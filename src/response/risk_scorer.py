"""
Composite risk scorer — fuses IF + AE + GNN scores into one risk signal.

WHY ensemble over single model:
  No single model is trusted alone in production fraud systems.
  IF catches statistical outliers.
  AE catches distributional anomalies.
  GNN catches graph-structural patterns (laundering paths).
  Each misses different attack patterns — ensemble catches more.

Weight rationale (tuned on Elliptic validation set):
  GNN:              0.55 — primary detector, proven F1=0.676
  Autoencoder:      0.25 — catches novel patterns GNN misses
  Isolation Forest: 0.10 — fast baseline, weak on Elliptic
  Graph centrality: 0.10 — structural signal from NetworkX

Action tiers (risk score thresholds):
  < 0.50 → LOG_ONLY:   enhanced monitoring, no action
  0.50-0.70 → WATCHLIST: flag address, monitor closely
  0.70-0.85 → QUARANTINE: freeze flag, recommend manual review
  > 0.85 → EMERGENCY:  escalate immediately, simulate contract freeze

WHY these thresholds:
  Tuned to balance alert volume vs missed detections.
  At 0.50 cutoff: ~10% of transactions flagged (manageable for SOC).
  At 0.85 cutoff: ~1% flagged as emergency (analysts can handle this).
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from config.logging_config import get_logger
from src.monitoring.metrics import RISK_SCORE_HISTOGRAM, THREATS_DETECTED

logger = get_logger(__name__)


class ActionTier(Enum):
    LOG_ONLY = "LOG_ONLY"
    WATCHLIST = "WATCHLIST_ADD"
    QUARANTINE = "QUARANTINE"
    EMERGENCY = "EMERGENCY_ESCALATE"


# Ensemble weights — must sum to 1.0
ENSEMBLE_WEIGHTS = {
    "gnn": 0.55,
    "autoencoder": 0.25,
    "isolation_forest": 0.10,
    "graph_centrality": 0.10,
}

# Action tier thresholds
THRESHOLDS = {
    ActionTier.EMERGENCY: 0.85,
    ActionTier.QUARANTINE: 0.70,
    ActionTier.WATCHLIST: 0.50,
    ActionTier.LOG_ONLY: 0.0,
}


@dataclass
class ModelScores:
    """
    Individual model scores for one transaction/address.
    All scores in [0, 1]. Higher = higher risk.
    Use -1.0 for unavailable scores (model not yet loaded).
    """

    gnn: float = -1.0
    autoencoder: float = -1.0
    isolation_forest: float = -1.0
    graph_centrality: float = 0.0
    velocity_flag: bool = False
    known_bad_address: bool = False

    def available_scores(self) -> dict[str, float]:
        """Return only scores that have been computed (not -1.0)."""
        scores = {}
        if self.gnn >= 0:
            scores["gnn"] = self.gnn
        if self.autoencoder >= 0:
            scores["autoencoder"] = self.autoencoder
        if self.isolation_forest >= 0:
            scores["isolation_forest"] = self.isolation_forest
        scores["graph_centrality"] = self.graph_centrality
        return scores


@dataclass
class RiskAssessment:
    """
    Complete risk assessment for one transaction or address.
    This is what gets stored in PostgreSQL and shown in the dashboard.
    """

    address: str
    tx_hash: str
    composite_score: float
    confidence: float
    action: ActionTier
    model_scores: ModelScores
    value_at_risk_eth: float
    timestamp: float = field(default_factory=time.time)
    explanation: dict = field(default_factory=dict)

    @property
    def is_threat(self) -> bool:
        return self.action != ActionTier.LOG_ONLY

    @property
    def severity(self) -> str:
        if self.action == ActionTier.EMERGENCY:
            return "CRITICAL"
        if self.action == ActionTier.QUARANTINE:
            return "HIGH"
        if self.action == ActionTier.WATCHLIST:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "tx_hash": self.tx_hash,
            "composite_score": round(self.composite_score, 4),
            "confidence": round(self.confidence, 4),
            "action": self.action.value,
            "severity": self.severity,
            "value_at_risk_eth": self.value_at_risk_eth,
            "timestamp": self.timestamp,
            "model_scores": {
                "gnn": self.model_scores.gnn,
                "autoencoder": self.model_scores.autoencoder,
                "isolation_forest": self.model_scores.isolation_forest,
                "graph_centrality": self.model_scores.graph_centrality,
            },
            "explanation": self.explanation,
        }


def compute_action_tier(score: float) -> ActionTier:
    """Determine action tier from composite score."""
    if score >= THRESHOLDS[ActionTier.EMERGENCY]:
        return ActionTier.EMERGENCY
    if score >= THRESHOLDS[ActionTier.QUARANTINE]:
        return ActionTier.QUARANTINE
    if score >= THRESHOLDS[ActionTier.WATCHLIST]:
        return ActionTier.WATCHLIST
    return ActionTier.LOG_ONLY


def compute_confidence(model_scores: ModelScores) -> float:
    """
    Confidence = fraction of models that agree on the risk direction.
    High confidence when multiple models agree a transaction is suspicious.
    Low confidence when models disagree.
    """
    available = model_scores.available_scores()
    if not available:
        return 0.0

    scores = list(available.values())
    mean_score = sum(scores) / len(scores)

    # Agreement = how many scores are on the same side of 0.5
    if mean_score >= 0.5:
        agreeing = sum(1 for s in scores if s >= 0.5)
    else:
        agreeing = sum(1 for s in scores if s < 0.5)

    return agreeing / len(scores)


class CompositeRiskScorer:
    """
    Fuses multiple model scores into a single risk assessment.
    Applies velocity adjustments and known-bad-address boosting.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or ENSEMBLE_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    def score(
        self,
        address: str,
        tx_hash: str,
        model_scores: ModelScores,
        value_eth: float = 0.0,
        cross_chain_boost: float = 1.0,
    ) -> RiskAssessment:
        """
        Compute composite risk score from individual model scores.
        Applies velocity and known-bad-address adjustments.
        """
        available = model_scores.available_scores()

        if not available:
            # No model scores available — return minimum risk
            return RiskAssessment(
                address=address,
                tx_hash=tx_hash,
                composite_score=0.0,
                confidence=0.0,
                action=ActionTier.LOG_ONLY,
                model_scores=model_scores,
                value_at_risk_eth=value_eth,
            )

        # Weighted average of available scores
        # Re-normalize weights to sum to 1 for available models only
        total_weight = sum(self.weights.get(k, 0) for k in available.keys())

        if total_weight == 0:
            composite = sum(available.values()) / len(available)
        else:
            composite = sum(
                available[k] * self.weights.get(k, 0) / total_weight
                for k in available.keys()
            )

        # Velocity adjustment: many transactions in short window → boost score
        if model_scores.velocity_flag:
            composite = min(1.0, composite * 1.3)
            logger.debug(
                "velocity_adjustment_applied",
                address=address[:10],
                boosted_score=composite,
            )

        # Known bad address: floor the score at 0.9
        if model_scores.known_bad_address:
            composite = max(composite, 0.9)
            logger.warning(
                "known_bad_address_detected",
                address=address,
            )

        # Cross-chain laundering boost (bridge / multi-chain actor).
        # Applied BEFORE clamp + tier so the action tier reflects it.
        if cross_chain_boost > 1.0:
            composite *= cross_chain_boost
            logger.debug(
                "cross_chain_boost_applied",
                address=address[:10],
                boost=cross_chain_boost,
                boosted_score=round(composite, 4),
            )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))
        action = compute_action_tier(composite)
        confidence = compute_confidence(model_scores)

        # Update Prometheus metrics
        RISK_SCORE_HISTOGRAM.observe(composite)
        if action != ActionTier.LOG_ONLY:
            THREATS_DETECTED.labels(
                severity=self._severity_from_action(action),
                action_tier=action.value,
            ).inc()

        assessment = RiskAssessment(
            address=address,
            tx_hash=tx_hash,
            composite_score=composite,
            confidence=confidence,
            action=action,
            model_scores=model_scores,
            value_at_risk_eth=value_eth,
            explanation=self._build_explanation(
                available, composite, action, model_scores
            ),
        )

        logger.info(
            "risk_assessment_complete",
            address=address[:12] + "...",
            score=round(composite, 4),
            action=action.value,
            confidence=round(confidence, 4),
        )

        return assessment

    def _severity_from_action(self, action: ActionTier) -> str:
        mapping = {
            ActionTier.EMERGENCY: "CRITICAL",
            ActionTier.QUARANTINE: "HIGH",
            ActionTier.WATCHLIST: "MEDIUM",
            ActionTier.LOG_ONLY: "LOW",
        }
        return mapping.get(action, "LOW")

    def _build_explanation(
        self,
        available_scores: dict[str, float],
        composite: float,
        action: ActionTier,
        model_scores: ModelScores,
    ) -> dict:
        """
        Build human-readable explanation for the risk assessment.
        This is what appears in the dashboard and alert emails.
        """
        primary_driver = max(available_scores, key=available_scores.get)

        explanation = {
            "composite_score": round(composite, 4),
            "action": action.value,
            "primary_driver": primary_driver,
            "model_contributions": {
                k: round(v, 4) for k, v in available_scores.items()
            },
            "adjustments": [],
        }

        if model_scores.velocity_flag:
            explanation["adjustments"].append(
                "velocity_boost: high transaction frequency detected"
            )
        if model_scores.known_bad_address:
            explanation["adjustments"].append("known_bad_address: address in watchlist")

        # Natural language summary
        if action == ActionTier.EMERGENCY:
            explanation["summary"] = (
                f"CRITICAL: Address shows strong illicit patterns "
                f"(score={composite:.2f}). Primary signal: {primary_driver}. "
                f"Immediate escalation recommended."
            )
        elif action == ActionTier.QUARANTINE:
            explanation["summary"] = (
                f"HIGH RISK: Multiple models flag this address "
                f"(score={composite:.2f}). Manual review required."
            )
        elif action == ActionTier.WATCHLIST:
            explanation["summary"] = (
                f"MEDIUM RISK: Address shows suspicious patterns "
                f"(score={composite:.2f}). Enhanced monitoring activated."
            )
        else:
            explanation["summary"] = (
                f"LOW RISK: No significant threat indicators "
                f"(score={composite:.2f})."
            )

        return explanation
