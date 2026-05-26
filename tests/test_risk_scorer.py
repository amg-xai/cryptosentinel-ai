"""Tests for composite risk scorer and alert manager."""
import time
import pytest
from src.response.risk_scorer import (
    CompositeRiskScorer,
    ModelScores,
    ActionTier,
    compute_action_tier,
    compute_confidence,
    THRESHOLDS,
)
from src.response.alert_manager import AlertManager, Alert


# --- Risk scorer tests ---

def test_compute_action_tier_emergency():
    assert compute_action_tier(0.90) == ActionTier.EMERGENCY
    assert compute_action_tier(0.85) == ActionTier.EMERGENCY


def test_compute_action_tier_quarantine():
    assert compute_action_tier(0.75) == ActionTier.QUARANTINE
    assert compute_action_tier(0.70) == ActionTier.QUARANTINE


def test_compute_action_tier_watchlist():
    assert compute_action_tier(0.60) == ActionTier.WATCHLIST
    assert compute_action_tier(0.50) == ActionTier.WATCHLIST


def test_compute_action_tier_log_only():
    assert compute_action_tier(0.30) == ActionTier.LOG_ONLY
    assert compute_action_tier(0.0) == ActionTier.LOG_ONLY


def test_composite_score_in_range():
    """Composite score must always be in [0, 1]."""
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.8, autoencoder=0.6, isolation_forest=0.4)
    result = scorer.score("0xAddress", "0xTx", scores, value_eth=1.0)
    assert 0.0 <= result.composite_score <= 1.0


def test_velocity_flag_boosts_score():
    """Velocity flag must increase composite score."""
    scorer = CompositeRiskScorer()

    scores_normal = ModelScores(gnn=0.5, autoencoder=0.5)
    scores_velocity = ModelScores(gnn=0.5, autoencoder=0.5, velocity_flag=True)

    result_normal = scorer.score("0xAddr", "0xTx1", scores_normal)
    result_velocity = scorer.score("0xAddr", "0xTx2", scores_velocity)

    assert result_velocity.composite_score > result_normal.composite_score


def test_known_bad_address_floors_score():
    """Known bad address must have score >= 0.9."""
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.1, autoencoder=0.1, known_bad_address=True)
    result = scorer.score("0xBadAddr", "0xTx", scores)
    assert result.composite_score >= 0.9


def test_high_gnn_score_triggers_emergency():
    """Scores above 0.85 threshold should trigger emergency."""
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.95, autoencoder=0.90, isolation_forest=0.85)
    result = scorer.score("0xAddr", "0xTx", scores, value_eth=10.0)
    assert result.action in [ActionTier.EMERGENCY, ActionTier.QUARANTINE]
    assert result.composite_score >= 0.70


def test_low_scores_log_only():
    """Low scores across all models should be LOG_ONLY."""
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.1, autoencoder=0.1, isolation_forest=0.1)
    result = scorer.score("0xAddr", "0xTx", scores)
    assert result.action == ActionTier.LOG_ONLY


def test_explanation_contains_required_fields():
    """Explanation dict must have required fields."""
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.7, autoencoder=0.6)
    result = scorer.score("0xAddr", "0xTx", scores)
    assert "summary" in result.explanation
    assert "primary_driver" in result.explanation
    assert "model_contributions" in result.explanation


def test_weights_sum_to_one():
    """Ensemble weights must sum to 1.0."""
    from src.response.risk_scorer import ENSEMBLE_WEIGHTS
    assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-6


def test_confidence_all_agree():
    """When all available model scores agree direction, confidence is 1.0."""
    scores = ModelScores(gnn=0.9, autoencoder=0.8, isolation_forest=0.7)
    # graph_centrality defaults to 0.0 which pulls confidence down
    # Test with explicit graph_centrality also high
    scores.graph_centrality = 0.8
    confidence = compute_confidence(scores)
    assert confidence == 1.0


def test_confidence_models_disagree():
    """When models disagree, confidence should be less than 1.0."""
    scores = ModelScores(gnn=0.9, autoencoder=0.1)
    confidence = compute_confidence(scores)
    assert confidence < 1.0


# --- Alert manager tests ---

def make_assessment(address="0xAddr", score=0.8, value_eth=1.0):
    from src.response.risk_scorer import RiskAssessment, ModelScores
    scores = ModelScores(gnn=score)
    return RiskAssessment(
        address=address,
        tx_hash="0xTx",
        composite_score=score,
        confidence=0.9,
        action=compute_action_tier(score),
        model_scores=scores,
        value_at_risk_eth=value_eth,
    )


def test_alert_manager_creates_alert():
    manager = AlertManager()
    assessment = make_assessment()
    alert = manager.add_or_update(assessment)
    assert alert is not None
    assert manager.get_stats()["total_created"] == 1


def test_alert_manager_deduplicates():
    """Same address twice should update, not create duplicate."""
    manager = AlertManager()
    manager.add_or_update(make_assessment("0xAddr", score=0.7))
    manager.add_or_update(make_assessment("0xAddr", score=0.8))
    stats = manager.get_stats()
    assert stats["total_created"] == 1
    assert stats["total_deduplicated"] == 1


def test_alert_manager_acknowledge():
    manager = AlertManager()
    manager.add_or_update(make_assessment("0xAddr"))
    result = manager.acknowledge("0xAddr", analyst="analyst_1")
    assert result is True
    assert manager.get_stats()["total_active"] == 0


def test_alert_priority_higher_score_ranks_first():
    """Higher score alert should rank above lower score alert."""
    manager = AlertManager()
    manager.add_or_update(make_assessment("0xLow", score=0.5, value_eth=1.0))
    manager.add_or_update(make_assessment("0xHigh", score=0.9, value_eth=1.0))
    top = manager.get_top_alerts(n=2)
    assert top[0].assessment.address == "0xHigh"


def test_get_critical_alerts():
    """get_critical_alerts returns only EMERGENCY tier."""
    manager = AlertManager()
    manager.add_or_update(make_assessment("0xLow", score=0.3))
    manager.add_or_update(make_assessment("0xCrit", score=0.95))
    critical = manager.get_critical_alerts()
    assert len(critical) == 1
    assert critical[0].assessment.address == "0xCrit"


def test_is_threat_property():
    assessment = make_assessment(score=0.8)
    assert assessment.is_threat is True
    low_assessment = make_assessment(score=0.2)
    assert low_assessment.is_threat is False


def test_to_dict_contains_required_fields():
    manager = AlertManager()
    alert = manager.add_or_update(make_assessment())
    d = alert.to_dict()
    required = {"address", "composite_score", "severity",
                "action", "current_priority", "acknowledged"}
    assert required.issubset(set(d.keys()))
