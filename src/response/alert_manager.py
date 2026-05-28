"""
Alert manager — priority queue with deduplication and aging.

WHY priority queue:
  Not all alerts are equal. A 0.95-score address moving 100 ETH
  must be seen before a 0.51-score address moving 0.001 ETH.
  Priority = composite_score * log(value_at_risk + 1)

WHY deduplication:
  The same suspicious address may appear in hundreds of transactions.
  Without dedup, analysts see 200 alerts for one criminal wallet.
  We update the existing alert instead of creating duplicates.

WHY aging:
  An alert from 6 hours ago with no new activity is less urgent.
  Priority decays by 10% per hour after the first hour.
  This surfaces fresh threats over stale ones automatically.
"""

import heapq
import math
import time
from dataclasses import dataclass, field

from config.logging_config import get_logger
from src.response.risk_scorer import ActionTier, RiskAssessment

logger = get_logger(__name__)

# How long before an alert's priority starts decaying (seconds)
DECAY_START_SECONDS = 3600  # 1 hour
DECAY_RATE_PER_HOUR = 0.10  # 10% priority reduction per hour


@dataclass
class Alert:
    """
    A single threat alert in the priority queue.
    Comparable by priority for heapq (higher priority = processed first).
    """

    assessment: RiskAssessment
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    update_count: int = 0
    acknowledged: bool = False
    acknowledged_by: str | None = None

    @property
    def raw_priority(self) -> float:
        """Base priority before aging."""
        score = self.assessment.composite_score
        value = self.assessment.value_at_risk_eth
        return score * math.log1p(value + 1)

    @property
    def current_priority(self) -> float:
        """Priority after time-based decay."""
        age_seconds = time.time() - self.updated_at
        if age_seconds <= DECAY_START_SECONDS:
            return self.raw_priority
        hours_past_decay = (age_seconds - DECAY_START_SECONDS) / 3600
        decay = (1 - DECAY_RATE_PER_HOUR) ** hours_past_decay
        return self.raw_priority * decay

    def __lt__(self, other: "Alert") -> bool:
        """Higher priority = processed first (max-heap via negation)."""
        return self.current_priority > other.current_priority

    def to_dict(self) -> dict:
        return {
            "address": self.assessment.address,
            "tx_hash": self.assessment.tx_hash,
            "composite_score": self.assessment.composite_score,
            "severity": self.assessment.severity,
            "action": self.assessment.action.value,
            "value_at_risk_eth": self.assessment.value_at_risk_eth,
            "current_priority": round(self.current_priority, 4),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "update_count": self.update_count,
            "acknowledged": self.acknowledged,
            "explanation": self.assessment.explanation,
        }


class AlertManager:
    """
    In-memory alert store with priority queue, deduplication, and aging.
    In production this persists to PostgreSQL — stubs added for Week 8.
    """

    def __init__(self, max_alerts: int = 10_000):
        self.max_alerts = max_alerts
        self._alerts: dict[str, Alert] = {}  # address -> Alert
        self._queue: list[Alert] = []  # heapq
        self._total_created = 0
        self._total_deduplicated = 0

    def add_or_update(self, assessment: RiskAssessment) -> Alert:
        """
        Add new alert or update existing one for same address.
        Deduplicates by address — one active alert per address.
        """
        address = assessment.address

        if address in self._alerts:
            # Update existing alert
            existing = self._alerts[address]
            existing.assessment = assessment
            existing.updated_at = time.time()
            existing.update_count += 1
            self._total_deduplicated += 1

            logger.debug(
                "alert_updated",
                address=address[:12] + "...",
                update_count=existing.update_count,
                new_score=assessment.composite_score,
            )
            return existing

        # Create new alert
        alert = Alert(assessment=assessment)
        self._alerts[address] = alert
        heapq.heappush(self._queue, alert)
        self._total_created += 1

        logger.info(
            "alert_created",
            address=address[:12] + "...",
            score=assessment.composite_score,
            action=assessment.action.value,
            severity=assessment.severity,
        )

        # Evict lowest priority if over limit
        if len(self._alerts) > self.max_alerts:
            self._evict_lowest_priority()

        return alert

    def get_top_alerts(self, n: int = 50) -> list[Alert]:
        """Get top N alerts by current priority."""
        active = [a for a in self._alerts.values() if not a.acknowledged]
        return sorted(active, reverse=False)[:n]

    def get_critical_alerts(self) -> list[Alert]:
        """Get all EMERGENCY tier alerts."""
        return [
            a
            for a in self._alerts.values()
            if a.assessment.action == ActionTier.EMERGENCY and not a.acknowledged
        ]

    def acknowledge(self, address: str, analyst: str) -> bool:
        """Mark alert as acknowledged by an analyst."""
        if address not in self._alerts:
            return False
        alert = self._alerts[address]
        alert.acknowledged = True
        alert.acknowledged_by = analyst
        logger.info(
            "alert_acknowledged",
            address=address[:12] + "...",
            analyst=analyst,
            score=alert.assessment.composite_score,
        )
        return True

    def get_stats(self) -> dict:
        """Summary statistics for the dashboard."""
        active = [a for a in self._alerts.values() if not a.acknowledged]
        by_tier = {}
        for tier in ActionTier:
            by_tier[tier.value] = sum(1 for a in active if a.assessment.action == tier)
        return {
            "total_active": len(active),
            "total_created": self._total_created,
            "total_deduplicated": self._total_deduplicated,
            "by_tier": by_tier,
        }

    def _evict_lowest_priority(self) -> None:
        """Remove the lowest-priority non-emergency alert."""
        candidates = [
            (addr, alert)
            for addr, alert in self._alerts.items()
            if alert.assessment.action != ActionTier.EMERGENCY
        ]
        if not candidates:
            return
        lowest_addr = min(
            candidates,
            key=lambda x: x[1].current_priority,
        )[0]
        del self._alerts[lowest_addr]
        logger.debug("alert_evicted", address=lowest_addr[:12] + "...")
