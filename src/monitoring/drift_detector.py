"""
Model drift detection via Population Stability Index (PSI).

WHY PSI over mean-shift:
  The existing Prometheus rule compares today's average risk score to
  yesterday's. That catches a shifting MEAN but is blind to shape changes
  — scores splitting into two clusters, variance collapsing, a tail
  fattening. PSI compares the full BINNED distribution, so it catches
  shape drift the mean misses.

PSI formula:
  PSI = Σ (live_pct - base_pct) * ln(live_pct / base_pct)
  over a fixed set of score bins.

Interpretation (industry convention, credit-risk origin):
  PSI < 0.10  -> stable, no action
  0.10–0.25   -> moderate drift, investigate
  PSI > 0.25  -> significant drift, model likely stale / population changed

WHY it matters here:
  A blockchain threat model trained on 2019 Elliptic data sees a 2026
  transaction mix. Attack patterns evolve (new mixers, new bridge exploits).
  Drift detection tells you WHEN to retrain instead of guessing.
"""
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)

BASELINE_PATH = Path("data/models/drift_baseline.json")

# Fixed bins over the [0, 1] risk-score range. Edges chosen to match the
# Prometheus histogram buckets so the two drift signals stay comparable.
BIN_EDGES = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


def _binned_pct(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Fraction of scores falling in each bin; eps-floored to avoid log(0)."""
    counts, _ = np.histogram(scores, bins=BIN_EDGES)
    pct = counts / max(counts.sum(), 1)
    return np.clip(pct, eps, None)


def compute_psi(baseline: np.ndarray, live: np.ndarray) -> float:
    """PSI between a baseline score array and a live score array."""
    base_pct = _binned_pct(np.asarray(baseline, dtype=float))
    live_pct = _binned_pct(np.asarray(live, dtype=float))
    psi = np.sum((live_pct - base_pct) * np.log(live_pct / base_pct))
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


class DriftDetector:
    """
    Maintains a rolling window of live risk scores and compares it to a
    saved baseline distribution. Emits PSI to Prometheus.
    """

    def __init__(self, window_size: int = 500, min_samples: int = 100):
        self.window_size = window_size
        self.min_samples = min_samples
        self._window = deque(maxlen=window_size)
        self._baseline: Optional[np.ndarray] = None
        self._load_baseline()

    def _load_baseline(self) -> None:
        if BASELINE_PATH.exists():
            try:
                data = json.loads(BASELINE_PATH.read_text())
                self._baseline = np.array(data["scores"], dtype=float)
                logger.info("drift_baseline_loaded",
                            samples=len(self._baseline))
            except Exception as e:
                logger.warning("drift_baseline_load_failed", error=str(e))

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None and len(self._baseline) > 0

    def observe(self, score: float) -> None:
        """Record one live risk score into the rolling window."""
        self._window.append(float(score))

    def current_psi(self) -> Optional[float]:
        """
        PSI of the current window vs baseline.
        Returns None if no baseline or too few samples yet.
        """
        if not self.has_baseline:
            return None
        if len(self._window) < self.min_samples:
            return None
        return compute_psi(self._baseline, np.array(self._window))

    def update_metrics(self) -> Optional[float]:
        """Compute PSI and push it to Prometheus gauges."""
        psi = self.current_psi()
        try:
            from src.monitoring.metrics import (
                MODEL_SCORE_DRIFT_PSI, MODEL_SCORE_DRIFT_SAMPLES,
            )
            MODEL_SCORE_DRIFT_SAMPLES.set(len(self._window))
            if psi is not None:
                MODEL_SCORE_DRIFT_PSI.set(psi)
        except Exception as e:
            logger.debug("drift_metric_update_skipped", error=str(e))
        return psi


def save_baseline(scores: np.ndarray) -> bool:
    """Persist a baseline score distribution for future drift comparison."""
    try:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scores": [round(float(s), 6) for s in scores],
            "n": int(len(scores)),
        }
        BASELINE_PATH.write_text(json.dumps(payload))
        logger.info("drift_baseline_saved", samples=len(scores))
        return True
    except Exception as e:
        logger.error("drift_baseline_save_failed", error=str(e))
        return False
