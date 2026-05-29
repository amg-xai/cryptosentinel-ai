"""Tests for live drift baseline collection + comparison behavior."""
import numpy as np
from src.monitoring.drift_detector import compute_psi, classify_psi


def test_collector_module_imports():
    """The live baseline collector should import without side effects."""
    import src.monitoring.collect_live_baseline as m
    assert hasattr(m, "collect")
    assert hasattr(m, "_score_tx")


def test_low_variance_baseline_stable_self():
    """A low-variance baseline compared to itself reads zero drift."""
    base = np.full(500, 0.3747) + np.random.normal(0, 0.0005, 500)
    psi = compute_psi(base, base)
    assert psi < 0.01
    assert classify_psi(psi) == "stable"


def test_low_variance_baseline_catches_shift():
    """Even a low-variance baseline catches a real mean shift."""
    base = np.full(500, 0.3747) + np.random.normal(0, 0.0005, 500)
    shifted = base + 0.15
    psi = compute_psi(base, shifted)
    assert psi > 0.25
    assert classify_psi(psi) == "significant"
