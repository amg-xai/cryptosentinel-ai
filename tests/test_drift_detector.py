"""Tests for PSI-based model drift detection."""
import numpy as np
from src.monitoring.drift_detector import (
    compute_psi, classify_psi, DriftDetector, BIN_EDGES,
)


def test_identical_distributions_zero_psi():
    """Same data should give PSI very close to 0."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 2000)
    psi = compute_psi(data, data)
    assert psi < 0.01


def test_resampled_same_distribution_low_psi():
    """Resampling the same distribution stays in 'stable' range."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.4, 0.15, 5000).clip(0, 1)
    live = rng.choice(base, 1000)
    psi = compute_psi(base, live)
    assert psi < 0.10
    assert classify_psi(psi) == "stable"


def test_shifted_distribution_high_psi():
    """A clear mean shift should produce significant PSI."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.4, 0.1, 3000).clip(0, 1)
    live = (base[:1000] + 0.3).clip(0, 1)
    psi = compute_psi(base, live)
    assert psi > 0.25
    assert classify_psi(psi) == "significant"


def test_variance_collapse_high_psi():
    """
    Variance collapse keeps the mean ~same but changes shape.
    PSI must catch this (the whole point vs mean-shift detection).
    """
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, 3000)          # spread out, mean ~0.5
    live = rng.normal(0.5, 0.02, 1000).clip(0, 1)  # collapsed at 0.5
    # Means are similar...
    assert abs(base.mean() - live.mean()) < 0.1
    # ...but PSI flags the shape change
    psi = compute_psi(base, live)
    assert psi > 0.25


def test_classify_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.30) == "significant"
    assert classify_psi(0.10) == "moderate"   # boundary
    assert classify_psi(0.25) == "significant"  # boundary


def test_psi_symmetric_bins():
    """BIN_EDGES should span the full [0,1] score range."""
    assert BIN_EDGES[0] == 0.0
    assert BIN_EDGES[-1] == 1.0


def test_detector_no_baseline_returns_none(tmp_path, monkeypatch):
    """Without a baseline file, current_psi returns None."""
    import src.monitoring.drift_detector as dd
    monkeypatch.setattr(dd, "BASELINE_PATH", tmp_path / "nope.json")
    d = dd.DriftDetector()
    assert d.has_baseline is False
    assert d.current_psi() is None


def test_detector_min_samples_gate():
    """Detector returns None until min_samples observed."""
    d = DriftDetector(window_size=500, min_samples=100)
    # Force a baseline so the gate is the only blocker
    d._baseline = np.random.uniform(0, 1, 1000)
    for _ in range(50):
        d.observe(0.5)
    assert d.current_psi() is None  # only 50 < 100
    for _ in range(60):
        d.observe(0.5)
    assert d.current_psi() is not None  # now 110 >= 100


def test_detector_rolling_window_caps():
    """Window should not exceed window_size."""
    d = DriftDetector(window_size=100, min_samples=10)
    for i in range(250):
        d.observe(i / 250)
    assert len(d._window) == 100


def test_save_and_load_baseline(tmp_path, monkeypatch):
    """Baseline round-trips through disk."""
    import src.monitoring.drift_detector as dd
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(dd, "BASELINE_PATH", path)
    scores = np.random.uniform(0, 1, 500)
    assert dd.save_baseline(scores) is True
    d = dd.DriftDetector()
    assert d.has_baseline is True
    assert len(d._baseline) == 500
