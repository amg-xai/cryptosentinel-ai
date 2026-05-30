"""Tests for the live-native model (Option C) and its engine integration."""
import numpy as np
import pytest
from pathlib import Path

MODEL_EXISTS = (Path("data/models/live_model.joblib").exists()
                and Path("data/models/live_scaler.joblib").exists())


@pytest.mark.skipif(not MODEL_EXISTS, reason="live model not trained")
def test_engine_loads_live_model():
    from src.ml.inference_engine import engine
    status = engine.load()
    assert status.get("live_model") is True
    assert engine.live_model is not None


@pytest.mark.skipif(not MODEL_EXISTS, reason="live model not trained")
def test_score_live_returns_probability():
    from src.ml.inference_engine import engine
    engine.load()
    vec = np.zeros(16, dtype=np.float32)
    score = engine.score_live(vec)
    assert -1.0 <= score <= 1.0
    assert score != -1.0  # model is loaded, should produce a real score


@pytest.mark.skipif(not MODEL_EXISTS, reason="live model not trained")
def test_score_live_discriminates_on_gas():
    """Higher gas price (dominant feature) should raise the score."""
    from src.ml.inference_engine import engine
    engine.load()
    low = np.zeros(16, dtype=np.float32); low[4] = 1.0
    high = np.zeros(16, dtype=np.float32); high[4] = 60.0
    assert engine.score_live(high) > engine.score_live(low)


@pytest.mark.skipif(not MODEL_EXISTS, reason="live model not trained")
def test_score_live_handles_nan():
    from src.ml.inference_engine import engine
    engine.load()
    vec = np.full(16, np.nan, dtype=np.float32)
    score = engine.score_live(vec)
    assert -1.0 <= score <= 1.0  # must not crash


def test_score_live_unavailable_returns_minus_one():
    """A fresh engine with no live model returns -1.0, not a crash."""
    from src.ml.inference_engine import InferenceEngine
    eng = InferenceEngine()
    # don't load — live_model is None
    vec = np.zeros(16, dtype=np.float32)
    assert eng.score_live(vec) == -1.0
