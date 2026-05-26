"""
Tests for Isolation Forest model.
Uses synthetic data — fast, no CSV files needed.
"""

import numpy as np
import pytest

from src.ml.tabular.isolation_forest import ThreatIsolationForest


def make_synthetic_data(n_normal=500, n_anomaly=50, n_features=165, seed=42):
    """
    Create synthetic data with clear normal/anomaly separation.
    Normal: centered at 0, small variance
    Anomaly: centered at 5, large variance
    """
    rng = np.random.RandomState(seed)
    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    X_anomaly = (rng.randn(n_anomaly, n_features) * 2 + 5).astype(np.float32)
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    return X, y, mask


def test_model_trains_without_error():
    """Model should train on synthetic data without exceptions."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    result = model.train(X, y, mask)
    assert model._is_fitted is True
    assert "training_samples" in result


def test_score_samples_returns_float32():
    """Scores must be float32 arrays in [0, 1]."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    model.train(X, y, mask)
    scores = model.score_samples(X[:10])
    assert scores.dtype == np.float32
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_predict_returns_binary():
    """Predictions must be 0 or 1."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    model.train(X, y, mask)
    preds = model.predict(X[:20])
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape():
    """predict_proba should return 1D array matching input length."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    model.train(X, y, mask)
    proba = model.predict_proba(X[:15])
    assert proba.shape == (15,)


def test_unfitted_model_raises():
    """Scoring before training must raise RuntimeError."""
    model = ThreatIsolationForest()
    X = np.random.randn(5, 165).astype(np.float32)
    with pytest.raises(RuntimeError):
        model.score_samples(X)


def test_evaluate_returns_required_metrics():
    """evaluate() must return all required metric keys."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    model.train(X, y, mask)
    metrics = model.evaluate(X, y, mask)
    required_keys = {
        "f1_score",
        "pr_auc",
        "roc_auc",
        "threshold",
        "test_samples",
        "illicit_detected",
        "illicit_total",
    }
    assert required_keys.issubset(set(metrics.keys()))


def test_save_and_load(tmp_path):
    """Model should save and load correctly."""
    X, y, mask = make_synthetic_data()
    model = ThreatIsolationForest(n_estimators=10, random_state=42)
    model.train(X, y, mask)

    save_path = str(tmp_path / "test_model.joblib")
    model.save(save_path)

    loaded = ThreatIsolationForest.load(save_path)
    assert loaded._is_fitted is True

    original_scores = model.score_samples(X[:5])
    loaded_scores = loaded.score_samples(X[:5])
    np.testing.assert_array_almost_equal(original_scores, loaded_scores)


def test_anomalies_score_higher_on_synthetic():
    """
    On clearly separated synthetic data, anomalies should score higher.
    NOTE: This does NOT hold on Elliptic dataset — documented in MODELS.md.
    IF is a weak baseline for Elliptic. GNN is the primary detector.
    """
    X, y, mask = make_synthetic_data(n_normal=1000, n_anomaly=100)
    model = ThreatIsolationForest(
        n_estimators=100,
        contamination=0.09,
        random_state=42,
    )
    model.train(X, y, mask)
    scores = model.score_samples(X)
    mean_normal = scores[y == 0].mean()
    mean_anomaly = scores[y == 1].mean()
    assert mean_anomaly > mean_normal
