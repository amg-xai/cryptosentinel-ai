"""
Tests for VAE anomaly detector.
Uses small synthetic data — fast CPU execution, no GPU needed.
"""

import numpy as np
import pytest
import torch

from src.ml.tabular.autoencoder import (
    VAE,
    ThreatAutoencoder,
    vae_loss,
)


def make_data(n_normal=200, n_anomaly=20, n_features=165, seed=42):
    rng = np.random.RandomState(seed)
    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    X_anomaly = (rng.randn(n_anomaly, n_features) * 3 + 4).astype(np.float32)
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    return X, y, mask


def test_vae_forward_pass():
    """VAE forward pass returns correct shapes."""
    model = VAE(input_dim=165, latent_dim=16)
    x = torch.randn(8, 165)
    x_recon, mu, log_var = model(x)
    assert x_recon.shape == (8, 165)
    assert mu.shape == (8, 16)
    assert log_var.shape == (8, 16)


def test_vae_loss_returns_three_values():
    """vae_loss returns total, recon, and kl losses."""
    x = torch.randn(16, 165)
    x_recon = torch.randn(16, 165)
    mu = torch.randn(16, 16)
    log_var = torch.randn(16, 16)
    total, recon, kl = vae_loss(x, x_recon, mu, log_var)
    assert total.item() > 0
    assert recon.item() > 0


def test_reconstruction_error_shape():
    """Reconstruction error must be per-sample."""
    model = VAE(input_dim=165, latent_dim=16)
    x = torch.randn(10, 165)
    errors = model.reconstruction_error(x)
    assert errors.shape == (10,)
    assert (errors >= 0).all()


def test_autoencoder_trains_without_error():
    """ThreatAutoencoder trains on synthetic data."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8, learning_rate=1e-3)
    history = ae.train(X, y, mask, epochs=2, batch_size=64)
    assert ae._is_fitted is True
    assert len(history) == 2
    assert "total_loss" in history[0]


def test_score_samples_in_range():
    """Scores must be float32 in [0, 1]."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8)
    ae.train(X, y, mask, epochs=2, batch_size=64)
    scores = ae.score_samples(X[:10])
    assert scores.dtype == np.float32
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_predict_returns_binary():
    """Predictions must be 0 or 1."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8)
    ae.train(X, y, mask, epochs=2, batch_size=64)
    preds = ae.predict(X[:20], threshold=0.5)
    assert set(preds).issubset({0, 1})


def test_unfitted_raises():
    """Scoring before training raises RuntimeError."""
    ae = ThreatAutoencoder(input_dim=165)
    X = np.random.randn(5, 165).astype(np.float32)
    with pytest.raises(RuntimeError):
        ae.score_samples(X)


def test_threshold_from_percentile():
    """Threshold setting from percentile must update _threshold."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8)
    ae.train(X, y, mask, epochs=2, batch_size=64)
    licit_mask = mask & (y == 0)
    t = ae.set_threshold_from_percentile(X[licit_mask], percentile=95.0)
    assert 0.0 <= t <= 1.0
    assert ae._threshold == t


def test_evaluate_returns_required_keys():
    """evaluate() must return all required metric keys."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8)
    ae.train(X, y, mask, epochs=2, batch_size=64)
    metrics = ae.evaluate(X, y, mask)
    required = {
        "f1_score",
        "pr_auc",
        "roc_auc",
        "threshold",
        "test_samples",
        "illicit_detected",
        "illicit_total",
    }
    assert required.issubset(set(metrics.keys()))


def test_save_and_load(tmp_path):
    """Save and load must preserve model behavior."""
    X, y, mask = make_data()
    ae = ThreatAutoencoder(input_dim=165, latent_dim=8)
    ae.train(X, y, mask, epochs=2, batch_size=64)

    save_path = str(tmp_path / "test_ae.pt")
    ae.save(save_path)

    loaded = ThreatAutoencoder.load(save_path)
    assert loaded._is_fitted is True

    original = ae.score_samples(X[:5])
    restored = loaded.score_samples(X[:5])
    np.testing.assert_array_almost_equal(original, restored, decimal=5)


def test_licit_reconstruct_better_than_anomaly():
    """
    On clearly separated synthetic data, normal samples should
    have lower reconstruction error than anomalies.
    NOTE: This does NOT reliably hold on Elliptic — documented in MODELS.md.
    """
    X, y, mask = make_data(n_normal=500, n_anomaly=50)
    ae = ThreatAutoencoder(input_dim=165, latent_dim=16, learning_rate=1e-3)
    ae.train(X, y, mask, epochs=30, batch_size=128)

    scores = ae.score_samples(X)
    normal_mean = scores[y == 0].mean()
    anomaly_mean = scores[y == 1].mean()
    assert anomaly_mean > normal_mean
