"""
Training script for VAE anomaly detector.
Run: python -m src.ml.tabular.train_autoencoder
"""

import json
from pathlib import Path

import numpy as np

from config.logging_config import get_logger
from src.ml.tabular.autoencoder import ThreatAutoencoder

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")


def main():
    logger.info("loading_processed_data")

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    train_mask = np.load(PROCESSED_DIR / "train_mask.npy")
    test_mask = np.load(PROCESSED_DIR / "test_mask.npy")

    # Train
    ae = ThreatAutoencoder(
        input_dim=165,
        latent_dim=16,
        learning_rate=1e-3,
        kl_weight=0.001,
    )

    history = ae.train(
        X_train,
        y_train,
        train_mask,
        epochs=50,
        batch_size=512,
    )

    # Set threshold from licit training samples
    licit_mask = train_mask & (y_train == 0)
    X_licit = X_train[licit_mask]
    ae.set_threshold_from_percentile(X_licit, percentile=95.0)

    # Evaluate
    metrics = ae.evaluate(X_test, y_test, test_mask)

    print("\n=== Autoencoder (VAE) Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Training History (last 5 epochs) ===")
    for record in history[-5:]:
        print(
            f"  Epoch {record['epoch']:3d}: "
            f"total={record['total_loss']:.6f} "
            f"recon={record['recon_loss']:.6f} "
            f"kl={record['kl_loss']:.6f}"
        )

    # Save
    ae.save()

    with open(MODELS_DIR / "autoencoder_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODELS_DIR / "autoencoder_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("autoencoder_training_pipeline_complete")
    return metrics


if __name__ == "__main__":
    main()
