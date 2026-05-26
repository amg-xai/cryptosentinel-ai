"""
GNN training script.
Run: python -m src.ml.gnn.train_gnn
Expected: F1 > 0.4, PR-AUC > 0.3 on Elliptic test set.
"""
import json
from pathlib import Path

import numpy as np

from config.logging_config import get_logger
from src.ml.gnn.model import ThreatGNN
from src.ml.gnn.trainer import GNNTrainer, build_pyg_data

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
    edge_index = np.load(PROCESSED_DIR / "edge_index.npy")

    # Build PyG Data object
    data = build_pyg_data(
        X_train, X_test,
        y_train, y_test,
        train_mask, test_mask,
        edge_index,
    )

    # Train
    trainer = GNNTrainer(
        hidden_channels=128,
        heads=4,
        dropout=0.3,
        learning_rate=0.001,
        weight_decay=1e-4,
    )

    history = trainer.train(
        data,
        epochs=100,
        batch_size=512,
        num_neighbors=[25, 10, 5],
    )

    # Evaluate
    metrics = trainer.evaluate(data)

    print("\n=== GNN Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Training History (last 5 epochs) ===")
    for record in history[-5:]:
        print(
            f"  Epoch {record['epoch']:3d}: "
            f"loss={record['loss']:.6f} "
            f"lr={record['lr']:.6f}"
        )

    # Save final model
    trainer.save(str(MODELS_DIR / "gnn_final.pt"))

    with open(MODELS_DIR / "gnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODELS_DIR / "gnn_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("gnn_training_pipeline_complete")
    return metrics


if __name__ == "__main__":
    main()
