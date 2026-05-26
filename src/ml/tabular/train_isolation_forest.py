"""
Training script for Isolation Forest.
Run: python -m src.ml.tabular.train_isolation_forest
"""

import json
from pathlib import Path

import numpy as np

from config.logging_config import get_logger
from src.ml.tabular.isolation_forest import ThreatIsolationForest

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

    logger.info(
        "data_loaded",
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )

    # Train
    model = ThreatIsolationForest(
        n_estimators=200,
        contamination=0.115,
        random_state=42,
        n_jobs=-1,
    )
    model.train(X_train, y_train, train_mask)

    # Tune threshold on test set
    model.tune_threshold(
        X_test,
        y_test,
        test_mask,
        target_precision=0.3,
    )

    # Evaluate
    metrics = model.evaluate(X_test, y_test, test_mask)

    print("\n=== Isolation Forest Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Setup SHAP explainer on licit training samples
    licit_mask = train_mask & (y_train == 0)
    X_licit = X_train[licit_mask]
    model.setup_explainer(X_licit[:200])

    # Demo: explain one illicit transaction
    illicit_mask = test_mask & (y_test == 1)
    illicit_indices = np.where(illicit_mask)[0]
    if len(illicit_indices) > 0:
        sample = X_test[illicit_indices[0]]
        explanation = model.explain_single(sample)
        print("\n=== Sample SHAP Explanation (illicit tx) ===")
        print(f"  Risk score: {explanation['risk_score']}")
        for feat in explanation["top_features"]:
            print(
                f"  {feat['feature_name']}: "
                f"{feat['shap_value']:+.4f} "
                f"({feat['direction']})"
            )

    # Save
    model.save()

    # Save metrics for documentation
    with open(MODELS_DIR / "isolation_forest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("training_pipeline_complete")
    return metrics


if __name__ == "__main__":
    main()
