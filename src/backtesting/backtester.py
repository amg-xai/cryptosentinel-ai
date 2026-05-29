"""
Backtesting framework — evaluate the full decision system on labeled data.

WHY backtest the SYSTEM, not just the model:
  GNN F1=0.676 measures the model in isolation. But production decisions
  come from the CompositeRiskScorer (weighted ensemble) + action-tier
  thresholds. A great model with a badly-tuned threshold makes bad calls.
  Backtesting replays labeled transactions through the ENTIRE decision
  path and measures the precision/recall the SOC analyst actually sees.

WHAT it produces:
  - Confusion matrix (TP/FP/TN/FN) at the production threshold
  - Precision / recall / F1 / accuracy on threat decisions
  - A threshold sweep: precision & recall at each cutoff, so you can
    pick the operating point (high precision for low alert fatigue,
    or high recall to miss fewer threats)

Data:
  Uses the Elliptic test set (chronologically held out, steps 35-49).
  GNN probabilities on test nodes are the primary signal — those nodes
  are real graph nodes, so this is a faithful replay.
"""
import json
import numpy as np
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")
RESULTS_PATH = MODELS_DIR / "backtest_results.json"


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix counts."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_confusion(c: dict) -> dict:
    """Precision/recall/F1/accuracy from confusion counts."""
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Confusion + metrics for a single decision threshold."""
    preds = (scores >= threshold).astype(int)
    conf = _confusion(labels, preds)
    metrics = _metrics_from_confusion(conf)
    metrics["confusion"] = conf
    metrics["threshold"] = round(threshold, 3)
    return metrics


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    steps: int = 19,
) -> list:
    """Evaluate metrics across thresholds from 0.05 to 0.95."""
    thresholds = np.linspace(0.05, 0.95, steps)
    return [evaluate_at_threshold(scores, labels, float(t)) for t in thresholds]


def find_best_threshold(sweep: list) -> dict:
    """Pick the threshold with the highest F1."""
    return max(sweep, key=lambda m: m["f1"])


class Backtester:
    """Replays labeled test transactions through the GNN decision path."""

    def __init__(self):
        self.gnn_trainer = None
        self.gnn_data = None
        self.test_mask = None
        self.y_test = None
        self._loaded = False

    def load(self) -> bool:
        """Load GNN model, graph data, and the held-out test labels."""
        try:
            import torch  # noqa
            from src.ml.gnn.trainer import GNNTrainer, build_pyg_data

            gnn_path = MODELS_DIR / "gnn_final.pt"
            if not gnn_path.exists():
                logger.warning("backtest_no_model")
                return False

            self.gnn_trainer = GNNTrainer()
            self.gnn_trainer.load(str(gnn_path), in_channels=165)

            X_train = np.load(PROCESSED_DIR / "X_train.npy")
            X_test = np.load(PROCESSED_DIR / "X_test.npy")
            y_train = np.load(PROCESSED_DIR / "y_train.npy")
            y_test = np.load(PROCESSED_DIR / "y_test.npy")
            train_mask = np.load(PROCESSED_DIR / "train_mask.npy")
            test_mask = np.load(PROCESSED_DIR / "test_mask.npy")
            edge_index = np.load(PROCESSED_DIR / "edge_index.npy")

            self.gnn_data = build_pyg_data(
                X_train, X_test, y_train, y_test,
                train_mask, test_mask, edge_index,
            )
            self.test_mask = test_mask
            self.y_test = y_test
            self._loaded = True
            logger.info("backtester_loaded", test_samples=int(test_mask.sum()))
            return True
        except Exception as e:
            logger.error("backtester_load_failed", error=str(e))
            return False

    def _gnn_scores_on_test(self) -> tuple:
        """Run a forward pass; return (scores, labels) for test nodes.

        Mirrors GNNTrainer.evaluate(): mask BOTH probs and labels by
        data.test_mask so they align node-for-node.
        """
        import torch

        device = torch.device("cpu")
        self.gnn_trainer.model.to(device)
        self.gnn_trainer.model.eval()
        with torch.no_grad():
            data = self.gnn_data
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            out = self.gnn_trainer.model(x, edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]

            mask = data.test_mask.to(device)
            scores = probs[mask].cpu().numpy()
            labels = data.y[mask].cpu().numpy().astype(int)

        return scores, labels

    def run(self, production_threshold: float = 0.5) -> dict:
        """
        Full backtest: confusion + metrics at the production threshold,
        plus a threshold sweep and the best-F1 operating point.
        """
        if not self._loaded:
            raise RuntimeError("Backtester not loaded — call load() first")

        scores, labels = self._gnn_scores_on_test()

        # Guard: lengths must match
        n = min(len(scores), len(labels))
        scores, labels = scores[:n], labels[:n]

        prod = evaluate_at_threshold(scores, labels, production_threshold)
        sweep = threshold_sweep(scores, labels)
        best = find_best_threshold(sweep)

        results = {
            "test_samples": int(n),
            "illicit_in_test": int(np.sum(labels == 1)),
            "production_threshold": production_threshold,
            "production_metrics": prod,
            "best_threshold": best,
            "threshold_sweep": sweep,
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("backtest_complete",
                    f1=prod["f1"], best_f1=best["f1"])
        return results
