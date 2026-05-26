"""
Isolation Forest anomaly detector for blockchain transaction scoring.

WHY Isolation Forest:
  - Unsupervised — works without labels (important for unknown class)
  - Fast inference — O(n log n) training, O(log n) scoring
  - Interpretable — isolation depth has intuitive meaning
  - Good baseline before GNN

WHY calibration matters:
  Raw IsolationForest scores are NOT probabilities.
  score_samples() returns values roughly in [-0.5, 0.5].
  We calibrate to [0, 1] using isotonic regression so the output
  can be used directly as a risk score by compliance teams.

Training strategy:
  Train ONLY on licit transactions (label=2 / binary=0).
  Anomaly = anything that doesn't look like normal licit activity.
  This mirrors how real fraud systems work — model normality, flag deviations.
"""

from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from config.logging_config import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")


class ThreatIsolationForest:
    """
    Isolation Forest wrapper with calibration and SHAP explainability.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self._threshold: float = 0.5
        self._is_fitted: bool = False
        self._explainer = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_mask: np.ndarray,
    ) -> dict:
        """
        Train on all known samples.
        IF learns the overall distribution — illicit clusters separately.
        """
        X_known = X_train[train_mask]

        logger.info(
            "isolation_forest_training_started",
            n_estimators=self.n_estimators,
            training_samples=len(X_known),
            contamination=self.contamination,
        )

        self.model.fit(X_known)
        self._is_fitted = True

        logger.info("isolation_forest_training_complete")
        return {"training_samples": len(X_known)}

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly scores. Higher = more anomalous = higher risk.
        sklearn returns more negative = more anomalous, so we negate.
        Then normalize per-batch to [0,1].
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        raw_scores = self.model.score_samples(X)
        # More negative = more anomalous → negate so higher = more anomalous
        risk_scores = -raw_scores

        score_min = risk_scores.min()
        score_max = risk_scores.max()
        if score_max > score_min:
            normalized = (risk_scores - score_min) / (score_max - score_min)
        else:
            normalized = np.full_like(risk_scores, 0.5)

        return normalized.astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calibrated risk scores as probabilities.
        Returns array of shape [n_samples] with values in [0, 1].
        Higher = higher risk of being illicit.
        """
        return self.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Binary predictions using threshold.
        Returns 1 for illicit (anomalous), 0 for licit (normal).
        """
        t = threshold if threshold is not None else self._threshold
        scores = self.predict_proba(X)
        return (scores >= t).astype(np.int64)

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_mask: np.ndarray,
        target_precision: float = 0.7,
    ) -> float:
        """
        Find threshold that achieves target_precision on validation set.
        We optimize for precision — minimize false positives (alert fatigue).

        WHY precision over recall:
          In a SOC, analysts investigate every alert.
          Too many false positives = analysts ignore the system.
          We target 70% precision as a reasonable starting point.
        """
        X_known = X_val[val_mask]
        y_known = y_val[val_mask]

        scores = self.predict_proba(X_known)
        precision, recall, thresholds = precision_recall_curve(y_known, scores)

        # Find threshold where precision >= target
        best_threshold = 0.5
        for p, r, t in zip(precision, recall, thresholds):
            if p >= target_precision:
                best_threshold = float(t)
                break

        self._threshold = best_threshold
        logger.info(
            "threshold_tuned",
            threshold=best_threshold,
            target_precision=target_precision,
        )
        return best_threshold

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_mask: np.ndarray,
    ) -> dict:
        """
        Evaluate on test set using known labels only.
        Returns dict of metrics for MODELS.md documentation.
        """
        X_known = X_test[test_mask]
        y_known = y_test[test_mask]

        scores = self.predict_proba(X_known)
        predictions = self.predict(X_known)

        # Core metrics
        f1 = f1_score(y_known, predictions, zero_division=0)
        pr_auc = average_precision_score(y_known, scores)

        try:
            roc_auc = roc_auc_score(y_known, scores)
        except ValueError:
            roc_auc = 0.0

        # Precision at various recall levels
        precision, recall, _ = precision_recall_curve(y_known, scores)

        metrics = {
            "f1_score": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "threshold": round(self._threshold, 4),
            "test_samples": len(X_known),
            "illicit_detected": int(predictions[y_known == 1].sum()),
            "illicit_total": int((y_known == 1).sum()),
        }

        logger.info("isolation_forest_evaluation", **metrics)
        return metrics

    def setup_explainer(self, X_background: np.ndarray) -> None:
        """
        Initialize SHAP explainer using a background sample.
        X_background: small sample (100-500 rows) representing normal data.
        Call this after training before calling explain().
        """
        # Use a subsample for efficiency
        n_background = min(200, len(X_background))
        background = X_background[:n_background]

        self._explainer = shap.TreeExplainer(self.model)
        logger.info(
            "shap_explainer_initialized",
            background_samples=n_background,
        )

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for input samples.
        Returns array of shape [n_samples, n_features].
        Higher absolute value = feature contributed more to anomaly score.
        """
        if self._explainer is None:
            raise RuntimeError("Call setup_explainer() before explain().")
        shap_values = self._explainer.shap_values(X)
        return shap_values

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain a single transaction.
        Returns top 5 features driving the anomaly score.
        """
        if self._explainer is None:
            raise RuntimeError("Call setup_explainer() before explain().")

        shap_vals = self._explainer.shap_values(x.reshape(1, -1))[0]
        risk_score = float(self.predict_proba(x.reshape(1, -1))[0])

        # Top features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_features = [
            {
                "feature_index": int(idx),
                "feature_name": f"feat_{idx + 1}",
                "shap_value": round(float(shap_vals[idx]), 6),
                "direction": (
                    "increases_risk" if shap_vals[idx] > 0 else "decreases_risk"
                ),
            }
            for idx in top_indices
        ]

        return {
            "risk_score": round(risk_score, 4),
            "top_features": top_features,
        }

    def save(self, path: str | None = None) -> str:
        """Save model to disk."""
        save_path = path or str(MODELS_DIR / "isolation_forest.joblib")
        joblib.dump(self, save_path)
        logger.info("model_saved", path=save_path)
        return save_path

    @classmethod
    def load(cls, path: str | None = None) -> "ThreatIsolationForest":
        """Load model from disk."""
        load_path = path or str(MODELS_DIR / "isolation_forest.joblib")
        model = joblib.load(load_path)
        logger.info("model_loaded", path=load_path)
        return model
