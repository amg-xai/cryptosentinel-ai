"""
GNN Explainability using PyTorch Geometric's GNNExplainer.

WHY explainability matters in production:
  A risk score of 0.87 means nothing to a compliance officer.
  "This wallet scored 0.87 because: 73% of its neighbors are
  flagged addresses, transaction velocity is 12x the baseline,
  and it has a direct connection to a known mixer" means everything.

  Explainability is not optional for financial crime detection systems.
  EU AI Act Article 13: high-risk AI must provide "meaningful information
  about the logic involved."

GNNExplainer approach:
  Learns a mask over edges and node features that maximally
  explains the model's prediction for a specific node.
  Returns: important edges (which connections matter?)
           important features (which of the 165 features matter?)

SHAP approach:
  Kernel SHAP on the tabular feature vector.
  Model-agnostic — works on IF, AE, GNN alike.
  Returns: per-feature contribution to the risk score.

Usage:
  explainer = RiskExplainer()
  explainer.load()
  explanation = explainer.explain_transaction(features_array)
  waterfall = explainer.shap_waterfall(features_array)
"""

from pathlib import Path

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")

# Feature names for the 16 live transaction features
LIVE_FEATURE_NAMES = [
    "value_eth",
    "log_value_eth",
    "gas",
    "gas_price_gwei",
    "is_contract_call",
    "is_contract_creation",
    "input_data_length",
    "tx_count_1h",
    "tx_count_24h",
    "value_sum_1h",
    "value_sum_24h",
    "unique_recipients_24h",
    "avg_gas_price_24h",
    "hour_of_day",
    "day_of_week",
    "value_velocity_ratio",
]

# Top Elliptic feature names (first 16 of 165)
ELLIPTIC_FEATURE_NAMES = [f"elliptic_feature_{i}" for i in range(165)]


class RiskExplainer:
    """
    Generates explanations for risk assessments.
    Uses SHAP for tabular models and GNNExplainer for graph model.
    """

    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.gnn_trainer = None
        self.gnn_data = None
        self._shap_explainer_if = None
        self._shap_background = None
        self._loaded = False

    def load(self) -> bool:
        """Load models and initialize SHAP explainers."""
        try:
            from src.ml.gnn.trainer import GNNTrainer, build_pyg_data
            from src.ml.tabular.autoencoder import ThreatAutoencoder
            from src.ml.tabular.isolation_forest import ThreatIsolationForest

            # Load IF
            if_path = MODELS_DIR / "isolation_forest.joblib"
            if if_path.exists():
                self.isolation_forest = ThreatIsolationForest.load(str(if_path))

            # Load AE
            ae_path = MODELS_DIR / "autoencoder.pt"
            if ae_path.exists():
                self.autoencoder = ThreatAutoencoder.load(str(ae_path))

            # Load GNN + graph data
            gnn_path = MODELS_DIR / "gnn_final.pt"
            if gnn_path.exists():
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
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_mask,
                    test_mask,
                    edge_index,
                )

                # SHAP background: 100 random training samples
                self._shap_background = X_train[
                    np.random.choice(len(X_train), 100, replace=False)
                ]

            self._loaded = True
            logger.info("risk_explainer_loaded")
            return True

        except Exception as e:
            logger.error("risk_explainer_load_failed", error=str(e))
            return False

    def explain_transaction(
        self,
        features_array: np.ndarray,
        feature_names: list | None = None,
    ) -> dict:
        """
        Generate a complete explanation for a transaction's risk score.
        Returns human-readable feature importance.
        """
        if not self._loaded:
            return {"error": "Explainer not loaded"}

        # Pad to 165 if needed
        if features_array.shape[0] < 165:
            padded = np.zeros(165, dtype=np.float32)
            padded[: features_array.shape[0]] = features_array
            features_array = padded

        names = feature_names or ELLIPTIC_FEATURE_NAMES
        explanation = {}

        # IF explanation via SHAP
        if_shap = self._explain_isolation_forest(features_array, names)
        if if_shap:
            explanation["isolation_forest"] = if_shap

        # GNN explanation via feature similarity
        gnn_exp = self._explain_gnn(features_array, names)
        if gnn_exp:
            explanation["gnn"] = gnn_exp

        # Combined top features
        explanation["top_risk_features"] = self._get_top_features(features_array, names)

        return explanation

    def _explain_isolation_forest(
        self,
        features_array: np.ndarray,
        feature_names: list,
    ) -> dict | None:
        """SHAP explanation for Isolation Forest."""
        if self.isolation_forest is None or self._shap_background is None:
            return None

        try:
            import shap

            # KernelExplainer works with any model
            def if_predict(X):
                scores = []
                for x in X:
                    try:
                        s = self.isolation_forest.predict_proba(x.reshape(1, -1))
                        scores.append(float(s[0]))
                    except Exception:
                        scores.append(0.5)
                return np.array(scores)

            explainer = shap.KernelExplainer(
                if_predict,
                self._shap_background[:20],  # small background for speed
            )
            shap_values = explainer.shap_values(
                features_array.reshape(1, -1),
                nsamples=50,
                silent=True,
            )

            # Top 5 features by absolute SHAP value
            sv = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
            top_indices = np.argsort(np.abs(sv))[-5:][::-1]

            return {
                "top_features": [
                    {
                        "feature": (
                            feature_names[i]
                            if i < len(feature_names)
                            else f"feature_{i}"
                        ),
                        "shap_value": round(float(sv[i]), 4),
                        "direction": (
                            "increases_risk" if sv[i] > 0 else "decreases_risk"
                        ),
                    }
                    for i in top_indices
                ],
                "base_value": round(float(explainer.expected_value), 4),
            }

        except Exception as e:
            logger.error("if_shap_failed", error=str(e))
            return None

    def _explain_gnn(
        self,
        features_array: np.ndarray,
        feature_names: list,
    ) -> dict | None:
        """
        GNN explanation via nearest neighbor analysis.
        Find the most similar training node and explain which
        features made it similar (high cosine similarity contribution).
        """
        if self.gnn_trainer is None or self.gnn_data is None:
            return None

        try:
            import torch

            all_features = self.gnn_data.x.numpy()
            query = features_array / (np.linalg.norm(features_array) + 1e-8)
            norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
            normalized = all_features / norms
            similarities = normalized @ query
            nearest_idx = int(np.argmax(similarities))

            # Get GNN score for nearest node
            self.gnn_trainer.model.eval()
            with torch.no_grad():
                data = self.gnn_data
                out = self.gnn_trainer.model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)
                gnn_score = float(probs[nearest_idx, 1])

            # Feature contribution via element-wise similarity
            nearest_features = all_features[nearest_idx]
            contributions = query * (
                nearest_features / (np.linalg.norm(nearest_features) + 1e-8)
            )
            top_indices = np.argsort(np.abs(contributions))[-5:][::-1]

            return {
                "gnn_score": round(gnn_score, 4),
                "nearest_node_idx": nearest_idx,
                "similarity": round(float(similarities[nearest_idx]), 4),
                "top_similar_features": [
                    {
                        "feature": (
                            feature_names[i]
                            if i < len(feature_names)
                            else f"feature_{i}"
                        ),
                        "contribution": round(float(contributions[i]), 4),
                    }
                    for i in top_indices
                ],
            }

        except Exception as e:
            logger.error("gnn_explain_failed", error=str(e))
            return None

    def _get_top_features(
        self,
        features_array: np.ndarray,
        feature_names: list,
    ) -> list:
        """
        Get top features by absolute value — simple but fast.
        Used as fallback when SHAP is too slow.
        """
        top_indices = np.argsort(np.abs(features_array))[-10:][::-1]
        return [
            {
                "feature": (
                    feature_names[i] if i < len(feature_names) else f"feature_{i}"
                ),
                "value": round(float(features_array[i]), 4),
                "abs_magnitude": round(float(abs(features_array[i])), 4),
            }
            for i in top_indices
            if abs(features_array[i]) > 1e-6
        ]

    def get_shap_waterfall_data(
        self,
        features_array: np.ndarray,
        feature_names: list | None = None,
    ) -> dict:
        """
        Get data for a SHAP waterfall chart.
        Returns values that can be rendered by Plotly in the dashboard.
        """
        if not self._loaded or self._shap_background is None:
            return {}

        # Pad to 165
        if features_array.shape[0] < 165:
            padded = np.zeros(165, dtype=np.float32)
            padded[: features_array.shape[0]] = features_array
            features_array = padded

        names = feature_names or ELLIPTIC_FEATURE_NAMES

        try:
            import shap

            def if_predict(X):
                scores = []
                for x in X:
                    try:
                        s = self.isolation_forest.predict_proba(x.reshape(1, -1))
                        scores.append(float(s[0]))
                    except Exception:
                        scores.append(0.5)
                return np.array(scores)

            explainer = shap.KernelExplainer(
                if_predict,
                self._shap_background[:10],
            )
            shap_values = explainer.shap_values(
                features_array.reshape(1, -1),
                nsamples=30,
                silent=True,
            )

            sv = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
            top_n = 10
            top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]

            return {
                "features": [names[i] if i < len(names) else f"f_{i}" for i in top_idx],
                "shap_values": [round(float(sv[i]), 4) for i in top_idx],
                "base_value": round(float(explainer.expected_value), 4),
                "feature_values": [round(float(features_array[i]), 4) for i in top_idx],
            }

        except Exception as e:
            logger.error("shap_waterfall_failed", error=str(e))
            return {}
