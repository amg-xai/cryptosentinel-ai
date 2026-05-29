"""
Model registry — loads and caches all ML models at startup.

WHY singleton pattern:
  ML models are large (100MB+). Loading per-request is too slow.
  Load once at startup, share across all API requests.
  Thread-safe reads — models are read-only after loading.

WHY lazy loading with fallback:
  If a model file doesn't exist (not yet trained), return None.
  The API degrades gracefully — scores that model as 0.5 (neutral).
  This lets you start the API before all models are trained.
"""

from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("data/models")


class ModelRegistry:
    """Singleton that holds all loaded ML models."""

    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.gnn_trainer = None
        self.gnn_data = None
        self.scaler = None
        self._loaded = False

    def load_all(self) -> dict:
        """
        Load all available models. Called once at API startup.
        Returns dict of {model_name: loaded_bool}.
        """
        status = {}

        # Isolation Forest
        try:
            from src.ml.tabular.isolation_forest import ThreatIsolationForest

            if_path = MODELS_DIR / "isolation_forest.joblib"
            if if_path.exists():
                self.isolation_forest = ThreatIsolationForest.load(str(if_path))
                status["isolation_forest"] = True
                logger.info("isolation_forest_loaded")
            else:
                status["isolation_forest"] = False
                logger.warning("isolation_forest_not_found")
        except Exception as e:
            status["isolation_forest"] = False
            logger.error("isolation_forest_load_failed", error=str(e))

        # Autoencoder
        try:
            from src.ml.tabular.autoencoder import ThreatAutoencoder

            ae_path = MODELS_DIR / "autoencoder.pt"
            if ae_path.exists():
                self.autoencoder = ThreatAutoencoder.load(str(ae_path))
                status["autoencoder"] = True
                logger.info("autoencoder_loaded")
            else:
                status["autoencoder"] = False
                logger.warning("autoencoder_not_found")
        except Exception as e:
            status["autoencoder"] = False
            logger.error("autoencoder_load_failed", error=str(e))

        # Scaler
        try:
            import joblib

            scaler_path = MODELS_DIR / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
                status["scaler"] = True
            else:
                status["scaler"] = False
        except Exception:
            status["scaler"] = False

        # GNN
        try:
            from src.ml.gnn.trainer import GNNTrainer

            gnn_path = MODELS_DIR / "gnn_best.pt"
            if gnn_path.exists():
                self.gnn_trainer = GNNTrainer()
                self.gnn_trainer.load(str(gnn_path), in_channels=165)
                status["gnn"] = True
                logger.info("gnn_loaded")
            else:
                status["gnn"] = False
                logger.warning("gnn_not_found")
        except Exception as e:
            status["gnn"] = False
            logger.error("gnn_load_failed", error=str(e))

        self._loaded = True
        logger.info("model_registry_loaded", status=status)
        return status

    def score_tabular(self, features_array) -> dict:
        """
        Score a feature vector with IF and AE.
        Returns dict of scores, -1.0 for unavailable models.
        """
        import numpy as np

        scores = {
            "isolation_forest": -1.0,
            "autoencoder": -1.0,
        }

        if features_array.shape[0] < 165:
            padded = np.zeros(165, dtype=np.float32)
            padded[: features_array.shape[0]] = features_array
            features_array = padded

        if self.isolation_forest and self.isolation_forest._is_fitted:
            try:
                x = features_array.reshape(1, -1)
                scores["isolation_forest"] = float(
                    self.isolation_forest.predict_proba(x)[0]
                )
            except Exception as e:
                logger.error("if_scoring_failed", error=str(e))

        if self.autoencoder and self.autoencoder._is_fitted:
            try:
                x = features_array.reshape(1, -1)
                scores["autoencoder"] = float(self.autoencoder.predict_proba(x)[0])
            except Exception as e:
                logger.error("ae_scoring_failed", error=str(e))

        return scores


# Global singleton
registry = ModelRegistry()
