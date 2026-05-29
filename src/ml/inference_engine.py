"""
Real-time inference engine — scores transactions as they arrive from Kafka.

Design:
  - Loads all trained models once at startup
  - Scores each transaction with IF + AE + GNN ensemble
  - Returns a RiskAssessment for every transaction
  - Runs in a ProcessPoolExecutor so GPU inference doesn't block event loop

WHY singleton pattern:
  Loading PyTorch models takes 2-5 seconds and uses significant RAM.
  We load once, score millions of transactions with the same model instance.
  Thread-safe reads — models are read-only after loading.
"""

from pathlib import Path

import numpy as np

from config.logging_config import get_logger
from src.monitoring.metrics import GNN_INFERENCE_LATENCY

logger = get_logger(__name__)

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")


class InferenceEngine:
    """
    Loads and runs all ML models for real-time transaction scoring.
    Call load() once at startup, then score() for every transaction.
    """

    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.gnn_trainer = None
        self.gnn_data = None
        self.scaler = None
        self._loaded = False

    def load(self) -> dict:
        """Load all trained models. Returns status dict."""
        import joblib

        status = {}

        # Scaler — needed for feature normalization
        try:
            scaler_path = MODELS_DIR / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
                status["scaler"] = True
                logger.info("scaler_loaded")
            else:
                status["scaler"] = False
                logger.warning("scaler_not_found")
        except Exception as e:
            status["scaler"] = False
            logger.error("scaler_load_failed", error=str(e))

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
        except Exception as e:
            status["autoencoder"] = False
            logger.error("autoencoder_load_failed", error=str(e))

        # GNN — load model and full graph for inference
        try:
            from src.ml.gnn.trainer import GNNTrainer, build_pyg_data

            gnn_path = MODELS_DIR / "gnn_final.pt"
            if gnn_path.exists():
                self.gnn_trainer = GNNTrainer()
                self.gnn_trainer.load(str(gnn_path), in_channels=165)

                # Load processed graph data for GNN inference context
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
                status["gnn"] = True
                logger.info("gnn_loaded_with_graph_data")
            else:
                status["gnn"] = False
        except Exception as e:
            status["gnn"] = False
            logger.error("gnn_load_failed", error=str(e))

        self._loaded = True
        logger.info("inference_engine_ready", status=status)
        return status

    def score_features(self, features_array: np.ndarray) -> dict:
        """
        Score a feature vector with IF and AE.

        NOTE: IF and AE were trained on 165-feature Elliptic dataset.
        Live blockchain transactions have 16 features.
        We pad to 165 for compatibility — zeros for missing features.
        This gives approximate scores. GNN (via similarity) is the
        primary scorer for live transactions.
        """
        scores = {
            "isolation_forest": -1.0,
            "autoencoder": -1.0,
        }

        # Pad features to 165 if needed
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
                import traceback

                logger.error("if_score_failed", error=str(e), tb=traceback.format_exc())

        if self.autoencoder and self.autoencoder._is_fitted:
            try:
                x = features_array.reshape(1, -1)
                scores["autoencoder"] = float(self.autoencoder.score_samples(x)[0])
            except Exception as e:
                import traceback

                logger.error("ae_score_failed", error=str(e), tb=traceback.format_exc())

        return scores

    def score_with_gnn(self, features_array: np.ndarray) -> float:
        """Score using GNN on CPU (thread-safe)."""
        if self.gnn_trainer is None or self.gnn_data is None:
            return -1.0

        try:
            import time

            import torch

            # Pad to 165 if needed
            if features_array.shape[0] < 165:
                padded = np.zeros(165, dtype=np.float32)
                padded[: features_array.shape[0]] = features_array
                features_array = padded

            start = time.perf_counter()

            # Force CPU for thread safety
            cpu_device = torch.device("cpu")
            model = self.gnn_trainer.model.to(cpu_device)
            data = self.gnn_data

            all_features = data.x.numpy()
            query = features_array / (np.linalg.norm(features_array) + 1e-8)
            norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
            normalized = all_features / norms
            similarities = normalized @ query
            nearest_idx = int(np.argmax(similarities))

            model.eval()
            with torch.no_grad():
                x_cpu = data.x.to(cpu_device)
                ei_cpu = data.edge_index.to(cpu_device)
                out = model(x_cpu, ei_cpu)
                probs = torch.softmax(out, dim=1)
                gnn_score = float(probs[nearest_idx, 1])

            elapsed = time.perf_counter() - start
            GNN_INFERENCE_LATENCY.observe(elapsed)

            return gnn_score

        except Exception as e:
            import traceback

            logger.error(
                "gnn_score_failed",
                error=str(e),
                tb=traceback.format_exc(),
            )
            return -1.0
        """
        Score a new transaction using the trained GNN.

        For a new transaction not in the training graph:
        We use the GNN to get the score of its nearest neighbor
        in feature space from the training set.

        WHY this approach:
          GNN is transductive on the Elliptic graph — it learned embeddings
          for those specific nodes. For NEW transactions from live blockchain,
          we find the most similar training node and use its GNN score.
          This is called "inductive inference via feature similarity."
          A proper production system would add the new node to the graph
          and run incremental GNN inference — that's the Week 6 extension.
        """
        if self.gnn_trainer is None or self.gnn_data is None:
            return -1.0

        # Pad to 165 features if needed (live transactions have 16 features)
        if features_array.shape[0] < 165:
            padded = np.zeros(165, dtype=np.float32)
            padded[: features_array.shape[0]] = features_array
            features_array = padded

        try:
            import time

            import torch

            start = time.perf_counter()

            # Get all node features from training graph
            all_features = self.gnn_data.x.numpy()

            # Find most similar node using cosine similarity
            query = features_array / (np.linalg.norm(features_array) + 1e-8)
            norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
            normalized = all_features / norms
            similarities = normalized @ query
            nearest_idx = int(np.argmax(similarities))

            # Get GNN score for that node
            self.gnn_trainer.model.eval()
            data = self.gnn_data.to(self.gnn_trainer.device)

            with torch.no_grad():
                out = self.gnn_trainer.model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)
                gnn_score = float(probs[nearest_idx, 1].cpu())

            elapsed = time.perf_counter() - start
            GNN_INFERENCE_LATENCY.observe(elapsed)

            logger.debug(
                "gnn_inference_complete",
                nearest_idx=nearest_idx,
                gnn_score=round(gnn_score, 4),
                latency_ms=round(elapsed * 1000, 2),
            )

            return gnn_score

        except Exception as e:
            import traceback

            logger.error("gnn_score_failed", error=str(e), tb=traceback.format_exc())
            return -1.0


# Global singleton — loaded once at pipeline startup
engine = InferenceEngine()
