"""
Elliptic Bitcoin dataset preprocessor.

Dataset facts:
  - 203,769 transactions (nodes)
  - 234,355 edges (directed payment flows)
  - 49 features per transaction (local + aggregated)
  - 3 classes: 1=illicit, 2=licit, 0=unknown
  - 49 time steps (temporal dimension)

CRITICAL design decisions:
  1. Chronological split — NEVER random split on time-series data
     Train: time steps 1-34, Test: time steps 35-49
     Random split causes temporal leakage — model sees future data during training

  2. Unknown class (0) handling:
     Excluded from supervised training labels
     BUT kept in graph structure for GNN message passing
     Unknown nodes still have neighbors that carry signal

  3. Feature normalization:
     Fit StandardScaler on TRAINING data only
     Transform test data using training scaler
     Fitting on test data = data leakage

  4. Feature groups:
     Column 0: txId (drop — not a feature)
     Column 1: time_step (use for splitting, then drop)
     Columns 2-94: local features (transaction-level)
     Columns 95-166: aggregated features (neighborhood stats)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from config.logging_config import get_logger

logger = get_logger(__name__)

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")

FEATURES_FILE = RAW_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = RAW_DIR / "elliptic_txs_classes.csv"
EDGELIST_FILE = RAW_DIR / "elliptic_txs_edgelist.csv"

# Chronological split boundary
TRAIN_TIME_STEPS = list(range(1, 35))  # Steps 1-34 → training
TEST_TIME_STEPS = list(range(35, 50))  # Steps 35-49 → testing


class EllipticProcessor:
    """
    Full preprocessing pipeline for the Elliptic dataset.
    Call process() to run everything end to end.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all three CSV files."""
        logger.info("loading_raw_data")

        # Features file has no header — we add column names
        # Column 0: txId, Column 1: time_step, Columns 2-166: features
        n_feature_cols = 165
        feature_col_names = ["txId", "time_step"] + [
            f"feat_{i}" for i in range(1, n_feature_cols + 1)
        ]

        features_df = pl.read_csv(
            FEATURES_FILE,
            has_header=False,
            new_columns=feature_col_names,
        )

        classes_df = pl.read_csv(
            CLASSES_FILE,
            has_header=True,
        )

        edgelist_df = pl.read_csv(
            EDGELIST_FILE,
            has_header=True,
        )

        logger.info(
            "raw_data_loaded",
            transactions=len(features_df),
            labeled=len(classes_df),
            edges=len(edgelist_df),
        )

        return features_df, classes_df, edgelist_df

    def merge_labels(
        self,
        features_df: pl.DataFrame,
        classes_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Join features with class labels.
        Maps class strings to integers:
          'illicit' -> 1
          'licit'   -> 2
          'unknown' -> 0
        """
        # Rename for clean join
        classes_renamed = classes_df.rename({"txId": "txId", "class": "label_str"})

        merged = features_df.join(
            classes_renamed,
            on="txId",
            how="left",
        )

        # Map string labels to integers
        merged = merged.with_columns(
            pl.col("label_str")
            .map_elements(
                lambda x: 1 if x == "1" else (2 if x == "2" else 0),
                return_dtype=pl.Int32,
            )
            .alias("label")
        )

        label_counts = merged.group_by("label").len().sort("label")
        logger.info("label_distribution", counts=label_counts.to_dicts())

        return merged

    def chronological_split(
        self,
        df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split by time step — NEVER by random index.

        WHY: Blockchain data is temporal. A random split means your model
        sees transactions from time step 40 during training, then predicts
        transactions from time step 30 during testing. This is impossible
        in production (you can't predict the past) and inflates metrics.
        """
        train_df = df.filter(pl.col("time_step").is_in(TRAIN_TIME_STEPS))
        test_df = df.filter(pl.col("time_step").is_in(TEST_TIME_STEPS))

        logger.info(
            "chronological_split_complete",
            train_size=len(train_df),
            test_size=len(test_df),
            train_time_steps="1-34",
            test_time_steps="35-49",
        )

        return train_df, test_df

    def get_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Return only the feature columns — drop txId, time_step, labels."""
        exclude = {"txId", "time_step", "label", "label_str"}
        return [c for c in df.columns if c not in exclude]

    def normalize_features(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler.
        CRITICAL: fit on training data only, transform both sets.
        """
        feature_cols = self.get_feature_columns(train_df)

        # Convert to numpy for sklearn
        X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
        X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)

        # Fit ONLY on training data
        self.scaler.fit(X_train)

        # Transform both using training statistics
        X_train_scaled = self.scaler.transform(X_train).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)

        logger.info(
            "features_normalized",
            n_features=len(feature_cols),
            train_shape=X_train_scaled.shape,
            test_shape=X_test_scaled.shape,
        )

        return X_train_scaled, X_test_scaled

    def get_labels(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract labels and masks.

        Returns:
          y_train: all labels including unknown (0)
          y_test: all labels including unknown (0)
          train_mask: boolean mask — True where label is known (1 or 2)
          test_mask: boolean mask — True where label is known (1 or 2)

        WHY masks: Unknown samples are excluded from supervised loss
        but kept in graph for message passing.
        """
        y_train = train_df["label"].to_numpy().astype(np.int64)
        y_test = test_df["label"].to_numpy().astype(np.int64)

        # Known = illicit (1) or licit (2)
        train_mask = (y_train == 1) | (y_train == 2)
        test_mask = (y_test == 1) | (y_test == 2)

        # Convert to binary: illicit=1, licit=0 (for binary classification)
        y_train_binary = (y_train == 1).astype(np.int64)
        y_test_binary = (y_test == 1).astype(np.int64)

        logger.info(
            "labels_extracted",
            train_illicit=int(y_train_binary[train_mask].sum()),
            train_licit=int((y_train_binary[train_mask] == 0).sum()),
            test_illicit=int(y_test_binary[test_mask].sum()),
            test_licit=int((y_test_binary[test_mask] == 0).sum()),
        )

        return y_train_binary, y_test_binary, train_mask, test_mask

    def build_edge_index(
        self,
        edgelist_df: pl.DataFrame,
        tx_id_to_idx: dict[int, int],
    ) -> np.ndarray:
        """
        Convert edgelist to PyTorch Geometric edge_index format.
        edge_index shape: [2, num_edges]
        Row 0: source nodes, Row 1: destination nodes

        Filters out edges where either node is not in our node set.
        """
        sources = []
        targets = []

        txid1_col = edgelist_df["txId1"].to_list()
        txid2_col = edgelist_df["txId2"].to_list()

        for src_id, dst_id in zip(txid1_col, txid2_col):
            if src_id in tx_id_to_idx and dst_id in tx_id_to_idx:
                sources.append(tx_id_to_idx[src_id])
                targets.append(tx_id_to_idx[dst_id])

        edge_index = np.array([sources, targets], dtype=np.int64)

        logger.info(
            "edge_index_built",
            total_edges=len(edgelist_df),
            valid_edges=edge_index.shape[1],
        )

        return edge_index

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        edge_index: np.ndarray,
        tx_ids_train: list,
        tx_ids_test: list,
    ) -> None:
        """Save all processed arrays to disk."""
        np.save(PROCESSED_DIR / "X_train.npy", X_train)
        np.save(PROCESSED_DIR / "X_test.npy", X_test)
        np.save(PROCESSED_DIR / "y_train.npy", y_train)
        np.save(PROCESSED_DIR / "y_test.npy", y_test)
        np.save(PROCESSED_DIR / "train_mask.npy", train_mask)
        np.save(PROCESSED_DIR / "test_mask.npy", test_mask)
        np.save(PROCESSED_DIR / "edge_index.npy", edge_index)

        # Save scaler for inference
        joblib.dump(self.scaler, MODELS_DIR / "scaler.joblib")

        # Save metadata
        metadata = {
            "n_features": X_train.shape[1],
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_edges": edge_index.shape[1],
            "train_time_steps": "1-34",
            "test_time_steps": "35-49",
            "feature_dim": X_train.shape[1],
        }
        with open(PROCESSED_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "processed_data_saved",
            directory=str(PROCESSED_DIR),
            files=list(metadata.keys()),
        )

    def process(self) -> dict:
        """
        Run the full preprocessing pipeline end to end.
        Returns metadata dict with dataset statistics.
        """
        logger.info("elliptic_preprocessing_started")

        # 1. Load
        features_df, classes_df, edgelist_df = self.load_raw_data()

        # 2. Merge labels
        merged_df = self.merge_labels(features_df, classes_df)

        # 3. Chronological split
        train_df, test_df = self.chronological_split(merged_df)

        # 4. Build tx_id → index mapping for edge construction
        all_tx_ids = merged_df["txId"].to_list()
        tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}

        # 5. Normalize features
        X_train, X_test = self.normalize_features(train_df, test_df)

        # 6. Extract labels and masks
        y_train, y_test, train_mask, test_mask = self.get_labels(train_df, test_df)

        # 7. Build graph edge index
        edge_index = self.build_edge_index(edgelist_df, tx_id_to_idx)

        # 8. Save everything
        self.save_processed_data(
            X_train,
            X_test,
            y_train,
            y_test,
            train_mask,
            test_mask,
            edge_index,
            train_df["txId"].to_list(),
            test_df["txId"].to_list(),
        )

        metadata = {
            "n_features": X_train.shape[1],
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_edges": edge_index.shape[1],
            "train_illicit": int(y_train[train_mask].sum()),
            "train_licit": int((y_train[train_mask] == 0).sum()),
            "test_illicit": int(y_test[test_mask].sum()),
            "test_licit": int((y_test[test_mask] == 0).sum()),
        }

        logger.info("elliptic_preprocessing_complete", **metadata)
        return metadata


if __name__ == "__main__":
    processor = EllipticProcessor()
    stats = processor.process()
    print("\n=== Preprocessing Complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
