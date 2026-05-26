"""
Tests for Elliptic dataset preprocessor.
Uses small synthetic data — no real CSV files needed for unit tests.
"""

import numpy as np
import polars as pl

from src.ml.tabular.elliptic_processor import EllipticProcessor


def make_synthetic_features(n_rows: int = 20) -> pl.DataFrame:
    """Create a minimal synthetic features dataframe."""
    data = {
        "txId": list(range(1, n_rows + 1)),
        "time_step": [1] * 10 + [40] * 10,  # 10 train, 10 test
    }
    for i in range(1, 166):
        data[f"feat_{i}"] = [float(i) * 0.1] * n_rows
    return pl.DataFrame(data)


def make_synthetic_classes(n_rows: int = 20) -> pl.DataFrame:
    """Create synthetic class labels."""
    return pl.DataFrame(
        {
            "txId": list(range(1, n_rows + 1)),
            "class": (["1"] * 5 + ["2"] * 10 + ["unknown"] * 5),
        }
    )


def test_chronological_split():
    """Train must contain only early time steps, test only later ones."""
    processor = EllipticProcessor()
    features = make_synthetic_features()
    classes = make_synthetic_classes()
    merged = processor.merge_labels(features, classes)
    train_df, test_df = processor.chronological_split(merged)

    assert all(t in range(1, 35) for t in train_df["time_step"].to_list())
    assert all(t in range(35, 50) for t in test_df["time_step"].to_list())


def test_no_temporal_leakage():
    """Train and test time steps must not overlap."""
    processor = EllipticProcessor()
    features = make_synthetic_features()
    classes = make_synthetic_classes()
    merged = processor.merge_labels(features, classes)
    train_df, test_df = processor.chronological_split(merged)

    train_steps = set(train_df["time_step"].to_list())
    test_steps = set(test_df["time_step"].to_list())
    assert len(train_steps & test_steps) == 0


def test_label_mapping():
    """Class '1' -> illicit=1, class '2' -> licit=0, unknown -> excluded."""
    processor = EllipticProcessor()
    features = make_synthetic_features_n(6)
    classes = pl.DataFrame(
        {
            "txId": list(range(100, 106)),
            "class": ["1", "2", "unknown", "1", "2", "unknown"],
        }
    )
    merged = processor.merge_labels(features, classes)
    labels = merged["label"].to_list()
    assert labels[0] == 1  # illicit
    assert labels[1] == 2  # licit
    assert labels[2] == 0  # unknown


def test_scaler_fit_on_train_only():
    """Scaler must be fit on training data, not test data."""
    processor = EllipticProcessor()
    features = make_synthetic_features()
    classes = make_synthetic_classes()
    merged = processor.merge_labels(features, classes)
    train_df, test_df = processor.chronological_split(merged)
    X_train, X_test = processor.normalize_features(train_df, test_df)

    # Scaler should be fitted — mean and scale should be set
    assert processor.scaler.mean_ is not None
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32


def test_feature_dimensions():
    """Feature arrays must have 165 columns."""
    processor = EllipticProcessor()
    features = make_synthetic_features()
    classes = make_synthetic_classes()
    merged = processor.merge_labels(features, classes)
    train_df, test_df = processor.chronological_split(merged)
    X_train, X_test = processor.normalize_features(train_df, test_df)

    assert X_train.shape[1] == 165
    assert X_test.shape[1] == 165


def test_edge_index_shape():
    """Edge index must have shape [2, n_edges]."""
    processor = EllipticProcessor()
    edgelist = pl.DataFrame(
        {
            "txId1": [1, 2, 3],
            "txId2": [2, 3, 1],
        }
    )
    tx_id_to_idx = {1: 0, 2: 1, 3: 2}
    edge_index = processor.build_edge_index(edgelist, tx_id_to_idx)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 3


def test_unknown_labels_excluded_from_mask():
    """Unknown samples must be False in the training mask."""
    processor = EllipticProcessor()
    features = make_synthetic_features_n(6)
    classes = pl.DataFrame(
        {
            "txId": list(range(100, 106)),
            "class": ["1", "2", "unknown", "1", "2", "unknown"],
        }
    )
    merged = processor.merge_labels(features, classes)
    y, _, mask, _ = processor.get_labels(merged, merged)
    # Indices 2 and 5 are unknown — must be False in mask
    assert mask[2] == False
    assert mask[5] == False
    # Indices 0 and 3 are illicit — must be True in mask
    assert mask[0] == True
    assert mask[3] == True


def make_synthetic_features_n(n_rows: int) -> pl.DataFrame:
    """Create synthetic features with exactly n_rows, all in time_step=1."""
    data = {
        "txId": list(range(100, 100 + n_rows)),
        "time_step": [1] * n_rows,
    }
    for i in range(1, 166):
        data[f"feat_{i}"] = [float(i) * 0.1] * n_rows
    return pl.DataFrame(data)
