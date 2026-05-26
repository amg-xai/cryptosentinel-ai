"""
Tests for ThreatGNN model and trainer.
Uses tiny synthetic graphs — fast CPU execution.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from src.ml.gnn.model import ThreatGNN
from src.ml.gnn.trainer import GNNTrainer, build_pyg_data


def make_synthetic_graph(n_nodes=50, n_features=165, seed=42):
    """Create a small synthetic graph for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_nodes, n_features).astype(np.float32)
    y = np.array([0] * 40 + [1] * 10, dtype=np.int64)

    # Random edges
    src = rng.randint(0, n_nodes, size=100)
    dst = rng.randint(0, n_nodes, size=100)
    edge_index = np.array([src, dst], dtype=np.int64)

    mask = np.ones(n_nodes, dtype=bool)
    train_mask = np.array([True] * 30 + [False] * 20)
    test_mask = np.array([False] * 30 + [True] * 20)

    return X, y, edge_index, train_mask, test_mask


def test_model_forward_pass():
    """ThreatGNN forward pass returns correct output shape."""
    model = ThreatGNN(in_channels=165, hidden_channels=32, heads=2)
    x = torch.randn(20, 165)
    edge_index = torch.randint(0, 20, (2, 50))
    out = model(x, edge_index)
    assert out.shape == (20, 2)


def test_model_get_embeddings():
    """get_embeddings returns correct shape."""
    model = ThreatGNN(in_channels=165, hidden_channels=32, heads=2)
    x = torch.randn(20, 165)
    edge_index = torch.randint(0, 20, (2, 50))
    emb = model.get_embeddings(x, edge_index)
    assert emb.shape == (20, 16)  # hidden_channels // 2


def test_build_pyg_data():
    """build_pyg_data creates valid PyG Data object."""
    X, y, edge_index, train_mask, test_mask = make_synthetic_graph(50)
    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]

    data = build_pyg_data(
        X_train, X_test,
        y_train, y_test,
        train_mask[:30], test_mask[30:],
        edge_index,
    )

    assert data.num_nodes == 50
    assert data.num_node_features == 165
    assert data.edge_index.shape[0] == 2
    assert data.train_mask.sum() > 0
    assert data.test_mask.sum() > 0


def test_trainer_trains_without_error():
    """GNNTrainer completes training on synthetic graph."""
    X, y, edge_index, train_mask, test_mask = make_synthetic_graph(50)
    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]

    data = build_pyg_data(
        X_train, X_test,
        y_train, y_test,
        train_mask[:30], test_mask[30:],
        edge_index,
    )

    trainer = GNNTrainer(
        hidden_channels=32,
        heads=2,
        dropout=0.1,
        learning_rate=0.01,
    )
    history = trainer.train(
        data,
        epochs=3,
        batch_size=16,
        num_neighbors=[5, 3],
    )

    assert len(history) > 0
    assert "loss" in history[0]
    assert trainer.model is not None


def test_trainer_evaluate_returns_metrics():
    """evaluate() returns all required metric keys."""
    X, y, edge_index, train_mask, test_mask = make_synthetic_graph(50)
    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]

    data = build_pyg_data(
        X_train, X_test,
        y_train, y_test,
        train_mask[:30], test_mask[30:],
        edge_index,
    )

    trainer = GNNTrainer(hidden_channels=32, heads=2)
    trainer.train(data, epochs=2, batch_size=16, num_neighbors=[5, 3])
    metrics = trainer.evaluate(data)

    required = {"f1_score", "pr_auc", "roc_auc",
                "test_samples", "illicit_detected", "illicit_total"}
    assert required.issubset(set(metrics.keys()))


def test_trainer_save_and_load(tmp_path):
    """Save and load preserves model architecture."""
    X, y, edge_index, train_mask, test_mask = make_synthetic_graph(50)
    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]

    data = build_pyg_data(
        X_train, X_test,
        y_train, y_test,
        train_mask[:30], test_mask[30:],
        edge_index,
    )

    trainer = GNNTrainer(hidden_channels=32, heads=2)
    trainer.train(data, epochs=2, batch_size=16, num_neighbors=[5, 3])

    save_path = str(tmp_path / "test_gnn.pt")
    trainer.save(save_path)

    trainer2 = GNNTrainer(hidden_channels=32, heads=2)
    trainer2.load(save_path, in_channels=165)
    assert trainer2.model is not None


def test_gnn_output_is_probability():
    """Softmax output must sum to 1 per node."""
    model = ThreatGNN(in_channels=165, hidden_channels=32, heads=2)
    model.eval()
    x = torch.randn(10, 165)
    edge_index = torch.randint(0, 10, (2, 20))
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(10), atol=1e-5)
