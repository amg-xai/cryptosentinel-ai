"""Tests for GNN explainability and SHAP integration."""
import pytest
import numpy as np
from src.ml.explainability.gnn_explainer import RiskExplainer, LIVE_FEATURE_NAMES


def test_explainer_initializes():
    explainer = RiskExplainer()
    assert explainer is not None
    assert explainer._loaded is False


def test_explain_without_loading_returns_error():
    explainer = RiskExplainer()
    features = np.random.randn(16).astype(np.float32)
    result = explainer.explain_transaction(features)
    assert "error" in result


def test_top_features_returns_list():
    explainer = RiskExplainer()
    features = np.array([1.0, 0.0, 2.0, 0.5, 0.0, 3.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float32)
    names = LIVE_FEATURE_NAMES
    result = explainer._get_top_features(features, names)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "feature" in result[0]
    assert "value" in result[0]


def test_top_features_sorted_by_magnitude():
    explainer = RiskExplainer()
    features = np.array([0.1, 0.5, 3.0, 0.2, 0.8, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    names = [f"f{i}" for i in range(16)]
    result = explainer._get_top_features(features, names)
    magnitudes = [r["abs_magnitude"] for r in result]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_live_feature_names_correct_count():
    assert len(LIVE_FEATURE_NAMES) == 16


def test_explain_api_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    response = client.post("/explain/transaction", json={
        "tx_hash": "0xtest",
        "from_addr": "0xSender",
        "value_eth": 1.0,
        "gas": 21000,
        "gas_price": 1000000000,
        "block_timestamp": 1000000.0,
        "chain_name": "ethereum-sepolia",
    })
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "tx_hash" in data


def test_explain_returns_top_risk_features():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    response = client.post("/explain/transaction", json={
        "tx_hash": "0xtest2",
        "from_addr": "0xSender2",
        "value_eth": 5.0,
        "gas": 100000,
        "gas_price": 50000000000,
        "is_contract_call": True,
        "block_timestamp": 1000000.0,
        "chain_name": "ethereum-sepolia",
    })
    assert response.status_code == 200
    data = response.json()
    explanation = data.get("explanation", {})
    assert "top_risk_features" in explanation
    assert isinstance(explanation["top_risk_features"], list)
