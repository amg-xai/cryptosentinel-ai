"""Tests for FastAPI routes."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_model_status():
    response = client.get("/health")
    data = response.json()
    assert "models_loaded" in data
    assert "uptime_seconds" in data


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_analyze_wallet_returns_200():
    response = client.post(
        "/analyze/wallet",
        json={"address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"},
    )
    assert response.status_code == 200


def test_analyze_wallet_has_required_fields():
    response = client.post(
        "/analyze/wallet",
        json={"address": "0xTestAddress"},
    )
    data = response.json()
    assert "composite_score" in data
    assert "action" in data
    assert "severity" in data
    assert "explanation" in data


def test_analyze_wallet_score_in_range():
    response = client.post(
        "/analyze/wallet",
        json={"address": "0xTestAddress"},
    )
    data = response.json()
    assert 0.0 <= data["composite_score"] <= 1.0


def test_analyze_transaction_returns_200():
    response = client.post(
        "/analyze/transaction",
        json={
            "tx_hash": "0xabc123",
            "from_addr": "0xSender",
            "to_addr": "0xRecipient",
            "value_eth": 1.5,
            "gas": 21000,
            "gas_price": 1000000000,
            "is_contract_call": False,
            "is_contract_creation": False,
            "input_data": "0x",
            "block_timestamp": 1000000.0,
            "chain_name": "ethereum-sepolia",
        },
    )
    assert response.status_code == 200


def test_analyze_transaction_has_risk_score():
    response = client.post(
        "/analyze/transaction",
        json={
            "tx_hash": "0xdef456",
            "from_addr": "0xSender2",
            "value_eth": 0.0,
            "gas": 21000,
            "gas_price": 1000000000,
            "block_timestamp": 1000000.0,
            "chain_name": "ethereum-sepolia",
        },
    )
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0


def test_get_alerts_returns_200():
    response = client.get("/alerts")
    assert response.status_code == 200


def test_get_alerts_has_structure():
    response = client.get("/alerts")
    data = response.json()
    assert "total_active" in data
    assert "alerts" in data
    assert "stats" in data


def test_get_critical_alerts_returns_200():
    response = client.get("/alerts/critical")
    assert response.status_code == 200


def test_graph_stats_returns_200():
    response = client.get("/graph/stats/summary")
    assert response.status_code == 200


def test_graph_wallet_returns_200():
    response = client.get("/graph/0xTestAddress")
    assert response.status_code == 200


def test_scan_contract_selfdestruct():
    response = client.post(
        "/scan/contract",
        json={
            "source_code": "pragma solidity ^0.8.0;\ncontract Test {\n  function kill() public { selfdestruct(payable(msg.sender)); }\n}"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_critical"] is True
    assert data["combined_risk_score"] > 0


def test_scan_contract_no_input_returns_400():
    response = client.post("/scan/contract", json={})
    assert response.status_code == 400


def test_scan_safe_contract():
    response = client.post(
        "/scan/contract",
        json={
            "source_code": "pragma solidity ^0.8.0;\ncontract Safe {\n  uint256 public value;\n}"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_critical"] is False
