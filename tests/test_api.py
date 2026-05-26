"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_healthy():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cryptosentinel-api"
    assert "uptime_seconds" in data


def test_metrics_endpoint_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_returns_prometheus_format():
    response = client.get("/metrics")
    # Prometheus format always contains TYPE and HELP lines
    assert "# HELP" in response.text
    assert "# TYPE" in response.text


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "health" in data
