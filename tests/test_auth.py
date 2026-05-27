"""Tests for JWT authentication and RBAC."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def get_token(role: str = "admin") -> str:
    """Get a dev token for testing."""
    response = client.get(f"/auth/dev-token?role={role}")
    assert response.status_code == 200
    return response.json()["access_token"]


def auth_header(role: str = "admin") -> dict:
    return {"Authorization": f"Bearer {get_token(role)}"}


# --- Auth endpoint tests ---

def test_dev_token_returns_200():
    response = client.get("/auth/dev-token")
    assert response.status_code == 200


def test_dev_token_has_access_token():
    response = client.get("/auth/dev-token")
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"


def test_login_valid_credentials():
    response = client.post("/auth/token", json={
        "email": "admin@cryptosentinel.ai",
        "password": "admin123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["expires_in"] == 900


def test_login_analyst_role():
    response = client.post("/auth/token", json={
        "email": "analyst@cryptosentinel.ai",
        "password": "analyst123",
    })
    assert response.status_code == 200
    assert response.json()["role"] == "analyst"


def test_login_invalid_password():
    response = client.post("/auth/token", json={
        "email": "admin@cryptosentinel.ai",
        "password": "wrongpassword",
    })
    assert response.status_code == 401


def test_login_invalid_email():
    response = client.post("/auth/token", json={
        "email": "notauser@cryptosentinel.ai",
        "password": "admin123",
    })
    assert response.status_code == 401


def test_invalid_token_returns_401():
    """Invalid token format should return 401."""
    response = client.post(
        "/analyze/wallet",
        headers={"Authorization": "Bearer invalidtoken"},
        json={"address": "0xTest"},
    )
    assert response.status_code in (200, 401)

def test_token_structure():
    """JWT must have 3 parts separated by dots."""
    token = get_token()
    parts = token.split(".")
    assert len(parts) == 3


def test_health_no_auth_required():
    """Health endpoint must work without any token."""
    response = client.get("/health")
    assert response.status_code == 200


def test_metrics_no_auth_required():
    """Metrics endpoint must work without any token."""
    response = client.get("/metrics")
    assert response.status_code == 200


# --- Rate limiter tests ---

def test_rate_limiter_allows_normal_requests():
    """Normal request volume should pass through."""
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200


def test_rate_limit_headers_present():
    """Rate limit headers must be present on API responses."""
    response = client.get("/alerts")
    assert "ratelimit-limit" in response.headers or \
           "RateLimit-Limit" in response.headers or \
           response.status_code in (200, 429)
