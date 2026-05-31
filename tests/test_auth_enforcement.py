"""Auth enforcement + JWT refresh/revocation tests."""
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app)


def _login(client):
    r = client.post("/auth/token", json={
        "email": "analyst@cryptosentinel.ai", "password": "analyst123"})
    assert r.status_code == 200
    return r.json()


def test_login_returns_token_pair(client):
    data = _login(client)
    assert data["access_token"]
    assert data["refresh_token"]


def test_protected_route_requires_auth(client):
    # No Authorization header -> 401
    r = client.post("/analyze/wallet", json={"address": "0xabc"})
    assert r.status_code == 401


def test_valid_token_allows_access(client):
    token = _login(client)["access_token"]
    r = client.post("/analyze/wallet", json={"address": "0xabc"},
                    headers={"Authorization": f"Bearer {token}"})
    # 200 (or 422 if body invalid) — the point is it's NOT 401
    assert r.status_code != 401


def test_logout_revokes_token(client):
    token = _login(client)["access_token"]
    # works before logout
    r1 = client.post("/analyze/wallet", json={"address": "0xabc"},
                     headers={"Authorization": f"Bearer {token}"})
    assert r1.status_code != 401
    # logout revokes
    rlo = client.post("/auth/logout",
                      headers={"Authorization": f"Bearer {token}"})
    assert rlo.status_code == 200
    # now rejected
    r2 = client.post("/analyze/wallet", json={"address": "0xabc"},
                     headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 401


def test_refresh_issues_new_access_token(client):
    refresh = _login(client)["refresh_token"]
    r = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_refresh_rejects_access_token(client):
    """An access token must not be usable as a refresh token."""
    access = _login(client)["access_token"]
    r = client.post("/auth/refresh", json={"refresh_token": access})
    assert r.status_code == 401


def test_auth_routes_stay_open(client):
    """Login must be reachable without a token."""
    r = client.post("/auth/token", json={
        "email": "analyst@cryptosentinel.ai", "password": "analyst123"})
    assert r.status_code == 200
