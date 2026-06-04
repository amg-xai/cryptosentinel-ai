"""
Dashboard API client — calls the FastAPI backend.
All dashboard data goes through these functions.
Handles connection errors gracefully — dashboard never crashes.
"""

import requests

from config.logging_config import get_logger

logger = get_logger(__name__)

import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 5

# Dashboard authenticates with a dev token (dashboard is an internal SOC
# tool). Cached after first fetch. Protected routes require this post-auth.
_token_cache = {"token": None}


def _auth_headers() -> dict:
    if _token_cache["token"] is None:
        try:
            r = requests.get(f"{API_BASE}/auth/dev-token?role=admin", timeout=TIMEOUT)
            _token_cache["token"] = r.json().get("access_token", "")
        except Exception:
            _token_cache["token"] = ""
    return {"Authorization": f"Bearer {_token_cache['token']}"}


def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": {}}


def get_alerts(limit: int = 50) -> dict:
    """Pull alerts from the PostgreSQL-backed history (shared across
    processes), not the API's in-memory list which is empty when the
    pipeline runs as a separate process."""
    try:
        h = _auth_headers()
        # DB-backed recent alerts + DB stats (the real, persisted data)
        hist = requests.get(f"{API_BASE}/alerts/history?limit={limit}",
                            headers=h, timeout=TIMEOUT).json()
        stats = requests.get(f"{API_BASE}/alerts/stats/database",
                             headers=h, timeout=TIMEOUT).json()
        alerts = hist.get("alerts", hist if isinstance(hist, list) else [])
        return {
            "total_active": stats.get("total", 0),
            "alerts": alerts,
            "stats": stats,
        }
    except Exception as e:
        logger.warning("dashboard_get_alerts_failed", error=str(e))
        return {"total_active": 0, "alerts": [], "stats": {}}


def get_critical_alerts() -> dict:
    try:
        r = requests.get(f"{API_BASE}/alerts/critical", timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {"count": 0, "alerts": []}


def analyze_wallet(address: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/analyze/wallet",
            json={"address": address, "include_graph": True},
            timeout=TIMEOUT,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_graph_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/graph/stats/summary", timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {"num_nodes": 0, "num_edges": 0}


def get_wallet_graph(address: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/graph/{address}", timeout=TIMEOUT)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def scan_contract(source_code: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/scan/contract",
            json={"source_code": source_code},
            timeout=30,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def acknowledge_alert(address: str, analyst: str = "dashboard_user") -> bool:
    try:
        r = requests.post(
            f"{API_BASE}/alerts/{address}/acknowledge?analyst={analyst}",
            timeout=TIMEOUT,
        )
        return r.status_code == 200
    except Exception:
        return False
