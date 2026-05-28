"""
Dashboard API client — calls the FastAPI backend.
All dashboard data goes through these functions.
Handles connection errors gracefully — dashboard never crashes.
"""

import requests

from config.logging_config import get_logger

logger = get_logger(__name__)

API_BASE = "http://localhost:8000"
TIMEOUT = 5


def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": {}}


def get_alerts(limit: int = 50) -> dict:
    try:
        r = requests.get(f"{API_BASE}/alerts?limit={limit}", timeout=TIMEOUT)
        return r.json()
    except Exception:
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
