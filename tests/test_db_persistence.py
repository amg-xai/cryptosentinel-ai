"""Tests for PostgreSQL alert persistence."""
import pytest
from src.db.session import init_db, is_available
from src.db.alert_repository import (
    save_alert,
    get_recent_alerts,
    get_alerts_by_address,
    acknowledge_alert_db,
    get_alert_stats,
    save_scan_result,
)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Initialize DB once for all tests in this module."""
    init_db()


def make_alert_dict(address="0xPyTest", score=0.8, severity="HIGH"):
    return {
        "address": address,
        "tx_hash": "0xtest_tx",
        "composite_score": score,
        "confidence": 0.9,
        "action": "QUARANTINE",
        "severity": severity,
        "value_at_risk_eth": 1.5,
        "model_scores": {"gnn": score},
        "explanation": {"summary": "pytest alert"},
    }


def test_db_initializes():
    """Database should initialize (or skip if unavailable)."""
    result = init_db()
    assert isinstance(result, bool)


def test_save_alert_returns_id():
    if not is_available():
        pytest.skip("Database not available")
    alert_id = save_alert(make_alert_dict())
    assert alert_id is not None
    assert alert_id > 0


def test_get_recent_alerts():
    if not is_available():
        pytest.skip("Database not available")
    save_alert(make_alert_dict())
    alerts = get_recent_alerts(limit=10)
    assert isinstance(alerts, list)
    assert len(alerts) > 0


def test_get_recent_alerts_by_severity():
    if not is_available():
        pytest.skip("Database not available")
    save_alert(make_alert_dict(severity="CRITICAL"))
    critical = get_recent_alerts(limit=10, severity="CRITICAL")
    assert all(a["severity"] == "CRITICAL" for a in critical)


def test_get_alerts_by_address():
    if not is_available():
        pytest.skip("Database not available")
    unique_addr = "0xUniqueTestAddr999"
    save_alert(make_alert_dict(address=unique_addr))
    alerts = get_alerts_by_address(unique_addr)
    assert len(alerts) > 0
    assert all(a["address"] == unique_addr for a in alerts)


def test_acknowledge_alert():
    if not is_available():
        pytest.skip("Database not available")
    addr = "0xAckTestAddr"
    save_alert(make_alert_dict(address=addr))
    result = acknowledge_alert_db(addr, "test_analyst")
    assert result is True
    # Verify acknowledged
    alerts = get_alerts_by_address(addr)
    assert any(a["acknowledged"] for a in alerts)


def test_get_alert_stats():
    if not is_available():
        pytest.skip("Database not available")
    save_alert(make_alert_dict())
    stats = get_alert_stats()
    assert "total" in stats
    assert "by_severity" in stats
    assert stats["total"] > 0


def test_save_scan_result():
    if not is_available():
        pytest.skip("Database not available")
    scan_id = save_scan_result({
        "combined_risk_score": 0.9,
        "has_critical": True,
        "vulnerability_count": {"CRITICAL": 1},
        "vulnerabilities": [],
        "scan_duration_ms": 5.0,
    }, contract_hash="0xcontract")
    assert scan_id is not None
    assert scan_id > 0


def test_graceful_degradation_when_unavailable():
    """Functions should return safe defaults when DB unavailable."""
    # These functions check is_available() internally
    # If DB is down, they return [] or None, never crash
    alerts = get_recent_alerts()
    assert isinstance(alerts, list)
