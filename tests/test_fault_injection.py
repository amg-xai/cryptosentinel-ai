"""Fault injection + graceful degradation tests."""
import time
import asyncio
import pytest
from src.testing.fault_injector import (
    FaultInjector, FaultMode, inject_graph_failure,
)


def _ok():
    return "ok"


def test_fault_none_passthrough():
    fi = FaultInjector(_ok, FaultMode.NONE)
    assert fi.inject() == "ok"


def test_fault_latency_delays():
    fi = FaultInjector(_ok, FaultMode.LATENCY, latency_ms=50)
    t0 = time.perf_counter()
    assert fi.inject() == "ok"
    assert (time.perf_counter() - t0) >= 0.045


def test_fault_error_raises():
    fi = FaultInjector(_ok, FaultMode.ERROR, error_rate=1.0)
    with pytest.raises(Exception):
        fi.inject()


def test_fault_partial_mixed():
    fi = FaultInjector(_ok, FaultMode.PARTIAL)
    ok = fail = 0
    for _ in range(200):
        try:
            fi.inject()
            ok += 1
        except Exception:
            fail += 1
    assert ok > 0 and fail > 0
    assert 0.3 < fi.fault_rate < 0.7


def test_fault_stats():
    fi = FaultInjector(_ok, FaultMode.ERROR, error_rate=1.0)
    for _ in range(5):
        try:
            fi.inject()
        except Exception:
            pass
    s = fi.stats()
    assert s["total_calls"] == 5
    assert s["fault_count"] == 5


def test_graph_endpoint_degrades_gracefully():
    from src.graph.threat_graph import ThreatGraph
    from src.api.routes.graph import get_wallet_graph

    g = ThreatGraph()
    g.add_transaction("0xtest", "0xa", "0xb", 1.0, 1779000000.0, 1, False)

    r = asyncio.run(get_wallet_graph("0xa", max_depth=3, threat_graph=g))
    assert r.degraded is False

    with inject_graph_failure():
        r = asyncio.run(get_wallet_graph("0xa", max_depth=3, threat_graph=g))
    assert r.degraded is True
    assert r.wallet_stats is not None
    assert r.laundering_paths == []
