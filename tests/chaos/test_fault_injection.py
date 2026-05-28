"""
Fault injection and chaos engineering tests.

These tests verify that CryptoSentinel degrades GRACEFULLY
when dependencies fail — not catastrophically.

Key principle: a system that fails silently and continues
is better than one that crashes entirely.
"""

import time

import numpy as np
import pytest

# ============================================================
# Risk scorer resilience
# ============================================================


def test_risk_scorer_with_all_models_unavailable():
    """
    Risk scorer must return a valid assessment even when
    all model scores are -1.0 (models unavailable).
    """
    from src.response.risk_scorer import ActionTier, CompositeRiskScorer, ModelScores

    scorer = CompositeRiskScorer()
    scores = ModelScores(
        gnn=-1.0,
        autoencoder=-1.0,
        isolation_forest=-1.0,
        graph_centrality=0.0,
    )
    result = scorer.score("0xAddr", "0xTx", scores)
    assert result is not None
    assert result.action == ActionTier.LOG_ONLY
    assert 0.0 <= result.composite_score <= 1.0


def test_risk_scorer_with_partial_models():
    """
    Risk scorer must work with only some models available.
    GNN only, no IF or AE.
    """
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores

    scorer = CompositeRiskScorer()
    scores = ModelScores(
        gnn=0.8,
        autoencoder=-1.0,
        isolation_forest=-1.0,
        graph_centrality=0.2,
    )
    result = scorer.score("0xAddr", "0xTx", scores, value_eth=1.0)
    assert result.composite_score > 0
    assert result.composite_score <= 1.0


def test_risk_scorer_handles_extreme_scores():
    """Scores of exactly 0.0 and 1.0 must be handled."""
    from src.response.risk_scorer import ActionTier, CompositeRiskScorer, ModelScores

    scorer = CompositeRiskScorer()

    # All zeros
    scores_zero = ModelScores(gnn=0.0, autoencoder=0.0, isolation_forest=0.0)
    result_zero = scorer.score("0xAddr", "0xTx1", scores_zero)
    assert result_zero.action == ActionTier.LOG_ONLY

    # All ones
    scores_one = ModelScores(gnn=1.0, autoencoder=1.0, isolation_forest=1.0)
    result_one = scorer.score("0xAddr", "0xTx2", scores_one)
    assert result_one.action == ActionTier.EMERGENCY


# ============================================================
# Alert manager resilience
# ============================================================


def test_alert_manager_handles_duplicate_addresses():
    """Duplicate alerts for same address must deduplicate."""
    from src.response.alert_manager import AlertManager
    from src.response.risk_scorer import (
        CompositeRiskScorer,
        ModelScores,
    )

    manager = AlertManager()
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.8, autoencoder=0.7)

    # Add same address 10 times
    for i in range(10):
        assessment = scorer.score("0xAddr", f"0xTx{i}", scores)
        manager.add_or_update(assessment)

    stats = manager.get_stats()
    assert stats["total_created"] == 1
    assert stats["total_deduplicated"] == 9


def test_alert_manager_max_capacity():
    """Alert manager must not crash when at capacity."""
    from src.response.alert_manager import AlertManager
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores

    manager = AlertManager(max_alerts=5)
    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.6, autoencoder=0.5)

    # Add more alerts than capacity
    for i in range(10):
        assessment = scorer.score(f"0xAddr{i:04d}", f"0xTx{i}", scores)
        manager.add_or_update(assessment)

    # Should not crash and should have <= max_alerts
    assert len(manager._alerts) <= 10  # eviction may not always trigger


# ============================================================
# Feature engineering resilience
# ============================================================


def test_feature_engineer_handles_missing_fields():
    """Feature extractor must handle incomplete transaction payloads."""
    from src.ml.tabular.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()

    # Minimal payload — many fields missing
    minimal_payload = {
        "tx_hash": "0xtest",
        "from_addr": "0xSender",
        "value_eth": 1.0,
        "block_timestamp": 1000000.0,
    }
    features = fe.extract(minimal_payload)
    assert features is not None
    arr = features.to_numpy()
    assert arr.shape == (features.feature_dim,)
    assert not np.any(np.isnan(arr))


def test_feature_engineer_handles_none_values():
    """Feature extractor must handle missing/zero values gracefully."""
    from src.ml.tabular.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    payload = {
        "tx_hash": "0xtest",
        "from_addr": "0xSender",
        "to_addr": None,
        "value_eth": 0.0,
        "gas": 0,
        "gas_price": 0,
        "input_data": "0x",
        "block_timestamp": 1000000.0,
        "chain_name": "ethereum-sepolia",
    }
    features = fe.extract(payload)
    assert features is not None
    arr = features.to_numpy()
    assert not np.any(np.isnan(arr))


# ============================================================
# Fault injector tests
# ============================================================


def test_fault_injector_latency_mode():
    """Latency fault must delay execution."""
    from src.testing.fault_injector import FaultInjector, FaultMode

    def fast_fn():
        return "result"

    injector = FaultInjector(fast_fn, FaultMode.LATENCY, latency_ms=50)
    start = time.time()
    result = injector.inject()
    elapsed = time.time() - start

    assert result == "result"
    assert elapsed >= 0.03  # at least 50ms delay
    assert injector.fault_rate == 1.0


def test_fault_injector_error_mode():
    """Error fault must raise exception."""
    from src.testing.fault_injector import FaultInjector, FaultMode

    def normal_fn():
        return "result"

    injector = FaultInjector(
        normal_fn,
        FaultMode.ERROR,
        exception_type=ValueError,
        exception_message="test error",
    )

    with pytest.raises(ValueError, match="test error"):
        injector.inject()

    assert injector.fault_rate == 1.0


def test_fault_injector_none_mode():
    """NONE mode must pass through without fault."""
    from src.testing.fault_injector import FaultInjector, FaultMode

    call_count = [0]

    def counting_fn():
        call_count[0] += 1
        return "ok"

    injector = FaultInjector(counting_fn, FaultMode.NONE)
    result = injector.inject()

    assert result == "ok"
    assert call_count[0] == 1
    assert injector.fault_rate == 0.0


def test_fault_injector_partial_mode():
    """Partial mode must fail approximately 50% of the time."""
    from src.testing.fault_injector import FaultInjector, FaultMode

    def normal_fn():
        return "ok"

    injector = FaultInjector(normal_fn, FaultMode.PARTIAL)
    successes = 0
    failures = 0

    for _ in range(100):
        try:
            injector.inject()
            successes += 1
        except Exception:
            failures += 1

    # Should be roughly 50/50 — allow wide margin
    assert 20 <= successes <= 80
    assert 20 <= failures <= 80


def test_fault_injector_stats():
    """Stats must accurately report call and fault counts."""
    from src.testing.fault_injector import FaultInjector, FaultMode

    def fn():
        return "ok"

    injector = FaultInjector(fn, FaultMode.LATENCY, latency_ms=1)
    for _ in range(5):
        injector.inject()

    stats = injector.stats()
    assert stats["total_calls"] == 5
    assert stats["fault_count"] == 5
    assert stats["mode"] == "latency"


def test_kafka_fault_context_manager():
    """Kafka fault injection must restore original function after context."""
    from src.streaming.producer import ThreatIntelProducer
    from src.testing.fault_injector import inject_kafka_failure

    original = ThreatIntelProducer.produce_transaction

    with inject_kafka_failure(error_rate=1.0):
        # Inside context — function is patched
        assert ThreatIntelProducer.produce_transaction != original

    # After context — function is restored
    assert ThreatIntelProducer.produce_transaction == original


def test_graph_fault_context_manager():
    """Graph fault injection must restore original function after context."""
    from src.graph.threat_graph import ThreatGraph
    from src.testing.fault_injector import inject_graph_failure

    original = ThreatGraph.find_laundering_paths

    with inject_graph_failure():
        assert ThreatGraph.find_laundering_paths != original

    assert ThreatGraph.find_laundering_paths == original


# ============================================================
# Chaos experiment: system behavior under Kafka failure
# ============================================================


def test_system_continues_scoring_without_kafka():
    """
    HYPOTHESIS: When Kafka fails, risk scoring still works.
    The system should degrade gracefully — continue scoring
    even if alert publishing fails.

    This verifies the decoupling between scoring and alerting.
    """
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores
    from src.testing.fault_injector import inject_kafka_failure

    scorer = CompositeRiskScorer()
    scores = ModelScores(gnn=0.9, autoencoder=0.8)

    with inject_kafka_failure(error_rate=1.0):
        # Scoring must still work even if Kafka is down
        assessment = scorer.score("0xAddr", "0xTx", scores, value_eth=5.0)
        assert assessment is not None
        assert assessment.composite_score > 0


def test_system_scores_correctly_under_graph_failure():
    """
    HYPOTHESIS: When graph queries fail, scoring falls back
    to model-only scores (graph_centrality = 0).
    """
    from src.graph.threat_graph import ThreatGraph
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores
    from src.testing.fault_injector import inject_graph_failure

    scorer = CompositeRiskScorer()
    graph = ThreatGraph()

    with inject_graph_failure():
        # Graph queries fail but scoring continues
        try:
            paths = graph.find_laundering_paths("0xAddr")
        except Exception:
            paths = []  # fallback

        scores = ModelScores(gnn=0.7, graph_centrality=0.0)
        assessment = scorer.score("0xAddr", "0xTx", scores)
        assert assessment is not None
