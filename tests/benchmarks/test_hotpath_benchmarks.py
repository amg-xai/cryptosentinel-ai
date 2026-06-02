"""
Hot-path benchmarks — the paths that actually matter for latency, identified
by load testing + py-spy profiling (see docs/PERFORMANCE.md).

These guard against silent performance regression. CI compares against a
committed baseline and fails if any benchmark regresses beyond the threshold
(see Makefile bench-check / .github CI).

Run:        make bench           (run + save baseline)
Compare:    make bench-check     (fail if regressed > threshold)
"""
import numpy as np
import pytest


# ---- 1. Composite scoring path (made 4.9x faster on Day 54) ----
def test_bench_risk_score(benchmark):
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores
    scorer = CompositeRiskScorer()
    ms = ModelScores(gnn=0.6, autoencoder=0.5, isolation_forest=0.4,
                     graph_centrality=0.3)

    def score():
        return scorer.score("0xabc", "0xtx", ms, value_eth=1.0)

    result = benchmark(score)
    assert result is not None


# ---- 2. Feature extraction (per-transaction, runs on every score) ----
def test_bench_feature_extraction(benchmark):
    from src.ml.tabular.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    payload = {
        "tx_hash": "0x" + "a" * 64,
        "from_addr": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to_addr": "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
        "value_eth": 1.5, "gas": 50000, "gas_price": 20_000_000_000,
        "is_contract_call": True, "is_contract_creation": False,
        "input_data": "0x", "block_timestamp": 1779000000.0,
        "chain_name": "ethereum-sepolia",
    }

    def extract():
        return fe.extract(payload)

    result = benchmark(extract)
    assert result is not None


# ---- 3. Numba velocity (the 83x+ vectorization win) ----
def test_bench_velocity_numba(benchmark):
    from src.ml.tabular.vectorized_features import compute_velocity_numba
    rng = np.random.default_rng(42)
    ts = np.sort(rng.uniform(0, 86400, size=1000)).astype(np.float64)
    # Warm up JIT compile outside the timed region
    compute_velocity_numba(ts, 3600.0)

    def run():
        return compute_velocity_numba(ts, 3600.0)

    result = benchmark(run)
    assert len(result) == 1000
