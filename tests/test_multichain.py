"""Tests for multi-chain support."""
import pytest
from src.blockchain.chains import (
    ChainConfig, get_chains, get_configured_chains,
)
from src.blockchain.multichain import should_sample


def test_get_chains_returns_both():
    chains = get_chains()
    assert "ethereum-sepolia" in chains
    assert "polygon-mainnet" in chains


def test_chain_config_fields():
    chains = get_chains()
    eth = chains["ethereum-sepolia"]
    assert eth.native_symbol == "ETH"
    assert eth.chain_id == 11155111
    poly = chains["polygon-mainnet"]
    assert poly.native_symbol == "MATIC"
    assert poly.chain_id == 137


def test_polygon_has_sampling():
    """Polygon mainnet should sample below 1.0 for volume control."""
    poly = get_chains()["polygon-mainnet"]
    assert poly.sample_rate < 1.0
    assert poly.sample_rate > 0.0


def test_ethereum_no_sampling():
    """Sepolia is low-volume, scores everything."""
    eth = get_chains()["ethereum-sepolia"]
    assert eth.sample_rate == 1.0


def test_is_configured_with_real_url():
    chain = ChainConfig(
        name="test", http_url="https://real.example.com/v2/key",
        native_symbol="X", block_time_s=1.0, explorer="",
    )
    assert chain.is_configured is True


def test_is_configured_with_localhost_default():
    chain = ChainConfig(
        name="test", http_url="https://localhost",
        native_symbol="X", block_time_s=1.0, explorer="",
    )
    assert chain.is_configured is False


def test_should_sample_always_true_at_rate_1():
    chain = ChainConfig(
        name="t", http_url="x", native_symbol="X",
        block_time_s=1.0, explorer="", sample_rate=1.0,
    )
    # 100 calls should all be True
    assert all(should_sample(chain) for _ in range(100))


def test_should_sample_distribution():
    """At sample_rate=0.2, roughly 20% should pass over many trials."""
    chain = ChainConfig(
        name="t", http_url="x", native_symbol="X",
        block_time_s=1.0, explorer="", sample_rate=0.2,
    )
    results = [should_sample(chain) for _ in range(1000)]
    passed = sum(results)
    # Allow wide margin: 100-300 out of 1000
    assert 100 <= passed <= 300


def test_get_configured_chains_excludes_unconfigured():
    """Only chains with real URLs are returned."""
    configured = get_configured_chains()
    for name, chain in configured.items():
        assert chain.is_configured
