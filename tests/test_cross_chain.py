"""Tests for cross-chain correlation and bridge detection."""
from src.graph.cross_chain import (
    CrossChainAnalyzer, is_bridge, bridge_name,
    is_bridged_asset, bridged_asset_name,
    KNOWN_BRIDGES, BRIDGED_ASSETS,
)

ROOT_CHAIN_MGR = "0xA0c68C638235ee32657e8f720a23ceC1bFc77C77"
USDC_E = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"


def test_is_bridge_known():
    assert is_bridge(ROOT_CHAIN_MGR) is True
    assert bridge_name(ROOT_CHAIN_MGR) == "Polygon PoS: RootChainManager"


def test_is_bridge_case_insensitive():
    assert is_bridge(ROOT_CHAIN_MGR.lower()) is True
    assert is_bridge(ROOT_CHAIN_MGR.upper()) is True


def test_is_bridge_unknown():
    assert is_bridge("0xdeadbeef") is False
    assert is_bridge(None) is False
    assert bridge_name(None) is None


def test_is_bridged_asset():
    assert is_bridged_asset(USDC_E) is True
    assert "USDC" in bridged_asset_name(USDC_E)
    assert is_bridged_asset("0xnope") is False


def test_single_chain_no_flags():
    a = CrossChainAnalyzer()
    flags = a.observe("0xA", "0xB", "polygon-mainnet")
    assert "cross_chain_actor" not in flags
    assert "bridge_interaction" not in flags


def test_cross_chain_actor_detection():
    a = CrossChainAnalyzer()
    a.observe("0xA", "0xB", "polygon-mainnet")
    flags = a.observe("0xA", "0xC", "ethereum-sepolia")
    assert "cross_chain_actor" in flags
    assert a.is_cross_chain_actor("0xA") is True
    assert set(a.chains_for("0xA")) == {"polygon-mainnet", "ethereum-sepolia"}


def test_bridge_interaction_detection():
    a = CrossChainAnalyzer()
    flags = a.observe("0xA", ROOT_CHAIN_MGR, "ethereum-sepolia")
    assert flags["bridge_interaction"] == "Polygon PoS: RootChainManager"
    assert flags["bridge_actor"] == "0xA"
    assert a.bridge_interaction_count("0xA") == 1


def test_bridged_asset_detection():
    a = CrossChainAnalyzer()
    flags = a.observe("0xA", USDC_E, "polygon-mainnet")
    assert "bridged_asset" in flags


def test_risk_boost_levels():
    a = CrossChainAnalyzer()
    # No signal
    assert a.cross_chain_risk_boost("0xNobody") == 1.0
    # Bridged asset only -> 1.08
    a.observe("0xAsset", USDC_E, "polygon-mainnet")
    assert a.cross_chain_risk_boost("0xAsset") == 1.08
    # Cross-chain only -> 1.15
    a.observe("0xCross", "0xX", "polygon-mainnet")
    a.observe("0xCross", "0xY", "ethereum-sepolia")
    assert a.cross_chain_risk_boost("0xCross") == 1.15
    # Bridge only -> 1.25
    a.observe("0xBridge", ROOT_CHAIN_MGR, "polygon-mainnet")
    assert a.cross_chain_risk_boost("0xBridge") == 1.25


def test_risk_boost_max_cross_and_bridge():
    a = CrossChainAnalyzer()
    a.observe("0xHop", ROOT_CHAIN_MGR, "polygon-mainnet")
    a.observe("0xHop", "0xZ", "ethereum-sepolia")
    # Both cross-chain AND bridge -> 1.4
    assert a.cross_chain_risk_boost("0xHop") == 1.4


def test_get_cross_chain_actors():
    a = CrossChainAnalyzer()
    a.observe("0xMulti", "0xB", "polygon-mainnet")
    a.observe("0xMulti", "0xC", "ethereum-sepolia")
    a.observe("0xSingle", "0xD", "polygon-mainnet")
    actors = a.get_cross_chain_actors()
    assert "0xMulti" in actors
    assert "0xSingle" not in actors


def test_stats_structure():
    a = CrossChainAnalyzer()
    a.observe("0xA", USDC_E, "polygon-mainnet")
    s = a.stats()
    assert s["known_bridges"] == len(KNOWN_BRIDGES)
    assert s["known_bridged_assets"] == len(BRIDGED_ASSETS)
    assert s["bridged_asset_interactions"] == 1
