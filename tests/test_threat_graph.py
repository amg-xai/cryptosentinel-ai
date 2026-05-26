"""Tests for ThreatGraph and graph algorithms."""
import time
import pytest
from src.graph.threat_graph import ThreatGraph, UnionFind


def make_graph_with_transactions() -> ThreatGraph:
    g = ThreatGraph()
    now = time.time()
    g.add_transaction("tx1", "0xA", "0xB", 1.0, now, 100)
    g.add_transaction("tx2", "0xB", "0xC", 0.9, now + 10, 101)
    g.add_transaction("tx3", "0xC", "0xD", 0.8, now + 20, 102)
    g.add_transaction("tx4", "0xA", "0xE", 0.5, now + 30, 103)
    g.add_transaction("tx5", "0xE", "0xA", 0.4, now + 3600, 104)
    return g


def test_union_find_same_cluster():
    uf = UnionFind()
    uf.union("0xA", "0xB")
    assert uf.same_cluster("0xA", "0xB")


def test_union_find_different_clusters():
    uf = UnionFind()
    uf.union("0xA", "0xB")
    uf.union("0xC", "0xD")
    assert not uf.same_cluster("0xA", "0xC")


def test_union_find_transitivity():
    uf = UnionFind()
    uf.union("0xA", "0xB")
    uf.union("0xB", "0xC")
    assert uf.same_cluster("0xA", "0xC")


def test_union_find_get_clusters():
    uf = UnionFind()
    uf.union("0xA", "0xB")
    uf.union("0xC", "0xD")
    clusters = uf.get_clusters()
    sizes = sorted([len(v) for v in clusters.values()])
    assert sizes == [2, 2]


def test_add_transaction_creates_nodes():
    g = ThreatGraph()
    g.add_transaction("tx1", "0xSender", "0xRecipient", 1.0, time.time(), 100)
    assert g.G.has_node("0xSender")
    assert g.G.has_node("0xRecipient")


def test_add_transaction_creates_edge():
    g = ThreatGraph()
    g.add_transaction("tx1", "0xA", "0xB", 1.0, time.time(), 100)
    assert g.G.has_edge("0xA", "0xB")


def test_add_transaction_is_idempotent():
    g = ThreatGraph()
    g.add_transaction("tx1", "0xA", "0xB", 1.0, time.time(), 100)
    g.add_transaction("tx1", "0xA", "0xB", 1.0, time.time(), 100)
    assert g.num_nodes == 2
    assert g.num_edges == 1


def test_flag_address_sets_risk():
    g = make_graph_with_transactions()
    g.flag_address("0xD", 0.9)
    assert g.G.nodes["0xD"]["risk_score"] == 0.9
    assert g.G.nodes["0xD"]["is_flagged"] is True


def test_flag_address_propagates_to_predecessors():
    g = make_graph_with_transactions()
    g.flag_address("0xD", 0.9)
    assert g.G.nodes["0xC"]["risk_score"] == pytest.approx(0.45)


def test_peel_back_finds_ancestors():
    g = make_graph_with_transactions()
    ancestors = g.peel_back("0xD", max_depth=3)
    assert "0xC" in ancestors
    assert "0xB" in ancestors
    assert "0xA" in ancestors


def test_peel_back_excludes_source():
    g = make_graph_with_transactions()
    ancestors = g.peel_back("0xD", max_depth=3)
    assert "0xD" not in ancestors


def test_laundering_paths_found():
    g = make_graph_with_transactions()
    paths = g.find_laundering_paths("0xA", max_depth=4)
    assert len(paths) > 0
    assert all(p[0] == "0xA" for p in paths)


def test_round_trip_detection():
    g = make_graph_with_transactions()
    round_trips = g.detect_round_trips("0xA")
    assert len(round_trips) > 0
    intermediates = [rt["intermediate"] for rt in round_trips]
    assert "0xE" in intermediates


def test_wallet_clustering_union_find():
    g = make_graph_with_transactions()
    g.cluster_wallets_union_find([
        ["0xA", "0xB"],
        ["0xC", "0xD"],
    ])
    uf = g._union_find
    assert uf.same_cluster("0xA", "0xB")
    assert uf.same_cluster("0xC", "0xD")
    assert not uf.same_cluster("0xA", "0xC")


def test_get_wallet_stats():
    g = make_graph_with_transactions()
    stats = g.get_wallet_stats("0xA")
    assert stats["address"] == "0xA"
    assert "risk_score" in stats
    assert "in_degree" in stats
    assert "out_degree" in stats


def test_get_wallet_stats_unknown_address():
    g = ThreatGraph()
    stats = g.get_wallet_stats("0xUnknown")
    assert stats == {}


def test_num_nodes_and_edges():
    g = make_graph_with_transactions()
    assert g.num_nodes == 5
    assert g.num_edges >= 4


def test_cluster_risk_propagation():
    g = make_graph_with_transactions()
    g.cluster_wallets_union_find([["0xA", "0xB"]])
    g.flag_address("0xA", 0.9)
    g.propagate_cluster_risk()
    assert g.G.nodes["0xB"]["risk_score"] > 0
