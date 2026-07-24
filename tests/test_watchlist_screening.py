"""Sanctions/darklist screening at serving time.

The analyze endpoints screen addresses against a bundled OFAC + community
darklist snapshot. A listed address must escalate; a clean one must not.
"""
import json
from pathlib import Path

import pytest

from src.response.watchlist import is_known_bad, watchlist_size


def test_watchlist_loaded():
    assert watchlist_size() > 0


def test_clean_address_not_flagged():
    assert is_known_bad("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045") is False


def test_empty_address_not_flagged():
    assert is_known_bad("") is False
    assert is_known_bad(None) is False


def test_listed_address_flagged():
    addrs = json.loads(Path("data/known_bad_addresses.json").read_text())["addresses"]
    assert is_known_bad(addrs[0]) is True


def test_screening_is_case_insensitive():
    addrs = json.loads(Path("data/known_bad_addresses.json").read_text())["addresses"]
    assert is_known_bad(addrs[0].upper()) is True


def test_known_bad_escalates_score():
    """A listed address must escalate above LOG_ONLY."""
    from src.response.risk_scorer import CompositeRiskScorer, ModelScores, ActionTier
    scorer = CompositeRiskScorer()
    clean = scorer.score("0xabc", "0xtx",
                         ModelScores(gnn=-1.0, autoencoder=-1.0,
                                     isolation_forest=-1.0, graph_centrality=0.0),
                         value_eth=1.0)
    flagged = scorer.score("0xabc", "0xtx",
                           ModelScores(gnn=-1.0, autoencoder=-1.0,
                                       isolation_forest=-1.0, graph_centrality=0.0,
                                       known_bad_address=True),
                           value_eth=1.0)
    assert flagged.composite_score > clean.composite_score
    assert flagged.action != ActionTier.LOG_ONLY
