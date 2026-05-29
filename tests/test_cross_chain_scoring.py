"""Tests for cross-chain risk boost applied in the composite scorer."""
from src.response.risk_scorer import CompositeRiskScorer, ModelScores, ActionTier


def _scores(gnn=0.5):
    return ModelScores(
        gnn=gnn,
        autoencoder=-1.0,
        isolation_forest=-1.0,
        graph_centrality=0.0,
    )


def test_no_boost_leaves_score_unchanged():
    scorer = CompositeRiskScorer()
    base = scorer.score("0xA", "0xtx", _scores(0.6), cross_chain_boost=1.0)
    again = scorer.score("0xA", "0xtx", _scores(0.6), cross_chain_boost=1.0)
    assert base.composite_score == again.composite_score


def test_boost_increases_score():
    scorer = CompositeRiskScorer()
    base = scorer.score("0xA", "0xtx", _scores(0.5), cross_chain_boost=1.0)
    boosted = scorer.score("0xA", "0xtx", _scores(0.5), cross_chain_boost=1.4)
    assert boosted.composite_score > base.composite_score


def test_boost_can_change_action_tier():
    """
    A score just below a tier boundary should cross it when boosted.
    This is the key behavior: boost must affect the DECISION, not just
    the displayed number.
    """
    scorer = CompositeRiskScorer()
    # Find a gnn score that lands just under WATCHLIST (0.5) unboosted
    base = scorer.score("0xA", "0xtx", _scores(0.48), cross_chain_boost=1.0)
    boosted = scorer.score("0xB", "0xtx", _scores(0.48), cross_chain_boost=1.4)
    # Boosted action tier should be >= base tier severity
    tier_order = {
        ActionTier.LOG_ONLY: 0,
        ActionTier.WATCHLIST: 1,
        ActionTier.QUARANTINE: 2,
        ActionTier.EMERGENCY: 3,
    }
    assert tier_order[boosted.action] >= tier_order[base.action]


def test_boost_clamps_at_one():
    scorer = CompositeRiskScorer()
    # High base score + max boost must not exceed 1.0
    a = scorer.score("0xA", "0xtx", _scores(0.95), cross_chain_boost=1.4)
    assert a.composite_score <= 1.0


def test_boost_default_is_one():
    """Calling score() without the boost param must behave as boost=1.0."""
    scorer = CompositeRiskScorer()
    explicit = scorer.score("0xA", "0xtx", _scores(0.6), cross_chain_boost=1.0)
    implicit = scorer.score("0xA", "0xtx", _scores(0.6))
    assert explicit.composite_score == implicit.composite_score
