"""Tests for live-mode scoring (Option B: heuristic + graph, ML dampened)."""
from src.response.risk_scorer import CompositeRiskScorer, ModelScores, ActionTier


def _ms(cent=0.0, vel=False, known_bad=False):
    m = ModelScores(
        gnn=-1.0, autoencoder=-1.0, isolation_forest=-1.0,
        graph_centrality=cent, velocity_flag=vel,
    )
    m.known_bad_address = known_bad
    return m


def test_live_mode_high_centrality_alone_not_flagged():
    """A legit high-degree address (exchange/router) must NOT be flagged."""
    s = CompositeRiskScorer()
    r = s.score("0xExchange", "0xtx", _ms(cent=0.9), live_mode=True)
    assert r.action == ActionTier.LOG_ONLY
    assert r.is_threat is False


def test_live_mode_centrality_plus_velocity_watchlist():
    s = CompositeRiskScorer()
    r = s.score("0xA", "0xtx", _ms(cent=0.8, vel=True), live_mode=True)
    assert r.action == ActionTier.WATCHLIST


def test_live_mode_multi_signal_escalates():
    s = CompositeRiskScorer()
    r = s.score("0xHop", "0xtx", _ms(cent=0.8, vel=True),
                live_mode=True, cross_chain_boost=1.4)
    assert r.composite_score > 0.7


def test_live_mode_known_bad_still_emergency():
    """Known-bad floor must survive live-mode dampening."""
    s = CompositeRiskScorer()
    r = s.score("0xBad", "0xtx", _ms(cent=0.3, known_bad=True), live_mode=True)
    assert r.action == ActionTier.EMERGENCY


def test_live_mode_dampens_vs_normal():
    """Same centrality scores lower in live mode than non-live mode."""
    s = CompositeRiskScorer()
    live = s.score("0xA", "0xtx", _ms(cent=0.8), live_mode=True)
    normal = s.score("0xB", "0xtx", _ms(cent=0.8), live_mode=False)
    assert live.composite_score < normal.composite_score


def test_offline_mode_unaffected_when_ml_present():
    """With ML scores present, live_mode dampening does NOT apply."""
    s = CompositeRiskScorer()
    m = ModelScores(gnn=0.8, autoencoder=0.7, isolation_forest=0.6,
                    graph_centrality=0.8)
    # Even with live_mode=True, ML present means no centrality dampening
    r_live = s.score("0xA", "0xtx", m, live_mode=True)
    r_off = s.score("0xB", "0xtx", m, live_mode=False)
    assert abs(r_live.composite_score - r_off.composite_score) < 1e-9
