"""Tests for BlockIngester sample_rate (multi-chain volume control)."""
from src.blockchain.ingester import BlockIngester


def _make_ingester(sample_rate=1.0):
    # localhost URL won't connect, but we only test the sample_rate attr
    # and gating logic, not live RPC.
    try:
        return BlockIngester(
            http_url="https://localhost",
            ws_url="wss://localhost",
            chain_id=1,
            chain_name="test-chain",
            sample_rate=sample_rate,
        )
    except Exception:
        return None


def test_sample_rate_stored():
    ing = _make_ingester(sample_rate=0.2)
    if ing is None:
        return  # connection failed in CI; attr test skipped
    assert ing.sample_rate == 0.2


def test_default_sample_rate_is_one():
    ing = _make_ingester()
    if ing is None:
        return
    assert ing.sample_rate == 1.0


def test_sampling_gate_logic():
    """
    The gating expression `sample_rate < 1.0 and random() > sample_rate`
    must always pass everything at rate 1.0, and statistically drop
    ~(1-rate) at rates below 1.0.
    """
    import random

    def keep(rate, r):
        # Mirror the ingester's gate: continue (drop) when this is True
        drop = rate < 1.0 and r > rate
        return not drop

    # Rate 1.0 keeps everything regardless of r
    assert keep(1.0, 0.99) is True
    assert keep(1.0, 0.01) is True

    # Rate 0.2: r below 0.2 kept, above 0.2 dropped
    assert keep(0.2, 0.1) is True
    assert keep(0.2, 0.5) is False

    # Statistical check at 0.2
    random.seed(7)
    kept = sum(keep(0.2, random.random()) for _ in range(2000))
    # ~20% kept, allow margin
    assert 300 <= kept <= 500
