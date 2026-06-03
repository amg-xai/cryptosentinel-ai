"""Kafka producer resilience: failures buffer locally and drain on recovery,
rather than crashing or silently losing alerts (verified via fault injection)."""
from collections import deque
from src.streaming.producer import ThreatIntelProducer


class _FakeKafka:
    def __init__(self):
        self.down = True
        self.sent = []

    def produce(self, topic, key, value, on_delivery=None):
        if self.down:
            raise Exception("Kafka down")
        self.sent.append((topic, value))

    def poll(self, t):
        pass


class _Tx:
    tx_hash = "0xabc"; block_number = 1; block_timestamp = 1.0
    from_addr = "0xaaa"; to_addr = "0xbbb"; value_wei = 0; value_eth = 0.0
    gas = 21000; gas_price = 1; is_contract_creation = False
    is_contract_call = False; chain_id = 1; chain_name = "ethereum-sepolia"


def _producer():
    p = ThreatIntelProducer.__new__(ThreatIntelProducer)
    p._buffer = deque(maxlen=ThreatIntelProducer._BUFFER_MAXLEN)
    p._producer = _FakeKafka()
    return p


def test_failure_buffers_not_loses():
    p = _producer()
    for _ in range(3):
        p.produce_transaction(_Tx())  # Kafka down
    assert len(p._buffer) == 3        # buffered, not lost
    assert len(p._producer.sent) == 0


def test_no_crash_on_failure():
    """A Kafka failure must not raise out of produce_transaction."""
    p = _producer()
    p.produce_transaction(_Tx())  # must not raise
    assert len(p._buffer) == 1


def test_recovery_drains_buffer():
    p = _producer()
    for _ in range(3):
        p.produce_transaction(_Tx())
    p._producer.down = False
    p.produce_transaction(_Tx())      # triggers drain + sends new
    assert len(p._buffer) == 0
    assert len(p._producer.sent) == 4


def test_buffer_is_bounded():
    """Buffer must not grow unbounded under a sustained outage."""
    assert ThreatIntelProducer._BUFFER_MAXLEN == 10000
    p = _producer()
    for _ in range(10050):
        p.produce_transaction(_Tx())
    assert len(p._buffer) == 10000  # capped, oldest evicted
