"""
Tests for blockchain ingestion layer.
Uses mocked web3 calls — no real RPC needed for unit tests.
"""

from unittest.mock import MagicMock, patch

from src.blockchain.ingester import BlockIngester
from src.blockchain.models import Transaction


def make_raw_tx(
    tx_hash="0xabc123",
    from_addr="0xSender",
    to_addr="0xRecipient",
    value=1_000_000_000_000_000_000,  # 1 ETH in Wei
    input_data="0x",
):
    """Factory for raw web3 transaction dicts."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {
        "hash": MagicMock(hex=lambda: tx_hash),
        "blockNumber": 100,
        "from": from_addr,
        "to": to_addr,
        "value": value,
        "gas": 21000,
        "gasPrice": 1_000_000_000,
        "input": input_data,
    }[key]
    mock.get = lambda key, default=None: {
        "to": to_addr,
        "input": input_data,
        "gasPrice": 1_000_000_000,
    }.get(key, default)
    return mock


def make_ingester():
    """Make a BlockIngester with mocked web3."""
    with patch("src.blockchain.ingester.Web3"):
        ingester = BlockIngester(
            http_url="https://mock",
            ws_url="wss://mock",
            chain_id=11155111,
            chain_name="ethereum-sepolia",
        )
    return ingester


def test_transaction_model_value_eth():
    """1 ETH in Wei should equal 1.0 ETH."""
    tx = Transaction(
        tx_hash="0xabc",
        block_number=1,
        block_timestamp=1000000,
        from_addr="0xSender",
        to_addr="0xRecipient",
        value_wei=2_000_000_000_000_000_000,  # 2 ETH in Wei
        gas=21000,
        gas_price=1_000_000_000,
        input_data="0x",
        is_contract_creation=False,
        is_contract_call=False,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )
    assert tx.value_eth == 2.0
    assert tx.is_high_value is True


def test_transaction_contract_creation():
    """to_addr=None means contract creation."""
    tx = Transaction(
        tx_hash="0xdef",
        block_number=2,
        block_timestamp=1000001,
        from_addr="0xDeployer",
        to_addr=None,
        value_wei=0,
        gas=500000,
        gas_price=1_000_000_000,
        input_data="0x6080604052",
        is_contract_creation=True,
        is_contract_call=False,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )
    assert tx.is_contract_creation is True
    assert tx.to_addr is None


def test_transaction_model_summary():
    """model_summary should return a dict with truncated fields."""
    tx = Transaction(
        tx_hash="0xabcdefghij123456",
        block_number=1,
        block_timestamp=1000000,
        from_addr="0xSenderAddress",
        to_addr="0xRecipientAddress",
        value_wei=500_000_000_000_000_000,
        gas=21000,
        gas_price=1_000_000_000,
        input_data="0x",
        is_contract_creation=False,
        is_contract_call=False,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )
    summary = tx.model_summary()
    assert "tx_hash" in summary
    assert "value_eth" in summary
    assert summary["value_eth"] == 0.5


def test_zero_value_transaction():
    """Zero value transactions are valid — common for contract calls."""
    tx = Transaction(
        tx_hash="0x999",
        block_number=3,
        block_timestamp=1000002,
        from_addr="0xCaller",
        to_addr="0xContract",
        value_wei=0,
        gas=100000,
        gas_price=1_000_000_000,
        input_data="0xa9059cbb",
        is_contract_creation=False,
        is_contract_call=True,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )
    assert tx.value_eth == 0.0
    assert tx.is_high_value is False
    assert tx.is_contract_call is True
