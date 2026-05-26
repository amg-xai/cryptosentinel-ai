"""Tests for feature engineering pipeline."""

import numpy as np

from src.ml.tabular.feature_engineer import FeatureEngineer, TransactionFeatures


def make_tx_payload(
    tx_hash="0xtest",
    from_addr="0xSender",
    value_eth=1.5,
    gas=21000,
    gas_price=1_000_000_000,
    is_contract_call=False,
    is_contract_creation=False,
    input_data="0x",
    block_timestamp=1000000.0,
    chain_name="ethereum-sepolia",
):
    return {
        "tx_hash": tx_hash,
        "from_addr": from_addr,
        "value_eth": value_eth,
        "gas": gas,
        "gas_price": gas_price,
        "is_contract_call": is_contract_call,
        "is_contract_creation": is_contract_creation,
        "input_data": input_data,
        "block_timestamp": block_timestamp,
        "chain_name": chain_name,
    }


def test_feature_extraction_basic():
    """Basic feature extraction returns TransactionFeatures."""
    fe = FeatureEngineer()
    tx = make_tx_payload(value_eth=2.0)
    features = fe.extract(tx)
    assert isinstance(features, TransactionFeatures)
    assert features.value_eth == 2.0
    assert features.is_high_value == 1.0


def test_log_value_feature():
    """log_value_eth should be log(value + 1)."""
    import math

    fe = FeatureEngineer()
    tx = make_tx_payload(value_eth=1.0)
    features = fe.extract(tx)
    assert abs(features.log_value_eth - math.log1p(1.0)) < 1e-6


def test_zero_value_transaction():
    """Zero ETH transactions are valid — common for contract calls."""
    fe = FeatureEngineer()
    tx = make_tx_payload(value_eth=0.0, is_contract_call=True)
    features = fe.extract(tx)
    assert features.value_eth == 0.0
    assert features.is_high_value == 0.0
    assert features.is_contract_call == 1.0


def test_velocity_feature_accumulates():
    """tx_count_1h increases with each transaction from same address."""
    fe = FeatureEngineer()
    addr = "0xRepeatedSender"

    # Send 5 transactions from same address
    for i in range(5):
        tx = make_tx_payload(
            from_addr=addr,
            block_timestamp=1000000.0 + i,
        )
        features = fe.extract(tx)

    # After 5 transactions, count should be 5
    assert features.tx_count_1h == 5.0


def test_features_to_numpy():
    """to_numpy returns a float32 array."""
    fe = FeatureEngineer()
    tx = make_tx_payload()
    features = fe.extract(tx)
    arr = features.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert len(arr) == features.feature_dim


def test_gas_price_in_gwei():
    """Gas price should be converted from Wei to Gwei."""
    fe = FeatureEngineer()
    tx = make_tx_payload(gas_price=5_000_000_000)  # 5 Gwei in Wei
    features = fe.extract(tx)
    assert features.gas_price_gwei == 5.0


def test_contract_call_detection():
    """is_contract_call flag should be set correctly."""
    fe = FeatureEngineer()
    tx = make_tx_payload(
        is_contract_call=True,
        input_data="0xa9059cbb0000",
    )
    features = fe.extract(tx)
    assert features.is_contract_call == 1.0
    assert features.input_data_length > 0
