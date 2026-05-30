"""
Feature engineering pipeline.

Each feature is a pure function: takes transaction data, returns a float.
Features are grouped by type for organized computation.

WHY pure functions:
- Testable in isolation
- Can be vectorized with NumPy (Week 2)
- Can be JIT-compiled with Numba (Week 2)
- Easy to add/remove features without touching other code
"""

import time
from dataclasses import dataclass

import numpy as np

from config.logging_config import get_logger
from src.ml.tabular.vectorized_features import log_transform_values

logger = get_logger(__name__)

from src.ml.tabular.vectorized_features import warmup_jit

warmup_jit()  # Compile JIT functions at module import time


@dataclass
class TransactionFeatures:
    """
    Feature vector for a single transaction.
    This is what gets fed into ML models.
    All fields are floats — ML models need numeric input.
    """

    # Identity
    tx_hash: str = ""
    from_addr: str = ""
    chain_name: str = ""

    # Value features
    value_eth: float = 0.0
    log_value_eth: float = 0.0  # log(value + 1) — handles zero values
    is_high_value: float = 0.0  # 1.0 if > 1 ETH

    # Gas features
    gas: float = 0.0
    gas_price_gwei: float = 0.0
    gas_price_percentile: float = 0.5  # placeholder until we have history

    # Contract features
    is_contract_call: float = 0.0
    is_contract_creation: float = 0.0
    input_data_length: float = 0.0  # bytes of input data

    # Temporal features (require address history — populated later)
    tx_count_1h: float = 0.0  # transactions from this address in last 1h
    tx_count_24h: float = 0.0  # transactions from this address in last 24h
    value_sum_1h: float = 0.0  # total ETH sent in last 1h
    time_since_last_tx: float = -1.0  # seconds (-1 = first seen)

    # Graph features (populated by graph engine — Week 3)
    address_in_degree: float = 0.0
    address_out_degree: float = 0.0
    cluster_risk_score: float = 0.0

    FEATURE_NAMES = [
        "value_eth", "log_value_eth", "is_high_value", "gas",
        "gas_price_gwei", "gas_price_percentile", "is_contract_call",
        "is_contract_creation", "input_data_length", "tx_count_1h",
        "tx_count_24h", "value_sum_1h", "time_since_last_tx",
        "address_in_degree", "address_out_degree", "cluster_risk_score",
    ]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array(
            [
                self.value_eth,
                self.log_value_eth,
                self.is_high_value,
                self.gas,
                self.gas_price_gwei,
                self.gas_price_percentile,
                self.is_contract_call,
                self.is_contract_creation,
                self.input_data_length,
                self.tx_count_1h,
                self.tx_count_24h,
                self.value_sum_1h,
                self.time_since_last_tx,
                self.address_in_degree,
                self.address_out_degree,
                self.cluster_risk_score,
            ],
            dtype=np.float32,
        )

    @property
    def feature_dim(self) -> int:
        return len(self.to_numpy())


class FeatureEngineer:
    """
    Computes features for transactions.
    Maintains a short in-memory history for velocity features.
    """

    def __init__(self, history_window_seconds: int = 3600):
        self.history_window = history_window_seconds
        # address -> list of (timestamp, value_eth)
        self._address_history: dict[str, list[tuple[float, float]]] = {}

    def extract(self, tx_payload: dict) -> TransactionFeatures:
        """
        Main entry point — compute all features for one transaction.
        Called by the Kafka consumer for every message.
        """
        now = time.time()
        from_addr = tx_payload.get("from_addr", "")
        value_eth = float(tx_payload.get("value_eth", 0.0))
        block_timestamp = float(tx_payload.get("block_timestamp", now))

        # Update address history
        self._update_history(from_addr, block_timestamp, value_eth)

        # Compute velocity features from history
        history = self._get_recent_history(from_addr, block_timestamp)
        tx_count_1h = len(history)
        value_sum_1h = sum(v for _, v in history)

        # Time since last tx
        if len(history) > 1:
            sorted_times = sorted(t for t, _ in history)
            time_since_last = block_timestamp - sorted_times[-2]
        else:
            time_since_last = -1.0

        # Input data
        input_data = tx_payload.get("input_data", "0x") or "0x"
        input_len = max(0, (len(input_data) - 2) // 2)  # bytes

        # Gas price in Gwei
        gas_price_wei = float(tx_payload.get("gas_price", 0))
        gas_price_gwei = gas_price_wei / 1e9

        features = TransactionFeatures(
            tx_hash=tx_payload.get("tx_hash", ""),
            from_addr=from_addr,
            chain_name=tx_payload.get("chain_name", ""),
            value_eth=value_eth,
            log_value_eth=float(log_transform_values(np.array([value_eth]))[0]),
            is_high_value=1.0 if value_eth > 1.0 else 0.0,
            gas=float(tx_payload.get("gas", 0)),
            gas_price_gwei=gas_price_gwei,
            is_contract_call=1.0 if tx_payload.get("is_contract_call") else 0.0,
            is_contract_creation=1.0 if tx_payload.get("is_contract_creation") else 0.0,
            input_data_length=float(input_len),
            tx_count_1h=float(tx_count_1h),
            tx_count_24h=float(tx_count_1h),  # simplified for now
            value_sum_1h=float(value_sum_1h),
            time_since_last_tx=float(time_since_last),
        )

        logger.debug(
            "features_extracted",
            tx_hash=features.tx_hash[:12],
            feature_dim=features.feature_dim,
            tx_count_1h=tx_count_1h,
        )

        return features

    def _update_history(
        self,
        address: str,
        timestamp: float,
        value_eth: float,
    ) -> None:
        """Add transaction to address history, prune old entries."""
        if address not in self._address_history:
            self._address_history[address] = []

        self._address_history[address].append((timestamp, value_eth))

        # Prune entries older than history window
        cutoff = timestamp - self.history_window
        self._address_history[address] = [
            (t, v) for t, v in self._address_history[address] if t >= cutoff
        ]

    def _get_recent_history(
        self,
        address: str,
        current_timestamp: float,
    ) -> list[tuple[float, float]]:
        """Get transactions from the last hour for an address."""
        cutoff = current_timestamp - self.history_window
        return [
            (t, v) for t, v in self._address_history.get(address, []) if t >= cutoff
        ]
