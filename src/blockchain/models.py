"""
Core blockchain data models.
Every transaction ingested from any chain is normalized to this shape.
Downstream modules (ML, graph, streaming) all consume Transaction objects.
"""

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """
    Normalized transaction model.
    Produced by BlockIngester, consumed by everything downstream.
    """

    tx_hash: str = Field(description="Transaction hash — unique identifier")
    block_number: int = Field(description="Block this transaction was mined in")
    block_timestamp: int = Field(description="Unix timestamp of the block")
    from_addr: str = Field(description="Sender address")
    to_addr: str | None = Field(
        default=None,
        description="Recipient address. None for contract creation transactions.",
    )
    value_wei: int = Field(description="Value transferred in Wei (1 ETH = 1e18 Wei)")
    gas: int = Field(description="Gas limit set by sender")
    gas_price: int = Field(description="Gas price in Wei")
    input_data: str = Field(
        description="Hex-encoded input data. Non-empty = contract call."
    )
    is_contract_creation: bool = Field(
        description="True if this transaction deploys a new contract"
    )
    is_contract_call: bool = Field(
        description="True if to_addr is a contract (input_data non-empty)"
    )
    chain_id: int = Field(description="Chain identifier. 11155111 = Sepolia.")
    chain_name: str = Field(description="Human readable chain name")

    @property
    def value_eth(self) -> float:
        """Value in ETH for human-readable display."""
        return self.value_wei / 1e18

    @property
    def is_high_value(self) -> bool:
        """Transactions over 1 ETH are flagged for closer inspection."""
        return self.value_eth > 1.0

    def model_summary(self) -> dict:
        """Compact summary for logging — avoids logging full input_data."""
        return {
            "tx_hash": self.tx_hash[:12] + "...",
            "from": self.from_addr[:10] + "...",
            "to": (self.to_addr[:10] + "...") if self.to_addr else "CONTRACT_CREATION",
            "value_eth": round(self.value_eth, 6),
            "is_contract_call": self.is_contract_call,
            "chain": self.chain_name,
        }
