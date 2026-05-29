"""
Multi-chain configuration registry.

WHY abstract chains:
  The detection logic (ML models, graph, risk scoring) is chain-agnostic.
  Only ingestion differs per chain: RPC endpoint, native token, block time,
  explorer URL. This registry isolates those differences so adding a third
  chain (Arbitrum, Base, etc.) is a config change, not a code rewrite.

Each chain carries:
  name:           canonical identifier used in Kafka topics + tx payloads
  http_url:       JSON-RPC endpoint
  native_symbol:  ETH, MATIC, etc. (for value display)
  block_time_s:   expected seconds between blocks (poll cadence)
  explorer:       block explorer base URL (for dashboard links)
  sample_rate:    fraction of txs to score (mainnet volume control)
"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class ChainConfig:
    name: str
    http_url: str
    native_symbol: str
    block_time_s: float
    explorer: str
    sample_rate: float = 1.0
    chain_id: int = 0

    @property
    def is_configured(self) -> bool:
        """True if a real RPC URL is set (not the localhost default)."""
        return bool(self.http_url) and "localhost" not in self.http_url


def get_chains() -> dict:
    """Build the chain registry from current settings."""
    return {
        "ethereum-sepolia": ChainConfig(
            name="ethereum-sepolia",
            http_url=settings.eth_http_url,
            native_symbol="ETH",
            block_time_s=12.0,
            explorer="https://sepolia.etherscan.io",
            sample_rate=1.0,
            chain_id=11155111,
        ),
        "polygon-mainnet": ChainConfig(
            name="polygon-mainnet",
            http_url=settings.polygon_http_url,
            native_symbol="MATIC",
            block_time_s=2.0,
            explorer="https://polygonscan.com",
            # Polygon mainnet is high-volume (~2s blocks, 50-100 tx/block).
            # Sample 20% to stay within Alchemy free-tier compute units.
            sample_rate=0.20,
            chain_id=137,
        ),
    }


def get_configured_chains() -> dict:
    """Return only chains with a real RPC URL set."""
    return {k: v for k, v in get_chains().items() if v.is_configured}
