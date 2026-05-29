"""
Multi-chain ingester factory + sampling.

Builds one BlockIngester per configured chain from the chain registry.
The scoring pipeline runs one ingestion loop per chain concurrently;
all chains feed the same downstream scoring path (models, graph, alerts).

Sampling:
  High-volume chains (Polygon mainnet) set sample_rate < 1.0.
  should_sample() applies deterministic-ish random sampling so we
  score a representative subset without exhausting RPC compute units.
"""
import random
from typing import Dict

from config.logging_config import get_logger
from src.blockchain.ingester import BlockIngester
from src.blockchain.chains import ChainConfig, get_configured_chains

logger = get_logger(__name__)


def build_ingester(chain: ChainConfig) -> BlockIngester:
    """Construct a BlockIngester for a single chain."""
    ws_url = ""  # WebSocket unused; HTTP polling is the active path
    ingester = BlockIngester(
        http_url=chain.http_url,
        ws_url=ws_url,
        chain_id=chain.chain_id,
        chain_name=chain.name,
    )
    logger.info(
        "ingester_built",
        chain=chain.name,
        symbol=chain.native_symbol,
        sample_rate=chain.sample_rate,
    )
    return ingester


def build_all_ingesters() -> Dict[str, BlockIngester]:
    """Build one ingester per configured chain."""
    ingesters = {}
    for name, chain in get_configured_chains().items():
        try:
            ingesters[name] = build_ingester(chain)
        except Exception as e:
            logger.error("ingester_build_failed", chain=name, error=str(e))
    logger.info("multichain_ready", chains=list(ingesters.keys()))
    return ingesters


def should_sample(chain: ChainConfig) -> bool:
    """
    Decide whether to score this transaction based on chain sample_rate.
    sample_rate=1.0 always returns True; 0.2 returns True ~20% of the time.
    """
    if chain.sample_rate >= 1.0:
        return True
    return random.random() < chain.sample_rate
