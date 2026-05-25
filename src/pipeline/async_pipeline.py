"""
Async pipeline — the nervous system of CryptoSentinel.

Architecture:
  [BlockIngester] → asyncio.Queue (bounded, backpressure) → [Workers]

The bounded queue is critical:
  If ML processing slows down, the queue fills up.
  When full, queue.put() blocks the ingester automatically.
  This prevents memory overflow — no data loss, just natural throttling.

Workers run concurrently via asyncio.gather().
CPU-bound work (ML inference) is offloaded to ProcessPoolExecutor
so it never blocks the event loop.
"""

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

import uvloop

from config.logging_config import get_logger
from config.settings import settings
from src.blockchain.ingester import BlockIngester
from src.blockchain.models import Transaction

logger = get_logger(__name__)

# Install uvloop — 2-4x faster than standard asyncio for I/O workloads
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Process pool for CPU-bound ML inference
# Uses all cores except one (keep one for the event loop)
_process_pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))

# Bounded queue — backpressure mechanism
# 10_000 transactions can sit here while ML catches up
QUEUE_MAX_SIZE = 10_000


async def transaction_logger_worker(queue: asyncio.Queue) -> None:
    """
    Stub worker — logs every transaction.
    This will be replaced by the full ML scoring pipeline in Week 2.
    For now it proves the pipeline is working end to end.
    """
    processed = 0
    while True:
        tx: Transaction = await queue.get()
        processed += 1

        if processed % 10 == 0:
            logger.info(
                "transactions_processed",
                count=processed,
                latest_tx=tx.model_summary(),
                queue_size=queue.qsize(),
            )

        queue.task_done()


async def run_pipeline() -> None:
    """
    Main pipeline entry point.
    Starts all ingesters and workers concurrently.
    """
    queue: asyncio.Queue[Transaction] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

    # Ethereum Sepolia ingester
    eth_ingester = BlockIngester(
        http_url=settings.eth_http_url,
        ws_url=settings.eth_ws_url,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )

    logger.info("pipeline_starting", queue_max_size=QUEUE_MAX_SIZE)

    # Test connection before starting
    if not eth_ingester.test_connection():
        logger.error("pipeline_aborted", reason="ethereum_connection_failed")
        return

    logger.info(
        "pipeline_connections_verified",
        latest_eth_block=eth_ingester.get_latest_block(),
    )

    # Run ingester and worker concurrently
    # asyncio.gather runs all coroutines in the same event loop
    await asyncio.gather(
        eth_ingester.stream(queue),
        transaction_logger_worker(queue),
        transaction_logger_worker(queue),  # Two workers for throughput
    )


def start() -> None:
    """Entrypoint — called from CLI or main script."""
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    start()
