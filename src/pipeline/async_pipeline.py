"""
Async pipeline — connects blockchain ingestion to Kafka streaming.

Flow:
  BlockIngester -> asyncio.Queue -> worker -> KafkaProducer -> topics
"""

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

import uvloop

from config.logging_config import get_logger
from config.settings import settings
from src.blockchain.ingester import BlockIngester
from src.blockchain.models import Transaction
from src.monitoring.metrics import TRANSACTIONS_SCANNED

logger = get_logger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_process_pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))

QUEUE_MAX_SIZE = 10_000


async def kafka_producer_worker(queue: asyncio.Queue) -> None:
    """
    Consumes transactions from the queue and publishes to Kafka.
    """
    from src.streaming.producer import ThreatIntelProducer

    producer = ThreatIntelProducer()
    processed = 0

    logger.info("kafka_worker_started")

    try:
        while True:
            tx: Transaction = await queue.get()
            processed += 1

            producer.produce_transaction(tx)
            TRANSACTIONS_SCANNED.labels(chain=tx.chain_name).inc()

            if processed % 50 == 0:
                logger.info(
                    "pipeline_progress",
                    processed=processed,
                    queue_size=queue.qsize(),
                    latest_block=tx.block_number,
                    chain=tx.chain_name,
                )

            queue.task_done()

    finally:
        producer.flush()
        logger.info("kafka_worker_stopped", total_processed=processed)


async def run_pipeline() -> None:
    """Main pipeline — starts ingester and Kafka worker concurrently."""
    queue: asyncio.Queue[Transaction] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

    eth_ingester = BlockIngester(
        http_url=settings.eth_http_url,
        ws_url=settings.eth_ws_url,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )

    logger.info("pipeline_starting", queue_max_size=QUEUE_MAX_SIZE)

    if not eth_ingester.test_connection():
        logger.error("pipeline_aborted", reason="ethereum_connection_failed")
        return

    logger.info(
        "pipeline_connections_verified",
        latest_eth_block=eth_ingester.get_latest_block(),
    )

    await asyncio.gather(
        eth_ingester.stream(queue),
        kafka_producer_worker(queue),
        kafka_producer_worker(queue),
    )


def start() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    start()
