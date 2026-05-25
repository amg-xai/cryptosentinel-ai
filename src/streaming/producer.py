"""
Kafka producer — publishes normalized transactions to topics.
"""

import json
import time

from confluent_kafka import KafkaException, Producer

from config.logging_config import get_logger
from config.settings import settings
from src.blockchain.models import Transaction
from src.monitoring.metrics import KAFKA_MESSAGES_PRODUCED

logger = get_logger(__name__)

CHAIN_TOPIC_MAP = {
    "ethereum-sepolia": "raw.transactions.ethereum",
    "ethereum-mainnet": "raw.transactions.ethereum",
    "polygon-mumbai": "raw.transactions.polygon",
    "polygon-mainnet": "raw.transactions.polygon",
}


def _delivery_callback(err, msg) -> None:
    if err:
        logger.error(
            "kafka_delivery_failed",
            error=str(err),
            topic=msg.topic(),
        )
    else:
        KAFKA_MESSAGES_PRODUCED.labels(topic=msg.topic()).inc()


class ThreatIntelProducer:
    def __init__(self, bootstrap_servers: str | None = None):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._producer = self._create_producer()

    def _create_producer(self) -> Producer:
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "acks": "all",
            "retries": 5,
            "retry.backoff.ms": 500,
            "linger.ms": 10,
            "batch.size": 16384,
            "enable.idempotence": True,
        }
        producer = Producer(config)
        logger.info(
            "kafka_producer_created",
            bootstrap_servers=self.bootstrap_servers,
        )
        return producer

    def produce_transaction(self, tx: Transaction) -> None:
        topic = CHAIN_TOPIC_MAP.get(tx.chain_name, "raw.transactions.ethereum")

        payload = {
            "tx_hash": tx.tx_hash,
            "block_number": tx.block_number,
            "block_timestamp": tx.block_timestamp,
            "from_addr": tx.from_addr,
            "to_addr": tx.to_addr,
            "value_wei": tx.value_wei,
            "value_eth": tx.value_eth,
            "gas": tx.gas,
            "gas_price": tx.gas_price,
            "is_contract_creation": tx.is_contract_creation,
            "is_contract_call": tx.is_contract_call,
            "chain_id": tx.chain_id,
            "chain_name": tx.chain_name,
            "produced_at": int(time.time()),
        }

        try:
            self._producer.produce(
                topic=topic,
                key=tx.from_addr.encode(),
                value=json.dumps(payload).encode(),
                on_delivery=_delivery_callback,
            )
            self._producer.poll(0)
        except KafkaException as e:
            logger.error(
                "kafka_produce_failed",
                error=str(e),
                topic=topic,
                tx_hash=tx.tx_hash,
            )

    def produce_alert(self, alert: dict, severity: str = "high") -> None:
        topic = f"alerts.{severity}"
        try:
            self._producer.produce(
                topic=topic,
                key=alert.get("address", "unknown").encode(),
                value=json.dumps(alert).encode(),
                on_delivery=_delivery_callback,
            )
            self._producer.poll(0)
        except KafkaException as e:
            logger.error(
                "alert_produce_failed",
                error=str(e),
                severity=severity,
            )

    def flush(self, timeout: float = 10.0) -> None:
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("kafka_flush_incomplete", remaining=remaining)
        else:
            logger.info("kafka_flush_complete")
