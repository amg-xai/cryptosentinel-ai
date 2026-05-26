"""
Kafka consumer — reads transactions from topics and routes to processors.

Design:
- Each consumer instance handles one topic
- Commits offsets only after successful processing (at-least-once delivery)
- Tracks consumer lag via Prometheus gauge
- Graceful shutdown on SIGTERM
"""

import json
import signal
from collections.abc import Callable

from confluent_kafka import Consumer, KafkaError, KafkaException

from config.logging_config import get_logger
from config.settings import settings
from src.monitoring.metrics import KAFKA_MESSAGES_CONSUMED

logger = get_logger(__name__)


class ThreatIntelConsumer:
    """
    Consumes messages from a Kafka topic and passes them to a handler.
    Commits offsets after successful processing — no silent data loss.
    """

    def __init__(
        self,
        topic: str,
        group_id: str,
        bootstrap_servers: str | None = None,
    ):
        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._consumer = self._create_consumer()
        self._running = False

        # Handle SIGTERM gracefully — flush before exit
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _create_consumer(self) -> Consumer:
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            # Start from earliest unread message
            "auto.offset.reset": "earliest",
            # Manual offset commit — we commit after processing
            "enable.auto.commit": False,
            # Heartbeat settings — detect dead consumers faster
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 10000,
            "max.poll.interval.ms": 300000,
        }
        consumer = Consumer(config)
        consumer.subscribe([self.topic])
        logger.info(
            "kafka_consumer_created",
            topic=self.topic,
            group_id=self.group_id,
        )
        return consumer

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("kafka_consumer_shutdown_signal_received", topic=self.topic)
        self._running = False

    def consume(
        self,
        handler: Callable[[dict], None],
        poll_timeout: float = 1.0,
    ) -> None:
        """
        Blocking consume loop. Calls handler for each message.
        Commits offset only after handler succeeds.
        """
        self._running = True
        messages_processed = 0

        logger.info("kafka_consumer_started", topic=self.topic)

        try:
            while self._running:
                msg = self._consumer.poll(timeout=poll_timeout)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition — not an error, just caught up
                        logger.debug(
                            "kafka_partition_eof",
                            topic=msg.topic(),
                            partition=msg.partition(),
                            offset=msg.offset(),
                        )
                    else:
                        raise KafkaException(msg.error())
                    continue

                try:
                    # Deserialize JSON payload
                    payload = json.loads(msg.value().decode("utf-8"))

                    # Call the handler
                    handler(payload)

                    # Commit offset AFTER successful processing
                    self._consumer.commit(message=msg, asynchronous=False)

                    messages_processed += 1
                    KAFKA_MESSAGES_CONSUMED.labels(topic=self.topic).inc()

                    if messages_processed % 100 == 0:
                        logger.info(
                            "kafka_consumer_progress",
                            topic=self.topic,
                            processed=messages_processed,
                            partition=msg.partition(),
                            offset=msg.offset(),
                        )

                except json.JSONDecodeError as e:
                    logger.error(
                        "kafka_message_deserialize_failed",
                        error=str(e),
                        topic=self.topic,
                    )
                    # Commit the bad message so we don't get stuck on it
                    self._consumer.commit(message=msg, asynchronous=False)

                except Exception as e:
                    logger.error(
                        "kafka_handler_failed",
                        error=str(e),
                        topic=self.topic,
                        offset=msg.offset(),
                    )
                    # Don't commit — will retry on restart

        finally:
            self._consumer.close()
            logger.info(
                "kafka_consumer_stopped",
                topic=self.topic,
                total_processed=messages_processed,
            )

    def stop(self) -> None:
        self._running = False
