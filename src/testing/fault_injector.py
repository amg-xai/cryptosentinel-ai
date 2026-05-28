"""
Fault injection framework for resilience testing.

WHY fault injection:
  Every dependency WILL fail in production.
  The question is: does your system fail gracefully or catastrophically?
  Fault injection proves graceful degradation BEFORE production failures.

Fault modes:
  LATENCY:  Add random delay to external calls (network slowness)
  ERROR:    Raise exceptions from dependencies (service failures)
  TIMEOUT:  Never respond (hung connections)
  PARTIAL:  Succeed sometimes, fail sometimes (flaky services)

Usage:
  with FaultInjector(target_fn, FaultMode.LATENCY, latency_ms=200):
      result = target_fn(args)  # will be delayed 200ms

  # Or as a context that patches a module function:
  with patch_with_fault("src.streaming.producer.ThreatIntelProducer.produce_transaction",
                         FaultMode.ERROR):
      pipeline.run()  # producer will raise exceptions
"""
import asyncio
import random
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Any
from unittest.mock import patch, MagicMock

from config.logging_config import get_logger

logger = get_logger(__name__)


class FaultMode(Enum):
    NONE = "none"
    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class FaultInjector:
    """
    Wraps a function to inject faults for testing.
    Use as a context manager or call inject() directly.
    """

    def __init__(
        self,
        target: Callable,
        mode: FaultMode,
        latency_ms: float = 200.0,
        error_rate: float = 1.0,
        timeout_seconds: float = 30.0,
        exception_type: type = Exception,
        exception_message: str = "Injected fault",
    ):
        self.target = target
        self.mode = mode
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.timeout_seconds = timeout_seconds
        self.exception_type = exception_type
        self.exception_message = exception_message
        self._call_count = 0
        self._fault_count = 0

    def inject(self, *args, **kwargs) -> Any:
        """Call the target with fault injection applied."""
        self._call_count += 1

        if self.mode == FaultMode.NONE:
            return self.target(*args, **kwargs)

        elif self.mode == FaultMode.LATENCY:
            delay = self.latency_ms / 1000.0
            logger.debug("fault_latency_injected", delay_ms=self.latency_ms)
            time.sleep(delay)
            self._fault_count += 1
            return self.target(*args, **kwargs)

        elif self.mode == FaultMode.ERROR:
            if random.random() < self.error_rate:
                self._fault_count += 1
                logger.debug(
                    "fault_error_injected",
                    exception=self.exception_message,
                )
                raise self.exception_type(self.exception_message)
            return self.target(*args, **kwargs)

        elif self.mode == FaultMode.TIMEOUT:
            logger.debug("fault_timeout_injected",
                        seconds=self.timeout_seconds)
            time.sleep(self.timeout_seconds)
            raise TimeoutError(f"Injected timeout after {self.timeout_seconds}s")

        elif self.mode == FaultMode.PARTIAL:
            # 50% success, 50% failure
            if random.random() < 0.5:
                self._fault_count += 1
                raise self.exception_type(self.exception_message)
            return self.target(*args, **kwargs)

        return self.target(*args, **kwargs)

    @property
    def fault_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._fault_count / self._call_count

    def stats(self) -> dict:
        return {
            "mode": self.mode.value,
            "total_calls": self._call_count,
            "fault_count": self._fault_count,
            "fault_rate": round(self.fault_rate, 3),
        }


@contextmanager
def inject_kafka_failure(error_rate: float = 1.0):
    """
    Context manager: makes Kafka producer raise exceptions.
    Use to test: does the system buffer transactions locally?
    """
    logger.warning("chaos_kafka_failure_start", error_rate=error_rate)
    original_produce = None

    try:
        from src.streaming.producer import ThreatIntelProducer
        original_produce = ThreatIntelProducer.produce_transaction

        def failing_produce(self, tx):
            if random.random() < error_rate:
                raise Exception("Injected Kafka failure")
            return original_produce(self, tx)

        ThreatIntelProducer.produce_transaction = failing_produce
        yield
    finally:
        if original_produce:
            from src.streaming.producer import ThreatIntelProducer
            ThreatIntelProducer.produce_transaction = original_produce
        logger.warning("chaos_kafka_failure_end")


@contextmanager
def inject_model_latency(latency_ms: float = 500.0):
    """
    Context manager: adds latency to ML model scoring.
    Use to test: does the API remain responsive under slow ML?
    """
    logger.warning("chaos_model_latency_start", latency_ms=latency_ms)
    original_score = None

    try:
        from src.ml.inference_engine import InferenceEngine
        original_score = InferenceEngine.score_features

        def slow_score(self, features_array):
            time.sleep(latency_ms / 1000.0)
            return original_score(self, features_array)

        InferenceEngine.score_features = slow_score
        yield
    finally:
        if original_score:
            from src.ml.inference_engine import InferenceEngine
            InferenceEngine.score_features = original_score
        logger.warning("chaos_model_latency_end")


@contextmanager
def inject_graph_failure():
    """
    Context manager: makes graph queries raise exceptions.
    Use to test: does risk scorer fall back gracefully?
    """
    logger.warning("chaos_graph_failure_start")
    original_query = None

    try:
        from src.graph.threat_graph import ThreatGraph
        original_query = ThreatGraph.find_laundering_paths

        def failing_query(self, *args, **kwargs):
            raise Exception("Injected graph query failure")

        ThreatGraph.find_laundering_paths = failing_query
        yield
    finally:
        if original_query:
            from src.graph.threat_graph import ThreatGraph
            ThreatGraph.find_laundering_paths = original_query
        logger.warning("chaos_graph_failure_end")
