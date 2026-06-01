"""
OpenTelemetry distributed tracing setup.
This module is imported first in every service entrypoint.
Every span created anywhere in the codebase becomes a child
of the active trace — giving you a full flamegraph of every request.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ParentBased, TraceIdRatioBased, ALWAYS_ON,
)
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "cryptosentinel") -> trace.Tracer:
    """
    Initialize OTel tracing. Call this once at service startup,
    before anything else runs.
    """
    global _tracer
    resource = Resource.create({"service.name": service_name})

    # Sample a configurable fraction of traces. ParentBased keeps a trace
    # intact end-to-end (if the root is sampled, children are too), avoiding
    # broken partial traces. ratio=1.0 traces everything (dev default).
    ratio = settings.otel_traces_sample_ratio
    if ratio >= 1.0:
        sampler = ALWAYS_ON
    else:
        sampler = ParentBased(root=TraceIdRatioBased(ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)

    import os
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )
    try:
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        )
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info("otel_tracing_initialized", service=service_name)
    except Exception as e:
        # Jaeger not running yet — that is fine for now.
        # Tracing will be a no-op until Jaeger is up in Week 8.
        logger.warning("otel_exporter_unavailable", error=str(e))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer. Setup must have been called first."""
    global _tracer
    if _tracer is None:
        return setup_tracing()
    return _tracer
