"""The loguru patcher must stamp the active OTel trace_id onto log records,
so logs correlate to traces in Jaeger. Falls back to '-' with no span."""
import re
from config.logging_config import _inject_trace_context


def test_no_span_falls_back_to_placeholder():
    record = {"extra": {}}
    _inject_trace_context(record)
    assert record["extra"]["trace_id"] == "-"


def test_active_span_injects_real_trace_id():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    # Use a real provider so spans have valid contexts
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("unit-span") as span:
        record = {"extra": {}}
        _inject_trace_context(record)
        expected = format(span.get_span_context().trace_id, "032x")
        assert record["extra"]["trace_id"] == expected
        # 32 lowercase hex chars, not the placeholder
        assert re.fullmatch(r"[a-f0-9]{32}", record["extra"]["trace_id"])


def test_preserves_existing_extra_keys():
    record = {"extra": {"module": "src.api.main"}}
    _inject_trace_context(record)
    assert record["extra"]["module"] == "src.api.main"
    assert "trace_id" in record["extra"]
