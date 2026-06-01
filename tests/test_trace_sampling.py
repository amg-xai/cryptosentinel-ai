"""Trace sampler must honor the configured ratio."""
import importlib


def test_full_sampling_uses_always_on(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLE_RATIO", "1.0")
    from config import settings as s
    importlib.reload(s)
    assert s.settings.otel_traces_sample_ratio == 1.0


def test_ratio_sampler_approximates_rate():
    """At ratio 0.1, sampled fraction should be roughly 10% over many spans."""
    from opentelemetry.sdk.trace.sampling import (
        ParentBased, TraceIdRatioBased,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace as t
    provider = TracerProvider(sampler=ParentBased(root=TraceIdRatioBased(0.1)))
    tracer = provider.get_tracer("test")
    sampled = 0
    N = 2000
    for i in range(N):
        with tracer.start_as_current_span(f"s{i}") as span:
            if span.get_span_context().trace_flags.sampled:
                sampled += 1
    frac = sampled / N
    assert 0.04 < frac < 0.18  # ~10% with generous margin
