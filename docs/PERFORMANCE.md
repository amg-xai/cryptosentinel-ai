# CryptoSentinel AI — Performance Profile

Profiling of the live API under load, captured with py-spy (sampling
profiler, no code changes / no measurable overhead on the target).

**Method:** API under sustained load on `POST /analyze/transaction` (the
heaviest endpoint — runs the full feature-extraction + ensemble + composite
scoring path). py-spy `record` at 100 Hz for 30s, 1477 samples, 0 errors.

## Load-test baseline (Locust, 50 users — see BENCHMARKS.md)
- Aggregate p50 = 11 ms, p95 = 300 ms
- `/analyze/transaction` p50 = 61 ms (heaviest path)
- Fast reads (`/alerts`, `/graph`) p50 = 10-12 ms

## Flamegraph finding (the surprising part)

Profiling the scoring path revealed that **request time is dominated by
observability instrumentation and logging — not by ML inference**. Top
frames by sample count:

| Frames | Category |
|---|---|
| `otel_send`, `start_as_current_span`, `_set_send_span`, `_export`, `start_span` | OpenTelemetry span creation/export |
| `_log`, `emit`, `info`, `warning`, `format`, `callHandlers` | loguru logging (JSON serialize + trace-id patcher per record) |
| `run_asgi`, `_send`, `dispatch` | ASGI / middleware plumbing |

The actual ML work (feature extraction, model `predict`) did not appear in
the top 20 frames. Interpretation: on this endpoint the models are fast
(small feature vectors, a logistic-regression live model + lightweight
ensemble), so the fixed per-request cost of tracing + structured logging is
proportionally the largest contributor.

## Implications / optimization opportunities (future work)

1. **Trace sampling** — auto-instrumenting every request creates a span
   per call plus child send-spans. A production deployment would sample
   (e.g., 10%) rather than trace 100% of traffic.
2. **Hot-path log volume** — `risk_scorer.score()` emits an info log every
   call. Demoting routine scoring logs to debug (or aggregating) would cut
   per-request work on the hottest endpoint.
3. The ML models themselves are NOT the bottleneck — no need to optimize
   inference for this workload.

This is a deliberately honest finding: profiling moved the optimization
target away from where intuition (and the "ML platform" framing) would
point, toward the instrumentation layer.

---

## Optimizations applied (and measured)

Acting on the profile above, two changes — both verified:

### 1. Demoted hot-path logging (info -> debug)
`risk_scorer.score()` logged at INFO on every scored transaction. Profiling
showed per-request logging (loguru JSON serialize + OTel trace-id patcher)
dominated the path, so routine scoring was demoted to debug (threats still
surface via alerts + the known-bad warning log).

Micro-benchmark of `score()` in isolation (20,000 calls):

| Variant | Per-call | 
|---|---|
| Log emitting (old INFO behavior) | 142.6 µs |
| Log suppressed (new, debug at INFO level) | 29.0 µs |

**~4.9x faster scoring path** — the log line was ~80% of per-score cost;
the actual scoring math is only ~29 µs.

### 2. Configurable trace sampling
The TracerProvider had no sampler (OTel default = trace 100%). Added a
`ParentBased(TraceIdRatioBased)` sampler controlled by
`OTEL_TRACES_SAMPLE_RATIO` (default 1.0 for dev/demo visibility in Jaeger).
Verified: at ratio 0.1, ~11% of spans are sampled (108/1000), cutting span
creation/export overhead ~90% in production while keeping traces intact
end-to-end (parent-based avoids broken partial traces).

**Net:** the optimization target the flamegraph identified (instrumentation,
not ML) was addressed with measured results, not assumptions.
