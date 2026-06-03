# CryptoSentinel AI — Operations Runbook

Operational procedures for running, monitoring, and troubleshooting
CryptoSentinel AI. Grounded in the system's real components and verified
failure modes.

---

## 1. Starting the system

```bash
cd ~/cryptosentinel-ai && source .venv/bin/activate

# Infrastructure (Kafka, Postgres, Redis, Vault, Prometheus, Grafana,
# Traefik, Jaeger, Loki, promtail)
cd docker && docker compose up -d && cd ..

# API (loads IF/AE/GNN + live model + registry on startup)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &

# Scoring pipeline (Ethereum + Polygon, concurrent)
python -m src.pipeline.scoring_pipeline > logs/pipeline.log 2>&1 &

# Dashboard
python -m streamlit run src/dashboard/app.py
```

**Expected startup time:** the API takes **15-30 seconds** to become healthy
because it loads all models + initializes the DB on startup. Do not assume a
failure until you have polled `/health` for ~30s:

```bash
for i in $(seq 1 20); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)" = "200" ] \
    && echo "ready" && break; sleep 2; done
```

## 2. Health checks

| Component | Check |
|---|---|
| API | `curl localhost:8000/health` → 200, `models_loaded` all true |
| Gateway | `curl localhost:8080/health` (via Traefik) → 200 |
| Pipeline metrics | `curl localhost:8001/metrics` → `transactions_scanned_total` rising |
| Prometheus | `localhost:9090` |
| Grafana | `localhost:3000` (admin/changeme) |
| Jaeger traces | `localhost:16686` |
| Loki logs | Grafana → Explore → Loki, `{job="cryptosentinel"}` |

## 3. Incident response (by alert)

### APIDown
API `/metrics` unreachable. Check the process (`ps aux | grep uvicorn`) and
`logs/api.log`. If models failed to load, check disk + `data/models/`.
Through the gateway, the **Traefik circuit breaker** will already be serving
fast 503s — it auto-recovers once `/health` returns 200 again.

### ModelScoreDriftPSI / ModelScoreDriftPSIModerate
PSI exceeded 0.25 (significant) or 0.10 (moderate). Means the live score
distribution shifted vs the baseline. Investigate: is it a real traffic
change (new attack pattern, new chain volume) or a model issue? The baseline
is collected from the live scoring path:
```bash
python -m src.monitoring.collect_live_baseline --target 500
```
Only re-baseline if the shift is a legitimate new normal, not an incident.

### GNNLatencyHigh
GNN inference p95 elevated. Note: profiling (docs/PERFORMANCE.md) showed
per-request **logging/tracing** usually dominates, not inference. Check the
OTel sample ratio (`OTEL_TRACES_SAMPLE_RATIO`) — 1.0 in prod is expensive;
set to 0.1. Confirm hot-path logs are at debug, not info.

### KafkaConsumerLag
Consumers falling behind. The **producer dead-letter buffer** (bounded 10k)
holds messages if Kafka is unreachable and drains on recovery — so a short
outage delays alerts rather than losing them. For sustained lag, scale
consumers or check broker health (`docker logs cryptosentinel-kafka`).

### CriticalThreatDetected / HighThreatDetectionRate
Expected during real threat activity. Verify via `/alerts/critical`. A sudden
spike with no corresponding on-chain event may indicate a scoring miscalibration
— cross-check `ThreatGraphGrowthAnomaly`.

## 4. Common operational tasks

**Rotate exposed API keys (Alchemy):** regenerate in the Alchemy dashboard,
update `.env` locally (never commit it — gitleaks enforces this in CI).

**Retrain the GNN** (e.g., after drift): `python -m src.ml.gnn.train_gnn`,
then verify the served checkpoint with the backtester before trusting it
(a corrupted checkpoint once scored F1=0.065 — always re-verify):
```bash
python -c "from src.backtesting.backtester import Backtester; b=Backtester(); b.load(); print(b.run()['production_metrics'])"
```

**Performance regression check (local):** `make bench-check`

**Run resilience/chaos tests:** `pytest tests/test_fault_injection.py tests/test_kafka_resilience.py -v`

## 5. Known operational quirks (not bugs)

- **liboqs compiles at container startup (~2 min)** — PQC unavailable until done.
- **OTel trace-export errors if Jaeger is down** — harmless; tracing degrades
  gracefully (the exporter logs a warning and the app continues).
- **`/graph/{address}` may return `degraded: true`** — graph analysis failed
  but wallet stats are still returned (graceful degradation, by design).
- **Model load timing varies 15-30s** — see startup note above.

## 6. References
- Architecture: docs/ARCHITECTURE.md
- Verified metrics: docs/BENCHMARKS.md
- Performance profile: docs/PERFORMANCE.md
- Security posture + accepted risks: SECURITY.md
- Model details + caveats: docs/MODELS.md
