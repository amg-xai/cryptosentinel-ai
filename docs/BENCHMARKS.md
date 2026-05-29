# CryptoSentinel AI — Performance Benchmarks

Hardware: WSL2 Ubuntu 22.04, Intel CPU (12 threads), NVIDIA RTX 3050 6GB

## Feature Engineering Vectorization

Velocity computation benchmark (transactions per time window):

| Implementation | N=100 | N=1000 | N=5000 |
|---|---|---|---|
| Python loop | 2.69ms | 100.69ms | ~2500ms (est.) |
| NumPy vectorized | 0.06ms | 5.71ms | 92.17ms |
| Numba JIT (parallel) | 0.03ms | 7.83ms | 9.76ms |
| **NumPy speedup** | **43.5x** | **17.6x** | — |
| **Numba speedup** | **83.6x** | **12.9x** | **~256x** |

Key insight: Numba wins at large N (5000+) due to CPU parallelization.
NumPy wins at small-medium N due to lower thread overhead.
Production pipeline uses Numba for address history windows (large N).

## ML Model Performance (Elliptic Bitcoin Dataset)

| Model | F1 | PR-AUC | ROC-AUC | Train time |
|---|---|---|---|---|
| Isolation Forest | 0.001 | 0.036 | 0.168 | ~2 min |
| VAE Autoencoder | 0.004 | 0.038 | 0.198 | ~3 min |
| GNN GraphSAGE+GAT | 0.708 | 0.682 | 0.905 | ~15 min (GPU) |

## API Latency

| Endpoint | p50 | p95 | Notes |
|---|---|---|---|
| GET /health | <1ms | <2ms | No ML |
| POST /analyze/wallet | <10ms | <25ms | Graph lookup only |
| POST /analyze/transaction | <50ms | <100ms | Tabular ML |
| POST /scan/contract | <5ms | <15ms | Rule engine |
| GET /alerts | <5ms | <10ms | In-memory |

## PQC vs Classical Cryptography

| Algorithm | Sign (ms) | Verify (ms) | Size (bytes) | Quantum Safe |
|---|---|---|---|---|
| ECDSA-secp256k1 | 0.697 | 0.713 | 71 | ❌ |
| ML-DSA-65 (Dilithium3) | 0.250 | 0.096 | 3309 | ✅ FIPS 204 |
| ML-KEM-768 (Kyber-768) | 0.021 | 0.031 | 1088 | ✅ FIPS 203 |

Key finding: Dilithium3 signs 2.8x faster than ECDSA.
Tradeoff: signatures are 46x larger (3309 vs 71 bytes).

## GNN Training

- Final training loss: ~0.012
- GPU utilization: ~87% (RTX 3050, AMP enabled)
- Training time: ~15 minutes (100 epochs)
- Graph size: 203,769 nodes, 234,355 edges

## Test Coverage
- Total tests: 230 passing (228 fast + 2 slow SHAP integration)
- Coverage: ~59%
- Fast suite (`make test`): ~37 seconds (slow SHAP tests gated)
- Full suite (`make test-full`): ~2 minutes

## Observability Notes

### Multi-process metrics (resolved)
The API and scoring pipeline run as separate processes with separate
Prometheus registries. The pipeline exposes its own metrics endpoint on
port 8001 (`start_http_server`), and Prometheus scrapes both the API
(:8000) and the pipeline (:8001) as distinct jobs. This makes
pipeline-only metrics — transactions scanned, PSI drift, cross-chain
counts — visible in Prometheus and Grafana.

### Drift baseline caveat
The PSI drift baseline is built from the GNN's score distribution on the
Elliptic Bitcoin test set. Live Ethereum/Polygon transactions are scored
via zero-padded live features, a different distribution, so live PSI
against this baseline reads very high by construction. A production
deployment would build the baseline from live-feature scores collected
during a stable window. The detector logic and Prometheus wiring are
production-correct; only the baseline source differs.