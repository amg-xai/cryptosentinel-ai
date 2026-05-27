# CryptoSentinel AI — Performance Benchmarks

Hardware: WSL2 Ubuntu 22.04, Intel CPU (12 threads), NVIDIA RTX 3050 6GB

## ML Model Performance

| Model | F1 | PR-AUC | ROC-AUC | Train time |
|---|---|---|---|---|
| Isolation Forest | 0.001 | 0.036 | 0.168 | ~2 min |
| VAE Autoencoder | 0.004 | 0.038 | 0.198 | ~3 min |
| GNN GraphSAGE+GAT | 0.676 | 0.650 | 0.899 | ~15 min (GPU) |

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
| ML-DSA-65 | 0.250 | 0.096 | 3309 | ✅ |
| ML-KEM-768 (encap) | 0.021 | 0.031 | 1088 | ✅ |

## GNN Training

- Final training loss: ~0.012
- GPU utilization: ~87% (RTX 3050)
- Training time: ~15 minutes (100 epochs, AMP enabled)
- Graph size: 203,769 nodes, 234,355 edges

## Test Coverage

- Total tests: 137 passing
- Coverage: ~68%
- Test execution time: ~25 seconds
