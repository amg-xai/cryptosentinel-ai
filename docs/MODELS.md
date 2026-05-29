# CryptoSentinel AI — Model Documentation

## Dataset: Elliptic Bitcoin Dataset

- 203,769 Bitcoin transactions
- 234,355 directed edges (payment flows)
- 165 features per transaction (local + aggregated neighborhood)
- Labels: 1=illicit (4,545), 2=licit (42,019), unknown (157,205)
- 49 time steps (temporal structure)

**Critical preprocessing decision: chronological split**
- Train: time steps 1-34 (136,265 transactions)
- Test: time steps 35-49 (67,504 transactions)
- Random split would cause temporal leakage — model sees future during training

## Model Results on Elliptic Test Set

| Model | F1 Score | PR-AUC | ROC-AUC | Illicit Detected |
|---|---|---|---|---|
| Isolation Forest | 0.001 | 0.036 | 0.168 | 1/1083 |
| VAE Autoencoder | 0.004 | 0.038 | 0.198 | 2/1083 |
| **GNN (GraphSAGE+GAT)** | **0.708** | **0.682** | **0.905** | **635/1083** |

**Why tabular models perform poorly:**
Illicit transactions do not form statistical outliers in raw feature space.
Graph structure (which wallets transact with which) is the primary signal.
This is consistent with published literature (Weber et al. 2019).

## GNN Architecture
**Training details:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Class weights: illicit weight = n_licit/n_illicit ≈ 7.6
- Mini-batch: NeighborLoader([25,10,5], batch_size=512)
- Mixed precision: AMP with GradScaler (RTX 3050)
- Epochs: 100, LR scheduler: ReduceLROnPlateau(patience=10)

## Ensemble Weights

| Model | Weight | Rationale |
|---|---|---|
| GNN | 0.55 | Primary detector, proven F1=0.708 |
| Autoencoder | 0.25 | Catches novel patterns GNN misses |
| Isolation Forest | 0.10 | Fast baseline, weak on Elliptic |
| Graph centrality | 0.10 | Structural signal from NetworkX |

## Post-Quantum Cryptography Benchmarks

| Algorithm | Keygen (ms) | Sign/Encap (ms) | Size (bytes) | Quantum Safe |
|---|---|---|---|---|
| ECDSA-secp256k1 | 0.504 | 0.697 | 71 | ❌ |
| ML-DSA-65 (Dilithium3) | 0.141 | 0.250 | 3309 | ✅ FIPS 204 |
| ML-KEM-768 (Kyber-768) | 0.030 | 0.021 | 1088 | ✅ FIPS 203 |

Key finding: Dilithium3 signs **2.8x faster** than ECDSA.
Tradeoff: signatures are **46x larger** (3309 vs 71 bytes).
