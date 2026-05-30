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
### Offline benchmark vs live scoring (important)

The ensemble weights above apply to the **offline benchmark** on the
Elliptic dataset, where all models score their native 165-feature space.
This is where the F1=0.708 GNN result comes from, and it is fully
legitimate.

On **live EVM transactions** the picture is different and we are explicit
about it. Live transactions yield 16 engineered features, zero-padded to
165 to fit the models' input shape. The models were trained on the
Elliptic feature semantics, so on padded live vectors their output
collapses to a near-constant value (verified: live composite std ~0.0005).
Those scores are not trustworthy, so the live pipeline sets
`trust_live_model_scores=False` by default and marks IF/AE/GNN as
unavailable. The composite then re-normalizes onto signals that *do* work
on live data:

- **graph centrality** — treated as a weak structural prior (dampened
  ×0.5 when it is the only signal, so high-degree exchanges/routers are
  NOT flagged on structure alone)
- **velocity** — burst behavior (×1.3)
- **cross-chain** — bridge / multi-chain actor (×1.15–1.4)
- **known-bad address** — floors the score at 0.9

Net effect: live detection is graph-structural + behavioral, not ML.
Closing this gap — training on a consistent live-feature space with weak
labels from public scam-address lists — is the next planned step.

## Post-Quantum Cryptography Benchmarks

| Algorithm | Keygen (ms) | Sign/Encap (ms) | Size (bytes) | Quantum Safe |
|---|---|---|---|---|
| ECDSA-secp256k1 | 0.504 | 0.697 | 71 | ❌ |
| ML-DSA-65 (Dilithium3) | 0.141 | 0.250 | 3309 | ✅ FIPS 204 |
| ML-KEM-768 (Kyber-768) | 0.030 | 0.021 | 1088 | ✅ FIPS 203 |

Key finding: Dilithium3 signs **2.8x faster** than ECDSA.
Tradeoff: signatures are **46x larger** (3309 vs 71 bytes).
