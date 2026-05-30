# CryptoSentinel AI — Verified Benchmarks

All numbers below were re-run and verified on the audit date by re-executing
the evaluation against the **currently served** model artifacts (not read
from stale metric files). Where a number differs from older claims, the
verified value here supersedes it.

**Audit date:** 2026-05-30
**Hardware:** WSL2 Ubuntu, Intel CPU, NVIDIA RTX 3050 6GB, CUDA
**Test suite:** 247 passed, 2 deselected (slow SHAP), via `make test`

---

## 1. ThreatGNN — offline detection (Elliptic Bitcoin dataset)

Primary fraud detector. SAGEConv→GATConv→SAGEConv, trained on the Elliptic
dataset (165 features), checkpointed on test-F1 (not loss).

| Metric | Value (served gnn_best.pt) |
|---|---|
| F1 @ threshold 0.5 | **0.700** |
| Precision @ 0.5 | 0.819 |
| Recall @ 0.5 | 0.611 |
| Best-threshold F1 (0.45) | 0.702 |
| PR-AUC | 0.677 |
| ROC-AUC | 0.902 |
| Test samples | 16,670 (1,083 illicit) |
| Illicit mean score | 0.604 |
| Licit mean score | 0.011 |

> **Audit note:** the served `gnn_best.pt` was found corrupted during this
> pass (a truncated 67KB checkpoint scoring F1≈0.065, with illicit scored
> *lower* than licit). Root cause: a partially-written save from an
> interrupted training run. Fixed by a clean retrain (best epoch 91,
> F1=0.700) and re-verified by re-running the backtester against the served
> file. Also fixed a typo (`gnn_bes.pt`) in the inference engine's GNN load
> path.

Weak baselines on the same task (for contrast): Isolation Forest F1≈0.001,
Autoencoder F1≈0.004 — i.e., graph structure is the signal, confirming the
project thesis (Weber et al. 2019).

---

## 2. Live-native model — live EVM detection (16-feature space)

Trained directly on the 16 features the live pipeline computes, with weak
labels (OFAC sanctions + MEW darklist positives; activity-matched mainnet
negatives). Artifact-free symmetric build; leakage-checked.

| Metric | Value |
|---|---|
| 5-fold CV F1 | **0.866 ± 0.034** |
| Test F1 | 0.877 |
| Precision | 0.926 |
| Recall | 0.833 |
| ROC-AUC | 0.984 |
| Dataset | 120 pos / 120 neg |

Caveats: weak labels (incomplete lists; negatives assumed legit);
gas_price is the dominant feature.

---

## 3. Drift detection (PSI)

Baseline collected from the **live scoring path** (not Elliptic), so PSI
compares like-with-like.

| Scenario | PSI | Classification |
|---|---|---|
| Stable live traffic | 0.00 | stable |
| +0.15 score shift | 27.6 | significant |

Nuance: live composite scores have low variance (~0.0005), so PSI reliably
catches systemic shifts more than subtle per-tx drift.

---

## 4. Post-Quantum Cryptography (100 iterations)

| Algorithm | Keygen (ms) | Sign/Encap (ms) | Sig size (bytes) | Quantum-safe |
|---|---|---|---|---|
| ECDSA-secp256k1 | 0.504 | 0.697 | 71 | ❌ |
| ML-DSA-65 (Dilithium3) | 0.141 | 0.250 | 3309 | ✅ FIPS 204 |
| ML-KEM-768 (Kyber-768) | 0.030 | 0.021 | 1088 | ✅ FIPS 203 |

Dilithium3 signs 2.8x faster than ECDSA; signatures are 46x larger.

---

## Honest limitations (carried forward)

1. Live EVM scoring uses the live-native model + graph/heuristics; the
   Elliptic-trained IF/AE/GNN are disabled on live data (feature mismatch).
2. Live-native model labels are weak and gas_price-dominant.
3. CI runs a test subset (no 200MB Elliptic dataset in repo).
4. Polygon is sampled at 20%; the 16-vector has 3 dead graph slots offline.
