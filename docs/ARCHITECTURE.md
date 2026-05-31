# CryptoSentinel AI — System Architecture

## Overview

CryptoSentinel AI is a production-grade blockchain threat intelligence platform
that detects money laundering, smart contract exploits, and coordinated attacks
using a three-tier ML detection system with post-quantum cryptographic signing.

## System Architecture Diagram
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│                                                                 │
│  Ethereum Sepolia ──► BlockIngester ──► asyncio.Queue           │
│  Polygon Mumbai   ──► (Web3.py)         (backpressure)          │
│                              │                                  │
│                              ▼                                  │
│                     Kafka Topics                                │
│              raw.transactions.ethereum                          │
│              raw.transactions.polygon                           │
│              processed.features                                 │
│              alerts.critical / alerts.high                      │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML DETECTION LAYER                          │
│                                                                 │
│  Isolation Forest ──► calibrated anomaly score (weight: 0.10)  │
│  VAE Autoencoder  ──► reconstruction error  (weight: 0.25)     │
│  GNN (SAGE+GAT)   ──► graph-based score     (weight: 0.55)     │
│                              │                                  │
│                              ▼                                  │
│              Ensemble Composite Risk Score [0,1]                │
│         LOG(<0.5) │ WATCHLIST(0.5-0.7) │ QUARANTINE(0.7-0.85) │
│                              │ EMERGENCY(>0.85)                 │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPH INTELLIGENCE LAYER                     │
│                                                                 │
│  ThreatGraph (NetworkX DiGraph)                                 │
│  ├── Laundering path detection (depth-limited BFS)             │
│  ├── Peel-back fund tracing (backward from flagged wallet)     │
│  ├── Round-trip detection (24h window)                         │
│  ├── Union-Find wallet clustering (common-input-ownership)     │
│  └── Louvain community detection (modularity optimization)     │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                  POST-QUANTUM CRYPTO LAYER                      │
│                                                                 │
│  ML-DSA-65 (Dilithium3) ──► signs every threat alert           │
│  ML-KEM-768 (Kyber-768) ──► key exchange for intel sharing     │
│  Hybrid: ECDSA + Dilithium3 (NIST transition strategy)         │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                       API + DASHBOARD                           │
│                                                                 │
│  FastAPI ──► /analyze/wallet  /analyze/transaction             │
│             /graph/{address}  /scan/contract                   │
│             /alerts           /health  /metrics                │
│                                                                 │
│  Streamlit SOC Dashboard (dark theme)                          │
│  ├── Page 1: SOC Overview (KPIs, alert feed)                   │
│  ├── Page 2: Threat Graph (laundering paths, clustering)       │
│  ├── Page 3: Contract Scanner (vulnerability detection)        │
│  └── Page 4: Model Monitor (benchmarks, training history)      │
└─────────────────────────────────────────────────────────────────┘
## Key Design Decisions

### Why GNN over pure tabular ML?
Isolation Forest and VAE achieve F1 < 0.01 on the Elliptic dataset.
GNN achieves F1=0.676. Graph structure — wallet-to-wallet transaction
patterns — is the primary signal for money laundering detection.
This is consistent with Weber et al. 2019 (Anti-Money Laundering in Bitcoin).

### Why GraphSAGE + GAT hybrid?
- GraphSAGE: inductive learning — works on new wallets not seen during training
- GAT: attention weights per edge — explains WHICH neighbors drove the prediction
- GCN (the alternative): transductive only — cannot handle new nodes at inference time

### Why Kafka over direct database writes?
- Multiple consumers read the same transaction independently
- ML scorer, graph engine, and alert manager consume concurrently
- Kafka provides replay — reprocess historical blocks if a new model is deployed
- Bounded queue provides backpressure — no data loss under load

### Why post-quantum cryptography?
Shor's algorithm on a cryptographically-relevant quantum computer breaks
ECDSA (secp256k1) in polynomial time. "Harvest now, decrypt later" attacks
are already happening. NIST finalized ML-DSA and ML-KEM in 2024.
Our hybrid approach (ECDSA + Dilithium3) is the NIST-recommended transition strategy.

### Why ensemble scoring over single model?
No single model is trusted alone in production fraud systems.
Each model catches different attack patterns:
- IF: statistical outliers in feature space
- AE: distributional anomalies (novel patterns)
- GNN: graph-structural laundering patterns
Combined weight: IF(0.10) + AE(0.25) + GNN(0.55) + graph_centrality(0.10)

## Technology Choices

| Component | Technology | Reason |
|---|---|---|
| Blockchain | Web3.py + Alchemy | Production RPC, WebSocket support |
| Streaming | Kafka (confluent-kafka) | Enterprise-grade, replayable, backpressure |
| ML tabular | scikit-learn + PyTorch | Industry standard |
| GNN | PyTorch Geometric | State-of-the-art GNN library |
| Graph | NetworkX | Flexible, well-documented graph algorithms |
| PQC | liboqs (NIST reference) | Official NIST PQC implementation |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Dashboard | Streamlit | Rapid SOC UI development |
| Observability | Prometheus + Grafana | Industry standard metrics stack |
| Container | Docker + Kubernetes | Production deployment standard |
| API gateway | Traefik v3 | Reverse proxy, circuit breaker, active health checks |


## API Gateway & Resilience

Traefik v3 sits in front of the FastAPI service as the single entry point
(`:8080`), providing production-grade resilience patterns:

- **Circuit breaker** — trips OPEN when the backend shows >30% 5xx
  responses, >25% network errors, or p50 latency >3s. While open, the
  gateway fails fast (503) instead of piling requests onto a struggling
  backend, then transitions through a half-open *recovering* state to probe
  for recovery before closing.
- **Active health checks** — Traefik polls `/health` every 10s and pulls an
  unhealthy backend out of rotation automatically.
- **Bounded retry** — up to 3 retries on transient network errors.
- **Observable** — Traefik exposes its own Prometheus metrics, so gateway
  behavior (request rates, circuit-breaker state) is visible alongside
  application metrics.

Verified behavior: with the backend killed, the gateway returns fast 503s
and the breaker trips; on backend restart, the breaker auto-recovers and
routing resumes — demonstrated against real process failures, not mocks.

**Dev vs production note:** in development the gateway routes to a
host-run API via `host.docker.internal` and the Traefik dashboard is
exposed insecurely on `:8090`. In production the API runs as a
containerized service behind Traefik with the dashboard secured/disabled,
and routing uses Kubernetes service discovery rather than a static host
address.
