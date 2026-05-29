# 🛡️ CryptoSentinel AI

> Quantum-resistant blockchain threat intelligence platform with GNN-based money laundering detection

[![CI](https://github.com/amg-xai/cryptosentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/amg-xai/cryptosentinel-ai/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-215%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

CryptoSentinel AI monitors live Ethereum blockchain transactions, detects money laundering and fraud using Graph Neural Networks, scans smart contracts for vulnerabilities, and signs all threat alerts with post-quantum cryptography.

**Headline result:** GNN achieves **F1=0.708, ROC-AUC=0.905** on the Elliptic Bitcoin dataset — compared to Isolation Forest F1=0.001. Graph structure is the key signal for blockchain fraud detection.

## Quick Start

```bash
# Start infrastructure
cd docker && docker compose up -d && cd ..

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start dashboard
python -m streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full system diagram.

**Data flow:**
Ethereum Sepolia → BlockIngester → Kafka → Feature Extraction
→ Ensemble ML (IF + VAE + GNN) → ThreatGraph → Risk Scorer
→ PQC-signed Alerts → FastAPI → Streamlit SOC Dashboard
## ML Results (Elliptic Bitcoin Dataset)

| Model | F1 | ROC-AUC |
|---|---|---|
| Isolation Forest | 0.001 | 0.168 |
| VAE Autoencoder | 0.004 | 0.198 |
| **GNN (GraphSAGE+GAT)** | **0.708** | **0.905** |

## Tech Stack

| Layer | Technology |
|---|---|
| Blockchain | Web3.py, Alchemy, Ethereum Sepolia |
| Streaming | Apache Kafka (confluent-kafka) |
| ML | PyTorch, PyTorch Geometric, scikit-learn |
| GNN | GraphSAGE + GAT, NeighborLoader |
| Graph | NetworkX, Union-Find, Louvain |
| PQC | liboqs (ML-DSA-65, ML-KEM-768) |
| API | FastAPI, Pydantic, Prometheus |
| Dashboard | Streamlit (dark SOC theme) |
| Infrastructure | Docker, Kubernetes, GitHub Actions |

## Key Features

- **Live blockchain ingestion** — real Ethereum Sepolia transactions via WebSocket
- **3-tier ML detection** — Isolation Forest + VAE Autoencoder + GNN ensemble
- **Threat graph engine** — laundering path detection, wallet clustering, peel-back tracing
- **Smart contract scanner** — reentrancy, access control, selfdestruct detection
- **Post-quantum cryptography** — NIST FIPS 203/204 (ML-DSA-65 + ML-KEM-768)
- **SOC dashboard** — 4-page dark theme UI with real-time alerts
- **Production infrastructure** — Docker, Kubernetes with HPA, Prometheus + Grafana

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Models & Benchmarks](docs/MODELS.md)
- [Performance](docs/BENCHMARKS.md)
- [K8s Deployment](k8s/README.md)

## Tests
```bash
make test       # 213 fast tests, ~37s (skips slow SHAP integration tests)
make test-full  # all 215 tests incl. SHAP explainability, ~2min
```

## Research Extensions

- Federated learning across multiple blockchain monitoring nodes
- Additional chains (Solana, Bitcoin) with cross-chain laundering detection (Ethereum + Polygon already live)
- Transformer-based smart contract vulnerability detection
- Real-time SOAR integration (Splunk, Palo Alto XSOAR)
- ZK-proof based privacy-preserving threat intelligence sharing
