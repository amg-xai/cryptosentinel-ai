"""
API request and response schemas.
Every endpoint uses typed Pydantic models — no raw dicts.

WHY Pydantic schemas:
  Type safety at the API boundary.
  Auto-generates OpenAPI documentation at /docs.
  Input validation is automatic — invalid requests return 422.
  Response serialization is controlled and predictable.
"""

from pydantic import BaseModel, Field


class WalletAnalysisRequest(BaseModel):
    address: str = Field(description="Ethereum wallet address (0x...)")
    include_graph: bool = Field(
        default=True,
        description="Include graph-based risk signals",
    )
    include_history: bool = Field(
        default=False,
        description="Include transaction history in response",
    )


class ModelScoresResponse(BaseModel):
    gnn: float = Field(description="GNN risk score [0,1]")
    autoencoder: float = Field(description="Autoencoder risk score [0,1]")
    isolation_forest: float = Field(description="IF risk score [0,1]")
    graph_centrality: float = Field(description="Graph centrality score [0,1]")


class WalletAnalysisResponse(BaseModel):
    address: str
    composite_score: float
    confidence: float
    action: str
    severity: str
    value_at_risk_eth: float
    model_scores: ModelScoresResponse
    explanation: dict
    graph_stats: dict | None = None


class TransactionScanRequest(BaseModel):
    tx_hash: str
    from_addr: str
    to_addr: str | None = None
    value_eth: float = 0.0
    gas: int = 21000
    gas_price: int = 1_000_000_000
    is_contract_call: bool = False
    is_contract_creation: bool = False
    input_data: str = "0x"
    block_timestamp: float = 0.0
    chain_name: str = "ethereum-sepolia"


class TransactionScanResponse(BaseModel):
    tx_hash: str
    risk_score: float
    action: str
    severity: str
    features_extracted: int
    explanation: dict


class ContractScanRequest(BaseModel):
    source_code: str | None = Field(
        default=None,
        description="Solidity source code",
    )
    bytecode: str | None = Field(
        default=None,
        description="EVM bytecode hex string",
    )


class ContractScanResponse(BaseModel):
    combined_risk_score: float
    has_critical: bool
    vulnerability_count: dict
    vulnerabilities: list[dict]
    scan_duration_ms: float


class AlertResponse(BaseModel):
    address: str
    tx_hash: str
    composite_score: float
    severity: str
    action: str
    value_at_risk_eth: float
    current_priority: float
    created_at: float
    acknowledged: bool
    explanation: dict


class AlertsListResponse(BaseModel):
    total_active: int
    alerts: list[AlertResponse]
    stats: dict


class GraphQueryResponse(BaseModel):
    address: str
    wallet_stats: dict
    laundering_paths: list[list[str]]
    ancestors: list[str]
    round_trips: list[dict]
    cluster_id: int | None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    uptime_seconds: int
    models_loaded: dict
