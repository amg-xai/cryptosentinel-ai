"""Analysis routes — wallet and transaction risk scoring."""

from fastapi import APIRouter, Depends

from config.logging_config import get_logger
from src.api.dependencies import get_alert_manager, get_risk_scorer, get_threat_graph
from src.api.model_registry import registry
from src.api.schemas import (
    ModelScoresResponse,
    TransactionScanRequest,
    TransactionScanResponse,
    WalletAnalysisRequest,
    WalletAnalysisResponse,
)
from src.ml.tabular.feature_engineer import FeatureEngineer
from src.response.risk_scorer import ModelScores

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])
_feature_engineer = FeatureEngineer()


@router.post("/wallet", response_model=WalletAnalysisResponse)
async def analyze_wallet(
    request: WalletAnalysisRequest,
    threat_graph=Depends(get_threat_graph),
    risk_scorer=Depends(get_risk_scorer),
    alert_manager=Depends(get_alert_manager),
):
    address = request.address
    graph_stats = threat_graph.get_wallet_stats(address)
    graph_centrality = 0.0
    if graph_stats:
        total_deg = graph_stats.get("in_degree", 0) + graph_stats.get("out_degree", 0)
        graph_centrality = min(1.0, total_deg / 100)

    model_scores = ModelScores(
        gnn=-1.0,
        autoencoder=-1.0,
        isolation_forest=-1.0,
        graph_centrality=graph_centrality,
    )
    value_eth = graph_stats.get("total_value_eth", 0.0) if graph_stats else 0.0
    assessment = risk_scorer.score(
        address=address,
        tx_hash="wallet_analysis",
        model_scores=model_scores,
        value_eth=value_eth,
    )
    if assessment.is_threat:
        alert_manager.add_or_update(assessment)

    return WalletAnalysisResponse(
        address=address,
        composite_score=assessment.composite_score,
        confidence=assessment.confidence,
        action=assessment.action.value,
        severity=assessment.severity,
        value_at_risk_eth=assessment.value_at_risk_eth,
        model_scores=ModelScoresResponse(
            gnn=model_scores.gnn,
            autoencoder=model_scores.autoencoder,
            isolation_forest=model_scores.isolation_forest,
            graph_centrality=graph_centrality,
        ),
        explanation=assessment.explanation,
        graph_stats=graph_stats if request.include_graph else None,
    )


@router.post("/transaction", response_model=TransactionScanResponse)
async def analyze_transaction(
    request: TransactionScanRequest,
    risk_scorer=Depends(get_risk_scorer),
    alert_manager=Depends(get_alert_manager),
):
    tx_payload = {
        "tx_hash": request.tx_hash,
        "from_addr": request.from_addr,
        "to_addr": request.to_addr,
        "value_eth": request.value_eth,
        "gas": request.gas,
        "gas_price": request.gas_price,
        "is_contract_call": request.is_contract_call,
        "is_contract_creation": request.is_contract_creation,
        "input_data": request.input_data,
        "block_timestamp": request.block_timestamp,
        "chain_name": request.chain_name,
    }
    features = _feature_engineer.extract(tx_payload)
    features_array = features.to_numpy()
    tabular_scores = registry.score_tabular(features_array)
    model_scores = ModelScores(
        gnn=-1.0,
        autoencoder=tabular_scores.get("autoencoder", -1.0),
        isolation_forest=tabular_scores.get("isolation_forest", -1.0),
        graph_centrality=0.0,
    )
    assessment = risk_scorer.score(
        address=request.from_addr,
        tx_hash=request.tx_hash,
        model_scores=model_scores,
        value_eth=request.value_eth,
    )
    if assessment.is_threat:
        alert_manager.add_or_update(assessment)

    return TransactionScanResponse(
        tx_hash=request.tx_hash,
        risk_score=assessment.composite_score,
        action=assessment.action.value,
        severity=assessment.severity,
        features_extracted=features.feature_dim,
        explanation=assessment.explanation,
    )
