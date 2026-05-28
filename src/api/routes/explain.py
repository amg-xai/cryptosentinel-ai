"""Explainability routes — SHAP + GNN explanations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.logging_config import get_logger
from src.ml.tabular.feature_engineer import FeatureEngineer

logger = get_logger(__name__)
router = APIRouter(prefix="/explain", tags=["Explainability"])

_feature_engineer = FeatureEngineer()
_explainer = None


def get_explainer():
    global _explainer
    if _explainer is None:
        from src.ml.explainability.gnn_explainer import RiskExplainer

        _explainer = RiskExplainer()
        _explainer.load()
    return _explainer


class ExplainRequest(BaseModel):
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


@router.post("/transaction")
async def explain_transaction(request: ExplainRequest):
    """
    Generate SHAP + GNN explanation for a transaction risk score.
    Returns top features driving the risk assessment.
    """
    try:
        explainer = get_explainer()

        tx_payload = request.model_dump()
        features = _feature_engineer.extract(tx_payload)
        features_array = features.to_numpy()

        explanation = explainer.explain_transaction(features_array)

        return {
            "tx_hash": request.tx_hash,
            "from_addr": request.from_addr,
            "explanation": explanation,
        }

    except Exception as e:
        logger.error("explain_transaction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shap-waterfall")
async def get_shap_waterfall(request: ExplainRequest):
    """
    Get SHAP waterfall chart data for a transaction.
    Returns data formatted for Plotly waterfall chart.
    """
    try:
        explainer = get_explainer()

        tx_payload = request.model_dump()
        features = _feature_engineer.extract(tx_payload)
        features_array = features.to_numpy()

        waterfall_data = explainer.get_shap_waterfall_data(features_array)

        return {
            "tx_hash": request.tx_hash,
            "waterfall_data": waterfall_data,
        }

    except Exception as e:
        logger.error("shap_waterfall_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
