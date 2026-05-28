"""Smart contract scanner routes."""

from fastapi import APIRouter, Depends, HTTPException

from config.logging_config import get_logger
from src.api.dependencies import get_contract_scanner
from src.api.schemas import ContractScanRequest, ContractScanResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/scan", tags=["Scanner"])


@router.post("/contract", response_model=ContractScanResponse)
async def scan_contract(
    request: ContractScanRequest,
    scanner=Depends(get_contract_scanner),
):
    if not request.source_code and not request.bytecode:
        raise HTTPException(status_code=400, detail="Provide source_code or bytecode")
    result = scanner.scan_full(
        source_code=request.source_code,
        bytecode_hex=request.bytecode,
    )
    source = result.get("source_analysis", {})
    return ContractScanResponse(
        combined_risk_score=result.get("combined_risk_score", 0.0),
        has_critical=source.get("has_critical", False),
        vulnerability_count=source.get("vulnerability_count", {}),
        vulnerabilities=source.get("vulnerabilities", []),
        scan_duration_ms=source.get("scan_duration_ms", 0.0),
    )
