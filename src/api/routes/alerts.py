"""Alert management routes."""
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import AlertsListResponse, AlertResponse
from src.api.dependencies import get_alert_manager
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=AlertsListResponse)
async def get_alerts(
    limit: int = 50,
    alert_manager=Depends(get_alert_manager),
):
    top_alerts = alert_manager.get_top_alerts(n=limit)
    stats = alert_manager.get_stats()
    return AlertsListResponse(
        total_active=stats["total_active"],
        alerts=[AlertResponse(**alert.to_dict()) for alert in top_alerts],
        stats=stats,
    )


@router.get("/critical")
async def get_critical_alerts(alert_manager=Depends(get_alert_manager)):
    critical = alert_manager.get_critical_alerts()
    return {"count": len(critical), "alerts": [a.to_dict() for a in critical]}


@router.post("/{address}/acknowledge")
async def acknowledge_alert(
    address: str,
    analyst: str = "analyst",
    alert_manager=Depends(get_alert_manager),
):
    success = alert_manager.acknowledge(address, analyst)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "address": address, "analyst": analyst}
