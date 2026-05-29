"""Alert management routes."""
from src.db.alert_repository import (
    get_recent_alerts as db_get_recent,
    get_alerts_by_address as db_get_by_address,
    get_alert_stats as db_get_stats,
)
from src.db.session import is_available as db_available
from fastapi import APIRouter, Depends, HTTPException

from config.logging_config import get_logger
from src.api.dependencies import get_alert_manager
from src.api.schemas import AlertResponse, AlertsListResponse

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

@router.get("/history")
async def get_alert_history(
    limit: int = 50,
    severity: str = None,
):
    """Get historical alerts from PostgreSQL."""
    if not db_available():
        return {"alerts": [], "source": "database_unavailable"}
    alerts = db_get_recent(limit=limit, severity=severity)
    return {"alerts": alerts, "count": len(alerts), "source": "postgresql"}


@router.get("/history/{address}")
async def get_address_history(address: str):
    """Get all historical alerts for a specific address."""
    if not db_available():
        return {"alerts": [], "source": "database_unavailable"}
    alerts = db_get_by_address(address)
    return {"address": address, "alerts": alerts, "count": len(alerts)}


@router.get("/stats/database")
async def get_database_stats():
    """Get aggregate alert statistics from PostgreSQL."""
    if not db_available():
        return {"total": 0, "source": "database_unavailable"}
    stats = db_get_stats()
    stats["source"] = "postgresql"
    return stats