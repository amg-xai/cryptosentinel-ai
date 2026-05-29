"""Alert repository — persistence layer for threat alerts."""
from typing import Optional
from datetime import datetime, timedelta

from config.logging_config import get_logger
from src.db.session import get_session, is_available
from src.db.models import AlertRecord, ScanRecord

logger = get_logger(__name__)


def save_alert(assessment_dict: dict) -> Optional[int]:
    if not is_available():
        return None
    try:
        with get_session() as session:
            alert = AlertRecord(
                address=assessment_dict.get("address", ""),
                tx_hash=assessment_dict.get("tx_hash", ""),
                composite_score=assessment_dict.get("composite_score", 0.0),
                confidence=assessment_dict.get("confidence", 0.0),
                action=assessment_dict.get("action", "LOG_ONLY"),
                severity=assessment_dict.get("severity", "LOW"),
                value_at_risk_eth=assessment_dict.get("value_at_risk_eth", 0.0),
                model_scores=assessment_dict.get("model_scores", {}),
                explanation=assessment_dict.get("explanation", {}),
            )
            session.add(alert)
            session.flush()
            alert_id = alert.id
            logger.info("alert_persisted", alert_id=alert_id)
            return alert_id
    except Exception as e:
        logger.error("save_alert_failed", error=str(e))
        return None


def get_recent_alerts(limit: int = 50, severity: Optional[str] = None) -> list:
    if not is_available():
        return []
    try:
        with get_session() as session:
            query = session.query(AlertRecord)
            if severity:
                query = query.filter(AlertRecord.severity == severity)
            query = query.order_by(AlertRecord.created_at.desc()).limit(limit)
            return [a.to_dict() for a in query.all()]
    except Exception as e:
        logger.error("get_recent_alerts_failed", error=str(e))
        return []


def get_alerts_by_address(address: str) -> list:
    if not is_available():
        return []
    try:
        with get_session() as session:
            alerts = (
                session.query(AlertRecord)
                .filter(AlertRecord.address == address)
                .order_by(AlertRecord.created_at.desc())
                .all()
            )
            return [a.to_dict() for a in alerts]
    except Exception as e:
        logger.error("get_alerts_by_address_failed", error=str(e))
        return []


def acknowledge_alert_db(address: str, analyst: str) -> bool:
    if not is_available():
        return False
    try:
        with get_session() as session:
            updated = (
                session.query(AlertRecord)
                .filter(AlertRecord.address == address)
                .filter(AlertRecord.acknowledged == False)
                .update({"acknowledged": True, "acknowledged_by": analyst})
            )
            logger.info("alerts_acknowledged_db", address=address[:12], count=updated)
            return updated > 0
    except Exception as e:
        logger.error("acknowledge_alert_db_failed", error=str(e))
        return False


def get_alert_stats() -> dict:
    if not is_available():
        return {"total": 0, "by_severity": {}}
    try:
        with get_session() as session:
            from sqlalchemy import func
            total = session.query(func.count(AlertRecord.id)).scalar()
            by_severity = {}
            results = (
                session.query(AlertRecord.severity, func.count(AlertRecord.id))
                .group_by(AlertRecord.severity)
                .all()
            )
            for severity, count in results:
                by_severity[severity] = count
            since = datetime.utcnow() - timedelta(hours=24)
            last_24h = (
                session.query(func.count(AlertRecord.id))
                .filter(AlertRecord.created_at >= since)
                .scalar()
            )
            return {"total": total, "by_severity": by_severity, "last_24h": last_24h}
    except Exception as e:
        logger.error("get_alert_stats_failed", error=str(e))
        return {"total": 0, "by_severity": {}}


def save_scan_result(scan_dict: dict, contract_hash: Optional[str] = None) -> Optional[int]:
    if not is_available():
        return None
    try:
        with get_session() as session:
            scan = ScanRecord(
                contract_hash=contract_hash,
                combined_risk_score=scan_dict.get("combined_risk_score", 0.0),
                has_critical=scan_dict.get("has_critical", False),
                vulnerability_count=scan_dict.get("vulnerability_count", {}),
                vulnerabilities=scan_dict.get("vulnerabilities", []),
                scan_duration_ms=scan_dict.get("scan_duration_ms", 0.0),
            )
            session.add(scan)
            session.flush()
            return scan.id
    except Exception as e:
        logger.error("save_scan_result_failed", error=str(e))
        return None
