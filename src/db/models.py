"""SQLAlchemy ORM models for persistent storage."""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Index, JSON,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), nullable=False, index=True)
    tx_hash = Column(String(66), nullable=False)
    composite_score = Column(Float, nullable=False, index=True)
    confidence = Column(Float, default=0.0)
    action = Column(String(32), nullable=False)
    severity = Column(String(16), nullable=False, index=True)
    value_at_risk_eth = Column(Float, default=0.0)
    model_scores = Column(JSON, default=dict)
    explanation = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_severity_created", "severity", "created_at"),
        Index("idx_address_score", "address", "composite_score"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "address": self.address,
            "tx_hash": self.tx_hash,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "action": self.action,
            "severity": self.severity,
            "value_at_risk_eth": self.value_at_risk_eth,
            "model_scores": self.model_scores,
            "explanation": self.explanation,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ScanRecord(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    contract_hash = Column(String(66), nullable=True, index=True)
    combined_risk_score = Column(Float, nullable=False)
    has_critical = Column(Boolean, default=False, index=True)
    vulnerability_count = Column(JSON, default=dict)
    vulnerabilities = Column(JSON, default=list)
    scan_duration_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contract_hash": self.contract_hash,
            "combined_risk_score": self.combined_risk_score,
            "has_critical": self.has_critical,
            "vulnerability_count": self.vulnerability_count,
            "scan_duration_ms": self.scan_duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
