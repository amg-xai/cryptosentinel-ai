"""
FastAPI dependency injection — shared resources across routes.
"""

from src.crypto.pqc_handler import PQCAlertSigner
from src.graph.threat_graph import ThreatGraph
from src.graph.cross_chain import CrossChainAnalyzer
from src.response.alert_manager import AlertManager
from src.response.risk_scorer import CompositeRiskScorer
from src.scanner.contract_scanner import ContractScanner

# Singletons initialized once
_threat_graph = ThreatGraph()
_risk_scorer = CompositeRiskScorer()
_alert_manager = AlertManager()
_contract_scanner = ContractScanner()
_pqc_signer = PQCAlertSigner()
_cross_chain = CrossChainAnalyzer()


def get_threat_graph() -> ThreatGraph:
    return _threat_graph


def get_risk_scorer() -> CompositeRiskScorer:
    return _risk_scorer


def get_alert_manager() -> AlertManager:
    return _alert_manager


def get_contract_scanner() -> ContractScanner:
    return _contract_scanner


def get_pqc_signer() -> PQCAlertSigner:
    return _pqc_signer

def get_cross_chain() -> CrossChainAnalyzer:
    return _cross_chain