"""
Contract scanner — orchestrates rule-based + ML scanning.
"""

import time

from config.logging_config import get_logger
from src.monitoring.metrics import CONTRACTS_SCANNED
from src.scanner.ml_classifier import BytecodeFeatureExtractor, BytecodeRiskScorer
from src.scanner.rule_engine import ScanResult, SolidityRuleEngine

logger = get_logger(__name__)


class ContractScanner:
    """
    Orchestrates rule-based Solidity scanning and bytecode ML scoring.
    """

    def __init__(self):
        self.rule_engine = SolidityRuleEngine()
        self.bytecode_extractor = BytecodeFeatureExtractor()
        self.bytecode_scorer = BytecodeRiskScorer()

    def scan_source(self, source_code: str) -> ScanResult:
        """Scan Solidity source code for vulnerabilities."""
        start = time.perf_counter()
        vulns = self.rule_engine.scan(source_code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Compute overall risk score from vulnerability severities
        if vulns:
            max_score = max(v.severity_score for v in vulns)
            avg_score = sum(v.severity_score for v in vulns) / len(vulns)
            overall_risk = (max_score * 0.7) + (avg_score * 0.3)
        else:
            overall_risk = 0.0

        result = "vulnerable" if vulns else "clean"
        CONTRACTS_SCANNED.labels(result=result).inc()

        logger.info(
            "contract_scan_complete",
            vulnerabilities=len(vulns),
            risk_score=round(overall_risk, 4),
            duration_ms=round(elapsed_ms, 2),
        )

        return ScanResult(
            source_code=source_code,
            vulnerabilities=vulns,
            overall_risk_score=overall_risk,
            lines_of_code=len(source_code.split("\n")),
            scan_duration_ms=elapsed_ms,
        )

    def scan_bytecode(self, bytecode_hex: str) -> dict:
        """Score contract risk from EVM bytecode."""
        features = self.bytecode_extractor.extract(bytecode_hex)
        score = self.bytecode_scorer.score(features)
        logger.info("bytecode_scan_complete", **score)
        return score

    def scan_full(
        self,
        source_code: str | None = None,
        bytecode_hex: str | None = None,
    ) -> dict:
        """Run both source and bytecode analysis."""
        result = {}

        if source_code:
            scan = self.scan_source(source_code)
            result["source_analysis"] = scan.to_dict()

        if bytecode_hex:
            result["bytecode_analysis"] = self.scan_bytecode(bytecode_hex)

        # Combined risk: max of both signals
        scores = []
        if "source_analysis" in result:
            scores.append(result["source_analysis"]["overall_risk_score"])
        if "bytecode_analysis" in result:
            scores.append(result["bytecode_analysis"]["bytecode_risk_score"])

        result["combined_risk_score"] = max(scores) if scores else 0.0
        return result
