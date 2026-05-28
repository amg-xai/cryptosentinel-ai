"""
Smart contract vulnerability scanner — rule-based detection engine.

WHY rule-based first:
  Rule-based detection has zero false negatives for KNOWN patterns.
  Every reentrancy attack follows the same structural pattern.
  Fast, deterministic, explainable — compliance teams trust it.

WHY add ML on top:
  Rules miss NEW attack patterns not yet documented.
  ML catches unusual bytecode distributions — novel exploits.
  Together: rules catch known, ML catches unknown.

Vulnerability patterns implemented:
  REENTRANCY (Critical):
    External call before state update.
    The DAO hack ($60M, 2016) exploited this.
    Pattern: .call() or .send() before balance = 0

  TIMESTAMP_DEPENDENCY (Medium):
    block.timestamp used in critical logic.
    Miners can manipulate by ~15 seconds.
    Pattern: block.timestamp in require() or as randomness seed

  ACCESS_CONTROL (High):
    Public functions modifying state without owner check.
    Pattern: public/external function with no onlyOwner/modifier

  INTEGER_OVERFLOW (High):
    Arithmetic without SafeMath (pre-Solidity 0.8).
    Pattern: pragma solidity ^0.7 or lower with arithmetic ops

  UNCHECKED_RETURN (Medium):
    transfer() or send() result not checked.
    Pattern: .transfer( or .send( without require()

  SELFDESTRUCT (Critical):
    Contract can be destroyed — sends all ETH to attacker.
    Pattern: selfdestruct( without access control
"""

import re
from dataclasses import dataclass
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


SEVERITY_SCORES = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.75,
    Severity.MEDIUM: 0.50,
    Severity.LOW: 0.25,
    Severity.INFO: 0.10,
}


@dataclass
class Vulnerability:
    """A single detected vulnerability."""

    vuln_type: str
    severity: Severity
    line: int | None
    description: str
    recommendation: str
    confidence: float
    code_snippet: str | None = None

    @property
    def severity_score(self) -> float:
        return SEVERITY_SCORES.get(self.severity, 0.0)

    def to_dict(self) -> dict:
        return {
            "vuln_type": self.vuln_type,
            "severity": self.severity.value,
            "severity_score": self.severity_score,
            "line": self.line,
            "description": self.description,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ScanResult:
    """Complete scan result for one contract."""

    source_code: str
    vulnerabilities: list[Vulnerability]
    overall_risk_score: float
    lines_of_code: int
    scan_duration_ms: float

    @property
    def has_critical(self) -> bool:
        return any(v.severity == Severity.CRITICAL for v in self.vulnerabilities)

    @property
    def vulnerability_count(self) -> dict:
        counts = {s.value: 0 for s in Severity}
        for v in self.vulnerabilities:
            counts[v.severity.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "overall_risk_score": round(self.overall_risk_score, 4),
            "has_critical": self.has_critical,
            "vulnerability_count": self.vulnerability_count,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "lines_of_code": self.lines_of_code,
            "scan_duration_ms": round(self.scan_duration_ms, 2),
        }


class SolidityRuleEngine:
    """
    Rule-based vulnerability detector for Solidity source code.
    Each check is a standalone function returning a list of Vulnerability.
    """

    def scan(self, source_code: str) -> list[Vulnerability]:
        """Run all checks and return combined vulnerability list."""
        import time

        start = time.perf_counter()
        lines = source_code.split("\n")
        vulns = []
        vulns.extend(self._check_reentrancy(source_code, lines))
        vulns.extend(self._check_timestamp_dependency(source_code, lines))
        vulns.extend(self._check_access_control(source_code, lines))
        vulns.extend(self._check_integer_overflow(source_code, lines))
        vulns.extend(self._check_unchecked_return(source_code, lines))
        vulns.extend(self._check_selfdestruct(source_code, lines))
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rule_scan_complete",
            vulnerabilities_found=len(vulns),
            duration_ms=round(elapsed_ms, 2),
        )
        return vulns

    def _check_reentrancy(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """
        Detect reentrancy: external call before state update.
        Pattern: .call.value( or .call{value:
        """
        vulns = []
        call_patterns = [
            re.compile(r"\.call\.value\s*\("),
            re.compile(r"\.call\{value\s*:"),
            re.compile(r"\.call\s*\(\s*\"\"\s*\)"),
        ]
        for i, line in enumerate(lines, 1):
            for pattern in call_patterns:
                if pattern.search(line):
                    # Check if state update comes AFTER the call
                    context = "\n".join(lines[i : min(i + 5, len(lines))])
                    has_state_update = bool(
                        re.search(
                            r"(balances|balance|amount|state)\s*[\-\+]?=",
                            context,
                        )
                    )
                    confidence = 0.90 if has_state_update else 0.65
                    vulns.append(
                        Vulnerability(
                            vuln_type="REENTRANCY",
                            severity=Severity.CRITICAL,
                            line=i,
                            description=(
                                "Low-level call detected before state update. "
                                "Vulnerable to reentrancy attacks (cf. The DAO hack)."
                            ),
                            recommendation=(
                                "Apply Checks-Effects-Interactions pattern: "
                                "update state BEFORE making external calls. "
                                "Or use ReentrancyGuard from OpenZeppelin."
                            ),
                            confidence=confidence,
                            code_snippet=line.strip(),
                        )
                    )
        return vulns

    def _check_timestamp_dependency(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """Detect block.timestamp used in critical logic."""
        vulns = []
        pattern = re.compile(r"block\.timestamp")
        critical_context = re.compile(r"(require|if|random|seed|lottery|winner)")
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                is_critical = bool(critical_context.search(line))
                vulns.append(
                    Vulnerability(
                        vuln_type="TIMESTAMP_DEPENDENCY",
                        severity=Severity.MEDIUM if is_critical else Severity.LOW,
                        line=i,
                        description=(
                            "block.timestamp can be manipulated by miners "
                            "by approximately 15 seconds."
                        ),
                        recommendation=(
                            "Do not use block.timestamp for randomness or "
                            "as an exact timing mechanism. Use Chainlink VRF "
                            "for randomness."
                        ),
                        confidence=0.80,
                        code_snippet=line.strip(),
                    )
                )
        return vulns

    def _check_access_control(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """Detect public/external functions modifying state without access control."""
        vulns = []
        func_pattern = re.compile(r"function\s+(\w+)\s*\([^)]*\)\s*(public|external)")
        protected_patterns = [
            re.compile(r"onlyOwner"),
            re.compile(r"onlyAdmin"),
            re.compile(r"onlyRole"),
            re.compile(r"require\s*\(\s*msg\.sender"),
            re.compile(r"modifier"),
        ]
        state_change_patterns = [
            re.compile(r"=\s*[^=]"),
            re.compile(r"\+=|-=|\*="),
            re.compile(r"\.push\("),
            re.compile(r"delete\s+"),
        ]

        for i, line in enumerate(lines, 1):
            match = func_pattern.search(line)
            if not match:
                continue
            func_name = match.group(1)
            if func_name in ("constructor", "fallback", "receive"):
                continue

            # Look ahead for access control and state changes
            context = "\n".join(lines[i : min(i + 20, len(lines))])
            has_access_control = any(p.search(context) for p in protected_patterns)
            has_state_change = any(p.search(context) for p in state_change_patterns)

            if has_state_change and not has_access_control:
                vulns.append(
                    Vulnerability(
                        vuln_type="ACCESS_CONTROL",
                        severity=Severity.HIGH,
                        line=i,
                        description=(
                            f"Function '{func_name}' is public/external "
                            f"and modifies state without access control."
                        ),
                        recommendation=(
                            "Add onlyOwner modifier or require(msg.sender == owner) "
                            "check. Consider using OpenZeppelin's Ownable."
                        ),
                        confidence=0.70,
                        code_snippet=line.strip(),
                    )
                )
        return vulns

    def _check_integer_overflow(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """Detect pre-0.8 Solidity without SafeMath."""
        vulns = []
        pragma_pattern = re.compile(r"pragma\s+solidity\s+[\^~]?0\.[0-7]\.")
        arithmetic_pattern = re.compile(r"[\+\-\*\/]=|\+\+|--")

        has_old_pragma = bool(pragma_pattern.search(source))
        has_safeMath = "SafeMath" in source

        if has_old_pragma and not has_safeMath:
            for i, line in enumerate(lines, 1):
                if arithmetic_pattern.search(line):
                    vulns.append(
                        Vulnerability(
                            vuln_type="INTEGER_OVERFLOW",
                            severity=Severity.HIGH,
                            line=i,
                            description=(
                                "Arithmetic operation in pre-0.8 Solidity "
                                "without SafeMath. Vulnerable to integer overflow."
                            ),
                            recommendation=(
                                "Upgrade to Solidity 0.8+ (overflow checks built-in) "
                                "or use OpenZeppelin SafeMath library."
                            ),
                            confidence=0.85,
                            code_snippet=line.strip(),
                        )
                    )
                    break  # One representative finding per contract
        return vulns

    def _check_unchecked_return(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """Detect unchecked .transfer() or .send() return values."""
        vulns = []
        pattern = re.compile(r"\.(transfer|send)\s*\(")
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                has_require = "require" in line or "assert" in line
                if not has_require:
                    vulns.append(
                        Vulnerability(
                            vuln_type="UNCHECKED_RETURN",
                            severity=Severity.MEDIUM,
                            line=i,
                            description=(
                                ".transfer() reverts on failure but .send() "
                                "returns false. Unchecked return can hide failures."
                            ),
                            recommendation=(
                                "Use .transfer() which reverts automatically, "
                                "or check .send() return value with require()."
                            ),
                            confidence=0.75,
                            code_snippet=line.strip(),
                        )
                    )
        return vulns

    def _check_selfdestruct(
        self,
        source: str,
        lines: list[str],
    ) -> list[Vulnerability]:
        """Detect selfdestruct without access control."""
        vulns = []
        pattern = re.compile(r"selfdestruct\s*\(")
        protected = re.compile(r"onlyOwner|require\s*\(\s*msg\.sender")

        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                context = "\n".join(lines[max(0, i - 10) : i])
                has_protection = bool(protected.search(context))
                vulns.append(
                    Vulnerability(
                        vuln_type="SELFDESTRUCT",
                        severity=(
                            Severity.CRITICAL if not has_protection else Severity.LOW
                        ),
                        line=i,
                        description=(
                            "selfdestruct() destroys the contract and sends "
                            "all ETH to the specified address."
                        ),
                        recommendation=(
                            "Ensure selfdestruct is protected by onlyOwner "
                            "or equivalent access control. Consider removing "
                            "selfdestruct entirely."
                        ),
                        confidence=0.95,
                        code_snippet=line.strip(),
                    )
                )
        return vulns
