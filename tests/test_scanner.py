"""Tests for smart contract vulnerability scanner."""

from src.scanner.contract_scanner import ContractScanner
from src.scanner.ml_classifier import BytecodeFeatureExtractor, BytecodeRiskScorer
from src.scanner.rule_engine import Severity, SolidityRuleEngine

# --- Vulnerable contract samples ---

REENTRANCY_CONTRACT = """
pragma solidity ^0.8.0;
contract Vulnerable {
    mapping(address => uint) balances;
    function withdraw() public {
        uint amount = balances[msg.sender];
        (bool success,) = msg.sender.call.value(amount)("");
        require(success);
        balances[msg.sender] = 0;
    }
}
"""

TIMESTAMP_CONTRACT = """
pragma solidity ^0.8.0;
contract TimestampVuln {
    function isWinner() public view returns (bool) {
        require(block.timestamp % 2 == 0, "Not a winner");
        return true;
    }
}
"""

SELFDESTRUCT_CONTRACT = """
pragma solidity ^0.8.0;
contract Killable {
    function kill(address payable recipient) public {
        selfdestruct(recipient);
    }
}
"""

SAFE_CONTRACT = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/access/Ownable.sol";
contract SafeContract is Ownable {
    uint256 public value;
    function setValue(uint256 _value) public onlyOwner {
        value = _value;
    }
}
"""

OVERFLOW_CONTRACT = """
pragma solidity ^0.6.0;
contract OverflowVuln {
    uint256 public balance;
    function add(uint256 amount) public {
        balance += amount;
    }
}
"""


# --- Rule engine tests ---


def test_reentrancy_detection():
    engine = SolidityRuleEngine()
    vulns = engine.scan(REENTRANCY_CONTRACT)
    types = [v.vuln_type for v in vulns]
    assert "REENTRANCY" in types


def test_timestamp_detection():
    engine = SolidityRuleEngine()
    vulns = engine.scan(TIMESTAMP_CONTRACT)
    types = [v.vuln_type for v in vulns]
    assert "TIMESTAMP_DEPENDENCY" in types


def test_selfdestruct_detection():
    engine = SolidityRuleEngine()
    vulns = engine.scan(SELFDESTRUCT_CONTRACT)
    types = [v.vuln_type for v in vulns]
    assert "SELFDESTRUCT" in types


def test_selfdestruct_without_access_control_is_critical():
    engine = SolidityRuleEngine()
    vulns = engine.scan(SELFDESTRUCT_CONTRACT)
    selfdestruct_vulns = [v for v in vulns if v.vuln_type == "SELFDESTRUCT"]
    assert any(v.severity == Severity.CRITICAL for v in selfdestruct_vulns)


def test_integer_overflow_detection():
    engine = SolidityRuleEngine()
    vulns = engine.scan(OVERFLOW_CONTRACT)
    types = [v.vuln_type for v in vulns]
    assert "INTEGER_OVERFLOW" in types


def test_safe_contract_has_no_critical():
    engine = SolidityRuleEngine()
    vulns = engine.scan(SAFE_CONTRACT)
    critical = [v for v in vulns if v.severity == Severity.CRITICAL]
    assert len(critical) == 0


def test_vulnerability_has_required_fields():
    engine = SolidityRuleEngine()
    vulns = engine.scan(REENTRANCY_CONTRACT)
    assert len(vulns) > 0
    v = vulns[0]
    assert v.vuln_type != ""
    assert v.description != ""
    assert v.recommendation != ""
    assert 0.0 <= v.confidence <= 1.0


def test_vulnerability_to_dict():
    engine = SolidityRuleEngine()
    vulns = engine.scan(REENTRANCY_CONTRACT)
    d = vulns[0].to_dict()
    required = {
        "vuln_type",
        "severity",
        "severity_score",
        "description",
        "recommendation",
        "confidence",
    }
    assert required.issubset(set(d.keys()))


# --- Bytecode classifier tests ---

SAMPLE_BYTECODE = "0x6080604052348015600f57600080fd5b50603f80601d6000396000f3"


def test_bytecode_feature_extraction():
    extractor = BytecodeFeatureExtractor()
    features = extractor.extract(SAMPLE_BYTECODE)
    assert features.opcode_frequencies.shape == (256,)
    assert features.bytecode_length > 0


def test_bytecode_features_to_numpy():
    extractor = BytecodeFeatureExtractor()
    features = extractor.extract(SAMPLE_BYTECODE)
    arr = features.to_numpy()
    assert len(arr) == 263  # 256 opcodes + 7 extra features


def test_bytecode_scorer_returns_score():
    extractor = BytecodeFeatureExtractor()
    scorer = BytecodeRiskScorer()
    features = extractor.extract(SAMPLE_BYTECODE)
    result = scorer.score(features)
    assert "bytecode_risk_score" in result
    assert 0.0 <= result["bytecode_risk_score"] <= 1.0


def test_selfdestruct_bytecode_scores_high():
    """Bytecode with SELFDESTRUCT opcode (0xFF) should score high."""
    extractor = BytecodeFeatureExtractor()
    scorer = BytecodeRiskScorer()
    # Craft bytecode with SELFDESTRUCT opcode
    bytecode_with_selfdestruct = "0x60806040" + "ff" + "6000"
    features = extractor.extract(bytecode_with_selfdestruct)
    result = scorer.score(features)
    assert result["selfdestruct"] is True
    assert result["bytecode_risk_score"] >= 0.30


def test_invalid_bytecode_returns_empty():
    extractor = BytecodeFeatureExtractor()
    features = extractor.extract("not_valid_hex!!!")
    assert features.bytecode_length == 0


# --- ContractScanner integration tests ---


def test_scanner_scan_source():
    scanner = ContractScanner()
    result = scanner.scan_source(REENTRANCY_CONTRACT)
    assert result.overall_risk_score > 0
    assert len(result.vulnerabilities) > 0


def test_scanner_scan_bytecode():
    scanner = ContractScanner()
    result = scanner.scan_bytecode(SAMPLE_BYTECODE)
    assert "bytecode_risk_score" in result


def test_scanner_scan_full():
    scanner = ContractScanner()
    result = scanner.scan_full(
        source_code=SELFDESTRUCT_CONTRACT,
        bytecode_hex=SAMPLE_BYTECODE,
    )
    assert "source_analysis" in result
    assert "bytecode_analysis" in result
    assert "combined_risk_score" in result
    assert result["combined_risk_score"] > 0


def test_scan_result_has_critical_flag():
    scanner = ContractScanner()
    result = scanner.scan_source(REENTRANCY_CONTRACT)
    assert isinstance(result.has_critical, bool)


def test_scan_result_vulnerability_count():
    scanner = ContractScanner()
    result = scanner.scan_source(REENTRANCY_CONTRACT)
    counts = result.vulnerability_count
    assert "CRITICAL" in counts
    assert "HIGH" in counts
