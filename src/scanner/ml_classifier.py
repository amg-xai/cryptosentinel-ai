"""
ML-based smart contract bytecode classifier.

WHY bytecode analysis:
  Source code is not always available (closed-source contracts).
  Bytecode analysis catches patterns in compiled output.
  Opcode frequency vectors are the feature representation.

WHY XGBoost:
  Gradient boosting on tabular opcode features.
  Fast inference, good on structured data, interpretable.
  SHAP values available for per-prediction explanation.

Feature extraction:
  EVM opcodes are 1-byte instructions.
  We count frequency of each opcode in the bytecode.
  256 possible opcodes → 256-dimensional feature vector.
  High CALL frequency → likely external calls (reentrancy risk).
  High SSTORE/SLOAD → heavy state access (storage patterns).
"""

from dataclasses import dataclass

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

# EVM opcodes most relevant to vulnerability detection
VULNERABILITY_OPCODES = {
    0x00: "STOP",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xFA: "STATICCALL",
    0xFF: "SELFDESTRUCT",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x5A: "GAS",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x33: "CALLER",
    0x32: "ORIGIN",
}


@dataclass
class BytecodeFeatures:
    """Feature vector extracted from contract bytecode."""

    opcode_frequencies: np.ndarray  # 256-dim vector
    call_frequency: float
    sstore_frequency: float
    sload_frequency: float
    selfdestruct_present: bool
    timestamp_present: bool
    delegatecall_present: bool
    bytecode_length: int

    def to_numpy(self) -> np.ndarray:
        """Full feature vector for ML model."""
        extra = np.array(
            [
                self.call_frequency,
                self.sstore_frequency,
                self.sload_frequency,
                float(self.selfdestruct_present),
                float(self.timestamp_present),
                float(self.delegatecall_present),
                float(self.bytecode_length) / 10000,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.opcode_frequencies.astype(np.float32),
                extra,
            ]
        )


class BytecodeFeatureExtractor:
    """Extracts vulnerability-relevant features from EVM bytecode."""

    def extract(self, bytecode_hex: str) -> BytecodeFeatures:
        """
        Extract features from hex-encoded bytecode.
        bytecode_hex: hex string with or without 0x prefix.
        """
        # Clean bytecode
        hex_str = bytecode_hex.replace("0x", "").replace(" ", "")
        if len(hex_str) % 2 != 0:
            hex_str = hex_str[:-1]

        try:
            bytecode = bytes.fromhex(hex_str)
        except ValueError:
            logger.warning("invalid_bytecode_hex")
            return self._empty_features()

        # Count opcode frequencies
        opcode_counts = np.zeros(256, dtype=np.int32)
        i = 0
        while i < len(bytecode):
            opcode = bytecode[i]
            opcode_counts[opcode] += 1
            # PUSH instructions consume N following bytes
            if 0x60 <= opcode <= 0x7F:
                push_size = opcode - 0x60 + 1
                i += push_size + 1
            else:
                i += 1

        total = max(1, len(bytecode))
        opcode_frequencies = opcode_counts / total

        return BytecodeFeatures(
            opcode_frequencies=opcode_frequencies,
            call_frequency=float(opcode_counts[0xF1]) / total,
            sstore_frequency=float(opcode_counts[0x55]) / total,
            sload_frequency=float(opcode_counts[0x54]) / total,
            selfdestruct_present=bool(opcode_counts[0xFF] > 0),
            timestamp_present=bool(opcode_counts[0x42] > 0),
            delegatecall_present=bool(opcode_counts[0xF4] > 0),
            bytecode_length=len(bytecode),
        )

    def _empty_features(self) -> BytecodeFeatures:
        return BytecodeFeatures(
            opcode_frequencies=np.zeros(256, dtype=np.float32),
            call_frequency=0.0,
            sstore_frequency=0.0,
            sload_frequency=0.0,
            selfdestruct_present=False,
            timestamp_present=False,
            delegatecall_present=False,
            bytecode_length=0,
        )


class BytecodeRiskScorer:
    """
    Heuristic risk scorer for bytecode features.
    Used when no trained ML model is available.
    Full XGBoost classifier added when SmartBugs dataset is integrated.
    """

    def score(self, features: BytecodeFeatures) -> dict:
        """
        Score bytecode risk from 0 to 1.
        Returns score and contributing factors.
        """
        risk = 0.0
        factors = []

        if features.selfdestruct_present:
            risk += 0.30
            factors.append("selfdestruct_opcode_present")

        if features.delegatecall_present:
            risk += 0.25
            factors.append("delegatecall_present")

        if features.call_frequency > 0.02:
            risk += 0.20
            factors.append(f"high_call_frequency_{features.call_frequency:.3f}")

        if features.timestamp_present:
            risk += 0.10
            factors.append("timestamp_opcode_present")

        if features.sstore_frequency > 0.05:
            risk += 0.10
            factors.append(f"high_sstore_frequency_{features.sstore_frequency:.3f}")

        if features.bytecode_length > 20000:
            risk += 0.05
            factors.append("large_contract_size")

        risk = min(1.0, risk)

        return {
            "bytecode_risk_score": round(risk, 4),
            "risk_factors": factors,
            "bytecode_length": features.bytecode_length,
            "selfdestruct": features.selfdestruct_present,
            "delegatecall": features.delegatecall_present,
        }
