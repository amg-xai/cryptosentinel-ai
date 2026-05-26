"""Tests for post-quantum cryptography layer."""
import json
import pytest
from src.crypto.pqc_handler import (
    PQCAlertSigner,
    PQCKeyExchange,
    SignedAlert,
    DILITHIUM_ALG,
    KYBER_ALG,
)


def test_signer_initializes():
    """PQCAlertSigner initializes with a valid public key."""
    signer = PQCAlertSigner()
    assert signer.public_key is not None
    assert len(signer.public_key) > 0


def test_sign_alert_returns_signed_alert():
    """sign_alert returns a SignedAlert with all fields."""
    signer = PQCAlertSigner()
    alert = {"address": "0xTest", "score": 0.9, "action": "EMERGENCY"}
    signed = signer.sign_alert(alert)
    assert isinstance(signed, SignedAlert)
    assert signed.pqc_algorithm == DILITHIUM_ALG
    assert len(signed.pqc_signature) > 0
    assert signed.payload_hash != ""


def test_verify_valid_signature():
    """Valid signature must verify as True."""
    signer = PQCAlertSigner()
    alert = {"address": "0xTest", "score": 0.85}
    signed = signer.sign_alert(alert)
    assert signer.verify_alert(signed) is True


def test_verify_tampered_payload():
    """Tampered payload must fail verification."""
    signer = PQCAlertSigner()
    alert = {"address": "0xTest", "score": 0.85}
    signed = signer.sign_alert(alert)
    # Tamper with payload after signing
    signed.payload["score"] = 0.01
    assert signer.verify_alert(signed) is False


def test_signature_size():
    """ML-DSA-65 signature must be ~3309 bytes."""
    signer = PQCAlertSigner()
    signed = signer.sign_alert({"test": True})
    assert 3000 <= len(signed.pqc_signature) <= 3500


def test_signed_alert_to_dict():
    """to_dict returns JSON-serializable dict."""
    signer = PQCAlertSigner()
    signed = signer.sign_alert({"address": "0xTest"})
    d = signed.to_dict()
    assert "pqc_signature" in d
    assert "pqc_algorithm" in d
    assert "payload_hash" in d
    # Must be JSON serializable
    json.dumps(d)


def test_kem_initializes():
    """PQCKeyExchange initializes with a public key."""
    kem = PQCKeyExchange()
    assert kem.public_key is not None
    assert len(kem.public_key) > 0


def test_kem_shared_secrets_match():
    """Encapsulated and decapsulated secrets must be identical."""
    kem = PQCKeyExchange()
    result = kem.encapsulate(kem.public_key)
    recovered = kem.decapsulate(result.ciphertext)
    assert result.shared_secret == recovered


def test_kem_ciphertext_size():
    """ML-KEM-768 ciphertext must be 1088 bytes."""
    kem = PQCKeyExchange()
    result = kem.encapsulate(kem.public_key)
    assert len(result.ciphertext) == 1088


def test_kem_shared_secret_size():
    """Shared secret must be 32 bytes (AES-256 key size)."""
    kem = PQCKeyExchange()
    result = kem.encapsulate(kem.public_key)
    assert len(result.shared_secret) == 32


def test_different_keys_different_secrets():
    """Two different KEM instances produce different shared secrets."""
    kem1 = PQCKeyExchange()
    kem2 = PQCKeyExchange()
    result1 = kem1.encapsulate(kem1.public_key)
    result2 = kem2.encapsulate(kem2.public_key)
    assert result1.shared_secret != result2.shared_secret


def test_sign_complex_alert():
    """Signing works for complex nested alert payloads."""
    signer = PQCAlertSigner()
    alert = {
        "address": "0xComplexAddress",
        "score": 0.95,
        "model_scores": {"gnn": 0.97, "autoencoder": 0.88},
        "action": "EMERGENCY",
        "explanation": {"primary_driver": "gnn", "summary": "Critical threat"},
    }
    signed = signer.sign_alert(alert)
    assert signer.verify_alert(signed) is True
