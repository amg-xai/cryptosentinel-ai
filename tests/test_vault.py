"""Tests for HashiCorp Vault integration."""
import os
import pytest
from src.security.vault_client import VaultClient


def test_vault_client_initializes():
    """VaultClient initializes (may or may not connect)."""
    client = VaultClient()
    assert client is not None
    assert isinstance(client.is_available, bool)


def test_vault_read_fallback_to_env():
    """When key not in Vault, falls back to environment variable."""
    os.environ["TEST_FALLBACK_SECRET"] = "from_env"
    client = VaultClient()
    value = client.read_secret("nonexistent_key", fallback_env="TEST_FALLBACK_SECRET")
    assert value == "from_env"
    del os.environ["TEST_FALLBACK_SECRET"]


def test_vault_read_missing_returns_none():
    """Missing key with no fallback returns None."""
    client = VaultClient()
    value = client.read_secret("definitely_does_not_exist_12345")
    assert value is None


def test_vault_unavailable_graceful_degradation():
    """Client with bad address degrades gracefully."""
    client = VaultClient(addr="http://localhost:9999", token="bad")
    assert client.is_available is False
    # Should not crash, returns None or env fallback
    value = client.read_secret("any_key")
    assert value is None


def test_vault_write_when_available():
    """Writing a secret works when Vault is available."""
    client = VaultClient()
    if client.is_available:
        result = client.write_secret("pytest_key", "pytest_value")
        assert result is True
        # Read it back
        value = client.read_secret("pytest_key")
        assert value == "pytest_value"
    else:
        pytest.skip("Vault not available")


def test_vault_list_secrets():
    """Listing secrets returns a list."""
    client = VaultClient()
    secrets = client.list_secrets()
    assert isinstance(secrets, list)
