"""
HashiCorp Vault client for secrets management.

WHY Vault over environment variables:
  Environment variables leak via process listings, crash dumps, logs.
  Vault provides encrypted storage, dynamic secrets, audit logging,
  secret rotation, and fine-grained access policies.

Graceful degradation:
  If Vault is unavailable, fall back to environment variables.
"""
import os
from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)

VAULT_ADDR = os.getenv("VAULT_ADDR", "http://localhost:8200")
VAULT_TOKEN = os.getenv("VAULT_TOKEN", "cryptosentinel-dev-token")
SECRET_MOUNT = "secret"
SECRET_PATH = "cryptosentinel"


class VaultClient:
    """Wraps HashiCorp Vault. Falls back to env vars if unavailable."""

    def __init__(self, addr: str = VAULT_ADDR, token: str = VAULT_TOKEN):
        self.addr = addr
        self.token = token
        self._client = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            import hvac
            self._client = hvac.Client(url=self.addr, token=self.token)
            if self._client.is_authenticated():
                self._available = True
                logger.info("vault_connected", addr=self.addr)
            else:
                logger.warning("vault_auth_failed")
        except Exception as e:
            logger.warning("vault_unavailable", error=str(e))
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def write_secret(self, key: str, value: str) -> bool:
        if not self._available:
            return False
        try:
            existing = {}
            try:
                resp = self._client.secrets.kv.v2.read_secret_version(
                    path=SECRET_PATH, mount_point=SECRET_MOUNT,
                )
                existing = resp["data"]["data"]
            except Exception:
                pass
            existing[key] = value
            self._client.secrets.kv.v2.create_or_update_secret(
                path=SECRET_PATH, secret=existing, mount_point=SECRET_MOUNT,
            )
            logger.info("vault_secret_written", key=key)
            return True
        except Exception as e:
            logger.error("vault_write_failed", key=key, error=str(e))
            return False

    def read_secret(self, key: str, fallback_env: Optional[str] = None) -> Optional[str]:
        if self._available:
            try:
                resp = self._client.secrets.kv.v2.read_secret_version(
                    path=SECRET_PATH, mount_point=SECRET_MOUNT,
                )
                value = resp["data"]["data"].get(key)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning("vault_read_failed", key=key, error=str(e))
        env_key = fallback_env or key.upper()
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        return None

    def write_all_secrets(self, secrets: dict) -> bool:
        if not self._available:
            return False
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=SECRET_PATH, secret=secrets, mount_point=SECRET_MOUNT,
            )
            logger.info("vault_secrets_written", count=len(secrets))
            return True
        except Exception as e:
            logger.error("vault_write_all_failed", error=str(e))
            return False

    def list_secrets(self) -> list:
        if not self._available:
            return []
        try:
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=SECRET_PATH, mount_point=SECRET_MOUNT,
            )
            return list(resp["data"]["data"].keys())
        except Exception:
            return []


vault = VaultClient()


def bootstrap_secrets() -> bool:
    if not vault.is_available:
        return False
    secrets_to_store = {
        "jwt_algorithm": "RS256",
        "grafana_password": os.getenv("GRAFANA_PASSWORD", "changeme"),
    }
    eth_http = os.getenv("ETH_HTTP_URL")
    if eth_http:
        secrets_to_store["eth_http_url"] = eth_http
    return vault.write_all_secrets(secrets_to_store)
