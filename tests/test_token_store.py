"""Token revocation store tests (graceful fallback)."""
import time
from src.api.token_store import TokenStore


def test_revoke_and_check():
    store = TokenStore()
    store._redis = None  # force in-memory path for determinism
    jti = "test-jti-123"
    assert store.is_revoked(jti) is False
    store.revoke(jti, time.time() + 60)
    assert store.is_revoked(jti) is True


def test_expired_revocation_self_cleans():
    store = TokenStore()
    store._redis = None
    jti = "test-jti-expired"
    store.revoke(jti, time.time() - 1)  # already expired
    assert store.is_revoked(jti) is False
