"""
Token revocation store — tracks revoked JWT IDs (jti) so a token can be
invalidated before its natural expiry (logout, compromise, rotation).

Design: Redis-backed when reachable, with a graceful in-memory fallback
(same pattern as the Vault client). Revoked jtis are stored with a TTL
equal to the token's remaining lifetime, so the denylist self-cleans and
never grows unbounded — once a token would have expired anyway, there's no
need to keep denying it.
"""
import time
import threading

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_REVOKE_PREFIX = "revoked_jti:"


class TokenStore:
    def __init__(self):
        self._redis = None
        self._mem: dict[str, float] = {}  # jti -> expiry epoch (fallback)
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            import redis
            client = redis.Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            logger.info("token_store_redis_connected")
        except Exception as e:
            self._redis = None
            logger.warning("token_store_redis_unavailable_using_memory",
                           error=str(e))

    def revoke(self, jti: str, expires_at: float) -> None:
        """Revoke a jti until its original expiry (TTL self-cleans)."""
        ttl = max(1, int(expires_at - time.time()))
        if self._redis is not None:
            try:
                self._redis.setex(_REVOKE_PREFIX + jti, ttl, "1")
                logger.info("token_revoked", jti=jti[:8], backend="redis")
                return
            except Exception as e:
                logger.warning("token_store_redis_revoke_failed",
                               error=str(e))
        with self._lock:
            self._mem[jti] = expires_at
        logger.info("token_revoked", jti=jti[:8], backend="memory")

    def is_revoked(self, jti: str) -> bool:
        if self._redis is not None:
            try:
                return self._redis.exists(_REVOKE_PREFIX + jti) == 1
            except Exception as e:
                logger.warning("token_store_redis_check_failed",
                               error=str(e))
        # In-memory fallback (lazily evict expired entries)
        with self._lock:
            exp = self._mem.get(jti)
            if exp is None:
                return False
            if exp < time.time():
                self._mem.pop(jti, None)
                return False
            return True


# Module-level singleton
token_store = TokenStore()
