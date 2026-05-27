"""
Sliding window rate limiter using Redis.

WHY sliding window over fixed window:
  Fixed window allows 2x the rate at window boundaries.
  Example: 100 req/min limit, send 100 at 11:59 and 100 at 12:00
  = 200 requests in 2 seconds. Sliding window prevents this.

Limits by endpoint sensitivity:
  GET /alerts → 100/min (read, cheap)
  POST /analyze/wallet → 30/min (ML scoring)
  POST /analyze/transaction → 30/min (ML scoring)
  POST /scan/contract → 10/min (expensive rule engine)
  POST /auth/token → 10/min (prevent brute force)
"""
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.logging_config import get_logger

logger = get_logger(__name__)

# Endpoint rate limits: path_prefix -> (limit, window_seconds)
RATE_LIMITS = {
    "/auth/token": (10, 60),
    "/scan/contract": (10, 60),
    "/analyze/wallet": (30, 60),
    "/analyze/transaction": (30, 60),
    "/alerts": (100, 60),
    "/graph": (50, 60),
}

DEFAULT_LIMIT = (200, 60)

# In-memory sliding window store
# {key: [(timestamp, count), ...]}
_windows: dict[str, list[float]] = {}


def _get_limit(path: str) -> tuple[int, int]:
    for prefix, limit in RATE_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return DEFAULT_LIMIT


def _check_rate_limit(key: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Sliding window check.
    Returns (allowed, remaining).
    """
    now = time.time()
    window_start = now - window

    if key not in _windows:
        _windows[key] = []

    # Remove expired entries
    _windows[key] = [t for t in _windows[key] if t > window_start]

    count = len(_windows[key])
    remaining = max(0, limit - count)

    if count >= limit:
        return False, 0

    _windows[key].append(now)
    return True, remaining - 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiting middleware."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health/metrics
        if request.url.path in ("/health", "/metrics", "/", "/docs", "/redoc"):
            return await call_next(request)

        # Key by IP for auth endpoints, by IP+path for others
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        key = f"{client_ip}:{path}"

        limit, window = _get_limit(path)
        allowed, remaining = _check_rate_limit(key, limit, window)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                ip=client_ip,
                path=path,
                limit=limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit: {limit} requests per {window}s",
                    "retry_after": window,
                },
                headers={
                    "RateLimit-Limit": str(limit),
                    "RateLimit-Remaining": "0",
                    "RateLimit-Reset": str(int(time.time()) + window),
                    "Retry-After": str(window),
                },
            )

        response = await call_next(request)
        response.headers["RateLimit-Limit"] = str(limit)
        response.headers["RateLimit-Remaining"] = str(remaining)
        return response
