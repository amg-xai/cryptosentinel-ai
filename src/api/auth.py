"""
JWT RS256 authentication + RBAC for CryptoSentinel API.

WHY RS256 over HS256:
  HS256 uses the same secret to sign and verify.
  Any service that can verify can also forge tokens.
  RS256 uses private key to sign (only auth service has this)
  and public key to verify (safe to distribute everywhere).
  This is the principle of least privilege applied to tokens.

WHY short access token lifetime (15 min):
  If a token is stolen, it's useless after 15 minutes.
  Refresh tokens (7 days) are stored in HttpOnly cookies —
  invisible to JavaScript, safe from XSS attacks.

RBAC roles:
  analyst:      read alerts, view dashboard, search wallets
  investigator: analyst + acknowledge alerts, export graphs
  admin:        investigator + modify watchlists, approve quarantine
  system:       internal service-to-service calls
"""

import time
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)

# Role hierarchy
ROLES = ["analyst", "investigator", "admin", "system"]
ROLE_PERMISSIONS = {
    "analyst": ["read:alerts", "read:graph", "read:dashboard", "scan:contract"],
    "investigator": [
        "read:alerts",
        "read:graph",
        "read:dashboard",
        "scan:contract",
        "write:acknowledge",
        "export:graph",
    ],
    "admin": [
        "read:alerts",
        "read:graph",
        "read:dashboard",
        "scan:contract",
        "write:acknowledge",
        "export:graph",
        "write:watchlist",
        "write:quarantine",
        "admin:users",
    ],
    "system": ["*"],  # all permissions
}


class TokenPayload(BaseModel):
    sub: str  # user ID
    role: str  # analyst / investigator / admin / system
    jti: str  # JWT ID — used for revocation
    exp: float  # expiry timestamp
    iat: float  # issued at timestamp
    type: str = "access"  # access or refresh


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str = ""
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds


def _load_private_key() -> str:
    try:
        with open(settings.jwt_private_key_path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(
            "jwt_private_key_not_found",
            path=settings.jwt_private_key_path,
        )
        return ""


def _load_public_key() -> str:
    try:
        with open(settings.jwt_public_key_path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(
            "jwt_public_key_not_found",
            path=settings.jwt_public_key_path,
        )
        return ""


def create_access_token(
    user_id: str,
    role: str = "analyst",
    expires_minutes: int = 15,
) -> str:
    """Create a signed RS256 JWT access token."""
    private_key = _load_private_key()
    if not private_key:
        raise HTTPException(
            status_code=500,
            detail="Auth service not configured",
        )

    now = time.time()
    payload = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "exp": now + (expires_minutes * 60),
        "iat": now,
        "type": "access",
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
    )

    logger.info(
        "token_created",
        user_id=user_id,
        role=role,
        expires_in=expires_minutes * 60,
    )
    return token


def create_refresh_token(
    user_id: str,
    role: str = "analyst",
    expires_days: int = 7,
) -> str:
    """Create a long-lived RS256 refresh token (type=refresh)."""
    private_key = _load_private_key()
    if not private_key:
        raise HTTPException(status_code=500, detail="Auth service not configured")
    now = time.time()
    payload = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "exp": now + (expires_days * 86400),
        "iat": now,
        "type": "refresh",
    }
    token = jwt.encode(payload, private_key, algorithm="RS256")
    logger.info("refresh_token_created", user_id=user_id, role=role)
    return token


def create_dev_token(role: str = "admin") -> str:
    """
    Create a development token without expiry check.
    Only works in development environment.
    Used for testing and dashboard access during development.
    """
    return create_access_token(
        user_id="dev_user",
        role=role,
        expires_minutes=60 * 24,  # 24 hours for dev
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> TokenPayload | None:
    """
    Extract and validate JWT from Authorization header.
    Returns None if no token provided (allows optional auth).
    Raises 401 if token is invalid.
    """
    if credentials is None:
        return None

    token = credentials.credentials
    public_key = _load_public_key()

    if not public_key:
        # Auth not configured — allow in development
        if settings.environment == "development":
            return TokenPayload(
                sub="dev_user",
                role="admin",
                jti=str(uuid.uuid4()),
                exp=time.time() + 3600,
                iat=time.time(),
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service not configured",
        )

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_exp": True},
        )
        token_data = TokenPayload(**payload)
        # Reject revoked tokens (logout / compromise) even before expiry
        from src.api.token_store import token_store
        if token_store.is_revoked(token_data.jti):
            logger.warning("revoked_token_rejected", jti=token_data.jti[:8])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token_data

    except JWTError as e:
        logger.warning("token_validation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_auth(
    user: TokenPayload | None = Depends(get_current_user),
) -> TokenPayload:
    """Require authentication — raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*roles: str):
    """
    Dependency factory — require specific role(s).
    Usage: Depends(require_role("analyst", "investigator"))
    """

    async def dependency(
        user: TokenPayload = Depends(require_auth),
    ) -> TokenPayload:
        if user.role not in roles and user.role != "admin":
            logger.warning(
                "access_denied",
                user_id=user.sub,
                user_role=user.role,
                required_roles=roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' cannot access this endpoint. "
                f"Required: {roles}",
            )
        return user

    return dependency
