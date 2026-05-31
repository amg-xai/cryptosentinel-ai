"""Auth routes — token issuance, refresh, and revocation."""
import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings
from src.api.auth import (
    create_access_token,
    create_refresh_token,
    create_dev_token,
    get_current_user,
    TokenPayload,
    _load_public_key,
)
from src.api.token_store import token_store

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

DEMO_USERS = {
    "analyst@cryptosentinel.ai": {"password": "analyst123", "role": "analyst"},
    "investigator@cryptosentinel.ai": {"password": "invest123", "role": "investigator"},
    "admin@cryptosentinel.ai": {"password": "admin123", "role": "admin"},
}


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str = ""
    token_type: str = "bearer"
    role: str
    expires_in: int = 900


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/token", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Issue an access + refresh token pair for valid credentials."""
    user = DEMO_USERS.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access = create_access_token(user_id=request.email, role=user["role"])
    refresh = create_refresh_token(user_id=request.email, role=user["role"])
    logger.info("user_logged_in", email=request.email, role=user["role"])
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        role=user["role"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest):
    """
    Exchange a valid refresh token for a new access token.
    Rejects access tokens, expired tokens, and revoked tokens.
    """
    from jose import jwt, JWTError
    public_key = _load_public_key()
    if not public_key:
        raise HTTPException(status_code=503, detail="Auth service not configured")
    try:
        payload = jwt.decode(
            request.refresh_token, public_key, algorithms=["RS256"],
            options={"verify_exp": True},
        )
    except JWTError as e:
        logger.warning("refresh_token_invalid", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")
    if token_store.is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")

    new_access = create_access_token(
        user_id=payload["sub"], role=payload.get("role", "analyst"),
    )
    logger.info("access_token_refreshed", user_id=payload["sub"])
    return TokenResponse(
        access_token=new_access,
        refresh_token=request.refresh_token,  # caller keeps the same refresh token
        role=payload.get("role", "analyst"),
    )


@router.post("/logout")
async def logout(user: TokenPayload = Depends(get_current_user)):
    """
    Revoke the caller's current token by adding its jti to the denylist.
    The token is rejected on all future requests until it would have expired.
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    token_store.revoke(user.jti, user.exp)
    logger.info("user_logged_out", user_id=user.sub, jti=user.jti[:8])
    return {"detail": "Token revoked", "jti": user.jti[:8]}


@router.get("/dev-token")
async def get_dev_token(role: str = "admin"):
    """
    Get a development token — only works in development environment.
    """
    if settings.environment != "development":
        raise HTTPException(
            status_code=403,
            detail="Dev tokens only available in development",
        )
    token = create_dev_token(role=role)
    return {
        "access_token": token,
        "role": role,
        "usage": f"Add header: Authorization: Bearer {token[:20]}...",
        "note": "24-hour dev token — do not use in production",
    }
