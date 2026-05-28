"""Auth routes — token issuance."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings
from src.api.auth import create_access_token, create_dev_token

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
    token_type: str = "bearer"
    role: str
    expires_in: int = 900


@router.post("/token", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Issue JWT access token for valid credentials."""
    user = DEMO_USERS.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        user_id=request.email,
        role=user["role"],
    )
    logger.info("user_logged_in", email=request.email, role=user["role"])
    return TokenResponse(
        access_token=token,
        role=user["role"],
    )


@router.get("/dev-token")
async def get_dev_token(role: str = "admin"):
    """
    Get a development token — only works in development environment.
    Use this for testing endpoints without a full login flow.
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
