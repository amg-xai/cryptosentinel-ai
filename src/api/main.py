"""
CryptoSentinel AI — FastAPI application.
All routes registered here. Models loaded at startup.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from config.logging_config import get_logger
from config.settings import settings
from src.api.model_registry import registry
from src.api.routes import alerts, analysis, graph, scanner
from src.api.routes.auth import router as auth_router
from src.api.routes.explain import router as explain_router
from src.monitoring.tracing import setup_tracing

logger = get_logger(__name__)
setup_tracing("cryptosentinel-api")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_starting")
    model_status = registry.load_all()
    logger.info("models_loaded", status=model_status)

    # Initialize database for alert persistence
    from src.db.session import init_db
    db_ok = init_db()
    logger.info("database_init", available=db_ok)

    yield
    logger.info("api_stopping")


app = FastAPI(
    title="CryptoSentinel AI",
    description=(
        "Quantum-resistant blockchain threat intelligence platform.\n\n"
        "## Authentication\n"
        "Most endpoints require a Bearer JWT token.\n"
        "Get a dev token at `GET /auth/dev-token` (development only).\n\n"
        "Demo credentials:\n"
        "- analyst@cryptosentinel.ai / analyst123\n"
        "- admin@cryptosentinel.ai / admin123"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

from src.api.middleware.rate_limiter import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(auth_router)
app.include_router(analysis.router)
app.include_router(alerts.router)
app.include_router(graph.router)
app.include_router(scanner.router)
app.include_router(explain_router)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "cryptosentinel-api",
        "version": "0.1.0",
        "environment": settings.environment,
        "uptime_seconds": int(time.time() - _start_time),
        "models_loaded": {
            "isolation_forest": registry.isolation_forest is not None,
            "autoencoder": registry.autoencoder is not None,
            "gnn": registry.gnn_trainer is not None,
        },
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "CryptoSentinel AI",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "auth": "/auth/token",
        "dev_token": "/auth/dev-token (development only)",
    }
