"""
CryptoSentinel AI — FastAPI application.
All routes registered here. Models loaded at startup.
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from config.logging_config import get_logger
from config.settings import settings
from src.monitoring.tracing import setup_tracing
from src.api.model_registry import registry
from src.api.routes import analysis, alerts, graph, scanner

logger = get_logger(__name__)
setup_tracing("cryptosentinel-api")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("api_starting")
    model_status = registry.load_all()
    logger.info("models_loaded", status=model_status)
    yield
    logger.info("api_stopping")


app = FastAPI(
    title="CryptoSentinel AI",
    description="Quantum-resistant blockchain threat intelligence platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analysis.router)
app.include_router(alerts.router)
app.include_router(graph.router)
app.include_router(scanner.router)


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
    }
