"""
CryptoSentinel AI — FastAPI application.

Endpoints added progressively:
  Day 5:  GET /health, GET /metrics
  Week 3: GET /analyze/wallet, POST /analyze/transaction
  Week 4: GET /graph/{address}, POST /alerts/{id}/acknowledge
  Week 6: POST /scan/contract
  Week 7: GET /alerts, POST /auth/token
"""

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from config.logging_config import get_logger
from config.settings import settings
from src.monitoring.tracing import setup_tracing

logger = get_logger(__name__)

# Initialize tracing before anything else
setup_tracing("cryptosentinel-api")

app = FastAPI(
    title="CryptoSentinel AI",
    description="Quantum-resistant blockchain threat intelligence platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time for uptime reporting
_start_time = time.time()


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and uptime.
    Used by Docker health checks and K8s liveness probes.
    """
    uptime_seconds = int(time.time() - _start_time)
    return {
        "status": "healthy",
        "service": "cryptosentinel-api",
        "version": "0.1.0",
        "environment": settings.environment,
        "uptime_seconds": uptime_seconds,
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Scraped every 15 seconds by Prometheus.
    """
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


@app.on_event("startup")
async def startup_event():
    logger.info(
        "api_started",
        environment=settings.environment,
        docs_url="http://localhost:8000/docs",
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("api_stopping")
