"""Validate Traefik gateway configs are well-formed and contain the
circuit breaker + health check the response-engine design requires."""
from pathlib import Path
import yaml
import pytest

TRAEFIK_DIR = Path("docker/traefik")
STATIC = TRAEFIK_DIR / "traefik.yml"
DYNAMIC = TRAEFIK_DIR / "dynamic.yml"

pytestmark = pytest.mark.skipif(
    not STATIC.exists() or not DYNAMIC.exists(),
    reason="traefik configs not present",
)


def test_static_config_valid_yaml():
    cfg = yaml.safe_load(STATIC.read_text())
    assert "entryPoints" in cfg
    assert "web" in cfg["entryPoints"]


def test_dynamic_config_valid_yaml():
    cfg = yaml.safe_load(DYNAMIC.read_text())
    assert "http" in cfg


def test_circuit_breaker_present():
    cfg = yaml.safe_load(DYNAMIC.read_text())
    mws = cfg["http"]["middlewares"]
    cb = next((m for m in mws.values() if "circuitBreaker" in m), None)
    assert cb is not None, "no circuit breaker middleware defined"
    assert "expression" in cb["circuitBreaker"]


def test_api_service_has_health_check():
    cfg = yaml.safe_load(DYNAMIC.read_text())
    svc = cfg["http"]["services"]["cryptosentinel-api"]["loadBalancer"]
    assert "healthCheck" in svc
    assert svc["healthCheck"]["path"] == "/health"


def test_router_uses_circuit_breaker():
    cfg = yaml.safe_load(DYNAMIC.read_text())
    router = cfg["http"]["routers"]["api-router"]
    assert "api-circuit-breaker" in router["middlewares"]
