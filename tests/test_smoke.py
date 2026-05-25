"""
Smoke tests — these must always pass.
They verify the project structure and config load correctly.
"""

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def test_settings_loads():
    """Settings must load without errors."""
    assert settings is not None
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_logger_works():
    """Logger must be importable and callable."""
    logger.info("smoke_test_passed", test="test_logger_works")
    assert True


def test_model_artifacts_path_configured():
    """Model path must be set."""
    assert settings.model_artifacts_path == "data/models"
