"""Database session management with SQLAlchemy."""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from config.logging_config import get_logger
from src.db.models import Base

logger = get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinel:sentinel_dev_password@localhost:5432/cryptosentinel",
)
SYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

_engine = None
_SessionLocal = None
_available = False


def init_db() -> bool:
    global _engine, _SessionLocal, _available
    try:
        _engine = create_engine(
            SYNC_DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        _available = True
        logger.info("database_initialized")
        return True
    except Exception as e:
        logger.warning("database_unavailable", error=str(e))
        _available = False
        return False


def is_available() -> bool:
    return _available


@contextmanager
def get_session() -> Session:
    if not _available or _SessionLocal is None:
        raise RuntimeError("Database not available")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("db_session_error", error=str(e))
        raise
    finally:
        session.close()
