"""
Metadron Capital — SQLAlchemy database configuration.

Provides the engine, session factory, declarative base, and a FastAPI
dependency for obtaining a database session per request.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("metadron.db")

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError:
    raise RuntimeError(
        "SQLAlchemy is required. Install it with: pip install sqlalchemy"
    )

# ---------------------------------------------------------------------------
# Database path — defaults to <project_root>/data/metadron.db
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/Metadron-Capital
_DATA_DIR = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_DB_URL = os.getenv(
    "METADRON_DATABASE_URL",
    f"sqlite:///{_DATA_DIR / 'metadron.db'}",
)

# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------
try:
    engine = create_engine(
        _DB_URL,
        connect_args={"check_same_thread": False} if "sqlite" in _DB_URL else {},
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created: %s", _DB_URL)
except Exception as exc:
    logger.error("Failed to create database engine: %s", exc)
    engine = None
    SessionLocal = None

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_db():
    """Yield a SQLAlchemy session and ensure it is closed after the request.

    Usage::

        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    if SessionLocal is None:
        raise RuntimeError("Database is not available.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
