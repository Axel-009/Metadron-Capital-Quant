"""
Metadron Capital — Test Configuration & Fixtures

Provides shared fixtures for all test modules:
    - paper_broker: PaperBroker instance
    - test_db: in-memory SQLite session
    - test_client: FastAPI TestClient with auth bypass
    - api_key_header: creates a test API key and returns the header dict
"""

import hashlib
import os
import secrets

import pytest

# Force paper broker for all tests
os.environ["METADRON_BROKER_TYPE"] = "paper"
os.environ["METADRON_ENV"] = "development"

# One temporary SQLite file shared by the test session AND the API's auth middleware
# (which opens its own SessionLocal). Must be set before engine.db is imported.
import tempfile as _tempfile
_TEST_DB_PATH = os.path.join(_tempfile.mkdtemp(prefix="metadron-test-"), "test.db")
os.environ.setdefault("METADRON_DATABASE_URL", f"sqlite:///{_TEST_DB_PATH}")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset shared singletons between tests to ensure isolation."""
    yield
    try:
        from engine.api.shared import _reset_singletons
        _reset_singletons()
    except Exception:
        pass
    # Also clear rate limit store
    try:
        from engine.api.auth import _rate_store
        _rate_store.clear()
    except Exception:
        pass


@pytest.fixture
def paper_broker():
    """Return a fresh PaperBroker instance."""
    from engine.execution.paper_broker import PaperBroker
    return PaperBroker()


@pytest.fixture
def test_db():
    """Create an in-memory SQLite session for testing."""
    from engine.db.database import Base, SessionLocal, engine as db_engine
    from engine.db import tables as _tables  # noqa: F401  — registers the models on Base

    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def test_client(test_db):
    """FastAPI TestClient with in-memory DB and auth middleware active."""
    from fastapi.testclient import TestClient
    from engine.api.server import app
    from engine.db.database import get_db

    def _override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def api_key_header(test_db):
    """Create a test API key in the DB and return the header dict."""
    from engine.db.tables import ApiKey

    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    record = ApiKey(key_hash=key_hash, name="test-key", is_active=True)
    test_db.add(record)
    test_db.commit()
    return {"X-API-Key": raw_key}
