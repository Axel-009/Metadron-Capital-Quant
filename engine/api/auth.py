"""
Metadron Capital — API Key Authentication & Rate Limiting

Provides:
    - verify_api_key(): validates X-API-Key header against hashed keys in DB
    - require_api_key: FastAPI Depends() injectable
    - AuthMiddleware: ASGI middleware that enforces auth on all non-exempt paths
    - Rate limiting: sliding window 100 req/min per key
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("metadron-api.auth")

# ---------------------------------------------------------------------------
# Rate limiting — in-memory sliding window
# ---------------------------------------------------------------------------
_RATE_LIMIT = int(os.getenv("METADRON_RATE_LIMIT", "100"))  # requests per minute
_rate_store: dict[str, list[float]] = defaultdict(list)


_STALE_THRESHOLD = 72 * 3600  # 72 hours — delete keys inactive longer than this
_last_eviction: float = 0.0


def _check_rate_limit(key_hash: str) -> bool:
    """Return True if the request is within rate limits, False if exceeded."""
    global _last_eviction
    now = time.time()
    window_start = now - 60.0

    # Evict keys inactive for 72+ hours (runs at most once per hour)
    if now - _last_eviction > 3600:
        stale = [k for k, ts in _rate_store.items() if not ts or ts[-1] < now - _STALE_THRESHOLD]
        for k in stale:
            del _rate_store[k]
        _last_eviction = now

    # Prune old entries for this key
    timestamps = _rate_store[key_hash]
    _rate_store[key_hash] = [ts for ts in timestamps if ts > window_start]
    if len(_rate_store[key_hash]) >= _RATE_LIMIT:
        return False
    _rate_store[key_hash].append(now)
    return True


# ---------------------------------------------------------------------------
# Paths exempt from authentication
# ---------------------------------------------------------------------------
EXEMPT_PATHS = {
    "/api/engine/health",
    "/api/engine/docs",
    "/api/engine/openapi.json",
    "/api/engine/redoc",
}

# Prefix-based exemptions (bootstrap: POST to api-keys when no keys exist)
EXEMPT_PREFIXES = [
    "/api/engine/docs",
]


def _is_exempt(path: str, method: str) -> bool:
    """Check if a request path is exempt from authentication."""
    if path in EXEMPT_PATHS:
        return True
    for prefix in EXEMPT_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _is_bootstrap_request(path: str, method: str) -> bool:
    """Check if this is a bootstrap request to create the first API key."""
    return path.rstrip("/") == "/api/engine/api-keys" and method.upper() == "POST"


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------
def _hash_key(raw: str) -> str:
    """SHA-256 hash of a raw API key."""
    return hashlib.sha256(raw.encode()).hexdigest()


def _check_internal_token(request: Request) -> bool:
    """Check if request has a valid internal proxy token from the API Vault.

    The Express frontend injects X-Internal-Token on proxied requests.
    This bypasses API key auth for same-host proxy traffic.
    """
    token = request.headers.get("X-Internal-Token")
    if not token:
        return False
    try:
        from engine.api.vault import get_vault
        return get_vault().verify_internal_token(token)
    except Exception:
        return False


def _extract_key(request: Request) -> Optional[str]:
    """Extract API key from X-API-Key header or Authorization: Bearer."""
    key = request.headers.get("X-API-Key")
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def verify_api_key(request: Request) -> dict:
    """Validate the API key from the request.

    Returns a dict with key metadata on success.
    Raises HTTPException(401) on failure, HTTPException(429) on rate limit.
    """
    raw_key = _extract_key(request)
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = _hash_key(raw_key)

    # Check rate limit
    if not _check_rate_limit(key_hash):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (100 req/min)")

    # Validate against DB
    try:
        from engine.db.database import SessionLocal
        from engine.db.tables import ApiKey

        if SessionLocal is None:
            raise HTTPException(status_code=500, detail="Database unavailable")

        db = SessionLocal()
        try:
            record = (
                db.query(ApiKey)
                .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
                .first()
            )
            if record is None:
                raise HTTPException(status_code=401, detail="Invalid or revoked API key")
            return {"key_id": record.id, "key_name": record.name}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication service error")


# ---------------------------------------------------------------------------
# FastAPI Depends() injectable
# ---------------------------------------------------------------------------
async def require_api_key(request: Request) -> dict:
    """FastAPI dependency that enforces API key authentication."""
    return verify_api_key(request)


# ---------------------------------------------------------------------------
# ASGI Middleware — enforces auth on all non-exempt paths
# ---------------------------------------------------------------------------
class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key authentication on all requests.

    Exempt paths (health, docs, openapi) are allowed through without auth.
    Bootstrap: POST /api/engine/api-keys is allowed when no keys exist.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Allow exempt paths
        if _is_exempt(path, method):
            return await call_next(request)

        # Perimeter circuit breaker — lockdown blocks external requests
        try:
            from engine.security.integrity import get_security
            sec = get_security()
            if sec and sec.circuit_breaker.is_locked:
                if path not in EXEMPT_PATHS:
                    return JSONResponse(
                        status_code=503,
                        content={"detail": "System in lockdown mode — try again later"},
                    )
            # Record request for rate detection
            if sec:
                sec.circuit_breaker.record_request()
        except Exception:
            pass

        # Allow internal proxy token (frontend → backend same-host traffic)
        if _check_internal_token(request):
            request.state.api_key_id = 0
            request.state.api_key_name = "internal-proxy"
            return await call_next(request)

        # Bootstrap: allow creating first API key when DB has no keys
        if _is_bootstrap_request(path, method):
            try:
                from engine.db.database import SessionLocal
                from engine.db.tables import ApiKey

                if SessionLocal is not None:
                    db = SessionLocal()
                    try:
                        count = db.query(ApiKey).filter(ApiKey.is_active == True).count()
                        if count == 0:
                            logger.info("Bootstrap: allowing first API key creation without auth")
                            return await call_next(request)
                    finally:
                        db.close()
            except Exception as exc:
                logger.warning("Bootstrap check failed: %s", exc)

        # Extract and verify key
        raw_key = _extract_key(request)
        if not raw_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
            )

        key_hash = _hash_key(raw_key)

        # Rate limit
        if not _check_rate_limit(key_hash):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded (100 req/min)"},
            )

        # Validate key
        try:
            from engine.db.database import SessionLocal
            from engine.db.tables import ApiKey

            if SessionLocal is None:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Database unavailable"},
                )

            db = SessionLocal()
            try:
                record = (
                    db.query(ApiKey)
                    .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
                    .first()
                )
                if record is None:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or revoked API key"},
                    )
                # Attach key info to request state for downstream use
                request.state.api_key_id = record.id
                request.state.api_key_name = record.name
            finally:
                db.close()
        except Exception as exc:
            logger.error("Auth middleware DB error: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication service error"},
            )

        return await call_next(request)
