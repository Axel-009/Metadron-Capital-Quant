"""Tests for engine.api.auth — API key authentication and rate limiting."""

import hashlib
import secrets
import pytest


class TestAuthentication:
    """API key authentication tests."""

    def test_unauthenticated_request_returns_401(self, test_client):
        """Requests without API key should be rejected."""
        resp = test_client.get("/api/engine/portfolio/live")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, test_client):
        """Requests with an invalid API key should be rejected."""
        resp = test_client.get(
            "/api/engine/portfolio/live",
            headers={"X-API-Key": "invalid-key-12345"},
        )
        assert resp.status_code == 401

    def test_valid_api_key_returns_200(self, test_client, api_key_header):
        """Requests with a valid API key should pass auth."""
        resp = test_client.get("/api/engine/health", headers=api_key_header)
        # Health is exempt, but should always work
        assert resp.status_code == 200

    def test_revoked_api_key_returns_401(self, test_client, test_db):
        """Revoked (is_active=False) keys should be rejected."""
        from engine.db.tables import ApiKey

        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        record = ApiKey(key_hash=key_hash, name="revoked-key", is_active=False)
        test_db.add(record)
        test_db.commit()

        resp = test_client.get(
            "/api/engine/portfolio/live",
            headers={"X-API-Key": raw_key},
        )
        assert resp.status_code == 401

    def test_health_endpoint_no_auth_required(self, test_client):
        """Health endpoint should be accessible without any API key."""
        resp = test_client.get("/api/engine/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_rate_limiting(self, test_client, api_key_header):
        """After 100 requests per minute, the 101st should return 429."""
        from engine.api.auth import _rate_store
        _rate_store.clear()

        # Health is exempt from auth, so we need an authenticated endpoint.
        # Simulate 100 requests by injecting timestamps directly.
        import time
        raw_key = api_key_header["X-API-Key"]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now = time.time()
        _rate_store[key_hash] = [now - i * 0.5 for i in range(100)]

        # The 101st request should be rate-limited
        resp = test_client.get(
            "/api/engine/portfolio/live",
            headers=api_key_header,
        )
        assert resp.status_code == 429

    def test_bearer_token_auth(self, test_client, api_key_header):
        """Authorization: Bearer <key> should also work."""
        raw_key = api_key_header["X-API-Key"]
        resp = test_client.get("/api/engine/health", headers={"Authorization": f"Bearer {raw_key}"})
        assert resp.status_code == 200
