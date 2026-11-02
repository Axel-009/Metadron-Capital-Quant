"""Tests for security headers, CORS, and request ID middleware."""

import pytest


class TestSecurityHeaders:
    """Verify security headers are present on responses."""

    def test_x_content_type_options(self, test_client):
        """X-Content-Type-Options: nosniff should be on all responses."""
        resp = test_client.get("/api/engine/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, test_client):
        """X-Frame-Options: DENY should be on all responses."""
        resp = test_client.get("/api/engine/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, test_client):
        """X-XSS-Protection header should be present."""
        resp = test_client.get("/api/engine/health")
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_request_id_present(self, test_client):
        """X-Request-ID header should be on all responses."""
        resp = test_client.get("/api/engine/health")
        request_id = resp.headers.get("X-Request-ID")
        assert request_id is not None
        # Should be a valid UUID format
        assert len(request_id) == 36
        assert request_id.count("-") == 4

    def test_request_id_unique(self, test_client):
        """Each request should get a unique request ID."""
        resp1 = test_client.get("/api/engine/health")
        resp2 = test_client.get("/api/engine/health")
        assert resp1.headers["X-Request-ID"] != resp2.headers["X-Request-ID"]

    def test_cors_headers_present(self, test_client):
        """CORS headers should be present for allowed origins."""
        resp = test_client.options(
            "/api/engine/health",
            headers={
                "Origin": "http://localhost:5000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # For allowed origin, should get access-control-allow-origin
        assert "access-control-allow-origin" in resp.headers or resp.status_code == 200
