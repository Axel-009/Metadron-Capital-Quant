"""Tests for risk router endpoints."""

import pytest


class TestRiskRouter:
    """Verify risk router endpoint shapes and singleton consistency."""

    def test_risk_portfolio_shape(self, test_client, api_key_header):
        """GET /api/engine/risk/portfolio should return expected fields."""
        resp = test_client.get("/api/engine/risk/portfolio", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "current_beta" in data or "error" in data

    def test_risk_greeks_shape(self, test_client, api_key_header):
        """GET /api/engine/risk/greeks should return expected fields."""
        resp = test_client.get("/api/engine/risk/greeks", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data

    def test_risk_and_portfolio_share_broker(self):
        """Risk and portfolio routers should share the same broker instance."""
        from engine.api.shared import get_broker, get_engine
        broker = get_broker()
        engine = get_engine()
        assert broker is engine.broker
