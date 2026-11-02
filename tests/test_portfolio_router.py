"""Tests for portfolio router endpoints."""

import pytest


class TestPortfolioRouter:
    """Verify portfolio router endpoint shapes."""

    def test_portfolio_live_shape(self, test_client, api_key_header):
        """GET /api/engine/portfolio/live should return expected fields."""
        resp = test_client.get("/api/engine/portfolio/live", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        # Should have nav or error
        assert "nav" in data or "error" in data

    def test_portfolio_positions_returns_list(self, test_client, api_key_header):
        """GET /api/engine/portfolio/positions should return a positions list."""
        resp = test_client.get("/api/engine/portfolio/positions", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)

    def test_portfolio_trades_returns_list(self, test_client, api_key_header):
        """GET /api/engine/portfolio/trades should return a trades list."""
        resp = test_client.get("/api/engine/portfolio/trades", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "trades" in data
        assert isinstance(data["trades"], list)
