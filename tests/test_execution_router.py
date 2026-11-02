"""Tests for execution router endpoints."""

import pytest


class TestExecutionRouter:
    """Verify execution router endpoint shapes."""

    def test_pipeline_status_shape(self, test_client, api_key_header):
        """GET /api/engine/execution/pipeline-status should return expected fields."""
        resp = test_client.get("/api/engine/execution/pipeline-status", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data or "error" in data

    def test_reconciliation_shape(self, test_client, api_key_header):
        """GET /api/engine/execution/reconciliation should return expected fields."""
        resp = test_client.get("/api/engine/execution/reconciliation", headers=api_key_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data or "error" in data
