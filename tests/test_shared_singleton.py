"""Tests for engine.api.shared singleton getters."""

import os
import pytest

os.environ["METADRON_BROKER_TYPE"] = "paper"


class TestSharedSingletons:
    """Verify that shared getters return consistent, reusable instances."""

    def test_get_engine_returns_same_instance(self):
        from engine.api.shared import get_engine
        eng1 = get_engine()
        eng2 = get_engine()
        assert eng1 is eng2

    def test_get_broker_returns_engine_broker(self):
        from engine.api.shared import get_engine, get_broker
        eng = get_engine()
        broker = get_broker()
        assert broker is eng.broker

    def test_get_beta_returns_engine_beta(self):
        from engine.api.shared import get_engine, get_beta
        eng = get_engine()
        beta = get_beta()
        # Beta should be reused from engine if available
        if hasattr(eng, "beta") and eng.beta is not None:
            assert beta is eng.beta

    def test_reset_singletons(self):
        from engine.api.shared import get_engine, _reset_singletons
        eng1 = get_engine()
        _reset_singletons()
        eng2 = get_engine()
        assert eng1 is not eng2

    def test_paper_broker_fallback(self):
        """With METADRON_BROKER_TYPE=paper, broker should be PaperBroker."""
        os.environ["METADRON_BROKER_TYPE"] = "paper"
        from engine.api.shared import _reset_singletons, get_broker
        _reset_singletons()
        broker = get_broker()
        assert type(broker).__name__ == "PaperBroker"
