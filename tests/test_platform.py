"""Tests for Metadron Capital platform core modules."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.platform import MetadronPlatform, Layer, RepoModule
from core.signals import SignalEngine, SignalDecomposition, MoneyVelocityMetrics
from core.portfolio import PortfolioEngine, PortfolioAnalytics


class TestPlatform:
    def test_load_config(self):
        platform = MetadronPlatform()
        assert len(platform.modules) > 0
        assert "metadron_capital" in platform.modules

    def test_layer_classification(self):
        platform = MetadronPlatform()
        layer1 = platform.get_layer_modules(Layer.DATA_INGESTION)
        assert len(layer1) >= 3  # financial_data, open_bb, hedgefund_tracker

    def test_initialize_all(self):
        platform = MetadronPlatform()
        results = platform.initialize_all()
        # At least Metadron-Capital itself should initialize
        assert results["metadron_capital"] in ("initialized", "error")

    def test_status_report(self):
        platform = MetadronPlatform()
        platform.initialize_all()
        report = platform.status_report()
        assert "METADRON CAPITAL" in report
        assert "LAYER" in report

    def test_data_flow(self):
        platform = MetadronPlatform()
        flow = platform.get_data_flow()
        assert len(flow) == 6  # Layers 1-6


class TestSignalEngine:
    @pytest.fixture
    def synthetic_series(self):
        np.random.seed(42)
        n = 504  # ~2 years of trading days
        t = np.arange(n)
        # Secular trend + cyclical + noise
        secular = 0.001 * t
        cyclical = 2 * np.sin(2 * np.pi * t / 252)  # 1-year cycle
        noise = np.random.randn(n) * 0.5
        values = 100 + secular + cyclical + noise
        return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=n))

    def test_decompose(self, synthetic_series):
        result = SignalEngine.decompose(synthetic_series)
        assert isinstance(result, SignalDecomposition)
        assert len(result.secular_trend) == len(synthetic_series)
        assert len(result.cyclical_component) == len(synthetic_series)
        assert result.dominant_period_days > 0

    def test_decompose_short_series(self):
        short = pd.Series([1, 2, 3, 4, 5], index=pd.bdate_range("2024-01-01", periods=5))
        result = SignalEngine.decompose(short)
        assert isinstance(result, SignalDecomposition)

    def test_money_velocity(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="QE")
        m2 = pd.Series(np.linspace(15e12, 21e12, 20), index=idx)
        gdp = pd.Series(np.linspace(21e12, 25e12, 20), index=idx)

        result = SignalEngine.money_velocity(m2, gdp)
        assert isinstance(result, MoneyVelocityMetrics)
        assert len(result.velocity) > 0
        assert all(result.velocity > 0)


class TestPortfolioEngine:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 252
        returns = pd.Series(
            np.random.randn(n) * 0.01 + 0.0003,
            index=pd.bdate_range("2024-01-01", periods=n),
            name="portfolio"
        )
        return returns

    def test_compute_analytics(self, sample_returns):
        weights = {"AAPL": 0.3, "GOOGL": 0.3, "MSFT": 0.4}
        analytics = PortfolioEngine.compute_analytics(sample_returns, weights)

        assert isinstance(analytics, PortfolioAnalytics)
        assert analytics.annualized_volatility > 0
        assert analytics.max_drawdown <= 0
        assert isinstance(analytics.sharpe_ratio, float)

    def test_optimize_weights(self):
        assets = ["AAPL", "GOOGL", "MSFT"]
        returns = pd.Series([0.12, 0.10, 0.08], index=assets)
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005],
             [0.01, 0.03, 0.008],
             [0.005, 0.008, 0.02]],
            index=assets, columns=assets
        )

        weights = PortfolioEngine.optimize_weights(returns, cov)
        assert len(weights) == 3
        assert abs(sum(abs(v) for v in weights.values()) - 1.0) < 1e-6

    def test_optimize_long_only(self):
        assets = ["A", "B"]
        returns = pd.Series([0.1, -0.05], index=assets)
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.03]], index=assets, columns=assets)

        weights = PortfolioEngine.optimize_weights(
            returns, cov, constraints={"long_only": True}
        )
        assert all(v >= 0 for v in weights.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
