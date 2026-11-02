"""
Metadron Capital — Portfolio Analytics Engine
Generates comprehensive portfolio analytics beyond human neural assessment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PortfolioAnalytics:
    """Comprehensive portfolio analytics report."""
    # Returns
    total_return: float
    annualized_return: float
    cumulative_returns: pd.Series

    # Risk
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    drawdown_series: pd.Series
    var_95: float
    cvar_95: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float]

    # Regime awareness
    cyclical_beta: float        # Beta to cyclical factors
    secular_beta: float         # Beta to secular trends
    velocity_sensitivity: float # Sensitivity to money velocity changes

    # Allocation
    weights: dict[str, float]
    sector_exposure: dict[str, float]
    factor_exposure: dict[str, float]


class PortfolioEngine:
    """
    Portfolio analytics engine that integrates signals from all layers
    to generate investment intelligence surpassing human assessment.
    """

    @staticmethod
    def compute_analytics(
        returns: pd.Series,
        weights: dict[str, float],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> PortfolioAnalytics:
        """Compute full portfolio analytics suite."""

        returns = returns.dropna()
        n = len(returns)

        if n == 0:
            raise ValueError("No return data provided")

        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
        years = n / trading_days
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

        # Volatility
        vol = returns.std()
        ann_vol = vol * np.sqrt(trading_days)

        # Drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        # Sharpe
        excess_daily = returns.mean() - risk_free_rate / trading_days
        sharpe = (excess_daily / vol * np.sqrt(trading_days)) if vol > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else ann_vol
        sortino = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

        # Calmar
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Information ratio
        info_ratio = None
        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(returns.index).dropna()
            if len(bench) > 0:
                active = returns.loc[bench.index] - bench
                te = active.std() * np.sqrt(trading_days)
                info_ratio = (active.mean() * trading_days) / te if te > 0 else 0.0

        return PortfolioAnalytics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_returns=cumulative,
            volatility=vol,
            annualized_volatility=ann_vol,
            max_drawdown=max_dd,
            drawdown_series=drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            cyclical_beta=0.0,   # Populated by signal engine integration
            secular_beta=0.0,
            velocity_sensitivity=0.0,
            weights=weights,
            sector_exposure={},
            factor_exposure={},
        )

    @staticmethod
    def optimize_weights(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 1.0,
        constraints: Optional[dict] = None,
    ) -> dict[str, float]:
        """
        Mean-variance optimization with optional constraints.
        Uses analytical solution for unconstrained case.
        """
        n = len(expected_returns)
        if n == 0:
            return {}

        mu = expected_returns.values
        sigma = cov_matrix.values

        # Analytical solution: w = (1/gamma) * Sigma^{-1} * mu
        try:
            sigma_inv = np.linalg.inv(sigma)
            raw_weights = (1.0 / risk_aversion) * sigma_inv @ mu

            # Normalize to sum to 1 (long-only if all positive)
            if constraints and constraints.get("long_only", False):
                raw_weights = np.maximum(raw_weights, 0)

            weight_sum = np.sum(np.abs(raw_weights))
            if weight_sum > 0:
                weights = raw_weights / weight_sum
            else:
                weights = np.ones(n) / n

            return dict(zip(expected_returns.index, weights))

        except np.linalg.LinAlgError:
            # Fallback to equal weight
            return dict(zip(expected_returns.index, np.ones(n) / n))
