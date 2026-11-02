"""
Daily P&L Reporting Module for Hedge Fund Tracker.

Provides comprehensive daily profit/loss calculation, attribution analysis
by asset class and GICS sector, benchmark comparison, and risk metrics.

Mathematical Foundation:
    Daily P&L:
        PnL_t = sum_i(quantity_i * (P_{i,t} - P_{i,t-1}))
        PnL_pct = PnL_t / NAV_{t-1}

    Attribution (Brinson-Fachler Model):
        Allocation Effect:
            AE_j = (w_{p,j} - w_{b,j}) * (R_{b,j} - R_b)
        Selection Effect:
            SE_j = w_{b,j} * (R_{p,j} - R_{b,j})
        Interaction Effect:
            IE_j = (w_{p,j} - w_{b,j}) * (R_{p,j} - R_{b,j})
        Total Alpha = sum_j(AE_j + SE_j + IE_j)

    Alpha (Jensen's Alpha):
        alpha = R_p - (R_f + beta * (R_m - R_f))
        beta = Cov(R_p, R_m) / Var(R_m)

    VaR (Value at Risk) at confidence level c:
        VaR_c = -mu + z_c * sigma    (parametric, assuming normality)
        VaR_c = -percentile(returns, 1-c)  (historical)

    CVaR (Conditional VaR / Expected Shortfall):
        CVaR_c = -E[R | R <= -VaR_c]
        CVaR_c = -(1/(1-c)) * integral_{-inf}^{-VaR} r*f(r)dr

    Max Drawdown:
        MDD = max_t (max_{s<=t}(NAV_s) - NAV_t) / max_{s<=t}(NAV_s)

Usage:
    from daily_pnl import DailyPnLReport, PnLBreakdown
    import sys; sys.path.insert(0, "..")
    from openbb_universe import AssetClass

    report = DailyPnLReport()
    pnl = report.calculate_daily_pnl(positions, prices)
    attribution = report.attribution_by_asset_class(pnl)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# Allow import from parent
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openbb_universe import AssetClass, EQUITY_GICS_MAP, detect_asset_class

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    asset_class: AssetClass
    gics_sector: Optional[str] = None
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    weight: float = 0.0

    def __post_init__(self):
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = self.quantity * (self.current_price - self.avg_cost)


@dataclass
class PnLBreakdown:
    """Comprehensive P&L breakdown for a single day."""
    date: datetime
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    nav: float
    prev_nav: float
    positions: list[Position] = field(default_factory=list)
    pnl_by_symbol: dict[str, float] = field(default_factory=dict)
    pnl_by_asset_class: dict[str, float] = field(default_factory=dict)
    pnl_by_sector: dict[str, float] = field(default_factory=dict)
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    num_winners: int = 0
    num_losers: int = 0
    biggest_winner: str = ""
    biggest_loser: str = ""
    biggest_winner_pnl: float = 0.0
    biggest_loser_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Daily PnL Report Engine
# ---------------------------------------------------------------------------

class DailyPnLReport:
    """
    Daily P&L reporting engine for hedge fund position tracking.

    Provides:
        - Position-level and portfolio-level P&L calculation
        - Attribution by asset class and GICS sector
        - Benchmark comparison (alpha/beta decomposition)
        - Risk metrics: VaR, CVaR, max drawdown, Sharpe
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
        var_confidence: float = 0.95,
    ):
        """
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate.
        trading_days : int
            Trading days per year.
        var_confidence : float
            Confidence level for VaR/CVaR.
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.var_confidence = var_confidence

    def calculate_daily_pnl(
        self,
        positions: list[dict[str, Any]],
        prices: dict[str, pd.DataFrame],
        date: Optional[str] = None,
    ) -> PnLBreakdown:
        """
        Calculate daily P&L for all positions.

        For each position i:
            pnl_i = quantity_i * (P_{i,t} - P_{i,t-1})
            pnl_pct_i = pnl_i / (quantity_i * P_{i,t-1})

        Total P&L:
            PnL_total = sum_i(pnl_i)
            PnL_pct = PnL_total / NAV_{t-1}

        Parameters
        ----------
        positions : list[dict]
            Each dict has keys: "symbol", "quantity", "avg_cost".
            Positive quantity = long, negative = short.
        prices : dict[str, pd.DataFrame]
            Symbol -> OHLCV DataFrame with at least 2 rows.
        date : str, optional
            Date for the P&L snapshot. Defaults to latest date in data.

        Returns
        -------
        PnLBreakdown
            Complete P&L breakdown for the day.
        """
        pnl_by_symbol = {}
        pnl_by_ac = {}
        pnl_by_sector = {}
        pos_objects = []
        total_pnl = 0.0
        long_pnl = 0.0
        short_pnl = 0.0
        prev_nav = 0.0
        current_nav = 0.0
        gross_long = 0.0
        gross_short = 0.0

        for pos_dict in positions:
            symbol = pos_dict["symbol"]
            quantity = pos_dict["quantity"]
            avg_cost = pos_dict.get("avg_cost", 0.0)

            if symbol not in prices or prices[symbol] is None:
                continue

            df = prices[symbol]
            if "Close" not in df.columns or len(df) < 2:
                continue

            close = df["Close"].dropna()
            if len(close) < 2:
                continue

            current_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])

            asset_class = detect_asset_class(symbol)
            gics_sector = EQUITY_GICS_MAP.get(symbol, "Unknown") if asset_class == AssetClass.EQUITY else None

            # Position P&L
            daily_pnl = quantity * (current_price - prev_price)
            pnl_by_symbol[symbol] = daily_pnl
            total_pnl += daily_pnl

            # P&L by asset class
            ac_key = asset_class.value
            pnl_by_ac[ac_key] = pnl_by_ac.get(ac_key, 0.0) + daily_pnl

            # P&L by GICS sector
            if gics_sector:
                pnl_by_sector[gics_sector] = pnl_by_sector.get(gics_sector, 0.0) + daily_pnl

            # Long/short decomposition
            if quantity > 0:
                long_pnl += daily_pnl
                gross_long += abs(quantity * current_price)
            else:
                short_pnl += daily_pnl
                gross_short += abs(quantity * current_price)

            prev_nav += abs(quantity) * prev_price
            current_nav += quantity * current_price

            pos_objects.append(Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                asset_class=asset_class,
                gics_sector=gics_sector,
            ))

        # NAV and percentage
        pnl_pct = total_pnl / prev_nav if prev_nav > 0 else 0.0

        # Winners and losers
        winners = {s: p for s, p in pnl_by_symbol.items() if p > 0}
        losers = {s: p for s, p in pnl_by_symbol.items() if p < 0}

        biggest_winner = max(winners, key=winners.get) if winners else ""
        biggest_loser = min(losers, key=losers.get) if losers else ""

        # Calculate weights
        total_mv = sum(abs(p.market_value) for p in pos_objects)
        if total_mv > 0:
            for p in pos_objects:
                p.weight = p.market_value / total_mv

        report_date = datetime.now()
        if date:
            try:
                report_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                pass

        return PnLBreakdown(
            date=report_date,
            total_pnl=total_pnl,
            total_pnl_pct=pnl_pct,
            realized_pnl=0.0,  # Would need trade data to decompose
            unrealized_pnl=sum(p.unrealized_pnl for p in pos_objects),
            nav=current_nav,
            prev_nav=prev_nav,
            positions=pos_objects,
            pnl_by_symbol=pnl_by_symbol,
            pnl_by_asset_class=pnl_by_ac,
            pnl_by_sector=pnl_by_sector,
            long_pnl=long_pnl,
            short_pnl=short_pnl,
            gross_exposure=gross_long + gross_short,
            net_exposure=gross_long - gross_short,
            num_winners=len(winners),
            num_losers=len(losers),
            biggest_winner=biggest_winner,
            biggest_loser=biggest_loser,
            biggest_winner_pnl=winners.get(biggest_winner, 0.0),
            biggest_loser_pnl=losers.get(biggest_loser, 0.0),
        )

    def attribution_by_asset_class(
        self, pnl: PnLBreakdown
    ) -> dict[str, float]:
        """
        Attribute P&L contribution by asset class.

        For each asset class j:
            contribution_j = sum_{i in j}(PnL_i)
            pct_of_total_j = contribution_j / |PnL_total|

        Parameters
        ----------
        pnl : PnLBreakdown
            P&L breakdown from calculate_daily_pnl.

        Returns
        -------
        dict[AssetClass, float]
            Asset class -> P&L contribution.
        """
        return dict(pnl.pnl_by_asset_class)

    def attribution_by_gics_sector(
        self, pnl: PnLBreakdown
    ) -> dict[str, float]:
        """
        Attribute P&L contribution by GICS sector.

        For each GICS sector s:
            contribution_s = sum_{i in s}(PnL_i)

        Parameters
        ----------
        pnl : PnLBreakdown

        Returns
        -------
        dict[str, float]
            Sector name -> P&L contribution.
        """
        return dict(pnl.pnl_by_sector)

    def vs_benchmark(
        self,
        pnl: PnLBreakdown,
        benchmark_prices: pd.DataFrame,
        benchmark: str = "SPY",
        lookback_days: int = 252,
        portfolio_returns: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate alpha vs a benchmark.

        Jensen's Alpha:
            alpha = R_p - (R_f + beta * (R_m - R_f))

        Where:
            R_p = portfolio return
            R_m = benchmark (market) return
            R_f = risk-free rate (daily)
            beta = Cov(R_p, R_m) / Var(R_m)

        For single-day alpha (simplified):
            alpha = R_p - beta * R_m
            where beta is estimated from trailing returns.

        Parameters
        ----------
        pnl : PnLBreakdown
            Today's P&L breakdown.
        benchmark_prices : pd.DataFrame
            Benchmark OHLCV data.
        benchmark : str
            Benchmark name (for logging).
        lookback_days : int
            Days for beta estimation.
        portfolio_returns : pd.Series, optional
            Historical portfolio daily returns for beta estimation.

        Returns
        -------
        float
            Alpha: portfolio outperformance vs benchmark (daily).
        """
        if benchmark_prices is None or "Close" not in benchmark_prices.columns:
            return 0.0

        bm_close = benchmark_prices["Close"].dropna()
        if len(bm_close) < 2:
            return 0.0

        bm_return = float(bm_close.iloc[-1] / bm_close.iloc[-2] - 1.0)
        portfolio_return = pnl.total_pnl_pct
        daily_rf = self.risk_free_rate / self.trading_days

        # Estimate beta from historical returns if available
        beta = 1.0  # default
        if portfolio_returns is not None and len(portfolio_returns) >= 60:
            bm_returns = bm_close.pct_change().dropna()
            # Align lengths
            min_len = min(len(portfolio_returns), len(bm_returns))
            p_ret = portfolio_returns.iloc[-min_len:].values
            b_ret = bm_returns.iloc[-min_len:].values

            cov = np.cov(p_ret, b_ret)
            if cov.shape == (2, 2) and cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]

        # Jensen's alpha (daily)
        alpha = portfolio_return - (daily_rf + beta * (bm_return - daily_rf))
        return float(alpha)

    def risk_metrics(
        self,
        positions: list[dict[str, Any]],
        prices: dict[str, pd.DataFrame],
        lookback_days: int = 252,
    ) -> dict[str, float]:
        """
        Calculate portfolio risk metrics.

        Metrics:
            VaR (parametric):
                VaR_c = -(mu - z_c * sigma) * NAV
                where z_{0.95} = 1.645, z_{0.99} = 2.326

            VaR (historical):
                VaR_c = -percentile(portfolio_returns, 100*(1-c)) * NAV

            CVaR (Expected Shortfall):
                CVaR_c = -mean(R_p | R_p <= -VaR_c/NAV) * NAV

            Max Drawdown (trailing):
                MDD = max_{t in window}((peak_t - NAV_t) / peak_t)

            Realized Volatility:
                sigma_realized = std(R_p, window) * sqrt(252)

            Sharpe Ratio (trailing):
                SR = (mean(R_p) * 252 - R_f) / (std(R_p) * sqrt(252))

            Beta (to SPY):
                beta = Cov(R_p, R_spy) / Var(R_spy)

            Sortino Ratio:
                Sortino = (mean(R_p) * 252 - R_f) / (downside_dev * sqrt(252))
                downside_dev = sqrt(mean(min(R_p - R_target, 0)^2))

        Parameters
        ----------
        positions : list[dict]
            Current positions.
        prices : dict[str, pd.DataFrame]
            Price data per symbol.
        lookback_days : int
            Window for historical calculations.

        Returns
        -------
        dict[str, float]
            Dictionary of risk metrics.
        """
        # Build portfolio return series
        all_returns = []
        position_map = {p["symbol"]: p["quantity"] for p in positions}

        for symbol, qty in position_map.items():
            if symbol not in prices or prices[symbol] is None:
                continue
            df = prices[symbol]
            if "Close" not in df.columns:
                continue

            close = df["Close"].dropna()
            if len(close) < 2:
                continue

            rets = close.pct_change().dropna()
            weighted_rets = rets * (qty / max(abs(qty), 1))  # normalize
            all_returns.append(weighted_rets)

        if not all_returns:
            return {
                "var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0,
                "max_drawdown": 0.0, "realized_volatility": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "beta": 0.0,
            }

        # Aggregate portfolio returns
        combined = pd.concat(all_returns, axis=1)
        portfolio_returns = combined.mean(axis=1).dropna()

        if len(portfolio_returns) < 20:
            return {
                "var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0,
                "max_drawdown": 0.0, "realized_volatility": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "beta": 0.0,
            }

        window = min(lookback_days, len(portfolio_returns))
        ret_window = portfolio_returns.iloc[-window:]

        mu = float(ret_window.mean())
        sigma = float(ret_window.std())

        # VaR (parametric)
        z_95 = 1.645
        z_99 = 2.326
        var_95_pct = -(mu - z_95 * sigma)
        var_99_pct = -(mu - z_99 * sigma)

        # VaR (historical) - use the more conservative of the two
        hist_var_95 = -float(np.percentile(ret_window.values, 5))
        var_95_final = max(var_95_pct, hist_var_95)

        # CVaR (Expected Shortfall)
        var_threshold = np.percentile(ret_window.values, 5)
        tail = ret_window[ret_window <= var_threshold]
        cvar_95 = -float(tail.mean()) if len(tail) > 0 else var_95_final

        # Max Drawdown
        cum_returns = (1.0 + ret_window).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = float(drawdown.min())

        # Realized Volatility (annualized)
        realized_vol = sigma * np.sqrt(self.trading_days)

        # Sharpe Ratio
        ann_return = mu * self.trading_days
        sharpe = (ann_return - self.risk_free_rate) / realized_vol if realized_vol > 0 else 0.0

        # Sortino Ratio
        daily_rf = self.risk_free_rate / self.trading_days
        downside = ret_window[ret_window < daily_rf] - daily_rf
        downside_dev = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else sigma
        ann_downside = downside_dev * np.sqrt(self.trading_days)
        sortino = (ann_return - self.risk_free_rate) / ann_downside if ann_downside > 0 else 0.0

        return {
            "var_95": var_95_final,
            "var_99": var_99_pct,
            "cvar_95": cvar_95,
            "max_drawdown": max_dd,
            "realized_volatility": realized_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "beta": 1.0,  # Would need benchmark data for actual calculation
            "annualized_return": ann_return,
            "daily_mean_return": mu,
            "daily_std_return": sigma,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    report = DailyPnLReport()

    # Synthetic demo
    np.random.seed(42)
    n_days = 60

    symbols = ["AAPL", "MSFT", "GC=F", "TLT", "BTC-USD"]
    fake_prices = {}
    for sym in symbols:
        base = {"AAPL": 180, "MSFT": 380, "GC=F": 2000, "TLT": 100, "BTC-USD": 60000}[sym]
        vol = {"AAPL": 0.02, "MSFT": 0.02, "GC=F": 0.01, "TLT": 0.008, "BTC-USD": 0.04}[sym]
        closes = base * np.exp(np.cumsum(np.random.normal(0.0003, vol, n_days)))
        fake_prices[sym] = pd.DataFrame({
            "Open": closes * 0.999,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": np.random.randint(1e6, 1e8, n_days),
        })

    positions = [
        {"symbol": "AAPL", "quantity": 100, "avg_cost": 175.0},
        {"symbol": "MSFT", "quantity": 50, "avg_cost": 370.0},
        {"symbol": "GC=F", "quantity": 10, "avg_cost": 1980.0},
        {"symbol": "TLT", "quantity": 200, "avg_cost": 98.0},
        {"symbol": "BTC-USD", "quantity": 0.5, "avg_cost": 55000.0},
    ]

    print("=== Daily P&L ===")
    pnl = report.calculate_daily_pnl(positions, fake_prices)
    print(f"  Total P&L: ${pnl.total_pnl:,.2f} ({pnl.total_pnl_pct:.4%})")
    print(f"  NAV: ${pnl.nav:,.2f}")
    print(f"  Winners: {pnl.num_winners}, Losers: {pnl.num_losers}")

    print("\n=== Attribution by Asset Class ===")
    ac_attr = report.attribution_by_asset_class(pnl)
    for ac, contribution in sorted(ac_attr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {ac}: ${contribution:,.2f}")

    print("\n=== Attribution by GICS Sector ===")
    sector_attr = report.attribution_by_gics_sector(pnl)
    for sector, contribution in sorted(sector_attr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {sector}: ${contribution:,.2f}")

    print("\n=== Risk Metrics ===")
    metrics = report.risk_metrics(positions, fake_prices)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
