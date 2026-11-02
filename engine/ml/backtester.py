"""
Metadron Capital — Backtesting Engine
======================================
Walk-forward backtesting, Monte Carlo simulation, scenario analysis,
and strategy comparison for alpha extraction validation.

Pure numpy/pandas implementation with full transaction cost modeling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    from ..data.yahoo_data import get_adj_close, get_returns
    from ..data.universe_engine import SECTOR_ETFS, RV_PAIRS
    from ..portfolio.beta_corridor import ALPHA, R_LOW, R_HIGH, BETA_MAX, VOL_STANDARD
    from ..execution.paper_broker import SignalType
except ImportError:
    get_adj_close = None
    get_returns = None
    SECTOR_ETFS = [
        "XLK", "XLV", "XLF", "XLE", "XLI",
        "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    ]
    RV_PAIRS = [
        ("XLK", "XLI"), ("XLY", "XLP"), ("XLE", "XLU"),
    ]
    ALPHA = 0.05
    R_LOW = 0.02
    R_HIGH = 0.10
    BETA_MAX = 1.2
    VOL_STANDARD = 0.16

    class SignalType:
        LONG = "LONG"
        SHORT = "SHORT"
        FLAT = "FLAT"


# ── Trading days per year ─────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR) if np is not None else 15.875


# ═══════════════════════════════════════════════════════════════════════
# 1. Configuration & Result Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """All tunables for a single backtest run."""

    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    initial_nav: float = 1_000_000.0
    benchmark: str = "SPY"
    rebalance_freq: str = "daily"          # daily | weekly | monthly

    slippage_bps: float = 5.0              # one-way slippage in basis points
    commission_bps: float = 1.0            # one-way commission in basis points
    max_position_pct: float = 0.10         # 10 % single-name cap
    max_gross_exposure: float = 1.50       # 150 % gross

    risk_free_rate: float = 0.04           # annualized risk-free rate
    lookback_window: int = 60              # default lookback in trading days
    signal_threshold: float = 0.0          # minimum absolute signal to trade


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    # Core series
    nav_series: "pd.Series" = None                # type: ignore[assignment]
    returns_series: "pd.Series" = None             # type: ignore[assignment]
    benchmark_series: "pd.Series" = None           # type: ignore[assignment]

    # Summary statistics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0

    # Trade statistics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0

    # Breakdown tables
    monthly_returns: "pd.DataFrame" = None         # type: ignore[assignment]
    drawdown_series: "pd.Series" = None            # type: ignore[assignment]

    # Detailed trade log
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────
    def summary_dict(self) -> Dict[str, Any]:
        """Return flat dict of scalar metrics."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. Helper Utilities
# ═══════════════════════════════════════════════════════════════════════

def _compute_drawdown(nav: "pd.Series") -> "pd.Series":
    """Running drawdown from peak, as a negative fraction."""
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return dd


def _compute_max_drawdown(nav: "pd.Series") -> float:
    """Maximum drawdown magnitude (positive number)."""
    dd = _compute_drawdown(nav)
    return float(-dd.min()) if len(dd) > 0 else 0.0


def _annualized_return(total_ret: float, n_days: int) -> float:
    """Convert total return over n trading days to annualized."""
    if n_days <= 0:
        return 0.0
    years = n_days / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0.0
    return float((1.0 + total_ret) ** (1.0 / years) - 1.0)


def _sharpe(returns: "pd.Series", risk_free: float = 0.04) -> float:
    """Annualized Sharpe ratio."""
    if returns is None or len(returns) < 2:
        return 0.0
    daily_rf = (1.0 + risk_free) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = returns - daily_rf
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * ANNUALIZATION_FACTOR)


def _sortino(returns: "pd.Series", risk_free: float = 0.04) -> float:
    """Annualized Sortino ratio."""
    if returns is None or len(returns) < 2:
        return 0.0
    daily_rf = (1.0 + risk_free) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = returns - daily_rf
    downside = excess[excess < 0]
    down_vol = downside.std() if len(downside) > 1 else 0.0
    if down_vol == 0 or np.isnan(down_vol):
        return 0.0
    return float(excess.mean() / down_vol * ANNUALIZATION_FACTOR)


def _monthly_returns_table(nav: "pd.Series") -> "pd.DataFrame":
    """Pivot table of monthly returns (rows=year, cols=month)."""
    monthly = nav.resample("ME").last().pct_change().dropna()
    table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = table.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]
    return pivot


def _apply_transaction_costs(
    turnover: float, slippage_bps: float, commission_bps: float
) -> float:
    """Return total cost as a fraction for the given turnover."""
    cost_bps = (slippage_bps + commission_bps) * 2  # round-trip
    return turnover * cost_bps / 10_000.0


def _rebalance_mask(dates: "pd.DatetimeIndex", freq: str) -> "pd.Series":
    """Boolean mask indicating which dates are rebalance dates."""
    mask = pd.Series(False, index=dates)
    if freq == "daily":
        mask[:] = True
    elif freq == "weekly":
        mask[dates.weekday == 0] = True      # Mondays
        if not mask.iloc[0]:
            mask.iloc[0] = True
    elif freq == "monthly":
        month_end = dates.to_series().groupby(
            [dates.year, dates.month]
        ).transform("last")
        mask[dates.isin(month_end.unique())] = True
        if not mask.iloc[0]:
            mask.iloc[0] = True
    else:
        mask[:] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════
# 3. Signal Backtester
# ═══════════════════════════════════════════════════════════════════════

class SignalBacktester:
    """
    Walk-forward backtesting framework.

    Accepts a DataFrame of daily signals (columns=tickers, values in [-1, 1])
    and a corresponding prices DataFrame.  Simulates daily P&L with
    transaction cost modeling, position sizing, and drawdown tracking.
    """

    def __init__(self) -> None:
        self._trade_log: List[Dict[str, Any]] = []

    # ── public entry point ─────────────────────────────────────────────
    def run(
        self,
        signals_df: "pd.DataFrame",
        prices_df: "pd.DataFrame",
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run the backtest and return a full BacktestResult."""
        if config is None:
            config = BacktestConfig()

        self._trade_log = []

        # Align signals and prices on their shared date index
        common_idx = signals_df.index.intersection(prices_df.index).sort_values()
        if len(common_idx) < 2:
            warnings.warn("Insufficient overlapping dates for backtest.")
            return BacktestResult()

        signals = signals_df.loc[common_idx].fillna(0.0)
        prices = prices_df.loc[common_idx].ffill().bfill()
        tickers = [c for c in signals.columns if c in prices.columns]
        if not tickers:
            warnings.warn("No common tickers between signals and prices.")
            return BacktestResult()

        signals = signals[tickers]
        prices = prices[tickers]
        returns = prices.pct_change().fillna(0.0)

        rebal_mask = _rebalance_mask(common_idx, config.rebalance_freq)

        n_days = len(common_idx)
        nav = np.full(n_days, config.initial_nav)
        weights = pd.DataFrame(0.0, index=common_idx, columns=tickers)
        prev_weights = pd.Series(0.0, index=tickers)

        for i in range(1, n_days):
            date = common_idx[i]
            daily_ret = returns.iloc[i]

            if rebal_mask.iloc[i]:
                target_weights = self._compute_target_weights(
                    signals.iloc[i], config
                )
            else:
                # drift existing weights by daily returns
                drifted = prev_weights * (1.0 + daily_ret)
                total = drifted.sum()
                target_weights = drifted / total if abs(total) > 1e-12 else drifted

            # Turnover and costs
            turnover = (target_weights - prev_weights).abs().sum() / 2.0
            cost_frac = _apply_transaction_costs(
                turnover, config.slippage_bps, config.commission_bps
            )

            # Portfolio return for the day (net of costs)
            port_ret = (prev_weights * daily_ret).sum() - cost_frac
            nav[i] = nav[i - 1] * (1.0 + port_ret)

            # Log trades when weights change significantly
            if rebal_mask.iloc[i]:
                self._log_trades(date, prev_weights, target_weights, prices.iloc[i])

            prev_weights = target_weights.copy()
            weights.iloc[i] = target_weights

        nav_series = pd.Series(nav, index=common_idx, name="NAV")
        returns_series = nav_series.pct_change().fillna(0.0)
        returns_series.name = "returns"

        total_return = float(nav[-1] / nav[0] - 1.0)
        ann_ret = _annualized_return(total_return, n_days)
        ann_vol = float(returns_series.std() * ANNUALIZATION_FACTOR)
        sharpe = _sharpe(returns_series, config.risk_free_rate)
        sortino = _sortino(returns_series, config.risk_free_rate)
        max_dd = _compute_max_drawdown(nav_series)
        dd_series = _compute_drawdown(nav_series)

        # Trade statistics
        wins, losses = self._compute_trade_stats()

        monthly = _monthly_returns_table(nav_series)

        return BacktestResult(
            nav_series=nav_series,
            returns_series=returns_series,
            benchmark_series=None,
            total_return=total_return,
            annualized_return=ann_ret,
            annualized_vol=ann_vol,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            win_rate=wins["rate"],
            profit_factor=wins["profit_factor"],
            avg_win=wins["avg"],
            avg_loss=losses["avg"],
            total_trades=wins["count"] + losses["count"],
            monthly_returns=monthly,
            drawdown_series=dd_series,
            trade_log=list(self._trade_log),
        )

    # ── internals ──────────────────────────────────────────────────────
    def _compute_target_weights(
        self, signal_row: "pd.Series", config: BacktestConfig
    ) -> "pd.Series":
        """Convert raw signals to capped, normalized weights."""
        raw = signal_row.copy()
        # Zero out signals below threshold
        raw[raw.abs() < config.signal_threshold] = 0.0

        # Normalize so gross exposure == 1 (then scale)
        gross = raw.abs().sum()
        if gross < 1e-12:
            return pd.Series(0.0, index=raw.index)

        weights = raw / gross  # gross == 1

        # Cap individual positions
        weights = weights.clip(
            lower=-config.max_position_pct,
            upper=config.max_position_pct,
        )

        # Rescale to respect max gross exposure
        current_gross = weights.abs().sum()
        if current_gross > config.max_gross_exposure:
            weights *= config.max_gross_exposure / current_gross

        return weights

    def _log_trades(
        self,
        date: Any,
        old_w: "pd.Series",
        new_w: "pd.Series",
        prices: "pd.Series",
    ) -> None:
        """Record individual trade events."""
        diff = new_w - old_w
        for ticker in diff.index:
            delta = diff[ticker]
            if abs(delta) < 1e-6:
                continue
            self._trade_log.append({
                "date": str(date)[:10],
                "ticker": ticker,
                "side": "BUY" if delta > 0 else "SELL",
                "weight_delta": round(float(delta), 6),
                "price": round(float(prices.get(ticker, 0.0)), 4),
                "new_weight": round(float(new_w[ticker]), 6),
            })

    def _compute_trade_stats(self) -> Tuple[Dict, Dict]:
        """Aggregate trade log into win / loss stats."""
        if not self._trade_log:
            return (
                {"rate": 0.0, "avg": 0.0, "count": 0, "profit_factor": 0.0},
                {"avg": 0.0, "count": 0},
            )

        pnls: List[float] = []
        positions: Dict[str, List[Dict]] = {}
        for t in self._trade_log:
            ticker = t["ticker"]
            if ticker not in positions:
                positions[ticker] = []
            positions[ticker].append(t)

        # Simplified P&L: for each ticker, compare successive trade prices
        for ticker, trades in positions.items():
            for j in range(1, len(trades)):
                prev_price = trades[j - 1]["price"]
                cur_price = trades[j]["price"]
                if prev_price == 0:
                    continue
                direction = 1.0 if trades[j - 1]["side"] == "BUY" else -1.0
                ret = direction * (cur_price / prev_price - 1.0)
                pnls.append(ret)

        if not pnls:
            return (
                {"rate": 0.0, "avg": 0.0, "count": 0, "profit_factor": 0.0},
                {"avg": 0.0, "count": 0},
            )

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-12
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return (
            {
                "rate": round(win_rate, 4),
                "avg": round(avg_win, 6),
                "count": len(wins),
                "profit_factor": round(profit_factor, 4),
            },
            {"avg": round(avg_loss, 6), "count": len(losses)},
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Strategy Backtester
# ═══════════════════════════════════════════════════════════════════════

class StrategyBacktester:
    """
    Pre-built strategy signal generators that plug into SignalBacktester.
    Each method returns a signals DataFrame aligned to the prices index.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self._signal_bt = SignalBacktester()

    # ── Momentum ───────────────────────────────────────────────────────
    def momentum_signals(
        self, prices: "pd.DataFrame", lookback: int = 60
    ) -> "pd.DataFrame":
        """
        Cross-sectional momentum: rank tickers by total return over
        *lookback* days, go long the top quintile, short the bottom.
        """
        ret = prices.pct_change(lookback)
        ranks = ret.rank(axis=1, pct=True)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranks >= 0.80] = 1.0    # top 20 %
        signals[ranks <= 0.20] = -1.0   # bottom 20 %
        return signals

    def run_momentum(
        self, prices: "pd.DataFrame", lookback: int = 60
    ) -> BacktestResult:
        signals = self.momentum_signals(prices, lookback)
        return self._signal_bt.run(signals, prices, self.config)

    # ── Mean Reversion ─────────────────────────────────────────────────
    def mean_reversion_signals(
        self, prices: "pd.DataFrame", lookback: int = 20, z_threshold: float = 1.5
    ) -> "pd.DataFrame":
        """
        Buy assets that are oversold (z-score < -threshold),
        sell assets that are overbought (z-score > +threshold).
        """
        rolling_mean = prices.rolling(lookback).mean()
        rolling_std = prices.rolling(lookback).std()
        z = (prices - rolling_mean) / rolling_std.replace(0, np.nan)
        z = z.fillna(0.0)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[z < -z_threshold] = 1.0    # oversold  -> buy
        signals[z > z_threshold] = -1.0    # overbought -> sell
        return signals

    def run_mean_reversion(
        self, prices: "pd.DataFrame", lookback: int = 20, z_threshold: float = 1.5
    ) -> BacktestResult:
        signals = self.mean_reversion_signals(prices, lookback, z_threshold)
        return self._signal_bt.run(signals, prices, self.config)

    # ── Relative Value (Pair Trading) ──────────────────────────────────
    def relative_value_signals(
        self,
        prices: "pd.DataFrame",
        pairs: Optional[List[Tuple[str, str]]] = None,
        lookback: int = 40,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
    ) -> "pd.DataFrame":
        """
        For each pair, compute log-spread z-score.
        Enter when |z| > z_entry, exit when |z| < z_exit.
        """
        if pairs is None:
            pairs = [(a, b) for a, b in RV_PAIRS
                     if a in prices.columns and b in prices.columns]

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for leg_a, leg_b in pairs:
            if leg_a not in prices.columns or leg_b not in prices.columns:
                continue
            spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
            mu = spread.rolling(lookback).mean()
            sigma = spread.rolling(lookback).std().replace(0, np.nan)
            z = ((spread - mu) / sigma).fillna(0.0)

            # Entry: z > entry -> short A, long B; z < -entry -> long A, short B
            long_a = z < -z_entry
            short_a = z > z_entry
            # Exit
            flat = z.abs() < z_exit

            position = pd.Series(0.0, index=prices.index)
            in_pos = 0.0
            for i in range(len(z)):
                if long_a.iloc[i]:
                    in_pos = 1.0
                elif short_a.iloc[i]:
                    in_pos = -1.0
                elif flat.iloc[i]:
                    in_pos = 0.0
                position.iloc[i] = in_pos

            signals[leg_a] += position
            signals[leg_b] -= position

        return signals.clip(-1, 1)

    def run_relative_value(
        self, prices: "pd.DataFrame", pairs: Optional[List[Tuple[str, str]]] = None,
        lookback: int = 40,
    ) -> BacktestResult:
        signals = self.relative_value_signals(prices, pairs, lookback)
        return self._signal_bt.run(signals, prices, self.config)

    # ── Quality Factor ─────────────────────────────────────────────────
    def quality_signals(
        self, prices: "pd.DataFrame", lookback: int = 60
    ) -> "pd.DataFrame":
        """
        Quality proxy: low-volatility + positive trend.
        Rank by inverse realized vol and positive momentum.
        """
        ret = prices.pct_change()
        vol = ret.rolling(lookback).std()
        mom = prices.pct_change(lookback)

        inv_vol_rank = (-vol).rank(axis=1, pct=True)   # low vol = high rank
        mom_rank = mom.rank(axis=1, pct=True)
        composite = (inv_vol_rank + mom_rank) / 2.0

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[composite >= 0.75] = 1.0
        signals[composite <= 0.25] = -1.0
        return signals.fillna(0.0)

    def run_quality(
        self, prices: "pd.DataFrame", lookback: int = 60
    ) -> BacktestResult:
        signals = self.quality_signals(prices, lookback)
        return self._signal_bt.run(signals, prices, self.config)

    # ── Sector Rotation ────────────────────────────────────────────────
    def sector_rotation_signals(
        self, prices: "pd.DataFrame", lookback: int = 40, top_n: int = 3
    ) -> "pd.DataFrame":
        """
        Rotate into leading sectors based on momentum rank.
        Long top_n sectors, short bottom_n.
        """
        sector_cols = [c for c in SECTOR_ETFS if c in prices.columns]
        if not sector_cols:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        sector_prices = prices[sector_cols]
        mom = sector_prices.pct_change(lookback)
        ranks = mom.rank(axis=1, ascending=False)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in sector_cols:
            signals.loc[ranks[col] <= top_n, col] = 1.0
            signals.loc[ranks[col] > len(sector_cols) - top_n, col] = -1.0

        return signals

    def run_sector_rotation(
        self, prices: "pd.DataFrame", lookback: int = 40, top_n: int = 3
    ) -> BacktestResult:
        signals = self.sector_rotation_signals(prices, lookback, top_n)
        return self._signal_bt.run(signals, prices, self.config)


# ═══════════════════════════════════════════════════════════════════════
# 5. Monte Carlo Simulator
# ═══════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Generate simulated price paths via bootstrap or GBM.
    Produce VaR / CVaR estimates and confidence bands.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    # ── Bootstrap from historical returns ──────────────────────────────
    def bootstrap_paths(
        self,
        returns: "pd.Series",
        n_paths: int = 1000,
        n_days: int = 252,
        initial_price: float = 100.0,
    ) -> np.ndarray:
        """
        Resample daily returns with replacement to build
        *n_paths* simulated price trajectories of length *n_days*.

        Returns: ndarray of shape (n_paths, n_days + 1)
        """
        hist = returns.dropna().values
        if len(hist) < 5:
            raise ValueError("Need at least 5 historical returns for bootstrap.")

        sampled = self._rng.choice(hist, size=(n_paths, n_days), replace=True)
        cum = np.cumprod(1.0 + sampled, axis=1)
        paths = np.column_stack([np.ones(n_paths), cum]) * initial_price
        return paths

    # ── Geometric Brownian Motion ──────────────────────────────────────
    def gbm_paths(
        self,
        mu: float,
        sigma: float,
        n_paths: int = 1000,
        n_days: int = 252,
        initial_price: float = 100.0,
        dt: float = 1.0 / 252,
    ) -> np.ndarray:
        """
        Generate GBM paths:
            dS = mu * S * dt + sigma * S * dW

        Returns: ndarray of shape (n_paths, n_days + 1)
        """
        z = self._rng.standard_normal((n_paths, n_days))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        log_ret = drift + diffusion
        cum_log = np.cumsum(log_ret, axis=1)
        paths = np.column_stack([
            np.zeros(n_paths), cum_log
        ])
        paths = initial_price * np.exp(paths)
        return paths

    # ── Correlated multi-asset GBM ─────────────────────────────────────
    def correlated_gbm(
        self,
        mu_vec: np.ndarray,
        cov_matrix: np.ndarray,
        n_paths: int = 500,
        n_days: int = 252,
        initial_prices: Optional[np.ndarray] = None,
        dt: float = 1.0 / 252,
    ) -> np.ndarray:
        """
        Correlated multi-asset GBM simulation.

        Parameters
        ----------
        mu_vec : (n_assets,) annualized expected returns
        cov_matrix : (n_assets, n_assets) annualized covariance matrix
        initial_prices : (n_assets,) starting prices

        Returns: ndarray of shape (n_paths, n_days + 1, n_assets)
        """
        n_assets = len(mu_vec)
        if initial_prices is None:
            initial_prices = np.ones(n_assets) * 100.0

        L = np.linalg.cholesky(cov_matrix)
        sigma_vec = np.sqrt(np.diag(cov_matrix))
        drift = (mu_vec - 0.5 * sigma_vec ** 2) * dt

        z = self._rng.standard_normal((n_paths, n_days, n_assets))
        corr_z = z @ L.T  # apply correlation structure

        log_ret = drift[np.newaxis, np.newaxis, :] + np.sqrt(dt) * corr_z
        cum_log = np.cumsum(log_ret, axis=1)
        zeros = np.zeros((n_paths, 1, n_assets))
        cum_log = np.concatenate([zeros, cum_log], axis=1)
        paths = initial_prices[np.newaxis, np.newaxis, :] * np.exp(cum_log)
        return paths

    # ── VaR / CVaR ─────────────────────────────────────────────────────
    def estimate_var(
        self, terminal_values: np.ndarray, confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute Value-at-Risk and Conditional VaR from terminal values.

        Parameters
        ----------
        terminal_values : 1-D array of simulated terminal NAV or returns
        confidence : confidence level (e.g. 0.95)

        Returns dict with VaR, CVaR, mean, median, std.
        """
        returns = terminal_values / terminal_values.mean() - 1.0
        alpha = 1.0 - confidence
        var = float(-np.percentile(returns, alpha * 100))
        tail = returns[returns <= -var]
        cvar = float(-tail.mean()) if len(tail) > 0 else var

        return {
            "VaR": round(var, 6),
            "CVaR": round(cvar, 6),
            "mean": round(float(np.mean(terminal_values)), 4),
            "median": round(float(np.median(terminal_values)), 4),
            "std": round(float(np.std(terminal_values)), 4),
            "confidence": confidence,
        }

    # ── Confidence intervals ───────────────────────────────────────────
    def confidence_intervals(
        self, paths: np.ndarray, levels: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute percentile-based confidence bands across simulated paths.

        Parameters
        ----------
        paths : (n_paths, n_steps) array
        levels : list of confidence levels e.g. [0.05, 0.25, 0.50, 0.75, 0.95]

        Returns dict mapping percentile labels to 1-D arrays of length n_steps.
        """
        if levels is None:
            levels = [0.05, 0.25, 0.50, 0.75, 0.95]

        result: Dict[str, np.ndarray] = {}
        for pct in levels:
            key = f"p{int(pct * 100):02d}"
            result[key] = np.percentile(paths, pct * 100, axis=0)
        result["mean"] = np.mean(paths, axis=0)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 6. Walk-Forward Validator
# ═══════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Rolling window train/test split for strategy validation.
    Detects overfitting by comparing in-sample vs out-of-sample performance.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def generate_splits(
        self, dates: "pd.DatetimeIndex"
    ) -> List[Tuple["pd.DatetimeIndex", "pd.DatetimeIndex"]]:
        """Generate rolling (train, test) date index pairs."""
        splits = []
        n = len(dates)
        start = 0
        while start + self.train_window + self.test_window <= n:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            train_idx = dates[start:train_end]
            test_idx = dates[train_end:test_end]
            splits.append((train_idx, test_idx))
            start += self.step_size
        return splits

    def compute_ic(
        self, signals: "pd.Series", forward_returns: "pd.Series"
    ) -> float:
        """
        Information Coefficient: rank correlation between
        signal and subsequent realized returns.
        """
        valid = pd.DataFrame({"sig": signals, "ret": forward_returns}).dropna()
        if len(valid) < 5:
            return 0.0
        return float(valid["sig"].corr(valid["ret"], method="spearman"))

    def rolling_ic(
        self,
        signals_df: "pd.DataFrame",
        returns_df: "pd.DataFrame",
        window: int = 63,
    ) -> "pd.Series":
        """
        Compute rolling IC across time for all tickers combined.
        At each date, flatten the cross-section and compute rank corr.
        """
        dates = signals_df.index.intersection(returns_df.index).sort_values()
        ic_values = []
        ic_dates = []

        for i in range(window, len(dates)):
            window_dates = dates[i - window:i]
            sig_flat = signals_df.loc[window_dates].stack()
            ret_flat = returns_df.shift(-1).loc[window_dates].stack()
            valid = pd.DataFrame({"s": sig_flat, "r": ret_flat}).dropna()
            if len(valid) > 10:
                ic = float(valid["s"].corr(valid["r"], method="spearman"))
            else:
                ic = np.nan
            ic_values.append(ic)
            ic_dates.append(dates[i])

        return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name="IC")

    def detect_overfitting(
        self,
        is_sharpe: float,
        oos_sharpe: float,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Compare in-sample vs out-of-sample Sharpe.
        Flag overfitting if degradation exceeds threshold.
        """
        if abs(is_sharpe) < 1e-9:
            degradation = 0.0
        else:
            degradation = 1.0 - (oos_sharpe / is_sharpe)

        return {
            "is_sharpe": round(is_sharpe, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "degradation": round(degradation, 4),
            "is_overfit": degradation > threshold,
            "threshold": threshold,
        }

    def validate_strategy(
        self,
        strategy_fn,
        prices: "pd.DataFrame",
        config: Optional[BacktestConfig] = None,
    ) -> Dict[str, Any]:
        """
        Full walk-forward validation:
        1. Split data into rolling train/test windows.
        2. Generate signals in-sample, evaluate out-of-sample.
        3. Aggregate IS / OOS Sharpe and IC.
        """
        if config is None:
            config = BacktestConfig()

        dates = prices.index.sort_values()
        splits = self.generate_splits(dates)
        if not splits:
            return {"error": "Insufficient data for walk-forward splits."}

        bt = SignalBacktester()
        is_sharpes: List[float] = []
        oos_sharpes: List[float] = []
        ics: List[float] = []

        for train_idx, test_idx in splits:
            train_prices = prices.loc[train_idx]
            test_prices = prices.loc[test_idx]

            # Generate signals from in-sample data
            try:
                signals = strategy_fn(train_prices)
            except Exception:
                continue

            # In-sample backtest
            is_result = bt.run(signals, train_prices, config)
            is_sharpes.append(is_result.sharpe)

            # Out-of-sample: apply last day of signals to OOS period
            oos_signals = pd.DataFrame(
                np.tile(signals.iloc[-1].values, (len(test_idx), 1)),
                index=test_idx,
                columns=signals.columns,
            )
            oos_result = bt.run(oos_signals, test_prices, config)
            oos_sharpes.append(oos_result.sharpe)

            # IC
            fwd_ret = test_prices.pct_change().iloc[1:].mean()
            last_signal = signals.iloc[-1]
            common = last_signal.index.intersection(fwd_ret.index)
            if len(common) > 3:
                ic = float(last_signal[common].corr(fwd_ret[common], method="spearman"))
                ics.append(ic)

        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        avg_ic = float(np.nanmean(ics)) if ics else 0.0

        overfit_check = self.detect_overfitting(avg_is, avg_oos)

        return {
            "n_splits": len(splits),
            "avg_is_sharpe": round(avg_is, 4),
            "avg_oos_sharpe": round(avg_oos, 4),
            "avg_ic": round(avg_ic, 4),
            "ic_series_length": len(ics),
            "overfit_analysis": overfit_check,
        }


# ═══════════════════════════════════════════════════════════════════════
# 7. Scenario Engine
# ═══════════════════════════════════════════════════════════════════════

class ScenarioEngine:
    """
    Historical scenario replay and custom stress testing.
    """

    # ── Well-known historical scenarios ────────────────────────────────
    HISTORICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
        "2008_gfc": {
            "name": "2008 Global Financial Crisis",
            "start": "2008-09-01",
            "end": "2009-03-31",
            "description": "Lehman collapse, credit freeze, equity drawdown ~55%",
            "shocks": {
                "SPY": -0.45, "XLF": -0.70, "XLK": -0.40,
                "XLE": -0.50, "XLI": -0.45, "XLY": -0.50,
                "XLP": -0.15, "XLU": -0.20, "XLV": -0.25,
                "XLB": -0.45, "XLRE": -0.55,
            },
        },
        "2020_covid": {
            "name": "2020 COVID Crash",
            "start": "2020-02-19",
            "end": "2020-03-23",
            "description": "Pandemic-driven sell-off, S&P 500 -34% in 23 trading days",
            "shocks": {
                "SPY": -0.34, "XLF": -0.40, "XLK": -0.28,
                "XLE": -0.55, "XLI": -0.38, "XLY": -0.38,
                "XLP": -0.18, "XLU": -0.25, "XLV": -0.25,
                "XLB": -0.35, "XLRE": -0.30, "XLC": -0.28,
            },
        },
        "2022_rate_shock": {
            "name": "2022 Rate Shock",
            "start": "2022-01-03",
            "end": "2022-10-12",
            "description": "Fed tightening cycle, S&P 500 -25%, duration hit",
            "shocks": {
                "SPY": -0.25, "XLF": -0.18, "XLK": -0.33,
                "XLE": 0.30, "XLI": -0.20, "XLY": -0.35,
                "XLP": -0.05, "XLU": -0.08, "XLV": -0.10,
                "XLB": -0.20, "XLRE": -0.30, "XLC": -0.38,
            },
        },
        "flash_crash_2010": {
            "name": "2010 Flash Crash",
            "start": "2010-05-06",
            "end": "2010-05-07",
            "description": "Intraday crash, S&P 500 -9% and recovery",
            "shocks": {
                "SPY": -0.06, "XLF": -0.08, "XLK": -0.07,
                "XLE": -0.06, "XLI": -0.07, "XLY": -0.07,
                "XLP": -0.04, "XLU": -0.03, "XLV": -0.05,
            },
        },
    }

    def __init__(self) -> None:
        pass

    def replay_historical(
        self,
        portfolio_weights: Dict[str, float],
        scenario_name: str,
    ) -> Dict[str, Any]:
        """
        Apply historical scenario shocks to a portfolio.

        Parameters
        ----------
        portfolio_weights : {ticker: weight} of the current portfolio
        scenario_name : key in HISTORICAL_SCENARIOS

        Returns dict with portfolio P&L and per-asset contributions.
        """
        if scenario_name not in self.HISTORICAL_SCENARIOS:
            available = list(self.HISTORICAL_SCENARIOS.keys())
            return {"error": f"Unknown scenario. Available: {available}"}

        scenario = self.HISTORICAL_SCENARIOS[scenario_name]
        shocks = scenario["shocks"]

        contributions: Dict[str, float] = {}
        total_pnl = 0.0

        for ticker, weight in portfolio_weights.items():
            shock = shocks.get(ticker, 0.0)
            contrib = weight * shock
            contributions[ticker] = round(contrib, 6)
            total_pnl += contrib

        return {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "period": f"{scenario['start']} to {scenario['end']}",
            "portfolio_return": round(total_pnl, 6),
            "contributions": contributions,
        }

    def custom_scenario(
        self,
        portfolio_weights: Dict[str, float],
        shocks: Dict[str, float],
        scenario_name: str = "Custom",
    ) -> Dict[str, Any]:
        """Apply arbitrary return shocks to a portfolio."""
        contributions: Dict[str, float] = {}
        total_pnl = 0.0

        for ticker, weight in portfolio_weights.items():
            shock = shocks.get(ticker, 0.0)
            contrib = weight * shock
            contributions[ticker] = round(contrib, 6)
            total_pnl += contrib

        return {
            "scenario": scenario_name,
            "portfolio_return": round(total_pnl, 6),
            "contributions": contributions,
        }

    def stress_test_all(
        self, portfolio_weights: Dict[str, float]
    ) -> "pd.DataFrame":
        """
        Run portfolio through every historical scenario and return
        a summary DataFrame.
        """
        rows = []
        for key in self.HISTORICAL_SCENARIOS:
            result = self.replay_historical(portfolio_weights, key)
            rows.append({
                "scenario": result.get("scenario", key),
                "period": result.get("period", ""),
                "portfolio_return": result.get("portfolio_return", 0.0),
            })
        return pd.DataFrame(rows).set_index("scenario")

    def worst_case(
        self, portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Identify the worst historical scenario for this portfolio."""
        worst_name = ""
        worst_pnl = 0.0
        for key in self.HISTORICAL_SCENARIOS:
            result = self.replay_historical(portfolio_weights, key)
            pnl = result.get("portfolio_return", 0.0)
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_name = key
        if worst_name:
            return self.replay_historical(portfolio_weights, worst_name)
        return {"scenario": "None", "portfolio_return": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# 8. BacktestEngine — Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Top-level API that wires together all backtesting components.
    Provides convenience methods for common backtest workflows.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self.strategy_bt = StrategyBacktester(self.config)
        self.mc = MonteCarloSimulator()
        self.wf = WalkForwardValidator()
        self.scenario = ScenarioEngine()
        self._last_result: Optional[BacktestResult] = None

    # ── Strategy Backtests ─────────────────────────────────────────────
    def run_momentum_backtest(
        self, prices: "pd.DataFrame", lookback: int = 60
    ) -> BacktestResult:
        """Run momentum strategy backtest."""
        result = self.strategy_bt.run_momentum(prices, lookback)
        self._last_result = result
        return result

    def run_rv_backtest(
        self,
        prices: "pd.DataFrame",
        pairs: Optional[List[Tuple[str, str]]] = None,
        lookback: int = 40,
    ) -> BacktestResult:
        """Run relative-value pair-trading backtest."""
        result = self.strategy_bt.run_relative_value(prices, pairs, lookback)
        self._last_result = result
        return result

    def run_sector_rotation_backtest(
        self, prices: "pd.DataFrame", lookback: int = 40, top_n: int = 3
    ) -> BacktestResult:
        """Run sector rotation backtest."""
        result = self.strategy_bt.run_sector_rotation(prices, lookback, top_n)
        self._last_result = result
        return result

    def run_mean_reversion_backtest(
        self, prices: "pd.DataFrame", lookback: int = 20, z_threshold: float = 1.5
    ) -> BacktestResult:
        """Run mean-reversion backtest."""
        result = self.strategy_bt.run_mean_reversion(prices, lookback, z_threshold)
        self._last_result = result
        return result

    # ── Monte Carlo ────────────────────────────────────────────────────
    def run_monte_carlo(
        self,
        returns: "pd.Series",
        n_paths: int = 1000,
        n_days: int = 252,
        initial_price: float = 100.0,
        method: str = "bootstrap",
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation and return summary statistics.

        Parameters
        ----------
        returns : historical daily return series
        method : 'bootstrap' or 'gbm'
        """
        if method == "bootstrap":
            paths = self.mc.bootstrap_paths(returns, n_paths, n_days, initial_price)
        else:
            mu = float(returns.mean()) * TRADING_DAYS_PER_YEAR
            sigma = float(returns.std()) * ANNUALIZATION_FACTOR
            paths = self.mc.gbm_paths(mu, sigma, n_paths, n_days, initial_price)

        terminal = paths[:, -1]
        var_stats = self.mc.estimate_var(terminal, confidence=0.95)
        ci = self.mc.confidence_intervals(paths)

        return {
            "method": method,
            "n_paths": n_paths,
            "n_days": n_days,
            "initial_price": initial_price,
            "terminal_stats": var_stats,
            "confidence_intervals": {k: v.tolist() for k, v in ci.items()},
        }

    # ── Scenario Analysis ──────────────────────────────────────────────
    def run_scenario_analysis(
        self,
        portfolio_weights: Dict[str, float],
        scenarios: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run portfolio through specified (or all) historical scenarios.
        """
        if scenarios is None:
            scenarios = list(ScenarioEngine.HISTORICAL_SCENARIOS.keys())

        results = []
        for s in scenarios:
            r = self.scenario.replay_historical(portfolio_weights, s)
            results.append(r)
        return results

    # ── Strategy Comparison ────────────────────────────────────────────
    def compare_strategies(
        self, prices: "pd.DataFrame"
    ) -> "pd.DataFrame":
        """
        Run all built-in strategies on the same price data and return
        a side-by-side comparison DataFrame.
        """
        strategies = {
            "Momentum": lambda p: self.strategy_bt.run_momentum(p),
            "Mean Reversion": lambda p: self.strategy_bt.run_mean_reversion(p),
            "Quality": lambda p: self.strategy_bt.run_quality(p),
            "Sector Rotation": lambda p: self.strategy_bt.run_sector_rotation(p),
        }

        # Only add RV if we have valid pairs in the price columns
        rv_pairs = [(a, b) for a, b in RV_PAIRS
                    if a in prices.columns and b in prices.columns]
        if rv_pairs:
            strategies["Relative Value"] = lambda p: self.strategy_bt.run_relative_value(p)

        rows = []
        for name, fn in strategies.items():
            try:
                result = fn(prices)
                rows.append({
                    "strategy": name,
                    "total_return": round(result.total_return, 4),
                    "ann_return": round(result.annualized_return, 4),
                    "ann_vol": round(result.annualized_vol, 4),
                    "sharpe": round(result.sharpe, 4),
                    "sortino": round(result.sortino, 4),
                    "max_drawdown": round(result.max_drawdown, 4),
                    "win_rate": round(result.win_rate, 4),
                    "total_trades": result.total_trades,
                })
            except Exception as exc:
                rows.append({
                    "strategy": name,
                    "total_return": None,
                    "ann_return": None,
                    "ann_vol": None,
                    "sharpe": None,
                    "sortino": None,
                    "max_drawdown": None,
                    "win_rate": None,
                    "total_trades": None,
                    "error": str(exc),
                })

        return pd.DataFrame(rows).set_index("strategy")

    # ── ASCII Report ───────────────────────────────────────────────────
    def get_backtest_report(
        self, result: Optional[BacktestResult] = None
    ) -> str:
        """
        Generate a human-readable ASCII summary of a backtest.
        Falls back to the last run result if none provided.
        """
        r = result or self._last_result
        if r is None:
            return "No backtest result available. Run a backtest first."

        sep = "=" * 62
        thin = "-" * 62

        lines = [
            sep,
            "  METADRON CAPITAL — BACKTEST REPORT",
            sep,
            "",
            "  Performance Summary",
            thin,
            f"  Total Return          : {r.total_return:>10.2%}",
            f"  Annualized Return     : {r.annualized_return:>10.2%}",
            f"  Annualized Volatility : {r.annualized_vol:>10.2%}",
            f"  Sharpe Ratio          : {r.sharpe:>10.4f}",
            f"  Sortino Ratio         : {r.sortino:>10.4f}",
            f"  Max Drawdown          : {r.max_drawdown:>10.2%}",
            "",
            "  Trade Statistics",
            thin,
            f"  Total Trades          : {r.total_trades:>10d}",
            f"  Win Rate              : {r.win_rate:>10.2%}",
            f"  Profit Factor         : {r.profit_factor:>10.4f}",
            f"  Avg Win               : {r.avg_win:>10.4%}",
            f"  Avg Loss              : {r.avg_loss:>10.4%}",
        ]

        if r.nav_series is not None and len(r.nav_series) > 0:
            lines += [
                "",
                "  NAV",
                thin,
                f"  Start NAV             : {r.nav_series.iloc[0]:>14,.2f}",
                f"  End NAV               : {r.nav_series.iloc[-1]:>14,.2f}",
                f"  Start Date            : {str(r.nav_series.index[0])[:10]:>14s}",
                f"  End Date              : {str(r.nav_series.index[-1])[:10]:>14s}",
                f"  Trading Days          : {len(r.nav_series):>14d}",
            ]

        if r.monthly_returns is not None and not r.monthly_returns.empty:
            lines += [
                "",
                "  Monthly Returns (%)",
                thin,
            ]
            fmt = r.monthly_returns.map(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "     "
            )
            lines.append(fmt.to_string())

        lines += ["", sep]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Module-level convenience (for quick scripting)
# ═══════════════════════════════════════════════════════════════════════

def quick_backtest(
    signals: "pd.DataFrame",
    prices: "pd.DataFrame",
    **config_kwargs,
) -> BacktestResult:
    """One-liner backtest: pass signals + prices, get result."""
    cfg = BacktestConfig(**config_kwargs)
    return SignalBacktester().run(signals, prices, cfg)


def quick_monte_carlo(
    returns: "pd.Series",
    n_paths: int = 1000,
    n_days: int = 252,
) -> Dict[str, Any]:
    """One-liner Monte Carlo: pass returns, get VaR summary."""
    engine = BacktestEngine()
    return engine.run_monte_carlo(returns, n_paths, n_days)


def quick_stress_test(
    portfolio_weights: Dict[str, float],
) -> "pd.DataFrame":
    """One-liner stress test across all historical scenarios."""
    return ScenarioEngine().stress_test_all(portfolio_weights)


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Metadron Capital — Backtester self-test")
    print("=" * 50)

    # Build synthetic data
    np.random.seed(42)
    n = 504  # ~2 years
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # Synthetic prices: correlated random walks
    daily_ret = np.random.randn(n, len(tickers)) * 0.015 + 0.0003
    cum_ret = np.cumprod(1 + daily_ret, axis=0)
    prices_df = pd.DataFrame(
        cum_ret * 100, index=dates, columns=tickers
    )

    # 1. Signal backtest
    print("\n[1] Signal Backtest (random signals)")
    rand_signals = pd.DataFrame(
        np.random.randn(n, len(tickers)), index=dates, columns=tickers
    )
    cfg = BacktestConfig(slippage_bps=5, commission_bps=1)
    result = quick_backtest(rand_signals, prices_df, slippage_bps=5)
    print(f"    Total Return : {result.total_return:+.2%}")
    print(f"    Sharpe       : {result.sharpe:.4f}")
    print(f"    Max Drawdown : {result.max_drawdown:.2%}")
    print(f"    Trades       : {result.total_trades}")

    # 2. Strategy comparison
    print("\n[2] Strategy Comparison")
    engine = BacktestEngine(cfg)
    mom_result = engine.run_momentum_backtest(prices_df, lookback=40)
    print(f"    Momentum Return : {mom_result.total_return:+.2%}")
    print(f"    Momentum Sharpe : {mom_result.sharpe:.4f}")

    mr_result = engine.run_mean_reversion_backtest(prices_df, lookback=20)
    print(f"    MeanRev Return  : {mr_result.total_return:+.2%}")
    print(f"    MeanRev Sharpe  : {mr_result.sharpe:.4f}")

    # 3. Monte Carlo
    print("\n[3] Monte Carlo (bootstrap, 500 paths)")
    rets = prices_df["AAPL"].pct_change().dropna()
    mc_result = engine.run_monte_carlo(rets, n_paths=500, n_days=252)
    term = mc_result["terminal_stats"]
    print(f"    VaR (95%)  : {term['VaR']:.4f}")
    print(f"    CVaR (95%) : {term['CVaR']:.4f}")
    print(f"    Mean Term  : {term['mean']:.2f}")

    # 4. Scenario analysis
    print("\n[4] Scenario Analysis")
    weights = {"SPY": 0.40, "XLK": 0.25, "XLF": 0.15, "XLE": 0.10, "XLP": 0.10}
    stress = quick_stress_test(weights)
    print(stress.to_string())

    # 5. Full report
    print("\n[5] Backtest Report")
    print(engine.get_backtest_report(mom_result))

    print("\nSelf-test complete.")
