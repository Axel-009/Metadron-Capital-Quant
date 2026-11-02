"""Daily reporting — open/close reports with sector heatmap.

Generates:
    - Morning open report (pre-market macro scan)
    - Evening close report (reconciliation)
    - GICS sector heatmap (5-bucket ANSI)
    - Platinum report (comprehensive daily summary)
    - Performance attribution (sector, factor, signal type)
    - Risk metrics summary (VaR, CVaR, max drawdown)
    - P&L decomposition (alpha vs beta contribution)
    - Trade summary with fill quality analysis
    - Agent performance summary
    - Regime analysis section
    - HTML export
"""

import json
import re
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np
import pandas as pd

from ..data.yahoo_data import get_adj_close, get_returns
from ..data.universe_engine import SECTOR_ETFS
from ..signals.macro_engine import MacroSnapshot


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"


# ---------------------------------------------------------------------------
# Heatmap buckets (original interface)
# ---------------------------------------------------------------------------
HEATMAP_BUCKETS = {
    "STRONG_UP": {"min": 0.02, "color": "\033[92m", "symbol": "██"},
    "UP":        {"min": 0.005, "color": "\033[32m", "symbol": "▓▓"},
    "FLAT":      {"min": -0.005, "color": "\033[33m", "symbol": "░░"},
    "DOWN":      {"min": -0.02, "color": "\033[31m", "symbol": "▒▒"},
    "STRONG_DOWN": {"min": -999, "color": "\033[91m", "symbol": "██"},
}


def get_bucket(change: float) -> tuple[str, str, str]:
    """Classify a return into a heatmap bucket."""
    for name, cfg in HEATMAP_BUCKETS.items():
        if change >= cfg["min"]:
            return name, cfg["color"], cfg["symbol"]
    return "STRONG_DOWN", "\033[91m", "██"


# ---------------------------------------------------------------------------
# Risk metric helpers
# ---------------------------------------------------------------------------
def _compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at given confidence level."""
    if len(returns) < 5:
        return 0.0
    sorted_rets = np.sort(returns)
    index = int((1 - confidence) * len(sorted_rets))
    index = max(0, min(index, len(sorted_rets) - 1))
    return float(sorted_rets[index])


def _compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at given confidence level."""
    if len(returns) < 5:
        return 0.0
    var = _compute_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a return series."""
    if len(returns) < 2:
        return 0.0
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    return float(np.min(drawdowns))


def _compute_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    if len(returns) < 21:
        return 0.0
    excess = returns - rf / 252.0
    mu = float(np.mean(excess)) * 252.0
    sigma = float(np.std(excess, ddof=1)) * np.sqrt(252.0)
    if sigma < 1e-10:
        return 0.0
    return mu / sigma


def _compute_sortino(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sortino ratio."""
    if len(returns) < 21:
        return 0.0
    excess = returns - rf / 252.0
    mu = float(np.mean(excess)) * 252.0
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    down_std = float(np.std(downside, ddof=1)) * np.sqrt(252.0)
    if down_std < 1e-10:
        return 0.0
    return mu / down_std


def _compute_calmar(returns: np.ndarray) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    if len(returns) < 21:
        return 0.0
    ann_ret = float(np.mean(returns)) * 252.0
    mdd = abs(_compute_max_drawdown(returns))
    if mdd < 1e-10:
        return 0.0
    return ann_ret / mdd


# ---------------------------------------------------------------------------
# Performance attribution data classes
# ---------------------------------------------------------------------------
@dataclass
class SectorAttribution:
    """Performance attribution for a single sector."""
    sector: str
    weight: float = 0.0
    contribution: float = 0.0
    benchmark_return: float = 0.0
    portfolio_return: float = 0.0
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0


@dataclass
class FactorAttribution:
    """Performance attribution for a single factor."""
    factor: str
    exposure: float = 0.0
    factor_return: float = 0.0
    contribution: float = 0.0


@dataclass
class SignalAttribution:
    """Performance attribution for a signal type."""
    signal_type: str
    count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    hit_rate: float = 0.0


@dataclass
class RiskMetricsSummary:
    """Comprehensive risk metrics snapshot."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    annualized_vol: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    beta_to_spy: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0


@dataclass
class PnLDecomposition:
    """P&L decomposition into alpha and beta components."""
    total_pnl: float = 0.0
    beta_contribution: float = 0.0
    alpha_contribution: float = 0.0
    residual: float = 0.0
    sector_contributions: dict = field(default_factory=dict)
    factor_contributions: dict = field(default_factory=dict)
    idiosyncratic: float = 0.0


@dataclass
class TradeSummary:
    """Trade execution quality summary."""
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: float = 0.0
    avg_fill_price: float = 0.0
    avg_slippage_bps: float = 0.0
    vwap_performance: float = 0.0
    fill_rate: float = 0.0
    avg_time_to_fill_ms: float = 0.0
    signal_type_breakdown: dict = field(default_factory=dict)
    best_trade: dict = field(default_factory=dict)
    worst_trade: dict = field(default_factory=dict)
    turnover: float = 0.0


@dataclass
class AgentPerformanceSummary:
    """Summary of agent performance for the day."""
    total_agents: int = 0
    active_agents: int = 0
    total_signals: int = 0
    actionable_signals: int = 0
    agent_scores: dict = field(default_factory=dict)
    tier_distribution: dict = field(default_factory=dict)
    top_performer: str = ""
    worst_performer: str = ""
    avg_accuracy: float = 0.0
    avg_sharpe: float = 0.0


@dataclass
class RegimeAnalysis:
    """Regime analysis section."""
    current_regime: str = "UNKNOWN"
    cube_regime: str = "RANGE"
    regime_confidence: float = 0.0
    regime_duration_days: int = 0
    transition_probability: float = 0.0
    vix_percentile: float = 0.0
    credit_spread_percentile: float = 0.0
    yield_curve_signal: str = "NEUTRAL"
    liquidity_state: str = "NORMAL"
    risk_state: float = 0.0
    flow_state: float = 0.0
    regime_history: list = field(default_factory=list)


@dataclass
class PlatinumReport:
    """Comprehensive daily summary — the Platinum report."""
    report_date: str = ""
    generated_at: str = ""
    regime_analysis: RegimeAnalysis = field(default_factory=RegimeAnalysis)
    risk_metrics: RiskMetricsSummary = field(default_factory=RiskMetricsSummary)
    pnl_decomposition: PnLDecomposition = field(default_factory=PnLDecomposition)
    sector_attribution: list = field(default_factory=list)
    factor_attribution: list = field(default_factory=list)
    signal_attribution: list = field(default_factory=list)
    trade_summary: TradeSummary = field(default_factory=TradeSummary)
    agent_summary: AgentPerformanceSummary = field(
        default_factory=AgentPerformanceSummary
    )
    portfolio_summary: dict = field(default_factory=dict)
    market_summary: dict = field(default_factory=dict)
    heatmap: str = ""
    alerts: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Report generation — original interface (preserved)
# ---------------------------------------------------------------------------
def generate_sector_heatmap(date: Optional[str] = None) -> str:
    """ANSI sector heatmap for today."""
    start = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    prices = get_adj_close(list(SECTOR_ETFS.values()), start=start)
    inv = {v: k for k, v in SECTOR_ETFS.items()}

    lines = []
    lines.append("=" * 60)
    lines.append("METADRON CAPITAL — SECTOR HEATMAP")
    lines.append("=" * 60)

    if prices.empty:
        lines.append("  No data available")
        return "\n".join(lines)

    for col in prices.columns:
        sector = inv.get(col, col)
        r = prices[col].dropna()
        if len(r) < 2:
            continue
        change = float(r.iloc[-1] / r.iloc[-2] - 1)
        bucket, color, symbol = get_bucket(change)
        lines.append(
            f"  {color}{symbol}{RESET} {sector:<30} "
            f"{change:>+7.2%}  [{bucket}]"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_open_report(macro: Optional[MacroSnapshot] = None) -> dict:
    """Morning pre-market report."""
    report = {
        "type": "OPEN",
        "timestamp": datetime.now().isoformat(),
        "regime": macro.regime.value if macro else "UNKNOWN",
        "vix": macro.vix if macro else 0,
        "spy_1m": macro.spy_return_1m if macro else 0,
        "sector_rankings": macro.sector_rankings if macro else {},
        "heatmap": generate_sector_heatmap(),
    }
    return report


def generate_close_report(
    portfolio_summary: dict,
    trades: list[dict] = None,
    macro: Optional[MacroSnapshot] = None,
) -> dict:
    """Evening reconciliation report."""
    report = {
        "type": "CLOSE",
        "timestamp": datetime.now().isoformat(),
        "regime": macro.regime.value if macro else "UNKNOWN",
        "portfolio": portfolio_summary,
        "trades_today": trades or [],
        "heatmap": generate_sector_heatmap(),
    }
    return report


def save_report(report: dict, log_dir: Path = Path("logs/reports")):
    """Save report to JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    rtype = report.get("type", "unknown").lower()
    path = log_dir / f"{date_str}_{rtype}.json"
    # Strip ANSI codes from heatmap for JSON
    clean = dict(report)
    if "heatmap" in clean:
        clean["heatmap"] = re.sub(r'\033\[[0-9;]*m', '', clean["heatmap"])
    path.write_text(json.dumps(clean, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Enhanced: Performance attribution
# ---------------------------------------------------------------------------
class PerformanceAttributor:
    """Brinson-Fachler style performance attribution engine.

    Decomposes portfolio returns into allocation, selection, and
    interaction effects across sectors, factors, and signal types.
    """

    def __init__(self):
        self._sector_etfs = SECTOR_ETFS
        self._factor_tickers = {
            "Momentum": "MTUM",
            "Value": "VLUE",
            "Quality": "QUAL",
            "Size": "SIZE",
            "Low_Vol": "USMV",
        }

    def compute_sector_attribution(
        self,
        portfolio_weights: dict[str, float],
        portfolio_returns: dict[str, float],
        benchmark_weights: Optional[dict[str, float]] = None,
        benchmark_returns: Optional[dict[str, float]] = None,
    ) -> list[SectorAttribution]:
        """Compute Brinson-Fachler sector attribution.

        Args:
            portfolio_weights: sector -> weight in portfolio
            portfolio_returns: sector -> return in period
            benchmark_weights: sector -> weight in benchmark
                               (equal-weight if None)
            benchmark_returns: sector -> benchmark return
                               (fetched if None)

        Returns:
            List of SectorAttribution for each sector.
        """
        sectors = list(self._sector_etfs.keys())

        # Default: equal-weight benchmark
        if benchmark_weights is None:
            n = len(sectors)
            benchmark_weights = {s: 1.0 / n for s in sectors}

        # Fetch benchmark returns if not provided
        if benchmark_returns is None:
            benchmark_returns = self._fetch_sector_returns()

        # Total benchmark return
        total_bm_return = sum(
            benchmark_weights.get(s, 0.0) * benchmark_returns.get(s, 0.0)
            for s in sectors
        )

        attributions = []
        for sector in sectors:
            wp = portfolio_weights.get(sector, 0.0)
            wb = benchmark_weights.get(sector, 0.0)
            rp = portfolio_returns.get(sector, 0.0)
            rb = benchmark_returns.get(sector, 0.0)

            # Brinson-Fachler decomposition
            allocation = (wp - wb) * (rb - total_bm_return)
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)

            attr = SectorAttribution(
                sector=sector,
                weight=wp,
                contribution=wp * rp,
                benchmark_return=rb,
                portfolio_return=rp,
                allocation_effect=allocation,
                selection_effect=selection,
                interaction_effect=interaction,
            )
            attributions.append(attr)

        # Sort by total contribution descending
        attributions.sort(key=lambda a: a.contribution, reverse=True)
        return attributions

    def compute_factor_attribution(
        self,
        portfolio_returns: np.ndarray,
        lookback_days: int = 60,
    ) -> list[FactorAttribution]:
        """Compute factor attribution using regression-based approach.

        Regresses portfolio returns against factor ETF returns to
        estimate factor exposures and contributions.
        """
        factor_attrs = []
        start = (
            pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)
        ).strftime("%Y-%m-%d")

        try:
            factor_tickers = list(self._factor_tickers.values())
            factor_rets_df = get_returns(factor_tickers, start=start)
            if factor_rets_df.empty:
                return self._default_factor_attribution()
        except Exception:
            return self._default_factor_attribution()

        n = min(len(portfolio_returns), len(factor_rets_df))
        if n < 10:
            return self._default_factor_attribution()

        y = portfolio_returns[-n:]
        X = factor_rets_df.iloc[-n:].values

        # OLS regression: y = X @ beta + epsilon
        try:
            XtX = X.T @ X
            reg = np.linalg.lstsq(
                XtX + 1e-8 * np.eye(XtX.shape[0]),
                X.T @ y,
                rcond=None,
            )
            betas = reg[0]
        except Exception:
            betas = np.zeros(len(self._factor_tickers))

        inv_map = {v: k for k, v in self._factor_tickers.items()}
        for i, ticker in enumerate(factor_tickers):
            factor_name = inv_map.get(ticker, ticker)
            factor_ret_mean = float(
                np.mean(factor_rets_df[ticker].iloc[-n:].values)
            )
            exposure = float(betas[i]) if i < len(betas) else 0.0
            contribution = exposure * factor_ret_mean * n

            factor_attrs.append(FactorAttribution(
                factor=factor_name,
                exposure=exposure,
                factor_return=factor_ret_mean * 252.0,
                contribution=contribution,
            ))

        factor_attrs.sort(
            key=lambda a: abs(a.contribution), reverse=True
        )
        return factor_attrs

    def compute_signal_attribution(
        self,
        trades: list[dict],
    ) -> list[SignalAttribution]:
        """Attribution by signal type from trade list."""
        signal_groups: dict[str, list] = {}
        for trade in trades:
            sig = trade.get(
                "signal", trade.get("signal_type", "UNKNOWN")
            )
            signal_groups.setdefault(sig, []).append(trade)

        results = []
        for sig_type, group in signal_groups.items():
            pnls = []
            for t in group:
                pnl = t.get("pnl", t.get("realized_pnl", 0.0))
                pnls.append(float(pnl))

            wins = sum(1 for p in pnls if p > 0)
            total = len(pnls)
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / total if total > 0 else 0.0
            hit_rate = wins / total if total > 0 else 0.0

            results.append(SignalAttribution(
                signal_type=sig_type,
                count=total,
                win_count=wins,
                total_pnl=total_pnl,
                avg_pnl=avg_pnl,
                hit_rate=hit_rate,
            ))

        results.sort(key=lambda a: a.total_pnl, reverse=True)
        return results

    def _fetch_sector_returns(
        self, days: int = 1
    ) -> dict[str, float]:
        """Fetch recent sector returns."""
        start = (
            pd.Timestamp.now() - pd.Timedelta(days=days + 5)
        ).strftime("%Y-%m-%d")
        inv = {v: k for k, v in self._sector_etfs.items()}
        result = {}
        try:
            prices = get_adj_close(
                list(self._sector_etfs.values()), start=start
            )
            if prices.empty:
                return {s: 0.0 for s in self._sector_etfs}
            for col in prices.columns:
                sector = inv.get(col, col)
                r = prices[col].dropna()
                if len(r) >= 2:
                    result[sector] = float(
                        r.iloc[-1] / r.iloc[-2] - 1
                    )
                else:
                    result[sector] = 0.0
        except Exception:
            return {s: 0.0 for s in self._sector_etfs}
        return result

    def _default_factor_attribution(self) -> list[FactorAttribution]:
        """Return empty factor attribution when data unavailable."""
        return [
            FactorAttribution(
                factor=name,
                exposure=0.0,
                factor_return=0.0,
                contribution=0.0,
            )
            for name in self._factor_tickers
        ]


# ---------------------------------------------------------------------------
# Enhanced: Risk metrics engine
# ---------------------------------------------------------------------------
class RiskMetricsEngine:
    """Compute comprehensive risk metrics for a portfolio."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def compute_full_metrics(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> RiskMetricsSummary:
        """Compute all risk metrics from portfolio return series.

        Args:
            portfolio_returns: array of daily portfolio returns
            benchmark_returns: optional benchmark returns for beta,
                               information ratio, etc.

        Returns:
            RiskMetricsSummary with all computed metrics.
        """
        metrics = RiskMetricsSummary()

        if len(portfolio_returns) < 5:
            return metrics

        r = np.asarray(portfolio_returns, dtype=float)
        r = r[np.isfinite(r)]

        if len(r) < 5:
            return metrics

        # VaR and CVaR
        metrics.var_95 = _compute_var(r, 0.95)
        metrics.var_99 = _compute_var(r, 0.99)
        metrics.cvar_95 = _compute_cvar(r, 0.95)
        metrics.cvar_99 = _compute_cvar(r, 0.99)

        # Drawdown
        metrics.max_drawdown = _compute_max_drawdown(r)
        cumulative = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cumulative)
        if peak[-1] > 0:
            metrics.current_drawdown = float(
                cumulative[-1] / peak[-1] - 1.0
            )

        # Ratios
        metrics.sharpe_ratio = _compute_sharpe(
            r, self.risk_free_rate
        )
        metrics.sortino_ratio = _compute_sortino(
            r, self.risk_free_rate
        )
        metrics.calmar_ratio = _compute_calmar(r)

        # Volatility
        metrics.annualized_vol = float(
            np.std(r, ddof=1) * np.sqrt(252.0)
        )

        # Higher moments
        if len(r) > 3:
            mean_r = np.mean(r)
            std_r = np.std(r, ddof=1)
            if std_r > 1e-10:
                metrics.skewness = float(
                    np.mean(((r - mean_r) / std_r) ** 3)
                )
                metrics.kurtosis = float(
                    np.mean(((r - mean_r) / std_r) ** 4) - 3.0
                )

        # Beta, tracking error, IR (require benchmark)
        if benchmark_returns is not None:
            bm = np.asarray(benchmark_returns, dtype=float)
            n = min(len(r), len(bm))
            if n >= 10:
                r_n = r[-n:]
                bm_n = bm[-n:]
                cov = np.cov(r_n, bm_n)
                if cov.shape == (2, 2) and cov[1, 1] > 1e-12:
                    metrics.beta_to_spy = float(
                        cov[0, 1] / cov[1, 1]
                    )
                te = r_n - bm_n
                metrics.tracking_error = float(
                    np.std(te, ddof=1) * np.sqrt(252.0)
                )
                if metrics.tracking_error > 1e-10:
                    excess = float(np.mean(te) * 252.0)
                    metrics.information_ratio = (
                        excess / metrics.tracking_error
                    )

        return metrics


# ---------------------------------------------------------------------------
# Enhanced: P&L decomposition engine
# ---------------------------------------------------------------------------
class PnLDecomposer:
    """Decompose portfolio P&L into alpha, beta, sector, factor."""

    def decompose(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        positions: Optional[dict] = None,
        nav: float = 1_000_000.0,
    ) -> PnLDecomposition:
        """Full P&L decomposition.

        Args:
            portfolio_returns: daily returns of portfolio
            benchmark_returns: daily returns of benchmark (SPY)
            positions: dict of ticker -> {qty, sector, pnl}
            nav: net asset value

        Returns:
            PnLDecomposition.
        """
        decomp = PnLDecomposition()

        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return decomp

        r = np.asarray(portfolio_returns, dtype=float)
        bm = np.asarray(benchmark_returns, dtype=float)
        n = min(len(r), len(bm))
        r = r[-n:]
        bm = bm[-n:]

        # Total P&L
        decomp.total_pnl = float(np.sum(r)) * nav

        # Beta via OLS
        cov = np.cov(r, bm)
        beta = 0.0
        if cov.shape == (2, 2) and cov[1, 1] > 1e-12:
            beta = float(cov[0, 1] / cov[1, 1])

        # Beta contribution: beta * benchmark_return * NAV
        decomp.beta_contribution = (
            beta * float(np.sum(bm)) * nav
        )

        # Alpha contribution: total - beta
        decomp.alpha_contribution = (
            decomp.total_pnl - decomp.beta_contribution
        )

        # Residual (should be near zero in simple decomposition)
        decomp.residual = 0.0

        # Sector contributions from positions
        if positions:
            sector_pnl: dict[str, float] = {}
            for ticker, pos in positions.items():
                sector = pos.get("sector", "Unknown")
                pnl = pos.get(
                    "pnl", pos.get("unrealized_pnl", 0.0)
                )
                sector_pnl[sector] = (
                    sector_pnl.get(sector, 0.0) + float(pnl)
                )
            decomp.sector_contributions = sector_pnl

        # Idiosyncratic: alpha minus sector contributions
        total_sector = sum(decomp.sector_contributions.values())
        decomp.idiosyncratic = (
            decomp.alpha_contribution - total_sector
        )

        return decomp


# ---------------------------------------------------------------------------
# Enhanced: Trade quality analyzer
# ---------------------------------------------------------------------------
class TradeQualityAnalyzer:
    """Analyze trade execution quality: slippage, VWAP, timing."""

    def analyze_trades(
        self,
        trades: list[dict],
        nav: float = 1_000_000.0,
    ) -> TradeSummary:
        """Build trade summary from a list of trade dicts.

        Expected keys per trade: ticker, side, qty, price,
        fill_price, signal, signal_type, pnl, timestamp.
        """
        summary = TradeSummary()

        if not trades:
            return summary

        summary.total_trades = len(trades)
        summary.buy_trades = sum(
            1
            for t in trades
            if t.get("side", "").upper() in ("BUY", "COVER")
        )
        summary.sell_trades = sum(
            1
            for t in trades
            if t.get("side", "").upper() in ("SELL", "SHORT")
        )

        total_volume = 0.0
        slippages = []
        prices = []
        signal_breakdown: dict[str, dict] = {}
        best_pnl = -math.inf
        worst_pnl = math.inf
        best_trade: dict = {}
        worst_trade: dict = {}

        for trade in trades:
            qty = abs(
                float(trade.get("qty", trade.get("quantity", 0)))
            )
            price = float(
                trade.get("price", trade.get("fill_price", 0))
            )
            total_volume += qty * price
            prices.append(price)

            # Slippage estimation
            fill = float(trade.get("fill_price", price))
            ref = float(trade.get("price", fill))
            if ref > 0:
                slip_bps = abs(fill - ref) / ref * 10_000
                slippages.append(slip_bps)

            # Signal breakdown
            sig = trade.get(
                "signal", trade.get("signal_type", "UNKNOWN")
            )
            if sig not in signal_breakdown:
                signal_breakdown[sig] = {
                    "count": 0,
                    "pnl": 0.0,
                    "volume": 0.0,
                }
            signal_breakdown[sig]["count"] += 1
            signal_breakdown[sig]["pnl"] += float(
                trade.get("pnl", 0)
            )
            signal_breakdown[sig]["volume"] += qty * price

            # Best / worst
            pnl = float(trade.get("pnl", 0))
            if pnl > best_pnl:
                best_pnl = pnl
                best_trade = trade
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_trade = trade

        summary.total_volume = total_volume
        summary.avg_fill_price = (
            float(np.mean(prices)) if prices else 0.0
        )
        summary.avg_slippage_bps = (
            float(np.mean(slippages)) if slippages else 0.0
        )
        summary.fill_rate = 1.0  # trade log always fills
        summary.signal_type_breakdown = signal_breakdown
        summary.best_trade = best_trade
        summary.worst_trade = worst_trade
        summary.turnover = (
            total_volume / nav if nav > 0 else 0.0
        )

        return summary


# ---------------------------------------------------------------------------
# Enhanced: Agent performance summarizer
# ---------------------------------------------------------------------------
class AgentPerformanceSummarizer:
    """Summarize agent (sector bot) performance for the daily report."""

    def summarize(
        self,
        agent_scores: Optional[dict] = None,
        signals: Optional[list] = None,
    ) -> AgentPerformanceSummary:
        """Build agent performance summary.

        Args:
            agent_scores: dict agent_name ->
                {accuracy, sharpe, hit_rate, tier, ...}
            signals: list of signal dicts from sector bots
        """
        summary = AgentPerformanceSummary()

        if agent_scores:
            summary.total_agents = len(agent_scores)
            summary.active_agents = sum(
                1
                for s in agent_scores.values()
                if s.get(
                    "total_signals", s.get("count", 0)
                )
                > 0
            )
            summary.agent_scores = dict(agent_scores)

            tiers: dict[str, int] = {}
            accuracies = []
            sharpes = []
            best_score = -math.inf
            worst_score = math.inf
            best_name = ""
            worst_name = ""

            for name, score in agent_scores.items():
                tier = score.get("tier", "UNKNOWN")
                tiers[tier] = tiers.get(tier, 0) + 1
                acc = score.get("accuracy", 0.0)
                sh = score.get("sharpe", 0.0)
                accuracies.append(acc)
                sharpes.append(sh)

                composite = score.get("composite", 0.0)
                if composite > best_score:
                    best_score = composite
                    best_name = name
                if composite < worst_score:
                    worst_score = composite
                    worst_name = name

            summary.tier_distribution = tiers
            summary.top_performer = best_name
            summary.worst_performer = worst_name
            summary.avg_accuracy = (
                float(np.mean(accuracies))
                if accuracies
                else 0.0
            )
            summary.avg_sharpe = (
                float(np.mean(sharpes)) if sharpes else 0.0
            )

        if signals:
            summary.total_signals = len(signals)
            summary.actionable_signals = sum(
                1
                for s in signals
                if s.get(
                    "direction", s.get("signal", "HOLD")
                )
                not in ("HOLD", "UNKNOWN")
            )

        return summary


# ---------------------------------------------------------------------------
# Enhanced: Regime analysis builder
# ---------------------------------------------------------------------------
class RegimeAnalyzer:
    """Build regime analysis section from macro snapshot."""

    _VIX_PERCENTILES = [
        (10.0, 5), (12.0, 15), (14.0, 30), (16.0, 45),
        (18.0, 55), (20.0, 65), (25.0, 80), (30.0, 90),
        (35.0, 95), (40.0, 98), (50.0, 99),
    ]

    def analyze(
        self,
        macro: Optional[MacroSnapshot] = None,
        regime_history: Optional[list] = None,
    ) -> RegimeAnalysis:
        """Build regime analysis from macro snapshot.

        Args:
            macro: current MacroSnapshot
            regime_history: list of {date, regime} dicts

        Returns:
            RegimeAnalysis dataclass.
        """
        analysis = RegimeAnalysis()

        if macro is None:
            return analysis

        analysis.current_regime = macro.regime.value
        analysis.cube_regime = macro.cube_regime.value
        analysis.risk_state = macro.vix / 100.0
        analysis.flow_state = macro.spy_return_1m

        # VIX percentile
        analysis.vix_percentile = self._vix_to_percentile(
            macro.vix
        )

        # Yield curve signal
        if macro.yield_spread > 0.5:
            analysis.yield_curve_signal = "STEEPENING"
        elif macro.yield_spread > 0:
            analysis.yield_curve_signal = "FLAT_POSITIVE"
        elif macro.yield_spread > -0.5:
            analysis.yield_curve_signal = "FLAT_NEGATIVE"
        else:
            analysis.yield_curve_signal = "INVERTED"

        # Liquidity state
        if macro.credit_spread < 2.0:
            analysis.liquidity_state = "ABUNDANT"
        elif macro.credit_spread < 4.0:
            analysis.liquidity_state = "NORMAL"
        elif macro.credit_spread < 6.0:
            analysis.liquidity_state = "TIGHTENING"
        else:
            analysis.liquidity_state = "STRESSED"

        # Regime confidence heuristic
        if macro.regime.value in ("BULL", "STRESS"):
            analysis.regime_confidence = 0.75
        else:
            analysis.regime_confidence = 0.55

        # Transition probability
        analysis.transition_probability = (
            1.0 - analysis.regime_confidence
        )

        # Regime history
        if regime_history:
            analysis.regime_history = regime_history[-30:]
            current = analysis.current_regime
            duration = 0
            for entry in reversed(regime_history):
                if entry.get("regime", "") == current:
                    duration += 1
                else:
                    break
            analysis.regime_duration_days = duration

        return analysis

    def _vix_to_percentile(self, vix: float) -> float:
        """Approximate VIX percentile from historical dist."""
        for threshold, pct in self._VIX_PERCENTILES:
            if vix <= threshold:
                return float(pct)
        return 99.5


# ---------------------------------------------------------------------------
# Platinum report generator
# ---------------------------------------------------------------------------
class PlatinumReportGenerator:
    """Generates the comprehensive Platinum daily report.

    Combines all monitoring subsystems into one unified report.
    """

    def __init__(self):
        self.attributor = PerformanceAttributor()
        self.risk_engine = RiskMetricsEngine()
        self.pnl_decomposer = PnLDecomposer()
        self.trade_analyzer = TradeQualityAnalyzer()
        self.agent_summarizer = AgentPerformanceSummarizer()
        self.regime_analyzer = RegimeAnalyzer()

    def generate(
        self,
        portfolio_summary: dict,
        trades: Optional[list[dict]] = None,
        macro: Optional[MacroSnapshot] = None,
        portfolio_returns: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        positions: Optional[dict] = None,
        agent_scores: Optional[dict] = None,
        agent_signals: Optional[list] = None,
        regime_history: Optional[list] = None,
        portfolio_weights: Optional[dict[str, float]] = None,
        portfolio_sector_returns: Optional[dict[str, float]] = None,
    ) -> PlatinumReport:
        """Generate the full Platinum report.

        Args:
            portfolio_summary: from broker.get_portfolio_summary()
            trades: list of trade dicts for today
            macro: current MacroSnapshot
            portfolio_returns: historical daily returns (numpy)
            benchmark_returns: SPY daily returns (numpy)
            positions: dict ticker -> {qty, sector, pnl, ...}
            agent_scores: dict agent_name -> score dict
            agent_signals: list of signal dicts
            regime_history: list of {date, regime} entries
            portfolio_weights: sector -> weight for attribution
            portfolio_sector_returns: sector -> return

        Returns:
            PlatinumReport dataclass.
        """
        report = PlatinumReport()
        report.report_date = datetime.now().strftime("%Y-%m-%d")
        report.generated_at = datetime.now().isoformat()
        report.portfolio_summary = portfolio_summary or {}

        trades = trades or []

        # 1. Regime analysis
        report.regime_analysis = self.regime_analyzer.analyze(
            macro, regime_history
        )

        # 2. Risk metrics
        if (
            portfolio_returns is not None
            and len(portfolio_returns) > 0
        ):
            report.risk_metrics = (
                self.risk_engine.compute_full_metrics(
                    portfolio_returns, benchmark_returns
                )
            )
        else:
            report.risk_metrics = RiskMetricsSummary()

        # 3. P&L decomposition
        if (
            portfolio_returns is not None
            and benchmark_returns is not None
        ):
            nav = portfolio_summary.get("nav", 1_000_000.0)
            report.pnl_decomposition = (
                self.pnl_decomposer.decompose(
                    portfolio_returns,
                    benchmark_returns,
                    positions,
                    nav,
                )
            )
        else:
            report.pnl_decomposition = PnLDecomposition()

        # 4. Sector attribution
        if portfolio_weights and portfolio_sector_returns:
            report.sector_attribution = [
                asdict(a)
                for a in self.attributor.compute_sector_attribution(
                    portfolio_weights, portfolio_sector_returns
                )
            ]
        else:
            report.sector_attribution = []

        # 5. Factor attribution
        if (
            portfolio_returns is not None
            and len(portfolio_returns) >= 20
        ):
            report.factor_attribution = [
                asdict(a)
                for a in self.attributor.compute_factor_attribution(
                    portfolio_returns
                )
            ]
        else:
            report.factor_attribution = []

        # 6. Signal attribution
        if trades:
            report.signal_attribution = [
                asdict(a)
                for a in self.attributor.compute_signal_attribution(
                    trades
                )
            ]

        # 7. Trade summary
        nav = portfolio_summary.get("nav", 1_000_000.0)
        report.trade_summary = (
            self.trade_analyzer.analyze_trades(trades, nav)
        )

        # 8. Agent summary
        report.agent_summary = self.agent_summarizer.summarize(
            agent_scores, agent_signals
        )

        # 9. Heatmap
        try:
            report.heatmap = generate_sector_heatmap()
        except Exception:
            report.heatmap = "[heatmap unavailable]"

        # 10. Alerts
        report.alerts = self._generate_alerts(report)

        # 11. Market summary
        report.market_summary = self._build_market_summary(macro)

        return report

    def _generate_alerts(
        self, report: PlatinumReport
    ) -> list[dict]:
        """Generate alerts from report data."""
        alerts = []

        rm = report.risk_metrics
        if rm.var_95 < -0.03:
            alerts.append({
                "level": "WARNING",
                "category": "RISK",
                "message": (
                    f"Daily VaR(95%) at {rm.var_95:.2%} "
                    f"— elevated risk"
                ),
            })

        if rm.max_drawdown < -0.10:
            alerts.append({
                "level": "CRITICAL",
                "category": "RISK",
                "message": (
                    f"Max drawdown at {rm.max_drawdown:.2%} "
                    f"— exceeds 10% threshold"
                ),
            })

        if rm.current_drawdown < -0.05:
            alerts.append({
                "level": "WARNING",
                "category": "RISK",
                "message": (
                    f"Current drawdown at "
                    f"{rm.current_drawdown:.2%}"
                ),
            })

        ra = report.regime_analysis
        if ra.current_regime in ("STRESS", "CRASH"):
            alerts.append({
                "level": "CRITICAL",
                "category": "REGIME",
                "message": (
                    f"Regime: {ra.current_regime} — "
                    f"defensive posture recommended"
                ),
            })

        if ra.vix_percentile > 90:
            alerts.append({
                "level": "WARNING",
                "category": "VOLATILITY",
                "message": (
                    f"VIX at {ra.vix_percentile:.0f}th "
                    f"percentile — extreme volatility"
                ),
            })

        if ra.liquidity_state == "STRESSED":
            alerts.append({
                "level": "WARNING",
                "category": "LIQUIDITY",
                "message": (
                    "Liquidity conditions stressed — "
                    "widen execution limits"
                ),
            })

        ts = report.trade_summary
        if ts.avg_slippage_bps > 5.0:
            alerts.append({
                "level": "INFO",
                "category": "EXECUTION",
                "message": (
                    f"Average slippage {ts.avg_slippage_bps:.1f} "
                    f"bps — review fill quality"
                ),
            })

        return alerts

    def _build_market_summary(
        self, macro: Optional[MacroSnapshot]
    ) -> dict:
        """Build market summary from macro data."""
        if macro is None:
            return {}
        return {
            "vix": macro.vix,
            "spy_1m": macro.spy_return_1m,
            "spy_3m": macro.spy_return_3m,
            "yield_10y": macro.yield_10y,
            "yield_2y": macro.yield_2y,
            "yield_spread": macro.yield_spread,
            "credit_spread": macro.credit_spread,
            "gold_momentum": macro.gold_momentum,
            "regime": macro.regime.value,
            "cube_regime": macro.cube_regime.value,
        }


# ---------------------------------------------------------------------------
# ASCII Platinum report formatter
# ---------------------------------------------------------------------------
def format_platinum_report(report: PlatinumReport) -> str:
    """Render PlatinumReport as a rich ASCII text report.

    Returns a multi-section ANSI-formatted string suitable for
    terminal display.
    """
    lines: list[str] = []
    W = 80

    def header(title: str):
        lines.append("")
        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")
        lines.append(f"{BOLD}{WHITE}  {title}{RESET}")
        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

    def subheader(title: str):
        lines.append(
            f"\n{BOLD}{YELLOW}--- {title} ---{RESET}"
        )

    def kv(key: str, value: Any, width: int = 30):
        lines.append(f"  {key:<{width}} {value}")

    def color_pnl(val: float, fmt: str = "+.2%") -> str:
        s = f"{val:{fmt}}"
        if val > 0:
            return f"{GREEN}{s}{RESET}"
        elif val < 0:
            return f"{RED}{s}{RESET}"
        return f"{YELLOW}{s}{RESET}"

    # Title
    header(
        f"METADRON CAPITAL — PLATINUM DAILY REPORT  "
        f"[{report.report_date}]"
    )
    lines.append(f"  Generated: {report.generated_at}")

    # Regime
    ra = report.regime_analysis
    subheader("REGIME ANALYSIS")
    if ra.current_regime == "BULL":
        regime_color = GREEN
    elif ra.current_regime in ("STRESS", "CRASH"):
        regime_color = RED
    else:
        regime_color = YELLOW
    kv(
        "Market Regime:",
        f"{regime_color}{ra.current_regime}{RESET}",
    )
    kv("Cube Regime:", ra.cube_regime)
    kv("Regime Confidence:", f"{ra.regime_confidence:.0%}")
    kv("Regime Duration:", f"{ra.regime_duration_days} days")
    kv("VIX Percentile:", f"{ra.vix_percentile:.0f}th")
    kv("Yield Curve:", ra.yield_curve_signal)
    kv("Liquidity State:", ra.liquidity_state)
    kv(
        "Transition Prob:",
        f"{ra.transition_probability:.1%}",
    )

    # Portfolio summary
    ps = report.portfolio_summary
    if ps:
        subheader("PORTFOLIO SNAPSHOT")
        kv("NAV:", f"${ps.get('nav', 0):,.2f}")
        kv("Cash:", f"${ps.get('cash', 0):,.2f}")
        kv(
            "Total P&L:",
            color_pnl(ps.get("total_pnl", 0), "+,.2f"),
        )
        kv("Positions:", ps.get("positions", 0))
        kv(
            "Gross Exposure:",
            f"{ps.get('gross_exposure', 0):.2%}",
        )
        kv(
            "Net Exposure:",
            f"{ps.get('net_exposure', 0):.2%}",
        )
        kv("Win Rate:", f"{ps.get('win_rate', 0):.1%}")

    # Risk metrics
    rm = report.risk_metrics
    subheader("RISK METRICS")
    kv("VaR (95%):", color_pnl(rm.var_95))
    kv("VaR (99%):", color_pnl(rm.var_99))
    kv("CVaR (95%):", color_pnl(rm.cvar_95))
    kv("CVaR (99%):", color_pnl(rm.cvar_99))
    kv("Max Drawdown:", color_pnl(rm.max_drawdown))
    kv("Current Drawdown:", color_pnl(rm.current_drawdown))
    kv("Annualized Vol:", f"{rm.annualized_vol:.2%}")
    kv("Sharpe Ratio:", f"{rm.sharpe_ratio:.3f}")
    kv("Sortino Ratio:", f"{rm.sortino_ratio:.3f}")
    kv("Calmar Ratio:", f"{rm.calmar_ratio:.3f}")
    kv("Beta to SPY:", f"{rm.beta_to_spy:.3f}")
    kv("Tracking Error:", f"{rm.tracking_error:.2%}")
    kv("Information Ratio:", f"{rm.information_ratio:.3f}")
    kv("Skewness:", f"{rm.skewness:.3f}")
    kv("Kurtosis:", f"{rm.kurtosis:.3f}")

    # P&L decomposition
    pnl = report.pnl_decomposition
    subheader("P&L DECOMPOSITION")
    kv("Total P&L:", color_pnl(pnl.total_pnl, "+,.2f"))
    kv(
        "Beta Contribution:",
        color_pnl(pnl.beta_contribution, "+,.2f"),
    )
    kv(
        "Alpha Contribution:",
        color_pnl(pnl.alpha_contribution, "+,.2f"),
    )
    kv(
        "Idiosyncratic:",
        color_pnl(pnl.idiosyncratic, "+,.2f"),
    )
    if pnl.sector_contributions:
        lines.append(
            f"\n  {DIM}Sector P&L Breakdown:{RESET}"
        )
        for sector, sc_pnl in sorted(
            pnl.sector_contributions.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            kv(
                f"    {sector}:",
                color_pnl(sc_pnl, "+,.2f"),
            )

    # Sector attribution
    if report.sector_attribution:
        subheader("SECTOR ATTRIBUTION (Brinson-Fachler)")
        lines.append(
            f"  {'Sector':<25} {'Weight':>7} {'Return':>8} "
            f"{'Alloc':>8} {'Select':>8} {'Inter':>8}"
        )
        lines.append(f"  {'-' * 64}")
        for attr in report.sector_attribution:
            s = attr.get("sector", "?")
            w = attr.get("weight", 0)
            r = attr.get("portfolio_return", 0)
            a = attr.get("allocation_effect", 0)
            se = attr.get("selection_effect", 0)
            ie = attr.get("interaction_effect", 0)
            lines.append(
                f"  {s:<25} {w:>6.1%} {r:>+7.2%} "
                f"{a:>+7.4f} {se:>+7.4f} {ie:>+7.4f}"
            )

    # Factor attribution
    if report.factor_attribution:
        subheader("FACTOR ATTRIBUTION")
        lines.append(
            f"  {'Factor':<18} {'Exposure':>10} "
            f"{'Fac Return':>12} {'Contribution':>14}"
        )
        lines.append(f"  {'-' * 54}")
        for attr in report.factor_attribution:
            f_name = attr.get("factor", "?")
            exp = attr.get("exposure", 0)
            fr = attr.get("factor_return", 0)
            c = attr.get("contribution", 0)
            lines.append(
                f"  {f_name:<18} {exp:>+9.3f} "
                f"{fr:>+11.2%} {c:>+13.4f}"
            )

    # Signal attribution
    if report.signal_attribution:
        subheader("SIGNAL ATTRIBUTION")
        lines.append(
            f"  {'Signal Type':<22} {'Count':>6} {'Wins':>6} "
            f"{'Hit Rate':>9} {'Total P&L':>12}"
        )
        lines.append(f"  {'-' * 55}")
        for attr in report.signal_attribution:
            sig = attr.get("signal_type", "?")
            cnt = attr.get("count", 0)
            wins = attr.get("win_count", 0)
            hr = attr.get("hit_rate", 0)
            tp = attr.get("total_pnl", 0)
            lines.append(
                f"  {sig:<22} {cnt:>6} {wins:>6} "
                f"{hr:>8.1%} "
                f"{color_pnl(tp, '+12,.2f')}"
            )

    # Trade summary
    ts = report.trade_summary
    subheader("TRADE SUMMARY")
    kv("Total Trades:", ts.total_trades)
    kv("Buy Trades:", ts.buy_trades)
    kv("Sell Trades:", ts.sell_trades)
    kv("Total Volume:", f"${ts.total_volume:,.2f}")
    kv("Avg Slippage:", f"{ts.avg_slippage_bps:.1f} bps")
    kv("Fill Rate:", f"{ts.fill_rate:.0%}")
    kv("Turnover:", f"{ts.turnover:.2%}")

    # Agent summary
    ag = report.agent_summary
    subheader("AGENT PERFORMANCE")
    kv("Total Agents:", ag.total_agents)
    kv("Active Agents:", ag.active_agents)
    kv("Total Signals:", ag.total_signals)
    kv("Actionable Signals:", ag.actionable_signals)
    kv("Avg Accuracy:", f"{ag.avg_accuracy:.1%}")
    kv("Avg Sharpe:", f"{ag.avg_sharpe:.3f}")
    kv("Top Performer:", ag.top_performer or "N/A")
    kv("Worst Performer:", ag.worst_performer or "N/A")
    if ag.tier_distribution:
        lines.append(
            f"\n  {DIM}Tier Distribution:{RESET}"
        )
        for tier, count in sorted(
            ag.tier_distribution.items()
        ):
            kv(f"    {tier}:", count)

    # Alerts
    if report.alerts:
        subheader("ALERTS")
        for alert in report.alerts:
            level = alert.get("level", "INFO")
            cat = alert.get("category", "")
            msg = alert.get("message", "")
            if level == "CRITICAL":
                lines.append(
                    f"  {BG_RED}{WHITE} CRITICAL {RESET} "
                    f"[{cat}] {msg}"
                )
            elif level == "WARNING":
                lines.append(
                    f"  {BRIGHT_YELLOW} WARNING  {RESET} "
                    f"[{cat}] {msg}"
                )
            else:
                lines.append(
                    f"  {DIM} INFO     {RESET} "
                    f"[{cat}] {msg}"
                )

    # Heatmap
    if report.heatmap:
        subheader("SECTOR HEATMAP")
        lines.append(report.heatmap)

    # Footer
    lines.append("")
    lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")
    lines.append(
        f"{DIM}  End of Platinum Report "
        f"— Metadron Capital{RESET}"
    )
    lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------
def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


def export_platinum_html(
    report: PlatinumReport,
    output_path: Optional[Path] = None,
) -> str:
    """Export PlatinumReport to an HTML file.

    Args:
        report: PlatinumReport to export
        output_path: where to write HTML; defaults to logs/reports/

    Returns:
        Path string to the generated HTML file.
    """
    if output_path is None:
        output_dir = Path("logs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"platinum_{report.report_date}.html"
        )

    ra = report.regime_analysis
    rm = report.risk_metrics
    pnl = report.pnl_decomposition
    ts = report.trade_summary
    ag = report.agent_summary
    ps = report.portfolio_summary

    # Build sector attribution rows
    sector_rows = ""
    for attr in report.sector_attribution:
        sector_rows += (
            "<tr>"
            f"<td>{attr.get('sector', '')}</td>"
            f"<td>{attr.get('weight', 0):.1%}</td>"
            f"<td>{attr.get('portfolio_return', 0):+.2%}</td>"
            f"<td>{attr.get('allocation_effect', 0):+.4f}</td>"
            f"<td>{attr.get('selection_effect', 0):+.4f}</td>"
            f"<td>{attr.get('interaction_effect', 0):+.4f}</td>"
            "</tr>\n"
        )

    factor_rows = ""
    for attr in report.factor_attribution:
        factor_rows += (
            "<tr>"
            f"<td>{attr.get('factor', '')}</td>"
            f"<td>{attr.get('exposure', 0):+.3f}</td>"
            f"<td>{attr.get('factor_return', 0):+.2%}</td>"
            f"<td>{attr.get('contribution', 0):+.4f}</td>"
            "</tr>\n"
        )

    signal_rows = ""
    for attr in report.signal_attribution:
        signal_rows += (
            "<tr>"
            f"<td>{attr.get('signal_type', '')}</td>"
            f"<td>{attr.get('count', 0)}</td>"
            f"<td>{attr.get('win_count', 0)}</td>"
            f"<td>{attr.get('hit_rate', 0):.1%}</td>"
            f"<td>{attr.get('total_pnl', 0):+,.2f}</td>"
            "</tr>\n"
        )

    alert_rows = ""
    for alert in report.alerts:
        level = alert.get("level", "INFO")
        cls = (
            "critical"
            if level == "CRITICAL"
            else ("warning" if level == "WARNING" else "info")
        )
        alert_rows += (
            f'<div class="alert {cls}">'
            f"<strong>[{level}]</strong> "
            f"[{alert.get('category', '')}] "
            f"{alert.get('message', '')}"
            "</div>\n"
        )

    if ra.current_regime == "BULL":
        regime_color = "#2ecc71"
    elif ra.current_regime in ("STRESS", "CRASH"):
        regime_color = "#e74c3c"
    else:
        regime_color = "#f39c12"

    nav_val = ps.get("nav", 0)
    cash_val = ps.get("cash", 0)
    pnl_val = ps.get("total_pnl", 0)
    pnl_cls = "positive" if pnl_val >= 0 else "negative"
    pos_count = ps.get("positions", 0)
    gross_exp = ps.get("gross_exposure", 0)
    net_exp = ps.get("net_exposure", 0)
    total_cls = "positive" if pnl.total_pnl >= 0 else "negative"
    beta_cls = (
        "positive"
        if pnl.beta_contribution >= 0
        else "negative"
    )
    alpha_cls = (
        "positive"
        if pnl.alpha_contribution >= 0
        else "negative"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Metadron Capital Platinum Report {report.report_date}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0;
       margin: 0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #00d4ff;
     border-bottom: 2px solid #00d4ff;
     padding-bottom: 10px; }}
h2 {{ color: #f39c12; margin-top: 30px; }}
.card {{ background: #16213e; border-radius: 8px;
         padding: 20px; margin: 15px 0; }}
.metric {{ display: inline-block; width: 180px;
           margin: 10px; text-align: center; }}
.metric .value {{ font-size: 22px; font-weight: bold; }}
.metric .label {{ font-size: 12px; color: #888; }}
.positive {{ color: #2ecc71; }}
.negative {{ color: #e74c3c; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left;
          border-bottom: 1px solid #2a3a5e; }}
th {{ background: #0f3460; color: #00d4ff; }}
.alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
.alert.critical {{ background: #5e1a1a;
                   border-left: 4px solid #e74c3c; }}
.alert.warning {{ background: #5e4a1a;
                  border-left: 4px solid #f39c12; }}
.alert.info {{ background: #1a3a5e;
               border-left: 4px solid #3498db; }}
.regime-badge {{ display: inline-block; padding: 5px 15px;
                 border-radius: 15px; font-weight: bold;
                 color: white;
                 background: {regime_color}; }}
pre {{ background: #0a0a1a; padding: 15px;
       border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
<h1>Metadron Capital — Platinum Report</h1>
<p>Date: {report.report_date}</p>

<div class="card">
<h2>Regime</h2>
<span class="regime-badge">{ra.current_regime}</span>
 Cube: <strong>{ra.cube_regime}</strong>
<div class="metric">
  <div class="value">{ra.regime_confidence:.0%}</div>
  <div class="label">Confidence</div></div>
<div class="metric">
  <div class="value">{ra.vix_percentile:.0f}th</div>
  <div class="label">VIX Pctile</div></div>
<div class="metric">
  <div class="value">{ra.yield_curve_signal}</div>
  <div class="label">Yield Curve</div></div>
</div>

<div class="card">
<h2>Portfolio</h2>
<div class="metric">
  <div class="value">${nav_val:,.0f}</div>
  <div class="label">NAV</div></div>
<div class="metric">
  <div class="value">${cash_val:,.0f}</div>
  <div class="label">Cash</div></div>
<div class="metric">
  <div class="value {pnl_cls}">${pnl_val:+,.0f}</div>
  <div class="label">P&L</div></div>
<div class="metric">
  <div class="value">{pos_count}</div>
  <div class="label">Positions</div></div>
<div class="metric">
  <div class="value">{gross_exp:.1%}</div>
  <div class="label">Gross</div></div>
<div class="metric">
  <div class="value">{net_exp:.1%}</div>
  <div class="label">Net</div></div>
</div>

<div class="card">
<h2>Risk</h2>
<table>
<tr><th>Metric</th><th>Value</th>
    <th>Metric</th><th>Value</th></tr>
<tr><td>VaR 95</td><td>{rm.var_95:.2%}</td>
    <td>VaR 99</td><td>{rm.var_99:.2%}</td></tr>
<tr><td>Max DD</td><td>{rm.max_drawdown:.2%}</td>
    <td>Sharpe</td><td>{rm.sharpe_ratio:.3f}</td></tr>
<tr><td>Sortino</td><td>{rm.sortino_ratio:.3f}</td>
    <td>Vol</td><td>{rm.annualized_vol:.2%}</td></tr>
<tr><td>Beta</td><td>{rm.beta_to_spy:.3f}</td>
    <td>IR</td><td>{rm.information_ratio:.3f}</td></tr>
</table>
</div>

<div class="card">
<h2>P&L Decomposition</h2>
<div class="metric">
  <div class="value {total_cls}">${pnl.total_pnl:+,.0f}</div>
  <div class="label">Total</div></div>
<div class="metric">
  <div class="value {beta_cls}">${pnl.beta_contribution:+,.0f}</div>
  <div class="label">Beta</div></div>
<div class="metric">
  <div class="value {alpha_cls}">${pnl.alpha_contribution:+,.0f}</div>
  <div class="label">Alpha</div></div>
</div>

<div class="card">
<h2>Sector Attribution</h2>
<table>
<tr><th>Sector</th><th>Weight</th><th>Return</th>
    <th>Alloc</th><th>Select</th><th>Inter</th></tr>
{sector_rows}
</table>
</div>

<div class="card">
<h2>Factor Attribution</h2>
<table>
<tr><th>Factor</th><th>Exposure</th>
    <th>Return</th><th>Contribution</th></tr>
{factor_rows}
</table>
</div>

<div class="card">
<h2>Signal Attribution</h2>
<table>
<tr><th>Signal</th><th>Count</th><th>Wins</th>
    <th>Hit Rate</th><th>P&L</th></tr>
{signal_rows}
</table>
</div>

<div class="card">
<h2>Trades</h2>
<table>
<tr><td>Total</td><td>{ts.total_trades}</td>
    <td>Volume</td><td>${ts.total_volume:,.0f}</td></tr>
<tr><td>Slippage</td><td>{ts.avg_slippage_bps:.1f}bps</td>
    <td>Turnover</td><td>{ts.turnover:.2%}</td></tr>
</table>
</div>

<div class="card">
<h2>Alerts</h2>
{alert_rows if alert_rows else '<p style="color:#888">None</p>'}
</div>

<div class="card">
<h2>Heatmap</h2>
<pre>{_strip_ansi(report.heatmap)}</pre>
</div>

<footer style="text-align:center;color:#555;margin-top:30px">
Metadron Capital | {report.generated_at}
</footer>
</div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    return str(output_path)


# ---------------------------------------------------------------------------
# Convenience: save Platinum report to JSON
# ---------------------------------------------------------------------------
def save_platinum_report(
    report: PlatinumReport,
    log_dir: Path = Path("logs/reports"),
) -> Path:
    """Serialize PlatinumReport to JSON, stripping ANSI codes."""
    log_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    if "heatmap" in data:
        data["heatmap"] = _strip_ansi(data["heatmap"])
    path = log_dir / f"platinum_{report.report_date}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
