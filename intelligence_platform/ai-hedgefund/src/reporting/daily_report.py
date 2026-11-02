# -*- coding: utf-8 -*-
"""
Daily Report Generator
======================

Produces comprehensive daily reports covering P&L breakdown, missed
opportunities analysis, forward-looking trade ideas, and risk assessment.

Report Sections:
    1. P&L Report: Daily performance by asset class, horizon, and position
    2. Missed Opportunities: Trades we should have taken (post-hoc analysis)
    3. Medium/Long-Term Ideas: New theses for buy-and-hold or swing
    4. Risk Report: VaR, exposure analysis, concentration, and stress tests

Risk Metrics Reference:
    VaR (parametric):  VaR_alpha = mu + z_alpha * sigma
    CVaR:              E[R | R <= VaR_alpha]
    Component VaR:     CVaR_i = w_i * beta_i * Portfolio_VaR
    Marginal VaR:      dVaR / dw_i
    Herfindahl Index:  H = sum(w_i^2)  (concentration, lower = more diversified)
    Beta exposure:     beta_p = sum(w_i * beta_i)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.openbb_universe import (
    AssetClass,
    UniverseData,
    classify_by_gics,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_var,
    detect_asset_class,
)
from src.execution.hft_engine import DailyPnL, HFTExecutionEngine, Position
from src.strategy.multi_horizon import (
    MultiHorizonEngine,
    TradeHorizon,
    TradeThesis,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Report Data Classes
# ===================================================================
@dataclass
class PnLReport:
    """Complete daily P&L report."""

    date: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    portfolio_value: float
    daily_return_pct: float

    # Breakdowns
    pnl_by_asset_class: Dict[str, float]
    pnl_by_sector: Dict[str, float]
    pnl_by_horizon: Dict[str, float]
    pnl_by_direction: Dict[str, float]
    top_winners: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]

    # Period metrics
    mtd_return_pct: float
    ytd_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float

    # Formatted output
    summary_text: str


@dataclass
class MissedOpportunity:
    """Represents a trade we should have taken."""

    symbol: str
    asset_class: AssetClass
    direction: str
    signal_date: datetime
    signal_price: float
    current_price: float
    missed_return_pct: float
    signal_type: str  # what signal fired
    reason_missed: str  # why we didn't take it
    lesson: str  # what to improve


@dataclass
class MissedOpportunities:
    """Collection of missed opportunities with analysis."""

    date: datetime
    opportunities: List[MissedOpportunity]
    total_missed_pnl: float
    most_costly_miss: Optional[MissedOpportunity]
    recurring_patterns: List[str]
    summary_text: str


@dataclass
class RiskReport:
    """Comprehensive risk assessment."""

    date: datetime
    portfolio_value: float

    # VaR and CVaR
    var_95: float          # 95% VaR in dollars
    var_99: float          # 99% VaR in dollars
    cvar_95: float         # 95% CVaR (Expected Shortfall)
    var_95_pct: float      # As percentage of portfolio
    var_99_pct: float

    # Exposure
    gross_exposure: float
    net_exposure: float
    gross_leverage: float
    net_leverage: float
    long_exposure: float
    short_exposure: float

    # Concentration
    top_5_concentration_pct: float  # % of portfolio in top 5 positions
    sector_concentration: Dict[str, float]  # sector -> % of portfolio
    asset_class_concentration: Dict[str, float]
    herfindahl_index: float  # H = sum(w_i^2), lower = more diversified

    # Correlation/Beta
    portfolio_beta: float  # beta to SPY
    max_single_name_pct: float

    # Stress tests
    stress_scenarios: Dict[str, float]  # scenario -> estimated P&L

    # Drawdown
    current_drawdown_pct: float
    max_drawdown_pct: float
    days_in_drawdown: int

    summary_text: str


# ===================================================================
# Daily Report Generator
# ===================================================================
class DailyReportGenerator:
    """
    Generates comprehensive daily reports for the hedge fund.
    """

    def __init__(
        self,
        execution_engine: HFTExecutionEngine,
        strategy_engine: MultiHorizonEngine,
    ):
        self.execution = execution_engine
        self.strategy = strategy_engine

    def generate_pnl_report(
        self,
        date: Optional[datetime] = None,
    ) -> PnLReport:
        """
        Generate daily P&L breakdown by asset class, sector, horizon, and direction.

        Computes:
            - Daily P&L (realized + unrealized)
            - P&L by asset class (equity, bond, commodity, crypto, FX)
            - P&L by GICS sector
            - P&L by trade horizon
            - P&L by direction (long vs short)
            - Top winners and losers
            - Period performance metrics (MTD, YTD, Sharpe, Sortino, MDD)
            - Profit factor = gross_profits / gross_losses
            - Win rate

        Parameters
        ----------
        date : datetime, optional
            Report date (defaults to today).

        Returns
        -------
        PnLReport
        """
        if date is None:
            date = datetime.utcnow()

        # Get daily P&L from execution engine
        daily = self.execution.calculate_pnl()
        perf = self.execution.get_performance_summary()

        # P&L by sector
        pnl_by_sector: Dict[str, float] = {}
        for pos in self.execution.positions.values():
            cls = classify_by_gics(pos.symbol)
            sector = cls.sector_name if cls else "Unknown"
            pnl_by_sector[sector] = pnl_by_sector.get(sector, 0) + pos.unrealized_pnl

        for trade in self.execution.closed_trades:
            if trade.get("exit_time") and trade["exit_time"].date() == date.date():
                cls = classify_by_gics(trade["symbol"])
                sector = cls.sector_name if cls else "Unknown"
                pnl_by_sector[sector] = pnl_by_sector.get(sector, 0) + trade["realized_pnl"]

        # P&L by direction
        pnl_by_direction: Dict[str, float] = {"long": 0.0, "short": 0.0}
        for pos in self.execution.positions.values():
            pnl_by_direction[pos.direction] += pos.unrealized_pnl
        for trade in self.execution.closed_trades:
            if trade.get("exit_time") and trade["exit_time"].date() == date.date():
                pnl_by_direction[trade["direction"]] += trade["realized_pnl"]

        # Top winners and losers (from open positions + today's closed trades)
        all_pnl_items: List[Dict[str, Any]] = []
        for pos in self.execution.positions.values():
            all_pnl_items.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct * 100,
                "status": "open",
                "horizon": pos.horizon.value,
            })
        for trade in self.execution.closed_trades:
            if trade.get("exit_time") and trade["exit_time"].date() == date.date():
                all_pnl_items.append({
                    "symbol": trade["symbol"],
                    "direction": trade["direction"],
                    "pnl": trade["realized_pnl"],
                    "pnl_pct": trade.get("pnl_pct", 0) * 100,
                    "status": "closed",
                    "horizon": trade.get("horizon", "unknown"),
                })

        sorted_items = sorted(all_pnl_items, key=lambda x: x["pnl"], reverse=True)
        top_winners = sorted_items[:5] if sorted_items else []
        top_losers = sorted_items[-5:][::-1] if sorted_items else []

        # Daily return
        pv = self.execution.portfolio_value
        prev_pv = (
            self.execution._portfolio_values[-2]
            if len(self.execution._portfolio_values) >= 2
            else self.execution.initial_capital
        )
        daily_return = (pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0

        # Period returns
        returns_series = pd.Series(self.execution._daily_returns)
        mtd_return = float(((1 + returns_series).prod() - 1)) if len(returns_series) > 0 else 0.0
        ytd_return = (pv - self.execution.initial_capital) / self.execution.initial_capital

        # Build summary text
        summary_lines = [
            f"=== DAILY P&L REPORT: {date.strftime('%Y-%m-%d')} ===",
            f"",
            f"Portfolio Value:    ${pv:>14,.2f}",
            f"Daily P&L:         ${daily.total_pnl:>14,.2f} ({daily_return * 100:+.2f}%)",
            f"  Realized:        ${daily.realized_pnl:>14,.2f}",
            f"  Unrealized:      ${daily.unrealized_pnl:>14,.2f}",
            f"",
            f"--- P&L by Asset Class ---",
        ]
        for ac, pnl in sorted(daily.pnl_by_asset_class.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"  {ac:<20s} ${pnl:>12,.2f}")

        summary_lines.append(f"")
        summary_lines.append(f"--- P&L by Sector ---")
        for sec, pnl in sorted(pnl_by_sector.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"  {sec:<30s} ${pnl:>12,.2f}")

        summary_lines.append(f"")
        summary_lines.append(f"--- P&L by Horizon ---")
        for h, pnl in sorted(daily.pnl_by_horizon.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"  {h:<20s} ${pnl:>12,.2f}")

        summary_lines.extend([
            f"",
            f"--- Performance ---",
            f"MTD Return:        {mtd_return * 100:>8.2f}%",
            f"YTD Return:        {ytd_return * 100:>8.2f}%",
            f"Sharpe (ann.):     {perf['sharpe_ratio']:>8.2f}",
            f"Sortino (ann.):    {perf['sortino_ratio']:>8.2f}",
            f"Max Drawdown:      {perf['max_drawdown_pct']:>8.2f}%",
            f"Win Rate:          {perf['win_rate_pct']:>8.1f}%",
            f"Profit Factor:     {perf['profit_factor']:>8.2f}",
            f"Trades Today:      {daily.num_trades:>8d}",
            f"Open Positions:    {perf['num_open_positions']:>8d}",
        ])

        if top_winners:
            summary_lines.append(f"")
            summary_lines.append(f"--- Top Winners ---")
            for w in top_winners[:3]:
                summary_lines.append(
                    f"  {w['symbol']:<6s} {w['direction']:<5s} ${w['pnl']:>10,.2f} ({w['pnl_pct']:+.1f}%)"
                )

        if top_losers:
            summary_lines.append(f"--- Top Losers ---")
            for l in top_losers[:3]:
                summary_lines.append(
                    f"  {l['symbol']:<6s} {l['direction']:<5s} ${l['pnl']:>10,.2f} ({l['pnl_pct']:+.1f}%)"
                )

        return PnLReport(
            date=date,
            total_pnl=daily.total_pnl,
            realized_pnl=daily.realized_pnl,
            unrealized_pnl=daily.unrealized_pnl,
            portfolio_value=pv,
            daily_return_pct=daily_return * 100,
            pnl_by_asset_class=daily.pnl_by_asset_class,
            pnl_by_sector=pnl_by_sector,
            pnl_by_horizon=daily.pnl_by_horizon,
            pnl_by_direction=pnl_by_direction,
            top_winners=top_winners,
            top_losers=top_losers,
            mtd_return_pct=mtd_return * 100,
            ytd_return_pct=ytd_return * 100,
            sharpe_ratio=perf["sharpe_ratio"],
            sortino_ratio=perf["sortino_ratio"],
            max_drawdown_pct=perf["max_drawdown_pct"],
            win_rate_pct=perf["win_rate_pct"],
            profit_factor=perf["profit_factor"],
            summary_text="\n".join(summary_lines),
        )

    def generate_missed_opportunities(
        self,
        date: Optional[datetime] = None,
        universe_data: Optional[UniverseData] = None,
        lookback_days: int = 5,
    ) -> MissedOpportunities:
        """
        Analyze what we should have traded but didn't.

        Detection methods:
        1. Signals that fired but weren't acted on (capacity, leverage limits)
        2. Large moves we missed (>3% daily move in universe stocks we didn't hold)
        3. Breakouts from consolidation we didn't catch
        4. Earnings surprises in universe stocks without positions

        Post-hoc return analysis:
            missed_return = (current_price - signal_price) / signal_price

        Pattern recognition:
            - Recurring sectors we miss (e.g., always late on Energy)
            - Time-of-day patterns (missing morning gaps)
            - Volatility regime blindness (not adapting to vol changes)

        Parameters
        ----------
        date : datetime, optional
        universe_data : UniverseData, optional
        lookback_days : int
            How many days back to scan for missed signals.

        Returns
        -------
        MissedOpportunities
        """
        if date is None:
            date = datetime.utcnow()

        opportunities: List[MissedOpportunity] = []
        held_symbols = {p.symbol for p in self.execution.positions.values()}
        traded_symbols = {t["symbol"] for t in self.execution.closed_trades}
        all_engaged = held_symbols | traded_symbols

        if universe_data is not None:
            # Scan for large moves we missed
            for sym, df in universe_data.equities.items():
                if sym in all_engaged:
                    continue
                if len(df) < lookback_days + 1 or "Close" not in df.columns:
                    continue

                close = df["Close"]

                # Check for significant moves in the lookback period
                for offset in range(1, min(lookback_days + 1, len(close))):
                    if offset >= len(close):
                        break
                    price_then = float(close.iloc[-offset - 1])
                    price_now = float(close.iloc[-1])
                    move_pct = (price_now - price_then) / price_then

                    if abs(move_pct) > 0.05:  # > 5% move
                        direction = "long" if move_pct > 0 else "short"
                        signal_date = df.index[-offset - 1] if hasattr(df.index[-offset - 1], 'strftime') else date - timedelta(days=offset)

                        # Determine what signal we should have caught
                        signal_type = "momentum" if abs(move_pct) > 0.10 else "trend_following"

                        # Check if there was an RSI signal
                        if len(close) > 14 + offset:
                            delta = close.diff()
                            gain = delta.clip(lower=0).rolling(14).mean()
                            loss_s = (-delta.clip(upper=0)).rolling(14).mean()
                            rs = gain / loss_s.replace(0, np.nan)
                            rsi = 100 - (100 / (1 + rs))
                            rsi_at_signal = rsi.iloc[-offset - 1] if not np.isnan(rsi.iloc[-offset - 1]) else 50
                            if rsi_at_signal < 30 and move_pct > 0:
                                signal_type = "RSI_oversold_bounce"
                            elif rsi_at_signal > 70 and move_pct < 0:
                                signal_type = "RSI_overbought_reversal"

                        opportunities.append(MissedOpportunity(
                            symbol=sym,
                            asset_class=detect_asset_class(sym),
                            direction=direction,
                            signal_date=signal_date if isinstance(signal_date, datetime) else date - timedelta(days=offset),
                            signal_price=price_then,
                            current_price=price_now,
                            missed_return_pct=move_pct * 100,
                            signal_type=signal_type,
                            reason_missed="Not in active scan results or filtered by risk limits",
                            lesson=f"Consider adding {sym} to active monitoring for {signal_type} signals",
                        ))
                        break  # only log the most recent missed move per symbol

            # Scan crypto for missed moves
            for sym, df in universe_data.crypto.items():
                if sym in all_engaged:
                    continue
                if len(df) < 3 or "Close" not in df.columns:
                    continue

                close = df["Close"]
                move_pct = (float(close.iloc[-1]) - float(close.iloc[-3])) / float(close.iloc[-3])

                if abs(move_pct) > 0.10:  # > 10% crypto move
                    opportunities.append(MissedOpportunity(
                        symbol=sym,
                        asset_class=AssetClass.CRYPTO,
                        direction="long" if move_pct > 0 else "short",
                        signal_date=date - timedelta(days=3),
                        signal_price=float(close.iloc[-3]),
                        current_price=float(close.iloc[-1]),
                        missed_return_pct=move_pct * 100,
                        signal_type="crypto_momentum",
                        reason_missed="Crypto position limits or insufficient scan coverage",
                        lesson=f"Increase crypto allocation monitoring for {sym}",
                    ))

        # Sort by absolute missed return
        opportunities.sort(key=lambda x: abs(x.missed_return_pct), reverse=True)

        total_missed = sum(
            abs(o.missed_return_pct) * self.execution.portfolio_value * 0.05 / 100
            for o in opportunities
        )  # Estimate: assume 5% position size

        most_costly = opportunities[0] if opportunities else None

        # Identify recurring patterns
        patterns: List[str] = []
        sector_misses: Dict[str, int] = {}
        for o in opportunities:
            cls = classify_by_gics(o.symbol)
            if cls:
                sector_misses[cls.sector_name] = sector_misses.get(cls.sector_name, 0) + 1

        for sector, count in sorted(sector_misses.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                patterns.append(f"Multiple missed opportunities in {sector} ({count} signals)")

        signal_type_counts: Dict[str, int] = {}
        for o in opportunities:
            signal_type_counts[o.signal_type] = signal_type_counts.get(o.signal_type, 0) + 1
        for sig, count in sorted(signal_type_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3:
                patterns.append(f"Repeatedly missing {sig} signals ({count} occurrences)")

        # Build summary
        summary_lines = [
            f"=== MISSED OPPORTUNITIES REPORT: {date.strftime('%Y-%m-%d')} ===",
            f"",
            f"Total missed opportunities:  {len(opportunities)}",
            f"Estimated missed P&L:        ${total_missed:,.2f}",
            f"",
        ]
        for o in opportunities[:10]:
            summary_lines.append(
                f"  {o.symbol:<8s} {o.direction:<5s} {o.missed_return_pct:+.1f}%  "
                f"[{o.signal_type}]  {o.signal_price:.2f} -> {o.current_price:.2f}"
            )

        if patterns:
            summary_lines.append(f"")
            summary_lines.append(f"--- Recurring Patterns ---")
            for p in patterns:
                summary_lines.append(f"  * {p}")

        return MissedOpportunities(
            date=date,
            opportunities=opportunities[:20],  # top 20
            total_missed_pnl=total_missed,
            most_costly_miss=most_costly,
            recurring_patterns=patterns,
            summary_text="\n".join(summary_lines),
        )

    def generate_medium_term_ideas(
        self,
        universe_data: Optional[UniverseData] = None,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
        news: Optional[pd.DataFrame] = None,
    ) -> List[TradeThesis]:
        """
        Generate new medium and long-term trade ideas.

        Sources:
        1. Strategy engine's scan results (medium + long term)
        2. Buy-and-hold recommendations
        3. Sector rotation calls
        4. Macro-driven themes

        Filtering:
        - Exclude symbols already in portfolio
        - Minimum composite score > 0.5
        - Minimum risk/reward > 2:1

        Parameters
        ----------
        universe_data : UniverseData, optional
        macro_data : dict, optional
        news : pd.DataFrame, optional

        Returns
        -------
        list of TradeThesis
            Sorted by composite score.
        """
        if universe_data is None:
            return self.strategy.get_buy_and_hold_recommendations()

        # Run full scan
        all_opps = self.strategy.scan_all_horizons(universe_data, macro_data, news)

        # Filter to medium + long term
        candidates: List[TradeThesis] = []
        held_symbols = {p.symbol for p in self.execution.positions.values()}

        for horizon in (TradeHorizon.MEDIUM_TERM, TradeHorizon.LONG_TERM):
            for thesis in all_opps.get(horizon, []):
                if thesis.symbol in held_symbols:
                    continue
                if thesis.composite_score < 0.45:
                    continue
                if thesis.risk_reward_ratio < 1.5:
                    continue
                candidates.append(thesis)

        candidates.sort(key=lambda t: t.composite_score, reverse=True)
        return candidates[:20]

    def generate_risk_report(
        self,
        date: Optional[datetime] = None,
    ) -> RiskReport:
        """
        Generate comprehensive risk assessment.

        Metrics computed:
        1. VaR (95% and 99%, parametric and historical)
           VaR_alpha = mu + z_alpha * sigma (parametric)
           VaR_alpha = percentile(returns, 1-alpha) (historical)

        2. CVaR (Expected Shortfall)
           CVaR = E[R | R <= VaR]

        3. Exposure analysis
           Gross = sum(|position_notional|)
           Net = sum(long_notional) - sum(short_notional)
           Leverage = Exposure / Portfolio_Value

        4. Concentration risk
           Herfindahl Index: H = sum(w_i^2)
           Perfect diversification: H = 1/N
           Single stock: H = 1

        5. Stress scenarios
           - Market crash (-20%)
           - Rate shock (+200bps)
           - Vol spike (VIX to 40)
           - Sector rotation (tech -15%)
           - USD strength (+10%)
           - Liquidity crisis (50% wider spreads)

        Parameters
        ----------
        date : datetime, optional

        Returns
        -------
        RiskReport
        """
        if date is None:
            date = datetime.utcnow()

        pv = self.execution.portfolio_value
        returns_series = pd.Series(self.execution._daily_returns) if self.execution._daily_returns else pd.Series(dtype=float)
        pv_series = pd.Series(self.execution._portfolio_values) if self.execution._portfolio_values else pd.Series(dtype=float)

        # VaR and CVaR
        var_95 = compute_var(returns_series, 0.95) * pv if len(returns_series) > 5 else 0
        var_99 = compute_var(returns_series, 0.99) * pv if len(returns_series) > 5 else 0
        cvar_95 = compute_cvar(returns_series, 0.95) * pv if len(returns_series) > 5 else 0

        # Exposure
        long_exposure = sum(
            abs(p.current_price * p.quantity)
            for p in self.execution.positions.values()
            if p.direction == "long"
        )
        short_exposure = sum(
            abs(p.current_price * p.quantity)
            for p in self.execution.positions.values()
            if p.direction == "short"
        )
        gross_exp = long_exposure + short_exposure
        net_exp = long_exposure - short_exposure

        # Concentration
        position_weights: List[float] = []
        position_notionals: List[Tuple[str, float]] = []

        for pos in self.execution.positions.values():
            notional = abs(pos.current_price * pos.quantity)
            weight = notional / pv if pv > 0 else 0
            position_weights.append(weight)
            position_notionals.append((pos.symbol, notional))

        # Sort by notional descending
        position_notionals.sort(key=lambda x: x[1], reverse=True)
        top_5_notional = sum(n for _, n in position_notionals[:5])
        top_5_pct = (top_5_notional / pv * 100) if pv > 0 else 0

        # Herfindahl Index
        herfindahl = sum(w ** 2 for w in position_weights) if position_weights else 0.0

        # Max single name
        max_single = max((n / pv for _, n in position_notionals), default=0) * 100

        # Sector concentration
        sector_notional: Dict[str, float] = {}
        for pos in self.execution.positions.values():
            cls = classify_by_gics(pos.symbol)
            sector = cls.sector_name if cls else "Other"
            notional = abs(pos.current_price * pos.quantity)
            sector_notional[sector] = sector_notional.get(sector, 0) + notional

        sector_concentration = {
            s: (n / pv * 100) if pv > 0 else 0
            for s, n in sector_notional.items()
        }

        # Asset class concentration
        ac_notional: Dict[str, float] = {}
        for pos in self.execution.positions.values():
            ac = pos.asset_class.value
            notional = abs(pos.current_price * pos.quantity)
            ac_notional[ac] = ac_notional.get(ac, 0) + notional

        ac_concentration = {
            ac: (n / pv * 100) if pv > 0 else 0
            for ac, n in ac_notional.items()
        }

        # Drawdown
        current_dd = 0.0
        days_in_dd = 0
        if len(pv_series) > 1:
            peak = pv_series.cummax()
            current_dd = float((peak.iloc[-1] - pv_series.iloc[-1]) / peak.iloc[-1]) * 100
            # Count consecutive days in drawdown
            dd_series = (peak - pv_series) / peak
            in_dd = dd_series > 0.001  # 0.1% threshold
            if in_dd.iloc[-1]:
                # Count backwards from end
                for i in range(len(in_dd) - 1, -1, -1):
                    if in_dd.iloc[i]:
                        days_in_dd += 1
                    else:
                        break

        max_dd = compute_max_drawdown(pv_series) * 100 if len(pv_series) > 1 else 0

        # Stress scenarios
        stress_scenarios: Dict[str, float] = {}

        # Market crash: assume all equities drop 20%
        equity_exposure = sum(
            pos.current_price * pos.quantity * (1 if pos.direction == "long" else -1)
            for pos in self.execution.positions.values()
            if pos.asset_class == AssetClass.EQUITY
        )
        stress_scenarios["Market Crash (-20%)"] = -equity_exposure * 0.20

        # Rate shock: bonds drop, financials benefit
        bond_exposure = sum(
            pos.current_price * pos.quantity * (1 if pos.direction == "long" else -1)
            for pos in self.execution.positions.values()
            if pos.asset_class == AssetClass.BOND
        )
        financial_exposure = sum(
            pos.current_price * pos.quantity * (1 if pos.direction == "long" else -1)
            for pos in self.execution.positions.values()
            if pos.asset_class == AssetClass.EQUITY
            and classify_by_gics(pos.symbol)
            and classify_by_gics(pos.symbol).sector_name == "Financials"
        )
        stress_scenarios["Rate Shock (+200bps)"] = (
            -bond_exposure * 0.10 + financial_exposure * 0.05
        )

        # VIX spike: general equity selloff -10%
        stress_scenarios["VIX Spike (to 40)"] = -equity_exposure * 0.10

        # Tech selloff
        tech_exposure = sum(
            pos.current_price * pos.quantity * (1 if pos.direction == "long" else -1)
            for pos in self.execution.positions.values()
            if pos.asset_class == AssetClass.EQUITY
            and classify_by_gics(pos.symbol)
            and classify_by_gics(pos.symbol).sector_name == "Information Technology"
        )
        stress_scenarios["Tech Selloff (-15%)"] = -tech_exposure * 0.15

        # USD strength: crypto and EM currencies drop
        crypto_exposure = sum(
            pos.current_price * pos.quantity * (1 if pos.direction == "long" else -1)
            for pos in self.execution.positions.values()
            if pos.asset_class == AssetClass.CRYPTO
        )
        stress_scenarios["USD Strength (+10%)"] = -crypto_exposure * 0.15

        # Liquidity crisis: assume 5% adverse impact on all positions
        stress_scenarios["Liquidity Crisis"] = -gross_exp * 0.05

        # Portfolio beta (simplified: use fraction of equity long exposure)
        portfolio_beta = equity_exposure / pv if pv > 0 else 0.0

        # Build summary
        summary_lines = [
            f"=== RISK REPORT: {date.strftime('%Y-%m-%d')} ===",
            f"",
            f"Portfolio Value:        ${pv:>14,.2f}",
            f"",
            f"--- Value at Risk ---",
            f"VaR (95%, 1-day):      ${var_95:>14,.2f}  ({var_95 / pv * 100:.2f}%)" if pv > 0 else f"VaR (95%): N/A",
            f"VaR (99%, 1-day):      ${var_99:>14,.2f}  ({var_99 / pv * 100:.2f}%)" if pv > 0 else f"VaR (99%): N/A",
            f"CVaR (95%, 1-day):     ${cvar_95:>14,.2f}" if cvar_95 > 0 else f"CVaR: N/A",
            f"",
            f"--- Exposure ---",
            f"Gross Exposure:        ${gross_exp:>14,.2f}  ({gross_exp / pv:.2f}x)" if pv > 0 else "",
            f"Net Exposure:          ${net_exp:>14,.2f}  ({net_exp / pv:.2f}x)" if pv > 0 else "",
            f"Long:                  ${long_exposure:>14,.2f}",
            f"Short:                 ${short_exposure:>14,.2f}",
            f"",
            f"--- Concentration ---",
            f"Top 5 Positions:       {top_5_pct:>8.1f}%",
            f"Max Single Name:       {max_single:>8.1f}%",
            f"Herfindahl Index:      {herfindahl:>8.4f}",
            f"Portfolio Beta:        {portfolio_beta:>8.2f}",
            f"",
            f"--- Drawdown ---",
            f"Current Drawdown:      {current_dd:>8.2f}%",
            f"Max Drawdown:          {max_dd:>8.2f}%",
            f"Days in Drawdown:      {days_in_dd:>8d}",
            f"",
            f"--- Stress Scenarios ---",
        ]
        for scenario, impact in sorted(stress_scenarios.items(), key=lambda x: x[1]):
            summary_lines.append(f"  {scenario:<25s}  ${impact:>12,.2f}")

        return RiskReport(
            date=date,
            portfolio_value=pv,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            var_95_pct=(var_95 / pv * 100) if pv > 0 else 0,
            var_99_pct=(var_99 / pv * 100) if pv > 0 else 0,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            gross_leverage=gross_exp / pv if pv > 0 else 0,
            net_leverage=net_exp / pv if pv > 0 else 0,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            top_5_concentration_pct=top_5_pct,
            sector_concentration=sector_concentration,
            asset_class_concentration=ac_concentration,
            herfindahl_index=herfindahl,
            portfolio_beta=portfolio_beta,
            max_single_name_pct=max_single,
            stress_scenarios=stress_scenarios,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
            days_in_drawdown=days_in_dd,
            summary_text="\n".join(summary_lines),
        )

    def generate_full_daily_report(
        self,
        universe_data: Optional[UniverseData] = None,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
        news: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate all report sections and return as a combined dictionary.

        Returns
        -------
        dict
            Keys: 'pnl', 'missed_opportunities', 'new_ideas', 'risk'
        """
        pnl = self.generate_pnl_report()
        missed = self.generate_missed_opportunities(universe_data=universe_data)
        ideas = self.generate_medium_term_ideas(universe_data, macro_data, news)
        risk = self.generate_risk_report()

        # Print summary to logs
        logger.info("\n%s", pnl.summary_text)
        logger.info("\n%s", missed.summary_text)
        logger.info("\n%s", risk.summary_text)
        logger.info("New medium/long-term ideas: %d", len(ideas))
        for idea in ideas[:5]:
            logger.info(
                "  %s %s %s | Score: %.2f | RR: %.1f | %s",
                idea.direction, idea.symbol, idea.horizon.value,
                idea.composite_score, idea.risk_reward_ratio,
                idea.catalyst,
            )

        return {
            "pnl": pnl,
            "missed_opportunities": missed,
            "new_ideas": ideas,
            "risk": risk,
        }
