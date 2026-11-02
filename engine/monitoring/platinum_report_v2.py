"""PlatinumReportV2Generator — Enhanced 30-section v2 Platinum Report.

Extends the original platinum_report.py with a 9-part, 30-section structure
covering market overview, sector analysis, alpha pipeline, risk dashboard,
signal summary, portfolio state, intelligence, performance, and session meta.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available — PlatinumReportV2 degraded")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not available — PlatinumReportV2 degraded")

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_WHITE = "\033[97m"
ANSI_MAGENTA = "\033[95m"

DIVIDER = f"{ANSI_CYAN}{'=' * 90}{ANSI_RESET}"
THIN_DIV = f"{ANSI_DIM}{'-' * 90}{ANSI_RESET}"


def _safe_get(d: dict, *keys, default="N/A") -> Any:
    """Safely traverse nested dict keys."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def _color_pct(val, default_str: str = "N/A") -> str:
    """Color-code a percentage value."""
    try:
        v = float(val)
        color = ANSI_GREEN if v > 0 else ANSI_RED if v < 0 else ANSI_YELLOW
        return f"{color}{v:+.2f}%{ANSI_RESET}"
    except (TypeError, ValueError):
        return default_str


def _color_num(val, fmt: str = ".2f", default_str: str = "N/A") -> str:
    """Color-code a numeric value."""
    try:
        v = float(val)
        color = ANSI_GREEN if v > 0 else ANSI_RED if v < 0 else ANSI_YELLOW
        return f"{color}{v:{fmt}}{ANSI_RESET}"
    except (TypeError, ValueError):
        return default_str


class PlatinumReportV2Generator:
    """Enhanced 30-section Platinum Report (v2).

    Organized into 9 parts covering every aspect of engine state.
    """

    def __init__(self):
        logger.info("PlatinumReportV2Generator initialized")

    def generate_report(self, engine_state: dict) -> str:
        """Generate the full 30-section v2 Platinum Report.

        Parameters
        ----------
        engine_state : dict
            Complete engine state dictionary containing all subsystem outputs.

        Returns
        -------
        str
            Formatted ASCII report string.
        """
        try:
            state = engine_state or {}
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
            sections = []

            # Title block
            sections.append(DIVIDER)
            sections.append(f"{ANSI_BOLD}{ANSI_MAGENTA}  PLATINUM REPORT v2  —  {ts}{ANSI_RESET}")
            sections.append(f"{ANSI_BOLD}{ANSI_MAGENTA}  Metadron Capital — 30-Section Intelligence Brief{ANSI_RESET}")
            sections.append(DIVIDER)

            # Part 1: Market Overview (sections 1-4)
            sections.append(self._part_header("PART 1: MARKET OVERVIEW"))
            sections.append(self._s01_regime(state))
            sections.append(self._s02_vix(state))
            sections.append(self._s03_rates(state))
            sections.append(self._s04_macro(state))

            # Part 2: Sector Analysis (sections 5-6)
            sections.append(self._part_header("PART 2: SECTOR ANALYSIS"))
            sections.append(self._s05_sectors(state))
            sections.append(self._s06_rotation(state))

            # Part 3: Alpha Pipeline (sections 7-9)
            sections.append(self._part_header("PART 3: ALPHA PIPELINE"))
            sections.append(self._s07_optimizer(state))
            sections.append(self._s08_quality(state))
            sections.append(self._s09_candidates(state))

            # Part 4: Risk Dashboard (sections 10-13)
            sections.append(self._part_header("PART 4: RISK DASHBOARD"))
            sections.append(self._s10_var(state))
            sections.append(self._s11_beta(state))
            sections.append(self._s12_drawdown(state))
            sections.append(self._s13_exposure(state))

            # Part 5: Signal Summary (sections 14-16)
            sections.append(self._part_header("PART 5: SIGNAL SUMMARY"))
            sections.append(self._s14_ensemble(state))
            sections.append(self._s15_conviction(state))
            sections.append(self._s16_signal_counts(state))

            # Part 6: Portfolio State (sections 17-20)
            sections.append(self._part_header("PART 6: PORTFOLIO STATE"))
            sections.append(self._s17_positions(state))
            sections.append(self._s18_pnl(state))
            sections.append(self._s19_allocation(state))
            sections.append(self._s20_trades(state))

            # Part 7: Intelligence (sections 21-24)
            sections.append(self._part_header("PART 7: INTELLIGENCE"))
            sections.append(self._s21_research_bots(state))
            sections.append(self._s22_sentiment(state))
            sections.append(self._s23_events(state))
            sections.append(self._s24_contagion(state))

            # Part 8: Performance (sections 25-28)
            sections.append(self._part_header("PART 8: PERFORMANCE"))
            sections.append(self._s25_sharpe(state))
            sections.append(self._s26_returns(state))
            sections.append(self._s27_win_rate(state))
            sections.append(self._s28_benchmarks(state))

            # Part 9: Session Meta (sections 29-30)
            sections.append(self._part_header("PART 9: SESSION META"))
            sections.append(self._s29_runtime(state))
            sections.append(self._s30_memory(state))

            sections.append(DIVIDER)
            sections.append(f"{ANSI_BOLD}  END PLATINUM REPORT v2{ANSI_RESET}")
            sections.append(DIVIDER)

            return "\n".join(sections)
        except Exception as e:
            logger.error("PlatinumReportV2 generation failed: %s", e)
            return f"[PlatinumReportV2] Report generation error: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _part_header(self, title: str) -> str:
        return f"\n{ANSI_BOLD}{ANSI_WHITE}  {title}{ANSI_RESET}\n{THIN_DIV}"

    def _section(self, num: int, title: str, body: str) -> str:
        return f"\n  {ANSI_BOLD}[{num:02d}] {title}{ANSI_RESET}\n{body}"

    # ------------------------------------------------------------------
    # Part 1: Market Overview
    # ------------------------------------------------------------------
    def _s01_regime(self, s: dict) -> str:
        regime = _safe_get(s, "regime", "label")
        confidence = _safe_get(s, "regime", "confidence")
        return self._section(1, "Market Regime", (
            f"    Regime: {ANSI_BOLD}{regime}{ANSI_RESET}  |  "
            f"Confidence: {_color_num(confidence, '.1f')}"
        ))

    def _s02_vix(self, s: dict) -> str:
        vix = _safe_get(s, "vix", "spot")
        term = _safe_get(s, "vix", "term_structure")
        return self._section(2, "VIX & Volatility", (
            f"    VIX Spot: {_color_num(vix, '.1f')}  |  "
            f"Term Structure: {term}"
        ))

    def _s03_rates(self, s: dict) -> str:
        fed = _safe_get(s, "rates", "fed_funds")
        t10y2y = _safe_get(s, "rates", "t10y2y")
        sofr = _safe_get(s, "rates", "sofr")
        return self._section(3, "Rates & Yield Curve", (
            f"    Fed Funds: {fed}  |  10Y-2Y: {_color_num(t10y2y, '.2f')}  |  "
            f"SOFR: {sofr}"
        ))

    def _s04_macro(self, s: dict) -> str:
        m2v = _safe_get(s, "macro", "m2_velocity")
        cpi = _safe_get(s, "macro", "cpi")
        gdp = _safe_get(s, "macro", "gdp_growth")
        return self._section(4, "Macro Indicators", (
            f"    M2 Velocity: {m2v}  |  CPI: {cpi}  |  GDP Growth: {gdp}"
        ))

    # ------------------------------------------------------------------
    # Part 2: Sector Analysis
    # ------------------------------------------------------------------
    def _s05_sectors(self, s: dict) -> str:
        sectors = _safe_get(s, "sectors", default={})
        if not isinstance(sectors, dict) or not sectors:
            return self._section(5, "GICS Sector Performance", "    No sector data available")
        lines = []
        for name, data in list(sectors.items())[:11]:
            chg = data.get("change", 0) if isinstance(data, dict) else data
            lines.append(f"    {name:<28} {_color_pct(chg)}")
        return self._section(5, "GICS Sector Performance", "\n".join(lines))

    def _s06_rotation(self, s: dict) -> str:
        signal = _safe_get(s, "rotation", "signal")
        from_s = _safe_get(s, "rotation", "from_sector")
        to_s = _safe_get(s, "rotation", "to_sector")
        return self._section(6, "Rotation Signals", (
            f"    Signal: {signal}  |  From: {from_s}  ->  To: {to_s}"
        ))

    # ------------------------------------------------------------------
    # Part 3: Alpha Pipeline
    # ------------------------------------------------------------------
    def _s07_optimizer(self, s: dict) -> str:
        alpha = _safe_get(s, "alpha", "annual_alpha")
        n_pos = _safe_get(s, "alpha", "num_positions")
        return self._section(7, "Alpha Optimizer Output", (
            f"    Annual Alpha: {_color_pct(alpha)}  |  Positions: {n_pos}"
        ))

    def _s08_quality(self, s: dict) -> str:
        dist = _safe_get(s, "alpha", "quality_distribution", default={})
        if not isinstance(dist, dict) or not dist:
            return self._section(8, "Quality Distribution", "    No quality data")
        lines = [f"    {k}: {v}" for k, v in dist.items()]
        return self._section(8, "Quality Distribution", "\n".join(lines))

    def _s09_candidates(self, s: dict) -> str:
        candidates = _safe_get(s, "alpha", "candidates", default=[])
        if not isinstance(candidates, list) or not candidates:
            return self._section(9, "Alpha Candidates", "    No candidates")
        lines = [f"    {c}" for c in candidates[:10]]
        return self._section(9, "Alpha Candidates (top 10)", "\n".join(lines))

    # ------------------------------------------------------------------
    # Part 4: Risk Dashboard
    # ------------------------------------------------------------------
    def _s10_var(self, s: dict) -> str:
        var95 = _safe_get(s, "risk", "var_95")
        var99 = _safe_get(s, "risk", "var_99")
        return self._section(10, "Value at Risk", (
            f"    VaR 95%: {_color_num(var95)}  |  VaR 99%: {_color_num(var99)}"
        ))

    def _s11_beta(self, s: dict) -> str:
        beta = _safe_get(s, "risk", "portfolio_beta")
        target = _safe_get(s, "risk", "target_beta")
        return self._section(11, "Beta Exposure", (
            f"    Portfolio Beta: {_color_num(beta)}  |  Target: {target}"
        ))

    def _s12_drawdown(self, s: dict) -> str:
        dd = _safe_get(s, "risk", "max_drawdown")
        current_dd = _safe_get(s, "risk", "current_drawdown")
        return self._section(12, "Drawdown", (
            f"    Max Drawdown: {_color_pct(dd)}  |  Current: {_color_pct(current_dd)}"
        ))

    def _s13_exposure(self, s: dict) -> str:
        gross = _safe_get(s, "risk", "gross_exposure")
        net = _safe_get(s, "risk", "net_exposure")
        return self._section(13, "Exposure Summary", (
            f"    Gross: {_color_pct(gross)}  |  Net: {_color_pct(net)}"
        ))

    # ------------------------------------------------------------------
    # Part 5: Signal Summary
    # ------------------------------------------------------------------
    def _s14_ensemble(self, s: dict) -> str:
        votes = _safe_get(s, "signals", "ensemble_votes", default={})
        if not isinstance(votes, dict) or not votes:
            return self._section(14, "Ensemble Votes", "    No ensemble data")
        lines = [f"    {k}: {_color_num(v, '.2f')}" for k, v in list(votes.items())[:10]]
        return self._section(14, "Ensemble Votes", "\n".join(lines))

    def _s15_conviction(self, s: dict) -> str:
        levels = _safe_get(s, "signals", "conviction_levels", default={})
        if not isinstance(levels, dict) or not levels:
            return self._section(15, "Conviction Levels", "    No conviction data")
        lines = [f"    {k}: {v}" for k, v in levels.items()]
        return self._section(15, "Conviction Levels", "\n".join(lines))

    def _s16_signal_counts(self, s: dict) -> str:
        counts = _safe_get(s, "signals", "counts", default={})
        if not isinstance(counts, dict) or not counts:
            total = _safe_get(s, "signals", "total_count")
            return self._section(16, "Signal Counts", f"    Total signals: {total}")
        lines = [f"    {k}: {v}" for k, v in counts.items()]
        return self._section(16, "Signal Counts", "\n".join(lines))

    # ------------------------------------------------------------------
    # Part 6: Portfolio State
    # ------------------------------------------------------------------
    def _s17_positions(self, s: dict) -> str:
        positions = _safe_get(s, "portfolio", "positions", default=[])
        if not isinstance(positions, list) or not positions:
            return self._section(17, "Open Positions", "    No positions")
        lines = []
        for pos in positions[:15]:
            if isinstance(pos, dict):
                t = pos.get("ticker", "???")
                w = pos.get("weight", 0)
                pnl = pos.get("pnl", 0)
                lines.append(f"    {t:<8} wt={_color_pct(w)}  PnL={_color_num(pnl, '+,.0f')}")
            else:
                lines.append(f"    {pos}")
        return self._section(17, "Open Positions", "\n".join(lines))

    def _s18_pnl(self, s: dict) -> str:
        daily = _safe_get(s, "portfolio", "daily_pnl")
        total = _safe_get(s, "portfolio", "total_pnl")
        return self._section(18, "P&L Summary", (
            f"    Daily PnL: {_color_num(daily, '+,.2f')}  |  "
            f"Total PnL: {_color_num(total, '+,.2f')}"
        ))

    def _s19_allocation(self, s: dict) -> str:
        alloc = _safe_get(s, "portfolio", "sleeve_allocation", default={})
        if not isinstance(alloc, dict) or not alloc:
            return self._section(19, "Sleeve Allocation", "    No allocation data")
        lines = [f"    {k}: {_color_pct(v)}" for k, v in alloc.items()]
        return self._section(19, "Sleeve Allocation", "\n".join(lines))

    def _s20_trades(self, s: dict) -> str:
        trades = _safe_get(s, "portfolio", "recent_trades", default=[])
        if not isinstance(trades, list) or not trades:
            return self._section(20, "Recent Trades", "    No recent trades")
        lines = [f"    {t}" for t in trades[:10]]
        return self._section(20, "Recent Trades", "\n".join(lines))

    # ------------------------------------------------------------------
    # Part 7: Intelligence
    # ------------------------------------------------------------------
    def _s21_research_bots(self, s: dict) -> str:
        bots = _safe_get(s, "intelligence", "research_bots", default={})
        if not isinstance(bots, dict) or not bots:
            return self._section(21, "Research Bots", "    No bot data")
        lines = [f"    {k}: {v}" for k, v in list(bots.items())[:11]]
        return self._section(21, "Research Bots", "\n".join(lines))

    def _s22_sentiment(self, s: dict) -> str:
        agg = _safe_get(s, "intelligence", "sentiment", "aggregate")
        social = _safe_get(s, "intelligence", "sentiment", "social_score")
        return self._section(22, "Sentiment Analysis", (
            f"    Aggregate: {_color_num(agg, '+.2f')}  |  Social: {_color_num(social, '+.2f')}"
        ))

    def _s23_events(self, s: dict) -> str:
        events = _safe_get(s, "intelligence", "events", default=[])
        if not isinstance(events, list) or not events:
            return self._section(23, "Upcoming Events", "    No events")
        lines = [f"    {e}" for e in events[:8]]
        return self._section(23, "Upcoming Events", "\n".join(lines))

    def _s24_contagion(self, s: dict) -> str:
        risk = _safe_get(s, "intelligence", "contagion_risk")
        nodes = _safe_get(s, "intelligence", "contagion_nodes")
        return self._section(24, "Contagion Risk", (
            f"    Risk Level: {_color_num(risk, '.2f')}  |  Active Nodes: {nodes}"
        ))

    # ------------------------------------------------------------------
    # Part 8: Performance
    # ------------------------------------------------------------------
    def _s25_sharpe(self, s: dict) -> str:
        sharpe = _safe_get(s, "performance", "sharpe")
        sortino = _safe_get(s, "performance", "sortino")
        calmar = _safe_get(s, "performance", "calmar")
        return self._section(25, "Risk-Adjusted Returns", (
            f"    Sharpe: {_color_num(sharpe)}  |  "
            f"Sortino: {_color_num(sortino)}  |  "
            f"Calmar: {_color_num(calmar)}"
        ))

    def _s26_returns(self, s: dict) -> str:
        daily = _safe_get(s, "performance", "daily_return")
        weekly = _safe_get(s, "performance", "weekly_return")
        total = _safe_get(s, "performance", "total_return")
        return self._section(26, "Return Summary", (
            f"    Daily: {_color_pct(daily)}  |  Weekly: {_color_pct(weekly)}  |  "
            f"Total: {_color_pct(total)}"
        ))

    def _s27_win_rate(self, s: dict) -> str:
        wr = _safe_get(s, "performance", "win_rate")
        avg_win = _safe_get(s, "performance", "avg_win")
        avg_loss = _safe_get(s, "performance", "avg_loss")
        return self._section(27, "Win Rate & Expectancy", (
            f"    Win Rate: {_color_pct(wr)}  |  "
            f"Avg Win: {_color_num(avg_win, '+,.2f')}  |  Avg Loss: {_color_num(avg_loss, '+,.2f')}"
        ))

    def _s28_benchmarks(self, s: dict) -> str:
        benchmarks = _safe_get(s, "performance", "vs_benchmarks", default={})
        if not isinstance(benchmarks, dict) or not benchmarks:
            return self._section(28, "vs Benchmarks", "    No benchmark data")
        lines = [f"    vs {k}: {_color_pct(v)}" for k, v in benchmarks.items()]
        return self._section(28, "vs Benchmarks", "\n".join(lines))

    # ------------------------------------------------------------------
    # Part 9: Session Meta
    # ------------------------------------------------------------------
    def _s29_runtime(self, s: dict) -> str:
        uptime = _safe_get(s, "session", "uptime")
        run_id = _safe_get(s, "session", "run_id")
        pipeline_ms = _safe_get(s, "session", "pipeline_ms")
        return self._section(29, "Runtime", (
            f"    Uptime: {uptime}  |  Run ID: {run_id}  |  Pipeline: {pipeline_ms}ms"
        ))

    def _s30_memory(self, s: dict) -> str:
        mem_mb = _safe_get(s, "session", "memory_mb")
        signals_total = _safe_get(s, "session", "signals_generated")
        trades_total = _safe_get(s, "session", "trades_executed")
        return self._section(30, "Memory & Counters", (
            f"    Memory: {mem_mb}MB  |  Signals Generated: {signals_total}  |  "
            f"Trades Executed: {trades_total}"
        ))
