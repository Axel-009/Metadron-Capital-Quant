"""L7 Execution Surface Dashboard — Risk + TCA panels for live dashboard.

Provides two dashboard panels that integrate with the existing LiveDashboard:
    1. L7 Risk Panel: real-time risk state, gate status, kill switch, VaR
    2. L7 TCA Panel: transaction cost analysis, cost decomposition, trends

Both panels support Rich (rich library) and ASCII fallback rendering.

Usage:
    from engine.monitoring.l7_dashboard import L7DashboardRenderer
    renderer = L7DashboardRenderer()
    risk_text = renderer.render_risk_panel(risk_state)
    tca_text = renderer.render_tca_panel(tca_aggregate, tca_history)
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Any

try:
    from ..execution.l7_unified_execution_surface import (
        RiskState, TCAAggregate, TCASnapshot, L7UnifiedExecutionSurface,
    )
except ImportError:
    RiskState = None  # type: ignore[assignment,misc]
    TCAAggregate = None  # type: ignore[assignment,misc]

# Rich library (optional)
try:
    from rich.text import Text
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ANSI codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"


def _bar(value: float, max_val: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """ASCII progress bar."""
    pct = min(value / max(max_val, 0.001), 1.0)
    filled = int(pct * width)
    return fill * filled + empty * (width - filled)


def _color_risk(level: str) -> str:
    """Color code for risk level."""
    return {
        "NORMAL": _GREEN, "ELEVATED": _YELLOW,
        "HIGH": _RED, "CRITICAL": f"{_BOLD}{_RED}",
    }.get(level, _WHITE)


def _gate_icon(passed: bool) -> str:
    return f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"


class L7DashboardRenderer:
    """Renders L7 risk and TCA dashboard panels."""

    # ------------------------------------------------------------------
    # ASCII Risk Panel
    # ------------------------------------------------------------------

    def render_risk_panel(self, risk: Optional[Any] = None, width: int = 60) -> str:
        """Render L7 risk state as ASCII panel."""
        if risk is None:
            return self._box("L7 RISK MANAGEMENT", ["No risk data available"], width)

        lines = []
        rc = _color_risk(risk.risk_level)

        # Header
        ks = f"{_RED}ACTIVE{_RESET}" if risk.kill_switch_active else f"{_GREEN}OFF{_RESET}"
        lines.append(f"  Risk Level: {rc}{risk.risk_level}{_RESET}    Kill Switch: {ks}")
        lines.append("")

        # NAV + P&L
        pnl_c = _GREEN if risk.daily_pnl >= 0 else _RED
        lines.append(f"  NAV:        ${risk.nav:>12,.2f}    Cash: ${risk.cash:>12,.2f}")
        lines.append(f"  Daily P&L:  {pnl_c}${risk.daily_pnl:>+12,.2f}{_RESET}  ({risk.daily_pnl_pct:>+.2%})")
        lines.append(f"  VaR 95/1d:  ${risk.var_95_1d:>12,.2f}")
        lines.append("")

        # Leverage
        lines.append(f"  Gross Leverage: {risk.gross_leverage:>6.1%}  {_bar(risk.gross_leverage, 2.5)}")
        lines.append(f"  Net Leverage:   {risk.net_leverage:>6.1%}  {_bar(risk.net_leverage, 1.5)}")
        lines.append("")

        # Position concentration
        lines.append(f"  Max Position:   {risk.max_position_pct:>6.1%}  [{risk.max_position_ticker}]")
        lines.append(f"  Positions:      {risk.position_count:>6d}")
        lines.append(f"  Drawdown:       {risk.intraday_drawdown_pct:>6.2%}  {_bar(risk.intraday_drawdown_pct, 0.10)}")
        lines.append("")

        # Gate status
        lines.append(f"  {'Gate':<20} {'Status':<10}")
        lines.append(f"  {'─'*20} {'─'*10}")
        for gate, passed in (risk.gates_status or {}).items():
            lines.append(f"  {gate:<20} {_gate_icon(passed)}")

        return self._box("L7 RISK MANAGEMENT", lines, width)

    # ------------------------------------------------------------------
    # ASCII TCA Panel
    # ------------------------------------------------------------------

    def render_tca_panel(
        self,
        tca: Optional[Any] = None,
        daily_costs: Optional[Dict[str, float]] = None,
        width: int = 60,
    ) -> str:
        """Render TCA analysis as ASCII panel."""
        if tca is None:
            return self._box("L7 TCA ANALYSIS", ["No TCA data available"], width)

        lines = []

        # Summary
        trend_c = _GREEN if tca.cost_trend == "IMPROVING" else (_RED if tca.cost_trend == "DEGRADING" else _YELLOW)
        lines.append(f"  Total Trades:   {tca.total_trades:>8d}    Volume: ${tca.total_volume_usd:>12,.0f}")
        lines.append(f"  Cost Trend:     {trend_c}{tca.cost_trend}{_RESET}")
        lines.append("")

        # Cost decomposition
        lines.append(f"  {'Component':<22} {'Avg (bps)':<10}")
        lines.append(f"  {'─'*22} {'─'*10}")
        lines.append(f"  {'Spread':<22} {tca.avg_spread_cost_bps:>8.2f}")
        lines.append(f"  {'Market Impact':<22} {tca.avg_market_impact_bps:>8.2f}")
        lines.append(f"  {'Timing':<22} {tca.avg_timing_cost_bps:>8.2f}")
        lines.append(f"  {'Commission':<22} {tca.avg_commission_bps:>8.2f}")
        lines.append(f"  {'─'*22} {'─'*10}")
        tc = _GREEN if tca.avg_total_cost_bps < 5 else (_YELLOW if tca.avg_total_cost_bps < 15 else _RED)
        lines.append(f"  {_BOLD}{'TOTAL':<22}{_RESET} {tc}{tca.avg_total_cost_bps:>8.2f}{_RESET}")
        lines.append("")

        # Per product
        lines.append(f"  {'Product':<22} {'Avg Cost (bps)':<14}")
        lines.append(f"  {'─'*22} {'─'*14}")
        lines.append(f"  {'Equity':<22} {tca.equity_avg_cost_bps:>12.2f}")
        lines.append(f"  {'Options':<22} {tca.option_avg_cost_bps:>12.2f}")
        lines.append(f"  {'Futures':<22} {tca.future_avg_cost_bps:>12.2f}")
        lines.append("")

        # Implementation shortfall
        is_c = _GREEN if tca.total_implementation_shortfall_usd <= 0 else _RED
        lines.append(f"  Impl. Shortfall: {is_c}${tca.total_implementation_shortfall_usd:>+12,.2f}{_RESET}")
        lines.append("")

        # Best/worst
        if tca.best_execution_ticker:
            lines.append(f"  Best Execution:  {_GREEN}{tca.best_execution_ticker}{_RESET}")
        if tca.worst_execution_ticker:
            lines.append(f"  Worst Execution: {_RED}{tca.worst_execution_ticker}{_RESET}")

        # Daily cost sparkline
        if daily_costs:
            costs = list(daily_costs.values())[-20:]
            spark = self._mini_sparkline(costs)
            lines.append("")
            lines.append(f"  Daily Cost Trend (last {len(costs)}d): {spark}")

        return self._box("L7 TCA ANALYSIS", lines, width)

    # ------------------------------------------------------------------
    # Combined L7 Panel
    # ------------------------------------------------------------------

    def render_l7_panel(self, l7: Optional[Any] = None, width: int = 80) -> str:
        """Render combined L7 execution summary panel."""
        if l7 is None:
            return self._box("L7 EXECUTION SURFACE", ["No L7 data available"], width)

        summary = l7.get_execution_summary() if hasattr(l7, 'get_execution_summary') else {}

        lines = []
        lines.append(f"  {'NAV:':<18} ${summary.get('nav', 0):>12,.2f}")
        lines.append(f"  {'Daily P&L:':<18} ${summary.get('daily_pnl', 0):>+12,.2f}")
        lines.append(f"  {'Positions:':<18} {summary.get('positions_count', 0):>12d}")
        lines.append(f"  {'Orders Today:':<18} {summary.get('total_orders_today', 0):>12d}")
        lines.append(f"  {'Fills Today:':<18} {summary.get('total_fills_today', 0):>12d}")
        lines.append("")

        # Routing stats
        routing = summary.get('routing_stats', {})
        lines.append(f"  Routing: EQ={routing.get('EQUITY', 0)} "
                     f"OPT={routing.get('OPTION', 0)} "
                     f"FUT={routing.get('FUTURE', 0)} "
                     f"REJ={routing.get('REJECTED', 0)}")
        lines.append("")

        # Risk
        rc = _color_risk(summary.get('risk_level', 'UNKNOWN'))
        ks = f"{_RED}ACTIVE{_RESET}" if summary.get('kill_switch') else f"{_GREEN}OFF{_RESET}"
        lines.append(f"  Risk: {rc}{summary.get('risk_level', 'UNKNOWN')}{_RESET}  "
                     f"Kill Switch: {ks}  "
                     f"VaR: ${summary.get('var_95_1d', 0):,.2f}")

        # TCA
        tc = summary.get('avg_tca_cost_bps', 0)
        trend = summary.get('tca_trend', 'STABLE')
        trend_c = _GREEN if trend == "IMPROVING" else (_RED if trend == "DEGRADING" else _YELLOW)
        lines.append(f"  TCA: {tc:.1f}bps avg  Trend: {trend_c}{trend}{_RESET}  "
                     f"Patterns: {summary.get('patterns_learned', 0)}")

        lines.append(f"  Heartbeat: #{summary.get('heartbeat', 0)}")

        return self._box("L7 EXECUTION SURFACE", lines, width)

    # ------------------------------------------------------------------
    # Rich panels (when rich library available)
    # ------------------------------------------------------------------

    def render_risk_panel_rich(self, risk: Optional[Any] = None) -> Any:
        """Render L7 risk as Rich Table for live dashboard integration."""
        if not RICH_AVAILABLE:
            return self.render_risk_panel(risk)

        table = Table(title="L7 Risk Management", show_header=True, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        if risk is None:
            table.add_row("Status", "No data", "[yellow]WAITING[/yellow]")
            return table

        # Risk level
        level_style = {"NORMAL": "green", "ELEVATED": "yellow",
                       "HIGH": "red", "CRITICAL": "bold red"}.get(risk.risk_level, "white")
        table.add_row("Risk Level", f"[{level_style}]{risk.risk_level}[/{level_style}]", "")

        # Kill switch
        ks_style = "bold red" if risk.kill_switch_active else "green"
        ks_text = "ACTIVE" if risk.kill_switch_active else "OFF"
        table.add_row("Kill Switch", f"[{ks_style}]{ks_text}[/{ks_style}]", "")

        # NAV
        table.add_row("NAV", f"${risk.nav:,.2f}", "")
        pnl_style = "green" if risk.daily_pnl >= 0 else "red"
        table.add_row("Daily P&L", f"[{pnl_style}]${risk.daily_pnl:+,.2f}[/{pnl_style}]",
                      f"[{pnl_style}]{risk.daily_pnl_pct:+.2%}[/{pnl_style}]")
        table.add_row("VaR 95/1d", f"${risk.var_95_1d:,.2f}", "")
        table.add_row("Gross Leverage", f"{risk.gross_leverage:.1%}", "")
        table.add_row("Net Leverage", f"{risk.net_leverage:.1%}", "")
        table.add_row("Drawdown", f"{risk.intraday_drawdown_pct:.2%}", "")

        # Gates
        for gate, passed in (risk.gates_status or {}).items():
            g_style = "green" if passed else "bold red"
            table.add_row(gate, f"[{g_style}]{'PASS' if passed else 'FAIL'}[/{g_style}]", "")

        return table

    def render_tca_panel_rich(self, tca: Optional[Any] = None) -> Any:
        """Render TCA as Rich Table."""
        if not RICH_AVAILABLE:
            return self.render_tca_panel(tca)

        table = Table(title="L7 TCA Analysis", show_header=True, expand=True)
        table.add_column("Component", style="cyan")
        table.add_column("Avg (bps)", justify="right")

        if tca is None:
            table.add_row("Status", "No data")
            return table

        table.add_row("Spread", f"{tca.avg_spread_cost_bps:.2f}")
        table.add_row("Market Impact", f"{tca.avg_market_impact_bps:.2f}")
        table.add_row("Timing", f"{tca.avg_timing_cost_bps:.2f}")
        table.add_row("Commission", f"{tca.avg_commission_bps:.2f}")

        cost_style = "green" if tca.avg_total_cost_bps < 5 else ("yellow" if tca.avg_total_cost_bps < 15 else "red")
        table.add_row(f"[bold]TOTAL[/bold]",
                      f"[{cost_style}][bold]{tca.avg_total_cost_bps:.2f}[/bold][/{cost_style}]")

        table.add_row("", "")
        table.add_row("Equity Avg", f"{tca.equity_avg_cost_bps:.2f}")
        table.add_row("Options Avg", f"{tca.option_avg_cost_bps:.2f}")
        table.add_row("Futures Avg", f"{tca.future_avg_cost_bps:.2f}")

        table.add_row("", "")
        table.add_row("Total Trades", f"{tca.total_trades}")
        is_style = "green" if tca.total_implementation_shortfall_usd <= 0 else "red"
        table.add_row("Impl. Shortfall",
                      f"[{is_style}]${tca.total_implementation_shortfall_usd:+,.2f}[/{is_style}]")

        trend_style = {"IMPROVING": "green", "DEGRADING": "red"}.get(tca.cost_trend, "yellow")
        table.add_row("Trend", f"[{trend_style}]{tca.cost_trend}[/{trend_style}]")

        return table

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box(title: str, lines: List[str], width: int = 60) -> str:
        """Simple ASCII box around content."""
        border = "═" * (width - 2)
        header = f"╔{border}╗\n║ {_BOLD}{title}{_RESET}" + " " * max(0, width - len(title) - 4) + "║\n"
        header += f"╠{border}╣\n"
        body = ""
        for line in lines:
            # Pad line (ignore ANSI for width calc)
            import re
            visible = len(re.sub(r"\033\[[0-9;]*m", "", line))
            padding = max(0, width - visible - 3)
            body += f"║{line}{' ' * padding}║\n"
        footer = f"╚{border}╝"
        return header + body + footer

    @staticmethod
    def _mini_sparkline(values: List[float], width: int = 20) -> str:
        """Tiny ASCII sparkline."""
        if not values:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        lo, hi = min(values), max(values)
        spread = hi - lo if hi != lo else 1.0
        scaled = [int((v - lo) / spread * (len(blocks) - 1)) for v in values]
        if len(scaled) > width:
            step = len(scaled) / width
            scaled = [scaled[int(i * step)] for i in range(width)]
        return "".join(blocks[s] for s in scaled)
