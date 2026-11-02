"""Portfolio Analytics Report — performance deep dive, scenario engine, risk analytics.

Generates comprehensive ASCII reports for morning open and evening close.
Imported by run_open.py and run_close.py:
    from engine.monitoring.portfolio_report import PortfolioReportGenerator
    port_gen = PortfolioReportGenerator()
    port_report = port_gen.generate_open_report()   # morning
    port_report = port_gen.generate_close_report()   # evening

Report structure (20 sections):
    Part 1 — Performance Deep Dive (7 sections)
        1. NAV Waterfall
        2. Attribution by Sleeve (P1-P5)
        3. Attribution by Sector (11 GICS)
        4. Factor Exposure
        5. Sharpe / Sortino / Calmar
        6. Max Drawdown History
        7. Rolling Returns

    Part 2 — Scenario Engine (6 sections)
        8. Base Case projection
        9. Bull scenario
        10. Bear scenario
        11. 2008 GFC replay
        12. 2020 COVID replay
        13. 2022 Rate Hike replay

    Part 3 — Risk Analytics (7 sections)
        14. VaR breakdown (parametric + historical)
        15. Expected Shortfall (CVaR)
        16. Correlation Matrix
        17. Beta Decomposition
        18. Liquidity Risk (ADV ratios)
        19. Concentration Risk (HHI)
        20. Stress Test Summary

Pure Python + numpy + datetime.  No pandas, no external broker.
"""

import math
from datetime import datetime, timedelta

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_WIDTH = 78
INITIAL_NAV = 1_000_000.0

SLEEVES = [
    ("P1", "Directional Equities"),
    ("P2", "Factor Rotation"),
    ("P3", "Commodities/Macro"),
    ("P4", "Options Convexity"),
    ("P5", "Hedges/Volatility"),
]

GICS_SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

FACTORS = ["Beta", "Momentum", "Value", "Quality", "Size"]

TOP_POSITIONS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "JPM", "V", "UNH",
]

SCENARIO_NAMES = {
    "base":      "Base Case",
    "bull":      "Bull Scenario",
    "bear":      "Bear Scenario",
    "gfc_2008":  "2008 GFC Replay",
    "covid_2020": "2020 COVID Replay",
    "rate_2022":  "2022 Rate Hike Replay",
}


# ---------------------------------------------------------------------------
# Deterministic seed helper — paper-trading placeholder data
# ---------------------------------------------------------------------------
def _day_seed() -> int:
    """Seed derived from today's date for reproducible daily placeholder data."""
    d = datetime.now()
    return d.year * 10000 + d.month * 100 + d.day


def _rng() -> np.random.RandomState:
    return np.random.RandomState(_day_seed())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _hline(char: str = "=", width: int = REPORT_WIDTH) -> str:
    return char * width


def _center(text: str, width: int = REPORT_WIDTH) -> str:
    return text.center(width)


def _header(title: str, char: str = "=") -> str:
    line = _hline(char)
    return f"{line}\n{_center(title)}\n{line}"


def _section(num: int, title: str) -> str:
    """Section header with number tag."""
    tag = f"[{num:02d}]"
    label = f"{tag}  {title}"
    bar = "-" * REPORT_WIDTH
    return f"\n{bar}\n{label}\n{bar}"


def _fmt_pct(val: float, width: int = 8) -> str:
    """Format percentage with sign, e.g. ' +1.23%'."""
    s = f"{val:+.2f}%"
    return s.rjust(width)


def _fmt_dollar(val: float, width: int = 14) -> str:
    """Format dollar amount with comma separators."""
    if val < 0:
        s = f"-${abs(val):,.2f}"
    else:
        s = f"${val:,.2f}"
    return s.rjust(width)


def _fmt_ratio(val: float, width: int = 8) -> str:
    s = f"{val:+.3f}"
    return s.rjust(width)


def _bar_chart(val: float, max_abs: float, bar_len: int = 20) -> str:
    """Simple inline ASCII bar.  Positive → right, negative → left."""
    if max_abs == 0:
        return " " * bar_len
    frac = val / max_abs
    frac = max(-1.0, min(1.0, frac))
    half = bar_len // 2
    if frac >= 0:
        n = int(round(frac * half))
        return " " * half + "|" + "#" * n + " " * (half - n)
    else:
        n = int(round(-frac * half))
        return " " * (half - n) + "#" * n + "|" + " " * half


def _table_row(cols: list, widths: list) -> str:
    """Build a fixed-width table row from columns."""
    parts = []
    for col, w in zip(cols, widths):
        parts.append(str(col).ljust(w) if w > 0 else str(col).rjust(-w))
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Placeholder data generators — deterministic per day
# ---------------------------------------------------------------------------
class _PlaceholderPortfolio:
    """Generate consistent paper-trading placeholder data for one day."""

    def __init__(self):
        self.rng = _rng()
        self._build()

    # -- core NAV waterfall ---------------------------------------------------
    def _build(self):
        self.nav_start = INITIAL_NAV + self.rng.uniform(-20000, 40000)
        self.trade_impact = self.rng.uniform(-3000, 5000)
        self.fees = -(abs(self.rng.normal(150, 50)))
        self.gross_pnl = self.rng.normal(1200, 4000)
        self.net_pnl = self.gross_pnl + self.fees
        self.nav_end = self.nav_start + self.trade_impact + self.net_pnl

        # sleeve weights & returns
        base_weights = np.array([0.35, 0.22, 0.15, 0.14, 0.14])
        noise = self.rng.normal(0, 0.02, 5)
        self.sleeve_weights = np.clip(base_weights + noise, 0.02, 0.60)
        self.sleeve_weights /= self.sleeve_weights.sum()
        self.sleeve_returns = self.rng.normal(0.05, 1.2, 5)  # daily bps → %

        # sector returns
        self.sector_returns = self.rng.normal(0.03, 0.8, 11)
        self.sector_weights = np.abs(self.rng.dirichlet(np.ones(11) * 3))

        # factor exposures
        self.factor_exposures = {
            "Beta":     round(self.rng.uniform(0.70, 1.15), 3),
            "Momentum": round(self.rng.uniform(-0.30, 0.45), 3),
            "Value":    round(self.rng.uniform(-0.20, 0.35), 3),
            "Quality":  round(self.rng.uniform(0.05, 0.50), 3),
            "Size":     round(self.rng.uniform(-0.40, 0.10), 3),
        }

        # ratios
        self.sharpe = round(self.rng.uniform(0.8, 2.5), 2)
        self.sortino = round(self.sharpe * self.rng.uniform(1.1, 1.6), 2)
        self.calmar = round(self.rng.uniform(0.5, 3.0), 2)

        # drawdown history (last 10 drawdowns)
        self.drawdowns = sorted(
            [round(self.rng.uniform(-18, -0.5), 2) for _ in range(10)]
        )
        self.max_dd = self.drawdowns[0]
        self.current_dd = round(self.rng.uniform(self.max_dd * 0.3, 0), 2)

        # rolling returns
        self.rolling = {
            "1d":  round(self.rng.normal(0.04, 0.8), 2),
            "5d":  round(self.rng.normal(0.20, 1.5), 2),
            "20d": round(self.rng.normal(0.80, 3.0), 2),
            "60d": round(self.rng.normal(2.50, 5.0), 2),
        }

        # positions for correlation / concentration
        n_pos = len(TOP_POSITIONS)
        self.position_weights = np.abs(self.rng.dirichlet(np.ones(n_pos) * 2))
        self.position_betas = self.rng.uniform(0.6, 1.6, n_pos)
        self.position_adv_ratio = self.rng.uniform(0.001, 0.05, n_pos)

        # correlation matrix (symmetric, 1 on diagonal)
        raw = self.rng.uniform(0.1, 0.7, (n_pos, n_pos))
        corr = (raw + raw.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        self.corr_matrix = np.round(corr, 2)

        # VaR / CVaR
        self.var_95_para = round(self.rng.uniform(-1.5, -0.5), 2)
        self.var_99_para = round(self.var_95_para * self.rng.uniform(1.3, 1.6), 2)
        self.var_95_hist = round(self.var_95_para * self.rng.uniform(0.9, 1.1), 2)
        self.var_99_hist = round(self.var_99_para * self.rng.uniform(0.9, 1.1), 2)
        self.cvar_95 = round(self.var_95_para * self.rng.uniform(1.2, 1.5), 2)
        self.cvar_99 = round(self.var_99_para * self.rng.uniform(1.2, 1.5), 2)

        # HHI
        self.hhi = round(float(np.sum(self.position_weights ** 2) * 10000), 1)

        # scenario shocks
        self.scenarios = self._build_scenarios()

    def _build_scenarios(self) -> dict:
        """Build scenario projections for the current portfolio."""
        nav = self.nav_end
        scenarios = {}
        configs = {
            "base":       {"equity": 0.06, "vol": 0.12, "horizon_d": 252},
            "bull":       {"equity": 0.18, "vol": 0.10, "horizon_d": 252},
            "bear":       {"equity": -0.15, "vol": 0.22, "horizon_d": 252},
            "gfc_2008":   {"equity": -0.38, "vol": 0.45, "horizon_d": 126},
            "covid_2020": {"equity": -0.34, "vol": 0.65, "horizon_d":  23},
            "rate_2022":  {"equity": -0.19, "vol": 0.28, "horizon_d": 200},
        }
        beta = self.factor_exposures["Beta"]
        for key, cfg in configs.items():
            eq_ret = cfg["equity"]
            vol = cfg["vol"]
            horizon = cfg["horizon_d"]
            port_ret = eq_ret * beta
            port_vol = vol * beta
            nav_proj = nav * (1 + port_ret)
            nav_low = nav * (1 + port_ret - 1.65 * port_vol * math.sqrt(horizon / 252))
            nav_high = nav * (1 + port_ret + 1.65 * port_vol * math.sqrt(horizon / 252))
            pnl = nav_proj - nav
            drawdown = min(0.0, port_ret - 1.65 * port_vol * math.sqrt(horizon / 252))
            scenarios[key] = {
                "name": SCENARIO_NAMES[key],
                "equity_return": eq_ret,
                "vol": vol,
                "horizon_days": horizon,
                "port_return": round(port_ret * 100, 2),
                "nav_projected": round(nav_proj, 2),
                "nav_low": round(nav_low, 2),
                "nav_high": round(nav_high, 2),
                "pnl": round(pnl, 2),
                "max_dd_est": round(drawdown * 100, 2),
            }
        return scenarios


# ===========================================================================
#  PortfolioReportGenerator
# ===========================================================================
class PortfolioReportGenerator:
    """Builds ASCII portfolio analytics reports for open and close sessions."""

    def __init__(self):
        self.data = _PlaceholderPortfolio()
        self.timestamp = datetime.now()

    # -----------------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------------
    def generate_open_report(self) -> str:
        """Morning open report — full performance + scenario + risk."""
        lines: list[str] = []
        self._append_banner(lines, "MORNING OPEN")
        self._append_part1(lines)
        self._append_part2(lines)
        self._append_part3(lines)
        self._append_footer(lines, "END MORNING PORTFOLIO REPORT")
        return "\n".join(lines)

    def generate_close_report(self) -> str:
        """Evening close report — same structure, labelled as close."""
        lines: list[str] = []
        self._append_banner(lines, "EVENING CLOSE")
        self._append_part1(lines)
        self._append_part2(lines)
        self._append_part3(lines)
        self._append_footer(lines, "END EVENING PORTFOLIO REPORT")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    #  Banner / Footer
    # -----------------------------------------------------------------------
    def _append_banner(self, lines: list, session: str):
        lines.append("")
        lines.append(_hline("="))
        lines.append(_center("METADRON CAPITAL — PORTFOLIO ANALYTICS REPORT"))
        lines.append(_center(f"Session: {session}"))
        lines.append(_center(self.timestamp.strftime("%Y-%m-%d  %H:%M:%S ET")))
        lines.append(_hline("="))
        lines.append("")
        lines.append(_center("Paper Trading System — OpenBB Data"))
        lines.append(_center("All figures are simulated placeholder values"))
        lines.append("")

    def _append_footer(self, lines: list, label: str):
        lines.append("")
        lines.append(_hline("="))
        lines.append(_center(label))
        lines.append(_center(self.timestamp.strftime("%Y-%m-%d  %H:%M:%S ET")))
        lines.append(_hline("="))
        lines.append("")

    # =======================================================================
    #  PART 1 — Performance Deep Dive
    # =======================================================================
    def _append_part1(self, lines: list):
        lines.append(_header("PART 1 — PERFORMANCE DEEP DIVE", "="))
        lines.append("")
        self._section_01_nav_waterfall(lines)
        self._section_02_sleeve_attribution(lines)
        self._section_03_sector_attribution(lines)
        self._section_04_factor_exposure(lines)
        self._section_05_ratios(lines)
        self._section_06_drawdown(lines)
        self._section_07_rolling_returns(lines)

    # -- 01 NAV Waterfall ----------------------------------------------------
    def _section_01_nav_waterfall(self, lines: list):
        d = self.data
        lines.append(_section(1, "NAV WATERFALL"))
        lines.append("")
        lines.append("  Tracks NAV progression from start of day through each")
        lines.append("  component to arrive at end-of-day NAV.")
        lines.append("")

        w_label = 30
        w_val = 16
        w_bar = 24

        items = [
            ("Starting NAV",     d.nav_start,    0),
            ("+ Trade Impact",   d.trade_impact, d.trade_impact),
            ("- Fees & Costs",   d.fees,         d.fees),
            ("+ Gross PnL",      d.gross_pnl,    d.gross_pnl),
            ("= Net PnL",        d.net_pnl,      d.net_pnl),
            ("= Ending NAV",     d.nav_end,      0),
        ]

        max_abs = max(abs(x[2]) for x in items) or 1.0

        lines.append(f"  {'Component':<{w_label}} {'Amount':>{w_val}}  {'':>{w_bar}}")
        lines.append(f"  {'-' * w_label} {'-' * w_val}  {'-' * w_bar}")

        for label, val, bar_val in items:
            dollar = _fmt_dollar(val, w_val)
            bar = _bar_chart(bar_val, max_abs, w_bar) if bar_val != 0 else ""
            lines.append(f"  {label:<{w_label}} {dollar}  {bar}")

        lines.append("")
        day_ret = (d.nav_end / d.nav_start - 1) * 100
        lines.append(f"  Day Return: {day_ret:+.4f}%")
        lines.append(f"  Annualised: {day_ret * 252:+.2f}%  (approx)")
        lines.append("")

    # -- 02 Attribution by Sleeve --------------------------------------------
    def _section_02_sleeve_attribution(self, lines: list):
        d = self.data
        lines.append(_section(2, "ATTRIBUTION BY SLEEVE (P1-P5)"))
        lines.append("")

        hdr = f"  {'Sleeve':<6} {'Name':<24} {'Weight':>8} {'Return':>8} {'Contrib':>8}  {'Bar':>20}"
        lines.append(hdr)
        lines.append(f"  {'-' * 6} {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 20}")

        contribs = d.sleeve_weights * d.sleeve_returns
        max_c = max(abs(contribs.max()), abs(contribs.min()), 0.01)

        total_w = 0.0
        total_c = 0.0
        for i, (code, name) in enumerate(SLEEVES):
            w = d.sleeve_weights[i] * 100
            r = d.sleeve_returns[i]
            c = contribs[i]
            total_w += w
            total_c += c
            bar = _bar_chart(c, max_c, 20)
            lines.append(
                f"  {code:<6} {name:<24} {w:>7.1f}% {r:>+7.2f}% {c:>+7.3f}%  {bar}"
            )

        lines.append(f"  {'-' * 6} {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 20}")
        lines.append(
            f"  {'TOTAL':<6} {'':<24} {total_w:>7.1f}% {'':<8} {total_c:>+7.3f}%"
        )
        lines.append("")

    # -- 03 Attribution by Sector (11 GICS) ----------------------------------
    def _section_03_sector_attribution(self, lines: list):
        d = self.data
        lines.append(_section(3, "ATTRIBUTION BY SECTOR (11 GICS)"))
        lines.append("")

        hdr = f"  {'Sector':<28} {'Weight':>8} {'Return':>8} {'Contrib':>8}  {'Bar':>20}"
        lines.append(hdr)
        lines.append(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 20}")

        contribs = d.sector_weights * d.sector_returns
        max_c = max(abs(contribs.max()), abs(contribs.min()), 0.01)

        order = np.argsort(-contribs)
        total_c = 0.0
        for idx in order:
            sec = GICS_SECTORS[idx]
            w = d.sector_weights[idx] * 100
            r = d.sector_returns[idx]
            c = contribs[idx]
            total_c += c
            bar = _bar_chart(c, max_c, 20)
            lines.append(
                f"  {sec:<28} {w:>7.1f}% {r:>+7.2f}% {c:>+7.3f}%  {bar}"
            )

        lines.append(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 20}")
        lines.append(
            f"  {'TOTAL':<28} {'100.0%':>8} {'':<8} {total_c:>+7.3f}%"
        )
        lines.append("")

        # top / bottom contributors
        best_idx = order[0]
        worst_idx = order[-1]
        lines.append(f"  Best  contributor: {GICS_SECTORS[best_idx]:<28} {contribs[best_idx]:>+.3f}%")
        lines.append(f"  Worst contributor: {GICS_SECTORS[worst_idx]:<28} {contribs[worst_idx]:>+.3f}%")
        lines.append("")

    # -- 04 Factor Exposure ---------------------------------------------------
    def _section_04_factor_exposure(self, lines: list):
        d = self.data
        lines.append(_section(4, "FACTOR EXPOSURE"))
        lines.append("")
        lines.append("  Factor exposures estimated from portfolio holdings.")
        lines.append("  Benchmark: SPY (S&P 500)")
        lines.append("")

        hdr = f"  {'Factor':<14} {'Exposure':>10} {'Target':>10} {'Delta':>10}  {'Bar':>20}"
        lines.append(hdr)
        lines.append(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 20}")

        targets = {"Beta": 1.0, "Momentum": 0.15, "Value": 0.10, "Quality": 0.20, "Size": -0.10}
        exposures = d.factor_exposures

        vals = list(exposures.values())
        tgts = [targets[f] for f in FACTORS]
        deltas = [exposures[f] - targets[f] for f in FACTORS]
        max_d = max(abs(x) for x in deltas) or 0.01

        for i, factor in enumerate(FACTORS):
            exp = exposures[factor]
            tgt = targets[factor]
            delta = deltas[i]
            bar = _bar_chart(delta, max_d, 20)
            lines.append(
                f"  {factor:<14} {exp:>+10.3f} {tgt:>+10.3f} {delta:>+10.3f}  {bar}"
            )

        lines.append("")
        beta = exposures["Beta"]
        if beta > 1.10:
            lines.append("  WARNING: Portfolio beta elevated (>1.10) — consider hedging")
        elif beta < 0.80:
            lines.append("  NOTE: Portfolio beta defensive (<0.80)")
        else:
            lines.append("  STATUS: Beta within normal range (0.80 - 1.10)")
        lines.append("")

    # -- 05 Sharpe / Sortino / Calmar ----------------------------------------
    def _section_05_ratios(self, lines: list):
        d = self.data
        lines.append(_section(5, "RISK-ADJUSTED RETURN RATIOS"))
        lines.append("")

        ratios = [
            ("Sharpe Ratio",  d.sharpe,  "Excess return / total volatility"),
            ("Sortino Ratio", d.sortino, "Excess return / downside volatility"),
            ("Calmar Ratio",  d.calmar,  "Annualised return / max drawdown"),
        ]

        lines.append(f"  {'Metric':<20} {'Value':>10}   {'Description':<36}")
        lines.append(f"  {'-' * 20} {'-' * 10}   {'-' * 36}")
        for name, val, desc in ratios:
            grade = self._ratio_grade(val)
            lines.append(f"  {name:<20} {val:>+10.2f}   {desc:<36}  [{grade}]")

        lines.append("")
        lines.append("  Grading:  [A] >= 2.0  |  [B] >= 1.5  |  [C] >= 1.0  |  [D] < 1.0")
        lines.append("")

        # trailing annualised table
        lines.append("  Trailing Annualised Ratios:")
        lines.append(f"  {'Period':<14} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
        lines.append(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8}")
        rng = self.data.rng
        for period in ["1 Month", "3 Months", "6 Months", "YTD", "1 Year"]:
            s = round(d.sharpe + rng.normal(0, 0.3), 2)
            so = round(d.sortino + rng.normal(0, 0.4), 2)
            ca = round(d.calmar + rng.normal(0, 0.5), 2)
            lines.append(f"  {period:<14} {s:>+8.2f} {so:>+8.2f} {ca:>+8.2f}")
        lines.append("")

    @staticmethod
    def _ratio_grade(val: float) -> str:
        if val >= 2.0:
            return "A"
        if val >= 1.5:
            return "B"
        if val >= 1.0:
            return "C"
        return "D"

    # -- 06 Max Drawdown History ---------------------------------------------
    def _section_06_drawdown(self, lines: list):
        d = self.data
        lines.append(_section(6, "MAX DRAWDOWN HISTORY"))
        lines.append("")

        lines.append(f"  Current Drawdown:  {d.current_dd:+.2f}%")
        lines.append(f"  Maximum Drawdown:  {d.max_dd:+.2f}%")
        lines.append("")

        lines.append("  Historical Drawdowns (deepest first):")
        lines.append(f"  {'Rank':>4}  {'Drawdown':>10}  {'Depth Bar':<40}")
        lines.append(f"  {'-' * 4}  {'-' * 10}  {'-' * 40}")

        abs_max = abs(d.max_dd) or 1.0
        for i, dd in enumerate(d.drawdowns):
            n_blocks = int(round(abs(dd) / abs_max * 30))
            bar = "#" * n_blocks
            lines.append(f"  {i + 1:>4}  {dd:>+10.2f}%  {bar}")

        lines.append("")

        # drawdown duration table
        lines.append("  Drawdown Duration Statistics:")
        lines.append(f"  {'Metric':<30} {'Days':>8}")
        lines.append(f"  {'-' * 30} {'-' * 8}")
        rng = self.data.rng
        lines.append(f"  {'Avg Drawdown Duration':<30} {int(rng.uniform(5, 25)):>8}")
        lines.append(f"  {'Max Drawdown Duration':<30} {int(rng.uniform(20, 90)):>8}")
        lines.append(f"  {'Current DD Duration':<30} {int(rng.uniform(0, 15)):>8}")
        lines.append(f"  {'Time to Recovery (est.)':<30} {int(rng.uniform(3, 30)):>8}")
        lines.append("")

    # -- 07 Rolling Returns --------------------------------------------------
    def _section_07_rolling_returns(self, lines: list):
        d = self.data
        lines.append(_section(7, "ROLLING RETURNS"))
        lines.append("")

        lines.append(f"  {'Window':<10} {'Return':>8} {'Ann. Vol':>10} {'Sharpe':>8}  {'Bar':>20}")
        lines.append(f"  {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 8}  {'-' * 20}")

        max_r = max(abs(v) for v in d.rolling.values()) or 0.01
        rng = self.data.rng

        for window, ret in d.rolling.items():
            vol = abs(ret) * rng.uniform(1.5, 4.0)
            sharpe_est = ret / vol if vol > 0 else 0.0
            bar = _bar_chart(ret, max_r, 20)
            lines.append(
                f"  {window:<10} {ret:>+7.2f}% {vol:>9.2f}% {sharpe_est:>+7.2f}  {bar}"
            )

        lines.append("")

        # cumulative returns block
        lines.append("  Cumulative Return Milestones:")
        lines.append(f"  {'Period':<16} {'Cumulative':>10} {'Benchmark':>10} {'Alpha':>10}")
        lines.append(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 10}")
        for period in ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "YTD"]:
            cum = round(rng.normal(0.5, 3.0), 2)
            bench = round(rng.normal(0.4, 2.5), 2)
            alpha = round(cum - bench, 2)
            lines.append(
                f"  {period:<16} {cum:>+9.2f}% {bench:>+9.2f}% {alpha:>+9.2f}%"
            )
        lines.append("")

    # =======================================================================
    #  PART 2 — Scenario Engine
    # =======================================================================
    def _append_part2(self, lines: list):
        lines.append(_header("PART 2 — SCENARIO ENGINE", "="))
        lines.append("")
        lines.append("  Projects portfolio NAV under various macro scenarios.")
        lines.append(f"  Current Portfolio Beta: {self.data.factor_exposures['Beta']:.3f}")
        lines.append(f"  Current NAV: {_fmt_dollar(self.data.nav_end)}")
        lines.append("")

        self._section_08_base(lines)
        self._section_09_bull(lines)
        self._section_10_bear(lines)
        self._section_11_gfc(lines)
        self._section_12_covid(lines)
        self._section_13_rate(lines)
        self._scenario_comparison_table(lines)

    def _render_scenario(self, lines: list, section_num: int, key: str):
        """Render a single scenario section."""
        sc = self.data.scenarios[key]
        lines.append(_section(section_num, sc["name"].upper()))
        lines.append("")

        lines.append(f"  Scenario Assumptions:")
        lines.append(f"    Equity Market Return:   {sc['equity_return'] * 100:>+8.2f}%")
        lines.append(f"    Market Volatility:      {sc['vol'] * 100:>8.1f}%")
        lines.append(f"    Horizon:                {sc['horizon_days']:>8} days")
        lines.append("")

        lines.append(f"  Portfolio Projections:")
        lines.append(f"    Expected Return:        {sc['port_return']:>+8.2f}%")
        lines.append(f"    Projected NAV:          {_fmt_dollar(sc['nav_projected'])}")
        lines.append(f"    NAV Range (90% CI):     {_fmt_dollar(sc['nav_low'])}  —  {_fmt_dollar(sc['nav_high'])}")
        lines.append(f"    Expected PnL:           {_fmt_dollar(sc['pnl'])}")
        lines.append(f"    Max Drawdown (est.):    {sc['max_dd_est']:>+8.2f}%")
        lines.append("")

        # mini bar chart of NAV range
        nav_lo = sc["nav_low"]
        nav_hi = sc["nav_high"]
        nav_proj = sc["nav_projected"]
        nav_now = self.data.nav_end
        rng_span = nav_hi - nav_lo if nav_hi != nav_lo else 1.0
        bar_width = 50
        pos_now = int((nav_now - nav_lo) / rng_span * bar_width)
        pos_proj = int((nav_proj - nav_lo) / rng_span * bar_width)
        pos_now = max(0, min(bar_width - 1, pos_now))
        pos_proj = max(0, min(bar_width - 1, pos_proj))

        bar_chars = ["-"] * bar_width
        bar_chars[0] = "|"
        bar_chars[-1] = "|"
        if 0 <= pos_now < bar_width:
            bar_chars[pos_now] = "N"
        if 0 <= pos_proj < bar_width:
            bar_chars[pos_proj] = "P"

        bar_str = "".join(bar_chars)
        lines.append(f"  NAV Range: [{''.join(bar_str)}]")
        lines.append(f"             {'Low':>{pos_now + 4}}{'Now (N)':>{8}}    Proj (P)    {'High':>{4}}")
        lines.append("")

    def _section_08_base(self, lines: list):
        self._render_scenario(lines, 8, "base")

    def _section_09_bull(self, lines: list):
        self._render_scenario(lines, 9, "bull")

    def _section_10_bear(self, lines: list):
        self._render_scenario(lines, 10, "bear")

    def _section_11_gfc(self, lines: list):
        self._render_scenario(lines, 11, "gfc_2008")

    def _section_12_covid(self, lines: list):
        self._render_scenario(lines, 12, "covid_2020")

    def _section_13_rate(self, lines: list):
        self._render_scenario(lines, 13, "rate_2022")

    def _scenario_comparison_table(self, lines: list):
        """Summary comparison table across all 6 scenarios."""
        lines.append("")
        lines.append(f"  {'-' * 74}")
        lines.append(f"  SCENARIO COMPARISON SUMMARY")
        lines.append(f"  {'-' * 74}")
        lines.append("")

        hdr = (
            f"  {'Scenario':<24} {'Ret %':>8} {'PnL':>14} "
            f"{'NAV Proj':>14} {'MaxDD':>8}"
        )
        lines.append(hdr)
        lines.append(
            f"  {'-' * 24} {'-' * 8} {'-' * 14} {'-' * 14} {'-' * 8}"
        )

        for key in ["base", "bull", "bear", "gfc_2008", "covid_2020", "rate_2022"]:
            sc = self.data.scenarios[key]
            lines.append(
                f"  {sc['name']:<24} {sc['port_return']:>+7.2f}% "
                f"{_fmt_dollar(sc['pnl'])} "
                f"{_fmt_dollar(sc['nav_projected'])} "
                f"{sc['max_dd_est']:>+7.2f}%"
            )

        lines.append(
            f"  {'-' * 24} {'-' * 8} {'-' * 14} {'-' * 14} {'-' * 8}"
        )

        # worst case
        worst_key = min(self.data.scenarios, key=lambda k: self.data.scenarios[k]["pnl"])
        worst = self.data.scenarios[worst_key]
        lines.append(f"  Worst case: {worst['name']}  =>  PnL {_fmt_dollar(worst['pnl'])}")
        lines.append("")

    # =======================================================================
    #  PART 3 — Risk Analytics
    # =======================================================================
    def _append_part3(self, lines: list):
        lines.append(_header("PART 3 — RISK ANALYTICS", "="))
        lines.append("")
        self._section_14_var(lines)
        self._section_15_cvar(lines)
        self._section_16_correlation(lines)
        self._section_17_beta_decomp(lines)
        self._section_18_liquidity(lines)
        self._section_19_concentration(lines)
        self._section_20_stress_summary(lines)

    # -- 14 VaR Breakdown ---------------------------------------------------
    def _section_14_var(self, lines: list):
        d = self.data
        lines.append(_section(14, "VALUE AT RISK (VaR) BREAKDOWN"))
        lines.append("")
        lines.append("  VaR estimates potential portfolio loss at given confidence level.")
        lines.append(f"  Portfolio NAV: {_fmt_dollar(d.nav_end)}")
        lines.append("")

        lines.append(f"  {'Method':<22} {'95% VaR':>10} {'99% VaR':>10} {'95% $':>14} {'99% $':>14}")
        lines.append(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 14} {'-' * 14}")

        var95p_d = d.var_95_para / 100 * d.nav_end
        var99p_d = d.var_99_para / 100 * d.nav_end
        var95h_d = d.var_95_hist / 100 * d.nav_end
        var99h_d = d.var_99_hist / 100 * d.nav_end

        lines.append(
            f"  {'Parametric (Normal)':<22} {d.var_95_para:>+9.2f}% {d.var_99_para:>+9.2f}% "
            f"{_fmt_dollar(var95p_d)} {_fmt_dollar(var99p_d)}"
        )
        lines.append(
            f"  {'Historical (500d)':<22} {d.var_95_hist:>+9.2f}% {d.var_99_hist:>+9.2f}% "
            f"{_fmt_dollar(var95h_d)} {_fmt_dollar(var99h_d)}"
        )
        lines.append("")

        # VaR by sleeve
        lines.append("  VaR Contribution by Sleeve:")
        lines.append(f"  {'Sleeve':<28} {'Marginal VaR':>12} {'Component VaR':>14}")
        lines.append(f"  {'-' * 28} {'-' * 12} {'-' * 14}")

        rng = self.data.rng
        total_mvar = 0.0
        total_cvar = 0.0
        for code, name in SLEEVES:
            mvar = round(rng.uniform(0.05, 0.50), 2)
            cvar = round(mvar * rng.uniform(0.5, 1.2), 2)
            total_mvar += mvar
            total_cvar += cvar
            lines.append(
                f"  {code + ' ' + name:<28} {mvar:>+11.2f}% {cvar:>+13.2f}%"
            )
        lines.append(f"  {'-' * 28} {'-' * 12} {'-' * 14}")
        lines.append(
            f"  {'TOTAL':<28} {total_mvar:>+11.2f}% {total_cvar:>+13.2f}%"
        )
        lines.append("")

    # -- 15 Expected Shortfall (CVaR) ----------------------------------------
    def _section_15_cvar(self, lines: list):
        d = self.data
        lines.append(_section(15, "EXPECTED SHORTFALL (CVaR)"))
        lines.append("")
        lines.append("  CVaR = average loss in the worst (1-alpha)% of scenarios.")
        lines.append("")

        lines.append(f"  {'Metric':<30} {'Value':>10} {'Dollar':>14}")
        lines.append(f"  {'-' * 30} {'-' * 10} {'-' * 14}")

        cvar95_d = d.cvar_95 / 100 * d.nav_end
        cvar99_d = d.cvar_99 / 100 * d.nav_end

        lines.append(f"  {'CVaR 95%':<30} {d.cvar_95:>+9.2f}% {_fmt_dollar(cvar95_d)}")
        lines.append(f"  {'CVaR 99%':<30} {d.cvar_99:>+9.2f}% {_fmt_dollar(cvar99_d)}")
        lines.append(f"  {'VaR 95% (for reference)':<30} {d.var_95_para:>+9.2f}% {_fmt_dollar(d.var_95_para / 100 * d.nav_end)}")
        lines.append(f"  {'VaR 99% (for reference)':<30} {d.var_99_para:>+9.2f}% {_fmt_dollar(d.var_99_para / 100 * d.nav_end)}")
        lines.append("")

        # tail risk profile
        lines.append("  Tail Risk Profile:")
        lines.append(f"  {'Percentile':>12} {'Loss':>10} {'$ Impact':>14}")
        lines.append(f"  {'-' * 12} {'-' * 10} {'-' * 14}")
        rng = self.data.rng
        for pct in [99.5, 99.0, 97.5, 95.0, 90.0]:
            loss = round(d.var_99_para * (pct / 99.0) * rng.uniform(0.8, 1.2), 2)
            dollar = loss / 100 * d.nav_end
            lines.append(f"  {pct:>11.1f}% {loss:>+9.2f}% {_fmt_dollar(dollar)}")
        lines.append("")

        # CVaR ratio
        if d.var_95_para != 0:
            ratio = d.cvar_95 / d.var_95_para
        else:
            ratio = 0
        lines.append(f"  CVaR/VaR Ratio (95%): {ratio:.2f}")
        if ratio > 1.5:
            lines.append("  WARNING: Fat-tailed distribution — tail risk elevated")
        else:
            lines.append("  STATUS: Tail risk within normal bounds")
        lines.append("")

    # -- 16 Correlation Matrix -----------------------------------------------
    def _section_16_correlation(self, lines: list):
        d = self.data
        lines.append(_section(16, "CORRELATION MATRIX (TOP POSITIONS)"))
        lines.append("")

        tickers = TOP_POSITIONS
        n = len(tickers)

        # header row
        hdr = "          " + "".join(f"{t:>7}" for t in tickers)
        lines.append(hdr)
        lines.append("  " + "-" * (8 + 7 * n))

        for i in range(n):
            row_vals = []
            for j in range(n):
                val = d.corr_matrix[i, j]
                if i == j:
                    row_vals.append("  1.00")
                else:
                    row_vals.append(f"{val:>6.2f}")
            lines.append(f"  {tickers[i]:<6}" + " ".join(row_vals))

        lines.append("")

        # high correlation pairs
        lines.append("  High Correlation Pairs (|rho| > 0.55):")
        lines.append(f"  {'Pair':<16} {'Correlation':>12}")
        lines.append(f"  {'-' * 16} {'-' * 12}")
        pairs_found = 0
        for i in range(n):
            for j in range(i + 1, n):
                rho = d.corr_matrix[i, j]
                if abs(rho) > 0.55:
                    pair = f"{tickers[i]}/{tickers[j]}"
                    lines.append(f"  {pair:<16} {rho:>+11.2f}")
                    pairs_found += 1
        if pairs_found == 0:
            lines.append("  (none above threshold)")
        lines.append("")

        # average pairwise correlation
        upper_tri = d.corr_matrix[np.triu_indices(n, k=1)]
        avg_corr = float(np.mean(upper_tri))
        lines.append(f"  Average Pairwise Correlation: {avg_corr:+.3f}")
        if avg_corr > 0.50:
            lines.append("  WARNING: High average correlation — diversification limited")
        else:
            lines.append("  STATUS: Correlation levels acceptable")
        lines.append("")

    # -- 17 Beta Decomposition -----------------------------------------------
    def _section_17_beta_decomp(self, lines: list):
        d = self.data
        lines.append(_section(17, "BETA DECOMPOSITION"))
        lines.append("")
        lines.append("  Position-level beta contribution to portfolio beta.")
        lines.append("")

        port_beta = d.factor_exposures["Beta"]
        weighted_betas = d.position_weights * d.position_betas
        total_weighted = float(np.sum(weighted_betas))

        lines.append(
            f"  {'Ticker':<8} {'Weight':>8} {'Beta':>8} {'Contrib':>8} "
            f"{'% of Port':>10}  {'Bar':>16}"
        )
        lines.append(
            f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} "
            f"{'-' * 10}  {'-' * 16}"
        )

        max_wb = max(abs(weighted_betas.max()), abs(weighted_betas.min()), 0.01)
        order = np.argsort(-weighted_betas)

        for idx in order:
            t = TOP_POSITIONS[idx]
            w = d.position_weights[idx] * 100
            b = d.position_betas[idx]
            wb = weighted_betas[idx]
            pct = (wb / total_weighted * 100) if total_weighted != 0 else 0
            bar = _bar_chart(wb, max_wb, 16)
            lines.append(
                f"  {t:<8} {w:>7.1f}% {b:>+7.2f} {wb:>+7.3f} "
                f"{pct:>9.1f}%  {bar}"
            )

        lines.append(
            f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} "
            f"{'-' * 10}  {'-' * 16}"
        )
        lines.append(
            f"  {'TOTAL':<8} {'100.0%':>8} {'':<8} {total_weighted:>+7.3f} {'100.0%':>10}"
        )
        lines.append("")
        lines.append(f"  Portfolio Beta (factor model):  {port_beta:+.3f}")
        lines.append(f"  Portfolio Beta (position sum):  {total_weighted:+.3f}")
        diff = abs(port_beta - total_weighted)
        lines.append(f"  Residual (model vs sum):        {diff:+.3f}")
        lines.append("")

    # -- 18 Liquidity Risk ---------------------------------------------------
    def _section_18_liquidity(self, lines: list):
        d = self.data
        lines.append(_section(18, "LIQUIDITY RISK (ADV RATIOS)"))
        lines.append("")
        lines.append("  Position size as fraction of 20-day Average Daily Volume.")
        lines.append("  Threshold: >5% ADV = caution,  >10% ADV = warning")
        lines.append("")

        lines.append(
            f"  {'Ticker':<8} {'Weight':>8} {'Position $':>14} "
            f"{'ADV Ratio':>10} {'Days to Exit':>12}  {'Status':<10}"
        )
        lines.append(
            f"  {'-' * 8} {'-' * 8} {'-' * 14} "
            f"{'-' * 10} {'-' * 12}  {'-' * 10}"
        )

        rng = self.data.rng
        warnings = 0
        for i, ticker in enumerate(TOP_POSITIONS):
            w = d.position_weights[i] * 100
            pos_val = d.position_weights[i] * d.nav_end
            adv = d.position_adv_ratio[i] * 100
            days_exit = round(1.0 / d.position_adv_ratio[i]) if d.position_adv_ratio[i] > 0 else 999
            if adv > 10:
                status = "WARNING"
                warnings += 1
            elif adv > 5:
                status = "CAUTION"
            else:
                status = "OK"
            lines.append(
                f"  {ticker:<8} {w:>7.1f}% {_fmt_dollar(pos_val)} "
                f"{adv:>9.2f}% {days_exit:>12}  {status:<10}"
            )

        lines.append("")
        lines.append(f"  Portfolio Avg ADV Ratio: {float(np.mean(d.position_adv_ratio)) * 100:.2f}%")
        if warnings > 0:
            lines.append(f"  WARNINGS: {warnings} position(s) exceed 10% ADV threshold")
        else:
            lines.append("  STATUS: All positions within liquidity bounds")
        lines.append("")

        # liquidity tier breakdown
        lines.append("  Liquidity Tier Breakdown:")
        lines.append(f"  {'Tier':<20} {'% of Portfolio':>14} {'Avg Days Exit':>14}")
        lines.append(f"  {'-' * 20} {'-' * 14} {'-' * 14}")
        tiers = [
            ("Tier 1 (< 1% ADV)", round(rng.uniform(50, 75), 1), round(rng.uniform(1, 3), 1)),
            ("Tier 2 (1-5% ADV)", round(rng.uniform(15, 30), 1), round(rng.uniform(3, 10), 1)),
            ("Tier 3 (5-10% ADV)", round(rng.uniform(3, 12), 1), round(rng.uniform(10, 30), 1)),
            ("Tier 4 (> 10% ADV)", round(rng.uniform(0, 5), 1), round(rng.uniform(30, 100), 1)),
        ]
        for tier, pct, days in tiers:
            lines.append(f"  {tier:<20} {pct:>13.1f}% {days:>13.1f}")
        lines.append("")

    # -- 19 Concentration Risk (HHI) -----------------------------------------
    def _section_19_concentration(self, lines: list):
        d = self.data
        lines.append(_section(19, "CONCENTRATION RISK (HHI)"))
        lines.append("")
        lines.append("  Herfindahl-Hirschman Index measures portfolio concentration.")
        lines.append("  HHI < 1500 = diversified | 1500-2500 = moderate | > 2500 = concentrated")
        lines.append("")

        lines.append(f"  HHI (positions):  {d.hhi:.1f}")

        if d.hhi < 1500:
            hhi_status = "DIVERSIFIED"
        elif d.hhi < 2500:
            hhi_status = "MODERATE"
        else:
            hhi_status = "CONCENTRATED"
        lines.append(f"  Status:           {hhi_status}")
        lines.append("")

        # position weight distribution
        lines.append("  Position Weight Distribution:")
        lines.append(f"  {'Ticker':<8} {'Weight':>8} {'Cumul':>8}  {'Bar':>30}")
        lines.append(f"  {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 30}")

        order = np.argsort(-d.position_weights)
        cumul = 0.0
        max_w = float(d.position_weights.max())
        for idx in order:
            t = TOP_POSITIONS[idx]
            w = d.position_weights[idx] * 100
            cumul += w
            n_blocks = int(round(d.position_weights[idx] / max_w * 25))
            bar = "#" * n_blocks
            lines.append(f"  {t:<8} {w:>7.1f}% {cumul:>7.1f}%  {bar}")

        lines.append("")

        # concentration metrics
        lines.append("  Concentration Metrics:")
        lines.append(f"  {'Metric':<36} {'Value':>10}")
        lines.append(f"  {'-' * 36} {'-' * 10}")
        lines.append(f"  {'Top 1 Position Weight':<36} {d.position_weights[order[0]] * 100:>9.1f}%")
        top3 = float(np.sum(d.position_weights[order[:3]])) * 100
        top5 = float(np.sum(d.position_weights[order[:5]])) * 100
        lines.append(f"  {'Top 3 Positions Weight':<36} {top3:>9.1f}%")
        lines.append(f"  {'Top 5 Positions Weight':<36} {top5:>9.1f}%")
        eff_n = 1.0 / float(np.sum(d.position_weights ** 2)) if float(np.sum(d.position_weights ** 2)) > 0 else 0
        lines.append(f"  {'Effective # of Positions (1/HHI)':<36} {eff_n:>10.1f}")
        lines.append(f"  {'Actual # of Positions':<36} {len(TOP_POSITIONS):>10}")
        lines.append("")

    # -- 20 Stress Test Summary ----------------------------------------------
    def _section_20_stress_summary(self, lines: list):
        d = self.data
        lines.append(_section(20, "STRESS TEST SUMMARY"))
        lines.append("")
        lines.append("  Combined stress test results across all risk dimensions.")
        lines.append("")

        # summary table
        lines.append(f"  {'Risk Category':<26} {'Score':>6} {'Limit':>6} {'Status':>10} {'Detail':<20}")
        lines.append(f"  {'-' * 26} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 20}")

        rng = self.data.rng
        categories = [
            ("Market Risk (VaR)",       round(rng.uniform(20, 80), 0), 80, "Parametric + Hist"),
            ("Tail Risk (CVaR)",        round(rng.uniform(15, 70), 0), 75, "ES 95% / 99%"),
            ("Concentration (HHI)",     round(rng.uniform(25, 65), 0), 70, f"HHI={d.hhi:.0f}"),
            ("Liquidity (ADV)",         round(rng.uniform(10, 50), 0), 60, "20d ADV ratio"),
            ("Correlation Risk",        round(rng.uniform(20, 60), 0), 65, "Avg pairwise rho"),
            ("Beta Deviation",          round(rng.uniform(10, 55), 0), 50, f"Beta={d.factor_exposures['Beta']:.2f}"),
            ("Scenario (worst case)",   round(rng.uniform(30, 85), 0), 80, "GFC/COVID/Rate"),
            ("Drawdown Risk",           round(rng.uniform(15, 60), 0), 70, f"MaxDD={d.max_dd:.1f}%"),
        ]

        flags = 0
        for cat, score, limit, detail in categories:
            status = "PASS" if score <= limit else "BREACH"
            if status == "BREACH":
                flags += 1
            lines.append(
                f"  {cat:<26} {score:>5.0f} {limit:>5.0f}   {status:>8} {detail:<20}"
            )

        lines.append(f"  {'-' * 26} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 20}")
        lines.append("")

        # overall risk score
        scores = [c[1] for c in categories]
        avg_score = sum(scores) / len(scores)
        lines.append(f"  Overall Risk Score: {avg_score:.0f} / 100")
        if avg_score < 40:
            risk_grade = "LOW"
        elif avg_score < 60:
            risk_grade = "MODERATE"
        elif avg_score < 75:
            risk_grade = "ELEVATED"
        else:
            risk_grade = "HIGH"
        lines.append(f"  Risk Grade:         {risk_grade}")
        lines.append(f"  Breaches:           {flags} of {len(categories)}")
        lines.append("")

        # action items
        lines.append("  Recommended Actions:")
        lines.append(f"  {'-' * 50}")
        if flags == 0:
            lines.append("  - No breaches detected. Portfolio within all limits.")
        else:
            if d.factor_exposures["Beta"] > 1.10:
                lines.append("  - REDUCE beta exposure — currently above 1.10 target cap")
            if d.hhi > 2500:
                lines.append("  - DIVERSIFY positions — HHI indicates concentration")
            if abs(d.max_dd) > 15:
                lines.append("  - REVIEW drawdown management — historical max DD exceeds 15%")
            if avg_score > 60:
                lines.append("  - GENERAL: Elevated aggregate risk — review sizing")
            lines.append("  - Monitor intraday for further deterioration")
        lines.append("")

        # trailing risk heatmap (mini ASCII)
        lines.append("  5-Day Risk Score History:")
        lines.append(f"  {'Day':<8} {'Score':>6}  {'Heatmap':>30}")
        lines.append(f"  {'-' * 8} {'-' * 6}  {'-' * 30}")
        for i in range(5, 0, -1):
            day_score = round(avg_score + rng.normal(0, 8), 0)
            day_score = max(0, min(100, day_score))
            n_blocks = int(day_score / 100 * 25)
            if day_score < 40:
                block_char = "."
            elif day_score < 60:
                block_char = "o"
            elif day_score < 75:
                block_char = "O"
            else:
                block_char = "#"
            bar = block_char * n_blocks
            label = f"T-{i}"
            lines.append(f"  {label:<8} {day_score:>5.0f}  {bar}")
        today_score = round(avg_score, 0)
        n_blocks = int(today_score / 100 * 25)
        block_char = "#" if today_score >= 75 else ("O" if today_score >= 60 else ("o" if today_score >= 40 else "."))
        bar = block_char * n_blocks
        lines.append(f"  {'Today':<8} {today_score:>5.0f}  {bar}  <--")
        lines.append("")
        lines.append("  Legend:  . = low   o = moderate   O = elevated   # = high")
        lines.append("")


# ---------------------------------------------------------------------------
#  Module-level convenience (matches project pattern)
# ---------------------------------------------------------------------------
def generate_open_report() -> str:
    """Convenience wrapper for morning open."""
    return PortfolioReportGenerator().generate_open_report()


def generate_close_report() -> str:
    """Convenience wrapper for evening close."""
    return PortfolioReportGenerator().generate_close_report()


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = PortfolioReportGenerator()
    print(gen.generate_open_report())
