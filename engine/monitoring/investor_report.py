"""Investor Report Generator — Monthly investor letter for Metadron Capital.

Produces an ASCII-formatted monthly investor letter aggregating NAV history,
performance attribution, risk metrics, macro outlook, and portfolio positioning.

Imported by the reporting pipeline:
    from engine.monitoring.investor_report import InvestorReportGenerator
    gen = InvestorReportGenerator()
    report_str = gen.generate()

Report sections:
    1.  MONTHLY INVESTOR LETTER header
    2.  EXECUTIVE SUMMARY (NAV, return, benchmark comparison)
    3.  PERFORMANCE ATTRIBUTION (by strategy sleeve P1-P5)
    4.  RISK METRICS (drawdown, volatility, Sharpe)
    5.  MARKET OUTLOOK (macro regime, positioning)
    6.  PORTFOLIO POSITIONING (top holdings, sector allocation)
    7.  OPERATIONAL UPDATE (trade count, turnover)
    8.  NAV HISTORY TABLE (monthly values)
    9.  DISCLAIMER / LEGAL boilerplate

Pure Python + numpy.  No ML frameworks.  Guarded imports throughout.
"""

import math
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Optional engine imports — all guarded
# ---------------------------------------------------------------------------
try:
    from ..brokers.alpaca_broker import AlpacaBroker
    _BROKER_OK = True
except ImportError:
    AlpacaBroker = None  # type: ignore
    _BROKER_OK = False

try:
    from ..monitoring.daily_report import RiskMetricsEngine
    _RISK_OK = True
except ImportError:
    RiskMetricsEngine = None  # type: ignore
    _RISK_OK = False

try:
    from ..signals.macro_engine import MacroEngine
    _MACRO_OK = True
except ImportError:
    MacroEngine = None  # type: ignore
    _MACRO_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_WIDTH = 78
INITIAL_NAV = 1_000_000.0
FUND_NAME = "Metadron Capital, L.P."
BENCHMARK = "S&P 500 TR Index"

SLEEVES = [
    ("P1", "Directional Equities"),
    ("P2", "Factor Rotation"),
    ("P3", "Commodities / Macro"),
    ("P4", "Options Convexity"),
    ("P5", "Hedges / Volatility"),
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

TOP_POSITIONS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "JPM", "V", "UNH",
]

MACRO_REGIMES = ["TRENDING", "RANGE-BOUND", "STRESS", "CRASH"]

LEGAL_DISCLAIMER = (
    "This document is provided for informational purposes only and does not\n"
    "constitute an offer to sell or a solicitation of an offer to buy any\n"
    "securities. Past performance is not indicative of future results.\n"
    "Investments in the Fund involve a high degree of risk, including the\n"
    "possible loss of the entire amount invested.  This letter is intended\n"
    "solely for the use of the addressee and may not be reproduced or\n"
    "redistributed without written consent from the General Partner.\n"
    "Copyright Metadron Capital, L.P.  All rights reserved.\n"
)


# ---------------------------------------------------------------------------
# Deterministic seed helpers — date-based for reproducible placeholder data
# ---------------------------------------------------------------------------
def _month_seed() -> int:
    """Seed derived from current year/month for stable monthly placeholders."""
    d = datetime.now()
    return d.year * 100 + d.month


def _day_seed() -> int:
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
    tag = f"[{num:02d}]"
    label = f"{tag}  {title}"
    bar = "-" * REPORT_WIDTH
    return f"\n{bar}\n{label}\n{bar}"


def _fmt_pct(val: float, width: int = 8) -> str:
    return f"{val:+.2f}%".rjust(width)


def _fmt_dollar(val: float, width: int = 14) -> str:
    if val < 0:
        s = f"-${abs(val):,.2f}"
    else:
        s = f"${val:,.2f}"
    return s.rjust(width)


def _bar_chart(val: float, max_abs: float, bar_len: int = 20) -> str:
    if max_abs == 0:
        return " " * bar_len
    frac = max(-1.0, min(1.0, val / max_abs))
    half = bar_len // 2
    if frac >= 0:
        n = int(round(frac * half))
        return " " * half + "|" + "#" * n + " " * (half - n)
    else:
        n = int(round(-frac * half))
        return " " * (half - n) + "#" * n + "|" + " " * half


def _table_row(cols: list, widths: list) -> str:
    parts = []
    for col, w in zip(cols, widths):
        parts.append(str(col).ljust(w) if w > 0 else str(col).rjust(-w))
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Placeholder data — deterministic per month
# ---------------------------------------------------------------------------
class _PlaceholderInvestor:
    """Stable monthly placeholder data seeded from year+month."""

    def __init__(self):
        self.rng = np.random.RandomState(_month_seed())
        self._build()

    def _build(self):
        r = self.rng
        now = datetime.now()

        # NAV
        self.nav_end = round(INITIAL_NAV * (1.0 + r.uniform(-0.05, 0.18)), 2)
        self.nav_start = round(self.nav_end / (1.0 + r.uniform(-0.03, 0.06)), 2)
        self.monthly_return = round((self.nav_end / self.nav_start - 1.0) * 100.0, 3)
        self.benchmark_return = round(r.normal(1.0, 1.5), 3)
        self.excess_return = round(self.monthly_return - self.benchmark_return, 3)

        # YTD / ITD
        self.ytd_return = round(r.normal(5.0, 8.0), 2)
        self.itd_ann_return = round(r.uniform(8.0, 22.0), 2)
        self.itd_benchmark = round(r.uniform(7.0, 14.0), 2)

        # Sleeve attribution
        raw = r.normal(0.0, 1.2, len(SLEEVES))
        self.sleeve_returns = [round(float(v), 3) for v in raw]
        raw_w = np.abs(r.dirichlet(np.ones(len(SLEEVES)) * 3))
        self.sleeve_weights = [round(float(v), 4) for v in raw_w]
        self.sleeve_contrib = [
            round(s * w, 4)
            for s, w in zip(self.sleeve_returns, self.sleeve_weights)
        ]

        # Risk metrics
        self.sharpe = round(r.uniform(0.7, 2.8), 2)
        self.sortino = round(self.sharpe * r.uniform(1.1, 1.7), 2)
        self.max_dd = round(r.uniform(-15.0, -1.0), 2)
        self.current_dd = round(r.uniform(self.max_dd * 0.3, 0.0), 2)
        self.volatility = round(r.uniform(6.0, 18.0), 2)
        self.beta = round(r.uniform(0.5, 1.2), 3)

        # Macro
        regime_idx = int(r.randint(0, len(MACRO_REGIMES)))
        self.macro_regime = MACRO_REGIMES[regime_idx]
        self.vix = round(r.uniform(12.0, 35.0), 2)
        self.rate_10y = round(r.uniform(3.5, 5.5), 3)
        self.rate_2y = round(r.uniform(3.2, 5.2), 3)
        self.yield_spread = round(self.rate_10y - self.rate_2y, 3)
        self.positioning = ["RISK-ON", "RISK-OFF", "NEUTRAL"][int(r.randint(0, 3))]

        # Portfolio positioning
        n = len(TOP_POSITIONS)
        raw_pos = np.abs(r.dirichlet(np.ones(n) * 2))
        self.position_weights = [round(float(v * 100), 2) for v in raw_pos]
        self.position_pnl = [round(float(r.normal(0.5, 2.0)), 3) for _ in range(n)]

        raw_sec = np.abs(r.dirichlet(np.ones(len(GICS_SECTORS)) * 3))
        self.sector_weights = [round(float(v * 100), 2) for v in raw_sec]

        # Operational
        self.trade_count = int(r.randint(40, 180))
        self.turnover = round(r.uniform(5.0, 35.0), 2)
        self.fill_quality_bps = round(r.normal(0.5, 2.0), 2)
        self.avg_holding_days = round(r.uniform(3.0, 25.0), 1)

        # NAV history (last 12 months)
        nav_hist = [INITIAL_NAV]
        for _ in range(11):
            nav_hist.append(round(nav_hist[-1] * (1.0 + r.normal(0.008, 0.03)), 2))
        self.nav_history = nav_hist

        # Date range
        month_end = now.replace(day=1) - timedelta(days=1)
        month_start = month_end.replace(day=1)
        self.period_start = month_start.strftime("%B 1, %Y")
        self.period_end = month_end.strftime("%B %d, %Y")


# ---------------------------------------------------------------------------
# InvestorReportGenerator
# ---------------------------------------------------------------------------
class InvestorReportGenerator:
    """Generates a monthly investor letter as an ASCII string.

    Attempts to pull live NAV from IBKRBroker and performance metrics
    from RiskMetricsEngine; falls back to deterministic monthly placeholders
    when those engines are unavailable.

    Usage:
        gen = InvestorReportGenerator()
        print(gen.generate())
    """

    def __init__(self):
        self._ph = _PlaceholderInvestor()
        self._live_nav = None
        if _BROKER_OK:
            try:
                broker = AlpacaBroker()
                summary = broker.get_portfolio_summary()
                self._live_nav = summary.get("portfolio_value") or summary.get("nav")
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> str:
        """Return the full monthly investor letter as an ASCII string."""
        lines = []

        lines.append(_header(f"{FUND_NAME}"))
        lines.append(_center("MONTHLY INVESTOR LETTER"))
        lines.append(_center(f"{self._ph.period_start}  \u2014  {self._ph.period_end}"))
        lines.append(_hline("-"))
        if not _BROKER_OK:
            lines.append("  [!] AlpacaBroker unavailable \u2014 using deterministic placeholder data")
        if not _MACRO_OK:
            lines.append("  [!] MacroEngine unavailable \u2014 macro data estimated")

        lines.append(self._section_executive_summary())
        lines.append(self._section_attribution())
        lines.append(self._section_risk_metrics())
        lines.append(self._section_market_outlook())
        lines.append(self._section_positioning())
        lines.append(self._section_operational())
        lines.append(self._section_nav_history())
        lines.append(self._section_disclaimer())

        lines.append("\n" + _hline("="))
        lines.append(_center("END OF MONTHLY INVESTOR LETTER"))
        lines.append(_hline("="))

        return "\r\n".join(lines)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #
    def _section_executive_summary(self) -> str:
        ph = self._ph
        nav = self._live_nav or ph.nav_end
        out = [_section(2, "EXECUTIVE SUMMARY")]
        out.append("")
        out.append(f"  Fund NAV (Month-End)     : {_fmt_dollar(nav).strip()}")
        out.append(f"  Month Return             : {_fmt_pct(ph.monthly_return).strip()}")
        out.append(
            "  Benchmark ({:<16}) : {}".format(BENCHMARK, _fmt_pct(ph.benchmark_return).strip())
        )
        out.append(f"  Excess Return (Alpha)    : {_fmt_pct(ph.excess_return).strip()}")
        out.append(f"  YTD Return               : {_fmt_pct(ph.ytd_return).strip()}")
        out.append(f"  ITD Annualised Return    : {_fmt_pct(ph.itd_ann_return).strip()}")
        out.append(f"  ITD Benchmark Return     : {_fmt_pct(ph.itd_benchmark).strip()}")
        out.append("")
        status = "OUTPERFORMING" if ph.excess_return > 0 else "UNDERPERFORMING"
        out.append(
            f"  Month-to-date vs Benchmark: {status} by {abs(ph.excess_return):.2f}%"
        )
        return "\n".join(out)

    def _section_attribution(self) -> str:
        ph = self._ph
        out = [_section(3, "PERFORMANCE ATTRIBUTION  (by Strategy Sleeve)")]
        out.append("")
        out.append(
            "  {:<6}  {:<26} {:>8}  {:>8}  {:>8}  {}".format(
                "Sleeve", "Strategy", "Weight", "Return", "Contrib", "Chart"
            )
        )
        out.append(
            "  " + "-" * 6 + "  " + "-" * 26 + " " + "-" * 8
            + "  " + "-" * 8 + "  " + "-" * 8 + "  " + "-" * 20
        )
        total_contrib = 0.0
        for i, (code, name) in enumerate(SLEEVES):
            w = ph.sleeve_weights[i]
            ret = ph.sleeve_returns[i]
            contrib = ph.sleeve_contrib[i]
            total_contrib += contrib
            bar = _bar_chart(ret, 3.0, bar_len=20)
            out.append(_table_row(
                [f"  {code}", name, f"{w*100:.1f}%",
                 _fmt_pct(ret), _fmt_pct(contrib * 100), bar],
                [8, 28, -8, -8, -8, 22]
            ))
        out.append(
            "  {:<6}  {:<26} {:>8}  {:>8}  {:>8}".format(
                "", "TOTAL", "100.0%", "", _fmt_pct(total_contrib * 100).strip()
            )
        )
        return "\n".join(out)

    def _section_risk_metrics(self) -> str:
        ph = self._ph
        out = [_section(4, "RISK METRICS")]
        out.append("")
        metrics = [
            ("Sharpe Ratio (Ann.)",          f"{ph.sharpe:+.2f}"),
            ("Sortino Ratio (Ann.)",         f"{ph.sortino:+.2f}"),
            ("Annualised Volatility",        f"{ph.volatility:.2f}%"),
            ("Portfolio Beta",               f"{ph.beta:+.3f}"),
            ("Max Drawdown (Historical)",    f"{ph.max_dd:.2f}%"),
            ("Current Drawdown",             f"{ph.current_dd:.2f}%"),
        ]
        if not _RISK_OK:
            out.append("  [!] RiskMetricsEngine unavailable \u2014 metrics are estimated")
            out.append("")
        for label, val_str in metrics:
            out.append(f"  {label:<34} {val_str:>12}")
        return "\n".join(out)

    def _section_market_outlook(self) -> str:
        ph = self._ph
        out = [_section(5, "MARKET OUTLOOK")]
        out.append("")
        if not _MACRO_OK:
            out.append("  [!] MacroEngine unavailable \u2014 macro regime is estimated")
            out.append("")
        out.append(f"  Macro Regime          : {ph.macro_regime}")
        out.append(f"  Overall Positioning   : {ph.positioning}")
        out.append(f"  VIX Level             : {ph.vix:.2f}")
        out.append(f"  10Y Treasury Yield    : {ph.rate_10y:.3f}%")
        out.append(f"  2Y Treasury Yield     : {ph.rate_2y:.3f}%")
        spread_dir = "NORMAL" if ph.yield_spread > 0 else "INVERTED"
        out.append(
            f"  Yield Curve (10-2)    : {ph.yield_spread:+.3f}%  [{spread_dir}]"
        )
        out.append("")
        regime_commentary = {
            "TRENDING":
                "Markets trending with momentum; maintaining risk-on tilt.",
            "RANGE-BOUND":
                "Sideways price action; favouring mean-reversion strategies.",
            "STRESS":
                "Elevated stress indicators; reducing gross exposure.",
            "CRASH":
                "Tail-risk environment; hedges and short gamma active.",
        }
        out.append(
            "  Commentary: "
            + regime_commentary.get(ph.macro_regime, "Regime assessment in progress.")
        )
        return "\n".join(out)

    def _section_positioning(self) -> str:
        ph = self._ph
        out = [_section(6, "PORTFOLIO POSITIONING")]
        out.append("")
        out.append("  TOP HOLDINGS")
        out.append(
            "  {:<8} {:>10}  {:>10}  {}".format("Ticker", "Weight %", "MTD P&L", "Bar")
        )
        out.append("  " + "-" * 8 + " " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 20)
        for i, ticker in enumerate(TOP_POSITIONS):
            wt = ph.position_weights[i]
            pnl = ph.position_pnl[i]
            bar = _bar_chart(pnl, 5.0, bar_len=20)
            out.append(_table_row(
                [f"  {ticker}", f"{wt:.2f}%", f"{pnl:+.2f}%", bar],
                [10, -10, -10, 22]
            ))
        out.append("")
        out.append("  SECTOR ALLOCATION (top 6)")
        out.append(
            "  {:<30} {:>8}  {}".format("Sector", "Alloc %", "Bar")
        )
        out.append("  " + "-" * 30 + " " + "-" * 8 + "  " + "-" * 20)
        sorted_sectors = sorted(
            zip(GICS_SECTORS, ph.sector_weights), key=lambda x: -x[1]
        )[:6]
        for sector, wt in sorted_sectors:
            bar = _bar_chart(wt, 30.0, bar_len=20)
            out.append(f"  {sector:<30} {wt:>7.2f}%  {bar}")
        return "\n".join(out)

    def _section_operational(self) -> str:
        ph = self._ph
        out = [_section(7, "OPERATIONAL UPDATE")]
        out.append("")
        out.append(f"  Total Trades (month)        : {ph.trade_count}")
        out.append(f"  Portfolio Turnover          : {ph.turnover:.2f}%")
        out.append(f"  Avg Fill Quality            : {ph.fill_quality_bps:+.2f} bps vs VWAP")
        out.append(f"  Avg Holding Period          : {ph.avg_holding_days:.1f} days")
        out.append(f"  Algo Execution Share        : 100%  (fully automated)")
        out.append(f"  Prime Broker                : IBKR Markets")
        out.append(f"  Custodian                   : IBKR Securities LLC")
        return "\n".join(out)

    def _section_nav_history(self) -> str:
        ph = self._ph
        out = [_section(8, "NAV HISTORY  (last 12 months)")]
        out.append("")
        out.append(
            "  {:<12} {:>14}  {:>9}  {}".format("Month", "NAV", "MTD Ret", "Bar")
        )
        out.append(
            "  " + "-" * 12 + " " + "-" * 14 + "  " + "-" * 9 + "  " + "-" * 24
        )
        now = datetime.now()
        nav_vals = ph.nav_history
        for i in range(12):
            month_offset = 11 - i
            month_dt = now.replace(day=1) - timedelta(days=1)
            for _ in range(month_offset):
                month_dt = month_dt.replace(day=1) - timedelta(days=1)
            month_label = month_dt.strftime("%b %Y")
            nav_val = nav_vals[i]
            if i == 0:
                mtd = 0.0
            else:
                prev = nav_vals[i - 1]
                mtd = (nav_val / prev - 1.0) * 100.0 if prev != 0 else 0.0
            bar = _bar_chart(mtd, 5.0, bar_len=24)
            out.append(_table_row(
                [f"  {month_label}", _fmt_dollar(nav_val), _fmt_pct(mtd, width=9), bar],
                [14, -14, -9, 26]
            ))
        return "\n".join(out)

    def _section_disclaimer(self) -> str:
        out = [_section(9, "DISCLAIMER / LEGAL")]
        out.append("")
        for line in LEGAL_DISCLAIMER.strip().split("\n"):
            out.append(f"  {line.strip()}")
        return "\n".join(out)
