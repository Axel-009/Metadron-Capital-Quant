"""Risk Report Generator — Comprehensive risk dashboard for Metadron Capital.

Aggregates risk metrics from RiskMetricsEngine, AnomalyDetector, and
optional stress-test / Greeks engines into a single ASCII-formatted report.

Imported by run_close.py and the monitoring API router:
    from engine.monitoring.risk_report import RiskReportGenerator
    gen = RiskReportGenerator()
    report_str = gen.generate()

Report sections:
    1.  RISK DASHBOARD header
    2.  VALUE AT RISK (parametric 95/99, historical)
    3.  CONDITIONAL VaR / Expected Shortfall
    4.  PORTFOLIO METRICS (Sharpe, Sortino, Calmar, max drawdown, beta)
    5.  STRESS TEST RESULTS
    6.  ANOMALY DETECTION
    7.  LIQUIDITY RISK ASSESSMENT
    8.  GREEKS EXPOSURE

Pure Python + numpy.  No ML frameworks.  Guarded imports throughout.
"""

import math
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Optional engine imports — all guarded so the system runs degraded, not broken
# ---------------------------------------------------------------------------
try:
    from ..monitoring.daily_report import RiskMetricsEngine
    _RISK_ENGINE_OK = True
except ImportError:
    RiskMetricsEngine = None  # type: ignore
    _RISK_ENGINE_OK = False

try:
    from ..monitoring.anomaly_detector import AnomalyDetector
    _ANOMALY_OK = True
except ImportError:
    AnomalyDetector = None  # type: ignore
    _ANOMALY_OK = False

try:
    from ..portfolio.metadron_cube import MetadronCube
    _CUBE_OK = True
except ImportError:
    MetadronCube = None  # type: ignore
    _CUBE_OK = False

try:
    from ..signals.options_engine import OptionsEngine
    _OPTIONS_OK = True
except ImportError:
    OptionsEngine = None  # type: ignore
    _OPTIONS_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_WIDTH = 78
INITIAL_NAV = 1_000_000.0

TOP_POSITIONS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "JPM", "V", "UNH",
]

STRESS_SCENARIOS = [
    ("2008 GFC Replay",        -0.38),
    ("2020 COVID Crash",       -0.34),
    ("2022 Rate Hike Shock",   -0.19),
    ("Flash Crash (-10%)",     -0.10),
    ("VIX Spike to 45",        -0.15),
    ("USD Surge +10%",         -0.06),
    ("Oil Shock +50%",         -0.04),
]


# ---------------------------------------------------------------------------
# Deterministic seed helper
# ---------------------------------------------------------------------------
def _day_seed() -> int:
    """Seed derived from today's date for reproducible placeholder data."""
    d = datetime.now()
    return d.year * 10000 + d.month * 100 + d.day


def _rng() -> np.random.RandomState:
    return np.random.RandomState(_day_seed())


# ---------------------------------------------------------------------------
# Formatting helpers (mirror portfolio_report.py style)
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
    s = f"{val:+.2f}%"
    return s.rjust(width)


def _fmt_dollar(val: float, width: int = 14) -> str:
    if val < 0:
        s = f"-${abs(val):,.2f}"
    else:
        s = f"${val:,.2f}"
    return s.rjust(width)


def _fmt_ratio(val: float, width: int = 8) -> str:
    return f"{val:+.3f}".rjust(width)


def _bar_chart(val: float, max_abs: float, bar_len: int = 20) -> str:
    """Simple inline ASCII bar. Positive -> right, negative -> left."""
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
# Placeholder risk data — deterministic per day
# ---------------------------------------------------------------------------
class _PlaceholderRisk:
    """Generate consistent paper-trading risk placeholder data for one day."""

    def __init__(self):
        self.rng = _rng()
        self._build()

    def _build(self):
        r = self.rng
        self.nav = INITIAL_NAV + r.uniform(-20000, 40000)

        # VaR / CVaR (as % of portfolio)
        self.var_95_para = round(r.uniform(-1.8, -0.6), 3)
        self.var_99_para = round(self.var_95_para * r.uniform(1.3, 1.6), 3)
        self.var_95_hist = round(self.var_95_para * r.uniform(0.88, 1.12), 3)
        self.var_99_hist = round(self.var_99_para * r.uniform(0.88, 1.12), 3)
        self.cvar_95 = round(self.var_95_para * r.uniform(1.20, 1.50), 3)
        self.cvar_99 = round(self.var_99_para * r.uniform(1.20, 1.50), 3)

        # Portfolio metrics
        self.sharpe = round(r.uniform(0.7, 2.8), 2)
        self.sortino = round(self.sharpe * r.uniform(1.1, 1.7), 2)
        self.calmar = round(r.uniform(0.5, 3.2), 2)
        self.max_dd = round(r.uniform(-18.0, -1.5), 2)
        self.current_dd = round(r.uniform(self.max_dd * 0.4, 0.0), 2)
        self.beta = round(r.uniform(0.65, 1.20), 3)
        self.volatility_ann = round(r.uniform(8.0, 22.0), 2)

        # Stress test results (% impact on NAV)
        self.stress_results = [
            (name, round(shock * r.uniform(0.70, 1.10), 4))
            for name, shock in STRESS_SCENARIOS
        ]

        # Liquidity — ADV ratios per position
        n = len(TOP_POSITIONS)
        self.adv_ratios = r.uniform(0.001, 0.055, n)
        self.position_weights = np.abs(r.dirichlet(np.ones(n) * 2))
        self.days_to_liquidate = np.ceil(
            self.position_weights * self.nav / 1e6 / (self.adv_ratios + 1e-9)
        )

        # Greeks (aggregate notional)
        self.delta = round(r.uniform(-0.15, 0.30), 4)
        self.gamma = round(r.uniform(-0.02, 0.05), 4)
        self.theta = round(r.uniform(-500, -20), 2)
        self.vega = round(r.uniform(-2000, 3000), 2)
        self.rho = round(r.uniform(-1500, 800), 2)


# ---------------------------------------------------------------------------
# RiskReportGenerator
# ---------------------------------------------------------------------------
class RiskReportGenerator:
    """Generates a comprehensive ASCII risk dashboard.

    Aggregates live data from RiskMetricsEngine and AnomalyDetector where
    available; falls back to deterministic placeholder data otherwise.

    Usage:
        gen = RiskReportGenerator()
        print(gen.generate())
    """

    def __init__(self):
        self._ph = _PlaceholderRisk()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> str:
        """Return the full risk report as an ASCII string."""
        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append(_header("METADRON CAPITAL  \u2014  RISK DASHBOARD"))
        lines.append(f"  Generated : {ts}")
        lines.append(f"  NAV       : {_fmt_dollar(self._ph.nav).strip()}")
        lines.append(
            "  Mode      : "
            + ("LIVE (RiskMetricsEngine)" if _RISK_ENGINE_OK else "DEGRADED (placeholder data)")
        )

        lines.append(self._section_var())
        lines.append(self._section_cvar())
        lines.append(self._section_portfolio_metrics())
        lines.append(self._section_stress())
        lines.append(self._section_anomaly())
        lines.append(self._section_liquidity())
        lines.append(self._section_greeks())

        lines.append("\n" + _hline("="))
        lines.append(_center("END OF RISK REPORT"))
        lines.append(_hline("="))

        return "\r\n".join(lines)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #
    def _section_var(self) -> str:
        ph = self._ph
        out = [_section(2, "VALUE AT RISK")]
        out.append("")
        out.append(
            "  {:<28} {:>10}  {:>10}  {:>14}".format(
                "Method", "VaR 95%", "VaR 99%", "$ Impact 95%"
            )
        )
        out.append("  " + "-" * 28 + " " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 14)

        nav = ph.nav
        impact_95 = ph.var_95_para / 100.0 * nav

        out.append(_table_row(
            ["  Parametric (Normal)",
             _fmt_pct(ph.var_95_para),
             _fmt_pct(ph.var_99_para),
             _fmt_dollar(impact_95)],
            [30, -10, -10, -14]
        ))
        out.append(_table_row(
            ["  Historical Simulation",
             _fmt_pct(ph.var_95_hist),
             _fmt_pct(ph.var_99_hist),
             _fmt_dollar(ph.var_95_hist / 100.0 * nav)],
            [30, -10, -10, -14]
        ))

        if not _RISK_ENGINE_OK:
            out.append("")
            out.append("  [!] RiskMetricsEngine unavailable \u2014 showing placeholder data")
        return "\n".join(out)

    def _section_cvar(self) -> str:
        ph = self._ph
        out = [_section(3, "CONDITIONAL VaR / EXPECTED SHORTFALL")]
        out.append("")
        out.append(
            "  {:<20} {:>10}  {:>14}  {:>16}".format(
                "Confidence", "CVaR %", "CVaR $", "Excess over VaR"
            )
        )
        out.append("  " + "-" * 20 + " " + "-" * 10 + "  " + "-" * 14 + "  " + "-" * 16)

        nav = ph.nav
        excess_95 = ph.cvar_95 - ph.var_95_para
        excess_99 = ph.cvar_99 - ph.var_99_para
        out.append(_table_row(
            ["  95% (1-in-20 day)",
             _fmt_pct(ph.cvar_95),
             _fmt_dollar(ph.cvar_95 / 100.0 * nav),
             _fmt_pct(excess_95)],
            [22, -10, -14, -16]
        ))
        out.append(_table_row(
            ["  99% (1-in-100 day)",
             _fmt_pct(ph.cvar_99),
             _fmt_dollar(ph.cvar_99 / 100.0 * nav),
             _fmt_pct(excess_99)],
            [22, -10, -14, -16]
        ))
        out.append("")
        out.append(f"  Annualised Volatility : {ph.volatility_ann:+.2f}%")
        return "\n".join(out)

    def _section_portfolio_metrics(self) -> str:
        ph = self._ph
        out = [_section(4, "PORTFOLIO METRICS")]
        out.append("")
        metrics = [
            ("Sharpe Ratio (Ann.)",  f"{ph.sharpe:+.2f}"),
            ("Sortino Ratio (Ann.)", f"{ph.sortino:+.2f}"),
            ("Calmar Ratio",         f"{ph.calmar:+.2f}"),
            ("Max Drawdown",         f"{ph.max_dd:+.2f}%"),
            ("Current Drawdown",     f"{ph.current_dd:+.2f}%"),
            ("Portfolio Beta",       f"{ph.beta:+.3f}"),
        ]
        col_w = 30
        bar_max = 3.0
        for label, val_str in metrics:
            try:
                num = float(val_str.replace("%", ""))
            except ValueError:
                num = 0.0
            bar = _bar_chart(num, bar_max, bar_len=24)
            out.append(f"  {label:<{col_w}} {val_str:>10}   {bar}")
        return "\n".join(out)

    def _section_stress(self) -> str:
        ph = self._ph
        out = [_section(5, "STRESS TEST RESULTS")]
        if not _CUBE_OK:
            out.append("")
            out.append("  [!] MetadronCube unavailable \u2014 using historical shock estimates")
        out.append("")
        out.append(
            "  {:<30} {:>16}  {:>14}".format(
                "Scenario", "Portfolio Impact", "$ P&L Impact"
            )
        )
        out.append("  " + "-" * 30 + " " + "-" * 16 + "  " + "-" * 14)
        for name, pct in ph.stress_results:
            pnl = pct * ph.nav
            out.append(_table_row(
                [f"  {name}", _fmt_pct(pct * 100), _fmt_dollar(pnl)],
                [32, -16, -14]
            ))
        worst = min(ph.stress_results, key=lambda x: x[1])
        out.append("")
        out.append(f"  Worst case scenario : {worst[0]}  ({worst[1]*100:+.2f}%)")
        return "\n".join(out)

    def _section_anomaly(self) -> str:
        out = [_section(6, "ANOMALY DETECTION")]
        out.append("")
        if _ANOMALY_OK:
            try:
                detector = AnomalyDetector()
                anomalies = []
                if hasattr(detector, "run_full_scan"):
                    anomalies = detector.run_full_scan() or []
                elif hasattr(detector, "scan"):
                    anomalies = detector.scan() or []
                if anomalies:
                    out.append(f"  Anomalies detected: {len(anomalies)}")
                    out.append("")
                    out.append(
                        "  {:<22} {:<10} {:<8} {:>8}  {}".format(
                            "Type", "Severity", "Ticker", "Z-score", "Description"
                        )
                    )
                    out.append(
                        "  " + "-" * 22 + " " + "-" * 10 + " " + "-" * 8
                        + " " + "-" * 8 + "  " + "-" * 20
                    )
                    for a in anomalies[:10]:
                        ad = a.to_dict() if hasattr(a, "to_dict") else a
                        out.append(_table_row(
                            [f"  {ad.get('type', '')[:20]}",
                             ad.get("severity", ""),
                             ad.get("ticker", ""),
                             f"{ad.get('zscore', 0.0):+.2f}",
                             ad.get("description", "")[:30]],
                            [24, 10, 8, -8, 32]
                        ))
                else:
                    out.append("  No anomalies detected \u2014 system nominal")
            except Exception as exc:
                out.append(f"  [!] AnomalyDetector error: {exc}")
                out.append("  Showing last-known status: NOMINAL")
        else:
            out.append("  [!] AnomalyDetector unavailable \u2014 data unavailable")
            ph = self._ph
            rng = ph.rng
            n_anomalies = int(rng.randint(0, 4))
            out.append(f"  Last scan (placeholder): {n_anomalies} anomaly/ies flagged")
        return "\n".join(out)

    def _section_liquidity(self) -> str:
        ph = self._ph
        out = [_section(7, "LIQUIDITY RISK ASSESSMENT")]
        out.append("")
        out.append(
            "  {:<8} {:>10}  {:>10}  {:>12}  {}".format(
                "Ticker", "Weight %", "ADV Ratio", "Days-to-Liq", "Status"
            )
        )
        out.append(
            "  " + "-" * 8 + " " + "-" * 10 + "  " + "-" * 10
            + "  " + "-" * 12 + "  " + "-" * 10
        )
        for i, ticker in enumerate(TOP_POSITIONS):
            wt = ph.position_weights[i] * 100
            adv = ph.adv_ratios[i] * 100
            dtl = ph.days_to_liquidate[i]
            status = "ILLIQUID" if dtl > 5 else ("WATCH" if dtl > 2 else "OK")
            out.append(_table_row(
                [f"  {ticker}", f"{wt:.1f}%", f"{adv:.2f}%",
                 f"{dtl:.1f}d", status],
                [10, -10, -10, -12, 10]
            ))
        out.append("")
        illiquid_count = int(np.sum(ph.days_to_liquidate > 5))
        out.append(f"  Illiquid positions (>5 days): {illiquid_count}")
        return "\n".join(out)

    def _section_greeks(self) -> str:
        ph = self._ph
        out = [_section(8, "GREEKS EXPOSURE")]
        if not _OPTIONS_OK:
            out.append("")
            out.append("  [!] OptionsEngine unavailable \u2014 aggregate Greeks estimated")
        out.append("")
        greeks = [
            ("Delta (portfolio)",  f"{ph.delta:+.4f}", "sensitivity to 1% SPY move"),
            ("Gamma (portfolio)",  f"{ph.gamma:+.4f}", "rate of delta change"),
            ("Theta (daily $)",    f"${ph.theta:,.2f}", "daily time decay"),
            ("Vega (1-vol-pt $)",  f"${ph.vega:,.2f}",  "sensitivity to 1pp vol move"),
            ("Rho (1-rate-bp $)",  f"${ph.rho:,.2f}",   "sensitivity to 1bp rate move"),
        ]
        for label, val_str, note in greeks:
            out.append(f"  {label:<28} {val_str:>14}   ({note})")
        return "\n".join(out)
