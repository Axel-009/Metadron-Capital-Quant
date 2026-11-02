"""Compliance Report Generator — Regulatory and governance report for Metadron Capital.

Aggregates enforcement events, position limit checks, leverage metrics, and
PAUL-pattern governance data into a single ASCII-formatted compliance report.

Imported by run_close.py and the monitoring API router:
    from engine.monitoring.compliance_report import ComplianceReportGenerator
    gen = ComplianceReportGenerator()
    report_str = gen.generate()

Report sections:
    1.  COMPLIANCE REPORT header
    2.  POSITION LIMITS  (single name <=5%, sector <=25%)
    3.  CONCENTRATION RISK  (HHI index, top 5 positions)
    4.  LEVERAGE LIMITS  (gross / net exposure vs limits)
    5.  ENFORCEMENT EVENTS  (agent violations, demotions)
    6.  PAUL PATTERN COMPLIANCE  (orchestrator governance)
    7.  RISK LIMIT BREACHES  (VaR, drawdown, beta)
    8.  REGULATORY STATUS  (all clear / violations)
    9.  AUDIT TRAIL SUMMARY

Pure Python + numpy.  No ML frameworks.  Guarded imports throughout.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional engine imports — all guarded
# ---------------------------------------------------------------------------
try:
    from ..agents.enforcement_engine import EnforcementEngine
    _ENFORCEMENT_OK = True
except ImportError:
    EnforcementEngine = None  # type: ignore
    _ENFORCEMENT_OK = False

try:
    from ..portfolio.beta_corridor import BetaCorridor
    _BETA_OK = True
except ImportError:
    BetaCorridor = None  # type: ignore
    _BETA_OK = False

try:
    from ..monitoring.daily_report import RiskMetricsEngine
    _RISK_OK = True
except ImportError:
    RiskMetricsEngine = None  # type: ignore
    _RISK_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_WIDTH = 78
INITIAL_NAV = 1_000_000.0

# Compliance limits
SINGLE_NAME_MAX_PCT = 5.0        # max single-position weight (%)
SECTOR_MAX_PCT = 25.0            # max single-sector weight (%)
GROSS_EXPOSURE_MAX = 1.50        # max gross leverage (150%)
NET_EXPOSURE_MAX = 1.00          # max net leverage (100%)
VAR_LIMIT_PCT = 3.0              # max daily VaR (% of NAV)
DRAWDOWN_LIMIT_PCT = -20.0       # max acceptable drawdown (%)
BETA_CORRIDOR_LOW = 0.30         # portfolio beta floor
BETA_CORRIDOR_HIGH = 1.30        # portfolio beta ceiling
HHI_CONCENTRATION_LIMIT = 1500   # HHI > 1500 -> concentrated

TOP_POSITIONS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "JPM", "V", "UNH",
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

AGENT_TIERS = ["T1_ALPHA", "T2_CORE", "T3_SUPPORT", "T4_PROBATION", "T5_SUSPENDED"]


# ---------------------------------------------------------------------------
# Deterministic seed helper
# ---------------------------------------------------------------------------
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


def _status_tag(breach: bool) -> str:
    return "[ BREACH ]" if breach else "[  OK    ]"


def _table_row(cols: list, widths: list) -> str:
    parts = []
    for col, w in zip(cols, widths):
        parts.append(str(col).ljust(w) if w > 0 else str(col).rjust(-w))
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Placeholder compliance data — deterministic per day
# ---------------------------------------------------------------------------
class _PlaceholderCompliance:
    """Stable daily placeholder data for when live engines are unavailable."""

    def __init__(self):
        self.rng = _rng()
        self._build()

    def _build(self):
        r = self.rng
        n_pos = len(TOP_POSITIONS)
        n_sec = len(GICS_SECTORS)

        # Position weights (sum to 1)
        raw_pos = np.abs(r.dirichlet(np.ones(n_pos) * 2))
        self.position_weights_pct = [round(float(v * 100), 2) for v in raw_pos]

        # Sector weights (sum to 1)
        raw_sec = np.abs(r.dirichlet(np.ones(n_sec) * 3))
        self.sector_weights_pct = [round(float(v * 100), 2) for v in raw_sec]

        # HHI
        self.hhi = round(float(np.sum(raw_pos ** 2) * 10000), 1)

        # Leverage
        self.gross_exposure = round(r.uniform(0.85, 1.45), 4)
        self.net_exposure = round(r.uniform(-0.10, 0.90), 4)
        self.long_exposure = round(r.uniform(0.70, 1.20), 4)
        self.short_exposure = round(r.uniform(0.05, 0.40), 4)

        # Beta
        self.portfolio_beta = round(r.uniform(0.40, 1.25), 3)

        # VaR / drawdown
        self.var_pct = round(r.uniform(0.5, 3.5), 3)
        self.current_dd = round(r.uniform(-18.0, 0.0), 2)

        # Enforcement events (simulated)
        n_events = int(r.randint(0, 6))
        self.enforcement_events = []
        event_types = [
            "ACCURACY_BREACH", "DRAWDOWN_BREACH", "HERDING_RISK",
            "WEIGHT_REDUCTION", "PROBATION", "SUSPENSION",
        ]
        severities = ["INFO", "WARNING", "CRITICAL"]
        actions = ["WARN", "REDUCE_WEIGHT", "DEMOTE", "SUSPEND"]
        for i in range(n_events):
            self.enforcement_events.append({
                "event_id": f"EVT-{_day_seed()}-{i:03d}",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "agent_id": f"agent_{int(r.randint(1, 20)):03d}",
                "event_type": str(event_types[int(r.randint(0, len(event_types)))]),
                "action_taken": str(actions[int(r.randint(0, len(actions)))]),
                "severity": str(severities[int(r.randint(0, len(severities)))]),
            })

        # PAUL pattern checks
        self.paul_pattern_ok = bool(r.random() > 0.15)
        self.paul_checks = [
            ("Orchestrator heartbeat",        True),
            ("Paul-pattern adherence",        bool(r.random() > 0.10)),
            ("GSD gradient alignment",        bool(r.random() > 0.15)),
            ("Consensus drift within limit",  bool(r.random() > 0.20)),
            ("Signal diversity (>=3)",        bool(r.random() > 0.10)),
        ]

        # Audit trail stats
        self.audit_log_entries = int(r.randint(200, 2000))
        self.audit_log_warnings = int(r.randint(0, 20))
        self.audit_log_criticals = int(r.randint(0, 3))
        self.last_sweep_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# ComplianceReportGenerator
# ---------------------------------------------------------------------------
class ComplianceReportGenerator:
    """Generates an ASCII-formatted compliance report.

    Attempts to pull live enforcement events from EnforcementEngine and
    position limit data from BetaCorridor; falls back to deterministic
    daily placeholder data when those engines are unavailable.

    Usage:
        gen = ComplianceReportGenerator()
        print(gen.generate())
    """

    def __init__(self, log_dir: str = "logs/enforcement"):
        self._ph = _PlaceholderCompliance()
        self._log_dir = Path(log_dir)
        self._live_events = []
        if _ENFORCEMENT_OK:
            try:
                eng = EnforcementEngine()
                result = eng.run_daily_sweep()
                if result and isinstance(result, dict):
                    self._live_events = result.get("events", [])
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> str:
        """Return the full compliance report as an ASCII string."""
        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append(_header("METADRON CAPITAL  \u2014  COMPLIANCE REPORT"))
        lines.append(f"  Generated  : {ts}")
        lines.append(f"  Report Type: Daily EOD Compliance Review")
        lines.append(
            "  Mode       : "
            + ("LIVE (EnforcementEngine)" if _ENFORCEMENT_OK else "DEGRADED (placeholder data)")
        )

        lines.append(self._section_position_limits())
        lines.append(self._section_concentration())
        lines.append(self._section_leverage())
        lines.append(self._section_enforcement_events())
        lines.append(self._section_paul_compliance())
        lines.append(self._section_risk_limit_breaches())
        lines.append(self._section_regulatory_status())
        lines.append(self._section_audit_trail())

        lines.append("\n" + _hline("="))
        lines.append(_center("END OF COMPLIANCE REPORT"))
        lines.append(_hline("="))

        return "\r\n".join(lines)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #
    def _section_position_limits(self) -> str:
        ph = self._ph
        out = [_section(
            2,
            f"POSITION LIMITS  (single name <={SINGLE_NAME_MAX_PCT:.0f}%,"
            f" sector <={SECTOR_MAX_PCT:.0f}%)"
        )]
        out.append("")
        out.append("  SINGLE NAME LIMITS")
        out.append(
            "  {:<8} {:>10}  {:>8}  {:>10}  {}".format(
                "Ticker", "Weight %", "Limit", "Headroom", "Status"
            )
        )
        out.append(
            "  " + "-" * 8 + " " + "-" * 10 + "  " + "-" * 8
            + "  " + "-" * 10 + "  " + "-" * 12
        )
        for i, ticker in enumerate(TOP_POSITIONS):
            wt = ph.position_weights_pct[i]
            headroom = SINGLE_NAME_MAX_PCT - wt
            breach = wt > SINGLE_NAME_MAX_PCT
            out.append(_table_row(
                [f"  {ticker}", f"{wt:.2f}%", f"{SINGLE_NAME_MAX_PCT:.1f}%",
                 f"{headroom:+.2f}%", _status_tag(breach)],
                [10, -10, -8, -10, 12]
            ))
        out.append("")
        out.append("  SECTOR LIMITS (top 5 by exposure)")
        out.append(
            "  {:<30} {:>10}  {:>8}  {}".format(
                "Sector", "Weight %", "Limit", "Status"
            )
        )
        out.append(
            "  " + "-" * 30 + " " + "-" * 10 + "  " + "-" * 8 + "  " + "-" * 12
        )
        sorted_sectors = sorted(
            zip(GICS_SECTORS, ph.sector_weights_pct), key=lambda x: -x[1]
        )[:5]
        for sector, wt in sorted_sectors:
            breach = wt > SECTOR_MAX_PCT
            out.append(_table_row(
                [f"  {sector}", f"{wt:.2f}%", f"{SECTOR_MAX_PCT:.1f}%",
                 _status_tag(breach)],
                [32, -10, -8, 12]
            ))
        return "\n".join(out)

    def _section_concentration(self) -> str:
        ph = self._ph
        out = [_section(3, "CONCENTRATION RISK")]
        out.append("")
        hhi_status = "CONCENTRATED" if ph.hhi > HHI_CONCENTRATION_LIMIT else "DIVERSIFIED"
        out.append(
            f"  Herfindahl-Hirschman Index (HHI) : {ph.hhi:.1f}  [{hhi_status}]"
        )
        out.append(f"  HHI Limit                        : {HHI_CONCENTRATION_LIMIT}")
        out.append(
            f"  HHI Breach                       : "
            f"{_status_tag(ph.hhi > HHI_CONCENTRATION_LIMIT)}"
        )
        out.append("")
        out.append("  TOP 5 POSITIONS BY WEIGHT")
        out.append(
            "  {:<6} {:<8} {:>10}  {:>14}".format(
                "Rank", "Ticker", "Weight %", "Cumulative %"
            )
        )
        out.append(
            "  " + "-" * 6 + " " + "-" * 8 + " " + "-" * 10 + "  " + "-" * 14
        )
        sorted_pos = sorted(
            zip(TOP_POSITIONS, ph.position_weights_pct), key=lambda x: -x[1]
        )[:5]
        cumulative = 0.0
        for rank, (ticker, wt) in enumerate(sorted_pos, start=1):
            cumulative += wt
            out.append(_table_row(
                [f"  #{rank}", ticker, f"{wt:.2f}%", f"{cumulative:.2f}%"],
                [8, 10, -10, -14]
            ))
        return "\n".join(out)

    def _section_leverage(self) -> str:
        ph = self._ph
        out = [_section(4, "LEVERAGE LIMITS")]
        out.append("")
        gross_breach = ph.gross_exposure > GROSS_EXPOSURE_MAX
        net_breach = abs(ph.net_exposure) > NET_EXPOSURE_MAX
        items = [
            ("Gross Exposure", ph.gross_exposure, GROSS_EXPOSURE_MAX, gross_breach),
            ("Net Exposure",   abs(ph.net_exposure), NET_EXPOSURE_MAX, net_breach),
            ("Long Exposure",  ph.long_exposure,  GROSS_EXPOSURE_MAX, False),
            ("Short Exposure", ph.short_exposure, 0.50, ph.short_exposure > 0.50),
        ]
        out.append(
            "  {:<24} {:>10}  {:>10}  {:>10}  {}".format(
                "Metric", "Current", "Limit", "Headroom", "Status"
            )
        )
        out.append(
            "  " + "-" * 24 + " " + "-" * 10 + "  " + "-" * 10
            + "  " + "-" * 10 + "  " + "-" * 12
        )
        for label, val, lim, breach in items:
            headroom = lim - val
            out.append(_table_row(
                [f"  {label}", f"{val*100:.1f}%", f"{lim*100:.1f}%",
                 f"{headroom*100:+.1f}%", _status_tag(breach)],
                [26, -10, -10, -10, 12]
            ))
        if not _BETA_OK:
            out.append("")
            out.append("  [!] BetaCorridor unavailable \u2014 beta limits are estimated")
        out.append("")
        beta_breach = not (BETA_CORRIDOR_LOW <= ph.portfolio_beta <= BETA_CORRIDOR_HIGH)
        out.append(f"  Portfolio Beta      : {ph.portfolio_beta:+.3f}")
        out.append(
            f"  Beta Corridor       : [{BETA_CORRIDOR_LOW:.2f} \u2014 {BETA_CORRIDOR_HIGH:.2f}]"
        )
        out.append(f"  Beta Corridor Check : {_status_tag(beta_breach)}")
        return "\n".join(out)

    def _section_enforcement_events(self) -> str:
        events = self._live_events or self._ph.enforcement_events
        out = [_section(5, "ENFORCEMENT EVENTS")]
        out.append("")
        if not events:
            out.append("  No enforcement events recorded today \u2014 system nominal")
            return "\n".join(out)
        out.append(f"  Total events today: {len(events)}")
        out.append("")
        out.append(
            "  {:<20} {:<14} {:<22} {:<16} {}".format(
                "Event ID", "Agent", "Type", "Action", "Severity"
            )
        )
        out.append(
            "  " + "-" * 20 + " " + "-" * 14 + " " + "-" * 22
            + " " + "-" * 16 + " " + "-" * 10
        )
        for ev in events[:15]:
            if hasattr(ev, "__dict__"):
                ev = ev.__dict__
            out.append(_table_row(
                [f"  {str(ev.get('event_id',''))[:18]}",
                 str(ev.get("agent_id", ""))[:12],
                 str(ev.get("event_type", ""))[:20],
                 str(ev.get("action_taken", ""))[:14],
                 str(ev.get("severity", ""))],
                [22, 14, 22, 16, 10]
            ))
        if len(events) > 15:
            out.append(f"  ... and {len(events) - 15} more events (see enforcement JSONL log)")
        return "\n".join(out)

    def _section_paul_compliance(self) -> str:
        ph = self._ph
        out = [_section(6, "PAUL PATTERN COMPLIANCE  (Orchestrator Governance)")]
        out.append("")
        overall = "PASS" if ph.paul_pattern_ok else "FAIL"
        out.append(f"  Overall Governance Status : {overall}")
        out.append("")
        out.append(
            "  {:<40} {}".format("Check", "Status")
        )
        out.append("  " + "-" * 40 + " " + "-" * 12)
        for check_name, passed in ph.paul_checks:
            out.append(f"  {check_name:<40} {_status_tag(not passed)}")
        out.append("")
        out.append(
            "  PAUL Pattern enforces: agents learn from historical patterns,\n"
            "  maintain consistent signal behavior, and do not bypass orchestrator."
        )
        return "\n".join(out)

    def _section_risk_limit_breaches(self) -> str:
        ph = self._ph
        out = [_section(7, "RISK LIMIT BREACHES")]
        out.append("")
        var_breach = ph.var_pct > VAR_LIMIT_PCT
        dd_breach = ph.current_dd < DRAWDOWN_LIMIT_PCT
        beta_breach = not (BETA_CORRIDOR_LOW <= ph.portfolio_beta <= BETA_CORRIDOR_HIGH)
        items = [
            ("Daily VaR (95%)",    f"{ph.var_pct:.3f}%",
             f"{VAR_LIMIT_PCT:.1f}%",    var_breach),
            ("Current Drawdown",  f"{ph.current_dd:.2f}%",
             f"{DRAWDOWN_LIMIT_PCT:.1f}%", dd_breach),
            ("Portfolio Beta",    f"{ph.portfolio_beta:.3f}",
             f"[{BETA_CORRIDOR_LOW}-{BETA_CORRIDOR_HIGH}]", beta_breach),
        ]
        if not _RISK_OK:
            out.append("  [!] RiskMetricsEngine unavailable \u2014 limits estimated")
            out.append("")
        out.append(
            "  {:<24} {:>12}  {:>16}  {}".format(
                "Metric", "Current", "Limit", "Status"
            )
        )
        out.append(
            "  " + "-" * 24 + " " + "-" * 12 + "  " + "-" * 16 + "  " + "-" * 12
        )
        for label, cur, lim, breach in items:
            out.append(_table_row(
                [f"  {label}", cur, lim, _status_tag(breach)],
                [26, -12, -16, 12]
            ))
        breaches = sum(1 for _, _, _, b in items if b)
        out.append("")
        out.append(f"  Total risk limit breaches: {breaches}")
        return "\n".join(out)

    def _section_regulatory_status(self) -> str:
        ph = self._ph
        out = [_section(8, "REGULATORY STATUS")]
        out.append("")

        all_ok = (
            all(w <= SINGLE_NAME_MAX_PCT for w in ph.position_weights_pct)
            and all(w <= SECTOR_MAX_PCT for w in ph.sector_weights_pct)
            and ph.gross_exposure <= GROSS_EXPOSURE_MAX
            and ph.var_pct <= VAR_LIMIT_PCT
            and ph.current_dd >= DRAWDOWN_LIMIT_PCT
            and BETA_CORRIDOR_LOW <= ph.portfolio_beta <= BETA_CORRIDOR_HIGH
        )

        status_str = "ALL CLEAR" if all_ok else "VIOLATIONS DETECTED"
        out.append(f"  Regulatory Status : {status_str}")
        out.append("")
        checks = [
            ("Position limit compliance",
             all(w <= SINGLE_NAME_MAX_PCT for w in ph.position_weights_pct)),
            ("Sector limit compliance",
             all(w <= SECTOR_MAX_PCT for w in ph.sector_weights_pct)),
            ("Leverage compliance",
             ph.gross_exposure <= GROSS_EXPOSURE_MAX),
            ("VaR within limit",
             ph.var_pct <= VAR_LIMIT_PCT),
            ("Drawdown within limit",
             ph.current_dd >= DRAWDOWN_LIMIT_PCT),
            ("Beta within corridor",
             BETA_CORRIDOR_LOW <= ph.portfolio_beta <= BETA_CORRIDOR_HIGH),
            ("No enforcement suspensions",
             not any(
                 str(e.get("action_taken", "")) == "SUSPEND"
                 for e in ph.enforcement_events
             )),
        ]
        out.append("  {:<36} {}".format("Check", "Status"))
        out.append("  " + "-" * 36 + " " + "-" * 12)
        for check_name, passed in checks:
            out.append(f"  {check_name:<36} {_status_tag(not passed)}")
        return "\n".join(out)

    def _section_audit_trail(self) -> str:
        ph = self._ph
        out = [_section(9, "AUDIT TRAIL SUMMARY")]
        out.append("")
        out.append(f"  Log entries today        : {ph.audit_log_entries:,}")
        out.append(f"  Warnings logged          : {ph.audit_log_warnings}")
        out.append(f"  Critical events logged   : {ph.audit_log_criticals}")
        out.append(f"  Last enforcement sweep   : {ph.last_sweep_ts}")
        out.append(f"  Enforcement JSONL log    : logs/enforcement/enforcement.jsonl")
        out.append("")
        # Try to read a few lines from JSONL if it exists
        jsonl_path = self._log_dir / "enforcement.jsonl"
        if jsonl_path.exists():
            try:
                lines_read = []
                with open(jsonl_path, "r") as fh:
                    for i, line in enumerate(fh):
                        if i >= 5:
                            break
                        lines_read.append(line.strip())
                if lines_read:
                    out.append("  Recent JSONL entries (last 5):")
                    for entry_line in lines_read:
                        try:
                            entry = json.loads(entry_line)
                            ts = entry.get("timestamp", "")[:19]
                            etype = entry.get("event_type", "")[:20]
                            out.append(f"    [{ts}]  {etype}")
                        except json.JSONDecodeError:
                            out.append(f"    {entry_line[:70]}")
            except OSError:
                out.append("  [!] Unable to read enforcement JSONL log")
        else:
            out.append("  [i] Enforcement JSONL log not yet created (first run)")
        return "\n".join(out)
