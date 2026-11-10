"""Position protection, option expiry handling, event tail hedges, daily-loss breaker, CVaR check.

    equity / ETF positions   peak-to-now drawdown of the position's price since entry:
                             −6% → TRIM to half (once per peak), −10% → CLOSE
    long options             close when DTE ≤ 2 (straddles: also after the event passes); stop at −50% of
                             premium paid except LEAPS (≥ 270 DTE at entry, held for the recovery thesis);
                             LEAPS are rolled (closed, re-evaluated) when DTE < 90
    event tail hedge         for a long equity position ≥ 3% NAV with a known event (earnings, binary) within
                             5 days: buy a ~5% OTM put expiring after the event, premium ≤ 0.3% NAV per name;
                             the hedge is closed once the event has passed (days_to_event < 0)
    cooldown                 a trim or a stop holds for PROTECT_COOLDOWN_DAYS: the name's target is capped at the
                             shares left (0 after a close / resting-stop fill), so the next morning's research
                             cannot buy the position straight back
    daily-loss breaker       NAV ≤ day-open NAV × (1 − 3%) → no new entries today and halve the MES overlay
    CVaR check               book CVaR95 (normal: 2.063σ) ≤ 1.5 × regime VaR ceiling, else scale beta down
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TRIM_DD = 0.06
CLOSE_DD = 0.10
OPTION_STOP = 0.50
LEAPS_MIN_DTE_ENTRY = 270
LEAPS_ROLL_DTE = 90
EXPIRY_CLOSE_DTE = 2
HEDGE_MIN_WEIGHT = 0.03
HEDGE_EVENT_DAYS = 5
HEDGE_OTM = 0.05
HEDGE_PREMIUM = 0.003
HEDGE_DTE = (7, 45)
PROTECT_COOLDOWN_DAYS = 14      # calendar days a trimmed / stopped name may not be bought back (≈ 10 sessions)
DAILY_LOSS_BREAKER = 0.03
CVAR_Z = 2.063                  # CVaR95 / σ for a normal distribution
Z95 = 1.645


@dataclass
class PositionState:
    entry: float
    peak: float
    trimmed_at_peak: Optional[float] = None
    short: bool = False          # short position: `peak` holds the LOW since entry, drawdown = rise from it


@dataclass
class Action:
    kind: str                   # TRIM | CLOSE | HEDGE | UNHEDGE
    ticker: str
    reason: str
    fraction: float = 1.0       # TRIM: share of the position to sell


@dataclass
class Protector:
    states: Dict[str, PositionState] = field(default_factory=dict)

    def update(self, ticker: str, price: float, qty: float) -> Optional[Action]:
        if qty == 0 or price <= 0:
            self.states.pop(ticker, None)
            return None
        short = qty < 0
        st = self.states.get(ticker)
        if st is None or st.short != short:
            self.states[ticker] = PositionState(price, price, short=short)
            return None
        if short:
            st.peak = min(st.peak, price)                  # the low since entry
            dd = price / st.peak - 1                       # adverse move = rise from the low
        else:
            st.peak = max(st.peak, price)
            dd = 1 - price / st.peak
        if dd >= CLOSE_DD:
            self.states.pop(ticker, None)
            return Action("CLOSE", ticker, f"position drawdown {dd:.1%} ≥ {CLOSE_DD:.0%}")
        if dd >= TRIM_DD and st.trimmed_at_peak != st.peak:
            st.trimmed_at_peak = st.peak
            return Action("TRIM", ticker, f"position drawdown {dd:.1%} ≥ {TRIM_DD:.0%}", 0.5)
        return None


def option_exit(dte: int, entry_dte: int, value: float, cost: float, kind: str = "",
                days_to_event: Optional[int] = None) -> Optional[str]:
    """Reason to close a long option position, or None."""
    if dte <= EXPIRY_CLOSE_DTE:
        return f"expiry in {dte}d"
    leaps = entry_dte >= LEAPS_MIN_DTE_ENTRY
    if leaps and dte < LEAPS_ROLL_DTE:
        return f"LEAPS roll window ({dte}d left)"
    if not leaps and cost > 0 and value <= (1 - OPTION_STOP) * cost:
        return f"option stop {value / cost - 1:.0%}"
    if kind in ("STRADDLE", "HEDGE") and days_to_event is not None and days_to_event < 0:
        return "event passed"
    return None


def needs_hedge(weight: float, days_to_event: Optional[int], hedged: bool) -> Optional[str]:
    if hedged:
        return "UNHEDGE" if days_to_event is None or days_to_event < 0 else None
    if weight >= HEDGE_MIN_WEIGHT and days_to_event is not None and 0 <= days_to_event <= HEDGE_EVENT_DAYS:
        return "HEDGE"
    return None


def breaker(nav: float, day_open_nav: float) -> bool:
    return day_open_nav > 0 and nav <= day_open_nav * (1 - DAILY_LOSS_BREAKER)


def book_cvar(beta_leverage: float, sigma_m: float, alpha_var: float) -> float:
    """1-day CVaR95 of the book: market part + stock-specific part (independent), normal approximation."""
    s_beta = abs(beta_leverage) * sigma_m / math.sqrt(252)
    s_alpha = alpha_var / Z95
    return CVAR_Z * math.sqrt(s_beta ** 2 + s_alpha ** 2)


def cvar_scale(beta_leverage: float, sigma_m: float, alpha_var: float, ceiling_var: float, mult: float = 1.5) -> float:
    """Factor (≤ 1) to scale the beta overlay so CVaR95 ≤ mult × regime VaR ceiling."""
    limit = mult * ceiling_var
    if book_cvar(beta_leverage, sigma_m, alpha_var) <= limit or beta_leverage == 0:
        return 1.0
    s_alpha = alpha_var / Z95
    room = (limit / CVAR_Z) ** 2 - s_alpha ** 2
    if room <= 0:
        return 0.0
    allowed = math.sqrt(room) * math.sqrt(252) / sigma_m
    return max(0.0, min(1.0, allowed / abs(beta_leverage)))
