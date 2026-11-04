"""Feed guard — make a delayed, cut or tampered data feed expensive to exploit and quick to stop.

Threat model (owner): someone who can delay or cut the live feed (or alter it at the source) to make the
algorithm act on stale prices. Transport encryption already prevents silent in-transit edits; this module
handles what encryption cannot:

    1. STALENESS     every key price must be fresh during the session: SPY and MES ≤ 5 minutes, held single
                     names ≤ 15 minutes (daily-resolution backtests: multi-day gaps only)
    2. CROSS-FEED    SPY (ETF) vs an independent reference: phase 1 the S&P 500 index (SPX, calculated by S&P
                     from all 500 constituents), phase 2 the MES future (CME, IBKR feed) — different sources. Their ratio is stable intraday; a deviation > BASIS_TOL from its
                     rolling median means one feed is delayed or wrong. Faking both consistently is hard.
    3. JUMPS         a key price moving > JUMP_TOL between checks with no matching move in the other feed
    → any failure  = HALT: no new entries, no leverage increases, no market orders; existing protection
                     stays (resting broker-side stops). Clears only after CLEAR_CHECKS consecutive clean checks.
    4. COLLARS       every order is a limit order within COLLAR_BPS of the last trusted price (equities, ETFs,
                     futures); market-on-close is skipped while halted
    5. BROKER STOPS  long stock positions carry a resting stop at the position's −10% level (peak-based) and the
                     MES overlay a resting stop at the adverse move that costs ≈ 8% NAV — they live on the
                     broker's servers and execute even if the feed or the algorithm is cut entirely
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_LAG_MIN = 5.0             # minutes, SPY / MES (minute data, deep liquidity)
NAME_LAG_MIN = 15.0           # minutes, held single names (thin names can skip minutes)
DAILY_LAG_MIN = 4 * 24 * 60.0  # daily-resolution backtests: only multi-day gaps count
BASIS_TOL = 0.005             # 0.5% SPY/MES ratio deviation from its rolling median
BASIS_WINDOW = 30
JUMP_TOL = 0.03               # 3% move between 30-minute checks in one feed only
CLEAR_CHECKS = 2
COLLAR_BPS = {"EQUITY": 30.0, "FUTURE": 20.0}
POSITION_STOP = 0.10          # = protection.CLOSE_DD
FUTURES_NAV_LOSS = 0.08       # = risk_budget.DD_CLOSE: resting MES stop where the overlay alone loses ~8% NAV
FUTURES_STOP_MIN, FUTURES_STOP_MAX = 0.02, 0.10
STOP_REPLACE = 0.02           # re-place a resting stop only when its level moves > 2% (no order churn)


@dataclass
class GuardState:
    ok: bool = True
    reasons: List[str] = field(default_factory=list)
    clean_streak: int = 0


class FeedGuard:
    def __init__(self, daily: bool = False):
        self.daily = daily
        self.ratios: deque = deque(maxlen=BASIS_WINDOW)
        self.last: Dict[str, float] = {}
        self.state = GuardState()

    def check(self, now_minutes: float, last_bar_minutes: Dict[str, Optional[float]],
              spy: Optional[float], mes: Optional[float], ref_name: str = "MES") -> GuardState:
        """now_minutes / last_bar_minutes: minutes since any fixed epoch (e.g. timestamp() / 60).
        mes = the independent reference price (MES future in phase 2; the S&P 500 index SPX in phase 1)."""
        reasons = []
        for name, t in last_bar_minutes.items():
            lag = DAILY_LAG_MIN if self.daily else (MAX_LAG_MIN if name in ("SPY", ref_name) else NAME_LAG_MIN)
            if t is None:
                reasons.append(f"{name}: no data")
            elif now_minutes - t > lag:
                reasons.append(f"{name}: stale {now_minutes - t:.0f} min")
        if spy and mes and spy > 0 and mes > 0:
            r = mes / spy
            if len(self.ratios) >= 5:
                med = statistics.median(self.ratios)
                if abs(r / med - 1) > BASIS_TOL:
                    reasons.append(f"SPY/{ref_name} basis off {r / med - 1:+.2%} (feeds disagree)")
            if not reasons:
                self.ratios.append(r)
            for name, px, other, opx in (("SPY", spy, ref_name, mes), (ref_name, mes, "SPY", spy)):
                p0, o0 = self.last.get(name), self.last.get(other)
                if p0 and o0 and abs(px / p0 - 1) > JUMP_TOL and abs(opx / o0 - 1) < JUMP_TOL / 3:
                    reasons.append(f"{name} jumped {px / p0 - 1:+.1%} without {other}")
            self.last.update({"SPY": spy, ref_name: mes})
        st = self.state
        if reasons:
            st.ok, st.reasons, st.clean_streak = False, reasons, 0
        else:
            st.clean_streak += 1
            if not st.ok and st.clean_streak >= CLEAR_CHECKS:
                st.ok, st.reasons = True, []
        return st


def collar(price: float, buy: bool, kind: str = "EQUITY") -> float:
    """Limit price for a marketable-but-bounded order around the last trusted price."""
    bps = COLLAR_BPS.get(kind, COLLAR_BPS["EQUITY"]) / 1e4
    return round(price * (1 + bps) if buy else price * (1 - bps), 2)


def position_stop(peak: float) -> float:
    return round(peak * (1 - POSITION_STOP), 2)


def futures_stop(price: float, overlay: float) -> Optional[float]:
    """Resting stop for the MES overlay: the adverse move that costs ≈ 8% NAV (2%–10% of price)."""
    if overlay == 0 or price <= 0:
        return None
    move = min(max(FUTURES_NAV_LOSS / abs(overlay), FUTURES_STOP_MIN), FUTURES_STOP_MAX)
    return round(price * (1 - move) if overlay > 0 else price * (1 + move), 2)
