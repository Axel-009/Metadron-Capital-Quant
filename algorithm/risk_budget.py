"""Risk budget — split VaR budget + hard stops (owner-approved design).

Layers (CLAUDE.md "Risk budget"):
  1. Sizing: corridor k-Kelly leverage (corridor.py) — already scales with volatility.
  2. Regime VaR ceiling on MARKET-BETA risk (1-day 95%):
        TRENDING 7.0% · RANGE 4.5% · STRESS 2.25% · CRASH 1.0%
     (TRENDING 7.0% = VaR of the 4.5x max at 15% vol, so the owner's max is reachable)
     → converted to a leverage ceiling: L_max = VaR / (1.645 · σm / √252)
  3. Stock-specific (alpha) VaR budget, 1-day 95%, default 2.5% of NAV — applies to the
     idiosyncratic risk of the stock picks / options AFTER removing market beta.
     Per-name cap: one name's 1-day idiosyncratic VaR ≤ 0.5% NAV.
  4. Hard stops on the whole book: gross leverage ≤ 4.5x, CVaR95 ≤ 1.5 × regime ceiling,
     portfolio drawdown −4% (trim) / −8% (close all but hedges).

Long options count at max loss = premium (defined risk).
Pure Python + math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

Z95 = 1.645
SQRT_DAYS = math.sqrt(252)

REGIME_VAR_CEILING = {          # 1-day 95% VaR on market-beta risk, fraction of NAV
    "TRENDING": 0.070,
    "RANGE": 0.045,
    "STRESS": 0.0225,
    "CRASH": 0.010,
}
ALPHA_VAR_BUDGET = 0.025        # stock-specific 1-day 95% VaR, fraction of NAV
PER_NAME_VAR_CAP = 0.005        # one name's idiosyncratic 1-day VaR
MAX_GROSS_LEVERAGE = 4.5
MAX_FUTURES_LEVERAGE = 2.0      # owner: MES overlay notional ≤ 2.0x NAV, long or short (≈ 14% NAV initial margin)
CVAR_MULT = 1.5
DD_TRIM = 0.04
DD_CLOSE = 0.08


def daily_var(weight: float, annual_vol: float) -> float:
    """Parametric 1-day 95% VaR (fraction of NAV) of |weight| at annual_vol."""
    return Z95 * abs(weight) * annual_vol / SQRT_DAYS


def regime_leverage_ceiling(regime: str, sigma_market: float) -> float:
    """Max market-beta leverage allowed by the regime VaR ceiling."""
    ceiling = REGIME_VAR_CEILING.get(str(regime).upper(), REGIME_VAR_CEILING["RANGE"])
    per_unit = Z95 * max(sigma_market, 1e-4) / SQRT_DAYS
    return min(ceiling / per_unit, MAX_GROSS_LEVERAGE)


STRESS_REGIMES = ("STRESS", "CRASH")
NO_DOUBLE_SIZING_RM = 0.12      # Rm above the corridor ceiling (ROTATION zone)


# Beta burst: when momentum confirms the trend, the regime VaR ceiling is lifted by the legacy
# burst/cap ratios (TRENDING 0.70/0.65, RANGE 0.55/0.45, STRESS 0.20/0.15); never in CRASH and
# never above the 4.5x hard cap. Burst raises the CEILING only — Kelly still sets the target.
BURST_MULT = {"TRENDING": 0.70 / 0.65, "RANGE": 0.55 / 0.45, "STRESS": 0.20 / 0.15, "CRASH": 1.0}


def momentum_confirms(closes: list) -> bool:
    """Trend confirmation: price > 50-day MA > 200-day MA and the 63-day return is positive."""
    if len(closes) < 200:
        return False
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200
    return closes[-1] > ma50 > ma200 and closes[-1] > closes[-64]


def beta_cap(regime: str, sigma_market: float, regime_beta_cap: Optional[float] = None,
             rm: Optional[float] = None, burst: bool = False) -> float:
    """Binding cap for corridor leverage: min(4.5x, regime VaR ceiling, optional regime cap).

    No double sizing (owner rule): when the market itself is strong (Rm > 12%, ROTATION zone)
    but the regime reads STRESS/CRASH — typically a sector/theme crash or a post-crash rebound —
    the regime VaR ceiling is NOT applied on top of Kelly. Kelly's 1/σ² already sizes for the
    volatility; only the 4.5x hard cap (and an explicit regime_beta_cap) remain. The crash is
    handled by rotation (algorithm/rotation.py), not by cutting market beta twice.
    """
    if rm is not None and rm > NO_DOUBLE_SIZING_RM and str(regime).upper() in STRESS_REGIMES:
        cap = MAX_GROSS_LEVERAGE
    else:
        cap = regime_leverage_ceiling(regime, sigma_market)
        if burst:
            cap *= BURST_MULT.get(str(regime).upper(), 1.0)
    cap = min(cap, MAX_GROSS_LEVERAGE)
    if regime_beta_cap is not None:
        cap = min(cap, regime_beta_cap)
    return max(cap, 0.0)


@dataclass
class AlphaPosition:
    ticker: str
    weight: float                 # signed fraction of NAV (exposure for stock; premium for long options)
    idio_vol: float               # annualised stock-specific volatility
    sector: str = ""
    defined_risk: bool = False    # long option / spread: VaR capped at premium


def name_var(p: AlphaPosition) -> float:
    v = daily_var(p.weight, p.idio_vol)
    return min(v, abs(p.weight)) if p.defined_risk else v


def alpha_var(positions: Iterable[AlphaPosition], intra_sector_corr: float = 0.3) -> float:
    """Stock-specific 1-day VaR of the alpha book (names in the same sector correlated)."""
    pos = list(positions)
    var_sq = 0.0
    for i, a in enumerate(pos):
        va = name_var(a) * (1 if a.weight >= 0 else -1)
        for j, b in enumerate(pos):
            vb = name_var(b) * (1 if b.weight >= 0 else -1)
            if i == j:
                rho = 1.0
            elif a.sector and a.sector == b.sector:
                rho = intra_sector_corr
            else:
                rho = 0.0
            var_sq += rho * va * vb
    return math.sqrt(max(var_sq, 0.0))


@dataclass
class AlphaCheck:
    ok: bool
    book_var_before: float
    book_var_after: float
    name_var: float
    max_weight_allowed: float
    reasons: List[str] = field(default_factory=list)


def check_alpha_trade(book: List[AlphaPosition], trade: AlphaPosition,
                      budget: float = ALPHA_VAR_BUDGET,
                      per_name_cap: float = PER_NAME_VAR_CAP) -> AlphaCheck:
    """Marginal check of one stock-specific trade against the alpha budget and per-name cap.

    Returns the largest weight that fits (same sign as the trade), so callers can scale down.
    """
    before = alpha_var(book)
    merged = [p for p in book if p.ticker != trade.ticker]
    existing = next((p for p in book if p.ticker == trade.ticker), None)
    new_w = trade.weight + (existing.weight if existing else 0.0)
    combined = AlphaPosition(trade.ticker, new_w, trade.idio_vol, trade.sector, trade.defined_risk)
    after = alpha_var(merged + [combined])
    nv = name_var(combined)

    # largest weight (bisection) that satisfies both limits
    lo, hi = 0.0, abs(trade.weight)
    sign = 1.0 if trade.weight >= 0 else -1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        w = sign * mid + (existing.weight if existing else 0.0)
        cand = AlphaPosition(trade.ticker, w, trade.idio_vol, trade.sector, trade.defined_risk)
        if alpha_var(merged + [cand]) <= budget + 1e-12 and name_var(cand) <= per_name_cap + 1e-12:
            lo = mid
        else:
            hi = mid
    reasons = []
    if after > budget + 1e-12 and after > before:
        reasons.append(f"alpha VaR {after:.2%} > budget {budget:.2%}")
    if nv > per_name_cap + 1e-12:
        reasons.append(f"{trade.ticker} name VaR {nv:.2%} > cap {per_name_cap:.2%}")
    return AlphaCheck(not reasons, before, after, nv, sign * lo, reasons)


@dataclass
class HardStopState:
    action: str            # OK | TRIM | CLOSE
    drawdown: float
    gross_ok: bool
    reasons: List[str] = field(default_factory=list)


def hard_stops(nav: float, peak_nav: float, gross_leverage: float,
               book_cvar: Optional[float] = None, regime: str = "RANGE") -> HardStopState:
    dd = 1.0 - nav / peak_nav if peak_nav > 0 else 0.0
    reasons = []
    action = "OK"
    if dd >= DD_CLOSE:
        action = "CLOSE"
        reasons.append(f"drawdown {dd:.1%} ≥ {DD_CLOSE:.0%}: close all but hedges")
    elif dd >= DD_TRIM:
        action = "TRIM"
        reasons.append(f"drawdown {dd:.1%} ≥ {DD_TRIM:.0%}: trim losers")
    gross_ok = gross_leverage <= MAX_GROSS_LEVERAGE + 1e-9
    if not gross_ok:
        reasons.append(f"gross {gross_leverage:.2f}x > {MAX_GROSS_LEVERAGE}x")
        if action == "OK":
            action = "TRIM"
    if book_cvar is not None:
        lim = CVAR_MULT * REGIME_VAR_CEILING.get(str(regime).upper(), REGIME_VAR_CEILING["RANGE"])
        if book_cvar > lim:
            reasons.append(f"CVaR95 {book_cvar:.2%} > {lim:.2%}")
            if action == "OK":
                action = "TRIM"
    return HardStopState(action, dd, gross_ok, reasons)
