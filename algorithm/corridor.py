"""Smart-beta corridor — canonical implementation (v2, owner-approved).

Leverage on market beta (see CLAUDE.md "Smart Beta"):

    L(Rm, σm) = k · s(Rm) · (Rm − rf) / σm²          capped at MAX_GROSS_LEVERAGE (4.5)

    Rm    expected annualised S&P 500 return
    rf    risk-free rate (4%)
    σm    current annualised market volatility
    k     share of full Kelly (1.0 = full Kelly)
    s(Rm) corridor participation weight, a smoothstep:
              t = clip((Rm − 7%) / 5%, 0, 1),   s = 3t² − 2t³
    (Rm − rf)/σm²   the "gamma market": volatility-adjusted full-Kelly leverage

Zones:
    Rm < 7%      DEFENSIVE  s = 0  → no long market beta (shorts / hedges only)
    7% ≤ Rm ≤ 12% DYNAMIC   s rises 0 → 1 smoothly → leverage ramps in
    Rm > 12%     ROTATION   s = 1  → full vol-adjusted Kelly leverage + active rotation,
                            up to the 4.5x gross cap

Corridor gamma (the flattened bell) = ds/dRm = 6t(1 − t)/5%: zero outside the corridor,
peak 30 per unit Rm (0.30 per 1%) at 9.5%, area 1.

Legacy (v1) identifiers β = −0.029 … 0.425 (Merton with λσ² = 0.1101, h = 7.32%) are kept
below for reference only — they no longer drive leverage.

Pure Python + math only: runs unchanged on QuantConnect, locally and in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- corridor constants (single source of truth) -----------------------------
R_LOW = 0.07                     # S&P 500 historical earnings-return floor (annualised)
R_HIGH = 0.12                    # ... ceiling
RISK_FREE = 0.04                 # hurdle for the gamma market
KELLY_FRACTION = 1.0             # k — share of full Kelly
MAX_GROSS_LEVERAGE = 4.5         # owner-set hard cap
SIGMA_STD = 0.15                 # thesis standard volatility (reference / charts)
THROTTLE = 0.05                  # rebalance only if |Δleverage| exceeds this
MIN_SIGMA = 0.05                 # floor on σm to avoid division blow-ups

ZONE_DEFENSIVE = "DEFENSIVE"     # Rm < 7%
ZONE_DYNAMIC = "DYNAMIC"         # 7% ≤ Rm ≤ 12%
ZONE_ROTATION = "ROTATION"       # Rm > 12%

# --- legacy v1 identifiers (reference only) ----------------------------------
LEGACY_BETA_LOW = -0.029
LEGACY_BETA_HIGH = 0.425
LEGACY_LAMBDA_SIGMA2 = (R_HIGH - R_LOW) / (LEGACY_BETA_HIGH - LEGACY_BETA_LOW)   # 0.1101
LEGACY_HURDLE = R_LOW - LEGACY_BETA_LOW * LEGACY_LAMBDA_SIGMA2                    # 0.0732


def _t(rm: float) -> float:
    return min(max((rm - R_LOW) / (R_HIGH - R_LOW), 0.0), 1.0)


def participation(rm: float) -> float:
    """Corridor participation weight s(Rm): 0 below 7%, smooth S-curve, 1 above 12%."""
    t = _t(rm)
    return 3 * t * t - 2 * t * t * t


def corridor_gamma(rm: float) -> float:
    """Corridor gamma ds/dRm per unit of Rm — the flattened bell (peak 30 at 9.5%, area 1)."""
    if rm <= R_LOW or rm >= R_HIGH:
        return 0.0
    t = _t(rm)
    return 6 * t * (1 - t) / (R_HIGH - R_LOW)


def gamma_market(rm: float, sigma_market: float, rf: float = RISK_FREE) -> float:
    """Volatility-adjusted full-Kelly leverage (Rm − rf)/σm² (can be negative)."""
    s = max(sigma_market, MIN_SIGMA)
    return (rm - rf) / (s * s)


def zone(rm: float) -> str:
    if rm < R_LOW:
        return ZONE_DEFENSIVE
    if rm > R_HIGH:
        return ZONE_ROTATION
    return ZONE_DYNAMIC


@dataclass
class CorridorState:
    rm: float
    sigma: float
    zone: str
    participation: float
    gamma: float               # ds/dRm (per unit Rm)
    kelly: float               # (Rm − rf)/σ²
    leverage_raw: float        # k · s · kelly
    cap: float
    leverage: float            # final long-beta leverage after caps (≥ 0)
    leverage_slope: float      # dL/dRm at the current point (per unit Rm)

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def target_leverage(rm: float, sigma_market: float, k: float = KELLY_FRACTION,
                    cap: float = MAX_GROSS_LEVERAGE, rf: float = RISK_FREE) -> CorridorState:
    """Long market-beta leverage for the current Rm and volatility.

    cap — the binding ceiling chosen by the caller: min(MAX_GROSS_LEVERAGE, regime cap,
    regime-VaR-implied leverage). Leverage is never negative here: below 7% the corridor
    holds no long beta; short exposure is a separate decision (hedges / alpha shorts).
    """
    cap = min(cap, MAX_GROSS_LEVERAGE)
    s = participation(rm)
    g = corridor_gamma(rm)
    kel = gamma_market(rm, sigma_market, rf)
    raw = k * s * kel
    lev = min(max(raw, 0.0), cap)
    sig = max(sigma_market, MIN_SIGMA)
    slope = 0.0 if (raw <= 0 or raw >= cap) else k * (g * kel + s / (sig * sig))
    return CorridorState(rm, sigma_market, zone(rm), s, g, kel, raw, cap, lev, slope)


def should_rebalance(current_leverage: float, target: float, throttle: float = THROTTLE) -> bool:
    return abs(target - current_leverage) > throttle


def annualised_drift(closes: list[float], periods_per_year: int = 252) -> float:
    """Geometric annualised return of a price series (realised Rm estimate)."""
    if len(closes) < 2 or closes[0] <= 0:
        return 0.0
    n = len(closes) - 1
    return (closes[-1] / closes[0]) ** (periods_per_year / n) - 1.0


def realised_vol(closes: list[float], periods_per_year: int = 252) -> float:
    """Annualised realised volatility from close prices."""
    rets = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return SIGMA_STD
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * periods_per_year)


def legacy_beta(rm: float) -> float:
    """v1 identifier curve −0.029 … 0.425 (reference only; not used for sizing)."""
    return LEGACY_BETA_LOW + (LEGACY_BETA_HIGH - LEGACY_BETA_LOW) * participation(rm)
