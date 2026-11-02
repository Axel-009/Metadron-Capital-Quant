"""Forward Rm — the expected annualised S&P 500 return that drives the corridor.

The raw trailing drift is far too noisy for the 7–12% corridor (it swings −30% … +60%), so Rm is
built from slower, forward-looking components:

    anchor     valuation anchor: earnings yield + inflation (nominal Gordon-style expected return)
               when the macro agent supplies them, else the long-run 10%
    trend      EWMA trailing return SHRUNK halfway toward the anchor (momentum, not noise)
    regime     Σ P(regime) × regime drift  (HMM probabilities: TRENDING +3%, RANGE 0, STRESS −4%, CRASH −10%)
    tension    −8% × (GMTF tension percentile − 0.5)  (monetary tension above median lowers Rm)
    kronos     optional Kronos index forecast (annualised), blended at KRONOS_WEIGHT

    Rm = (1 − wK) · [0.5·anchor + 0.5·trend + regime + tension] + wK · kronos,  clipped to [−30%, +40%]

Every component is returned so the dashboard and agents can see why Rm moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

LONG_RUN = 0.10
TREND_SHRINK = 0.5
ANCHOR_WEIGHT = 0.5
REGIME_DRIFT = {"TRENDING": 0.03, "RANGE": 0.0, "STRESS": -0.04, "CRASH": -0.10}
TENSION_SLOPE = 0.08
KRONOS_WEIGHT = 0.25
RM_MIN, RM_MAX = -0.30, 0.40


@dataclass
class ForwardRm:
    rm: float
    components: Dict[str, float] = field(default_factory=dict)


def anchor(earnings_yield: Optional[float] = None, inflation: Optional[float] = None) -> float:
    if earnings_yield is not None and inflation is not None:
        return earnings_yield + inflation
    if earnings_yield is not None:
        return earnings_yield + 0.025
    return LONG_RUN


def estimate(trailing_ewma: float,
             regime_probs: Optional[Dict[str, float]] = None,
             gmtf_score: Optional[float] = None,
             earnings_yield: Optional[float] = None,
             inflation: Optional[float] = None,
             kronos: Optional[float] = None) -> ForwardRm:
    a = anchor(earnings_yield, inflation)
    trend = a + TREND_SHRINK * (trailing_ewma - a)
    reg = sum(p * REGIME_DRIFT.get(r, 0.0) for r, p in (regime_probs or {}).items())
    ten = -TENSION_SLOPE * (gmtf_score - 0.5) if gmtf_score is not None else 0.0
    core = ANCHOR_WEIGHT * a + (1 - ANCHOR_WEIGHT) * trend + reg + ten
    wk = KRONOS_WEIGHT if kronos is not None else 0.0
    rm = (1 - wk) * core + wk * (kronos if kronos is not None else 0.0)
    rm = max(RM_MIN, min(RM_MAX, rm))
    return ForwardRm(rm, {"anchor": a, "trend": trend, "regime_adj": reg, "tension_adj": ten,
                          "kronos": kronos if kronos is not None else float("nan"), "core": core})
