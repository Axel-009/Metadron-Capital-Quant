"""Macro regime — the trader's top-down question tree, as code.

Turns macro readings (incoming data vs expectation) and market data into
    1. factor states in [-1, +1] (see market_map.FACTORS),
    2. a diagnosis: the ordered questions a trader asks, the answers, and the implications,
    3. the stock–bond correlation regime (which hedge works),
    4. a market bias: BULLISH / NEUTRAL / BEARISH.

The question tree (owner's framework):
    Is inflation within mandate? → Is the economy hot / resilient? → Is labor tight?
    → Is labor gapping (deteriorating fast)? → Is there room for stimulus?
    → Is productivity rising? → Are real wages rising or stagnant?
      (productivity ↑ + wages flat + employment retained ⇒ skills being substituted ⇒ AI substitution:
       pressure on low-end SaaS / labor-intensive services, support for the AI chain)
    → How is system liquidity? → How are corporate bonds (credit spreads)?
      (spreads widening ⇒ pressure on debt / private-credit funded names)
    → Geopolitics escalating? (defense, energy, refiners) → Where is sentiment? (extremes = contrarian)

Inputs are optional; missing inputs leave a factor at 0 (neutral) and the question "UNKNOWN".
Macro readings come from FRED (via the macro agent / ObjectStore); market factors from prices.
Pure Python + math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INFLATION_TARGET = 0.02
INFLATION_TOLERANCE = 0.01        # "within mandate" = target ± 1 pt (core PCE / CPI YoY)
NAIRU = 0.042                     # natural unemployment rate (estimate)
GAP_TRIGGER = 0.005               # 3m-avg unemployment ≥ 0.5 pt above its 12m low (Sahm-style)


def _squash(x: float, scale: float) -> float:
    """Map x to [-1, 1] smoothly: tanh(x / scale)."""
    return math.tanh(x / scale) if scale else 0.0


# ------------------------------------------------------------------ factors
def factor_states(m: Dict[str, float]) -> Dict[str, float]:
    """Factor states from readings. Keys (all optional, decimals unless noted):

    growth_surprise        activity data vs consensus, in std-devs (e.g. a surprise index / 30)
    inflation_yoy          core PCE or CPI YoY
    unemployment           unemployment rate
    productivity_yoy       nonfarm productivity YoY
    real_wage_yoy          average hourly earnings YoY minus CPI YoY
    rates_change_3m        10y yield change over 3 months (decimal, e.g. 0.004 = +40 bp)
    liquidity_change_3m    Fed balance sheet minus TGA minus RRP, % change over 3 months
    hy_oas_change_1m       HY option-adjusted spread change over 1 month (decimal, + = widening)
    escalation             0..1 geopolitical escalation score
    vix                    VIX level
    ai_capex_growth        hyperscaler capex YoY growth
    commodities_21d        broad commodity return over 21 days
    """
    f = {}
    if "growth_surprise" in m:
        f["GROWTH"] = _squash(m["growth_surprise"], 1.0)
    if "inflation_yoy" in m:
        f["INFLATION"] = _squash(m["inflation_yoy"] - INFLATION_TARGET, 0.015)
    if "unemployment" in m:
        f["LABOR"] = _squash(NAIRU - m["unemployment"], 0.008)
    if "real_wage_yoy" in m:
        f["WAGES"] = _squash(m["real_wage_yoy"], 0.015)
    if "productivity_yoy" in m:
        f["PRODUCTIVITY"] = _squash(m["productivity_yoy"] - 0.015, 0.015)
    if "rates_change_3m" in m:
        f["RATES"] = _squash(m["rates_change_3m"], 0.004)
    if "liquidity_change_3m" in m:
        f["LIQUIDITY"] = _squash(m["liquidity_change_3m"], 0.03)
    if "hy_oas_change_1m" in m:
        f["CREDIT"] = _squash(-m["hy_oas_change_1m"], 0.004)      # widening → negative
    if "escalation" in m:
        # 0.25 = neutral background risk; calm is only mildly negative (no fake "peace dividend")
        f["GEOPOLITICS"] = max(-0.5, min(1.0, (m["escalation"] - 0.25) / 0.5))
    if "vix" in m:
        f["SENTIMENT"] = _squash(20.0 - m["vix"], 6.0)
    if "ai_capex_growth" in m:
        f["AI_CAPEX"] = _squash(m["ai_capex_growth"] - 0.15, 0.20)
    if "commodities_21d" in m:
        f["COMMODITIES"] = _squash(m["commodities_21d"], 0.06)
    return f


# ------------------------------------------------------------------ diagnosis
@dataclass
class Finding:
    question: str
    answer: str                     # YES / NO / UNKNOWN / short text
    implication: str
    tags: List[str] = field(default_factory=list)


# implication tag → sub-sector score adjustments (added to factor-model scores)
IMPLICATION_EFFECTS: Dict[str, Dict[str, float]] = {
    "AI_SUBSTITUTION": {"LOW_END_SAAS": -0.6, "LABOR_INTENSIVE_SERVICES": -0.4,
                        "AI_PLATFORMS": 0.3, "AI_CHIP_DESIGN": 0.3, "POWER_GRID": 0.2, "NETWORKING": 0.2},
    "CREDIT_STRESS": {"AI_NEOCLOUD_CREDIT": -0.7, "AI_HOSTING_PIVOT": -0.6, "REAL_ESTATE": -0.4,
                      "DATA_CENTERS": -0.3, "FINANCIALS": -0.3, "STAPLES": 0.2, "HEALTHCARE_PHARMA": 0.2},
    "ESCALATION": {"DEFENSE": 0.6, "ENERGY_INTEGRATED": 0.4, "REFINERS": 0.4,
                   "FOUNDRY_REGIONAL": -0.4, "SEMI_EQUIPMENT": -0.2},
    "INFLATION_HOT": {"COMMODITY_PRODUCERS": 0.3, "ENERGY_INTEGRATED": 0.2, "REAL_ESTATE": -0.3,
                      "CONSUMER_DISCRETIONARY": -0.3, "UTILITIES": -0.2},
    "LABOR_GAPPING": {"CONSUMER_DISCRETIONARY": -0.5, "FINANCIALS": -0.3, "INDUSTRIALS": -0.3,
                      "STAPLES": 0.3, "UTILITIES": 0.3, "HEALTHCARE_PHARMA": 0.3},
    "STIMULUS_ROOM": {"REAL_ESTATE": 0.3, "CONSUMER_DISCRETIONARY": 0.2, "DATA_CENTERS": 0.2},
    "LIQUIDITY_TIGHT": {"AI_NEOCLOUD_CREDIT": -0.4, "AI_HOSTING_PIVOT": -0.4, "AI_CHIP_DESIGN": -0.2},
    "EUPHORIA": {"AI_NEOCLOUD_CREDIT": -0.3, "AI_HOSTING_PIVOT": -0.3, "AI_CHIP_DESIGN": -0.1},
    "CAPITULATION": {"AI_CHIP_DESIGN": 0.2, "SEMI_EQUIPMENT": 0.2, "CONSUMER_DISCRETIONARY": 0.2},
}


def diagnose(m: Dict[str, float]) -> List[Finding]:
    """Walk the trader's question tree on the readings in m (see factor_states)."""
    out: List[Finding] = []
    infl = m.get("inflation_yoy")
    if infl is None:
        out.append(Finding("Is inflation within mandate?", "UNKNOWN", "no inflation reading"))
        within = None
    else:
        within = abs(infl - INFLATION_TARGET) <= INFLATION_TOLERANCE
        out.append(Finding("Is inflation within mandate?", "YES" if within else "NO",
                           f"inflation {infl:.1%} vs {INFLATION_TARGET:.0%} target",
                           [] if within or infl < INFLATION_TARGET else ["INFLATION_HOT"]))

    g = m.get("growth_surprise")
    hot = g is not None and g > 0.3
    out.append(Finding("Is the economy hot / resilient?", "UNKNOWN" if g is None else ("YES" if hot else "NO"),
                       "incoming data beating expectations" if hot else "data in line or missing"))

    u = m.get("unemployment")
    tight = u is not None and u < NAIRU
    out.append(Finding("Is the labor market tight?", "UNKNOWN" if u is None else ("YES" if tight else "NO"),
                       f"unemployment {u:.1%} vs NAIRU {NAIRU:.1%}" if u is not None else "no reading"))

    u_low = m.get("unemployment_12m_low")
    gapping = u is not None and u_low is not None and u - u_low >= GAP_TRIGGER
    out.append(Finding("Is labor gapping (deteriorating fast)?",
                       "UNKNOWN" if u_low is None else ("YES" if gapping else "NO"),
                       "unemployment ≥ 0.5 pt above its 12-month low" if gapping else "no gap",
                       ["LABOR_GAPPING"] if gapping else []))

    room = within is True and (gapping or not tight)
    out.append(Finding("Is there room for stimulus?", "YES" if room else ("NO" if within is not None else "UNKNOWN"),
                       "inflation in mandate and labor not tight" if room else "inflation or tight labor limits easing",
                       ["STIMULUS_ROOM"] if room else []))

    p, w = m.get("productivity_yoy"), m.get("real_wage_yoy")
    prod_up = p is not None and p > 0.015
    out.append(Finding("Is productivity rising?", "UNKNOWN" if p is None else ("YES" if prod_up else "NO"),
                       f"productivity {p:.1%} YoY" if p is not None else "no reading"))
    stagnant = w is not None and w <= 0.005
    substitution = prod_up and stagnant and (tight or u is None or not gapping)
    out.append(Finding("Are real wages rising?", "UNKNOWN" if w is None else ("NO — stagnant" if stagnant else "YES"),
                       "productivity up, wages flat, employment retained ⇒ skills being substituted (AI)"
                       if substitution else "no substitution signal",
                       ["AI_SUBSTITUTION"] if substitution else []))

    liq = m.get("liquidity_change_3m")
    tight_liq = liq is not None and liq < -0.02
    out.append(Finding("How is system liquidity?", "UNKNOWN" if liq is None else ("TIGHTENING" if tight_liq else "OK"),
                       f"net liquidity {liq:+.1%} over 3m" if liq is not None else "no reading",
                       ["LIQUIDITY_TIGHT"] if tight_liq else []))

    oas = m.get("hy_oas_change_1m")
    stress = oas is not None and oas >= 0.0035            # ≥ +35 bp in a month
    out.append(Finding("How are corporate bonds doing?", "UNKNOWN" if oas is None else ("SPREADS WIDENING" if stress else "STABLE"),
                       f"HY OAS {oas * 1e4:+.0f} bp over 1m" if oas is not None else "no reading",
                       ["CREDIT_STRESS"] if stress else []))

    esc = m.get("escalation")
    escal = esc is not None and esc >= 0.5
    out.append(Finding("Is geopolitics escalating?", "UNKNOWN" if esc is None else ("YES" if escal else "NO"),
                       f"escalation score {esc:.2f}" if esc is not None else "no reading",
                       ["ESCALATION"] if escal else []))

    vix = m.get("vix")
    tag = []
    if vix is not None and vix < 13:
        tag, ans = ["EUPHORIA"], "EUPHORIC"
    elif vix is not None and vix > 35:
        tag, ans = ["CAPITULATION"], "FEARFUL"
    else:
        ans = "UNKNOWN" if vix is None else "NORMAL"
    out.append(Finding("Where is investor sentiment?", ans,
                       f"VIX {vix:.1f}" if vix is not None else "no reading", tag))
    return out


def implication_adjustments(findings: List[Finding]) -> Dict[str, float]:
    adj: Dict[str, float] = {}
    for f in findings:
        for t in f.tags:
            for sub, v in IMPLICATION_EFFECTS.get(t, {}).items():
                adj[sub] = adj.get(sub, 0.0) + v
    return adj


# ------------------------------------------------------------------ stock–bond correlation
def correlation(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 10:
        return None
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def returns(closes: List[float]) -> List[float]:
    return [b / a - 1 for a, b in zip(closes[:-1], closes[1:]) if a > 0]


def stock_bond_regime(stock_closes: List[float], bond_closes: List[float], window: int = 63) -> dict:
    """POSITIVE corr (> +0.2): inflation-driven — bonds fall with stocks, hedge with commodities/gold/cash.
    NEGATIVE corr (< −0.2): growth-driven — Treasuries hedge equities.  Otherwise MIXED."""
    rho = correlation(returns(stock_closes)[-window:], returns(bond_closes)[-window:])
    if rho is None:
        return {"corr": None, "regime": "UNKNOWN", "hedge": "CASH"}
    if rho > 0.2:
        return {"corr": rho, "regime": "POSITIVE (inflation-driven)", "hedge": "COMMODITIES/GOLD"}
    if rho < -0.2:
        return {"corr": rho, "regime": "NEGATIVE (growth-driven)", "hedge": "TREASURIES"}
    return {"corr": rho, "regime": "MIXED", "hedge": "GOLD/CASH"}


# ------------------------------------------------------------------ market bias
def market_bias(factors: Dict[str, float], corridor_zone: Optional[str] = None) -> dict:
    """BULLISH / NEUTRAL / BEARISH from growth, liquidity, credit, sentiment and the corridor zone."""
    score = (0.30 * factors.get("GROWTH", 0) + 0.20 * factors.get("LIQUIDITY", 0)
             + 0.25 * factors.get("CREDIT", 0) + 0.15 * factors.get("SENTIMENT", 0)
             - 0.10 * factors.get("RATES", 0))
    if corridor_zone == "ROTATION":
        score += 0.15
    elif corridor_zone == "DEFENSIVE":
        score -= 0.25
    label = "BULLISH" if score > 0.15 else ("BEARISH" if score < -0.15 else "NEUTRAL")
    return {"score": round(score, 4), "bias": label}
