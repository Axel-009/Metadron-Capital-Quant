"""Decision matrix — 4 gates, pre-trade checks and full-Kelly sizing for the equity sleeves.

    gates (each 0..1)       FUNDAMENTALS 40% (must pass — fundamentals.gate)
                            FLOW         20% (dollar-volume trend + 21d strength vs SPY)
                            MACRO        20% (sub-sector edge + regime fit of the name's nature; < 0.3 = veto)
                            MOMENTUM     20% (12-1 momentum, trend, chart patterns)
    composite               Σ weight × gate; trade only when ≥ COMPOSITE_MIN and every veto passes
    expected alpha μα       blend of the composite-implied alpha and the model forecasts (alpha.py, IC-weighted)
    pre-trade checks        spread ≤ 30 bps (equity / ETF; no quote → skip check in backtests that lack quotes),
                            P(profit over HORIZON) ≥ 52% with μα > 0 (normal, idiosyncratic vol)
    sizing                  2x Kelly on idiosyncratic risk: w = k · μα / σε² (k = 2, owner), capped at 20% NAV per name;
                            then each (sleeve × GICS sector) bucket: sleeve budget × sector budget (sectors.py),
                            unused sector room spilled to the other sectors' names still below Kelly; then
                            risk_budget.check_alpha_trade scales into the 2.5% stock-specific VaR budget and the
                            0.5% per-name VaR cap (scaled, not rejected).
    allocation              engine/allocation/allocation_engine.py (THE reference): IG 40 · HY 10 · distressed 10 ·
                            event/CVR 10 single names; TLTW/cash-flow ETFs 15 · FI/macro 5; option premium 10% of NAV
                            (IG 4 · HY 4 · distressed 2). No money market; futures margin sized to the corridor.
Market beta of the sleeves is netted by the MES overlay (main.py): futures = corridor L − Σ wᵢβᵢ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

WEIGHTS = {"FUNDAMENTALS": 0.40, "FLOW": 0.20, "MACRO": 0.20, "MOMENTUM": 0.20}
COMPOSITE_MIN = 0.55
MACRO_VETO = 0.30
ALPHA_SCALE = 0.40              # composite 0.75 → +10%/yr alpha before model blend
HORIZON_DAYS = 21
P_PROFIT_MIN = 0.52
SPREAD_MAX_BPS = 30.0
MAX_POSITION = 0.20
KELLY_FRACTION = 2.0            # owner (Nov 2026): 2x Kelly on single-name positions (was full Kelly)
MIN_IDIO_VOL = 0.10
# Full deployment (owner): single-name budget left unfilled by the entry gate is deployed into the best
# remaining gate-passing names (composite ≥ FILL_MIN, no macro veto, spread ok), ≤ FILL_MAX_NAME each
FILL_MIN = 0.50
FILL_MAX_NAME = 0.04
SHORT_MAX_NAME = 0.05

# Allocation buckets (fractions of NAV, owner-approved; mirrored in engine/allocation AllocationRules)
SLEEVES = {"IG": 0.40, "HY": 0.10, "DISTRESSED": 0.10, "EVENT": 0.10,          # single names 70%
           "CASHFLOW": 0.15, "FI_MACRO": 0.05,                                  # ETFs 20%
           "OPTIONS": 0.10}                                                     # option PREMIUM 10% of NAV
EQUITY_SLEEVES = ("IG", "HY", "DISTRESSED", "EVENT")
OPTION_PREMIUM = {"IG": 0.04, "HY": 0.04, "DISTRESSED": 0.02, "EVENT": 0.04}  # per credit sleeve (EVENT shares IG's)
OPTIONS_TOTAL = SLEEVES["OPTIONS"]
# No money-market floor and no fixed margin bucket: MES initial margin is sized to the corridor and financed by the
# account's margin (reported, not capped here — the corridor's 4.5x and the regime VaR ceilings bound it).
SECTOR_MAX = 0.30               # = sectors.MAX_SECTOR: no GICS sector above 30% of a sleeve
MES_MARGIN_RATE = 0.07           # initial margin ≈ 7% of MES notional (check with IBKR; parameter)
NATURE_REGIME_FIT = {        # how well each nature fits the regime (0..1)
    "TRENDING": {"CYCLICAL": 0.8, "SECULAR": 0.8, "DEFENSIVE": 0.4},
    "RANGE": {"CYCLICAL": 0.5, "SECULAR": 0.6, "DEFENSIVE": 0.6},
    "STRESS": {"CYCLICAL": 0.2, "SECULAR": 0.4, "DEFENSIVE": 0.8},
    "CRASH": {"CYCLICAL": 0.1, "SECULAR": 0.3, "DEFENSIVE": 0.9},
}


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def returns(px: List[float]) -> List[float]:
    return [b / a - 1 for a, b in zip(px[:-1], px[1:]) if a > 0]


def beta_and_idio(px: List[float], spy: List[float], n: int = 252):
    """(beta, annualised idiosyncratic vol) from daily returns vs SPY."""
    a, b = returns(px[-(n + 1):]), returns(spy[-(n + 1):])
    m = min(len(a), len(b))
    if m < 40:
        return 1.0, None
    a, b = a[-m:], b[-m:]
    ma, mb = sum(a) / m, sum(b) / m
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (m - 1)
    var = sum((y - mb) ** 2 for y in b) / (m - 1)
    beta = cov / var if var > 0 else 1.0
    res = [x - ma - beta * (y - mb) for x, y in zip(a, b)]
    idio = math.sqrt(sum(e * e for e in res) / (m - 2) * 252)
    return beta, idio


def flow_gate(px: List[float], vol: List[float], spy: List[float]) -> float:
    if len(px) < 64 or len(spy) < 22:
        return 0.5
    dv = [p * v for p, v in zip(px[-60:], vol[-60:])] if vol and len(vol) >= 60 else []
    dv_trend = (sum(dv[-20:]) / 20) / (sum(dv) / 60) - 1 if dv and sum(dv) > 0 else 0.0
    rel = (px[-1] / px[-22] - 1) - (spy[-1] / spy[-22] - 1)
    return _sig(2.0 * dv_trend + 10.0 * rel)


def momentum_gate(px: List[float], pattern: float = 0.0) -> float:
    if len(px) < 253:
        if len(px) < 64:
            return 0.5
        m = px[-1] / px[-64] - 1
        return _sig(4 * m + pattern)
    m12_1 = px[-22] / px[-253] - 1
    ma50, ma200 = sum(px[-50:]) / 50, sum(px[-200:]) / 200
    trend = (1 if px[-1] > ma50 else -1) + (1 if ma50 > ma200 else -1)
    return _sig(2.5 * m12_1 + 0.4 * trend + pattern)


def macro_gate(subsector_edge: Optional[float], nature: str, regime: str) -> float:
    fit = NATURE_REGIME_FIT.get(regime, NATURE_REGIME_FIT["RANGE"]).get(nature, 0.5)
    edge = 0.0 if subsector_edge is None else max(-1.0, min(1.0, subsector_edge))
    return max(0.0, min(1.0, 0.5 * fit + 0.5 * (0.5 + 0.5 * edge)))


@dataclass
class Decision:
    ticker: str
    sleeve: str
    gates: Dict[str, float]
    composite: float
    mu_alpha: float = 0.0
    idio_vol: float = 0.3
    beta: float = 1.0
    p_profit: float = 0.0
    kelly: float = 0.0
    target: float = 0.0
    approved: bool = False
    reasons: List[str] = field(default_factory=list)
    sector: str = ""
    filled: bool = False


def decide(ticker: str, sleeve: str, fund_pass: bool, fund_score: float, px: List[float], vol: List[float],
           spy: List[float], subsector_edge: Optional[float], nature: str, regime: str,
           model_alpha: Optional[float] = None, model_weight: float = 0.0, pattern: float = 0.0,
           spread_bps: Optional[float] = None) -> Decision:
    g = {"FUNDAMENTALS": fund_score if fund_pass else 0.0, "FLOW": flow_gate(px, vol, spy),
         "MACRO": macro_gate(subsector_edge, nature, regime), "MOMENTUM": momentum_gate(px, pattern)}
    comp = sum(WEIGHTS[k] * v for k, v in g.items())
    d = Decision(ticker, sleeve, g, comp)
    d.beta, idio = beta_and_idio(px, spy)
    d.idio_vol = max(idio or 0.3, MIN_IDIO_VOL)
    score_alpha = ALPHA_SCALE * (comp - 0.5)
    w = max(0.0, min(1.0, model_weight)) if model_alpha is not None else 0.0
    d.mu_alpha = (1 - w) * score_alpha + w * (model_alpha or 0.0)
    h = HORIZON_DAYS / 252
    d.p_profit = _N(d.mu_alpha * h / (d.idio_vol * math.sqrt(h)))
    if not fund_pass:
        d.reasons.append("fundamentals gate failed")
    if g["MACRO"] < MACRO_VETO:
        d.reasons.append(f"macro veto {g['MACRO']:.2f}")
    if comp < COMPOSITE_MIN:
        d.reasons.append(f"composite {comp:.2f} < {COMPOSITE_MIN}")
    if spread_bps is not None and spread_bps > SPREAD_MAX_BPS:
        d.reasons.append(f"spread {spread_bps:.0f} bps > {SPREAD_MAX_BPS:.0f}")
    if d.mu_alpha <= 0 or d.p_profit < P_PROFIT_MIN:
        d.reasons.append(f"P(profit) {d.p_profit:.0%} / alpha {d.mu_alpha:+.1%}")
    d.kelly = KELLY_FRACTION * d.mu_alpha / d.idio_vol ** 2 if d.mu_alpha > 0 else 0.0
    d.approved = not d.reasons
    d.target = min(d.kelly, MAX_POSITION) if d.approved else 0.0
    if d.approved:
        d.reasons.append(f"composite {comp:.2f}, α {d.mu_alpha:+.1%}, σε {d.idio_vol:.0%}, {KELLY_FRACTION:g}x Kelly {d.kelly:.1%}")
    return d


def decide_short(ticker: str, quality: float, px: List[float], vol: List[float], spy: List[float],
                 subsector_edge: Optional[float], nature: str, regime: str, pattern: float = 0.0,
                 spread_bps: Optional[float] = None) -> Decision:
    """Short side (shorts.py): the long gates inverted. quality = fundamentals composite 0..1 (low = weak)."""
    g = {"FUNDAMENTALS": max(0.0, min(1.0, quality)), "FLOW": flow_gate(px, vol, spy),
         "MACRO": macro_gate(subsector_edge, nature, regime), "MOMENTUM": momentum_gate(px, pattern)}
    comp = sum(WEIGHTS[k] * (1.0 - v) for k, v in g.items())
    d = Decision(ticker, "SHORT", g, comp)
    d.beta, idio = beta_and_idio(px, spy)
    d.idio_vol = max(idio or 0.3, MIN_IDIO_VOL)
    d.mu_alpha = ALPHA_SCALE * (comp - 0.5)                 # expected alpha of the SHORT position
    h = HORIZON_DAYS / 252
    d.p_profit = _N(d.mu_alpha * h / (d.idio_vol * math.sqrt(h)))
    if g["MOMENTUM"] > 0.5:
        d.reasons.append(f"no downtrend (momentum {g['MOMENTUM']:.2f})")
    if comp < COMPOSITE_MIN:
        d.reasons.append(f"short composite {comp:.2f} < {COMPOSITE_MIN}")
    if spread_bps is not None and spread_bps > SPREAD_MAX_BPS:
        d.reasons.append(f"spread {spread_bps:.0f} bps > {SPREAD_MAX_BPS:.0f}")
    if d.mu_alpha <= 0 or d.p_profit < P_PROFIT_MIN:
        d.reasons.append(f"P(profit) {d.p_profit:.0%} / alpha {d.mu_alpha:+.1%}")
    d.kelly = KELLY_FRACTION * d.mu_alpha / d.idio_vol ** 2 if d.mu_alpha > 0 else 0.0
    d.approved = not d.reasons
    d.target = -min(d.kelly, SHORT_MAX_NAME) if d.approved else 0.0
    if d.approved:
        d.reasons.append(f"SHORT composite {comp:.2f}, α {d.mu_alpha:+.1%}, σε {d.idio_vol:.0%}")
    return d


def fit_shorts(decisions: List[Decision], budget: float) -> None:
    """Scale the short targets into the short budget (fraction of NAV)."""
    tot = sum(-d.target for d in decisions if d.target < 0)
    for d in decisions:
        d.target = 0.0 if budget <= 0 else (d.target * budget / tot if tot > budget else d.target)


def fill_book(decisions: List[Decision], pool: float, single_name_budget: float,
              caps: Optional[Dict[str, float]] = None) -> float:
    """Deploy single-name budget the entry gate left unfilled (owner: 100% of NAV deployed, not cash).

    Best composite first among names that pass the fundamentals gate, have no macro veto, composite ≥ FILL_MIN
    and a tradable spread; ≤ FILL_MAX_NAME per name and ≤ SECTOR_MAX of the single-name budget per GICS sector.
    HY / distressed / event names are filled only up to their OWN sleeve cap (caps); the rest of the pool goes to
    investment-grade names, so a fully deployed book never over-weights the weaker credit sleeves.
    Returns the part that could not be placed (stays in cash)."""
    sec: Dict[str, float] = {}
    for d in decisions:
        if d.target > 0:
            sec[d.sector] = sec.get(d.sector, 0.0) + d.target
    cands = [d for d in decisions if d.sleeve in EQUITY_SLEEVES and d.gates.get("FUNDAMENTALS", 0) > 0
             and d.gates.get("MACRO", 0) >= MACRO_VETO and d.composite >= FILL_MIN
             and not any(r.startswith("spread") for r in d.reasons)]
    used: Dict[str, float] = {}
    for d in decisions:
        if d.target > 0:
            used[d.sleeve] = used.get(d.sleeve, 0.0) + d.target
    for d in sorted(cands, key=lambda d: -d.composite):
        if pool <= 1e-4:
            break
        add = min(pool, FILL_MAX_NAME - d.target, SECTOR_MAX * single_name_budget - sec.get(d.sector, 0.0))
        if caps is not None and d.sleeve != "IG":
            add = min(add, caps.get(d.sleeve, 0.0) - used.get(d.sleeve, 0.0))
        if add < 0.002:
            continue
        if not d.approved:
            d.approved, d.filled = True, True
            d.reasons = [f"deployment fill: composite {d.composite:.2f} (best available, below the entry gate)"]
        d.target += add
        sec[d.sector] = sec.get(d.sector, 0.0) + add
        used[d.sleeve] = used.get(d.sleeve, 0.0) + add
        pool -= add
    return max(pool, 0.0)


def sleeve_caps() -> Dict[str, float]:
    """Bucket caps (no IG cut for futures margin — the margin is sized to the corridor, owner decision)."""
    return dict(SLEEVES)


def futures_margin(leverage: float) -> float:
    """Approximate MES initial margin as a fraction of NAV (reporting / data health)."""
    return abs(leverage) * MES_MARGIN_RATE


def fit_sleeves(decisions: List[Decision], caps: Optional[Dict[str, float]] = None,
                sector_budget: Optional[Dict[str, float]] = None) -> None:
    """Fit Kelly targets into (sleeve × GICS sector) buckets, spilling unused sector room to other names.

    Each decision needs .sector (GICS); with no sector_budget only the sleeve cap applies."""
    caps = caps or SLEEVES
    by: Dict[str, List[Decision]] = {}
    for d in decisions:
        by.setdefault(d.sleeve, []).append(d)
    for sleeve, ds in by.items():
        if sleeve == "SHORT":
            continue                                   # sized by fit_shorts
        cap = caps.get(sleeve, 0.0) if sleeve in EQUITY_SLEEVES else 0.0
        if cap <= 0:
            for d in ds:
                d.target = 0.0
            continue
        want = {id(d): d.target for d in ds}
        if sector_budget:
            groups: Dict[str, List[Decision]] = {}
            for d in ds:
                groups.setdefault(getattr(d, "sector", ""), []).append(d)
            for sec, g in groups.items():
                room = cap * sector_budget.get(sec, 0.0)
                tot = sum(d.target for d in g)
                if tot > room:
                    for d in g:
                        d.target = d.target * room / tot if tot > 0 else 0.0
            # spill: room left by sectors without enough candidates goes to favoured sectors that still want
            # more (below Kelly), never taking any sector above SECTOR_MAX of the sleeve
            leftover = cap - sum(d.target for d in ds)
            for sec, g in sorted(groups.items(), key=lambda kv: -sector_budget.get(kv[0], 0.0)):
                if leftover <= 1e-9 or sector_budget.get(sec, 0.0) <= 0:
                    continue
                deficit = sum(want[id(d)] - d.target for d in g)
                headroom = cap * SECTOR_MAX - sum(d.target for d in g)
                add = min(leftover, deficit, max(headroom, 0.0))
                if add <= 0 or deficit <= 0:
                    continue
                for d in g:
                    d.target += add * (want[id(d)] - d.target) / deficit
                leftover -= add
        tot = sum(d.target for d in ds)
        if tot > cap:
            for d in ds:
                d.target *= cap / tot


def book_beta(weights: Dict[str, float], betas: Dict[str, float]) -> float:
    return sum(w * betas.get(t, 1.0) for t, w in weights.items())


def spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (ask - bid) / (0.5 * (ask + bid)) * 1e4
