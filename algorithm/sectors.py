"""The 11 GICS sectors — the allocation map for single stocks and their options (owner rule).

The research decides how the equity book is split across the 11 GICS sectors; the decision matrix then fills
each sector's budget with single stocks from the fundamentals shortlist (and options on them). Sector ETFs are
NOT bought for rotation. When a theme crashes, its sector is marked down and its weight flows to the sectors
the bullish / bearish reasoning favours — by construction, because budgets are recomputed from scores every day
(idempotent: no "freed amount" bookkeeping).

    sector score   = mean edge of its market-map sub-sectors (macro question tree + scenarios + fundamentals)
                   + regime tilt   STRESS / CRASH: defensives +0.30, cyclicals −0.20, growth −0.10
                                   TRENDING: cyclicals +0.10, growth +0.10
                   + bias tilt     BEARISH: defensives +0.20 · BULLISH: cyclicals / growth +0.10
                   + escalation    (score ≥ 0.5): Energy +0.30, Industrials (aerospace & defense) +0.15
                   − 0.50 for the sector of a STRESS / CRASH theme
    sector budget  = share of each equity sleeve: ∝ max(score − SCORE_FLOOR, 0), each sector between
                     MIN_SECTOR (for any sector with a positive budget) and MAX_SECTOR; no sector positive →
                     the defensive trio (Health Care, Consumer Staples, Utilities)
"""

from __future__ import annotations

from typing import Dict, List, Optional

GICS = {   # GICS sector: style
    "Information Technology": "GROWTH",
    "Communication Services": "GROWTH",
    "Consumer Discretionary": "CYCLICAL",
    "Financials": "CYCLICAL",
    "Industrials": "CYCLICAL",
    "Materials": "CYCLICAL",
    "Real Estate": "CYCLICAL",
    "Energy": "DEFENSIVE",
    "Health Care": "DEFENSIVE",
    "Consumer Staples": "DEFENSIVE",
    "Utilities": "DEFENSIVE",
}
SECTORS = list(GICS)
DEFENSIVE_FALLBACK = ["Health Care", "Consumer Staples", "Utilities"]
SECTOR_ETF_FALLBACK = {"Energy": "XLE"}    # only as an option underlying when no Energy single name qualifies

SUBSECTOR_TO_GICS = {
    "AI_PLATFORMS": "Communication Services", "AI_CHIP_DESIGN": "Information Technology",
    "SEMI_EQUIPMENT": "Information Technology", "FOUNDRY_REGIONAL": "Information Technology",
    "MEMORY": "Information Technology", "NETWORKING": "Information Technology", "DATA_CENTERS": "Real Estate",
    "POWER_GRID": "Utilities", "AI_NEOCLOUD_CREDIT": "Information Technology",
    "AI_HOSTING_PIVOT": "Information Technology", "LOW_END_SAAS": "Information Technology",
    "SOFTWARE": "Information Technology", "LABOR_INTENSIVE_SERVICES": "Industrials", "MATERIALS": "Materials",
    "INDUSTRIALS": "Industrials", "FINANCIALS": "Financials", "CONSUMER_DISCRETIONARY": "Consumer Discretionary",
    "REAL_ESTATE": "Real Estate", "COMMODITY_PRODUCERS": "Energy", "HEALTHCARE_PHARMA": "Health Care",
    "MEDIA_TELECOM": "Communication Services", "STAPLES": "Consumer Staples", "UTILITIES": "Utilities",
    "ENERGY_INTEGRATED": "Energy", "REFINERS": "Energy", "DEFENSE": "Industrials",
}

ROTATION_TOP = 3
SCORE_FLOOR = -0.10
MIN_SECTOR = 0.03
MAX_SECTOR = 0.30
REGIME_TILT = {
    "STRESS": {"DEFENSIVE": 0.30, "CYCLICAL": -0.20, "GROWTH": -0.10},
    "CRASH": {"DEFENSIVE": 0.30, "CYCLICAL": -0.20, "GROWTH": -0.10},
    "TRENDING": {"CYCLICAL": 0.10, "GROWTH": 0.10},
    "RANGE": {},
}
BIAS_TILT = {"BEARISH": {"DEFENSIVE": 0.20}, "BULLISH": {"CYCLICAL": 0.10, "GROWTH": 0.10}}
ESCALATION_TILT = {"Energy": 0.30, "Industrials": 0.15}
THEME_PENALTY = 0.50


def sector_scores(subsector_edges: Optional[Dict[str, float]] = None, regime: str = "RANGE",
                  bias: str = "NEUTRAL", escalation: float = 0.0,
                  crashing_subsectors: Optional[List[str]] = None) -> Dict[str, float]:
    by: Dict[str, List[float]] = {s: [] for s in SECTORS}
    for sub, edge in (subsector_edges or {}).items():
        sec = SUBSECTOR_TO_GICS.get(sub)
        if sec:
            by[sec].append(edge)
    scores = {}
    for sec, style in GICS.items():
        v = by[sec]
        s = sum(v) / len(v) if v else 0.0
        s += REGIME_TILT.get(str(regime).upper(), {}).get(style, 0.0)
        s += BIAS_TILT.get(str(bias).upper(), {}).get(style, 0.0)
        if escalation >= 0.5:
            s += ESCALATION_TILT.get(sec, 0.0)
        scores[sec] = s
    for sub in crashing_subsectors or []:
        sec = SUBSECTOR_TO_GICS.get(sub)
        if sec:
            scores[sec] -= THEME_PENALTY
    return scores


def sector_budgets(scores: Dict[str, float]) -> Dict[str, float]:
    """Fractions of each equity sleeve per GICS sector (sum 1)."""
    raw = {s: max(v - SCORE_FLOOR, 0.0) for s, v in scores.items()}
    if sum(raw.values()) <= 0:
        return {s: (1 / len(DEFENSIVE_FALLBACK) if s in DEFENSIVE_FALLBACK else 0.0) for s in SECTORS}
    w = {s: v / sum(raw.values()) for s, v in raw.items()}
    for _ in range(20):                                    # clip to [MIN, MAX] for active sectors, renormalise
        active = [s for s in w if w[s] > 0]
        clipped = {s: (min(max(w[s], MIN_SECTOR), MAX_SECTOR) if s in active else 0.0) for s in w}
        tot = sum(clipped.values())
        free = [s for s in active if MIN_SECTOR < clipped[s] < MAX_SECTOR]
        if abs(tot - 1) < 1e-9 or not free:
            w = {s: v / tot for s, v in clipped.items()}
            break
        adj = (1 - tot) / len(free)
        w = {s: clipped[s] + (adj if s in free else 0.0) for s in clipped}
    return w


def rotation_sectors(scores: Dict[str, float], top: int = ROTATION_TOP) -> List[str]:
    best = [s for s, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0][:top]
    return best or list(DEFENSIVE_FALLBACK)
