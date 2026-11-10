"""Short book — the bearish half of the long / short rotation (owner, Nov 2026).

When the market regime turns bearish the single-name sleeves stop being long-only: a share of their budget is
rotated into SHORT positions in the weakest names (and puts on them, through the options desk), so the book
becomes an active long / short book instead of going to cash.

    short share of the single-name budget (70% of NAV)
        TRENDING 0 · RANGE 0 · STRESS 30% · CRASH 50%
        at least 15% as soon as the turn begins: corridor zone DEFENSIVE (expected S&P return < 7%)
        or P(STRESS) + P(CRASH) ≥ 35% on the regime model
    candidates   scored universe names that FAIL the fundamentals gate (or composite < 45), liquid enough to
                 borrow and cover: market cap ≥ $2B, $20M / day, price ≥ $10; weakest composite first, default
                 probability counted against the name; never a name on the long shortlist; 20 names
    entry        decision.decide_short: inverted gates (weak fundamentals, weak flow, weak macro fit, downtrend);
                 composite ≥ 0.55, downtrend required (never short a rising stock), spread ≤ 30 bps,
                 P(profit) ≥ 52%
    size         Kelly on the short alpha, ≤ 5% NAV per name (squeeze risk), the short budget in total, then the
                 2.5% stock-specific VaR budget
    protection   mirror of the longs: +6% from the low → cover half, +10% → cover all, resting buy-stop at
                 low +10%, 14-day cooldown
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

SHORT_SHARE = {"TRENDING": 0.0, "RANGE": 0.0, "STRESS": 0.30, "CRASH": 0.50}
TURNING_SHARE = 0.15            # the turn has begun (zone DEFENSIVE or bear probability ≥ TURNING_PROB)
TURNING_PROB = 0.35
SHORT_MAX_NAME = 0.05           # per-name short cap, fraction of NAV
SHORTLIST_N = 20
MIN_CAP = 2e9
MIN_DOLLAR_VOLUME = 2e7
MIN_PRICE = 10.0
WEAK_COMPOSITE = 45.0
BEAR_PUTS = 5                   # puts on the strongest short ideas


def short_share(regime: str, zone: str = "", probs: Optional[Dict[str, float]] = None) -> float:
    """Share of the single-name budget held short (0 … 0.5)."""
    s = SHORT_SHARE.get(str(regime).upper(), 0.0)
    bear = sum((probs or {}).get(k, 0.0) for k in ("STRESS", "CRASH"))
    if str(zone).upper() == "DEFENSIVE" or bear >= TURNING_PROB:
        s = max(s, TURNING_SHARE)
    return s


def candidates(scores: dict, data: Dict[str, dict], exclude: Iterable[str] = (), n: int = SHORTLIST_N) -> List[str]:
    """Weakest liquid names in the scored universe (fundamentals.Score objects + their raw data dicts)."""
    skip, ranked = set(exclude), []
    for t, s in scores.items():
        d = data.get(t) or {}
        if t in skip or (s.gate_pass and s.composite >= WEAK_COMPOSITE):
            continue
        if (d.get("market_cap") or 0) < MIN_CAP or (d.get("dollar_volume") or 0) < MIN_DOLLAR_VOLUME \
                or (d.get("price") or 0) < MIN_PRICE:
            continue
        pd = getattr(s.capital, "merton_pd", None) or 0.0
        ranked.append((s.composite - 50.0 * min(pd, 1.0), t))
    return [t for _, t in sorted(ranked)[:n]]
