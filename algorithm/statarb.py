"""Statistical arbitrage — cointegration mean reversion between stocks of the same GICS sector (owner, Nov 2026).

Engle-Granger, on one year of daily adjusted closes:
    1. log P_y = a + b · log P_x + e           (OLS; b = hedge ratio)
    2. e must be stationary: Dickey-Fuller t-statistic of rho in  Δe_t = rho · e_{t-1} + u_t  ≤ −3.34 (the
       Engle-Granger 5% critical value), and ≤ −2.5 on the first two thirds of the sample alone — testing
       hundreds of pairs finds some by chance; a relation that only appears in the last months is rejected
    3. half-life of the spread = −ln 2 / rho must be 3 … 40 trading days (fast enough to trade, not noise)
    4. return correlation ≥ 0.6 as a cheap pre-filter
Formation is refit every 21 trading days; the z-score of the spread is recomputed every day:
    z = (e_today − mean e) / std e
    |z| ≥ 2.0  open:  z > 0 → y is rich: SHORT y, LONG x     z < 0 → LONG y, SHORT x
    |z| ≤ 0.5  close (mean reverted)         |z| ≥ 4.0 or the pair stops being cointegrated → close (broken)
Sizing: PAIR_LEG of NAV on the y leg and PAIR_LEG × min(b, 1.5) on the x leg (beta-weighted), at most MAX_PAIRS
pairs, a name in at most one pair. The legs are added to the name's ordinary target, so a stock the book is
already long is rotated down when it is the rich leg and up when it is the cheap one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LOOKBACK = 252
ADF_ENTER = -3.34               # Engle-Granger 5% critical value (two series, constant)
ADF_STABLE = -2.5               # the same relation must already hold on the first two thirds of the sample
ADF_BROKEN = -1.8
HALF_LIFE = (3.0, 40.0)
MIN_CORR = 0.6
Z_OPEN, Z_CLOSE, Z_STOP = 2.0, 0.5, 4.0
PAIR_LEG = 0.015                # NAV fraction per leg
MAX_PAIRS = 8
MAX_HEDGE = 1.5
REFIT_DAYS = 21


@dataclass
class Pair:
    y: str
    x: str
    a: float
    b: float
    mean: float
    std: float
    adf: float
    half_life: float
    z: float = 0.0
    side: int = 0                # +1 long y / short x, −1 short y / long x, 0 flat


def _ols(y: List[float], x: List[float]) -> Tuple[float, float]:
    n = len(y)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return my, 0.0
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx
    return my - b * mx, b


def dickey_fuller(e: List[float]) -> Tuple[float, float]:
    """(t-statistic of rho, rho) for Δe_t = rho · e_{t−1} + u_t."""
    lag, d = e[:-1], [b - a for a, b in zip(e[:-1], e[1:])]
    sxx = sum(v * v for v in lag)
    if sxx <= 0 or len(d) < 30:
        return 0.0, 0.0
    rho = sum(u * v for u, v in zip(lag, d)) / sxx
    res = [v - rho * u for u, v in zip(lag, d)]
    s2 = sum(r * r for r in res) / (len(d) - 1)
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    return (rho / se if se > 0 else 0.0), rho


def _corr(a: List[float], b: List[float]) -> float:
    ra = [math.log(q / p) for p, q in zip(a[:-1], a[1:])]
    rb = [math.log(q / p) for p, q in zip(b[:-1], b[1:])]
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    va, vb = sum((v - ma) ** 2 for v in ra), sum((v - mb) ** 2 for v in rb)
    return sum((u - ma) * (v - mb) for u, v in zip(ra, rb)) / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0


def test_pair(y: str, x: str, py: List[float], px: List[float]) -> Optional[Pair]:
    n = min(len(py), len(px), LOOKBACK)
    if n < 120 or min(py[-n:]) <= 0 or min(px[-n:]) <= 0:
        return None
    py, px = py[-n:], px[-n:]
    if _corr(py, px) < MIN_CORR:
        return None
    ly, lx = [math.log(v) for v in py], [math.log(v) for v in px]
    a, b = _ols(ly, lx)
    if b <= 0.2 or b > 5:
        return None
    e = [u - a - b * v for u, v in zip(ly, lx)]
    t, rho = dickey_fuller(e)
    if t > ADF_ENTER or rho >= 0:
        return None
    if dickey_fuller(e[:2 * n // 3])[0] > ADF_STABLE:
        return None
    hl = -math.log(2) / rho if rho > -1 else 1.0
    if not HALF_LIFE[0] <= hl <= HALF_LIFE[1]:
        return None
    m = sum(e) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in e) / (n - 1))
    if sd <= 0:
        return None
    return Pair(y, x, a, b, m, sd, t, hl, (e[-1] - m) / sd)


def find_pairs(prices: Dict[str, List[float]], sectors: Dict[str, str], max_per_sector: int = 4) -> List[Pair]:
    """Cointegrated pairs inside each sector, strongest (most negative Dickey-Fuller) first."""
    by: Dict[str, List[str]] = {}
    for t, px in prices.items():
        if len(px) >= 120 and sectors.get(t):
            by.setdefault(sectors[t], []).append(t)
    out: List[Pair] = []
    for sec, names in by.items():
        found = []
        names = sorted(names)
        for i, y in enumerate(names):
            for x in names[i + 1:]:
                p = test_pair(y, x, prices[y], prices[x])
                if p is not None:
                    found.append(p)
        used, kept = set(), 0
        for p in sorted(found, key=lambda p: p.adf):
            if kept >= max_per_sector or p.y in used or p.x in used:
                continue
            used |= {p.y, p.x}
            out.append(p)
            kept += 1
    return sorted(out, key=lambda p: p.adf)


def update(pair: Pair, py: float, px: float) -> str:
    """Refresh z from today's prices and move the pair's state. Returns OPEN / CLOSE / STOP / HOLD / FLAT."""
    if py <= 0 or px <= 0:
        return "HOLD" if pair.side else "FLAT"
    pair.z = (math.log(py) - pair.a - pair.b * math.log(px) - pair.mean) / pair.std
    if pair.side == 0:
        if Z_OPEN <= abs(pair.z) < Z_STOP:
            pair.side = -1 if pair.z > 0 else 1
            return "OPEN"
        return "FLAT"
    if abs(pair.z) >= Z_STOP:
        pair.side = 0
        return "STOP"
    if abs(pair.z) <= Z_CLOSE or pair.side * pair.z > 0:      # reverted (or crossed through the mean)
        pair.side = 0
        return "CLOSE"
    return "HOLD"


def targets(pairs: List[Pair], leg: float = PAIR_LEG, max_pairs: int = MAX_PAIRS) -> Dict[str, float]:
    """Signed NAV weights of the open pairs' legs (a name appears in one open pair at most)."""
    out: Dict[str, float] = {}
    n = 0
    for p in pairs:
        if p.side == 0 or n >= max_pairs or p.y in out or p.x in out:
            continue
        out[p.y] = p.side * leg
        out[p.x] = -p.side * leg * min(p.b, MAX_HEDGE)
        n += 1
    return out
