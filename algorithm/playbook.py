"""Playbook — what history says each sector does in each kind of market (owner, Nov 2026).

The regime model says what the market is doing now. The playbook learns, from the past only, what tends to WORK
in that kind of market, and tilts the sector budgets toward it:

    market state (from SPY, each day)
        CRASH   drawdown from the 1-year high ≥ 20%
        BEAR    12-month return ≤ −5% (and not a crash)
        BULL    12-month return ≥ +10%
        FLAT    everything else
    sample      every 21 trading days over all available history (sector indexes back to 1999): the state on
                that day and each sector's NEXT 63-day return minus SPY's (the sector's excess return)
    learned     mean excess by (state, sector), annualised, shrunk toward zero by n / (n + 8)
    tilt        clip(learned / 0.25, −0.30, +0.30) added to the sector score for the CURRENT state
    natures     the same table averaged over cyclical / secular / defensive sectors (reporting, equity mix)

Walk-forward by construction: on any date only samples whose 63-day outcome is already known are used, so a
backtest never sees the future. Sector indexes (SPDR sector ETFs) are signals only — they are never held.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

STATES = ("BULL", "FLAT", "BEAR", "CRASH")
SECTOR_PROXY = {"Energy": "XLE", "Materials": "XLB", "Industrials": "XLI", "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP", "Health Care": "XLV", "Financials": "XLF",
                "Information Technology": "XLK", "Communication Services": "XLC", "Utilities": "XLU",
                "Real Estate": "XLRE"}
NATURE = {"Energy": "CYCLICAL", "Materials": "CYCLICAL", "Industrials": "CYCLICAL", "Financials": "CYCLICAL",
          "Consumer Discretionary": "CYCLICAL", "Real Estate": "CYCLICAL",
          "Information Technology": "SECULAR", "Communication Services": "SECULAR",
          "Consumer Staples": "DEFENSIVE", "Health Care": "DEFENSIVE", "Utilities": "DEFENSIVE"}
STEP, HORIZON, YEAR = 21, 63, 252
SHRINK_N = 8
TILT_SCALE, TILT_MAX = 0.25, 0.30
CRASH_DD, BEAR_R12, BULL_R12 = 0.20, -0.05, 0.10


def market_state(spy: List[float], i: Optional[int] = None) -> Optional[str]:
    """State on day i (default: the last day) from the trailing year of SPY closes."""
    i = len(spy) - 1 if i is None else i
    if i < YEAR:
        return None
    window = spy[i - YEAR:i + 1]
    dd = 1 - spy[i] / max(window)
    r12 = spy[i] / spy[i - YEAR] - 1
    if dd >= CRASH_DD:
        return "CRASH"
    if r12 <= BEAR_R12:
        return "BEAR"
    if r12 >= BULL_R12:
        return "BULL"
    return "FLAT"


def learn(spy: List[float], sectors: Dict[str, List[float]]) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """{state: {sector: (annualised mean excess return over the next 63 days, samples)}} — series are aligned on
    their LAST element (same end date); a sector with a shorter history simply has fewer samples."""
    table: Dict[str, Dict[str, List[float]]] = {s: {} for s in STATES}
    n = len(spy)
    for i in range(YEAR, n - HORIZON, STEP):
        st = market_state(spy, i)
        if st is None:
            continue
        m = spy[i + HORIZON] / spy[i] - 1
        for sec, px in sectors.items():
            j = i - (n - len(px))                      # same calendar day in the sector series
            if j < 0 or j + HORIZON >= len(px) or px[j] <= 0:
                continue
            table[st].setdefault(sec, []).append((px[j + HORIZON] / px[j] - 1) - m)
    out: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for st, by in table.items():
        out[st] = {}
        for sec, v in by.items():
            k = len(v)
            out[st][sec] = ((sum(v) / k) * (YEAR / HORIZON) * k / (k + SHRINK_N), k)
    return out


def tilts(table: Dict[str, Dict[str, Tuple[float, int]]], state: Optional[str]) -> Dict[str, float]:
    """Sector score tilts for the current state (0 for a sector with no evidence)."""
    row = table.get(state or "", {})
    return {sec: max(-TILT_MAX, min(TILT_MAX, mean / TILT_SCALE)) for sec, (mean, _) in row.items()}


def nature_view(table: Dict[str, Dict[str, Tuple[float, int]]]) -> Dict[str, Dict[str, float]]:
    """{state: {CYCLICAL / SECULAR / DEFENSIVE: mean annualised excess}} — how each nature fared in each state."""
    out: Dict[str, Dict[str, float]] = {}
    for st, row in table.items():
        agg: Dict[str, List[float]] = {}
        for sec, (mean, _) in row.items():
            agg.setdefault(NATURE.get(sec, "SECULAR"), []).append(mean)
        out[st] = {k: sum(v) / len(v) for k, v in agg.items()}
    return out


def describe(table: Dict[str, Dict[str, Tuple[float, int]]], state: Optional[str], n: int = 3) -> str:
    row = sorted(((mean, sec) for sec, (mean, _) in table.get(state or "", {}).items()), reverse=True)
    if not row:
        return f"{state}: no history yet"
    best = ", ".join(f"{s} {m:+.0%}" for m, s in row[:n])
    worst = ", ".join(f"{s} {m:+.0%}" for m, s in row[-n:])
    return f"{state}: best {best} | worst {worst}"


def leadership(returns: Dict[str, List[float]], min_names: int = 4) -> Dict[str, float]:
    """Theme leadership from the names themselves: {group: −1 … +1} = how far the group's median 63-day return
    sits from the median group, in units of the cross-group spread. Finds a theme that is running (AI build-out,
    an oil squeeze) or breaking, whatever it is called, without a ticker list."""
    med = {}
    for g, v in returns.items():
        if len(v) >= min_names:
            s = sorted(v)
            med[g] = s[len(s) // 2]
    if len(med) < 4:
        return {}
    vals = sorted(med.values())
    centre = vals[len(vals) // 2]
    spread = (vals[int(len(vals) * 0.85)] - vals[int(len(vals) * 0.15)]) or 1e-9
    return {g: max(-1.0, min(1.0, (m - centre) / spread)) for g, m in med.items()}
