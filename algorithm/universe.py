"""Universe: point-in-time S&P 1500, Morningstar fundamentals adapter, sub-sector mapping, shortlist.

    membership   S&P 500 + 400 + 600 = constituents of IVV + IJH + IJR (QuantConnect ETF-constituent
                 universes, point in time, refreshed daily). Fallback mode "fundamental": US names with
                 fundamentals, price ≥ $5, market cap ≥ $1B, dollar volume ≥ $5M.
    adapter      QC `Fundamental` (Morningstar) → plain dict for fundamentals.analyze. Every field is read
                 through guarded attribute paths with alternatives; a missing field is counted in data
                 health, never raised.
    sub-sector   fundamentals profile (profiles.py: capex intensity, leverage, cash burn, margins, size,
                 Morningstar codes); an override file (ObjectStore metadron/subsector_map.json) can pin a name.
    shortlist    names that pass the Fundamentals gate, ranked by gate score + macro tilt
                 (sub-sector edge), slots per sleeve split across the 11 GICS sectors by the sector budgets
                 (sectors.py), split by sleeve in the allocation ratios
                 IG 40 : HY 10 : DISTRESSED 10. Macro vetoes: a sub-sector with zero weight and edge < −0.15
                 contributes no names (macro can shrink / veto; micro cannot override).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


SP1500_ETFS = ("IVV", "IJH", "IJR")
EXPECTED_MEMBERS = 1500
MIN_PRICE = 5.0
MIN_CAP = 1e9
MIN_DOLLAR_VOLUME = 5e6
SHORTLIST_N = 60
SLEEVE_MIX = {"IG": 40, "HY": 10, "DISTRESSED": 10}
MACRO_TILT = 0.25            # rank = gate_score + 0.25 × clip(sub-sector edge, −1, 1)
VOL_WINDOW = 63
MACRO_VETO_EDGE = -0.15     # veto only clearly negative sub-sectors with zero weight

SECTOR_NAME = {101: "Materials", 102: "Consumer Discretionary", 103: "Financials", 104: "Real Estate",
               205: "Consumer Staples", 206: "Health Care", 207: "Utilities", 308: "Communication Services",
               309: "Energy", 310: "Industrials", 311: "Information Technology"}
SECTOR_TYPE = {207: "UTILITY", 103: "FINANCIAL", 104: "REAL_ESTATE", 309: "NATURAL_RESOURCE", 101: "NATURAL_RESOURCE"}


def subsector_for(ticker: str, d: dict, overrides: Optional[Dict[str, str]] = None) -> str:
    """Sub-sector from the name's fundamentals profile (profiles.classify) — never from a ticker list."""
    from profiles import classify
    return classify(d, ticker, overrides)


# ------------------------------------------------------------------------------- QC Fundamental adapter
def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read(obj, path: str, flow: bool = False) -> Optional[float]:
    """Walk a dotted attribute path; for Morningstar multi-period fields prefer twelve_months (flows)
    or value (stocks). Returns None when anything is missing or zero-filled."""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    if not isinstance(cur, (int, float)):
        for attr in (("twelve_months", "value") if flow else ("value", "twelve_months")):
            v = _num(getattr(cur, attr, None))
            if v not in (None, 0.0):
                return v
        return None
    v = _num(cur)
    return v if v not in (None, 0.0) else None


def first(obj, paths: Iterable[str], flow: bool = False) -> Optional[float]:
    for p in paths:
        v = read(obj, p, flow)
        if v is not None:
            return v
    return None


BS, IS, CF = "financial_statements.balance_sheet", "financial_statements.income_statement", \
    "financial_statements.cash_flow_statement"
FIELDS = {   # name: (paths, is_flow)
    "price": (["price"], False),
    "market_cap": (["market_cap"], False),
    "dollar_volume": (["dollar_volume"], False),
    "shares_out": (["company_profile.shares_outstanding", "earning_reports.basic_average_shares"], False),
    "eps_ttm": (["earning_reports.basic_eps", "earning_reports.diluted_eps"], True),
    "bvps": (["valuation_ratios.book_value_per_share"], False),
    "tangible_bvps": (["valuation_ratios.tangible_book_value_per_share", "valuation_ratios.tangible_bv_per_share"], False),
    "total_assets": ([f"{BS}.total_assets"], False),
    "total_liabilities": ([f"{BS}.total_liabilities_net_minority_interest", f"{BS}.total_liabilities"], False),
    "current_assets": ([f"{BS}.current_assets"], False),
    "current_liabilities": ([f"{BS}.current_liabilities"], False),
    "total_debt": ([f"{BS}.total_debt"], False),
    "cash": ([f"{BS}.cash_and_cash_equivalents", f"{BS}.cash"], False),
    "retained_earnings": ([f"{BS}.retained_earnings"], False),
    "revenue": ([f"{IS}.total_revenue", f"{IS}.operating_revenue"], True),
    "ebit": ([f"{IS}.ebit", f"{IS}.operating_income"], True),
    "net_income": ([f"{IS}.net_income", f"{IS}.net_income_common_stockholders"], True),
    "interest_expense": ([f"{IS}.interest_expense"], True),
    "depreciation": ([f"{IS}.depreciation_and_amortization", f"{CF}.depreciation_and_amortization"], True),
    "fcf": ([f"{CF}.free_cash_flow"], True),
    "capex": ([f"{CF}.capital_expenditure"], True),
    "roic": (["operation_ratios.roic"], False),
}
REQUIRED = ("price", "market_cap", "shares_out", "eps_ttm", "total_assets", "total_liabilities", "net_income")


def extract(f) -> Tuple[dict, List[str]]:
    """QC Fundamental → (dict, missing required fields)."""
    d = {}
    for name, (paths, flow) in FIELDS.items():
        v = first(f, paths, flow)
        if v is not None:
            d[name] = v
    ac = getattr(f, "asset_classification", None)
    d["sector_code"] = getattr(ac, "morningstar_sector_code", None) if ac is not None else None
    d["group_code"] = getattr(ac, "morningstar_industry_group_code", None) if ac is not None else None
    d["industry_code"] = getattr(ac, "morningstar_industry_code", None) if ac is not None else None
    try:
        d["sector"] = SECTOR_NAME.get(int(d["sector_code"]), "")
    except (TypeError, ValueError):
        d["sector"] = ""
    d["file_date"] = _file_date(f)                         # last earnings report filed (→ next earnings estimate)
    missing = [k for k in REQUIRED if k not in d]
    return d, missing


def _file_date(f) -> Optional[str]:
    """ISO date of the latest earnings report's filing, from Morningstar (None when unavailable)."""
    for path in ("earning_reports.file_date", "financial_statements.file_date"):
        cur = f
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        for cand in (cur, getattr(cur, "three_months", None), getattr(cur, "value", None),
                     getattr(cur, "twelve_months", None)):
            if cand is not None and hasattr(cand, "year") and getattr(cand, "year", 0) > 1990:
                try:
                    return f"{cand.year:04d}-{cand.month:02d}-{cand.day:02d}"
                except Exception:
                    continue
    return None


# ------------------------------------------------------------------------------- state between days
class EpsHistory:
    """Annual EPS samples built from the trailing-12m EPS seen at each new filing (up to 10 years)."""

    def __init__(self, years: int = 10):
        self.q: Dict[str, deque] = {}
        self.last: Dict[str, float] = {}
        self.years = years

    def update(self, ticker: str, eps_ttm: Optional[float]) -> None:
        if eps_ttm is None or self.last.get(ticker) == eps_ttm:
            return
        self.last[ticker] = eps_ttm
        self.q.setdefault(ticker, deque(maxlen=4 * self.years)).append(eps_ttm)

    def annual(self, ticker: str) -> List[float]:
        q = list(self.q.get(ticker, []))
        return q[::-1][::4][::-1]


class PriceBuffer:
    """Daily prices for every member (from the universe feed) → equity volatility for Merton."""

    def __init__(self, n: int = VOL_WINDOW + 1):
        self.p: Dict[str, deque] = {}
        self.n = n

    def update(self, ticker: str, price: Optional[float]) -> None:
        if price and price > 0:
            self.p.setdefault(ticker, deque(maxlen=self.n)).append(price)

    def vol(self, ticker: str) -> Optional[float]:
        px = list(self.p.get(ticker, []))
        if len(px) < 21:
            return None
        r = [math.log(b / a) for a, b in zip(px[:-1], px[1:]) if a > 0 and b > 0]
        m = sum(r) / len(r)
        return math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1) * 252)


def passes_liquidity(d: dict) -> bool:
    return (d.get("price") or 0) >= MIN_PRICE and (d.get("dollar_volume") or 0) >= MIN_DOLLAR_VOLUME


def passes_fallback(d: dict) -> bool:
    return passes_liquidity(d) and (d.get("market_cap") or 0) >= MIN_CAP


# ------------------------------------------------------------------------------- shortlist
@dataclass
class Shortlist:
    names: List[str] = field(default_factory=list)
    by_sleeve: Dict[str, List[str]] = field(default_factory=dict)
    rank: Dict[str, float] = field(default_factory=dict)
    vetoed: List[str] = field(default_factory=list)


def shortlist(scores: dict, subsector_views: Optional[Dict[str, dict]] = None, n: int = SHORTLIST_N,
              held: Iterable[str] = (), sector_budget: Optional[Dict[str, float]] = None) -> Shortlist:
    """scores: {ticker: fundamentals.Score}; subsector_views: {subsector: {"edge", "weight"}};
    sector_budget: {GICS sector: share} — each sleeve's slots are split across sectors in proportion, so the
    favoured sectors get enough single-name candidates (rest filled by best rank)."""
    views = subsector_views or {}
    out = Shortlist(by_sleeve={k: [] for k in SLEEVE_MIX})
    ranked: Dict[str, List[Tuple[float, str, str]]] = {k: [] for k in SLEEVE_MIX}
    for t, s in scores.items():
        if not s.gate_pass:
            continue
        v = views.get(s.subsector)
        if v is not None and v.get("weight", 1.0) <= 0 and v.get("edge", 0.0) < MACRO_VETO_EDGE:
            out.vetoed.append(t)
            continue
        edge = max(-1.0, min(1.0, v.get("edge", 0.0))) if v else 0.0
        r = s.gate_score + MACRO_TILT * edge
        out.rank[t] = r
        ranked.setdefault(s.capital.sleeve, []).append((r, t, s.sector))
    total = sum(SLEEVE_MIX.values())
    for sleeve, share in SLEEVE_MIX.items():
        k = max(1, round(n * share / total))
        cands = sorted(ranked.get(sleeve, []), reverse=True)
        picks: List[str] = []
        if sector_budget:
            for sec, w in sorted(sector_budget.items(), key=lambda x: -x[1]):
                slots = int(round(k * w))
                if w > 0 and slots == 0:
                    slots = 1
                picks += [t for _, t, sc in cands if sc == sec and t not in picks][:slots]
            picks = picks[:k]
        picks += [t for _, t, _ in cands if t not in picks][:k - len(picks)]
        out.by_sleeve[sleeve] = picks
        out.names.extend(picks)
    for t in held:                       # always keep held names in the data feed
        if t not in out.names:
            out.names.append(t)
    return out
