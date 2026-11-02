"""Reasoning profiles — sub-sectors and crash themes are detected from fundamentals, never from ticker lists.

Owner rule: names like CRWV / NBIS (debt-funded AI infrastructure) or IREN / HIVE (capex over-spenders) were
EXAMPLES of a pattern that crashed when AI-spend expectations met widening corporate-bond spreads. The algorithm
must find every name that fits the pattern — today's and tomorrow's — from its numbers.

Sub-sector profile (first rule that matches; an owner / agent override file can still pin a name):

    AI_NEOCLOUD_CREDIT       AI-chain sector (Technology, Communication, Real Estate, Financial Services*)
                             + capex ≥ 30% of revenue + credit-funded: debt/EBITDA ≥ 4 or coverage < 3 or D/E ≥ 1.5
    AI_HOSTING_PIVOT         AI-chain sector + capex ≥ 40% of revenue + free-cash-flow burn (FCF < 0)
    SEMI_EQUIPMENT / AI_CHIP_DESIGN / NETWORKING / DEFENSE / REFINERS / COMMODITY_PRODUCERS
                             Morningstar industry / industry-group codes (verify codes on QuantConnect)
    AI_PLATFORMS             Technology / Communication with market cap ≥ $200B and positive FCF (cash-funded demand)
    LOW_END_SAAS             software with market cap < $10B and operating margin < 10% (AI-substitution risk)
    SOFTWARE                 other software
    LABOR_INTENSIVE_SERVICES Industrials / Technology services: capex < 3% of revenue and operating margin < 8%
                             (asset-light, thin-margin, people-heavy — AI labor-substitution risk)
    MEDIA_TELECOM            other Communication Services
    sector default           Morningstar sector → market-map sub-sector
    * Financial Services only with capex ≥ 40% of revenue (e.g. crypto miners pivoting to AI hosting)

Themes = groups of profiles with the reasoning that makes them crash together. Members are the universe names
(S&P 1500 + a liquid off-index watch screen, market cap ≥ $2B) whose profile matches — largest first.
"""

from __future__ import annotations

from typing import Dict, List, Optional

AI_CHAIN_SECTORS = {311, 308, 104, 103}
CAPEX_CREDIT = 0.30
CAPEX_BURN = 0.40
LEV_DEBT_EBITDA = 4.0
LEV_COVERAGE = 3.0
LEV_DE = 1.5
PLATFORM_CAP = 200e9
SAAS_SMALL_CAP = 10e9
SAAS_MARGIN = 0.10
SERVICES_CAPEX = 0.03
SERVICES_MARGIN = 0.08
WATCH_MIN_CAP = 2e9
WATCH_MIN_DOLLAR_VOLUME = 2e7
THEME_MAX_NAMES = 10

SECTOR_DEFAULT = {
    101: "MATERIALS", 102: "CONSUMER_DISCRETIONARY", 103: "FINANCIALS", 104: "REAL_ESTATE",
    205: "STAPLES", 206: "HEALTHCARE_PHARMA", 207: "UTILITIES", 308: "MEDIA_TELECOM",
    309: "ENERGY_INTEGRATED", 310: "INDUSTRIALS", 311: "SOFTWARE",
}
GROUP_CODES = {31110: "SOFTWARE", 31120: "NETWORKING", 31130: "AI_CHIP_DESIGN", 31010: "DEFENSE"}
INDUSTRY_CODES = {31130010: "SEMI_EQUIPMENT", 30910050: "REFINERS",
                  30910010: "COMMODITY_PRODUCERS", 30910020: "COMMODITY_PRODUCERS", 30910060: "COMMODITY_PRODUCERS"}

THEMES = {
    "AI_CREDIT_HUBS": dict(
        subsectors=["AI_NEOCLOUD_CREDIT"],
        reasoning="AI infrastructure financed with debt / private credit: capex-heavy, levered, thin interest "
                  "cover. Breaks when HY spreads widen or AI-capex expectations reset (refinancing walls)."),
    "AI_OVERSPEND": dict(
        subsectors=["AI_HOSTING_PIVOT"],
        reasoning="Capex far beyond revenue with cash burn (incl. pivots into AI hosting). Breaks when funding "
                  "tightens or the payoff from AI spending is questioned."),
}
THEME_SUBSECTOR = {k: v["subsectors"][0] for k, v in THEMES.items()}


def _num(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def metrics(d: dict) -> Dict[str, float]:
    rev, capex = _num(d, "revenue"), abs(_num(d, "capex"))
    ebit, dep, debt = _num(d, "ebit"), _num(d, "depreciation"), _num(d, "total_debt")
    intexp = abs(_num(d, "interest_expense"))
    equity = _num(d, "total_assets") - _num(d, "total_liabilities")
    ebitda = ebit + dep
    return {
        "capex_intensity": capex / rev if rev > 0 else (9.9 if capex > 0 else 0.0),
        "debt_ebitda": debt / ebitda if ebitda > 0 else (99.0 if debt > 0 else 0.0),
        "coverage": ebit / intexp if intexp > 0 else 99.0,
        "debt_equity": debt / equity if equity > 0 else (99.0 if debt > 0 else 0.0),
        "op_margin": ebit / rev if rev > 0 else -1.0,
        "fcf": _num(d, "fcf"),
        "market_cap": _num(d, "market_cap"),
    }


def classify(d: dict, ticker: str = "", overrides: Optional[Dict[str, str]] = None) -> str:
    from market_map import SUBSECTORS
    if overrides and ticker in overrides and overrides[ticker] in SUBSECTORS:
        return overrides[ticker]
    m = metrics(d)
    sector, group, industry = _int(d.get("sector_code")), _int(d.get("group_code")), _int(d.get("industry_code"))
    if sector in AI_CHAIN_SECTORS:
        credit = (m["debt_ebitda"] >= LEV_DEBT_EBITDA or m["coverage"] < LEV_COVERAGE or m["debt_equity"] >= LEV_DE)
        floor = CAPEX_BURN if sector == 103 else CAPEX_CREDIT
        if m["capex_intensity"] >= floor and credit:
            return "AI_NEOCLOUD_CREDIT"
        if m["capex_intensity"] >= CAPEX_BURN and m["fcf"] < 0:
            return "AI_HOSTING_PIVOT"
    if industry in INDUSTRY_CODES:
        return INDUSTRY_CODES[industry]
    if sector in (311, 308) and m["market_cap"] >= PLATFORM_CAP and m["fcf"] > 0:
        return "AI_PLATFORMS"
    if group in GROUP_CODES and GROUP_CODES[group] != "SOFTWARE":
        return GROUP_CODES[group]
    if group == 31110 or (sector == 311 and group is None and m["capex_intensity"] < SERVICES_CAPEX
                          and m["op_margin"] >= SERVICES_MARGIN):
        if m["market_cap"] < SAAS_SMALL_CAP and m["op_margin"] < SAAS_MARGIN:
            return "LOW_END_SAAS"
        return "SOFTWARE"
    if sector in (310, 311) and m["capex_intensity"] < SERVICES_CAPEX and -1 < m["op_margin"] < SERVICES_MARGIN:
        return "LABOR_INTENSIVE_SERVICES"
    return SECTOR_DEFAULT.get(sector, "INDUSTRIALS")


def theme_members(profiles: Dict[str, str], caps: Dict[str, float], limit: int = THEME_MAX_NAMES) -> Dict[str, List[str]]:
    """{theme: tickers whose profile matches, largest market cap first}."""
    out = {}
    for theme, spec in THEMES.items():
        names = [t for t, sub in profiles.items() if sub in spec["subsectors"]]
        out[theme] = sorted(names, key=lambda t: -caps.get(t, 0.0))[:limit]
    return out


def watch_candidate(sector_code, market_cap, dollar_volume) -> bool:
    """Cheap pre-filter for off-index names that could fit a theme profile (full check after extract)."""
    return (_int(sector_code) in AI_CHAIN_SECTORS and (market_cap or 0) >= WATCH_MIN_CAP
            and (dollar_volume or 0) >= WATCH_MIN_DOLLAR_VOLUME)
