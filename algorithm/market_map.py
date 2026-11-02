"""Market map — how the market is wired (owner's trader framework, as data).

Every sub-sector carries:
    sector      GICS parent
    nature      CYCLICAL | SECULAR | DEFENSIVE
    chain       role in a value chain (e.g. the AI supply chain), or None
    proxy       liquid ETF proxy used for prices / trading when names are not specified
    illustrations  names that showed the pattern at one point in time — documentation only, never used to classify
    sens        sensitivity to each macro factor in [-1, +1]  (+ = benefits when the factor is high)

Factors (states in [-1, +1], computed by macro_regime.py or set by scenarios):
    GROWTH       incoming activity data vs expectation (+ hot / beating)
    INFLATION    inflation vs the 2% mandate (+ above mandate)
    LABOR        labor-market tightness (+ tight)
    WAGES        real wage growth (+ rising)
    PRODUCTIVITY productivity growth (+ rising)
    RATES        policy / long rates direction (+ rising / tightening)
    LIQUIDITY    system liquidity (Fed balance sheet, reserves, RRP) (+ abundant)
    CREDIT       corporate credit conditions (+ spreads tight / tightening)
    GEOPOLITICS  escalation risk (+ escalating)
    SENTIMENT    investor risk appetite (+ risk-on)
    AI_CAPEX     strength of the AI investment cycle (+ accelerating)
    COMMODITIES  commodity price momentum (+ rising)

The sensitivities are PRIORS (economic reasoning). The weekly walk-forward / ablation process
re-estimates them from data; agents may propose changes, the owner approves (design rule 3).
"""

from __future__ import annotations

FACTORS = ("GROWTH", "INFLATION", "LABOR", "WAGES", "PRODUCTIVITY", "RATES", "LIQUIDITY",
           "CREDIT", "GEOPOLITICS", "SENTIMENT", "AI_CAPEX", "COMMODITIES")

CYCLICAL, SECULAR, DEFENSIVE = "CYCLICAL", "SECULAR", "DEFENSIVE"


def _s(**kw):
    assert set(kw) <= set(FACTORS), set(kw) - set(FACTORS)
    return kw


SUBSECTORS = {
    # ---------------- AI supply chain (secular core + infrastructure) ----------------
    "AI_PLATFORMS": dict(sector="Communication / IT", nature=SECULAR, chain="AI:demand",
                         proxy="QQQ", illustrations=["MSFT", "GOOGL", "AMZN", "META"],
                         sens=_s(AI_CAPEX=0.8, GROWTH=0.4, RATES=-0.4, LIQUIDITY=0.4, SENTIMENT=0.5, PRODUCTIVITY=0.4)),
    "AI_CHIP_DESIGN": dict(sector="IT", nature=SECULAR, chain="AI:compute",
                           proxy="SMH", illustrations=["NVDA", "AMD", "AVGO"],
                           sens=_s(AI_CAPEX=1.0, GROWTH=0.4, RATES=-0.5, LIQUIDITY=0.5, SENTIMENT=0.7, GEOPOLITICS=-0.3)),
    "SEMI_EQUIPMENT": dict(sector="IT", nature=SECULAR, chain="AI:fab-tools",
                           proxy="SMH", illustrations=["ASML", "AMAT", "LRCX", "KLAC"],
                           sens=_s(AI_CAPEX=0.8, GROWTH=0.5, RATES=-0.4, GEOPOLITICS=-0.5, SENTIMENT=0.5)),
    "FOUNDRY_REGIONAL": dict(sector="IT", nature=SECULAR, chain="AI:fab (regional players)",
                             proxy="SMH", illustrations=["TSM", "INTC", "GFS"],
                             sens=_s(AI_CAPEX=0.8, GROWTH=0.4, GEOPOLITICS=-0.8, SENTIMENT=0.4)),
    "MEMORY": dict(sector="IT", nature=CYCLICAL, chain="AI:memory",
                   proxy="SMH", illustrations=["MU"],
                   sens=_s(AI_CAPEX=0.7, GROWTH=0.7, INFLATION=0.1, SENTIMENT=0.5, RATES=-0.3)),
    "NETWORKING": dict(sector="IT", nature=SECULAR, chain="AI:networking (incl. pivoters)",
                       proxy="IGV", illustrations=["ANET", "CSCO", "NOK"],
                       sens=_s(AI_CAPEX=0.7, GROWTH=0.3, RATES=-0.2, SENTIMENT=0.3)),
    "DATA_CENTERS": dict(sector="Real Estate", nature=SECULAR, chain="AI:data-centers",
                         proxy="SRVR", illustrations=["EQIX", "DLR"],
                         sens=_s(AI_CAPEX=0.7, RATES=-0.7, CREDIT=0.5, LIQUIDITY=0.4)),
    "POWER_GRID": dict(sector="Industrials / Utilities", nature=SECULAR, chain="AI:power-infrastructure",
                       proxy="GRID", illustrations=["VRT", "ETN", "CEG", "VST"],
                       sens=_s(AI_CAPEX=0.8, GROWTH=0.4, RATES=-0.3, COMMODITIES=0.2)),
    "AI_NEOCLOUD_CREDIT": dict(sector="IT", nature=SECULAR, chain="AI:neocloud (debt / private-credit funded)",
                               proxy=None, illustrations=["CRWV", "NBIS"],
                               sens=_s(AI_CAPEX=1.0, CREDIT=0.9, LIQUIDITY=0.8, RATES=-0.8, SENTIMENT=0.9)),
    "AI_HOSTING_PIVOT": dict(sector="IT / Financials", nature=SECULAR, chain="AI:hosting (miner pivots, capex-heavy)",
                             proxy=None, illustrations=["IREN", "HIVE"],
                             sens=_s(AI_CAPEX=0.9, CREDIT=0.7, LIQUIDITY=0.8, SENTIMENT=0.9, RATES=-0.6)),
    # ---------------- AI losers / at risk ----------------
    "LOW_END_SAAS": dict(sector="IT", nature=SECULAR, chain="AI:at-risk (substitution)",
                         proxy="IGV", illustrations=[],          # owner fills in names
                         sens=_s(AI_CAPEX=-0.7, PRODUCTIVITY=-0.4, GROWTH=0.3, RATES=-0.5, SENTIMENT=0.3)),
    "LABOR_INTENSIVE_SERVICES": dict(sector="Industrials", nature=CYCLICAL, chain="AI:at-risk (labor substitution)",
                                     proxy=None, illustrations=[],
                                     sens=_s(AI_CAPEX=-0.4, WAGES=-0.5, GROWTH=0.5)),
    # ---------------- cyclicals ----------------
    "MATERIALS": dict(sector="Materials", nature=CYCLICAL, chain=None, proxy="XLB", illustrations=["FCX", "NUE"],
                      sens=_s(GROWTH=0.8, COMMODITIES=0.8, INFLATION=0.4, RATES=-0.2)),
    "INDUSTRIALS": dict(sector="Industrials", nature=CYCLICAL, chain=None, proxy="XLI", illustrations=["CAT", "DE"],
                        sens=_s(GROWTH=0.8, CREDIT=0.4, RATES=-0.3, AI_CAPEX=0.2)),
    "FINANCIALS": dict(sector="Financials", nature=CYCLICAL, chain=None, proxy="XLF", illustrations=["JPM", "BAC"],
                       sens=_s(GROWTH=0.6, RATES=0.4, CREDIT=0.7, LIQUIDITY=0.3)),
    "CONSUMER_DISCRETIONARY": dict(sector="Consumer Discretionary", nature=CYCLICAL, chain=None, proxy="XLY",
                                   illustrations=["HD", "TSLA"],
                                   sens=_s(GROWTH=0.7, WAGES=0.7, LABOR=0.4, INFLATION=-0.5, RATES=-0.5, SENTIMENT=0.5)),
    "REAL_ESTATE": dict(sector="Real Estate", nature=CYCLICAL, chain=None, proxy="XLRE", illustrations=["PLD", "O"],
                        sens=_s(RATES=-0.9, CREDIT=0.5, GROWTH=0.3)),
    "COMMODITY_PRODUCERS": dict(sector="Energy / Materials", nature=CYCLICAL, chain=None, proxy="DBC", illustrations=[],
                                sens=_s(COMMODITIES=1.0, INFLATION=0.6, GROWTH=0.4, GEOPOLITICS=0.4)),
    # ---------------- secular (non-AI) ----------------
    "HEALTHCARE_PHARMA": dict(sector="Health Care", nature=SECULAR, chain=None, proxy="XLV",
                              illustrations=["LLY", "JNJ", "MRK", "ABBV"],
                              sens=_s(GROWTH=-0.2, SENTIMENT=-0.2, RATES=-0.2, GEOPOLITICS=0.1)),
    "SOFTWARE": dict(sector="IT", nature=SECULAR, chain=None, proxy="IGV", illustrations=[],
                     sens=_s(GROWTH=0.4, RATES=-0.5, AI_CAPEX=0.2, PRODUCTIVITY=0.2, SENTIMENT=0.4)),
    "MEDIA_TELECOM": dict(sector="Communication Services", nature=SECULAR, chain=None, proxy="XLC", illustrations=[],
                          sens=_s(GROWTH=0.3, RATES=-0.3, AI_CAPEX=0.2)),
    # ---------------- defensives ----------------
    "STAPLES": dict(sector="Consumer Staples", nature=DEFENSIVE, chain=None, proxy="XLP", illustrations=["PG", "KO"],
                    sens=_s(GROWTH=-0.5, SENTIMENT=-0.5, INFLATION=-0.3, RATES=-0.3)),
    "UTILITIES": dict(sector="Utilities", nature=DEFENSIVE, chain="AI:power (partial)", proxy="XLU",
                      illustrations=["NEE", "DUK"],
                      sens=_s(GROWTH=-0.4, RATES=-0.7, SENTIMENT=-0.4, AI_CAPEX=0.3)),
    "ENERGY_INTEGRATED": dict(sector="Energy", nature=DEFENSIVE, chain=None, proxy="XLE", illustrations=["XOM", "CVX"],
                              sens=_s(COMMODITIES=0.9, GEOPOLITICS=0.7, INFLATION=0.6, GROWTH=0.2)),
    "REFINERS": dict(sector="Energy", nature=DEFENSIVE, chain=None, proxy="CRAK", illustrations=["VLO", "MPC"],
                     sens=_s(GEOPOLITICS=0.7, COMMODITIES=0.5, GROWTH=0.3, INFLATION=0.4)),
    "DEFENSE": dict(sector="Industrials", nature=DEFENSIVE, chain=None, proxy="ITA", illustrations=["LMT", "NOC", "RTX", "GD"],
                    sens=_s(GEOPOLITICS=1.0, GROWTH=-0.1, SENTIMENT=-0.3)),
}

# Hedge assets used by the stock-bond correlation regime
HEDGES = {
    "TREASURIES": dict(proxy="TLT", note="hedge when stock-bond correlation is NEGATIVE (growth-driven regime)"),
    "GOLD": dict(proxy="GLD", note="hedge in inflation / geopolitical regimes"),
    "COMMODITIES": dict(proxy="DBC", note="hedge when stock-bond correlation is POSITIVE (inflation-driven)"),
    "CASH": dict(proxy=None, note="hedge of last resort"),
}


def by_nature(nature: str):
    return [k for k, v in SUBSECTORS.items() if v["nature"] == nature]


def chain(prefix: str = "AI"):
    """Sub-sectors in a value chain, e.g. chain('AI') → the AI supply chain map."""
    return {k: v["chain"] for k, v in SUBSECTORS.items() if v["chain"] and v["chain"].startswith(prefix)}
