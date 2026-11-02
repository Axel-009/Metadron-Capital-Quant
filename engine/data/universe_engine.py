"""L1 Data — Universe Engine.

Manages the complete cross-asset investment universe:
    - ~2,500+ equities: S&P 500 + S&P MidCap 400 + S&P SmallCap 600 + extras
    - Dynamic loading via OpenBB index.constituents() when API keys available
    - Static fallback from cross_asset_universe.py for offline operation
    - Full GICS 4-tier taxonomy (11 sectors / 25 industry groups / 74 industries / 163 sub-industries)
    - 70+ ETFs covering sectors, factors, commodities, fixed income, volatility
    - Asset class routing: TRADEABLE vs MACRO_ONLY (bonds/commodities/vol for analysis only)
    - 26 relative value (RV) pairs for pair trading
    - GIC pooling integration methodology
    - DailyUniverseScanner for morning pre-market scans
    - Fallen angel detection (credit downgrade candidates)
    - Quality scoring integration (A–G tiers)
    - Market cap categorisation (mega/large/mid/small/micro)
    - Sector rotation signals

All data via OpenBB — unified, free, no broker dependency.
try/except on ALL external imports — system runs degraded, never broken.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .openbb_data import (
        get_adj_close, get_returns, get_prices,
        get_fundamentals, get_bulk_fundamentals,
    )
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_prices(*a, **kw): return pd.DataFrame()
    def get_fundamentals(*a, **kw): return {}
    def get_bulk_fundamentals(*a, **kw): return {}

from .cross_asset_universe import (
    SP500_TICKERS, SP400_TICKERS, SP600_TICKERS, EXTRA_TICKERS,
    SECTOR_MAP, INDUSTRY_GROUP_MAP, OPENBB_INDEX_SYMBOLS,
    get_all_static_tickers, get_sector_for_ticker,
    get_industry_group_for_ticker,
)

logger = logging.getLogger("metadron.scan.universe_engine")

# ═══════════════════════════════════════════════════════════════════════════
# GICS Taxonomy — 11 sectors
# ═══════════════════════════════════════════════════════════════════════════
GICS_SECTORS = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}

GICS_INDUSTRY_GROUPS = {
    1010: ("Energy", "Energy Equipment & Services"),
    1020: ("Energy", "Oil, Gas & Consumable Fuels"),
    1510: ("Materials", "Chemicals"),
    1520: ("Materials", "Construction Materials"),
    1530: ("Materials", "Containers & Packaging"),
    1540: ("Materials", "Metals & Mining"),
    1550: ("Materials", "Paper & Forest Products"),
    2010: ("Industrials", "Capital Goods"),
    2020: ("Industrials", "Commercial & Professional Services"),
    2030: ("Industrials", "Transportation"),
    2510: ("Consumer Discretionary", "Automobiles & Components"),
    2520: ("Consumer Discretionary", "Consumer Durables & Apparel"),
    2530: ("Consumer Discretionary", "Consumer Services"),
    2550: ("Consumer Discretionary", "Retailing"),
    2560: ("Consumer Discretionary", "Consumer Discretionary Distribution & Retail"),
    3010: ("Consumer Staples", "Food & Staples Retailing"),
    3020: ("Consumer Staples", "Food, Beverage & Tobacco"),
    3030: ("Consumer Staples", "Household & Personal Products"),
    3510: ("Health Care", "Health Care Equipment & Services"),
    3520: ("Health Care", "Pharmaceuticals, Biotechnology & Life Sciences"),
    4010: ("Financials", "Banks"),
    4020: ("Financials", "Financial Services"),
    4030: ("Financials", "Insurance"),
    4040: ("Financials", "Diversified Financials"),
    4510: ("Information Technology", "Software & Services"),
    4520: ("Information Technology", "Technology Hardware & Equipment"),
    4530: ("Information Technology", "Semiconductors & Semiconductor Equipment"),
    5010: ("Communication Services", "Telecommunication Services"),
    5020: ("Communication Services", "Media & Entertainment"),
    5510: ("Utilities", "Utilities"),
    6010: ("Real Estate", "Equity Real Estate Investment Trusts (REITs)"),
    6020: ("Real Estate", "Real Estate Management & Development"),
}

GICS_INDUSTRIES = {
    101010: "Oil & Gas Drilling", 101020: "Oil & Gas Equipment & Services",
    102010: "Integrated Oil & Gas", 102020: "Oil & Gas Exploration & Production",
    102030: "Oil & Gas Refining & Marketing", 102040: "Oil & Gas Storage & Transportation",
    102050: "Coal & Consumable Fuels",
    151010: "Commodity Chemicals", 151020: "Diversified Chemicals",
    151030: "Fertilizers & Agricultural Chemicals", 151040: "Industrial Gases",
    151050: "Specialty Chemicals", 152010: "Construction Materials",
    153010: "Metal & Glass Containers", 153020: "Paper Packaging",
    154010: "Aluminum", 154020: "Diversified Metals & Mining", 154030: "Copper",
    154040: "Gold", 154050: "Precious Metals & Minerals", 154060: "Silver", 154070: "Steel",
    155010: "Forest Products", 155020: "Paper Products",
    201010: "Aerospace & Defense", 201020: "Building Products",
    201030: "Construction & Engineering", 201040: "Electrical Equipment",
    201050: "Industrial Conglomerates", 201060: "Machinery",
    201070: "Trading Companies & Distributors",
    202010: "Commercial Services & Supplies", 202020: "Professional Services",
    203010: "Air Freight & Logistics", 203020: "Passenger Airlines",
    203030: "Marine Transportation", 203040: "Ground Transportation",
    203050: "Transportation Infrastructure",
    251010: "Auto Components", 251020: "Automobiles",
    252010: "Household Durables", 252020: "Leisure Products",
    252030: "Textiles, Apparel & Luxury Goods",
    253010: "Hotels, Restaurants & Leisure", 253020: "Diversified Consumer Services",
    255010: "Distributors", 255020: "Internet & Direct Marketing Retail",
    255030: "Broadline Retail", 255040: "Specialty Retail",
    301010: "Food & Staples Retailing",
    302010: "Beverages", 302020: "Food Products", 302030: "Tobacco",
    303010: "Household Products", 303020: "Personal Care Products",
    351010: "Health Care Equipment & Supplies", 351020: "Health Care Providers & Services",
    351030: "Health Care Technology",
    352010: "Biotechnology", 352020: "Pharmaceuticals",
    352030: "Life Sciences Tools & Services",
    401010: "Diversified Banks", 401020: "Regional Banks",
    402010: "Diversified Financial Services", 402020: "Consumer Finance",
    402030: "Capital Markets", 402040: "Mortgage Real Estate Investment Trusts",
    403010: "Insurance",
    451010: "IT Consulting & Other Services", 451020: "Internet Services & Infrastructure",
    451030: "Application Software", 451040: "Systems Software",
    452010: "Communications Equipment",
    452020: "Technology Hardware, Storage & Peripherals",
    452030: "Electronic Equipment, Instruments & Components",
    453010: "Semiconductor Materials & Equipment", 453020: "Semiconductors",
    501010: "Alternative Carriers", 501020: "Integrated Telecommunication Services",
    501030: "Wireless Telecommunication Services",
    502010: "Advertising", 502020: "Broadcasting", 502030: "Cable & Satellite",
    502040: "Publishing", 502050: "Movies & Entertainment",
    502060: "Interactive Home Entertainment", 502070: "Interactive Media & Services",
    551010: "Electric Utilities", 551020: "Gas Utilities",
    551030: "Multi-Utilities", 551040: "Water Utilities",
    551050: "Independent Power & Renewable Electricity",
    601010: "Diversified REITs", 601025: "Industrial REITs",
    601030: "Hotel & Resort REITs", 601040: "Office REITs",
    601050: "Health Care REITs", 601060: "Residential REITs",
    601070: "Retail REITs", 601080: "Specialized REITs",
    602010: "Real Estate Operating Companies", 602020: "Real Estate Development",
    602030: "Real Estate Services",
}

# ═══════════════════════════════════════════════════════════════════════════
# Sector ETFs — 11 sectors
# ═══════════════════════════════════════════════════════════════════════════
SECTOR_ETFS = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Health Care": "XLV", "Financials": "XLF",
    "Information Technology": "XLK", "Communication Services": "XLC",
    "Utilities": "XLU", "Real Estate": "XLRE",
}

# ═══════════════════════════════════════════════════════════════════════════
# 70+ ETFs
# ═══════════════════════════════════════════════════════════════════════════
FACTOR_ETFS = {
    "Momentum": "MTUM", "Value": "VLUE", "Quality": "QUAL",
    "Low Volatility": "USMV", "Size": "SIZE", "Dividend Growth": "DGRO",
    "High Dividend": "HDV", "Growth": "VUG", "Deep Value": "RPV", "High Beta": "SPHB",
}
COMMODITY_ETFS = {
    "Gold": "GLD", "Silver": "SLV", "Oil": "USO", "Natural Gas": "UNG",
    "Broad Commodities": "DBC", "Broad Alt": "DJP",
    "Agriculture": "DBA", "Copper": "CPER", "Platinum": "PPLT",
}
FIXED_INCOME_ETFS = {
    "Treasury 20Y+": "TLT", "Treasury 7-10Y": "IEF", "Treasury 1-3Y": "SHY",
    "Aggregate Bond": "AGG", "Total Bond": "BND",
    "TIPS": "TIP", "Investment Grade Corp": "LQD", "IG Intermediate": "VCIT",
    "High Yield Corp": "HYG", "High Yield Junk": "JNK",
    "Fallen Angels": "ANGL", "Fallen Angels Alt": "FALN",
    "Muni Bond": "MUB", "Floating Rate": "FLOT", "Emerging Market Debt": "EMB",
}
VOLATILITY_ETFS = {
    "VIX Short-Term": "VXX", "VIX Ultra": "UVXY", "Short VIX": "SVXY",
    "VIX Short-Term Alt": "VIXY", "VIX Mid-Term": "VIXM",
}
INTERNATIONAL_ETFS = {
    "MSCI EAFE": "EFA", "Emerging Markets": "EEM", "EM Core": "IEMG",
    "China": "FXI", "Japan": "EWJ", "Europe": "VGK",
    "India": "INDA", "Brazil": "EWZ", "UK": "EWU",
}
INDEX_ETFS = {
    "S&P 500": "SPY", "NASDAQ 100": "QQQ", "Russell 2000": "IWM",
    "S&P MidCap": "MDY", "Dow Jones": "DIA", "Total Market": "VTI", "Equal Weight S&P": "RSP",
}
THEMATIC_ETFS = {
    "Semiconductors": "SMH", "Biotech": "XBI", "Clean Energy": "ICLN",
    "Cybersecurity": "HACK", "Robotics & AI": "BOTZ", "Cloud Computing": "SKYY",
    "Blockchain": "BLOK", "Space": "UFO",
}

ALL_ETFS = sorted(set(
    list(SECTOR_ETFS.values()) + list(FACTOR_ETFS.values()) +
    list(COMMODITY_ETFS.values()) + list(FIXED_INCOME_ETFS.values()) +
    list(VOLATILITY_ETFS.values()) + list(INTERNATIONAL_ETFS.values()) +
    list(INDEX_ETFS.values()) + list(THEMATIC_ETFS.values())
))

# ═══════════════════════════════════════════════════════════════════════════
# 26 Relative Value Pairs
# ═══════════════════════════════════════════════════════════════════════════
RV_PAIRS = [
    ("AAPL", "MSFT"), ("GOOGL", "META"), ("AMZN", "WMT"),
    ("JPM", "BAC"), ("GS", "MS"), ("XOM", "CVX"),
    ("PFE", "MRK"), ("JNJ", "ABT"), ("KO", "PEP"),
    ("HD", "LOW"), ("V", "MA"), ("DIS", "CMCSA"),
    ("UNH", "CI"), ("BA", "LMT"), ("CAT", "DE"),
    ("NVDA", "AMD"), ("CRM", "ORCL"), ("NFLX", "DIS"),
    ("COST", "TGT"), ("UPS", "FDX"), ("NEE", "DUK"),
    ("SPG", "O"), ("SLB", "HAL"), ("MCD", "SBUX"),
    ("NKE", "LULU"), ("TSLA", "F"),
]


class CreditRating(str, Enum):
    """Credit rating scale."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"


class SecurityType(str, Enum):
    """Classification of security type."""
    EQUITY = "EQUITY"
    SECTOR_ETF = "SECTOR_ETF"
    BROAD_ETF = "BROAD_ETF"
    FACTOR_ETF = "FACTOR_ETF"
    FIXED_INCOME_ETF = "FIXED_INCOME_ETF"
    COMMODITY_ETF = "COMMODITY_ETF"
    VOLATILITY_ETF = "VOLATILITY_ETF"
    INTERNATIONAL_ETF = "INTERNATIONAL_ETF"
    INDEX_ETF = "INDEX_ETF"
    THEMATIC_ETF = "THEMATIC_ETF"


class AssetClass(str, Enum):
    """Asset class routing — determines whether an instrument flows through
    the execution pipeline or is macro-analysis-only.

    TRADEABLE: Stocks, equity ETFs, futures, options → allocator → alpha →
               beta corridor → decision matrix → execution via broker.
    MACRO_ONLY: Bonds, currencies, commodities (non-ETF) → analysed for
                macro regime, sector allocation, cross-asset correlation,
                but NEVER sent to execution.
    """
    TRADEABLE = "TRADEABLE"
    MACRO_ONLY = "MACRO_ONLY"


# Map SecurityType → AssetClass for pipeline routing
# Stocks, sector/index/factor/thematic ETFs are tradeable via IBKRBroker.
# Fixed income, commodity, and volatility ETFs are tradeable as ETFs on
# Serve primarily as macro signals — we allow execution of
# sector/index/factor/thematic ETFs only.
SECURITY_TYPE_ASSET_CLASS = {
    SecurityType.EQUITY: AssetClass.TRADEABLE,
    SecurityType.SECTOR_ETF: AssetClass.TRADEABLE,
    SecurityType.INDEX_ETF: AssetClass.TRADEABLE,
    SecurityType.FACTOR_ETF: AssetClass.TRADEABLE,
    SecurityType.THEMATIC_ETF: AssetClass.TRADEABLE,
    SecurityType.BROAD_ETF: AssetClass.TRADEABLE,
    SecurityType.INTERNATIONAL_ETF: AssetClass.TRADEABLE,
    # Macro-only: analysed but NOT executed
    SecurityType.FIXED_INCOME_ETF: AssetClass.MACRO_ONLY,
    SecurityType.COMMODITY_ETF: AssetClass.MACRO_ONLY,
    SecurityType.VOLATILITY_ETF: AssetClass.MACRO_ONLY,
}

# Ticker-level override for ETFs that should be macro-only
# These tickers are used by CrossAssetMonitor/MacroEngine for regime
# detection but should never flow through the execution pipeline.
MACRO_ONLY_TICKERS = set(
    list(FIXED_INCOME_ETFS.values()) +
    list(COMMODITY_ETFS.values()) +
    list(VOLATILITY_ETFS.values())
)

# Tradeable ETFs — execution eligible
TRADEABLE_ETFS = set(
    list(SECTOR_ETFS.values()) +
    list(INDEX_ETFS.values()) +
    list(FACTOR_ETFS.values()) +
    list(THEMATIC_ETFS.values()) +
    list(INTERNATIONAL_ETFS.values())
)


class GICSSector(str, Enum):
    """GICS sector enum."""
    ENERGY = "Energy"
    MATERIALS = "Materials"
    INDUSTRIALS = "Industrials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    COMMUNICATION_SERVICES = "Communication Services"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"


class AvgVolTier(str, Enum):
    """Average daily volume tier."""
    ULTRA_LIQUID = "ULTRA_LIQUID"   # >50M shares/day
    HIGH = "HIGH"                    # 10-50M
    MEDIUM = "MEDIUM"                # 1-10M
    LOW = "LOW"                      # <1M


class MarketCapTier(str, Enum):
    MEGA = "MEGA"
    LARGE = "LARGE"
    MID = "MID"
    SMALL = "SMALL"
    MICRO = "MICRO"


def classify_market_cap(cap: float) -> MarketCapTier:
    if cap >= 200e9: return MarketCapTier.MEGA
    if cap >= 10e9: return MarketCapTier.LARGE
    if cap >= 2e9: return MarketCapTier.MID
    if cap >= 300e6: return MarketCapTier.SMALL
    return MarketCapTier.MICRO


@dataclass
class Security:
    """Security object with full GICS 4-tier classification.

    Each security in the universe carries complete classification data
    for use across all engine layers.
    """
    ticker: str = ""
    name: str = ""
    security_type: str = SecurityType.EQUITY.value
    sector: str = ""
    gics_sector: str = ""          # GICSSector enum value
    industry_group: str = ""       # GICS tier 2
    industry: str = ""             # GICS tier 3
    sub_industry: str = ""         # GICS tier 4
    gics_code: int = 0
    market_cap: float = 0.0
    market_cap_tier: str = ""
    quality_tier: str = "D"
    avg_volume: float = 0.0
    avg_vol_tier: str = AvgVolTier.MEDIUM.value
    price: float = 0.0
    beta: float = 1.0
    sharpe_12m: float = 0.0
    momentum_3m: float = 0.0
    momentum_12m: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    dividend_yield: float = 0.0
    roe: float = 0.0
    debt_equity: float = 0.0
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    free_cash_flow_yield: float = 0.0
    options_eligible: bool = True
    sector_etf: str = ""           # Associated sector ETF
    rv_pair: str = ""              # RV pair partner ticker
    credit_quality_score: float = 0.0
    credit_rating: str = ""
    interest_coverage: float = 0.0
    is_fallen_angel: bool = False
    is_rv_candidate: bool = False
    last_updated: str = ""

    @property
    def cap_tier(self) -> MarketCapTier:
        return classify_market_cap(self.market_cap)

    @property
    def asset_class(self) -> AssetClass:
        """Determine whether this security is tradeable or macro-only."""
        # Ticker-level override first
        if self.ticker in MACRO_ONLY_TICKERS:
            return AssetClass.MACRO_ONLY
        if self.ticker in TRADEABLE_ETFS:
            return AssetClass.TRADEABLE
        # Fall back to SecurityType mapping
        try:
            sec_type = SecurityType(self.security_type)
            return SECURITY_TYPE_ASSET_CLASS.get(sec_type, AssetClass.TRADEABLE)
        except ValueError:
            return AssetClass.TRADEABLE  # Default: equities are tradeable

    @property
    def is_tradeable(self) -> bool:
        """True if this instrument can flow through allocator → execution."""
        return self.asset_class == AssetClass.TRADEABLE

    @property
    def is_macro_only(self) -> bool:
        """True if this instrument is analysed for macro context only."""
        return self.asset_class == AssetClass.MACRO_ONLY


class GICPoolingEngine:
    def __init__(self):
        self._pools: dict[str, list[str]] = {}
        self._pool_weights: dict[str, dict[str, float]] = {}

    def create_pool(self, pool_name: str, tickers: list[str], weights: Optional[dict[str, float]] = None):
        self._pools[pool_name] = list(tickers)
        if weights:
            self._pool_weights[pool_name] = dict(weights)
        else:
            n = len(tickers)
            self._pool_weights[pool_name] = {t: 1.0 / n for t in tickers} if n > 0 else {}

    def get_pool_exposure(self, pool_name: str) -> dict[str, float]:
        return dict(self._pool_weights.get(pool_name, {}))

    def compute_pool_return(self, pool_name: str, returns: pd.DataFrame) -> pd.Series:
        tickers = self._pools.get(pool_name, [])
        weights = self._pool_weights.get(pool_name, {})
        if not tickers or returns.empty:
            return pd.Series(dtype=float)
        available = [t for t in tickers if t in returns.columns]
        if not available:
            return pd.Series(dtype=float)
        w = np.array([weights.get(t, 0) for t in available])
        s = w.sum()
        if s > 0:
            w = w / s
        return returns[available].dot(w)

    def list_pools(self) -> list[str]:
        return list(self._pools.keys())


class FallenAngelDetector:
    CRITERIA = {
        "max_drawdown_from_high": -0.30,
        "min_market_cap": 300e6,   # No mega cap restriction — include all caps
        "min_avg_volume": 500_000,
        "max_pe_ratio": 25.0,
        "min_roe": 0.05,
    }

    def scan(self, securities: list[Security]) -> list[Security]:
        angels = []
        for sec in securities:
            if sec.market_cap < self.CRITERIA["min_market_cap"]:
                continue
            if sec.momentum_3m > self.CRITERIA["max_drawdown_from_high"]:
                continue
            if sec.roe < self.CRITERIA["min_roe"]:
                continue
            if 0 < sec.pe_ratio <= self.CRITERIA["max_pe_ratio"]:
                sec.is_fallen_angel = True
                angels.append(sec)
        return angels


class SectorRotationEngine:
    def __init__(self):
        self._lookback_short = 21
        self._lookback_medium = 63
        self._lookback_long = 252

    def compute_sector_momentum(self, sector_returns: Optional[pd.DataFrame] = None) -> dict[str, float]:
        if sector_returns is None:
            try:
                start = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
                prices = get_adj_close(list(SECTOR_ETFS.values()), start=start)
                if prices.empty:
                    return {}
                sector_returns = prices.pct_change().dropna()
            except Exception:
                return {}

        scores = {}
        inv_map = {v: k for k, v in SECTOR_ETFS.items()}
        for col in sector_returns.columns:
            sector_name = inv_map.get(col, col)
            r = sector_returns[col].dropna()
            if len(r) < self._lookback_long:
                continue
            mom_1m = float(r.iloc[-self._lookback_short:].sum())
            mom_3m = float(r.iloc[-self._lookback_medium:].sum())
            mom_12m = float(r.iloc[-self._lookback_long:].sum())
            score = 0.50 * mom_3m + 0.30 * (mom_12m / 2) + 0.20 * mom_1m
            scores[sector_name] = score
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def compute_relative_strength(self, sector_returns: Optional[pd.DataFrame] = None,
                                   benchmark_returns: Optional[pd.Series] = None) -> dict[str, float]:
        if sector_returns is None:
            return {}
        scores = {}
        inv_map = {v: k for k, v in SECTOR_ETFS.items()}
        for col in sector_returns.columns:
            sector_name = inv_map.get(col, col)
            r = sector_returns[col].dropna()
            if len(r) < 63:
                continue
            sector_cum = float((1 + r.iloc[-63:]).prod() - 1)
            bench_cum = 0.0
            if benchmark_returns is not None and len(benchmark_returns) >= 63:
                bench_cum = float((1 + benchmark_returns.iloc[-63:]).prod() - 1)
            scores[sector_name] = sector_cum - bench_cum
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_overweight_sectors(self, scores: dict[str, float], top_n: int = 3) -> list[str]:
        return [s[0] for s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def get_underweight_sectors(self, scores: dict[str, float], bottom_n: int = 3) -> list[str]:
        return [s[0] for s in sorted(scores.items(), key=lambda x: x[1])[:bottom_n]]


class QualityScorer:
    TIER_THRESHOLDS = {
        "A": {"sharpe": 2.0, "momentum": 0.15},
        "B": {"sharpe": 1.5, "momentum": 0.10},
        "C": {"sharpe": 1.0, "momentum": 0.05},
        "D": {"sharpe": 0.5, "momentum": 0.00},
        "E": {"sharpe": 0.0, "momentum": -0.05},
        "F": {"sharpe": -0.5, "momentum": -0.15},
    }

    def classify(self, sharpe: float, momentum: float) -> str:
        for tier, thresholds in self.TIER_THRESHOLDS.items():
            if sharpe >= thresholds["sharpe"] and momentum >= thresholds["momentum"]:
                return tier
        return "G"

    def score_universe(self, securities: list[Security]) -> list[Security]:
        for sec in securities:
            sec.quality_tier = self.classify(sec.sharpe_12m, sec.momentum_3m)
        return securities


class DailyUniverseScanner:
    def __init__(self, min_market_cap: float = 300e6, min_avg_volume: float = 500_000, min_price: float = 5.0):
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        self.min_price = min_price
        self._fallen_angel_detector = FallenAngelDetector()
        self._quality_scorer = QualityScorer()
        self._sector_rotation = SectorRotationEngine()

    def scan(self, universe: "UniverseEngine") -> dict:
        equities = universe.get_all()
        filtered = [s for s in equities if s.market_cap >= self.min_market_cap
                     and s.avg_volume >= self.min_avg_volume and s.price >= self.min_price]
        scored = self._quality_scorer.score_universe(filtered)
        fallen_angels = self._fallen_angel_detector.scan(scored)
        rv_tickers = set()
        for a, b in RV_PAIRS:
            rv_tickers.add(a)
            rv_tickers.add(b)
        rv_candidates = [s for s in scored if s.ticker in rv_tickers]
        for s in rv_candidates:
            s.is_rv_candidate = True
        sector_scores = self._sector_rotation.compute_sector_momentum()
        overweight = self._sector_rotation.get_overweight_sectors(sector_scores)
        underweight = self._sector_rotation.get_underweight_sectors(sector_scores)
        tier_dist = {}
        for s in scored:
            tier_dist[s.quality_tier] = tier_dist.get(s.quality_tier, 0) + 1
        cap_dist = {}
        for s in scored:
            tier = classify_market_cap(s.market_cap).value
            cap_dist[tier] = cap_dist.get(tier, 0) + 1
        return {
            "timestamp": datetime.now().isoformat(),
            "total_universe": len(equities), "filtered_count": len(filtered),
            "quality_scored": len(scored), "fallen_angels": len(fallen_angels),
            "fallen_angel_tickers": [s.ticker for s in fallen_angels],
            "rv_candidates": len(rv_candidates),
            "sector_momentum": sector_scores,
            "overweight_sectors": overweight, "underweight_sectors": underweight,
            "tier_distribution": tier_dist, "cap_distribution": cap_dist,
        }


class RVPairAnalyzer:
    def __init__(self, zscore_threshold: float = 2.0, lookback: int = 60):
        self.zscore_threshold = zscore_threshold
        self.lookback = lookback

    def analyze_pair(self, ticker_a: str, ticker_b: str,
                     returns: Optional[pd.DataFrame] = None) -> dict:
        if returns is None:
            try:
                start = (pd.Timestamp.now() - pd.Timedelta(days=252)).strftime("%Y-%m-%d")
                prices = get_adj_close([ticker_a, ticker_b], start=start)
                if prices.empty or ticker_a not in prices.columns or ticker_b not in prices.columns:
                    return {"pair": (ticker_a, ticker_b), "signal": "NO_DATA"}
                returns = prices.pct_change().dropna()
            except Exception:
                return {"pair": (ticker_a, ticker_b), "signal": "ERROR"}
        if ticker_a not in returns.columns or ticker_b not in returns.columns:
            return {"pair": (ticker_a, ticker_b), "signal": "NO_DATA"}
        spread = (returns[ticker_a] - returns[ticker_b]).cumsum()
        if len(spread) < self.lookback:
            return {"pair": (ticker_a, ticker_b), "signal": "INSUFFICIENT_DATA"}
        recent = spread.iloc[-self.lookback:]
        mean = float(recent.mean())
        std = float(recent.std())
        if std == 0:
            return {"pair": (ticker_a, ticker_b), "signal": "NO_VARIANCE", "zscore": 0}
        current = float(spread.iloc[-1])
        zscore = (current - mean) / std
        if zscore > self.zscore_threshold:
            signal = "RV_SHORT_A_LONG_B"
        elif zscore < -self.zscore_threshold:
            signal = "RV_LONG_A_SHORT_B"
        else:
            signal = "NEUTRAL"
        return {
            "pair": (ticker_a, ticker_b), "signal": signal,
            "zscore": round(zscore, 3), "spread_current": round(current, 4),
            "spread_mean": round(mean, 4), "spread_std": round(std, 4),
            "correlation": round(float(returns[ticker_a].corr(returns[ticker_b])), 3),
        }

    def scan_all_pairs(self, returns: Optional[pd.DataFrame] = None) -> list[dict]:
        return [self.analyze_pair(a, b, returns) for a, b in RV_PAIRS]

    def get_actionable_pairs(self, results: Optional[list[dict]] = None) -> list[dict]:
        if results is None:
            results = self.scan_all_pairs()
        return [r for r in results if r.get("signal", "").startswith("RV_")]


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Asset Universe — ~2,500+ equities from cross_asset_universe.py
# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias — old code referencing SP500_TOP_HOLDINGS still works
SP500_TOP_HOLDINGS = SP500_TICKERS

# Unified sector + industry maps from cross_asset_universe
KNOWN_SECTORS = SECTOR_MAP
KNOWN_INDUSTRY_GROUPS = INDUSTRY_GROUP_MAP

# RV pair mapping for quick lookup
RV_PAIR_MAP = {
    "GOOGL": "META", "META": "GOOGL",
    "XOM": "CVX", "CVX": "XOM",
    "AMD": "INTC", "INTC": "AMD",
    "JPM": "BAC", "BAC": "JPM",
    "V": "MA", "MA": "V",
    "KO": "PEP", "PEP": "KO",
    "HD": "LOW", "LOW": "HD",
    "UNH": "CI", "CI": "UNH",
    "BA": "LMT", "LMT": "BA",
    "CAT": "DE", "DE": "CAT",
    "AAPL": "MSFT", "MSFT": "AAPL",
    "PG": "CL", "CL": "PG",
}


class UniverseEngine:
    """Central universe management for Metadron Capital."""

    def __init__(self, load: bool = False):
        self._equities: list[Security] = []
        self._etf_map: dict[str, str] = dict(SECTOR_ETFS)
        self._loaded: bool = False
        self._scanner = DailyUniverseScanner()
        self._rv_analyzer = RVPairAnalyzer()
        self._gic_pooling = GICPoolingEngine()
        self._sector_rotation = SectorRotationEngine()
        self._quality_scorer = QualityScorer()
        if load:
            self.load_universe()

    def load_universe(self) -> int:
        """Load the full cross-asset universe.

        Strategy:
        1. Try dynamic loading via OpenBB FMP index.constituents() — PRIMARY
        2. Merge with static universe as FLOOR (catches any tickers FMP missed)
        3. Apply EXCLUDE_TICKERS (delisted/failed companies)
        4. Enrich each ticker with GICS sector/industry classification

        Dynamic constituents come from FMP's live index data. Static fallback
        ensures the universe is never empty even if FMP is unreachable.
        """
        # Step 1: Try dynamic OpenBB loading (primary — live FMP data)
        dynamic_tickers = self._try_openbb_constituents()

        # Step 2: Merge with static universe (floor/fallback)
        static_tickers = get_all_static_tickers()

        if dynamic_tickers:
            # Dynamic is primary — use it, add static as supplementary
            all_tickers = sorted(set(dynamic_tickers) | set(static_tickers))
        else:
            # No dynamic data — use static only
            all_tickers = sorted(set(static_tickers))

        logger.info(
            "Universe loading: %d dynamic + %d static → %d unique tickers",
            len(dynamic_tickers), len(static_tickers), len(all_tickers),
        )

        # Step 3: Build Security objects
        for ticker in all_tickers:
            sector = get_sector_for_ticker(ticker)
            sector_etf = SECTOR_ETFS.get(sector, "")
            rv_pair = RV_PAIR_MAP.get(ticker, "")
            ig = get_industry_group_for_ticker(ticker)
            sec = Security(
                ticker=ticker, name=ticker,
                security_type=SecurityType.EQUITY.value,
                sector=sector, gics_sector=sector,
                industry_group=ig[0], industry=ig[1], sub_industry=ig[2],
                options_eligible=True,
                sector_etf=sector_etf,
                rv_pair=rv_pair,
                is_rv_candidate=bool(rv_pair),
            )
            self._equities.append(sec)
        self._loaded = True
        logger.info(
            "Universe loaded: %d equities across %d sectors",
            len(self._equities), len(self.get_sector_counts()),
        )
        return len(self._equities)

    def _try_openbb_constituents(self) -> list[str]:
        """Try to fetch index constituents from OpenBB.

        Attempts SP500, SP400, SP600 via obb.index.constituents().
        Returns empty list if OpenBB is unavailable or API keys missing.
        """
        try:
            from .openbb_data import _obb, _openbb_available
            if not _openbb_available or _obb is None:
                return []
        except ImportError:
            return []

        tickers = set()
        # Try each major index
        for index_name, symbol in [
            ("S&P 500", "^GSPC"),
            ("S&P MidCap 400", "^MID"),
            ("S&P SmallCap 600", "^SML"),
        ]:
            try:
                result = _obb.index.constituents(symbol, provider="fmp")
                df = result.to_dataframe()
                if df is not None and not df.empty:
                    # Constituents typically have a 'symbol' column
                    sym_col = None
                    for col in ["symbol", "Symbol", "ticker", "Ticker"]:
                        if col in df.columns:
                            sym_col = col
                            break
                    if sym_col:
                        idx_tickers = df[sym_col].dropna().astype(str).tolist()
                        tickers.update(t.strip() for t in idx_tickers if t.strip())
                        logger.info(
                            "OpenBB %s: fetched %d constituents",
                            index_name, len(idx_tickers),
                        )
            except Exception as e:
                logger.debug("OpenBB %s constituents unavailable: %s", index_name, e)
                continue

        if tickers:
            logger.info("OpenBB dynamic universe: %d unique tickers", len(tickers))
        return sorted(tickers)

    def enrich_sectors_from_openbb(self, batch_size: int = 50):
        """Enrich unclassified tickers with GICS sector from OpenBB equity.profile().

        Only fetches for tickers that have no sector assigned from the static map.
        Batches requests to avoid rate limits.
        """
        unclassified = [s for s in self._equities if not s.sector]
        if not unclassified:
            return

        try:
            from .openbb_data import _obb, _openbb_available
            if not _openbb_available or _obb is None:
                logger.debug("OpenBB unavailable, skipping sector enrichment for %d tickers", len(unclassified))
                return
        except ImportError:
            return

        enriched = 0
        for i in range(0, len(unclassified), batch_size):
            batch = unclassified[i:i + batch_size]
            batch_tickers = [s.ticker for s in batch]
            try:
                result = _obb.equity.profile(batch_tickers, provider="fmp")
                df = result.to_dataframe()
                if df is None or df.empty:
                    continue
                # Build lookup from result
                for _, row in df.iterrows():
                    ticker = str(row.get("symbol", ""))
                    sector = str(row.get("sector", "") or "")
                    industry = str(row.get("industry", "") or "")
                    if not ticker or not sector:
                        continue
                    for sec in batch:
                        if sec.ticker == ticker:
                            sec.sector = sector
                            sec.gics_sector = sector
                            sec.sector_etf = SECTOR_ETFS.get(sector, "")
                            if industry:
                                sec.industry = industry
                            enriched += 1
                            break
            except Exception as e:
                logger.debug("OpenBB profile batch failed: %s", e)
                continue

        if enriched:
            logger.info("Enriched %d/%d unclassified tickers with sector data", enriched, len(unclassified))

    def load_fundamentals(self):
        tickers = [s.ticker for s in self._equities]
        try:
            data = get_bulk_fundamentals(tickers)
            if data.empty:
                return
            for sec in self._equities:
                if sec.ticker not in data.index:
                    continue
                try:
                    info = data.loc[sec.ticker].to_dict()
                except Exception:
                    continue
                if info:
                    sec.market_cap = info.get("market_cap", 0)
                    sec.pe_ratio = info.get("pe_ratio", 0)
                    sec.avg_volume = info.get("avg_volume", 0)
                    sec.price = info.get("price", 0)
                    sec.beta = info.get("beta", 1.0)
                    sec.roe = info.get("roe", 0)
                    sec.sector = info.get("sector", sec.sector)
                    sec.last_updated = datetime.now().isoformat()
        except Exception as e:
            logger.warning(f"Failed to load fundamentals: {e}")

    def enrich_with_returns(self, lookback_days: int = 365):
        """Compute Sharpe + momentum from price history.
        Fetches in batches of 350 to avoid API timeouts."""
        tickers = [s.ticker for s in self._equities]
        start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        batch_size = 350
        all_prices = pd.DataFrame()
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                prices = get_adj_close(batch, start=start)
                if not prices.empty:
                    all_prices = pd.concat([all_prices, prices], axis=1)
            except Exception as e:
                logger.warning(f"Batch {i//batch_size} fetch failed: {e}")

        if all_prices.empty:
            return
        returns = all_prices.pct_change().dropna()
        for sec in self._equities:
            if sec.ticker not in returns.columns:
                continue
            r = returns[sec.ticker].dropna()
            if len(r) < 60:
                continue
            sec.momentum_3m = float((1 + r.iloc[-63:]).prod() - 1) if len(r) >= 63 else 0
            sec.momentum_12m = float((1 + r.iloc[-252:]).prod() - 1) if len(r) >= 252 else 0
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252))
            sec.sharpe_12m = ann_ret / ann_vol if ann_vol > 0 else 0
            sec.last_updated = datetime.now().isoformat()

    def screen(self, sectors: Optional[list[str]] = None, min_market_cap: Optional[float] = None,
               max_market_cap: Optional[float] = None, quality_tiers: Optional[list[str]] = None,
               min_volume: Optional[float] = None, fallen_angels_only: bool = False) -> list[Security]:
        result = list(self._equities)
        if sectors:
            result = [s for s in result if s.sector in sectors]
        if min_market_cap is not None:
            result = [s for s in result if s.market_cap >= min_market_cap]
        if max_market_cap is not None:
            result = [s for s in result if s.market_cap <= max_market_cap]
        if quality_tiers:
            result = [s for s in result if s.quality_tier in quality_tiers]
        if min_volume:
            result = [s for s in result if s.avg_volume >= min_volume]
        if fallen_angels_only:
            result = [s for s in result if s.is_fallen_angel]
        return result

    def get_all(self) -> list[Security]:
        return list(self._equities)

    def get_by_ticker(self, ticker: str) -> Optional[Security]:
        for s in self._equities:
            if s.ticker == ticker:
                return s
        return None

    def get_by_sector(self, sector: str) -> list[Security]:
        return [s for s in self._equities if s.sector == sector]

    def size(self) -> int:
        return len(self._equities)

    def get_sector_counts(self) -> dict[str, int]:
        counts = {}
        for s in self._equities:
            counts[s.sector] = counts.get(s.sector, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def run_morning_scan(self) -> dict:
        return self._scanner.scan(self)

    def scan_rv_pairs(self) -> list[dict]:
        return self._rv_analyzer.scan_all_pairs()

    def get_sector_momentum(self) -> dict[str, float]:
        return self._sector_rotation.compute_sector_momentum()

    def get_sectors(self) -> list[str]:
        """Return list of unique sectors in the universe."""
        return list(set(s.sector for s in self._equities if s.sector))

    def get_tradeable(self) -> list[Security]:
        """Return only tradeable securities (stocks, equity ETFs, futures, options).

        These are the instruments that flow through:
        allocator → alpha optimizer → beta corridor → decision matrix → execution.
        """
        return [s for s in self._equities if s.is_tradeable]

    def get_macro_only(self) -> list[str]:
        """Return macro-only instrument tickers (bonds, commodities, volatility ETFs).

        These are analysed for regime detection and sector allocation via GICS
        but are NEVER sent to the execution pipeline.
        """
        return sorted(MACRO_ONLY_TICKERS)

    def get_tradeable_tickers(self) -> set[str]:
        """Return set of all tradeable ticker symbols."""
        equity_tickers = {s.ticker for s in self._equities if s.is_tradeable}
        return equity_tickers | TRADEABLE_ETFS

    def is_ticker_tradeable(self, ticker: str) -> bool:
        """Check if a specific ticker is execution-eligible."""
        if ticker in MACRO_ONLY_TICKERS:
            return False
        if ticker in TRADEABLE_ETFS:
            return True
        # Check if it's an equity in our universe
        sec = self.get_by_ticker(ticker)
        if sec:
            return sec.is_tradeable
        # Unknown tickers default to tradeable (individual stocks)
        return True

    def load(self) -> int:
        """Alias for load_universe() — used by execution pipeline."""
        if not self._loaded:
            return self.load_universe()
        return self.size()

    def summary(self) -> str:
        lines = ["=" * 60, "METADRON CAPITAL — UNIVERSE ENGINE", "=" * 60,
                  f"  Total Securities: {self.size()}", f"  Loaded: {self._loaded}",
                  f"  Sectors: {len(self.get_sector_counts())}", f"  RV Pairs: {len(RV_PAIRS)}",
                  f"  ETFs Tracked: {len(ALL_ETFS)}", ""]
        for sector, count in self.get_sector_counts().items():
            lines.append(f"  {sector:<35} {count:>4}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_ENGINE_INSTANCE: Optional[UniverseEngine] = None


def get_engine() -> UniverseEngine:
    """Get or create the singleton UniverseEngine instance."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = UniverseEngine(load=True)
    return _ENGINE_INSTANCE
