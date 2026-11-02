"""
Unified OpenBB Universe Module for Investment Platform.

Provides the full investable universe across ALL asset classes using OpenBB
as the SOLE data source. Supports GICS classification for all 11 sectors,
asset class enumeration, and standardized accessor functions.

Usage:
    from openbb_data import (
        get_full_universe, get_equity_universe, get_bond_universe,
        get_commodity_universe, get_fx_universe, get_crypto_universe,
        classify_by_gics, get_sector_constituents, get_historical,
        get_fundamentals, get_options_chain, AssetClass, GICSSector,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    """Supported asset classes in the unified universe."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FX = "fx"
    CRYPTO = "crypto"
    INDEX = "index"
    ETF = "etf"
    FUND = "fund"


class GICSSector(str, Enum):
    """GICS Level-1 sectors (all 11)."""

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


# Canonical GICS mapping: sector -> industry groups -> representative tickers
GICS_SECTOR_MAP: dict[GICSSector, dict[str, list[str]]] = {
    GICSSector.ENERGY: {
        "Oil, Gas & Consumable Fuels": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES"],
        "Energy Equipment & Services": ["HAL", "BKR", "FTI", "NOV", "CHX"],
    },
    GICSSector.MATERIALS: {
        "Chemicals": ["LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "CE", "EMN", "ALB"],
        "Metals & Mining": ["NEM", "FCX", "NUE", "STLD", "RS"],
        "Containers & Packaging": ["BLL", "PKG", "IP", "AVY", "SEE"],
    },
    GICSSector.INDUSTRIALS: {
        "Aerospace & Defense": ["RTX", "BA", "LMT", "NOC", "GD", "LHX", "TDG", "HWM", "TXT", "HII"],
        "Industrial Conglomerates": ["GE", "HON", "MMM", "ITW", "EMR"],
        "Machinery": ["CAT", "DE", "PH", "ETN", "ROK", "AME", "IR", "DOV", "OTIS", "CMI"],
        "Airlines": ["DAL", "UAL", "LUV", "ALK", "JBLU"],
        "Railroads": ["UNP", "CSX", "NSC"],
    },
    GICSSector.CONSUMER_DISCRETIONARY: {
        "Automobiles": ["TSLA", "GM", "F", "RIVN", "LCID"],
        "Hotels, Restaurants & Leisure": ["MCD", "SBUX", "CMG", "MAR", "HLT", "LVS", "WYNN", "MGM"],
        "Retail": ["AMZN", "HD", "LOW", "TJX", "ROST", "BBY", "DG", "DLTR", "ORLY", "AZO"],
        "Textiles, Apparel & Luxury": ["NKE", "LULU", "TPR", "RL", "VFC"],
    },
    GICSSector.CONSUMER_STAPLES: {
        "Food & Staples Retailing": ["WMT", "COST", "KR", "SYY", "ADM"],
        "Beverages": ["KO", "PEP", "STZ", "MNST", "BF.B"],
        "Food Products": ["MDLZ", "GIS", "K", "HSY", "HRL", "MKC", "SJM", "CAG", "CPB"],
        "Household Products": ["PG", "CL", "CLX", "CHD", "SPB"],
        "Tobacco": ["PM", "MO", "BTI"],
    },
    GICSSector.HEALTH_CARE: {
        "Pharmaceuticals": ["JNJ", "LLY", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "VTRS", "TAK"],
        "Biotechnology": ["REGN", "VRTX", "MRNA", "BIIB", "SGEN", "ALNY", "BMRN", "INCY"],
        "Health Care Equipment": ["ABT", "MDT", "SYK", "BSX", "EW", "ISRG", "ZBH", "BAX"],
        "Health Care Providers": ["UNH", "ELV", "CI", "HUM", "CNC", "MOH", "HCA", "THC"],
    },
    GICSSector.FINANCIALS: {
        "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
        "Insurance": ["BRK.B", "AIG", "MET", "PRU", "ALL", "TRV", "AFL", "PGR", "CB", "HIG"],
        "Capital Markets": ["BLK", "ICE", "CME", "SPGI", "MCO", "MSCI", "NDAQ", "FDS"],
        "Consumer Finance": ["V", "MA", "AXP", "COF", "DFS", "SYF"],
    },
    GICSSector.INFORMATION_TECHNOLOGY: {
        "Software": ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "SNPS", "CDNS", "PANW", "FTNT"],
        "Semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "TXN", "QCOM", "MU", "AMAT", "LRCX", "KLAC"],
        "Hardware": ["AAPL", "CSCO", "IBM", "HPQ", "HPE", "DELL", "NTAP", "WDC", "STX"],
        "IT Services": ["ACN", "FIS", "FISV", "GPN", "IT", "CTSH", "WIT"],
    },
    GICSSector.COMMUNICATION_SERVICES: {
        "Media & Entertainment": ["GOOG", "GOOGL", "META", "DIS", "NFLX", "CMCSA", "WBD", "PARA", "FOX"],
        "Telecom Services": ["T", "VZ", "TMUS"],
        "Interactive Media": ["SNAP", "PINS", "MTCH", "ZG", "YELP"],
    },
    GICSSector.UTILITIES: {
        "Electric Utilities": ["NEE", "SO", "DUK", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED"],
        "Gas Utilities": ["NI", "ATO", "OGS", "SWX"],
        "Multi-Utilities": ["WEC", "ES", "ETR", "CMS", "DTE", "FE", "PPL", "AEE", "EVRG"],
        "Water Utilities": ["AWK", "WTR", "CWT", "SJW"],
    },
    GICSSector.REAL_ESTATE: {
        "Equity REITs": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
        "Mortgage REITs": ["AGNC", "NLY", "STWD", "BXMT"],
        "Real Estate Services": ["CBRE", "JLL", "CSGP", "RKT"],
    },
}

# Representative bond universe (corporate and sovereign)
BOND_UNIVERSE: dict[str, dict] = {
    # Investment grade corporate ETFs / benchmarks
    "LQD": {"name": "iShares iBoxx IG Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "ig_corporate"},
    "VCIT": {"name": "Vanguard Intermediate Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "ig_corporate"},
    "VCLT": {"name": "Vanguard Long-Term Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "ig_corporate"},
    "IGIB": {"name": "iShares 5-10Y IG Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "ig_corporate"},
    # High yield
    "HYG": {"name": "iShares iBoxx HY Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "hy_corporate"},
    "JNK": {"name": "SPDR Bloomberg HY Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "hy_corporate"},
    "SHYG": {"name": "iShares 0-5Y HY Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "hy_corporate"},
    "USHY": {"name": "iShares Broad USD HY Corp Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "hy_corporate"},
    # Treasuries
    "TLT": {"name": "iShares 20+ Year Treasury", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "treasury"},
    "IEF": {"name": "iShares 7-10 Year Treasury", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "treasury"},
    "SHY": {"name": "iShares 1-3 Year Treasury", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "treasury"},
    "TIP": {"name": "iShares TIPS Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "tips"},
    "BIL": {"name": "SPDR Bloomberg 1-3M T-Bill", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "tbill"},
    # Emerging market debt
    "EMB": {"name": "iShares JP Morgan EM Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "em_debt"},
    "PCY": {"name": "Invesco EM Sovereign Debt", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "em_debt"},
    # Municipals
    "MUB": {"name": "iShares National Muni Bond", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "municipal"},
    "HYD": {"name": "VanEck HY Muni", "asset_class": AssetClass.FIXED_INCOME, "sub_type": "municipal"},
}

# Commodity universe
COMMODITY_UNIVERSE: dict[str, dict] = {
    "GC=F": {"name": "Gold Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "precious_metals"},
    "SI=F": {"name": "Silver Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "precious_metals"},
    "PL=F": {"name": "Platinum Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "precious_metals"},
    "CL=F": {"name": "WTI Crude Oil", "asset_class": AssetClass.COMMODITY, "sub_type": "energy"},
    "BZ=F": {"name": "Brent Crude Oil", "asset_class": AssetClass.COMMODITY, "sub_type": "energy"},
    "NG=F": {"name": "Natural Gas", "asset_class": AssetClass.COMMODITY, "sub_type": "energy"},
    "RB=F": {"name": "RBOB Gasoline", "asset_class": AssetClass.COMMODITY, "sub_type": "energy"},
    "HO=F": {"name": "Heating Oil", "asset_class": AssetClass.COMMODITY, "sub_type": "energy"},
    "ZC=F": {"name": "Corn Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "agriculture"},
    "ZW=F": {"name": "Wheat Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "agriculture"},
    "ZS=F": {"name": "Soybean Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "agriculture"},
    "KC=F": {"name": "Coffee Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "agriculture"},
    "CT=F": {"name": "Cotton Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "agriculture"},
    "HG=F": {"name": "Copper Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "industrial_metals"},
    "ALI=F": {"name": "Aluminum Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "industrial_metals"},
    "LE=F": {"name": "Live Cattle", "asset_class": AssetClass.COMMODITY, "sub_type": "livestock"},
    "LBS=F": {"name": "Lumber Futures", "asset_class": AssetClass.COMMODITY, "sub_type": "forest"},
}

# FX pairs
FX_UNIVERSE: dict[str, dict] = {
    "EURUSD=X": {"name": "EUR/USD", "asset_class": AssetClass.FX, "sub_type": "major"},
    "GBPUSD=X": {"name": "GBP/USD", "asset_class": AssetClass.FX, "sub_type": "major"},
    "USDJPY=X": {"name": "USD/JPY", "asset_class": AssetClass.FX, "sub_type": "major"},
    "USDCHF=X": {"name": "USD/CHF", "asset_class": AssetClass.FX, "sub_type": "major"},
    "AUDUSD=X": {"name": "AUD/USD", "asset_class": AssetClass.FX, "sub_type": "major"},
    "USDCAD=X": {"name": "USD/CAD", "asset_class": AssetClass.FX, "sub_type": "major"},
    "NZDUSD=X": {"name": "NZD/USD", "asset_class": AssetClass.FX, "sub_type": "major"},
    "EURGBP=X": {"name": "EUR/GBP", "asset_class": AssetClass.FX, "sub_type": "cross"},
    "EURJPY=X": {"name": "EUR/JPY", "asset_class": AssetClass.FX, "sub_type": "cross"},
    "GBPJPY=X": {"name": "GBP/JPY", "asset_class": AssetClass.FX, "sub_type": "cross"},
    "USDMXN=X": {"name": "USD/MXN", "asset_class": AssetClass.FX, "sub_type": "em"},
    "USDBRL=X": {"name": "USD/BRL", "asset_class": AssetClass.FX, "sub_type": "em"},
    "USDZAR=X": {"name": "USD/ZAR", "asset_class": AssetClass.FX, "sub_type": "em"},
    "USDTRY=X": {"name": "USD/TRY", "asset_class": AssetClass.FX, "sub_type": "em"},
    "USDCNH=X": {"name": "USD/CNH", "asset_class": AssetClass.FX, "sub_type": "em"},
    "USDINR=X": {"name": "USD/INR", "asset_class": AssetClass.FX, "sub_type": "em"},
}

# Crypto universe
CRYPTO_UNIVERSE: dict[str, dict] = {
    "BTC-USD": {"name": "Bitcoin", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "ETH-USD": {"name": "Ethereum", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "SOL-USD": {"name": "Solana", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "BNB-USD": {"name": "Binance Coin", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "ADA-USD": {"name": "Cardano", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "XRP-USD": {"name": "XRP", "asset_class": AssetClass.CRYPTO, "sub_type": "payments"},
    "DOGE-USD": {"name": "Dogecoin", "asset_class": AssetClass.CRYPTO, "sub_type": "meme"},
    "DOT-USD": {"name": "Polkadot", "asset_class": AssetClass.CRYPTO, "sub_type": "layer0"},
    "AVAX-USD": {"name": "Avalanche", "asset_class": AssetClass.CRYPTO, "sub_type": "layer1"},
    "LINK-USD": {"name": "Chainlink", "asset_class": AssetClass.CRYPTO, "sub_type": "oracle"},
    "MATIC-USD": {"name": "Polygon", "asset_class": AssetClass.CRYPTO, "sub_type": "layer2"},
    "UNI-USD": {"name": "Uniswap", "asset_class": AssetClass.CRYPTO, "sub_type": "defi"},
    "AAVE-USD": {"name": "Aave", "asset_class": AssetClass.CRYPTO, "sub_type": "defi"},
}

# Index universe
INDEX_UNIVERSE: dict[str, dict] = {
    "^GSPC": {"name": "S&P 500", "asset_class": AssetClass.INDEX, "sub_type": "us_large"},
    "^DJI": {"name": "Dow Jones Industrial Average", "asset_class": AssetClass.INDEX, "sub_type": "us_large"},
    "^IXIC": {"name": "NASDAQ Composite", "asset_class": AssetClass.INDEX, "sub_type": "us_tech"},
    "^RUT": {"name": "Russell 2000", "asset_class": AssetClass.INDEX, "sub_type": "us_small"},
    "^VIX": {"name": "CBOE Volatility Index", "asset_class": AssetClass.INDEX, "sub_type": "volatility"},
    "^FTSE": {"name": "FTSE 100", "asset_class": AssetClass.INDEX, "sub_type": "uk"},
    "^GDAXI": {"name": "DAX", "asset_class": AssetClass.INDEX, "sub_type": "germany"},
    "^N225": {"name": "Nikkei 225", "asset_class": AssetClass.INDEX, "sub_type": "japan"},
    "^HSI": {"name": "Hang Seng", "asset_class": AssetClass.INDEX, "sub_type": "hong_kong"},
    "000001.SS": {"name": "Shanghai Composite", "asset_class": AssetClass.INDEX, "sub_type": "china"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Security:
    """Represents a single security in the universe."""

    ticker: str
    name: str
    asset_class: AssetClass
    sub_type: str = ""
    gics_sector: Optional[GICSSector] = None
    gics_industry_group: str = ""
    market_cap: Optional[float] = None
    currency: str = "USD"
    exchange: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class UniverseSnapshot:
    """Point-in-time snapshot of the full universe."""

    timestamp: datetime
    securities: list[Security]
    equity_count: int = 0
    bond_count: int = 0
    commodity_count: int = 0
    fx_count: int = 0
    crypto_count: int = 0
    index_count: int = 0

    def __post_init__(self) -> None:
        by_class: dict[AssetClass, int] = {}
        for sec in self.securities:
            by_class[sec.asset_class] = by_class.get(sec.asset_class, 0) + 1
        self.equity_count = by_class.get(AssetClass.EQUITY, 0)
        self.bond_count = by_class.get(AssetClass.FIXED_INCOME, 0)
        self.commodity_count = by_class.get(AssetClass.COMMODITY, 0)
        self.fx_count = by_class.get(AssetClass.FX, 0)
        self.crypto_count = by_class.get(AssetClass.CRYPTO, 0)
        self.index_count = by_class.get(AssetClass.INDEX, 0)


# ---------------------------------------------------------------------------
# OpenBB client wrapper (singleton)
# ---------------------------------------------------------------------------


class _OpenBBClient:
    """Lazy-initialized OpenBB SDK wrapper."""

    _instance: Optional["_OpenBBClient"] = None
    _obb = None

    def __new__(cls) -> "_OpenBBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def obb(self):
        if self._obb is None:
            try:
                from openbb import obb
                self._obb = obb
                logger.info("OpenBB SDK initialized successfully")
            except ImportError:
                raise ImportError(
                    "OpenBB SDK is required. Install via: pip install openbb"
                )
        return self._obb


def _get_obb():
    """Get the OpenBB client instance."""
    return _OpenBBClient().obb


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Execute a function with exponential backoff on failure."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------


def _build_equity_securities() -> list[Security]:
    """Build equity securities from GICS sector map."""
    securities: list[Security] = []
    seen: set[str] = set()
    for sector, industry_groups in GICS_SECTOR_MAP.items():
        for ig_name, tickers in industry_groups.items():
            for ticker in tickers:
                if ticker in seen:
                    continue
                seen.add(ticker)
                securities.append(
                    Security(
                        ticker=ticker,
                        name=ticker,
                        asset_class=AssetClass.EQUITY,
                        sub_type="common_stock",
                        gics_sector=sector,
                        gics_industry_group=ig_name,
                    )
                )
    return securities


def _build_bond_securities() -> list[Security]:
    """Build fixed-income securities from bond universe."""
    return [
        Security(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            sub_type=info["sub_type"],
        )
        for ticker, info in BOND_UNIVERSE.items()
    ]


def _build_commodity_securities() -> list[Security]:
    """Build commodity securities."""
    return [
        Security(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            sub_type=info["sub_type"],
        )
        for ticker, info in COMMODITY_UNIVERSE.items()
    ]


def _build_fx_securities() -> list[Security]:
    """Build FX pair securities."""
    return [
        Security(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            sub_type=info["sub_type"],
        )
        for ticker, info in FX_UNIVERSE.items()
    ]


def _build_crypto_securities() -> list[Security]:
    """Build crypto securities."""
    return [
        Security(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            sub_type=info["sub_type"],
        )
        for ticker, info in CRYPTO_UNIVERSE.items()
    ]


def _build_index_securities() -> list[Security]:
    """Build index securities."""
    return [
        Security(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            sub_type=info["sub_type"],
        )
        for ticker, info in INDEX_UNIVERSE.items()
    ]


# ---------------------------------------------------------------------------
# Public API: Universe accessors
# ---------------------------------------------------------------------------


def get_full_universe() -> UniverseSnapshot:
    """
    Get the full investable universe across all asset classes.

    Returns
    -------
    UniverseSnapshot
        Complete universe snapshot with all securities categorized.
    """
    securities: list[Security] = []
    securities.extend(_build_equity_securities())
    securities.extend(_build_bond_securities())
    securities.extend(_build_commodity_securities())
    securities.extend(_build_fx_securities())
    securities.extend(_build_crypto_securities())
    securities.extend(_build_index_securities())

    snapshot = UniverseSnapshot(
        timestamp=datetime.utcnow(),
        securities=securities,
    )
    logger.info(
        "Full universe: %d securities (E:%d B:%d C:%d FX:%d Crypto:%d Idx:%d)",
        len(securities),
        snapshot.equity_count,
        snapshot.bond_count,
        snapshot.commodity_count,
        snapshot.fx_count,
        snapshot.crypto_count,
        snapshot.index_count,
    )
    return snapshot


def get_equity_universe() -> list[Security]:
    """Get all equity securities with GICS classification."""
    return _build_equity_securities()


def get_bond_universe() -> list[Security]:
    """Get all fixed-income securities."""
    return _build_bond_securities()


def get_commodity_universe() -> list[Security]:
    """Get all commodity securities."""
    return _build_commodity_securities()


def get_fx_universe() -> list[Security]:
    """Get all FX pair securities."""
    return _build_fx_securities()


def get_crypto_universe() -> list[Security]:
    """Get all crypto securities."""
    return _build_crypto_securities()


def get_index_universe() -> list[Security]:
    """Get all index securities."""
    return _build_index_securities()


# ---------------------------------------------------------------------------
# Public API: GICS classification
# ---------------------------------------------------------------------------


def classify_by_gics(ticker: str) -> Optional[tuple[GICSSector, str]]:
    """
    Classify a ticker by its GICS sector and industry group.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.

    Returns
    -------
    tuple[GICSSector, str] or None
        (sector, industry_group) if found, else None.
    """
    for sector, industry_groups in GICS_SECTOR_MAP.items():
        for ig_name, tickers in industry_groups.items():
            if ticker.upper() in [t.upper() for t in tickers]:
                return sector, ig_name
    return None


def get_sector_constituents(sector: GICSSector) -> list[Security]:
    """
    Get all securities belonging to a GICS sector.

    Parameters
    ----------
    sector : GICSSector
        Target GICS sector.

    Returns
    -------
    list[Security]
        Securities in that sector.
    """
    securities: list[Security] = []
    if sector not in GICS_SECTOR_MAP:
        return securities
    for ig_name, tickers in GICS_SECTOR_MAP[sector].items():
        for ticker in tickers:
            securities.append(
                Security(
                    ticker=ticker,
                    name=ticker,
                    asset_class=AssetClass.EQUITY,
                    sub_type="common_stock",
                    gics_sector=sector,
                    gics_industry_group=ig_name,
                )
            )
    return securities


def get_all_sectors() -> list[GICSSector]:
    """Return all 11 GICS sectors."""
    return list(GICSSector)


def get_industry_groups(sector: GICSSector) -> list[str]:
    """Get all industry groups within a GICS sector."""
    if sector not in GICS_SECTOR_MAP:
        return []
    return list(GICS_SECTOR_MAP[sector].keys())


def get_tickers_by_asset_class(asset_class: AssetClass) -> list[str]:
    """Return all tickers for a given asset class."""
    builders = {
        AssetClass.EQUITY: _build_equity_securities,
        AssetClass.FIXED_INCOME: _build_bond_securities,
        AssetClass.COMMODITY: _build_commodity_securities,
        AssetClass.FX: _build_fx_securities,
        AssetClass.CRYPTO: _build_crypto_securities,
        AssetClass.INDEX: _build_index_securities,
    }
    builder = builders.get(asset_class)
    if builder is None:
        return []
    return [s.ticker for s in builder()]


# ---------------------------------------------------------------------------
# Public API: Data fetching (OpenBB as SOLE source)
# ---------------------------------------------------------------------------


def get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data via OpenBB.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str, optional
        Start date YYYY-MM-DD. Defaults to 1 year ago.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    interval : str
        Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with normalised column names.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    obb = _get_obb()

    def _fetch():
        result = obb.equity.price.historical(
            symbol=symbol,
            start_date=start,
            end_date=end,
            provider="yfinance",
        )
        return result.to_dataframe()

    df = _retry_with_backoff(_fetch)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    logger.info("Fetched %d rows for %s", len(df), symbol)
    return _normalise(df)


def get_fundamentals(symbol: str) -> dict:
    """
    Fetch fundamental data for a symbol via OpenBB.

    Returns a dict with keys: income_statement, balance_sheet, cash_flow,
    ratios, profile.
    """
    obb = _get_obb()
    result: dict = {}

    try:
        income = obb.equity.fundamental.income(symbol=symbol, provider="yfinance")
        result["income_statement"] = income.to_dataframe()
    except Exception as exc:
        logger.warning("Income statement fetch failed for %s: %s", symbol, exc)
        result["income_statement"] = pd.DataFrame()

    try:
        balance = obb.equity.fundamental.balance(symbol=symbol, provider="yfinance")
        result["balance_sheet"] = balance.to_dataframe()
    except Exception as exc:
        logger.warning("Balance sheet fetch failed for %s: %s", symbol, exc)
        result["balance_sheet"] = pd.DataFrame()

    try:
        cash = obb.equity.fundamental.cash(symbol=symbol, provider="yfinance")
        result["cash_flow"] = cash.to_dataframe()
    except Exception as exc:
        logger.warning("Cash flow fetch failed for %s: %s", symbol, exc)
        result["cash_flow"] = pd.DataFrame()

    try:
        ratios = obb.equity.fundamental.ratios(symbol=symbol, provider="yfinance")
        result["ratios"] = ratios.to_dataframe()
    except Exception as exc:
        logger.warning("Ratios fetch failed for %s: %s", symbol, exc)
        result["ratios"] = pd.DataFrame()

    try:
        profile = obb.equity.profile(symbol=symbol, provider="yfinance")
        result["profile"] = profile.to_dataframe()
    except Exception as exc:
        logger.warning("Profile fetch failed for %s: %s", symbol, exc)
        result["profile"] = pd.DataFrame()

    return result


def get_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch options chain for a symbol via OpenBB.

    Parameters
    ----------
    symbol : str
        Underlying ticker.
    expiration : str, optional
        Expiration date YYYY-MM-DD. If None, fetches nearest expiry.

    Returns
    -------
    pd.DataFrame
        Options chain with calls and puts.
    """
    obb = _get_obb()

    def _fetch():
        kwargs = {"symbol": symbol, "provider": "cboe"}
        if expiration:
            kwargs["expiration"] = expiration
        result = obb.derivatives.options.chains(**kwargs)
        return result.to_dataframe()

    return _retry_with_backoff(_fetch)


def get_treasury_rates() -> pd.DataFrame:
    """Fetch current US Treasury rates across the curve via OpenBB."""
    obb = _get_obb()

    def _fetch():
        result = obb.fixedincome.rate.treasury(provider="federal_reserve")
        return result.to_dataframe()

    return _retry_with_backoff(_fetch)


def get_economic_indicators(
    indicator: str = "GDP",
    country: str = "united_states",
) -> pd.DataFrame:
    """Fetch economic indicator data via OpenBB."""
    obb = _get_obb()

    def _fetch():
        if indicator.upper() == "GDP":
            result = obb.economy.gdp.nominal(provider="oecd")
        elif indicator.upper() == "CPI":
            result = obb.economy.cpi(provider="fred", country=country)
        elif indicator.upper() == "UNEMPLOYMENT":
            result = obb.economy.unemployment(provider="oecd", country=country)
        else:
            result = obb.economy.indicators(
                symbol=indicator, provider="econdb", country=country,
            )
        return result.to_dataframe()

    return _retry_with_backoff(_fetch)


def get_news(
    symbol: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Fetch financial news via OpenBB."""
    obb = _get_obb()

    def _fetch():
        if symbol:
            result = obb.news.company(symbol=symbol, limit=limit, provider="benzinga")
        else:
            result = obb.news.world(limit=limit, provider="benzinga")
        return result.to_dataframe()

    return _retry_with_backoff(_fetch)


def get_etf_holdings(symbol: str) -> pd.DataFrame:
    """Fetch ETF holdings via OpenBB."""
    obb = _get_obb()

    def _fetch():
        result = obb.etf.holdings(symbol=symbol, provider="yfinance")
        return result.to_dataframe()

    return _retry_with_backoff(_fetch)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to title-case standard."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "date":
            rename_map[col] = "Date"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower in ("close", "adj close", "adj_close"):
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"
    return df.rename(columns=rename_map)


def universe_summary() -> str:
    """Return a human-readable summary of the universe."""
    snapshot = get_full_universe()
    lines = [
        f"Universe Snapshot ({snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC)",
        f"  Total securities: {len(snapshot.securities)}",
        f"  Equities:         {snapshot.equity_count}",
        f"  Fixed Income:     {snapshot.bond_count}",
        f"  Commodities:      {snapshot.commodity_count}",
        f"  FX Pairs:         {snapshot.fx_count}",
        f"  Crypto:           {snapshot.crypto_count}",
        f"  Indices:          {snapshot.index_count}",
        "",
        "GICS Sector Breakdown:",
    ]
    for sector in GICSSector:
        constituents = get_sector_constituents(sector)
        lines.append(f"  {sector.value}: {len(constituents)} securities")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(universe_summary())
