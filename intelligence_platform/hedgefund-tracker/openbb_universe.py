"""
Unified OpenBB Universe Module for Hedge Fund Tracker.

Provides access to the FULL investment universe using OpenBB as the SOLE
data source. Designed for hedge fund position tracking, P&L reporting,
and opportunity scanning across all asset classes.

Asset Classes:
    EQUITY, FIXED_INCOME, COMMODITY, CRYPTO, FX, ETF, INDEX

Mathematical Foundation:
    Daily P&L:
        PnL_t = sum_i(position_i * (P_{i,t} - P_{i,t-1}))
        PnL_pct = PnL_t / NAV_{t-1}

    Attribution (Brinson model):
        Selection = sum_i(w_{p,i} * (r_{p,i} - r_{b,i}))
        Allocation = sum_i((w_{p,i} - w_{b,i}) * (r_{b,i} - r_b))
        Interaction = sum_i((w_{p,i} - w_{b,i}) * (r_{p,i} - r_{b,i}))

    Alpha:
        alpha = R_p - beta * R_m
        where beta = Cov(R_p, R_m) / Var(R_m)

Usage:
    from openbb_universe import get_full_universe, get_historical, AssetClass
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset Class Enum
# ---------------------------------------------------------------------------

class AssetClass(Enum):
    """Enumeration of all supported asset classes."""
    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    CRYPTO = "CRYPTO"
    FX = "FX"
    ETF = "ETF"
    INDEX = "INDEX"


# ---------------------------------------------------------------------------
# GICS Classification (Complete Hierarchy)
# ---------------------------------------------------------------------------

GICS_CLASSIFICATION: dict[str, dict[str, dict[str, list[str]]]] = {
    "10 - Energy": {
        "1010 - Energy": {
            "101010 - Energy Equipment & Services": [
                "10101010 - Oil & Gas Drilling",
                "10101020 - Oil & Gas Equipment & Services",
            ],
            "101020 - Oil, Gas & Consumable Fuels": [
                "10102010 - Integrated Oil & Gas",
                "10102020 - Oil & Gas Exploration & Production",
                "10102030 - Oil & Gas Refining & Marketing",
                "10102040 - Oil & Gas Storage & Transportation",
                "10102050 - Coal & Consumable Fuels",
            ],
        },
    },
    "15 - Materials": {
        "1510 - Materials": {
            "151010 - Chemicals": [
                "15101010 - Commodity Chemicals",
                "15101020 - Diversified Chemicals",
                "15101030 - Fertilizers & Agricultural Chemicals",
                "15101040 - Industrial Gases",
                "15101050 - Specialty Chemicals",
            ],
            "151040 - Metals & Mining": [
                "15104010 - Aluminum",
                "15104020 - Diversified Metals & Mining",
                "15104030 - Gold",
                "15104050 - Steel",
            ],
        },
    },
    "20 - Industrials": {
        "2010 - Capital Goods": {
            "201010 - Aerospace & Defense": ["20101010 - Aerospace & Defense"],
            "201060 - Machinery": ["20106020 - Industrial Machinery & Supplies & Components"],
        },
        "2030 - Transportation": {
            "203040 - Ground Transportation": ["20304010 - Rail Transportation"],
        },
    },
    "25 - Consumer Discretionary": {
        "2510 - Automobiles & Components": {
            "251020 - Automobiles": ["25102010 - Automobile Manufacturers"],
        },
        "2550 - Retailing": {
            "255030 - Broadline Retail": ["25503010 - Broadline Retail"],
        },
    },
    "30 - Consumer Staples": {
        "3020 - Food, Beverage & Tobacco": {
            "302010 - Beverages": ["30201030 - Soft Drinks & Non-alcoholic Beverages"],
            "302020 - Food Products": ["30202030 - Packaged Foods & Meats"],
        },
    },
    "35 - Health Care": {
        "3520 - Pharmaceuticals, Biotechnology & Life Sciences": {
            "352010 - Biotechnology": ["35201010 - Biotechnology"],
            "352020 - Pharmaceuticals": ["35202010 - Pharmaceuticals"],
        },
    },
    "40 - Financials": {
        "4010 - Banks": {
            "401010 - Banks": ["40101010 - Diversified Banks", "40101015 - Regional Banks"],
        },
        "4020 - Financial Services": {
            "402030 - Capital Markets": [
                "40203010 - Asset Management & Custody Banks",
                "40203020 - Investment Banking & Brokerage",
            ],
        },
    },
    "45 - Information Technology": {
        "4510 - Software & Services": {
            "451030 - Software": ["45103010 - Application Software", "45103020 - Systems Software"],
        },
        "4530 - Semiconductors & Semiconductor Equipment": {
            "453010 - Semiconductors & Semiconductor Equipment": ["45301020 - Semiconductors"],
        },
    },
    "50 - Communication Services": {
        "5020 - Media & Entertainment": {
            "502030 - Interactive Media & Services": ["50203010 - Interactive Media & Services"],
        },
    },
    "55 - Utilities": {
        "5510 - Utilities": {
            "551010 - Electric Utilities": ["55101010 - Electric Utilities"],
        },
    },
    "60 - Real Estate": {
        "6010 - Equity Real Estate Investment Trusts": {
            "601080 - Specialized REITs": ["60108010 - Data Center REITs"],
        },
    },
}

# ---------------------------------------------------------------------------
# Hardcoded Constituent Lists
# ---------------------------------------------------------------------------

SP500_TOP_50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "CVX",
    "MRK", "ABBV", "LLY", "COST", "PEP", "KO", "ADBE", "WMT", "MCD",
    "CSCO", "CRM", "ACN", "ABT", "TMO", "DHR", "LIN", "CMCSA", "NKE",
    "VZ", "NEE", "TXN", "PM", "INTC", "RTX", "HON", "UNP", "LOW",
    "QCOM", "ORCL", "AMGN", "IBM", "BA",
]

EQUITY_GICS_MAP: dict[str, str] = {
    "AAPL": "45 - Information Technology", "MSFT": "45 - Information Technology",
    "AMZN": "25 - Consumer Discretionary", "NVDA": "45 - Information Technology",
    "GOOGL": "50 - Communication Services", "META": "50 - Communication Services",
    "BRK-B": "40 - Financials", "TSLA": "25 - Consumer Discretionary",
    "UNH": "35 - Health Care", "XOM": "10 - Energy",
    "JNJ": "35 - Health Care", "JPM": "40 - Financials",
    "V": "40 - Financials", "PG": "30 - Consumer Staples",
    "MA": "40 - Financials", "HD": "25 - Consumer Discretionary",
    "AVGO": "45 - Information Technology", "CVX": "10 - Energy",
    "MRK": "35 - Health Care", "ABBV": "35 - Health Care",
    "LLY": "35 - Health Care", "COST": "30 - Consumer Staples",
    "PEP": "30 - Consumer Staples", "KO": "30 - Consumer Staples",
    "ADBE": "45 - Information Technology", "WMT": "30 - Consumer Staples",
    "MCD": "25 - Consumer Discretionary", "CSCO": "45 - Information Technology",
    "CRM": "45 - Information Technology", "ACN": "45 - Information Technology",
    "ABT": "35 - Health Care", "TMO": "35 - Health Care",
    "DHR": "35 - Health Care", "LIN": "15 - Materials",
    "CMCSA": "50 - Communication Services", "NKE": "25 - Consumer Discretionary",
    "VZ": "50 - Communication Services", "NEE": "55 - Utilities",
    "TXN": "45 - Information Technology", "PM": "30 - Consumer Staples",
    "INTC": "45 - Information Technology", "RTX": "20 - Industrials",
    "HON": "20 - Industrials", "UNP": "20 - Industrials",
    "LOW": "25 - Consumer Discretionary", "QCOM": "45 - Information Technology",
    "ORCL": "45 - Information Technology", "AMGN": "35 - Health Care",
    "IBM": "45 - Information Technology", "BA": "20 - Industrials",
}

MAJOR_BONDS = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "BND",
    "MUB", "EMB", "BNDX", "GOVT",
]

MAJOR_COMMODITIES = [
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F", "LE=F",
]

CRYPTO_TOP_20 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "FIL-USD",
    "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD", "MKR-USD",
]

FX_MAJORS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURCHF=X", "USDMXN=X", "USDZAR=X", "USDTRY=X",
    "USDCNY=X", "USDINR=X", "USDBRL=X",
]

MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM",
    "VWO", "GLD", "SLV", "USO", "XLF", "XLK", "XLE", "XLV",
    "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY",
    "ARKK", "VIG", "SCHD", "VYM", "DVY",
]

GLOBAL_INDICES = [
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    "^FCHI", "^N225", "^HSI", "000001.SS", "^STOXX50E", "^BVSP",
    "^AXJO", "^KS11", "^NSEI", "^GSPTSE",
]


# ---------------------------------------------------------------------------
# Symbol -> Asset Class Detection
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = {"-USD", "-USDT", "-BTC", "-ETH"}
_BOND_SYMBOLS = set(MAJOR_BONDS)
_COMMODITY_SYMBOLS = set(MAJOR_COMMODITIES)
_ETF_SYMBOLS = set(MAJOR_ETFS)
_INDEX_SYMBOLS = set(GLOBAL_INDICES)
_CRYPTO_SYMBOLS = set(CRYPTO_TOP_20)


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Detect the asset class for a given symbol.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    AssetClass
    """
    if symbol in _INDEX_SYMBOLS or symbol.startswith("^"):
        return AssetClass.INDEX
    if any(symbol.upper().endswith(s) for s in _CRYPTO_SUFFIXES) or symbol in _CRYPTO_SYMBOLS:
        return AssetClass.CRYPTO
    if symbol.endswith("=X") and symbol not in _COMMODITY_SYMBOLS:
        return AssetClass.FX
    if symbol.endswith("=F") or symbol in _COMMODITY_SYMBOLS:
        return AssetClass.COMMODITY
    if symbol in _BOND_SYMBOLS:
        return AssetClass.FIXED_INCOME
    if symbol in _ETF_SYMBOLS:
        return AssetClass.ETF
    return AssetClass.EQUITY


# ---------------------------------------------------------------------------
# Universe Functions
# ---------------------------------------------------------------------------

def get_full_universe() -> dict[AssetClass, list[str]]:
    """Return the complete investment universe organized by asset class."""
    return {
        AssetClass.EQUITY: SP500_TOP_50.copy(),
        AssetClass.FIXED_INCOME: MAJOR_BONDS.copy(),
        AssetClass.COMMODITY: MAJOR_COMMODITIES.copy(),
        AssetClass.CRYPTO: CRYPTO_TOP_20.copy(),
        AssetClass.FX: FX_MAJORS.copy(),
        AssetClass.ETF: MAJOR_ETFS.copy(),
        AssetClass.INDEX: GLOBAL_INDICES.copy(),
    }


def get_equity_universe() -> list[str]:
    return SP500_TOP_50.copy()

def get_bond_universe() -> list[str]:
    return MAJOR_BONDS.copy()

def get_commodity_universe() -> list[str]:
    return MAJOR_COMMODITIES.copy()

def get_crypto_universe() -> list[str]:
    return CRYPTO_TOP_20.copy()

def get_fx_universe() -> list[str]:
    return FX_MAJORS.copy()

def classify_by_gics(symbols: Optional[list[str]] = None) -> dict[str, dict[str, str]]:
    """Classify equity symbols by GICS sector."""
    if symbols is None:
        symbols = SP500_TOP_50
    return {
        sym: {"sector": EQUITY_GICS_MAP.get(sym, "Unknown"), "in_map": sym in EQUITY_GICS_MAP}
        for sym in symbols
    }


# ---------------------------------------------------------------------------
# OpenBB Data Retrieval (SOLE data source)
# ---------------------------------------------------------------------------

def _get_obb():
    from openbb import obb
    return obb


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to: Date, Open, High, Low, Close, Volume."""
    rename_map = {}
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
    df = df.rename(columns=rename_map)
    if "Date" not in df.columns and df.index.name and df.index.name.lower() == "date":
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
    return df


def get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using OpenBB as the sole data source.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str, optional
        Start date YYYY-MM-DD. Defaults to 1 year ago.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    provider : str
        OpenBB provider hint.

    Returns
    -------
    pd.DataFrame
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    asset_class = detect_asset_class(symbol)
    obb = _get_obb()

    try:
        if asset_class == AssetClass.CRYPTO:
            result = obb.crypto.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        elif asset_class == AssetClass.FX:
            result = obb.currency.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        else:
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )

        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"OpenBB returned empty DataFrame for {symbol}")
        return _normalise(df)

    except ImportError:
        raise RuntimeError("OpenBB SDK not installed. pip install openbb")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {symbol} via OpenBB: {exc}") from exc


def get_multiple(
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yfinance",
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for multiple symbols."""
    result = {}
    for sym in symbols:
        try:
            result[sym] = get_historical(sym, start=start, end=end, provider=provider)
        except Exception as exc:
            logger.error("Skipping %s: %s", sym, exc)
    return result


def get_fundamentals(symbol: str, provider: str = "yfinance") -> dict[str, Any]:
    """Fetch fundamental data for an equity symbol via OpenBB."""
    obb = _get_obb()
    result = {}
    try:
        overview = obb.equity.fundamental.overview(symbol=symbol, provider=provider)
        overview_df = overview.to_dataframe()
        if not overview_df.empty:
            row = overview_df.iloc[0]
            for key in [
                "market_cap", "pe_ratio", "forward_pe", "price_to_book",
                "dividend_yield", "beta", "eps", "sector", "industry",
            ]:
                result[key] = row.get(key, None)
    except Exception as exc:
        logger.warning("Could not fetch fundamentals for %s: %s", symbol, exc)
    return result


def get_macro_data(
    indicator: str = "GDP", country: str = "united_states", provider: str = "fred"
) -> pd.DataFrame:
    """Fetch macroeconomic data via OpenBB."""
    obb = _get_obb()
    try:
        result = obb.economy.fred_series(symbol=indicator, provider=provider)
        df = result.to_dataframe()
        if not df.empty:
            return df
    except Exception as exc:
        logger.warning("Failed to fetch macro data for %s: %s", indicator, exc)
    raise RuntimeError(f"Could not fetch macro indicator: {indicator}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    universe = get_full_universe()
    for ac, symbols in universe.items():
        print(f"{ac.value}: {len(symbols)} symbols")
