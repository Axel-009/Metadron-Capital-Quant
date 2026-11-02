"""
Unified OpenBB Universe Module for Stock-Chain.

Provides access to the FULL investment universe across all asset classes
using OpenBB as the SOLE data source. No yfinance fallback.

Asset Classes Covered:
    - EQUITY: Stocks with full GICS classification hierarchy
    - FIXED_INCOME: Government and corporate bonds
    - COMMODITY: Precious metals, energy, agriculture
    - CRYPTO: Digital assets (treated as standard asset class)
    - FX: Foreign exchange pairs
    - ETF: Exchange-traded funds across all asset classes
    - INDEX: Major global equity and bond indices

Mathematical Foundation:
    Returns:
        r_t = (P_t - P_{t-1}) / P_{t-1}                  (simple return)
        R_t = ln(P_t / P_{t-1})                           (log return)

    Volatility:
        sigma = sqrt( (1/(N-1)) * sum((r_i - r_bar)^2) )  (sample std dev)

    Sharpe Ratio:
        SR = (E[R_p] - R_f) / sigma_p

    Beta:
        beta = Cov(R_asset, R_market) / Var(R_market)

Usage:
    from openbb_data import get_full_universe, get_historical, AssetClass

    universe = get_full_universe()
    df = get_historical("AAPL", start="2023-01-01")
    asset_type = detect_asset_class("AAPL")
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
            "151020 - Construction Materials": [
                "15102010 - Construction Materials",
            ],
            "151030 - Containers & Packaging": [
                "15103010 - Metal, Glass & Plastic Containers",
                "15103020 - Paper & Plastic Packaging Products & Materials",
            ],
            "151040 - Metals & Mining": [
                "15104010 - Aluminum",
                "15104020 - Diversified Metals & Mining",
                "15104025 - Copper",
                "15104030 - Gold",
                "15104040 - Precious Metals & Minerals",
                "15104045 - Silver",
                "15104050 - Steel",
            ],
            "151050 - Paper & Forest Products": [
                "15105010 - Forest Products",
                "15105020 - Paper Products",
            ],
        },
    },
    "20 - Industrials": {
        "2010 - Capital Goods": {
            "201010 - Aerospace & Defense": [
                "20101010 - Aerospace & Defense",
            ],
            "201020 - Building Products": [
                "20102010 - Building Products",
            ],
            "201030 - Construction & Engineering": [
                "20103010 - Construction & Engineering",
            ],
            "201040 - Electrical Equipment": [
                "20104010 - Electrical Components & Equipment",
                "20104020 - Heavy Electrical Equipment",
            ],
            "201050 - Industrial Conglomerates": [
                "20105010 - Industrial Conglomerates",
            ],
            "201060 - Machinery": [
                "20106010 - Construction Machinery & Heavy Transportation Equipment",
                "20106015 - Agricultural & Farm Machinery",
                "20106020 - Industrial Machinery & Supplies & Components",
            ],
            "201070 - Trading Companies & Distributors": [
                "20107010 - Trading Companies & Distributors",
            ],
        },
        "2020 - Commercial & Professional Services": {
            "202010 - Commercial Services & Supplies": [
                "20201010 - Commercial Printing",
                "20201050 - Environmental & Facilities Services",
                "20201060 - Office Services & Supplies",
                "20201070 - Diversified Support Services",
                "20201080 - Security & Alarm Services",
            ],
            "202020 - Professional Services": [
                "20202010 - Human Resource & Employment Services",
                "20202020 - Research & Consulting Services",
                "20202030 - Data Processing & Outsourced Services",
            ],
        },
        "2030 - Transportation": {
            "203010 - Air Freight & Logistics": [
                "20301010 - Air Freight & Logistics",
            ],
            "203020 - Passenger Airlines": [
                "20302010 - Passenger Airlines",
            ],
            "203030 - Marine Transportation": [
                "20303010 - Marine Transportation",
            ],
            "203040 - Ground Transportation": [
                "20304010 - Rail Transportation",
                "20304020 - Cargo Ground Transportation",
                "20304030 - Passenger Ground Transportation",
            ],
            "203050 - Transportation Infrastructure": [
                "20305010 - Airport Services",
                "20305020 - Highways & Railtracks",
                "20305030 - Marine Ports & Services",
            ],
        },
    },
    "25 - Consumer Discretionary": {
        "2510 - Automobiles & Components": {
            "251010 - Automobile Components": [
                "25101010 - Automotive Parts & Equipment",
                "25101020 - Tires & Rubber",
            ],
            "251020 - Automobiles": [
                "25102010 - Automobile Manufacturers",
                "25102020 - Motorcycle Manufacturers",
            ],
        },
        "2520 - Consumer Durables & Apparel": {
            "252010 - Household Durables": [
                "25201010 - Consumer Electronics",
                "25201020 - Home Furnishings",
                "25201030 - Homebuilding",
                "25201040 - Household Appliances",
                "25201050 - Housewares & Specialties",
            ],
            "252020 - Leisure Products": [
                "25202010 - Leisure Products",
            ],
            "252030 - Textiles, Apparel & Luxury Goods": [
                "25203010 - Apparel, Accessories & Luxury Goods",
                "25203020 - Footwear",
                "25203030 - Textiles",
            ],
        },
        "2530 - Consumer Services": {
            "253010 - Hotels, Restaurants & Leisure": [
                "25301010 - Casinos & Gaming",
                "25301020 - Hotels, Resorts & Cruise Lines",
                "25301030 - Leisure Facilities",
                "25301040 - Restaurants",
            ],
            "253020 - Diversified Consumer Services": [
                "25302010 - Education Services",
                "25302020 - Specialized Consumer Services",
            ],
        },
        "2550 - Retailing": {
            "255010 - Distributors": [
                "25501010 - Distributors",
            ],
            "255020 - Internet & Direct Marketing Retail": [
                "25502020 - Internet & Direct Marketing Retail",
            ],
            "255030 - Broadline Retail": [
                "25503010 - Broadline Retail",
            ],
            "255040 - Specialty Retail": [
                "25504010 - Apparel Retail",
                "25504020 - Computer & Electronics Retail",
                "25504030 - Home Improvement Retail",
                "25504040 - Other Specialty Retail",
                "25504050 - Automotive Retail",
                "25504060 - Homefurnishing Retail",
            ],
        },
    },
    "30 - Consumer Staples": {
        "3010 - Food & Staples Retailing": {
            "301010 - Consumer Staples Distribution & Retail": [
                "30101010 - Drug Retail",
                "30101020 - Food Distributors",
                "30101030 - Food Retail",
                "30101040 - Consumer Staples Merchandise Retail",
            ],
        },
        "3020 - Food, Beverage & Tobacco": {
            "302010 - Beverages": [
                "30201010 - Brewers",
                "30201020 - Distillers & Vintners",
                "30201030 - Soft Drinks & Non-alcoholic Beverages",
            ],
            "302020 - Food Products": [
                "30202010 - Agricultural Products & Services",
                "30202030 - Packaged Foods & Meats",
            ],
            "302030 - Tobacco": [
                "30203010 - Tobacco",
            ],
        },
        "3030 - Household & Personal Products": {
            "303010 - Household Products": [
                "30301010 - Household Products",
            ],
            "303020 - Personal Care Products": [
                "30302010 - Personal Care Products",
            ],
        },
    },
    "35 - Health Care": {
        "3510 - Health Care Equipment & Services": {
            "351010 - Health Care Equipment & Supplies": [
                "35101010 - Health Care Equipment",
                "35101020 - Health Care Supplies",
            ],
            "351020 - Health Care Providers & Services": [
                "35102010 - Health Care Distributors",
                "35102015 - Health Care Services",
                "35102020 - Health Care Facilities",
                "35102030 - Managed Health Care",
            ],
            "351030 - Health Care Technology": [
                "35103010 - Health Care Technology",
            ],
        },
        "3520 - Pharmaceuticals, Biotechnology & Life Sciences": {
            "352010 - Biotechnology": [
                "35201010 - Biotechnology",
            ],
            "352020 - Pharmaceuticals": [
                "35202010 - Pharmaceuticals",
            ],
            "352030 - Life Sciences Tools & Services": [
                "35203010 - Life Sciences Tools & Services",
            ],
        },
    },
    "40 - Financials": {
        "4010 - Banks": {
            "401010 - Banks": [
                "40101010 - Diversified Banks",
                "40101015 - Regional Banks",
            ],
        },
        "4020 - Financial Services": {
            "402010 - Financial Services": [
                "40201020 - Diversified Financial Services",
                "40201030 - Multi-Sector Holdings",
                "40201040 - Specialized Finance",
                "40201050 - Commercial & Residential Mortgage Finance",
                "40201060 - Transaction & Payment Processing Services",
            ],
            "402020 - Consumer Finance": [
                "40202010 - Consumer Finance",
            ],
            "402030 - Capital Markets": [
                "40203010 - Asset Management & Custody Banks",
                "40203020 - Investment Banking & Brokerage",
                "40203030 - Diversified Capital Markets",
                "40203040 - Financial Exchanges & Data",
            ],
            "402040 - Mortgage Real Estate Investment Trusts": [
                "40204010 - Mortgage REITs",
            ],
        },
        "4030 - Insurance": {
            "403010 - Insurance": [
                "40301010 - Insurance Brokers",
                "40301020 - Life & Health Insurance",
                "40301030 - Multi-line Insurance",
                "40301040 - Property & Casualty Insurance",
                "40301050 - Reinsurance",
            ],
        },
    },
    "45 - Information Technology": {
        "4510 - Software & Services": {
            "451020 - IT Services": [
                "45102010 - IT Consulting & Other Services",
                "45102020 - Internet Services & Infrastructure",
            ],
            "451030 - Software": [
                "45103010 - Application Software",
                "45103020 - Systems Software",
            ],
        },
        "4520 - Technology Hardware & Equipment": {
            "452010 - Communications Equipment": [
                "45201020 - Communications Equipment",
            ],
            "452020 - Technology Hardware, Storage & Peripherals": [
                "45202030 - Technology Hardware, Storage & Peripherals",
            ],
            "452030 - Electronic Equipment, Instruments & Components": [
                "45203010 - Electronic Equipment & Instruments",
                "45203015 - Electronic Components",
                "45203020 - Electronic Manufacturing Services",
                "45203030 - Technology Distributors",
            ],
        },
        "4530 - Semiconductors & Semiconductor Equipment": {
            "453010 - Semiconductors & Semiconductor Equipment": [
                "45301010 - Semiconductor Materials & Equipment",
                "45301020 - Semiconductors",
            ],
        },
    },
    "50 - Communication Services": {
        "5010 - Telecommunication Services": {
            "501010 - Diversified Telecommunication Services": [
                "50101010 - Alternative Carriers",
                "50101020 - Integrated Telecommunication Services",
            ],
            "501020 - Wireless Telecommunication Services": [
                "50102010 - Wireless Telecommunication Services",
            ],
        },
        "5020 - Media & Entertainment": {
            "502010 - Media": [
                "50201010 - Advertising",
                "50201020 - Broadcasting",
                "50201030 - Cable & Satellite",
                "50201040 - Publishing",
            ],
            "502020 - Entertainment": [
                "50202010 - Movies & Entertainment",
                "50202020 - Interactive Home Entertainment",
            ],
            "502030 - Interactive Media & Services": [
                "50203010 - Interactive Media & Services",
            ],
        },
    },
    "55 - Utilities": {
        "5510 - Utilities": {
            "551010 - Electric Utilities": [
                "55101010 - Electric Utilities",
            ],
            "551020 - Gas Utilities": [
                "55102010 - Gas Utilities",
            ],
            "551030 - Multi-Utilities": [
                "55103010 - Multi-Utilities",
            ],
            "551040 - Water Utilities": [
                "55104010 - Water Utilities",
            ],
            "551050 - Independent Power and Renewable Electricity Producers": [
                "55105010 - Independent Power Producers & Energy Traders",
                "55105020 - Renewable Electricity",
            ],
        },
    },
    "60 - Real Estate": {
        "6010 - Equity Real Estate Investment Trusts": {
            "601010 - Diversified REITs": [
                "60101010 - Diversified REITs",
            ],
            "601025 - Industrial REITs": [
                "60102510 - Industrial REITs",
            ],
            "601030 - Hotel & Resort REITs": [
                "60103010 - Hotel & Resort REITs",
            ],
            "601040 - Office REITs": [
                "60104010 - Office REITs",
            ],
            "601050 - Health Care REITs": [
                "60105010 - Health Care REITs",
            ],
            "601060 - Residential REITs": [
                "60106010 - Multi-Family Residential REITs",
                "60106020 - Single-Family Residential REITs",
            ],
            "601070 - Retail REITs": [
                "60107010 - Retail REITs",
            ],
            "601080 - Specialized REITs": [
                "60108010 - Data Center REITs",
                "60108020 - Infrastructure REITs",
                "60108030 - Self-Storage REITs",
                "60108040 - Telecom Tower REITs",
                "60108050 - Timber REITs",
                "60108060 - Other Specialized REITs",
            ],
        },
        "6020 - Real Estate Management & Development": {
            "602010 - Real Estate Management & Development": [
                "60201010 - Diversified Real Estate Activities",
                "60201020 - Real Estate Operating Companies",
                "60201030 - Real Estate Development",
                "60201040 - Real Estate Services",
            ],
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
    "AAPL": "45 - Information Technology",
    "MSFT": "45 - Information Technology",
    "AMZN": "25 - Consumer Discretionary",
    "NVDA": "45 - Information Technology",
    "GOOGL": "50 - Communication Services",
    "META": "50 - Communication Services",
    "BRK-B": "40 - Financials",
    "TSLA": "25 - Consumer Discretionary",
    "UNH": "35 - Health Care",
    "XOM": "10 - Energy",
    "JNJ": "35 - Health Care",
    "JPM": "40 - Financials",
    "V": "40 - Financials",
    "PG": "30 - Consumer Staples",
    "MA": "40 - Financials",
    "HD": "25 - Consumer Discretionary",
    "AVGO": "45 - Information Technology",
    "CVX": "10 - Energy",
    "MRK": "35 - Health Care",
    "ABBV": "35 - Health Care",
    "LLY": "35 - Health Care",
    "COST": "30 - Consumer Staples",
    "PEP": "30 - Consumer Staples",
    "KO": "30 - Consumer Staples",
    "ADBE": "45 - Information Technology",
    "WMT": "30 - Consumer Staples",
    "MCD": "25 - Consumer Discretionary",
    "CSCO": "45 - Information Technology",
    "CRM": "45 - Information Technology",
    "ACN": "45 - Information Technology",
    "ABT": "35 - Health Care",
    "TMO": "35 - Health Care",
    "DHR": "35 - Health Care",
    "LIN": "15 - Materials",
    "CMCSA": "50 - Communication Services",
    "NKE": "25 - Consumer Discretionary",
    "VZ": "50 - Communication Services",
    "NEE": "55 - Utilities",
    "TXN": "45 - Information Technology",
    "PM": "30 - Consumer Staples",
    "INTC": "45 - Information Technology",
    "RTX": "20 - Industrials",
    "HON": "20 - Industrials",
    "UNP": "20 - Industrials",
    "LOW": "25 - Consumer Discretionary",
    "QCOM": "45 - Information Technology",
    "ORCL": "45 - Information Technology",
    "AMGN": "35 - Health Care",
    "IBM": "45 - Information Technology",
    "BA": "20 - Industrials",
}

MAJOR_BONDS = [
    "TLT",   # iShares 20+ Year Treasury Bond ETF (proxy for US long bonds)
    "IEF",   # iShares 7-10 Year Treasury Bond ETF
    "SHY",   # iShares 1-3 Year Treasury Bond ETF
    "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond ETF
    "HYG",   # iShares iBoxx $ High Yield Corporate Bond ETF
    "TIP",   # iShares TIPS Bond ETF
    "AGG",   # iShares Core U.S. Aggregate Bond ETF
    "BND",   # Vanguard Total Bond Market ETF
    "MUB",   # iShares National Muni Bond ETF
    "EMB",   # iShares J.P. Morgan USD Emerging Markets Bond ETF
    "BNDX",  # Vanguard Total International Bond ETF
    "GOVT",  # iShares U.S. Treasury Bond ETF
]

MAJOR_COMMODITIES = [
    "GC=F",   # Gold futures
    "SI=F",   # Silver futures
    "CL=F",   # Crude Oil WTI futures
    "BZ=F",   # Brent Crude futures
    "NG=F",   # Natural Gas futures
    "HG=F",   # Copper futures
    "PL=F",   # Platinum futures
    "PA=F",   # Palladium futures
    "ZC=F",   # Corn futures
    "ZW=F",   # Wheat futures
    "ZS=F",   # Soybean futures
    "KC=F",   # Coffee futures
    "SB=F",   # Sugar futures
    "CT=F",   # Cotton futures
    "LE=F",   # Live Cattle futures
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
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones Industrial Average
    "^IXIC",   # NASDAQ Composite
    "^RUT",    # Russell 2000
    "^VIX",    # CBOE Volatility Index
    "^FTSE",   # FTSE 100
    "^GDAXI",  # DAX
    "^FCHI",   # CAC 40
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "000001.SS",  # Shanghai Composite
    "^STOXX50E",  # Euro Stoxx 50
    "^BVSP",   # Bovespa (Brazil)
    "^AXJO",   # ASX 200 (Australia)
    "^KS11",   # KOSPI (South Korea)
    "^NSEI",   # Nifty 50 (India)
    "^GSPTSE", # S&P/TSX Composite (Canada)
]


# ---------------------------------------------------------------------------
# Symbol -> Asset Class Detection
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = {"-USD", "-USDT", "-BTC", "-ETH"}
_FX_SUFFIX = "=X"
_FUTURES_SUFFIX = "=F"
_INDEX_PREFIXES = {"^"}

_BOND_SYMBOLS = set(MAJOR_BONDS)
_COMMODITY_SYMBOLS = set(MAJOR_COMMODITIES)
_ETF_SYMBOLS = set(MAJOR_ETFS)
_INDEX_SYMBOLS = set(GLOBAL_INDICES)
_CRYPTO_SYMBOLS = set(CRYPTO_TOP_20)
_EQUITY_SYMBOLS = set(SP500_TOP_50)


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Detect the asset class for a given symbol based on naming conventions
    and constituent lists.

    Detection Logic (in priority order):
        1. If symbol is in the known index set or starts with '^' -> INDEX
        2. If symbol ends with crypto suffixes (-USD, -USDT, etc.) -> CRYPTO
        3. If symbol ends with '=X' -> FX
        4. If symbol ends with '=F' or is in commodity set -> COMMODITY
        5. If symbol is in the bond ETF set -> FIXED_INCOME
        6. If symbol is in the ETF set -> ETF
        7. Default -> EQUITY

    Parameters
    ----------
    symbol : str
        The ticker/symbol to classify.

    Returns
    -------
    AssetClass
        The detected asset class.
    """
    if symbol in _INDEX_SYMBOLS or symbol.startswith("^"):
        return AssetClass.INDEX
    if any(symbol.upper().endswith(s) for s in _CRYPTO_SUFFIXES) or symbol in _CRYPTO_SYMBOLS:
        return AssetClass.CRYPTO
    if symbol.endswith(_FX_SUFFIX) and symbol not in _COMMODITY_SYMBOLS:
        return AssetClass.FX
    if symbol.endswith(_FUTURES_SUFFIX) or symbol in _COMMODITY_SYMBOLS:
        return AssetClass.COMMODITY
    if symbol in _BOND_SYMBOLS:
        return AssetClass.FIXED_INCOME
    if symbol in _ETF_SYMBOLS:
        return AssetClass.ETF
    return AssetClass.EQUITY


# ---------------------------------------------------------------------------
# Universe Retrieval Functions
# ---------------------------------------------------------------------------

def get_full_universe() -> dict[AssetClass, list[str]]:
    """
    Return the complete investment universe organized by asset class.

    Returns
    -------
    dict[AssetClass, list[str]]
        Mapping from each AssetClass to its list of constituent symbols.
    """
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
    """Return the equity constituent list (S&P 500 top 50)."""
    return SP500_TOP_50.copy()


def get_bond_universe() -> list[str]:
    """Return the fixed-income constituent list."""
    return MAJOR_BONDS.copy()


def get_commodity_universe() -> list[str]:
    """Return the commodity futures constituent list."""
    return MAJOR_COMMODITIES.copy()


def get_crypto_universe() -> list[str]:
    """Return the top 20 cryptocurrency constituent list."""
    return CRYPTO_TOP_20.copy()


def get_fx_universe() -> list[str]:
    """Return the FX major pairs list."""
    return FX_MAJORS.copy()


def classify_by_gics(symbols: Optional[list[str]] = None) -> dict[str, dict[str, str]]:
    """
    Classify equity symbols by GICS sector.

    Parameters
    ----------
    symbols : list[str], optional
        Symbols to classify. Defaults to SP500_TOP_50.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of symbol -> {"sector": <GICS sector>, "detected": True/False}.
    """
    if symbols is None:
        symbols = SP500_TOP_50

    result = {}
    for sym in symbols:
        sector = EQUITY_GICS_MAP.get(sym, "Unknown")
        result[sym] = {"sector": sector, "in_map": sym in EQUITY_GICS_MAP}
    return result


# ---------------------------------------------------------------------------
# OpenBB Data Retrieval (SOLE data source)
# ---------------------------------------------------------------------------

def _get_obb():
    """Lazy import of OpenBB SDK."""
    from openbb import obb
    return obb


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names to title-case standard: Date, Open, High, Low, Close, Volume.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from OpenBB.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names.
    """
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

    # Ensure Date column exists (may be index)
    if "Date" not in df.columns and df.index.name and df.index.name.lower() == "date":
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})

    return df


def _resolve_dates(
    start: Optional[str], end: Optional[str], default_lookback_days: int = 365
) -> tuple[str, str]:
    """Resolve start/end dates with defaults."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=default_lookback_days)).strftime(
            "%Y-%m-%d"
        )
    return start, end


def get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using OpenBB as the sole data source.

    Mathematical Notes (for ML consumption):
        Simple return:  r_t = (Close_t - Close_{t-1}) / Close_{t-1}
        Log return:     R_t = ln(Close_t / Close_{t-1})
        Realized vol:   sigma = sqrt(252) * std(R_t)  (annualized)

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "AAPL", "BTC-USD", "GC=F", "EURUSD=X").
    start : str, optional
        Start date YYYY-MM-DD. Defaults to 1 year ago.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    provider : str
        OpenBB provider hint (default "yfinance").

    Returns
    -------
    pd.DataFrame
        Standardized OHLCV DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Raises
    ------
    RuntimeError
        If OpenBB cannot fetch the data.
    """
    start, end = _resolve_dates(start, end, default_lookback_days=365)
    asset_class = detect_asset_class(symbol)
    obb = _get_obb()

    logger.info("Fetching %s [%s] via OpenBB (%s)", symbol, asset_class.value, provider)

    try:
        if asset_class == AssetClass.CRYPTO:
            result = obb.crypto.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        elif asset_class == AssetClass.FX:
            result = obb.currency.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        elif asset_class in (
            AssetClass.INDEX,
            AssetClass.ETF,
            AssetClass.EQUITY,
            AssetClass.FIXED_INCOME,
            AssetClass.COMMODITY,
        ):
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        else:
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )

        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"OpenBB returned empty DataFrame for {symbol}")

        logger.info("OpenBB returned %d rows for %s", len(df), symbol)
        return _normalise(df)

    except ImportError:
        raise RuntimeError(
            "OpenBB SDK is not installed. Install with: pip install openbb"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {symbol} via OpenBB: {exc}") from exc


def get_fundamentals(
    symbol: str,
    provider: str = "yfinance",
) -> dict[str, Any]:
    """
    Fetch fundamental data for an equity symbol via OpenBB.

    Retrieves key financial metrics used in valuation:
        P/E Ratio:           P/E = Price / EPS
        EV/EBITDA:           EV/EBITDA = (Market Cap + Debt - Cash) / EBITDA
        Price-to-Book:       P/B = Price / (Total Equity / Shares Outstanding)
        Dividend Yield:      DY = Annual Dividends / Price
        Free Cash Flow Yield: FCFY = FCF / Market Cap

    Parameters
    ----------
    symbol : str
        Equity ticker symbol.
    provider : str
        OpenBB provider hint.

    Returns
    -------
    dict[str, Any]
        Dictionary of fundamental metrics.
    """
    obb = _get_obb()
    result = {}

    try:
        # Key metrics
        overview = obb.equity.fundamental.overview(symbol=symbol, provider=provider)
        overview_df = overview.to_dataframe()
        if not overview_df.empty:
            row = overview_df.iloc[0]
            result["market_cap"] = row.get("market_cap", None)
            result["pe_ratio"] = row.get("pe_ratio", None)
            result["forward_pe"] = row.get("forward_pe", None)
            result["price_to_book"] = row.get("price_to_book", None)
            result["dividend_yield"] = row.get("dividend_yield", None)
            result["beta"] = row.get("beta", None)
            result["eps"] = row.get("eps", None)
            result["sector"] = row.get("sector", None)
            result["industry"] = row.get("industry", None)
    except Exception as exc:
        logger.warning("Could not fetch overview for %s: %s", symbol, exc)

    try:
        # Income statement for EBITDA
        income = obb.equity.fundamental.income(
            symbol=symbol, provider=provider, limit=4
        )
        income_df = income.to_dataframe()
        if not income_df.empty:
            latest = income_df.iloc[0]
            result["revenue"] = latest.get("revenue", None)
            result["ebitda"] = latest.get("ebitda", None)
            result["net_income"] = latest.get("net_income", None)
            result["gross_profit"] = latest.get("gross_profit", None)
    except Exception as exc:
        logger.warning("Could not fetch income statement for %s: %s", symbol, exc)

    try:
        # Balance sheet for book value
        balance = obb.equity.fundamental.balance(
            symbol=symbol, provider=provider, limit=1
        )
        balance_df = balance.to_dataframe()
        if not balance_df.empty:
            latest = balance_df.iloc[0]
            result["total_assets"] = latest.get("total_assets", None)
            result["total_debt"] = latest.get("total_debt", None)
            result["total_equity"] = latest.get("total_stockholders_equity", None)
            result["cash"] = latest.get("cash_and_cash_equivalents", None)
    except Exception as exc:
        logger.warning("Could not fetch balance sheet for %s: %s", symbol, exc)

    try:
        # Cash flow for FCF
        cashflow = obb.equity.fundamental.cash(
            symbol=symbol, provider=provider, limit=1
        )
        cf_df = cashflow.to_dataframe()
        if not cf_df.empty:
            latest = cf_df.iloc[0]
            result["operating_cash_flow"] = latest.get("operating_cash_flow", None)
            result["capital_expenditure"] = latest.get("capital_expenditure", None)
            ocf = result.get("operating_cash_flow")
            capex = result.get("capital_expenditure")
            if ocf is not None and capex is not None:
                result["free_cash_flow"] = ocf - abs(capex)
    except Exception as exc:
        logger.warning("Could not fetch cash flow for %s: %s", symbol, exc)

    return result


def get_macro_data(
    indicator: str = "GDP",
    country: str = "united_states",
    provider: str = "fred",
) -> pd.DataFrame:
    """
    Fetch macroeconomic data via OpenBB.

    Common indicators and their economic significance:
        GDP:         Gross Domestic Product (total economic output)
        CPI:         Consumer Price Index (inflation measure)
        UNRATE:      Unemployment Rate
        FEDFUNDS:    Federal Funds Rate (monetary policy)
        T10Y2Y:      10Y-2Y Treasury Spread (recession indicator)
                     Inversion (< 0) historically precedes recessions by 12-18 months

    Parameters
    ----------
    indicator : str
        FRED series ID or macro indicator name.
    country : str
        Country for the data.
    provider : str
        OpenBB provider (default "fred").

    Returns
    -------
    pd.DataFrame
        Time series of the macro indicator.
    """
    obb = _get_obb()

    try:
        result = obb.economy.fred_series(
            symbol=indicator, provider=provider
        )
        df = result.to_dataframe()
        if not df.empty:
            return df
    except Exception as exc:
        logger.warning("Failed to fetch macro data for %s: %s", indicator, exc)

    raise RuntimeError(f"Could not fetch macro indicator: {indicator}")


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Full Universe ===")
    universe = get_full_universe()
    for ac, symbols in universe.items():
        print(f"  {ac.value}: {len(symbols)} symbols")

    print("\n=== GICS Classification (sample) ===")
    gics = classify_by_gics(["AAPL", "JPM", "XOM", "UNH", "NVDA"])
    for sym, info in gics.items():
        print(f"  {sym}: {info['sector']}")

    print("\n=== Asset Class Detection ===")
    test_symbols = ["AAPL", "BTC-USD", "GC=F", "EURUSD=X", "^GSPC", "SPY", "TLT"]
    for sym in test_symbols:
        print(f"  {sym}: {detect_asset_class(sym).value}")

    print("\n=== Historical Data (AAPL) ===")
    try:
        df = get_historical("AAPL")
        print(f"  Shape: {df.shape}")
        print(df.head())
    except RuntimeError as e:
        print(f"  Skipped: {e}")
