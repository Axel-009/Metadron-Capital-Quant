"""
OpenBB Universe Data Module — QLIB Provider Interface
=====================================================

The SOLE canonical data source for the entire Metadron Capital investment
platform, adapted for Microsoft QLIB's provider interface.  This module
wraps the unified OpenBB data functions AND implements QLIB's abstract
CalendarProvider, InstrumentProvider, and FeatureProvider classes so that
QLIB can source all market data exclusively through OpenBB.

No yfinance fallback.  No alternative providers.  OpenBB is the ONLY source.

This is a sibling of ``Financial-Data/openbb_universe.py`` — the same
universe definitions, GICS taxonomy, and data functions, plus QLIB-native
provider classes.

Usage
-----
    # Standalone (same API as Financial-Data version)
    from qlib.data.openbb_universe import get_historical, AssetClass

    # As a QLIB provider (register in qlib init config)
    from qlib.data.openbb_universe import (
        OpenBBCalendarProvider,
        OpenBBInstrumentProvider,
        OpenBBFeatureProvider,
    )
    # Then configure qlib.init(provider_uri="openbb", ...)

Mathematical Formulas (for ML model docstring training)
-------------------------------------------------------
Sharpe Ratio:
    S = (R_p - R_f) / sigma_p
    where R_p = portfolio return, R_f = risk-free rate, sigma_p = std dev of
    portfolio excess returns.

Sortino Ratio:
    S = (R_p - R_f) / sigma_d
    where sigma_d = sqrt(E[min(R - R_f, 0)^2]) is the downside deviation.

Kelly Criterion:
    f* = (b * p - q) / b
    where b = decimal odds (net profit per $1 wagered), p = probability of
    winning, q = 1 - p.

Black-Scholes European Call:
    C = S * N(d1) - K * e^(-rT) * N(d2)
    d1 = (ln(S/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    where S = spot price, K = strike, r = risk-free rate, T = time to
    expiry (years), sigma = implied volatility, N() = standard normal CDF.

Value at Risk (parametric):
    VaR_alpha = mu - z_alpha * sigma
    where mu = expected return, z_alpha = normal quantile at confidence
    level alpha, sigma = portfolio std dev.

Covariance:
    Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)]
    = E[XY] - mu_X * mu_Y

Capital Asset Pricing Model (CAPM):
    E[R_i] = R_f + beta_i * (E[R_m] - R_f)
    beta_i = Cov(R_i, R_m) / Var(R_m)

Information Ratio:
    IR = (R_p - R_b) / sigma_(p-b)
    where R_b = benchmark return, sigma_(p-b) = tracking error.

Maximum Drawdown:
    MDD = max_{t in [0,T]} ( max_{s in [0,t]} V_s - V_t ) / max_{s in [0,t]} V_s

Herfindahl-Hirschman Index (concentration):
    HHI = sum_{i=1}^{N} w_i^2
"""

from __future__ import annotations

import bisect
import logging
import re
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenBB SDK accessor (lazy import)
# ---------------------------------------------------------------------------

_obb = None


def _get_obb():
    """Lazy-import the OpenBB SDK so the module can be imported without it."""
    global _obb
    if _obb is None:
        try:
            from openbb import obb
            _obb = obb
        except ImportError as exc:
            raise ImportError(
                "OpenBB Platform is required.  Install with:  pip install openbb"
            ) from exc
    return _obb


# ═══════════════════════════════════════════════════════════════════════════
# Asset-class taxonomy
# ═══════════════════════════════════════════════════════════════════════════

class AssetClass(Enum):
    """Enumeration of all supported asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    ETF = "etf"
    INDEX = "index"


# ═══════════════════════════════════════════════════════════════════════════
# GICS Classification — complete hierarchy
# ═══════════════════════════════════════════════════════════════════════════

GICS_SECTORS: Dict[int, dict] = {
    10: {
        "name": "Energy",
        "groups": {
            1010: {
                "name": "Energy",
                "industries": {
                    101010: {
                        "name": "Energy Equipment & Services",
                        "sub_industries": {
                            10101010: "Oil & Gas Drilling",
                            10101020: "Oil & Gas Equipment & Services",
                        },
                    },
                    101020: {
                        "name": "Oil, Gas & Consumable Fuels",
                        "sub_industries": {
                            10102010: "Integrated Oil & Gas",
                            10102020: "Oil & Gas Exploration & Production",
                            10102030: "Oil & Gas Refining & Marketing",
                            10102040: "Oil & Gas Storage & Transportation",
                            10102050: "Coal & Consumable Fuels",
                        },
                    },
                },
            },
        },
    },
    15: {
        "name": "Materials",
        "groups": {
            1510: {
                "name": "Materials",
                "industries": {
                    151010: {
                        "name": "Chemicals",
                        "sub_industries": {
                            15101010: "Commodity Chemicals",
                            15101020: "Diversified Chemicals",
                            15101030: "Fertilizers & Agricultural Chemicals",
                            15101040: "Industrial Gases",
                            15101050: "Specialty Chemicals",
                        },
                    },
                    151020: {
                        "name": "Construction Materials",
                        "sub_industries": {
                            15102010: "Construction Materials",
                        },
                    },
                    151030: {
                        "name": "Containers & Packaging",
                        "sub_industries": {
                            15103010: "Metal & Glass Containers",
                            15103020: "Paper Packaging",
                        },
                    },
                    151040: {
                        "name": "Metals & Mining",
                        "sub_industries": {
                            15104010: "Aluminum",
                            15104020: "Diversified Metals & Mining",
                            15104025: "Copper",
                            15104030: "Gold",
                            15104040: "Precious Metals & Minerals",
                            15104045: "Silver",
                            15104050: "Steel",
                        },
                    },
                    151050: {
                        "name": "Paper & Forest Products",
                        "sub_industries": {
                            15105010: "Forest Products",
                            15105020: "Paper Products",
                        },
                    },
                },
            },
        },
    },
    20: {
        "name": "Industrials",
        "groups": {
            2010: {
                "name": "Capital Goods",
                "industries": {
                    201010: {
                        "name": "Aerospace & Defense",
                        "sub_industries": {
                            20101010: "Aerospace & Defense",
                        },
                    },
                    201020: {
                        "name": "Building Products",
                        "sub_industries": {
                            20102010: "Building Products",
                        },
                    },
                    201030: {
                        "name": "Construction & Engineering",
                        "sub_industries": {
                            20103010: "Construction & Engineering",
                        },
                    },
                    201040: {
                        "name": "Electrical Equipment",
                        "sub_industries": {
                            20104010: "Electrical Components & Equipment",
                            20104020: "Heavy Electrical Equipment",
                        },
                    },
                    201050: {
                        "name": "Industrial Conglomerates",
                        "sub_industries": {
                            20105010: "Industrial Conglomerates",
                        },
                    },
                    201060: {
                        "name": "Machinery",
                        "sub_industries": {
                            20106010: "Construction Machinery & Heavy Trucks",
                            20106015: "Agricultural & Farm Machinery",
                            20106020: "Industrial Machinery",
                        },
                    },
                    201070: {
                        "name": "Trading Companies & Distributors",
                        "sub_industries": {
                            20107010: "Trading Companies & Distributors",
                        },
                    },
                },
            },
            2020: {
                "name": "Commercial & Professional Services",
                "industries": {
                    202010: {
                        "name": "Commercial Services & Supplies",
                        "sub_industries": {
                            20201010: "Commercial Printing",
                            20201050: "Environmental & Facilities Services",
                            20201060: "Office Services & Supplies",
                            20201070: "Diversified Support Services",
                            20201080: "Security & Alarm Services",
                        },
                    },
                    202020: {
                        "name": "Professional Services",
                        "sub_industries": {
                            20202010: "Human Resource & Employment Services",
                            20202020: "Research & Consulting Services",
                        },
                    },
                },
            },
            2030: {
                "name": "Transportation",
                "industries": {
                    203010: {
                        "name": "Air Freight & Logistics",
                        "sub_industries": {
                            20301010: "Air Freight & Logistics",
                        },
                    },
                    203020: {
                        "name": "Airlines",
                        "sub_industries": {
                            20302010: "Airlines",
                        },
                    },
                    203030: {
                        "name": "Marine",
                        "sub_industries": {
                            20303010: "Marine",
                        },
                    },
                    203040: {
                        "name": "Road & Rail",
                        "sub_industries": {
                            20304010: "Railroads",
                            20304020: "Trucking",
                        },
                    },
                    203050: {
                        "name": "Transportation Infrastructure",
                        "sub_industries": {
                            20305010: "Airport Services",
                            20305020: "Highways & Railtracks",
                            20305030: "Marine Ports & Services",
                        },
                    },
                },
            },
        },
    },
    25: {
        "name": "Consumer Discretionary",
        "groups": {
            2510: {
                "name": "Automobiles & Components",
                "industries": {
                    251010: {
                        "name": "Auto Components",
                        "sub_industries": {
                            25101010: "Auto Parts & Equipment",
                            25101020: "Tires & Rubber",
                        },
                    },
                    251020: {
                        "name": "Automobiles",
                        "sub_industries": {
                            25102010: "Automobile Manufacturers",
                            25102020: "Motorcycle Manufacturers",
                        },
                    },
                },
            },
            2520: {
                "name": "Consumer Durables & Apparel",
                "industries": {
                    252010: {
                        "name": "Household Durables",
                        "sub_industries": {
                            25201010: "Consumer Electronics",
                            25201020: "Home Furnishings",
                            25201030: "Homebuilding",
                            25201040: "Household Appliances",
                            25201050: "Housewares & Specialties",
                        },
                    },
                    252020: {
                        "name": "Leisure Products",
                        "sub_industries": {
                            25202010: "Leisure Products",
                        },
                    },
                    252030: {
                        "name": "Textiles, Apparel & Luxury Goods",
                        "sub_industries": {
                            25203010: "Apparel, Accessories & Luxury Goods",
                            25203020: "Footwear",
                            25203030: "Textiles",
                        },
                    },
                },
            },
            2530: {
                "name": "Consumer Services",
                "industries": {
                    253010: {
                        "name": "Hotels, Restaurants & Leisure",
                        "sub_industries": {
                            25301010: "Casinos & Gaming",
                            25301020: "Hotels, Resorts & Cruise Lines",
                            25301030: "Leisure Facilities",
                            25301040: "Restaurants",
                        },
                    },
                    253020: {
                        "name": "Diversified Consumer Services",
                        "sub_industries": {
                            25302010: "Education Services",
                            25302020: "Specialized Consumer Services",
                        },
                    },
                },
            },
            2550: {
                "name": "Retailing",
                "industries": {
                    255010: {
                        "name": "Distributors",
                        "sub_industries": {
                            25501010: "Distributors",
                        },
                    },
                    255020: {
                        "name": "Internet & Direct Marketing Retail",
                        "sub_industries": {
                            25502020: "Internet & Direct Marketing Retail",
                        },
                    },
                    255030: {
                        "name": "Multiline Retail",
                        "sub_industries": {
                            25503010: "Department Stores",
                            25503020: "General Merchandise Stores",
                        },
                    },
                    255040: {
                        "name": "Specialty Retail",
                        "sub_industries": {
                            25504010: "Apparel Retail",
                            25504020: "Computer & Electronics Retail",
                            25504030: "Home Improvement Retail",
                            25504040: "Specialty Stores",
                            25504050: "Automotive Retail",
                            25504060: "Homefurnishing Retail",
                        },
                    },
                },
            },
        },
    },
    30: {
        "name": "Consumer Staples",
        "groups": {
            3010: {
                "name": "Food & Staples Retailing",
                "industries": {
                    301010: {
                        "name": "Food & Staples Retailing",
                        "sub_industries": {
                            30101010: "Drug Retail",
                            30101020: "Food Distributors",
                            30101030: "Food Retail",
                            30101040: "Hypermarkets & Super Centers",
                        },
                    },
                },
            },
            3020: {
                "name": "Food, Beverage & Tobacco",
                "industries": {
                    302010: {
                        "name": "Beverages",
                        "sub_industries": {
                            30201010: "Brewers",
                            30201020: "Distillers & Vintners",
                            30201030: "Soft Drinks",
                        },
                    },
                    302020: {
                        "name": "Food Products",
                        "sub_industries": {
                            30202010: "Agricultural Products",
                            30202030: "Packaged Foods & Meats",
                        },
                    },
                    302030: {
                        "name": "Tobacco",
                        "sub_industries": {
                            30203010: "Tobacco",
                        },
                    },
                },
            },
            3030: {
                "name": "Household & Personal Products",
                "industries": {
                    303010: {
                        "name": "Household Products",
                        "sub_industries": {
                            30301010: "Household Products",
                        },
                    },
                    303020: {
                        "name": "Personal Products",
                        "sub_industries": {
                            30302010: "Personal Products",
                        },
                    },
                },
            },
        },
    },
    35: {
        "name": "Health Care",
        "groups": {
            3510: {
                "name": "Health Care Equipment & Services",
                "industries": {
                    351010: {
                        "name": "Health Care Equipment & Supplies",
                        "sub_industries": {
                            35101010: "Health Care Equipment",
                            35101020: "Health Care Supplies",
                        },
                    },
                    351020: {
                        "name": "Health Care Providers & Services",
                        "sub_industries": {
                            35102010: "Health Care Distributors",
                            35102015: "Health Care Services",
                            35102020: "Health Care Facilities",
                            35102030: "Managed Health Care",
                        },
                    },
                    351030: {
                        "name": "Health Care Technology",
                        "sub_industries": {
                            35103010: "Health Care Technology",
                        },
                    },
                },
            },
            3520: {
                "name": "Pharmaceuticals, Biotechnology & Life Sciences",
                "industries": {
                    352010: {
                        "name": "Biotechnology",
                        "sub_industries": {
                            35201010: "Biotechnology",
                        },
                    },
                    352020: {
                        "name": "Pharmaceuticals",
                        "sub_industries": {
                            35202010: "Pharmaceuticals",
                        },
                    },
                    352030: {
                        "name": "Life Sciences Tools & Services",
                        "sub_industries": {
                            35203010: "Life Sciences Tools & Services",
                        },
                    },
                },
            },
        },
    },
    40: {
        "name": "Financials",
        "groups": {
            4010: {
                "name": "Banks",
                "industries": {
                    401010: {
                        "name": "Banks",
                        "sub_industries": {
                            40101010: "Diversified Banks",
                            40101015: "Regional Banks",
                        },
                    },
                    401020: {
                        "name": "Thrifts & Mortgage Finance",
                        "sub_industries": {
                            40102010: "Thrifts & Mortgage Finance",
                        },
                    },
                },
            },
            4020: {
                "name": "Diversified Financials",
                "industries": {
                    402010: {
                        "name": "Diversified Financial Services",
                        "sub_industries": {
                            40201020: "Other Diversified Financial Services",
                            40201030: "Multi-Sector Holdings",
                            40201040: "Specialized Finance",
                        },
                    },
                    402020: {
                        "name": "Consumer Finance",
                        "sub_industries": {
                            40202010: "Consumer Finance",
                        },
                    },
                    402030: {
                        "name": "Capital Markets",
                        "sub_industries": {
                            40203010: "Asset Management & Custody Banks",
                            40203020: "Investment Banking & Brokerage",
                            40203030: "Diversified Capital Markets",
                            40203040: "Financial Exchanges & Data",
                        },
                    },
                    402040: {
                        "name": "Mortgage Real Estate Investment Trusts (REITs)",
                        "sub_industries": {
                            40204010: "Mortgage REITs",
                        },
                    },
                },
            },
            4030: {
                "name": "Insurance",
                "industries": {
                    403010: {
                        "name": "Insurance",
                        "sub_industries": {
                            40301010: "Insurance Brokers",
                            40301020: "Life & Health Insurance",
                            40301030: "Multi-line Insurance",
                            40301040: "Property & Casualty Insurance",
                            40301050: "Reinsurance",
                        },
                    },
                },
            },
        },
    },
    45: {
        "name": "Information Technology",
        "groups": {
            4510: {
                "name": "Software & Services",
                "industries": {
                    451010: {
                        "name": "Internet Software & Services",
                        "sub_industries": {
                            45101010: "Internet Software & Services",
                        },
                    },
                    451020: {
                        "name": "IT Services",
                        "sub_industries": {
                            45102010: "IT Consulting & Other Services",
                            45102020: "Data Processing & Outsourced Services",
                        },
                    },
                    451030: {
                        "name": "Software",
                        "sub_industries": {
                            45103010: "Application Software",
                            45103020: "Systems Software",
                        },
                    },
                },
            },
            4520: {
                "name": "Technology Hardware & Equipment",
                "industries": {
                    452010: {
                        "name": "Communications Equipment",
                        "sub_industries": {
                            45201020: "Communications Equipment",
                        },
                    },
                    452020: {
                        "name": "Technology Hardware, Storage & Peripherals",
                        "sub_industries": {
                            45202030: "Technology Hardware, Storage & Peripherals",
                        },
                    },
                    452030: {
                        "name": "Electronic Equipment, Instruments & Components",
                        "sub_industries": {
                            45203010: "Electronic Equipment & Instruments",
                            45203015: "Electronic Components",
                            45203020: "Electronic Manufacturing Services",
                            45203030: "Technology Distributors",
                        },
                    },
                },
            },
            4530: {
                "name": "Semiconductors & Semiconductor Equipment",
                "industries": {
                    453010: {
                        "name": "Semiconductors & Semiconductor Equipment",
                        "sub_industries": {
                            45301010: "Semiconductor Equipment",
                            45301020: "Semiconductors",
                        },
                    },
                },
            },
        },
    },
    50: {
        "name": "Communication Services",
        "groups": {
            5010: {
                "name": "Telecommunication Services",
                "industries": {
                    501010: {
                        "name": "Diversified Telecommunication Services",
                        "sub_industries": {
                            50101010: "Alternative Carriers",
                            50101020: "Integrated Telecommunication Services",
                        },
                    },
                    501020: {
                        "name": "Wireless Telecommunication Services",
                        "sub_industries": {
                            50102010: "Wireless Telecommunication Services",
                        },
                    },
                },
            },
            5020: {
                "name": "Media & Entertainment",
                "industries": {
                    502010: {
                        "name": "Media",
                        "sub_industries": {
                            50201010: "Advertising",
                            50201020: "Broadcasting",
                            50201030: "Cable & Satellite",
                            50201040: "Publishing",
                        },
                    },
                    502020: {
                        "name": "Entertainment",
                        "sub_industries": {
                            50202010: "Movies & Entertainment",
                            50202020: "Interactive Home Entertainment",
                        },
                    },
                    502030: {
                        "name": "Interactive Media & Services",
                        "sub_industries": {
                            50203010: "Interactive Media & Services",
                        },
                    },
                },
            },
        },
    },
    55: {
        "name": "Utilities",
        "groups": {
            5510: {
                "name": "Utilities",
                "industries": {
                    551010: {
                        "name": "Electric Utilities",
                        "sub_industries": {
                            55101010: "Electric Utilities",
                        },
                    },
                    551020: {
                        "name": "Gas Utilities",
                        "sub_industries": {
                            55102010: "Gas Utilities",
                        },
                    },
                    551030: {
                        "name": "Multi-Utilities",
                        "sub_industries": {
                            55103010: "Multi-Utilities",
                        },
                    },
                    551040: {
                        "name": "Water Utilities",
                        "sub_industries": {
                            55104010: "Water Utilities",
                        },
                    },
                    551050: {
                        "name": "Independent Power and Renewable Electricity Producers",
                        "sub_industries": {
                            55105010: "Independent Power Producers & Energy Traders",
                            55105020: "Renewable Electricity",
                        },
                    },
                },
            },
        },
    },
    60: {
        "name": "Real Estate",
        "groups": {
            6010: {
                "name": "Real Estate",
                "industries": {
                    601010: {
                        "name": "Equity Real Estate Investment Trusts (REITs)",
                        "sub_industries": {
                            60101010: "Diversified REITs",
                            60101020: "Industrial REITs",
                            60101030: "Hotel & Resort REITs",
                            60101040: "Office REITs",
                            60101050: "Health Care REITs",
                            60101060: "Residential REITs",
                            60101070: "Retail REITs",
                            60101080: "Specialized REITs",
                        },
                    },
                    601020: {
                        "name": "Real Estate Management & Development",
                        "sub_industries": {
                            60102010: "Diversified Real Estate Activities",
                            60102020: "Real Estate Operating Companies",
                            60102030: "Real Estate Development",
                            60102040: "Real Estate Services",
                        },
                    },
                },
            },
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Universe constituent lists
# ═══════════════════════════════════════════════════════════════════════════

SP500_REPRESENTATIVE: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B",
    "TSLA", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO",
    "CVX", "MRK", "LLY", "ABBV", "COST", "PEP", "KO", "ADBE", "WMT",
    "MCD", "CRM", "CSCO", "TMO", "ACN", "ABT", "LIN", "NKE", "DHR",
    "ORCL", "TXN", "CMCSA", "PM", "NEE", "UNP", "RTX", "LOW", "INTC",
    "HON", "AMD", "AMGN", "UPS", "IBM", "GE", "CAT", "BA", "GS",
    "SBUX", "BLK", "ISRG", "MDLZ", "PLD", "DE",
]

BOND_ETFS: List[str] = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", "BNDX", "TIP",
    "VCSH", "VCIT", "VCLT", "MUB", "EMB", "JNK", "GOVT", "SCHO",
    "SCHR", "SCHZ", "IGIB", "IGSB", "USIG", "FLOT", "MINT",
]

COMMODITY_SYMBOLS: List[str] = [
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F",
    "LE=F", "HE=F",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "CPER",
    "PPLT", "PALL", "WEAT", "CORN", "SOYB", "JO", "NIB", "BAL",
    "IAU", "SGOL", "AAAU",
]

CRYPTO_SYMBOLS: List[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD", "LINK-USD",
    "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD", "XLM-USD",
    "ALGO-USD", "NEAR-USD", "FTM-USD", "AAVE-USD",
]

FX_MAJOR: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X",
]
FX_MINOR: List[str] = [
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDNZD=X", "EURCHF=X",
    "AUDCAD=X", "CADJPY=X", "NZDJPY=X", "GBPCHF=X", "EURAUD=X",
    "AUDCHF=X", "GBPAUD=X",
]
FX_EXOTIC: List[str] = [
    "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDBRL=X", "USDPLN=X",
    "USDSEK=X", "USDNOK=X", "USDDKK=X", "USDSGD=X", "USDHKD=X",
    "USDCNY=X", "USDINR=X", "USDTHB=X", "USDKRW=X", "USDTWD=X",
]
FX_ALL: List[str] = FX_MAJOR + FX_MINOR + FX_EXOTIC

INDEX_SYMBOLS: List[str] = [
    "^GSPC", "^DJI", "^IXIC", "^RUT",
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E",
    "^N225", "^HSI", "000001.SS", "^KS11",
    "^BSESN", "^NSEI",
    "^AXJO", "^NZ50",
    "^BVSP", "^MXX",
    "^GSPTSE",
    "^VIX",
]

ETF_CORE: List[str] = [
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "VEA", "VWO", "EFA",
    "EEM", "IEMG",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB",
    "XLRE", "XLC",
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
    "MTUM", "QUAL", "VLUE", "SIZE", "USMV",
    "FXI", "EWJ", "EWG", "EWU", "EWZ", "INDA",
]


# ═══════════════════════════════════════════════════════════════════════════
# GICS helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_gics_flat() -> dict:
    """Build flat lookup dicts from the nested GICS_SECTORS hierarchy."""
    sector_map: Dict[int, str] = {}
    group_map: Dict[int, str] = {}
    industry_map: Dict[int, str] = {}
    sub_industry_map: Dict[int, str] = {}

    for sector_code, sector_info in GICS_SECTORS.items():
        sector_map[sector_code] = sector_info["name"]
        for group_code, group_info in sector_info["groups"].items():
            group_map[group_code] = group_info["name"]
            for industry_code, industry_info in group_info["industries"].items():
                industry_map[industry_code] = industry_info["name"]
                for sub_code, sub_name in industry_info.get("sub_industries", {}).items():
                    sub_industry_map[sub_code] = sub_name

    return {
        "sector": sector_map,
        "group": group_map,
        "industry": industry_map,
        "sub_industry": sub_industry_map,
    }


_GICS_FLAT = _build_gics_flat()


def gics_sector_name(code: int) -> Optional[str]:
    return _GICS_FLAT["sector"].get(code)


def gics_industry_group_name(code: int) -> Optional[str]:
    return _GICS_FLAT["group"].get(code)


def gics_industry_name(code: int) -> Optional[str]:
    return _GICS_FLAT["industry"].get(code)


def gics_sub_industry_name(code: int) -> Optional[str]:
    return _GICS_FLAT["sub_industry"].get(code)


# ═══════════════════════════════════════════════════════════════════════════
# Asset-class detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_asset_class(symbol: str) -> AssetClass:
    """Auto-detect the asset class of a symbol based on its format."""
    s = symbol.upper().strip()

    if s.endswith("=X"):
        return AssetClass.FX
    if re.search(r"-USD$|-BTC$|-ETH$|-USDT$", s):
        return AssetClass.CRYPTO
    if s.startswith("^"):
        return AssetClass.INDEX
    if s.endswith("=F"):
        return AssetClass.COMMODITY
    if s in {c.upper() for c in COMMODITY_SYMBOLS}:
        return AssetClass.COMMODITY
    if s in {b.upper() for b in BOND_ETFS}:
        return AssetClass.FIXED_INCOME
    if s in {e.upper() for e in ETF_CORE}:
        return AssetClass.ETF
    return AssetClass.EQUITY


# ═══════════════════════════════════════════════════════════════════════════
# Core data functions — OpenBB as sole data source
# ═══════════════════════════════════════════════════════════════════════════

def _obb_result_to_df(result) -> pd.DataFrame:
    """Convert an OpenBB OBBject result to a pandas DataFrame."""
    if hasattr(result, "to_df"):
        return result.to_df()
    if hasattr(result, "results") and result.results is not None:
        return pd.DataFrame([r.model_dump() for r in result.results])
    return pd.DataFrame()


def get_full_universe(
    asset_classes: Optional[Sequence[AssetClass]] = None,
) -> Dict[AssetClass, List[str]]:
    """Return all tradeable symbols grouped by asset class."""
    full: Dict[AssetClass, List[str]] = {
        AssetClass.EQUITY: list(SP500_REPRESENTATIVE),
        AssetClass.FIXED_INCOME: list(BOND_ETFS),
        AssetClass.COMMODITY: list(COMMODITY_SYMBOLS),
        AssetClass.CRYPTO: list(CRYPTO_SYMBOLS),
        AssetClass.FX: list(FX_ALL),
        AssetClass.ETF: list(ETF_CORE),
        AssetClass.INDEX: list(INDEX_SYMBOLS),
    }

    try:
        obb = _get_obb()
        screener_result = obb.equity.screener(provider="fmp")
        df = _obb_result_to_df(screener_result)
        if not df.empty and "symbol" in df.columns:
            existing = set(full[AssetClass.EQUITY])
            for sym in df["symbol"].dropna().unique():
                if sym not in existing:
                    full[AssetClass.EQUITY].append(sym)
                    existing.add(sym)
    except Exception as exc:
        logger.warning("Could not augment equity universe from screener: %s", exc)

    if asset_classes is not None:
        requested = set(asset_classes)
        return {k: v for k, v in full.items() if k in requested}
    return full


def get_equity_universe() -> pd.DataFrame:
    """Return all equities with GICS classification columns."""
    obb = _get_obb()
    records: List[dict] = []
    batch_size = 20
    symbols = list(SP500_REPRESENTATIVE)

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            result = obb.equity.profile(symbol=",".join(batch), provider="fmp")
            df = _obb_result_to_df(result)
            if not df.empty:
                for _, row in df.iterrows():
                    record = {
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", row.get("company_name", "")),
                        "sector": row.get("sector", ""),
                        "industry_group": "",
                        "industry": row.get("industry", ""),
                        "sub_industry": "",
                        "market_cap": row.get("market_cap", row.get("mktCap", np.nan)),
                        "exchange": row.get("exchange", row.get("exchange_short_name", "")),
                        "country": row.get("country", ""),
                    }
                    for code, info in GICS_SECTORS.items():
                        if info["name"].lower() == str(record["sector"]).lower():
                            record["gics_sector_code"] = code
                            break
                    records.append(record)
        except Exception as exc:
            logger.warning("Profile fetch failed for batch %s: %s", batch[:3], exc)
            for sym in batch:
                records.append({"symbol": sym})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["symbol", "name", "sector", "industry_group", "industry",
                 "sub_industry", "market_cap", "exchange", "country"]
    )


def get_bond_universe() -> pd.DataFrame:
    """Return fixed-income universe: treasuries and corporates."""
    obb = _get_obb()
    frames: List[pd.DataFrame] = []

    try:
        result = obb.fixedincome.rate.treasury(provider="federal_reserve")
        df = _obb_result_to_df(result)
        if not df.empty:
            df["bond_type"] = "treasury"
            frames.append(df)
    except Exception as exc:
        logger.warning("Treasury rate fetch failed: %s", exc)

    try:
        result = obb.fixedincome.corporate.bond_prices(provider="tmx")
        df = _obb_result_to_df(result)
        if not df.empty:
            df["bond_type"] = "corporate"
            frames.append(df)
    except Exception as exc:
        logger.warning("Corporate bond fetch failed: %s", exc)

    try:
        for etf_sym in BOND_ETFS[:10]:
            result = obb.etf.price.historical(
                symbol=etf_sym,
                start_date=(date.today() - timedelta(days=5)).isoformat(),
                provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = etf_sym
                df["bond_type"] = "etf"
                frames.append(df.tail(1))
    except Exception as exc:
        logger.warning("Bond ETF fetch failed: %s", exc)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def get_commodity_universe() -> pd.DataFrame:
    """Return all commodity symbols with latest price data."""
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    for sym in COMMODITY_SYMBOLS:
        try:
            result = obb.commodity.price.historical(
                symbol=sym, start_date=start, provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                df["asset_class"] = "commodity"
                records.append(df.tail(1))
        except Exception:
            try:
                result = obb.equity.price.historical(
                    symbol=sym, start_date=start, provider="yfinance",
                )
                df = _obb_result_to_df(result)
                if not df.empty:
                    df["symbol"] = sym
                    df["asset_class"] = "commodity_etf"
                    records.append(df.tail(1))
            except Exception:
                pass

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def get_crypto_universe() -> pd.DataFrame:
    """Return top crypto assets with latest price data."""
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    for sym in CRYPTO_SYMBOLS:
        try:
            result = obb.crypto.price.historical(
                symbol=sym, start_date=start, provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                records.append(df.tail(1))
        except Exception as exc:
            logger.debug("Crypto fetch failed for %s: %s", sym, exc)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def get_fx_universe() -> pd.DataFrame:
    """Return FX universe: major, minor, and exotic pairs."""
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    pair_type_map = {}
    for s in FX_MAJOR:
        pair_type_map[s] = "major"
    for s in FX_MINOR:
        pair_type_map[s] = "minor"
    for s in FX_EXOTIC:
        pair_type_map[s] = "exotic"

    for sym in FX_ALL:
        try:
            result = obb.currency.price.historical(
                symbol=sym, start_date=start, provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                df["pair_type"] = pair_type_map.get(sym, "unknown")
                records.append(df.tail(1))
        except Exception as exc:
            logger.debug("FX fetch failed for %s: %s", sym, exc)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def classify_by_gics(symbols: List[str]) -> pd.DataFrame:
    """Map equity symbols to GICS classification."""
    obb = _get_obb()
    records: List[dict] = []

    sector_name_to_code: Dict[str, int] = {}
    for code, info in GICS_SECTORS.items():
        sector_name_to_code[info["name"].lower()] = code

    industry_name_lookup: Dict[str, tuple] = {}
    for sector_code, sector_info in GICS_SECTORS.items():
        for group_code, group_info in sector_info["groups"].items():
            for ind_code, ind_info in group_info["industries"].items():
                industry_name_lookup[ind_info["name"].lower()] = (
                    sector_code, group_code, ind_code
                )

    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            result = obb.equity.profile(symbol=",".join(batch), provider="fmp")
            df = _obb_result_to_df(result)
            if not df.empty:
                for _, row in df.iterrows():
                    sym = row.get("symbol", "")
                    sector_str = str(row.get("sector", "")).lower()
                    industry_str = str(row.get("industry", "")).lower()

                    record: Dict[str, Any] = {
                        "symbol": sym,
                        "sector": row.get("sector", ""),
                        "sector_code": sector_name_to_code.get(sector_str),
                        "industry_group": "",
                        "industry_group_code": None,
                        "industry": row.get("industry", ""),
                        "industry_code": None,
                        "sub_industry": "",
                        "sub_industry_code": None,
                    }

                    if industry_str in industry_name_lookup:
                        sc, gc, ic = industry_name_lookup[industry_str]
                        record["sector_code"] = sc
                        record["sector"] = GICS_SECTORS[sc]["name"]
                        record["industry_group_code"] = gc
                        record["industry_group"] = GICS_SECTORS[sc]["groups"][gc]["name"]
                        record["industry_code"] = ic
                        record["industry"] = GICS_SECTORS[sc]["groups"][gc]["industries"][ic]["name"]

                    records.append(record)
        except Exception as exc:
            logger.warning("GICS classification failed for batch %s: %s", batch[:3], exc)
            for sym in batch:
                records.append({"symbol": sym})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["symbol", "sector", "sector_code", "industry_group",
                 "industry_group_code", "industry", "industry_code",
                 "sub_industry", "sub_industry_code"]
    )


def get_historical(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    asset_class: Optional[AssetClass] = None,
    interval: str = "1d",
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch historical OHLCV data with smart routing to the correct OBB endpoint.

    Notes
    -----
    Sharpe Ratio: S = (R_p - R_f) / sigma_p
    """
    obb = _get_obb()

    if asset_class is None:
        asset_class = detect_asset_class(symbol)
    if end is None:
        end = date.today().isoformat()

    kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "start_date": start,
        "end_date": end,
        "provider": provider,
    }

    endpoint_map = {
        AssetClass.EQUITY: obb.equity.price.historical,
        AssetClass.ETF: obb.etf.price.historical,
        AssetClass.CRYPTO: obb.crypto.price.historical,
        AssetClass.FX: obb.currency.price.historical,
        AssetClass.INDEX: obb.index.price.historical,
        AssetClass.COMMODITY: obb.commodity.price.historical,
        AssetClass.FIXED_INCOME: obb.etf.price.historical,
    }

    endpoint = endpoint_map.get(asset_class, obb.equity.price.historical)
    result = endpoint(**kwargs)
    df = _obb_result_to_df(result)

    if not df.empty:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    return df


def get_multiple_historical(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    provider: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """Batch-fetch historical data for multiple symbols."""
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_historical(sym, start, end, interval=interval, provider=provider)
            if not df.empty:
                results[sym] = df
        except Exception as exc:
            logger.warning("Historical fetch failed for %s: %s", sym, exc)
    return results


def get_fundamentals(symbol: str, provider: str = "fmp") -> dict:
    """Fetch fundamental data: financials, ratios, profile."""
    obb = _get_obb()
    data: Dict[str, Any] = {}

    for key, fetcher in [
        ("profile", lambda: obb.equity.profile(symbol=symbol, provider=provider)),
        ("ratios", lambda: obb.equity.fundamental.ratios(symbol=symbol, provider=provider)),
        ("income", lambda: obb.equity.fundamental.income(symbol=symbol, provider=provider)),
        ("balance", lambda: obb.equity.fundamental.balance(symbol=symbol, provider=provider)),
        ("cash", lambda: obb.equity.fundamental.cash(symbol=symbol, provider=provider)),
    ]:
        try:
            data[key] = _obb_result_to_df(fetcher())
        except Exception as exc:
            logger.warning("%s fetch failed for %s: %s", key, symbol, exc)
            data[key] = pd.DataFrame()

    return data


def get_macro_data(provider: str = "federal_reserve") -> dict:
    """Fetch macroeconomic data: GDP, CPI, unemployment, treasury rates."""
    obb = _get_obb()
    data: Dict[str, pd.DataFrame] = {}

    fetchers = {
        "gdp": lambda: obb.economy.gdp.nominal(provider="oecd"),
        "cpi": lambda: obb.economy.cpi(provider="fred"),
        "unemployment": lambda: obb.economy.unemployment(provider="oecd"),
        "treasury_rates": lambda: obb.fixedincome.rate.treasury(provider="federal_reserve"),
    }

    for key, fetcher in fetchers.items():
        try:
            data[key] = _obb_result_to_df(fetcher())
        except Exception as exc:
            logger.warning("%s fetch failed: %s", key, exc)
            data[key] = pd.DataFrame()

    return data


def get_news_sentiment(
    symbol: Optional[str] = None,
    limit: int = 50,
    provider: str = "benzinga",
) -> pd.DataFrame:
    """Fetch news with sentiment scores."""
    obb = _get_obb()
    try:
        if symbol is not None:
            result = obb.news.company(symbol=symbol, limit=limit, provider=provider)
        else:
            result = obb.news.world(limit=limit, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return pd.DataFrame()


def get_options_chain(symbol: str, provider: str = "cboe") -> pd.DataFrame:
    """Fetch the full options chain for an equity symbol.

    Notes
    -----
    Black-Scholes: C = S*N(d1) - K*e^(-rT)*N(d2)
    """
    obb = _get_obb()
    try:
        result = obb.derivatives.options.chains(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Options chain fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def get_earnings_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch the earnings calendar."""
    obb = _get_obb()
    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() + timedelta(days=7)).isoformat()

    try:
        result = obb.equity.calendar.earnings(
            start_date=start_date, end_date=end_date, provider=provider,
        )
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Earnings calendar fetch failed: %s", exc)
        return pd.DataFrame()


def get_economic_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch the economic events calendar."""
    obb = _get_obb()
    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() + timedelta(days=7)).isoformat()

    try:
        result = obb.economy.calendar(
            start_date=start_date, end_date=end_date, provider=provider,
        )
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Economic calendar fetch failed: %s", exc)
        return pd.DataFrame()


def get_etf_holdings(symbol: str, provider: str = "fmp") -> pd.DataFrame:
    """Fetch the holdings of an ETF."""
    obb = _get_obb()
    try:
        result = obb.etf.holdings(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("ETF holdings fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def get_etf_info(symbol: str, provider: str = "fmp") -> pd.DataFrame:
    """Fetch ETF metadata."""
    obb = _get_obb()
    try:
        result = obb.etf.info(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("ETF info fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Analytics utilities
# ═══════════════════════════════════════════════════════════════════════════

def build_correlation_matrix(
    symbols: List[str], start: str, end: Optional[str] = None, field: str = "close",
) -> pd.DataFrame:
    """Build a correlation matrix of daily returns."""
    data = get_multiple_historical(symbols, start, end)
    if not data:
        return pd.DataFrame()
    closes = {sym: df[field] for sym, df in data.items() if field in df.columns}
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).pct_change().dropna().corr()


def build_covariance_matrix(
    symbols: List[str], start: str, end: Optional[str] = None,
    field: str = "close", annualise: bool = True,
) -> pd.DataFrame:
    """Build a covariance matrix of daily returns."""
    data = get_multiple_historical(symbols, start, end)
    if not data:
        return pd.DataFrame()
    closes = {sym: df[field] for sym, df in data.items() if field in df.columns}
    if not closes:
        return pd.DataFrame()
    cov = pd.DataFrame(closes).pct_change().dropna().cov()
    if annualise:
        cov = cov * 252
    return cov


def compute_var(
    returns: pd.Series, confidence: float = 0.95, method: str = "parametric",
) -> float:
    """Compute Value at Risk. VaR_alpha = mu - z_alpha * sigma"""
    from scipy import stats as scipy_stats
    if method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        z = scipy_stats.norm.ppf(1 - confidence)
        return -(mu + z * sigma)
    elif method == "historical":
        return -np.percentile(returns.dropna(), (1 - confidence) * 100)
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def compute_sharpe(
    returns: pd.Series, risk_free_rate: float = 0.0, annualise: bool = True,
) -> float:
    """Sharpe Ratio: S = (R_p - R_f) / sigma_p"""
    excess = returns - risk_free_rate
    if excess.std() == 0:
        return 0.0
    sharpe = excess.mean() / excess.std()
    if annualise:
        sharpe *= np.sqrt(252)
    return float(sharpe)


def compute_sortino(
    returns: pd.Series, risk_free_rate: float = 0.0, annualise: bool = True,
) -> float:
    """Sortino Ratio: S = (R_p - R_f) / sigma_d"""
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return float("inf")
    sortino = excess.mean() / downside_std
    if annualise:
        sortino *= np.sqrt(252)
    return float(sortino)


def compute_max_drawdown(prices: pd.Series) -> float:
    """MDD = max (peak - trough) / peak"""
    cummax = prices.cummax()
    drawdown = (cummax - prices) / cummax
    return float(drawdown.max())


def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly: f* = (b*p - q) / b"""
    q = 1.0 - win_prob
    return (win_loss_ratio * win_prob - q) / win_loss_ratio


# ═══════════════════════════════════════════════════════════════════════════
# QLIB Provider Interface Implementations
# ═══════════════════════════════════════════════════════════════════════════

# Import QLIB base classes (available when running inside the QLIB environment)
try:
    from qlib.data.data import CalendarProvider, InstrumentProvider, FeatureProvider
    _QLIB_AVAILABLE = True
except ImportError:
    _QLIB_AVAILABLE = False
    # Provide stub base classes so the module can still be imported standalone
    class CalendarProvider:  # type: ignore[no-redef]
        pass

    class InstrumentProvider:  # type: ignore[no-redef]
        @staticmethod
        def instruments(market="all", filter_pipe=None):
            if isinstance(market, list):
                return market
            return {"market": market, "filter_pipe": filter_pipe or []}

    class FeatureProvider:  # type: ignore[no-redef]
        pass


class OpenBBCalendarProvider(CalendarProvider):
    """QLIB CalendarProvider backed by OpenBB.

    Generates trading calendars by fetching historical data for a liquid
    reference instrument (SPY) and extracting the dates on which it traded.

    This replaces QLIB's default file-based LocalCalendarProvider.
    """

    # Internal cache: freq -> (np.array of timestamps, dict index)
    _cache: Dict[str, tuple] = {}

    def load_calendar(self, freq: str, future: bool = False) -> List[pd.Timestamp]:
        """Load trading calendar from OpenBB.

        Fetches SPY historical data going back ~10 years and extracts
        unique trading dates as the calendar.

        Parameters
        ----------
        freq : str
            Frequency: "day", "week", "month", etc.
        future : bool
            Whether to include future dates (ignored — OpenBB only returns
            historical data).

        Returns
        -------
        list[pd.Timestamp]
            Sorted list of trading-day timestamps.
        """
        cache_key = f"{freq}_future_{future}"
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        obb = _get_obb()
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=365 * 10)).isoformat()

        try:
            result = obb.equity.price.historical(
                symbol="SPY",
                start_date=start_date,
                end_date=end_date,
                provider="yfinance",
            )
            df = _obb_result_to_df(result)

            if df.empty:
                logger.warning("Empty calendar result from OpenBB; returning empty list")
                return []

            # Extract date index
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            else:
                dates = pd.to_datetime(df.index)

            dates = dates.normalize().unique().sort_values()

            # Resample if needed
            if freq == "week":
                dates = dates.to_series().resample("W-FRI").last().dropna().index
                dates = pd.DatetimeIndex(dates)
            elif freq == "month":
                dates = dates.to_series().resample("ME").last().dropna().index
                dates = pd.DatetimeIndex(dates)

            cal = [pd.Timestamp(d) for d in dates]
            self._cache[cache_key] = tuple(cal)
            return cal

        except Exception as exc:
            logger.error("Failed to load calendar from OpenBB: %s", exc)
            return []

    def _get_calendar(self, freq: str, future: bool):
        """Override to use our load_calendar with caching."""
        flag = f"{freq}_future_{future}"
        if flag not in self._cache:
            _calendar = np.array(self.load_calendar(freq, future))
            self._cache[flag] = _calendar
        _calendar = self._cache[flag]
        if isinstance(_calendar, tuple):
            _calendar = np.array(_calendar)
            self._cache[flag] = _calendar
        _calendar_index = {x: i for i, x in enumerate(_calendar)}
        return _calendar, _calendar_index

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        """Get calendar of trading days in the given range."""
        _calendar, _calendar_index = self._get_calendar(freq, future)

        if len(_calendar) == 0:
            return np.array([])

        if start_time == "None":
            start_time = None
        if end_time == "None":
            end_time = None

        if start_time:
            start_time = pd.Timestamp(start_time)
            if start_time > _calendar[-1]:
                return np.array([])
        else:
            start_time = _calendar[0]

        if end_time:
            end_time = pd.Timestamp(end_time)
            if end_time < _calendar[0]:
                return np.array([])
        else:
            end_time = _calendar[-1]

        si = bisect.bisect_left(_calendar, start_time)
        ei = bisect.bisect_right(_calendar, end_time)
        return _calendar[si:ei]

    def locate_index(self, start_time, end_time, freq, future=False):
        """Locate start/end indices in the calendar."""
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        calendar, calendar_index = self._get_calendar(freq=freq, future=future)

        if start_time not in calendar_index:
            idx = bisect.bisect_left(calendar, start_time)
            if idx >= len(calendar):
                raise IndexError(
                    "start_time uses a future date; use future=True for future trading days"
                )
            start_time = calendar[idx]
        start_index = calendar_index[start_time]

        if end_time not in calendar_index:
            end_time = calendar[bisect.bisect_right(calendar, end_time) - 1]
        end_index = calendar_index[end_time]

        return start_time, end_time, start_index, end_index


class OpenBBInstrumentProvider(InstrumentProvider):
    """QLIB InstrumentProvider backed by OpenBB.

    Lists available instruments (stock symbols) from the OpenBB universe.
    Supports market filters like "sp500", "etf", "crypto", "all".
    """

    # Mapping of market names to our universe lists
    _MARKET_MAP: Dict[str, List[str]] = {
        "all": SP500_REPRESENTATIVE + ETF_CORE,
        "sp500": SP500_REPRESENTATIVE,
        "etf": ETF_CORE,
        "bond": BOND_ETFS,
        "commodity": [s for s in COMMODITY_SYMBOLS if not s.endswith("=F")],
        "crypto": CRYPTO_SYMBOLS,
        "index": INDEX_SYMBOLS,
    }

    def list_instruments(
        self,
        instruments,
        start_time=None,
        end_time=None,
        freq="day",
        as_list=False,
    ):
        """List instruments based on a stock-pool config or list.

        Parameters
        ----------
        instruments : dict or list
            If dict, expects {"market": "sp500", "filter_pipe": [...]}.
            If list, returns as-is.
        start_time : str, optional
            Start of the time range.
        end_time : str, optional
            End of the time range.
        freq : str
            Time frequency.
        as_list : bool
            If True, return a flat list; otherwise return a dict with
            (start, end) tuples per instrument.

        Returns
        -------
        dict or list
            Instruments with trading spans or as a flat list.
        """
        if isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
            if as_list:
                return list(instruments)
            st = pd.Timestamp(start_time) if start_time else pd.Timestamp("2000-01-01")
            et = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()
            return {inst: (st, et) for inst in instruments}

        market = instruments.get("market", "all") if isinstance(instruments, dict) else "all"
        market_lower = market.lower()

        symbol_list = self._MARKET_MAP.get(market_lower, SP500_REPRESENTATIVE)

        # Attempt to augment from OpenBB screener for equity markets
        if market_lower in ("all", "sp500"):
            try:
                obb = _get_obb()
                screener_result = obb.equity.screener(provider="fmp")
                df = _obb_result_to_df(screener_result)
                if not df.empty and "symbol" in df.columns:
                    existing = set(symbol_list)
                    for sym in df["symbol"].dropna().unique():
                        if sym not in existing:
                            symbol_list.append(sym)
                            existing.add(sym)
            except Exception:
                pass

        if as_list:
            return list(symbol_list)

        st = pd.Timestamp(start_time) if start_time else pd.Timestamp("2000-01-01")
        et = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()
        return {inst: (st, et) for inst in symbol_list}


class OpenBBFeatureProvider(FeatureProvider):
    """QLIB FeatureProvider backed by OpenBB.

    Provides OHLCV feature data for any instrument via OpenBB's historical
    price endpoints.  Supports the standard QLIB field names:
      $open, $high, $low, $close, $volume, $vwap, $adj_close

    This replaces QLIB's default file-based LocalFeatureProvider.
    """

    # Map QLIB field names to DataFrame column names from OpenBB
    _FIELD_MAP: Dict[str, str] = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$vwap": "vwap",
        "$adj_close": "adj_close",
        # Also support bare names
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    # Internal cache: (instrument, freq) -> pd.DataFrame
    _cache: Dict[tuple, pd.DataFrame] = {}

    def _fetch_data(
        self, instrument: str, start_time: str, end_time: str, freq: str,
    ) -> pd.DataFrame:
        """Fetch and cache OHLCV data for an instrument."""
        cache_key = (instrument, freq)

        if cache_key in self._cache:
            df = self._cache[cache_key]
            # Check if cached data covers the requested range
            if not df.empty:
                cached_start = df.index.min()
                cached_end = df.index.max()
                req_start = pd.Timestamp(start_time)
                req_end = pd.Timestamp(end_time)
                if cached_start <= req_start and cached_end >= req_end:
                    return df.loc[req_start:req_end]

        # Fetch from OpenBB
        interval = "1d" if freq == "day" else freq
        try:
            df = get_historical(
                symbol=instrument,
                start=start_time,
                end=end_time,
                interval=interval,
            )
            if not df.empty:
                self._cache[cache_key] = df
            return df
        except Exception as exc:
            logger.warning("Feature fetch failed for %s: %s", instrument, exc)
            return pd.DataFrame()

    def feature(
        self,
        instrument: str,
        field: str,
        start_time: str,
        end_time: str,
        freq: str,
    ) -> pd.Series:
        """Get feature data for a single instrument and field.

        Parameters
        ----------
        instrument : str
            Ticker symbol (e.g. "AAPL").
        field : str
            Feature field name (e.g. "$close", "$volume").
        start_time : str
            Start date.
        end_time : str
            End date.
        freq : str
            Frequency ("day", "week", etc.).

        Returns
        -------
        pd.Series
            Time series of the requested feature, indexed by date.
        """
        col_name = self._FIELD_MAP.get(field, field.lstrip("$"))

        df = self._fetch_data(instrument, start_time, end_time, freq)

        if df.empty:
            return pd.Series(dtype=float, name=field)

        if col_name in df.columns:
            series = df[col_name].copy()
            series.name = field
            return series

        # If exact column not found, try case-insensitive match
        col_lower = {c.lower(): c for c in df.columns}
        if col_name.lower() in col_lower:
            series = df[col_lower[col_name.lower()]].copy()
            series.name = field
            return series

        logger.warning(
            "Field '%s' (mapped to '%s') not found in data for %s. "
            "Available columns: %s",
            field, col_name, instrument, list(df.columns),
        )
        return pd.Series(dtype=float, name=field)

    def clear_cache(self):
        """Clear the internal data cache."""
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# QLIB registration helper
# ═══════════════════════════════════════════════════════════════════════════

def register_openbb_providers():
    """Register OpenBB providers with QLIB's global provider registry.

    Call this after ``qlib.init()`` to replace the default local providers
    with OpenBB-backed ones.

    Example
    -------
        import qlib
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
        from qlib.data.openbb_universe import register_openbb_providers
        register_openbb_providers()
    """
    if not _QLIB_AVAILABLE:
        logger.warning(
            "QLIB is not installed; cannot register OpenBB providers. "
            "The module can still be used standalone."
        )
        return

    try:
        from qlib.data.data import Cal, Inst, FeatureD

        # Replace the global provider singletons
        Cal._provider = OpenBBCalendarProvider()
        Inst._provider = OpenBBInstrumentProvider()
        FeatureD._provider = OpenBBFeatureProvider()

        logger.info("OpenBB providers registered with QLIB successfully.")
    except Exception as exc:
        logger.error("Failed to register OpenBB providers with QLIB: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Module exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums / constants
    "AssetClass",
    "GICS_SECTORS",
    "SP500_REPRESENTATIVE",
    "BOND_ETFS",
    "COMMODITY_SYMBOLS",
    "CRYPTO_SYMBOLS",
    "FX_MAJOR",
    "FX_MINOR",
    "FX_EXOTIC",
    "FX_ALL",
    "INDEX_SYMBOLS",
    "ETF_CORE",
    # GICS helpers
    "gics_sector_name",
    "gics_industry_group_name",
    "gics_industry_name",
    "gics_sub_industry_name",
    # Detection
    "detect_asset_class",
    # Universe functions
    "get_full_universe",
    "get_equity_universe",
    "get_bond_universe",
    "get_commodity_universe",
    "get_crypto_universe",
    "get_fx_universe",
    "classify_by_gics",
    # Historical data
    "get_historical",
    "get_multiple_historical",
    # Fundamentals
    "get_fundamentals",
    # Macro
    "get_macro_data",
    # News
    "get_news_sentiment",
    # Derivatives
    "get_options_chain",
    # Calendars
    "get_earnings_calendar",
    "get_economic_calendar",
    # ETF
    "get_etf_holdings",
    "get_etf_info",
    # Analytics
    "build_correlation_matrix",
    "build_covariance_matrix",
    "compute_var",
    "compute_sharpe",
    "compute_sortino",
    "compute_max_drawdown",
    "kelly_criterion",
    # QLIB providers
    "OpenBBCalendarProvider",
    "OpenBBInstrumentProvider",
    "OpenBBFeatureProvider",
    "register_openbb_providers",
]
