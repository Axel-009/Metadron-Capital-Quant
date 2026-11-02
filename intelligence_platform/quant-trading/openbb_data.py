# -*- coding: utf-8 -*-
"""
Unified OpenBB Data Module for quant-trading backtests.

Uses OpenBB as the SOLE data source (no yfinance fallback).

Provides:
    - Full universe fetching across all asset classes (equities, bonds,
      commodities, crypto, FX)
    - GICS classification for equities (all 11 sectors, 25 industry groups,
      74 industries, 163 sub-industries)
    - Asset class enum and auto-detection
    - Historical data, fundamentals, macro data, and news sentiment

Mathematical Reference
======================
Sharpe Ratio:
    S = (R_p - R_f) / sigma_p

Sortino Ratio:
    So = (R_p - R_f) / sigma_d
    where sigma_d = sqrt( (1/N) * sum( min(R_i - R_f, 0)^2 ) )

Kelly Criterion:
    f* = (b*p - q) / b    (Half-Kelly: f = f*/2)

Black-Scholes (European call):
    C = S*N(d1) - K*e^{-rT}*N(d2)
    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Value at Risk (parametric):
    VaR_{alpha} = mu + z_{alpha} * sigma

Conditional VaR (Expected Shortfall):
    CVaR_{alpha} = mu - sigma * phi(z_{alpha}) / alpha

Usage:
    from openbb_data import get_historical, get_multiple, get_full_universe
    from openbb_data import AssetClass, classify_by_gics, detect_asset_class

    df = get_historical('AAPL', '2020-01-01', '2023-01-01')
    dfs = get_multiple(['AAPL', 'MSFT'], '2020-01-01', '2023-01-01')
    universe = get_full_universe('2023-01-01', '2024-01-01')
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenBB SDK import (sole data source)
# ---------------------------------------------------------------------------
try:
    from openbb import obb

    _HAS_OPENBB = True
except ImportError:
    obb = None  # type: ignore[assignment]
    _HAS_OPENBB = False
    logger.warning("OpenBB SDK not installed. Install with: pip install openbb")


# ===================================================================
# Asset Class Enum
# ===================================================================
class AssetClass(Enum):
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    INDEX = "index"
    UNKNOWN = "unknown"


# ===================================================================
# GICS Classification
# ===================================================================
GICS_SECTORS: Dict[int, str] = {
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

GICS_INDUSTRY_GROUPS: Dict[int, Tuple[str, int]] = {
    1010: ("Energy", 10),
    1510: ("Materials", 15),
    2010: ("Capital Goods", 20),
    2020: ("Commercial & Professional Services", 20),
    2030: ("Transportation", 20),
    2510: ("Automobiles & Components", 25),
    2520: ("Consumer Durables & Apparel", 25),
    2530: ("Consumer Services", 25),
    2550: ("Retailing", 25),
    2560: ("Consumer Discretionary Distribution & Retail", 25),
    3010: ("Food & Staples Retailing", 30),
    3020: ("Food, Beverage & Tobacco", 30),
    3030: ("Household & Personal Products", 30),
    3510: ("Health Care Equipment & Services", 35),
    3520: ("Pharmaceuticals, Biotechnology & Life Sciences", 35),
    4010: ("Banks", 40),
    4020: ("Diversified Financials", 40),
    4030: ("Insurance", 40),
    4510: ("Software & Services", 45),
    4520: ("Technology Hardware & Equipment", 45),
    4530: ("Semiconductors & Semiconductor Equipment", 45),
    5010: ("Telecommunication Services", 50),
    5020: ("Media & Entertainment", 50),
    5510: ("Utilities", 55),
    6010: ("Equity Real Estate Investment Trusts (REITs)", 60),
    6020: ("Real Estate Management & Development", 60),
}

GICS_INDUSTRIES: Dict[int, Tuple[str, int]] = {
    101010: ("Energy Equipment & Services", 1010),
    101020: ("Oil, Gas & Consumable Fuels", 1010),
    151010: ("Chemicals", 1510),
    151020: ("Construction Materials", 1510),
    151030: ("Containers & Packaging", 1510),
    151040: ("Metals & Mining", 1510),
    151050: ("Paper & Forest Products", 1510),
    201010: ("Aerospace & Defense", 2010),
    201020: ("Building Products", 2010),
    201030: ("Construction & Engineering", 2010),
    201040: ("Electrical Equipment", 2010),
    201050: ("Industrial Conglomerates", 2010),
    201060: ("Machinery", 2010),
    201070: ("Trading Companies & Distributors", 2010),
    202010: ("Commercial Services & Supplies", 2020),
    202020: ("Professional Services", 2020),
    203010: ("Air Freight & Logistics", 2030),
    203020: ("Passenger Airlines", 2030),
    203030: ("Marine Transportation", 2030),
    203040: ("Ground Transportation", 2030),
    203050: ("Transportation Infrastructure", 2030),
    251010: ("Automobile Components", 2510),
    251020: ("Automobiles", 2510),
    252010: ("Household Durables", 2520),
    252020: ("Leisure Products", 2520),
    252030: ("Textiles, Apparel & Luxury Goods", 2520),
    253010: ("Hotels, Restaurants & Leisure", 2530),
    253020: ("Diversified Consumer Services", 2530),
    255010: ("Distributors", 2560),
    255020: ("Internet & Direct Marketing Retail", 2560),
    255030: ("Broadline Retail", 2560),
    255040: ("Specialty Retail", 2560),
    301010: ("Consumer Staples Distribution & Retail", 3010),
    302010: ("Beverages", 3020),
    302020: ("Food Products", 3020),
    302030: ("Tobacco", 3020),
    303010: ("Household Products", 3030),
    303020: ("Personal Care Products", 3030),
    351010: ("Health Care Equipment & Supplies", 3510),
    351020: ("Health Care Providers & Services", 3510),
    351030: ("Health Care Technology", 3510),
    352010: ("Biotechnology", 3520),
    352020: ("Pharmaceuticals", 3520),
    352030: ("Life Sciences Tools & Services", 3520),
    401010: ("Banks", 4010),
    401020: ("Thrifts & Mortgage Finance", 4010),
    402010: ("Diversified Financial Services", 4020),
    402020: ("Consumer Finance", 4020),
    402030: ("Capital Markets", 4020),
    402040: ("Mortgage Real Estate Investment Trusts (REITs)", 4020),
    403010: ("Insurance", 4030),
    451010: ("Internet Software & Services", 4510),
    451020: ("IT Services", 4510),
    451030: ("Software", 4510),
    452010: ("Communications Equipment", 4520),
    452020: ("Technology Hardware, Storage & Peripherals", 4520),
    452030: ("Electronic Equipment, Instruments & Components", 4520),
    453010: ("Semiconductors & Semiconductor Equipment", 4530),
    501010: ("Diversified Telecommunication Services", 5010),
    501020: ("Wireless Telecommunication Services", 5010),
    502010: ("Media", 5020),
    502020: ("Entertainment", 5020),
    502030: ("Interactive Media & Services", 5020),
    551010: ("Electric Utilities", 5510),
    551020: ("Gas Utilities", 5510),
    551030: ("Multi-Utilities", 5510),
    551040: ("Water Utilities", 5510),
    551050: ("Independent Power and Renewable Electricity Producers", 5510),
    601010: ("Diversified REITs", 6010),
    601025: ("Industrial REITs", 6010),
    601030: ("Hotel & Resort REITs", 6010),
    601040: ("Office REITs", 6010),
    601050: ("Health Care REITs", 6010),
    601060: ("Residential REITs", 6010),
    601070: ("Retail REITs", 6010),
    601080: ("Specialized REITs", 6010),
    602010: ("Real Estate Management & Development", 6020),
}

GICS_SUB_INDUSTRIES: Dict[int, Tuple[str, int]] = {
    10101010: ("Oil & Gas Drilling", 101010),
    10101020: ("Oil & Gas Equipment & Services", 101010),
    10102010: ("Integrated Oil & Gas", 101020),
    10102020: ("Oil & Gas Exploration & Production", 101020),
    10102030: ("Oil & Gas Refining & Marketing", 101020),
    10102040: ("Oil & Gas Storage & Transportation", 101020),
    10102050: ("Coal & Consumable Fuels", 101020),
    15101010: ("Commodity Chemicals", 151010),
    15101020: ("Diversified Chemicals", 151010),
    15101030: ("Fertilizers & Agricultural Chemicals", 151010),
    15101040: ("Industrial Gases", 151010),
    15101050: ("Specialty Chemicals", 151010),
    15102010: ("Construction Materials", 151020),
    15103010: ("Metal, Glass & Plastic Containers", 151030),
    15103020: ("Paper Packaging", 151030),
    15104010: ("Aluminum", 151040),
    15104020: ("Diversified Metals & Mining", 151040),
    15104025: ("Copper", 151040),
    15104030: ("Gold", 151040),
    15104040: ("Precious Metals & Minerals", 151040),
    15104045: ("Silver", 151040),
    15104050: ("Steel", 151040),
    15105010: ("Forest Products", 151050),
    15105020: ("Paper Products", 151050),
    20101010: ("Aerospace & Defense", 201010),
    20102010: ("Building Products", 201020),
    20103010: ("Construction & Engineering", 201030),
    20104010: ("Electrical Components & Equipment", 201040),
    20104020: ("Heavy Electrical Equipment", 201040),
    20105010: ("Industrial Conglomerates", 201050),
    20106010: ("Construction Machinery & Heavy Transportation Equipment", 201060),
    20106015: ("Agricultural & Farm Machinery", 201060),
    20106020: ("Industrial Machinery & Supplies & Components", 201060),
    20107010: ("Trading Companies & Distributors", 201070),
    20201010: ("Commercial Printing", 202010),
    20201050: ("Environmental & Facilities Services", 202010),
    20201060: ("Office Services & Supplies", 202010),
    20201070: ("Diversified Support Services", 202010),
    20201080: ("Security & Alarm Services", 202010),
    20202010: ("Human Resource & Employment Services", 202020),
    20202020: ("Research & Consulting Services", 202020),
    20202030: ("Data Processing & Outsourced Services", 202020),
    20301010: ("Air Freight & Logistics", 203010),
    20302010: ("Passenger Airlines", 203020),
    20303010: ("Marine Transportation", 203030),
    20304010: ("Rail Transportation", 203040),
    20304020: ("Trucking", 203040),
    20305010: ("Airport Services", 203050),
    20305020: ("Highways & Railtracks", 203050),
    20305030: ("Marine Ports & Services", 203050),
    25101010: ("Automobile Components", 251010),
    25102010: ("Automobile Manufacturers", 251020),
    25102020: ("Motorcycle Manufacturers", 251020),
    25201010: ("Consumer Electronics", 252010),
    25201020: ("Home Furnishings", 252010),
    25201030: ("Homebuilding", 252010),
    25201040: ("Household Appliances", 252010),
    25201050: ("Housewares & Specialties", 252010),
    25202010: ("Leisure Products", 252020),
    25203010: ("Apparel, Accessories & Luxury Goods", 252030),
    25203020: ("Footwear", 252030),
    25203030: ("Textiles", 252030),
    25301010: ("Casinos & Gaming", 253010),
    25301020: ("Hotels, Resorts & Cruise Lines", 253010),
    25301030: ("Leisure Facilities", 253010),
    25301040: ("Restaurants", 253010),
    25302010: ("Education Services", 253020),
    25302020: ("Specialized Consumer Services", 253020),
    25501010: ("Distributors", 255010),
    25502020: ("Internet & Direct Marketing Retail", 255020),
    25503010: ("Broadline Retail", 255030),
    25504010: ("Apparel Retail", 255040),
    25504020: ("Computer & Electronics Retail", 255040),
    25504030: ("Home Improvement Retail", 255040),
    25504040: ("Other Specialty Retail", 255040),
    25504050: ("Automotive Retail", 255040),
    25504060: ("Homefurnishing Retail", 255040),
    30101010: ("Drug Retail", 301010),
    30101020: ("Food Distributors", 301010),
    30101030: ("Food Retail", 301010),
    30101040: ("Consumer Staples Merchandise Retail", 301010),
    30201010: ("Brewers", 302010),
    30201020: ("Distillers & Vintners", 302010),
    30201030: ("Soft Drinks & Non-alcoholic Beverages", 302010),
    30202010: ("Agricultural Products & Services", 302020),
    30202030: ("Packaged Foods & Meats", 302020),
    30203010: ("Tobacco", 302030),
    30301010: ("Household Products", 303010),
    30302010: ("Personal Care Products", 303020),
    35101010: ("Health Care Equipment", 351010),
    35101020: ("Health Care Supplies", 351010),
    35102010: ("Health Care Distributors", 351020),
    35102015: ("Health Care Services", 351020),
    35102020: ("Health Care Facilities", 351020),
    35102030: ("Managed Health Care", 351020),
    35103010: ("Health Care Technology", 351030),
    35201010: ("Biotechnology", 352010),
    35202010: ("Pharmaceuticals", 352020),
    35203010: ("Life Sciences Tools & Services", 352030),
    40101010: ("Diversified Banks", 401010),
    40101015: ("Regional Banks", 401010),
    40102010: ("Thrifts & Mortgage Finance", 401020),
    40201020: ("Other Diversified Financial Services", 402010),
    40201030: ("Multi-Sector Holdings", 402010),
    40201040: ("Specialized Finance", 402010),
    40202010: ("Consumer Finance", 402020),
    40203010: ("Asset Management & Custody Banks", 402030),
    40203020: ("Investment Banking & Brokerage", 402030),
    40203030: ("Diversified Capital Markets", 402030),
    40203040: ("Financial Exchanges & Data", 402030),
    40204010: ("Mortgage REITs", 402040),
    40301010: ("Insurance Brokers", 403010),
    40301020: ("Life & Health Insurance", 403010),
    40301030: ("Multi-line Insurance", 403010),
    40301040: ("Property & Casualty Insurance", 403010),
    40301050: ("Reinsurance", 403010),
    45101010: ("Internet Services & Infrastructure", 451010),
    45102010: ("IT Consulting & Other Services", 451020),
    45102020: ("Internet Services & Infrastructure", 451020),
    45102030: ("Application Software", 451030),
    45103010: ("Application Software", 451030),
    45103020: ("Systems Software", 451030),
    45201020: ("Communications Equipment", 452010),
    45202010: ("Technology Hardware, Storage & Peripherals", 452020),
    45203010: ("Electronic Equipment & Instruments", 452030),
    45203015: ("Electronic Components", 452030),
    45203020: ("Electronic Manufacturing Services", 452030),
    45203030: ("Technology Distributors", 452030),
    45301010: ("Semiconductor Materials & Equipment", 453010),
    45301020: ("Semiconductors", 453010),
    50101010: ("Alternative Carriers", 501010),
    50101020: ("Integrated Telecommunication Services", 501010),
    50102010: ("Wireless Telecommunication Services", 501020),
    50201010: ("Advertising", 502010),
    50201020: ("Broadcasting", 502010),
    50201030: ("Cable & Satellite", 502010),
    50201040: ("Publishing", 502010),
    50202010: ("Movies & Entertainment", 502020),
    50202020: ("Interactive Home Entertainment", 502020),
    50203010: ("Interactive Media & Services", 502030),
    55101010: ("Electric Utilities", 551010),
    55102010: ("Gas Utilities", 551020),
    55103010: ("Multi-Utilities", 551030),
    55104010: ("Water Utilities", 551040),
    55105010: ("Independent Power Producers & Energy Traders", 551050),
    55105020: ("Renewable Electricity", 551050),
    60101010: ("Diversified REITs", 601010),
    60102510: ("Industrial REITs", 601025),
    60103010: ("Hotel & Resort REITs", 601030),
    60104010: ("Office REITs", 601040),
    60105010: ("Health Care REITs", 601050),
    60106010: ("Multi-Family Residential REITs", 601060),
    60106020: ("Single-Family Residential REITs", 601060),
    60107010: ("Retail REITs", 601070),
    60108010: ("Diversified Real Estate Activities", 601080),
    60108020: ("Real Estate Operating Companies", 601080),
    60108030: ("Real Estate Development", 601080),
    60108040: ("Real Estate Services", 601080),
    60201010: ("Real Estate Operating Companies", 602010),
    60201020: ("Real Estate Development", 602010),
    60201030: ("Real Estate Services", 602010),
    60201040: ("Diversified Real Estate Activities", 602010),
}


# ===================================================================
# Default Universe Definitions
# ===================================================================
DEFAULT_EQUITY_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "INTC", "CSCO",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "DHR",
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA",
    "GE", "CAT", "RTX", "HON", "UPS", "BA", "DE", "LMT", "UNP", "ADP",
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
    "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "WELL", "DLR", "SPG", "VICI",
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "VMC", "MLM", "NUE",
]

DEFAULT_BOND_SYMBOLS: List[str] = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "BND",
    "MUB", "EMB", "BNDX", "GOVT", "VCSH", "VCLT", "IGIB",
]

DEFAULT_COMMODITY_SYMBOLS: List[str] = [
    "GLD", "SLV", "USO", "BNO", "UNG", "PPLT", "PALL", "CPER",
    "DBA", "DBC", "WEAT", "CORN", "SOYB", "URA", "WOOD",
]

DEFAULT_CRYPTO_SYMBOLS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "UNI-USD", "ATOM-USD", "LTC-USD", "NEAR-USD", "APT-USD",
]

DEFAULT_FX_PAIRS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "USDMXN=X", "USDBRL=X", "USDCNY=X", "USDINR=X", "USDTRY=X",
]


# ===================================================================
# Auto-detection of Asset Class
# ===================================================================
def detect_asset_class(symbol: str) -> AssetClass:
    """
    Automatically detect the asset class of a given symbol.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    AssetClass
    """
    s = symbol.upper().strip()
    if s.endswith("-USD") or s.endswith("/USD") or s.endswith("USDT"):
        return AssetClass.CRYPTO
    if s.endswith("=X") or s in {p.upper() for p in DEFAULT_FX_PAIRS}:
        return AssetClass.FX
    if s in {b.upper() for b in DEFAULT_BOND_SYMBOLS}:
        return AssetClass.BOND
    if s in {c.upper() for c in DEFAULT_COMMODITY_SYMBOLS}:
        return AssetClass.COMMODITY
    if s in {e.upper() for e in DEFAULT_EQUITY_SYMBOLS}:
        return AssetClass.EQUITY
    if s.isalpha() and 1 <= len(s) <= 5:
        return AssetClass.EQUITY
    return AssetClass.UNKNOWN


# ===================================================================
# GICS Classification
# ===================================================================
_TICKER_GICS_MAP: Dict[str, int] = {
    "AAPL": 45202010, "MSFT": 45103020, "NVDA": 45301020, "AVGO": 45301020,
    "ORCL": 45103020, "CRM": 45103010, "AMD": 45301020, "ADBE": 45103010,
    "INTC": 45301020, "CSCO": 45201020, "UNH": 35102030, "JNJ": 35202010,
    "LLY": 35202010, "ABBV": 35201010, "MRK": 35202010, "TMO": 35203010,
    "ABT": 35101010, "PFE": 35202010, "AMGN": 35201010, "DHR": 35203010,
    "JPM": 40101010, "V": 40201040, "MA": 40201040, "BAC": 40101010,
    "WFC": 40101010, "GS": 40203020, "MS": 40203020, "BLK": 40203010,
    "SCHW": 40203010, "AXP": 40202010, "AMZN": 25502020, "TSLA": 25102010,
    "HD": 25504030, "MCD": 25301040, "NKE": 25203010, "LOW": 25504030,
    "SBUX": 25301040, "TJX": 25504010, "BKNG": 25301020, "CMG": 25301040,
    "GOOGL": 50203010, "META": 50203010, "NFLX": 50202010, "DIS": 50202010,
    "CMCSA": 50201030, "TMUS": 50102010, "VZ": 50101020, "T": 50101020,
    "CHTR": 50201030, "EA": 50202020, "GE": 20104010, "CAT": 20106010,
    "RTX": 20101010, "HON": 20105010, "UPS": 20301010, "BA": 20101010,
    "DE": 20106015, "LMT": 20101010, "UNP": 20304010, "ADP": 45102010,
    "PG": 30301010, "KO": 30201030, "PEP": 30201030, "COST": 30101040,
    "WMT": 30101040, "PM": 30203010, "MO": 30203010, "CL": 30301010,
    "MDLZ": 30202030, "KHC": 30202030, "XOM": 10102010, "CVX": 10102010,
    "COP": 10102020, "SLB": 10101020, "EOG": 10102020, "MPC": 10102030,
    "PSX": 10102030, "VLO": 10102030, "OXY": 10102020, "HAL": 10101020,
    "NEE": 55101010, "DUK": 55101010, "SO": 55101010, "D": 55103010,
    "AEP": 55101010, "SRE": 55103010, "EXC": 55101010, "XEL": 55101010,
    "ED": 55101010, "WEC": 55103010, "PLD": 60102510, "AMT": 60108010,
    "CCI": 60108010, "EQIX": 60108010, "PSA": 60108010, "O": 60107010,
    "WELL": 60105010, "DLR": 60108010, "SPG": 60107010, "VICI": 60108010,
    "LIN": 15101040, "APD": 15101040, "SHW": 15101050, "ECL": 15101050,
    "DD": 15101020, "NEM": 15104030, "FCX": 15104025, "VMC": 15102010,
    "MLM": 15102010, "NUE": 15104050,
}


@dataclass
class GICSClassification:
    """Full GICS classification for a single equity."""
    symbol: str
    sub_industry_code: int
    sub_industry_name: str
    industry_code: int
    industry_name: str
    industry_group_code: int
    industry_group_name: str
    sector_code: int
    sector_name: str


def classify_by_gics(symbol: str) -> Optional[GICSClassification]:
    """
    Return the full GICS classification for an equity symbol.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    GICSClassification or None
    """
    sub_code = _TICKER_GICS_MAP.get(symbol.upper())
    if sub_code is None:
        return None

    sub_name, ind_code = GICS_SUB_INDUSTRIES.get(sub_code, ("Unknown", sub_code // 100))
    ind_name, ig_code = GICS_INDUSTRIES.get(ind_code, ("Unknown", ind_code // 10))
    ig_name, sec_code = GICS_INDUSTRY_GROUPS.get(ig_code, ("Unknown", ig_code // 100))
    sec_name = GICS_SECTORS.get(sec_code, "Unknown")

    return GICSClassification(
        symbol=symbol.upper(),
        sub_industry_code=sub_code,
        sub_industry_name=sub_name,
        industry_code=ind_code,
        industry_name=ind_name,
        industry_group_code=ig_code,
        industry_group_name=ig_name,
        sector_code=sec_code,
        sector_name=sec_name,
    )


def classify_universe_by_gics(
    symbols: Optional[List[str]] = None,
) -> Dict[str, List[GICSClassification]]:
    """Classify symbols by GICS sector."""
    if symbols is None:
        symbols = DEFAULT_EQUITY_SYMBOLS
    result: Dict[str, List[GICSClassification]] = {}
    for sym in symbols:
        cls = classify_by_gics(sym)
        if cls is not None:
            result.setdefault(cls.sector_name, []).append(cls)
    return result


# ===================================================================
# Data-Fetching Functions (OpenBB ONLY -- no yfinance)
# ===================================================================
def _ensure_openbb() -> None:
    if not _HAS_OPENBB:
        raise ImportError(
            "OpenBB SDK is required but not installed. Install with: pip install openbb"
        )


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame to standard column names and DatetimeIndex."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        low = col.lower().replace(" ", "_")
        if low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"
        elif low in ("adj_close", "adjusted_close"):
            rename_map[col] = "Adj Close"
    if rename_map:
        df = df.rename(columns=rename_map)

    if not isinstance(df.index, pd.DatetimeIndex):
        for date_col in ("date", "Date", "datetime", "Datetime"):
            if date_col in df.columns:
                df = df.set_index(date_col)
                break
    df.index.name = "Date"
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def get_historical(
    ticker: str,
    start: str,
    end: str,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single symbol via OpenBB.

    Parameters
    ----------
    ticker : str
    start : str - 'YYYY-MM-DD'
    end : str - 'YYYY-MM-DD'
    provider : str - OpenBB data provider

    Returns
    -------
    pd.DataFrame
    """
    _ensure_openbb()
    asset = detect_asset_class(ticker)

    if asset == AssetClass.CRYPTO:
        base = ticker.upper().replace("-USD", "").replace("/USD", "")
        result = obb.crypto.price.historical(
            symbol=base, start_date=start, end_date=end, provider=provider,
        )
    elif asset == AssetClass.FX:
        clean = ticker.replace("=X", "")
        result = obb.currency.price.historical(
            symbol=clean, start_date=start, end_date=end, provider=provider,
        )
    else:
        result = obb.equity.price.historical(
            symbol=ticker, start_date=start, end_date=end, provider=provider,
        )

    df = result.to_dataframe()
    return _normalize_ohlcv(df)


def get_multiple(
    tickers: List[str],
    start: str,
    end: str,
    provider: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers.

    Parameters
    ----------
    tickers : list of str
    start, end : str
    provider : str

    Returns
    -------
    dict - {ticker: DataFrame}
    """
    results: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = get_historical(ticker, start, end, provider=provider)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
    return results


# ===================================================================
# Universe Fetching
# ===================================================================
@dataclass
class UniverseData:
    """Container for full universe data across all asset classes."""
    equities: Dict[str, pd.DataFrame] = field(default_factory=dict)
    bonds: Dict[str, pd.DataFrame] = field(default_factory=dict)
    commodities: Dict[str, pd.DataFrame] = field(default_factory=dict)
    crypto: Dict[str, pd.DataFrame] = field(default_factory=dict)
    fx: Dict[str, pd.DataFrame] = field(default_factory=dict)
    gics_classification: Dict[str, List[GICSClassification]] = field(default_factory=dict)
    fetch_timestamp: Optional[datetime] = None

    @property
    def all_symbols(self) -> List[str]:
        syms: List[str] = []
        for d in (self.equities, self.bonds, self.commodities, self.crypto, self.fx):
            syms.extend(d.keys())
        return syms

    @property
    def all_dataframes(self) -> Dict[str, pd.DataFrame]:
        merged: Dict[str, pd.DataFrame] = {}
        for d in (self.equities, self.bonds, self.commodities, self.crypto, self.fx):
            merged.update(d)
        return merged


def get_equity_universe(start: str, end: str, symbols: Optional[List[str]] = None, provider: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch the equity universe."""
    return get_multiple(symbols or DEFAULT_EQUITY_SYMBOLS, start, end, provider=provider)


def get_bond_universe(start: str, end: str, symbols: Optional[List[str]] = None, provider: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch the bond ETF universe."""
    return get_multiple(symbols or DEFAULT_BOND_SYMBOLS, start, end, provider=provider)


def get_commodity_universe(start: str, end: str, symbols: Optional[List[str]] = None, provider: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch the commodity ETF universe."""
    return get_multiple(symbols or DEFAULT_COMMODITY_SYMBOLS, start, end, provider=provider)


def get_crypto_universe(start: str, end: str, symbols: Optional[List[str]] = None, provider: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch the crypto universe."""
    return get_multiple(symbols or DEFAULT_CRYPTO_SYMBOLS, start, end, provider=provider)


def get_fx_universe(start: str, end: str, symbols: Optional[List[str]] = None, provider: str = "yfinance") -> Dict[str, pd.DataFrame]:
    """Fetch the FX pairs universe."""
    return get_multiple(symbols or DEFAULT_FX_PAIRS, start, end, provider=provider)


def get_full_universe(
    start: str,
    end: str,
    provider: str = "yfinance",
    equity_symbols: Optional[List[str]] = None,
    bond_symbols: Optional[List[str]] = None,
    commodity_symbols: Optional[List[str]] = None,
    crypto_symbols: Optional[List[str]] = None,
    fx_symbols: Optional[List[str]] = None,
) -> UniverseData:
    """
    Fetch the FULL investment universe across all asset classes.

    Parameters
    ----------
    start, end : str - Date range
    provider : str
    *_symbols : optional overrides

    Returns
    -------
    UniverseData
    """
    logger.info("Fetching full universe from %s to %s ...", start, end)

    equities = get_equity_universe(start, end, symbols=equity_symbols, provider=provider)
    bonds = get_bond_universe(start, end, symbols=bond_symbols, provider=provider)
    commodities = get_commodity_universe(start, end, symbols=commodity_symbols, provider=provider)
    crypto = get_crypto_universe(start, end, symbols=crypto_symbols, provider=provider)
    fx = get_fx_universe(start, end, symbols=fx_symbols, provider=provider)

    gics = classify_universe_by_gics(list(equities.keys()))

    universe = UniverseData(
        equities=equities,
        bonds=bonds,
        commodities=commodities,
        crypto=crypto,
        fx=fx,
        gics_classification=gics,
        fetch_timestamp=datetime.utcnow(),
    )
    logger.info(
        "Universe fetched: %d equities, %d bonds, %d commodities, %d crypto, %d fx",
        len(equities), len(bonds), len(commodities), len(crypto), len(fx),
    )
    return universe


# ===================================================================
# Fundamentals, Macro, and Sentiment
# ===================================================================
def get_fundamentals(symbol: str, provider: str = "yfinance") -> Dict[str, Any]:
    """
    Fetch fundamental data for an equity symbol.

    Valuation formulas:
        P/E = Price / EPS
        P/B = Price / Book_Value_Per_Share
        EV/EBITDA = Enterprise_Value / EBITDA
        DCF: V = sum_{t=1}^{N} FCF_t / (1+r)^t + TV / (1+r)^N
    """
    _ensure_openbb()
    data: Dict[str, Any] = {}
    try:
        inc = obb.equity.fundamental.income(symbol=symbol, provider=provider)
        data["income_statement"] = inc.to_dataframe()
    except Exception as exc:
        logger.warning("Income statement fetch failed for %s: %s", symbol, exc)
        data["income_statement"] = pd.DataFrame()

    try:
        bal = obb.equity.fundamental.balance(symbol=symbol, provider=provider)
        data["balance_sheet"] = bal.to_dataframe()
    except Exception as exc:
        logger.warning("Balance sheet fetch failed for %s: %s", symbol, exc)
        data["balance_sheet"] = pd.DataFrame()

    try:
        ratios = obb.equity.fundamental.ratios(symbol=symbol, provider=provider)
        data["ratios"] = ratios.to_dataframe()
    except Exception as exc:
        logger.warning("Ratios fetch failed for %s: %s", symbol, exc)
        data["ratios"] = pd.DataFrame()

    try:
        profile = obb.equity.profile(symbol=symbol, provider=provider)
        data["profile"] = profile.to_dataframe()
    except Exception as exc:
        logger.warning("Profile fetch failed for %s: %s", symbol, exc)
        data["profile"] = pd.DataFrame()

    return data


def get_macro_data(
    indicators: Optional[List[str]] = None,
    provider: str = "fred",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch macroeconomic indicators.

    Default indicators:
        GDP, UNRATE, CPIAUCSL, FEDFUNDS, T10Y2Y, VIXCLS,
        INDPRO, UMCSENT, DTWEXBGS, BAMLH0A0HYM2
    """
    _ensure_openbb()
    if indicators is None:
        indicators = [
            "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10Y2Y",
            "VIXCLS", "INDPRO", "UMCSENT", "DTWEXBGS", "BAMLH0A0HYM2",
        ]
    results: Dict[str, pd.DataFrame] = {}
    for series_id in indicators:
        try:
            result = obb.economy.fred_series(symbol=series_id, provider="fred")
            df = result.to_dataframe()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                elif "Date" in df.columns:
                    df = df.set_index("Date")
            results[series_id] = df
        except Exception as exc:
            logger.warning("Failed to fetch macro indicator %s: %s", series_id, exc)
    return results


def get_news_sentiment(
    symbols: Optional[List[str]] = None,
    limit: int = 50,
    provider: str = "benzinga",
) -> pd.DataFrame:
    """
    Fetch news and sentiment data.

    Sentiment scoring:
        composite_score = 0.4*headline + 0.35*body + 0.25*social
    """
    _ensure_openbb()
    try:
        if symbols:
            result = obb.news.company(symbol=",".join(symbols), limit=limit, provider=provider)
        else:
            result = obb.news.world(limit=limit, provider=provider)
        return result.to_dataframe()
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return pd.DataFrame()


# ===================================================================
# Risk Metrics
# ===================================================================
def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """S = (R_p - R_f) / sigma_p, annualized."""
    if returns.empty or returns.std() == 0:
        return 0.0
    rf = risk_free_rate / periods_per_year
    excess = returns - rf
    return float(excess.mean() / excess.std() * math.sqrt(periods_per_year))


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """So = (R_p - R_f) / sigma_d where sigma_d = sqrt(mean(min(R-Rf,0)^2))."""
    if returns.empty:
        return 0.0
    rf = risk_free_rate / periods_per_year
    excess = returns - rf
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = math.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def compute_max_drawdown(prices: pd.Series) -> float:
    """MDD = max(peak - price) / peak."""
    if prices.empty:
        return 0.0
    peak = prices.cummax()
    dd = (peak - prices) / peak
    return float(dd.max())


def compute_var(returns: pd.Series, confidence: float = 0.95, method: str = "parametric") -> float:
    """VaR_alpha = mu + z_alpha * sigma (parametric) or percentile (historical)."""
    if returns.empty:
        return 0.0
    if method == "historical":
        var = float(np.percentile(returns.dropna(), (1 - confidence) * 100))
    else:
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        var = float(returns.mean() + z * returns.std())
    return abs(var)


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR = E[R | R <= VaR], historical method."""
    if returns.empty:
        return 0.0
    var = compute_var(returns, confidence, method="historical")
    tail = returns[returns <= -var]
    if tail.empty:
        return var
    return float(abs(tail.mean()))


def compute_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """f* = (bp - q)/b, half-Kelly = f*/2, capped at [0, 1]."""
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p
    kelly = (b * p - q) / b
    return float(max(0.0, min(1.0, kelly / 2.0)))


def compute_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    C = S*N(d1) - K*e^{-rT}*N(d2)
    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    """
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
