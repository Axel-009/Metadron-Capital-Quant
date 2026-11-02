"""
Unified OpenBB Universe Module for ML-Macro-Market.

Provides a comprehensive multi-asset universe across ALL asset classes
(equities, bonds, commodities, crypto, FX, ETFs, indices) with full
GICS classification. OpenBB is the SOLE data source -- no yfinance.

Usage:
    from openbb_universe import (
        get_historical, get_multiple, get_fundamentals,
        get_full_universe, AssetClass, detect_asset_class,
        GICSSector, GICS_SECTORS,
    )
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset-class taxonomy
# ---------------------------------------------------------------------------

class AssetClass(Enum):
    """Enumeration of supported asset classes."""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    ETF = "etf"
    INDEX = "index"


# ---------------------------------------------------------------------------
# GICS Classification (all 11 sectors, 25 groups, 74 industries,
# 163 sub-industries)
# ---------------------------------------------------------------------------

class GICSSector(Enum):
    """Global Industry Classification Standard -- 11 sectors."""
    ENERGY = "10"
    MATERIALS = "15"
    INDUSTRIALS = "20"
    CONSUMER_DISCRETIONARY = "25"
    CONSUMER_STAPLES = "30"
    HEALTH_CARE = "35"
    FINANCIALS = "40"
    INFORMATION_TECHNOLOGY = "45"
    COMMUNICATION_SERVICES = "50"
    UTILITIES = "55"
    REAL_ESTATE = "60"


GICS_SECTORS: Dict[str, Dict[str, Any]] = {
    "10": {"name": "Energy", "groups": {
        "1010": {"name": "Energy", "industries": {
            "101010": {"name": "Energy Equipment & Services", "sub_industries": {
                "10101010": "Oil & Gas Drilling",
                "10101020": "Oil & Gas Equipment & Services",
            }},
            "101020": {"name": "Oil, Gas & Consumable Fuels", "sub_industries": {
                "10102010": "Integrated Oil & Gas",
                "10102020": "Oil & Gas Exploration & Production",
                "10102030": "Oil & Gas Refining & Marketing",
                "10102040": "Oil & Gas Storage & Transportation",
                "10102050": "Coal & Consumable Fuels",
            }},
        }},
    }},
    "15": {"name": "Materials", "groups": {
        "1510": {"name": "Materials", "industries": {
            "151010": {"name": "Chemicals", "sub_industries": {
                "15101010": "Commodity Chemicals", "15101020": "Diversified Chemicals",
                "15101030": "Fertilizers & Agricultural Chemicals", "15101040": "Industrial Gases",
                "15101050": "Specialty Chemicals",
            }},
            "151020": {"name": "Construction Materials", "sub_industries": {"15102010": "Construction Materials"}},
            "151030": {"name": "Containers & Packaging", "sub_industries": {
                "15103010": "Metal, Glass & Plastic Containers",
                "15103020": "Paper & Plastic Packaging Products & Materials",
            }},
            "151040": {"name": "Metals & Mining", "sub_industries": {
                "15104010": "Aluminum", "15104020": "Diversified Metals & Mining",
                "15104025": "Copper", "15104030": "Gold", "15104040": "Precious Metals & Minerals",
                "15104045": "Silver", "15104050": "Steel",
            }},
            "151050": {"name": "Paper & Forest Products", "sub_industries": {
                "15105010": "Forest Products", "15105020": "Paper Products",
            }},
        }},
    }},
    "20": {"name": "Industrials", "groups": {
        "2010": {"name": "Capital Goods", "industries": {
            "201010": {"name": "Aerospace & Defense", "sub_industries": {"20101010": "Aerospace & Defense"}},
            "201020": {"name": "Building Products", "sub_industries": {"20102010": "Building Products"}},
            "201030": {"name": "Construction & Engineering", "sub_industries": {"20103010": "Construction & Engineering"}},
            "201040": {"name": "Electrical Equipment", "sub_industries": {
                "20104010": "Electrical Components & Equipment", "20104020": "Heavy Electrical Equipment",
            }},
            "201050": {"name": "Industrial Conglomerates", "sub_industries": {"20105010": "Industrial Conglomerates"}},
            "201060": {"name": "Machinery", "sub_industries": {
                "20106010": "Construction Machinery & Heavy Transportation Equipment",
                "20106015": "Agricultural & Farm Machinery",
                "20106020": "Industrial Machinery & Supplies & Components",
            }},
            "201070": {"name": "Trading Companies & Distributors", "sub_industries": {"20107010": "Trading Companies & Distributors"}},
        }},
        "2020": {"name": "Commercial & Professional Services", "industries": {
            "202010": {"name": "Commercial Services & Supplies", "sub_industries": {
                "20201010": "Commercial Printing", "20201050": "Environmental & Facilities Services",
                "20201060": "Office Services & Supplies", "20201070": "Diversified Support Services",
                "20201080": "Security & Alarm Services",
            }},
            "202020": {"name": "Professional Services", "sub_industries": {
                "20202010": "Human Resource & Employment Services",
                "20202020": "Research & Consulting Services",
                "20202030": "Data Processing & Outsourced Services",
            }},
        }},
        "2030": {"name": "Transportation", "industries": {
            "203010": {"name": "Air Freight & Logistics", "sub_industries": {"20301010": "Air Freight & Logistics"}},
            "203020": {"name": "Passenger Airlines", "sub_industries": {"20302010": "Passenger Airlines"}},
            "203030": {"name": "Marine Transportation", "sub_industries": {"20303010": "Marine Transportation"}},
            "203040": {"name": "Ground Transportation", "sub_industries": {
                "20304010": "Rail Transportation", "20304020": "Cargo Ground Transportation",
                "20304030": "Passenger Ground Transportation",
            }},
            "203050": {"name": "Transportation Infrastructure", "sub_industries": {
                "20305010": "Airport Services", "20305020": "Highways & Railtracks",
                "20305030": "Marine Ports & Services",
            }},
        }},
    }},
    "25": {"name": "Consumer Discretionary", "groups": {
        "2510": {"name": "Automobiles & Components", "industries": {
            "251010": {"name": "Automobile Components", "sub_industries": {
                "25101010": "Automobile Parts & Equipment", "25101020": "Tires & Rubber",
            }},
            "251020": {"name": "Automobiles", "sub_industries": {
                "25102010": "Automobile Manufacturers", "25102020": "Motorcycle Manufacturers",
            }},
        }},
        "2520": {"name": "Consumer Durables & Apparel", "industries": {
            "252010": {"name": "Household Durables", "sub_industries": {
                "25201010": "Consumer Electronics", "25201020": "Home Furnishings",
                "25201030": "Homebuilding", "25201040": "Household Appliances",
                "25201050": "Housewares & Specialties",
            }},
            "252020": {"name": "Leisure Products", "sub_industries": {"25202010": "Leisure Products"}},
            "252030": {"name": "Textiles, Apparel & Luxury Goods", "sub_industries": {
                "25203010": "Apparel, Accessories & Luxury Goods", "25203020": "Footwear",
                "25203030": "Textiles",
            }},
        }},
        "2530": {"name": "Consumer Services", "industries": {
            "253010": {"name": "Hotels, Restaurants & Leisure", "sub_industries": {
                "25301010": "Casinos & Gaming", "25301020": "Hotels, Resorts & Cruise Lines",
                "25301030": "Leisure Facilities", "25301040": "Restaurants",
            }},
            "253020": {"name": "Diversified Consumer Services", "sub_industries": {
                "25302010": "Education Services", "25302020": "Specialized Consumer Services",
            }},
        }},
        "2550": {"name": "Retailing", "industries": {
            "255010": {"name": "Distributors", "sub_industries": {"25501010": "Distributors"}},
            "255020": {"name": "Internet & Direct Marketing Retail", "sub_industries": {"25502020": "Internet & Direct Marketing Retail"}},
            "255030": {"name": "Broadline Retail", "sub_industries": {
                "25503010": "Department Stores", "25503020": "General Merchandise Stores",
                "25503030": "Apparel Retail", "25503040": "Computer & Electronics Retail",
                "25503050": "Home Improvement Retail", "25503060": "Automotive Retail",
            }},
            "255040": {"name": "Specialty Retail", "sub_industries": {"25504020": "Specialty Stores"}},
        }},
    }},
    "30": {"name": "Consumer Staples", "groups": {
        "3010": {"name": "Food & Staples Retailing", "industries": {
            "301010": {"name": "Consumer Staples Distribution & Retail", "sub_industries": {
                "30101010": "Drug Retail", "30101020": "Food Distributors",
                "30101030": "Food Retail", "30101040": "Consumer Staples Merchandise Retail",
            }},
        }},
        "3020": {"name": "Food, Beverage & Tobacco", "industries": {
            "302010": {"name": "Beverages", "sub_industries": {
                "30201010": "Brewers", "30201020": "Distillers & Vintners",
                "30201030": "Soft Drinks & Non-alcoholic Beverages",
            }},
            "302020": {"name": "Food Products", "sub_industries": {
                "30202010": "Agricultural Products & Services", "30202030": "Packaged Foods & Meats",
            }},
            "302030": {"name": "Tobacco", "sub_industries": {"30203010": "Tobacco"}},
        }},
        "3030": {"name": "Household & Personal Products", "industries": {
            "303010": {"name": "Household Products", "sub_industries": {"30301010": "Household Products"}},
            "303020": {"name": "Personal Care Products", "sub_industries": {"30302010": "Personal Care Products"}},
        }},
    }},
    "35": {"name": "Health Care", "groups": {
        "3510": {"name": "Health Care Equipment & Services", "industries": {
            "351010": {"name": "Health Care Equipment & Supplies", "sub_industries": {
                "35101010": "Health Care Equipment", "35101020": "Health Care Supplies",
            }},
            "351020": {"name": "Health Care Providers & Services", "sub_industries": {
                "35102010": "Health Care Distributors", "35102015": "Health Care Services",
                "35102020": "Health Care Facilities", "35102030": "Managed Health Care",
            }},
            "351030": {"name": "Health Care Technology", "sub_industries": {"35103010": "Health Care Technology"}},
        }},
        "3520": {"name": "Pharmaceuticals, Biotechnology & Life Sciences", "industries": {
            "352010": {"name": "Biotechnology", "sub_industries": {"35201010": "Biotechnology"}},
            "352020": {"name": "Pharmaceuticals", "sub_industries": {"35202010": "Pharmaceuticals"}},
            "352030": {"name": "Life Sciences Tools & Services", "sub_industries": {"35203010": "Life Sciences Tools & Services"}},
        }},
    }},
    "40": {"name": "Financials", "groups": {
        "4010": {"name": "Banks", "industries": {
            "401010": {"name": "Banks", "sub_industries": {
                "40101010": "Diversified Banks", "40101015": "Regional Banks",
            }},
        }},
        "4020": {"name": "Financial Services", "industries": {
            "402010": {"name": "Financial Services", "sub_industries": {
                "40201020": "Other Diversified Financial Services", "40201030": "Multi-Sector Holdings",
                "40201040": "Specialized Finance", "40201050": "Commercial & Residential Mortgage Finance",
                "40201060": "Transaction & Payment Processing Services",
            }},
            "402020": {"name": "Consumer Finance", "sub_industries": {"40202010": "Consumer Finance"}},
            "402030": {"name": "Capital Markets", "sub_industries": {
                "40203010": "Asset Management & Custody Banks", "40203020": "Investment Banking & Brokerage",
                "40203030": "Diversified Capital Markets", "40203040": "Financial Exchanges & Data",
            }},
            "402040": {"name": "Mortgage Real Estate Investment Trusts", "sub_industries": {"40204010": "Mortgage REITs"}},
        }},
        "4030": {"name": "Insurance", "industries": {
            "403010": {"name": "Insurance", "sub_industries": {
                "40301010": "Insurance Brokers", "40301020": "Life & Health Insurance",
                "40301030": "Multi-line Insurance", "40301040": "Property & Casualty Insurance",
                "40301050": "Reinsurance",
            }},
        }},
    }},
    "45": {"name": "Information Technology", "groups": {
        "4510": {"name": "Software & Services", "industries": {
            "451010": {"name": "IT Services", "sub_industries": {
                "45101010": "IT Consulting & Other Services", "45101020": "Internet Services & Infrastructure",
            }},
            "451020": {"name": "Software", "sub_industries": {
                "45102010": "Application Software", "45102020": "Systems Software",
            }},
        }},
        "4520": {"name": "Technology Hardware & Equipment", "industries": {
            "452010": {"name": "Communications Equipment", "sub_industries": {"45201020": "Communications Equipment"}},
            "452020": {"name": "Technology Hardware, Storage & Peripherals", "sub_industries": {"45202010": "Technology Hardware, Storage & Peripherals"}},
            "452030": {"name": "Electronic Equipment, Instruments & Components", "sub_industries": {
                "45203010": "Electronic Equipment & Instruments", "45203015": "Electronic Components",
                "45203020": "Electronic Manufacturing Services", "45203030": "Technology Distributors",
            }},
        }},
        "4530": {"name": "Semiconductors & Semiconductor Equipment", "industries": {
            "453010": {"name": "Semiconductors & Semiconductor Equipment", "sub_industries": {
                "45301010": "Semiconductor Materials & Equipment", "45301020": "Semiconductors",
            }},
        }},
    }},
    "50": {"name": "Communication Services", "groups": {
        "5010": {"name": "Telecommunication Services", "industries": {
            "501010": {"name": "Diversified Telecommunication Services", "sub_industries": {
                "50101010": "Alternative Carriers", "50101020": "Integrated Telecommunication Services",
            }},
            "501020": {"name": "Wireless Telecommunication Services", "sub_industries": {"50102010": "Wireless Telecommunication Services"}},
        }},
        "5020": {"name": "Media & Entertainment", "industries": {
            "502010": {"name": "Media", "sub_industries": {
                "50201010": "Advertising", "50201020": "Broadcasting",
                "50201030": "Cable & Satellite", "50201040": "Publishing",
            }},
            "502020": {"name": "Entertainment", "sub_industries": {
                "50202010": "Movies & Entertainment", "50202020": "Interactive Home Entertainment",
            }},
            "502030": {"name": "Interactive Media & Services", "sub_industries": {"50203010": "Interactive Media & Services"}},
        }},
    }},
    "55": {"name": "Utilities", "groups": {
        "5510": {"name": "Utilities", "industries": {
            "551010": {"name": "Electric Utilities", "sub_industries": {"55101010": "Electric Utilities"}},
            "551020": {"name": "Gas Utilities", "sub_industries": {"55102010": "Gas Utilities"}},
            "551030": {"name": "Multi-Utilities", "sub_industries": {"55103010": "Multi-Utilities"}},
            "551040": {"name": "Water Utilities", "sub_industries": {"55104010": "Water Utilities"}},
            "551050": {"name": "Independent Power and Renewable Electricity Producers", "sub_industries": {
                "55105010": "Independent Power Producers & Energy Traders", "55105020": "Renewable Electricity",
            }},
        }},
    }},
    "60": {"name": "Real Estate", "groups": {
        "6010": {"name": "Equity Real Estate Investment Trusts (REITs)", "industries": {
            "601010": {"name": "Diversified REITs", "sub_industries": {"60101010": "Diversified REITs"}},
            "601025": {"name": "Industrial REITs", "sub_industries": {"60102510": "Industrial REITs"}},
            "601030": {"name": "Hotel & Resort REITs", "sub_industries": {"60103010": "Hotel & Resort REITs"}},
            "601040": {"name": "Office REITs", "sub_industries": {"60104010": "Office REITs"}},
            "601050": {"name": "Health Care REITs", "sub_industries": {"60105010": "Health Care REITs"}},
            "601060": {"name": "Residential REITs", "sub_industries": {
                "60106010": "Multi-Family Residential REITs", "60106020": "Single-Family Residential REITs",
            }},
            "601070": {"name": "Retail REITs", "sub_industries": {"60107010": "Retail REITs"}},
            "601080": {"name": "Specialized REITs", "sub_industries": {
                "60108010": "Diversified Real Estate Activities", "60108020": "Other Specialized REITs",
                "60108030": "Self-Storage REITs", "60108040": "Telecom Tower REITs",
                "60108050": "Timber REITs", "60108060": "Data Center REITs",
            }},
        }},
        "6020": {"name": "Real Estate Management & Development", "industries": {
            "602010": {"name": "Real Estate Management & Development", "sub_industries": {
                "60201010": "Diversified Real Estate Activities", "60201020": "Real Estate Operating Companies",
                "60201030": "Real Estate Development", "60201040": "Real Estate Services",
            }},
        }},
    }},
}

# Flat lookups
GICS_INDUSTRY_GROUPS: Dict[str, str] = {}
GICS_INDUSTRIES: Dict[str, str] = {}
GICS_SUB_INDUSTRIES: Dict[str, str] = {}

for _sc, _s in GICS_SECTORS.items():
    for _gc, _g in _s["groups"].items():
        GICS_INDUSTRY_GROUPS[_gc] = _g["name"]
        for _ic, _ind in _g["industries"].items():
            GICS_INDUSTRIES[_ic] = _ind["name"]
            for _sic, _sin in _ind["sub_industries"].items():
                GICS_SUB_INDUSTRIES[_sic] = _sin


# ---------------------------------------------------------------------------
# Hardcoded universe constituents
# ---------------------------------------------------------------------------

SP500_TOP50: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA",
    "UNH", "LLY", "JPM", "XOM", "JNJ", "V", "PG", "MA", "AVGO", "HD",
    "MRK", "COST", "ABBV", "CVX", "PEP", "ADBE", "KO", "WMT", "CRM",
    "MCD", "BAC", "CSCO", "NFLX", "TMO", "AMD", "ABT", "LIN", "ACN",
    "DHR", "ORCL", "CMCSA", "TXN", "PM", "INTC", "WFC", "NEE", "INTU",
    "QCOM", "DIS", "AMGN", "CAT", "IBM",
]

EQUITY_GICS_MAP: Dict[str, Dict[str, Any]] = {
    "AAPL":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4520", "industry": "452020"},
    "MSFT":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451020"},
    "AMZN":  {"sector": GICSSector.CONSUMER_DISCRETIONARY, "industry_group": "2550", "industry": "255020"},
    "NVDA":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "GOOGL": {"sector": GICSSector.COMMUNICATION_SERVICES, "industry_group": "5020", "industry": "502030"},
    "META":  {"sector": GICSSector.COMMUNICATION_SERVICES, "industry_group": "5020", "industry": "502030"},
    "BRK-B": {"sector": GICSSector.FINANCIALS, "industry_group": "4020", "industry": "402010"},
    "TSLA":  {"sector": GICSSector.CONSUMER_DISCRETIONARY, "industry_group": "2510", "industry": "251020"},
    "UNH":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3510", "industry": "351020"},
    "LLY":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352020"},
    "JPM":   {"sector": GICSSector.FINANCIALS, "industry_group": "4010", "industry": "401010"},
    "XOM":   {"sector": GICSSector.ENERGY, "industry_group": "1010", "industry": "101020"},
    "JNJ":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352020"},
    "V":     {"sector": GICSSector.FINANCIALS, "industry_group": "4020", "industry": "402010"},
    "PG":    {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3030", "industry": "303010"},
    "MA":    {"sector": GICSSector.FINANCIALS, "industry_group": "4020", "industry": "402010"},
    "AVGO":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "HD":    {"sector": GICSSector.CONSUMER_DISCRETIONARY, "industry_group": "2550", "industry": "255030"},
    "MRK":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352020"},
    "COST":  {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3010", "industry": "301010"},
    "ABBV":  {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352020"},
    "CVX":   {"sector": GICSSector.ENERGY, "industry_group": "1010", "industry": "101020"},
    "PEP":   {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3020", "industry": "302010"},
    "ADBE":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451020"},
    "KO":    {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3020", "industry": "302010"},
    "WMT":   {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3010", "industry": "301010"},
    "CRM":   {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451020"},
    "MCD":   {"sector": GICSSector.CONSUMER_DISCRETIONARY, "industry_group": "2530", "industry": "253010"},
    "BAC":   {"sector": GICSSector.FINANCIALS, "industry_group": "4010", "industry": "401010"},
    "CSCO":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4520", "industry": "452010"},
    "NFLX":  {"sector": GICSSector.COMMUNICATION_SERVICES, "industry_group": "5020", "industry": "502020"},
    "TMO":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352030"},
    "AMD":   {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "ABT":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3510", "industry": "351010"},
    "LIN":   {"sector": GICSSector.MATERIALS, "industry_group": "1510", "industry": "151010"},
    "ACN":   {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451010"},
    "DHR":   {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352030"},
    "ORCL":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451020"},
    "CMCSA": {"sector": GICSSector.COMMUNICATION_SERVICES, "industry_group": "5010", "industry": "501010"},
    "TXN":   {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "PM":    {"sector": GICSSector.CONSUMER_STAPLES, "industry_group": "3020", "industry": "302030"},
    "INTC":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "WFC":   {"sector": GICSSector.FINANCIALS, "industry_group": "4010", "industry": "401010"},
    "NEE":   {"sector": GICSSector.UTILITIES, "industry_group": "5510", "industry": "551010"},
    "INTU":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451020"},
    "QCOM":  {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4530", "industry": "453010"},
    "DIS":   {"sector": GICSSector.COMMUNICATION_SERVICES, "industry_group": "5020", "industry": "502020"},
    "AMGN":  {"sector": GICSSector.HEALTH_CARE, "industry_group": "3520", "industry": "352010"},
    "CAT":   {"sector": GICSSector.INDUSTRIALS, "industry_group": "2010", "industry": "201060"},
    "IBM":   {"sector": GICSSector.INFORMATION_TECHNOLOGY, "industry_group": "4510", "industry": "451010"},
}

BOND_UNIVERSE: List[str] = [
    "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "AGG", "BND", "MUB", "EMB",
]

COMMODITY_UNIVERSE: List[str] = [
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "CT=F", "SB=F", "LE=F",
]

CRYPTO_UNIVERSE: List[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "UNI-USD", "ATOM-USD",
    "LTC-USD", "DOGE-USD", "SHIB-USD",
]

FX_UNIVERSE: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURCHF=X", "USDMXN=X", "USDBRL=X", "USDCNY=X",
]

INDEX_UNIVERSE: List[str] = [
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    "^N225", "^HSI", "^STOXX50E",
]

ETF_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU",
    "GLD", "SLV", "USO", "UNG",
    "EEM", "EFA", "VWO", "IEMG",
    "ARKK", "ARKG", "ARKW",
    "VNQ", "VNQI",
]

# Macro indicator symbols (for the macro engine)
MACRO_INDICATORS: Dict[str, str] = {
    "GDP": "GDP",
    "UNRATE": "UNRATE",           # Unemployment rate
    "CPIAUCSL": "CPIAUCSL",       # CPI
    "FEDFUNDS": "FEDFUNDS",       # Fed funds rate
    "GS10": "GS10",               # 10Y Treasury yield
    "GS2": "GS2",                 # 2Y Treasury yield
    "T10Y2Y": "T10Y2Y",           # 10Y-2Y spread
    "T10YIE": "T10YIE",           # 10Y breakeven inflation
    "DTWEXBGS": "DTWEXBGS",       # Trade-weighted USD index
    "INDPRO": "INDPRO",           # Industrial production
    "UMCSENT": "UMCSENT",         # Consumer sentiment
    "PAYEMS": "PAYEMS",           # Nonfarm payrolls
    "HOUST": "HOUST",             # Housing starts
    "RSAFS": "RSAFS",             # Retail sales
    "DCOILWTICO": "DCOILWTICO",   # WTI crude oil
}


# ---------------------------------------------------------------------------
# Asset-class detection
# ---------------------------------------------------------------------------

_BOND_TICKERS = set(BOND_UNIVERSE)
_COMMODITY_TICKERS = set(COMMODITY_UNIVERSE)
_CRYPTO_TICKERS = set(CRYPTO_UNIVERSE)
_FX_TICKERS = set(FX_UNIVERSE)
_INDEX_TICKERS = set(INDEX_UNIVERSE)
_ETF_TICKERS = set(ETF_UNIVERSE)
_EQUITY_TICKERS = set(SP500_TOP50)


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Detect asset class via hardcoded sets and pattern heuristics.
    """
    sym = symbol.upper().strip()
    if sym in _INDEX_TICKERS:
        return AssetClass.INDEX
    if sym in _COMMODITY_TICKERS:
        return AssetClass.COMMODITY
    if sym in _CRYPTO_TICKERS:
        return AssetClass.CRYPTO
    if sym in _FX_TICKERS:
        return AssetClass.FX
    if sym in _BOND_TICKERS:
        return AssetClass.BOND
    if sym in _ETF_TICKERS:
        return AssetClass.ETF
    if sym in _EQUITY_TICKERS:
        return AssetClass.EQUITY
    # Pattern heuristics
    if sym.startswith("^"):
        return AssetClass.INDEX
    if sym.endswith("=F"):
        return AssetClass.COMMODITY
    if sym.endswith("-USD") or sym.endswith("-USDT"):
        return AssetClass.CRYPTO
    if sym.endswith("=X"):
        return AssetClass.FX
    return AssetClass.EQUITY


def get_gics_classification(symbol: str) -> Optional[Dict[str, Any]]:
    """Return GICS classification for an equity symbol."""
    info = EQUITY_GICS_MAP.get(symbol.upper())
    if info is None:
        return None
    sector_enum: GICSSector = info["sector"]
    ig_code = info["industry_group"]
    ind_code = info["industry"]
    return {
        "sector_code": sector_enum.value,
        "sector_name": GICS_SECTORS[sector_enum.value]["name"],
        "industry_group_code": ig_code,
        "industry_group_name": GICS_INDUSTRY_GROUPS.get(ig_code, "Unknown"),
        "industry_code": ind_code,
        "industry_name": GICS_INDUSTRIES.get(ind_code, "Unknown"),
    }


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def get_full_universe() -> Dict[AssetClass, List[str]]:
    """Return complete hardcoded universe keyed by asset class."""
    return {
        AssetClass.EQUITY: list(SP500_TOP50),
        AssetClass.BOND: list(BOND_UNIVERSE),
        AssetClass.COMMODITY: list(COMMODITY_UNIVERSE),
        AssetClass.CRYPTO: list(CRYPTO_UNIVERSE),
        AssetClass.FX: list(FX_UNIVERSE),
        AssetClass.INDEX: list(INDEX_UNIVERSE),
        AssetClass.ETF: list(ETF_UNIVERSE),
    }


def get_all_symbols() -> List[str]:
    """Return flat list of every symbol."""
    symbols: List[str] = []
    for group in get_full_universe().values():
        symbols.extend(group)
    return symbols


# ---------------------------------------------------------------------------
# OpenBB data access (SOLE source)
# ---------------------------------------------------------------------------

def get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data via OpenBB (sole source, no yfinance).
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    from openbb import obb

    asset_class = detect_asset_class(symbol)
    logger.info("Fetching %s (asset_class=%s) via OpenBB", symbol, asset_class.value)

    try:
        if asset_class == AssetClass.CRYPTO:
            result = obb.crypto.price.historical(
                symbol=symbol.replace("-USD", "").replace("-USDT", ""),
                start_date=start, end_date=end, provider=provider,
            )
        elif asset_class == AssetClass.FX:
            result = obb.currency.price.historical(
                symbol=symbol.replace("=X", ""),
                start_date=start, end_date=end, provider=provider,
            )
        elif asset_class == AssetClass.ETF:
            result = obb.etf.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider,
            )
        elif asset_class == AssetClass.INDEX:
            result = obb.index.price.historical(
                symbol=symbol.replace("^", ""),
                start_date=start, end_date=end, provider=provider,
            )
        else:
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider,
            )

        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"OpenBB returned empty DataFrame for {symbol}")
        logger.info("OpenBB returned %d rows for %s", len(df), symbol)
        return _normalise(df)

    except Exception as exc:
        raise RuntimeError(f"OpenBB failed for {symbol}: {exc}") from exc


def get_macro_indicator(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> pd.DataFrame:
    """
    Fetch macroeconomic indicator data via OpenBB (FRED integration).

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g., "GDP", "UNRATE", "CPIAUCSL").
    start, end : str, optional
    provider : str

    Returns
    -------
    pd.DataFrame
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    from openbb import obb

    try:
        result = obb.economy.fred_series(
            symbol=series_id,
            start_date=start,
            end_date=end,
            provider=provider,
        )
        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"No macro data for {series_id}")
        return df
    except Exception as exc:
        raise RuntimeError(f"OpenBB macro data failed for {series_id}: {exc}") from exc


def get_multiple(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple symbols."""
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            results[sym] = get_historical(sym, start=start, end=end, provider=provider)
        except Exception as exc:
            logger.error("Skipping %s: %s", sym, exc)
    return results


def get_fundamentals(
    symbol: str,
    provider: str = "openbb",
) -> Dict[str, Any]:
    """Fetch fundamental data via OpenBB (equities only)."""
    from openbb import obb

    asset_class = detect_asset_class(symbol)
    if asset_class != AssetClass.EQUITY:
        raise ValueError(f"Fundamentals only for equities, got {asset_class.value}")

    result = obb.equity.profile(symbol=symbol, provider=provider)
    data = result.to_dataframe()
    if data.empty:
        raise RuntimeError(f"No fundamental data for {symbol}")
    return data.iloc[0].to_dict()


def get_universe_data(
    asset_classes: Optional[List[AssetClass]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for the universe (or selected asset classes)."""
    universe = get_full_universe()
    symbols: List[str] = []
    if asset_classes is None:
        symbols = get_all_symbols()
    else:
        for ac in asset_classes:
            symbols.extend(universe.get(ac, []))
    return get_multiple(symbols, start=start, end=end, provider=provider)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to title-case."""
    rename_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower == "date":
            rename_map[col] = "Date"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower in ("close", "adj close", "adj_close", "adjclose"):
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)
    if df.index.name and df.index.name.lower() == "date":
        df = df.reset_index()
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    universe = get_full_universe()
    for ac, syms in universe.items():
        print(f"{ac.value}: {len(syms)} symbols")
    print(f"Total: {len(get_all_symbols())} symbols")
    print(f"Macro indicators: {len(MACRO_INDICATORS)}")
    print(f"GICS Sectors: {len(GICS_SECTORS)}")
    print(f"GICS Industry Groups: {len(GICS_INDUSTRY_GROUPS)}")
    print(f"GICS Industries: {len(GICS_INDUSTRIES)}")
    print(f"GICS Sub-Industries: {len(GICS_SUB_INDUSTRIES)}")
