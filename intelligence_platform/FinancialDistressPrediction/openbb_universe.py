"""
Unified OpenBB Data Module for Metadron Capital Investment Platform.

Provides full universe coverage across ALL asset classes with OpenBB as the SOLE data source.
No yfinance fallback - OpenBB Platform is the single source of truth.

Asset Classes Covered:
    - Equities (S&P 500 top 50, global large caps)
    - Fixed Income (government bonds, corporate bonds, bond ETFs)
    - Commodities (energy, metals, agriculture futures)
    - Crypto (top 20 by market cap)
    - FX (G10 majors and EM pairs)
    - ETFs (sector, factor, thematic)
    - Indices (global benchmark indices)

GICS Classification (Global Industry Classification Standard):
    11 Sectors: Energy, Materials, Industrials, Consumer Discretionary,
    Consumer Staples, Health Care, Financials, Information Technology,
    Communication Services, Utilities, Real Estate

Key Financial Formulas:

    Sharpe Ratio: S = (R_p - R_f) / sigma_p
        R_p = portfolio return, R_f = risk-free rate, sigma_p = portfolio std dev

    Sortino Ratio: So = (R_p - R_f) / sigma_d
        sigma_d = downside deviation (only negative returns)

    Kelly Criterion: f* = (p * b - q) / b
        p = win probability, q = 1 - p, b = win/loss ratio

    Value at Risk (VaR): VaR_alpha = -mu + z_alpha * sigma
        Parametric: assumes normal distribution
        Historical: percentile of actual return distribution
        Monte Carlo: simulation-based

    Conditional VaR (CVaR / Expected Shortfall):
        CVaR_alpha = E[L | L > VaR_alpha] = (1/(1-alpha)) * integral from alpha to 1 of VaR_u du

    Maximum Drawdown: MDD = max over t in [0,T] of (peak_t - trough_t) / peak_t

    Information Ratio: IR = (R_p - R_b) / TE
        R_b = benchmark return, TE = tracking error

    Treynor Ratio: T = (R_p - R_f) / beta_p

    Calmar Ratio: C = CAGR / MDD

    Omega Ratio: Omega(r) = integral(r to inf) [1 - F(x)] dx / integral(-inf to r) F(x) dx
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


class AssetClass(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    ETF = "etf"
    INDEX = "index"


class GICSSector(Enum):
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


@dataclass
class AssetInfo:
    symbol: str
    name: str
    asset_class: AssetClass
    gics_sector: Optional[GICSSector] = None
    gics_industry_group: Optional[str] = None
    country: str = "US"
    currency: str = "USD"
    market_cap: Optional[float] = None
    exchange: Optional[str] = None


# ──────────────────────────────────────────────
# Hardcoded Universe Constituents
# ──────────────────────────────────────────────

SP500_TOP_50 = {
    "AAPL": ("Apple Inc.", GICSSector.INFORMATION_TECHNOLOGY, "Technology Hardware"),
    "MSFT": ("Microsoft Corp.", GICSSector.INFORMATION_TECHNOLOGY, "Software"),
    "AMZN": ("Amazon.com Inc.", GICSSector.CONSUMER_DISCRETIONARY, "Internet Retail"),
    "NVDA": ("NVIDIA Corp.", GICSSector.INFORMATION_TECHNOLOGY, "Semiconductors"),
    "GOOGL": ("Alphabet Inc. Class A", GICSSector.COMMUNICATION_SERVICES, "Interactive Media"),
    "META": ("Meta Platforms Inc.", GICSSector.COMMUNICATION_SERVICES, "Interactive Media"),
    "GOOG": ("Alphabet Inc. Class C", GICSSector.COMMUNICATION_SERVICES, "Interactive Media"),
    "BRK.B": ("Berkshire Hathaway B", GICSSector.FINANCIALS, "Multi-Sector Holdings"),
    "TSLA": ("Tesla Inc.", GICSSector.CONSUMER_DISCRETIONARY, "Automobiles"),
    "UNH": ("UnitedHealth Group", GICSSector.HEALTH_CARE, "Managed Health Care"),
    "XOM": ("Exxon Mobil Corp.", GICSSector.ENERGY, "Integrated Oil & Gas"),
    "JNJ": ("Johnson & Johnson", GICSSector.HEALTH_CARE, "Pharmaceuticals"),
    "JPM": ("JPMorgan Chase & Co.", GICSSector.FINANCIALS, "Diversified Banks"),
    "V": ("Visa Inc.", GICSSector.FINANCIALS, "Transaction Processing"),
    "PG": ("Procter & Gamble", GICSSector.CONSUMER_STAPLES, "Household Products"),
    "MA": ("Mastercard Inc.", GICSSector.FINANCIALS, "Transaction Processing"),
    "AVGO": ("Broadcom Inc.", GICSSector.INFORMATION_TECHNOLOGY, "Semiconductors"),
    "HD": ("Home Depot Inc.", GICSSector.CONSUMER_DISCRETIONARY, "Home Improvement"),
    "CVX": ("Chevron Corp.", GICSSector.ENERGY, "Integrated Oil & Gas"),
    "LLY": ("Eli Lilly & Co.", GICSSector.HEALTH_CARE, "Pharmaceuticals"),
    "MRK": ("Merck & Co.", GICSSector.HEALTH_CARE, "Pharmaceuticals"),
    "ABBV": ("AbbVie Inc.", GICSSector.HEALTH_CARE, "Biotechnology"),
    "KO": ("Coca-Cola Co.", GICSSector.CONSUMER_STAPLES, "Soft Drinks"),
    "PEP": ("PepsiCo Inc.", GICSSector.CONSUMER_STAPLES, "Soft Drinks"),
    "COST": ("Costco Wholesale", GICSSector.CONSUMER_STAPLES, "Hypermarkets"),
    "BAC": ("Bank of America", GICSSector.FINANCIALS, "Diversified Banks"),
    "WMT": ("Walmart Inc.", GICSSector.CONSUMER_STAPLES, "Hypermarkets"),
    "MCD": ("McDonald's Corp.", GICSSector.CONSUMER_DISCRETIONARY, "Restaurants"),
    "CSCO": ("Cisco Systems", GICSSector.INFORMATION_TECHNOLOGY, "Communications Equipment"),
    "CRM": ("Salesforce Inc.", GICSSector.INFORMATION_TECHNOLOGY, "Application Software"),
    "TMO": ("Thermo Fisher Scientific", GICSSector.HEALTH_CARE, "Life Sciences Tools"),
    "ABT": ("Abbott Laboratories", GICSSector.HEALTH_CARE, "Health Care Equipment"),
    "ADBE": ("Adobe Inc.", GICSSector.INFORMATION_TECHNOLOGY, "Application Software"),
    "ACN": ("Accenture plc", GICSSector.INFORMATION_TECHNOLOGY, "IT Consulting"),
    "NFLX": ("Netflix Inc.", GICSSector.COMMUNICATION_SERVICES, "Movies & Entertainment"),
    "AMD": ("Advanced Micro Devices", GICSSector.INFORMATION_TECHNOLOGY, "Semiconductors"),
    "LIN": ("Linde plc", GICSSector.MATERIALS, "Industrial Gases"),
    "TXN": ("Texas Instruments", GICSSector.INFORMATION_TECHNOLOGY, "Semiconductors"),
    "NEE": ("NextEra Energy", GICSSector.UTILITIES, "Electric Utilities"),
    "DHR": ("Danaher Corp.", GICSSector.HEALTH_CARE, "Life Sciences Tools"),
    "PM": ("Philip Morris Int'l", GICSSector.CONSUMER_STAPLES, "Tobacco"),
    "DIS": ("Walt Disney Co.", GICSSector.COMMUNICATION_SERVICES, "Movies & Entertainment"),
    "WFC": ("Wells Fargo & Co.", GICSSector.FINANCIALS, "Diversified Banks"),
    "BMY": ("Bristol-Myers Squibb", GICSSector.HEALTH_CARE, "Pharmaceuticals"),
    "QCOM": ("QUALCOMM Inc.", GICSSector.INFORMATION_TECHNOLOGY, "Semiconductors"),
    "UPS": ("United Parcel Service", GICSSector.INDUSTRIALS, "Air Freight & Logistics"),
    "RTX": ("RTX Corp.", GICSSector.INDUSTRIALS, "Aerospace & Defense"),
    "AMGN": ("Amgen Inc.", GICSSector.HEALTH_CARE, "Biotechnology"),
    "PLD": ("Prologis Inc.", GICSSector.REAL_ESTATE, "Industrial REITs"),
    "SPGI": ("S&P Global Inc.", GICSSector.FINANCIALS, "Financial Exchanges"),
}

BOND_ETFS = {
    "TLT": ("iShares 20+ Year Treasury Bond", "US Treasury Long"),
    "IEF": ("iShares 7-10 Year Treasury Bond", "US Treasury Intermediate"),
    "SHY": ("iShares 1-3 Year Treasury Bond", "US Treasury Short"),
    "LQD": ("iShares Investment Grade Corporate Bond", "Corporate Investment Grade"),
    "HYG": ("iShares High Yield Corporate Bond", "Corporate High Yield"),
    "TIP": ("iShares TIPS Bond", "Inflation-Protected"),
    "AGG": ("iShares Core U.S. Aggregate Bond", "Aggregate"),
    "BND": ("Vanguard Total Bond Market", "Aggregate"),
    "MUB": ("iShares National Muni Bond", "Municipal"),
    "EMB": ("iShares J.P. Morgan USD Emerging Markets Bond", "Emerging Market Debt"),
    "BNDX": ("Vanguard Total International Bond", "International Aggregate"),
    "VCSH": ("Vanguard Short-Term Corporate Bond", "Corporate Short"),
}

COMMODITY_FUTURES = {
    "CL=F": ("Crude Oil WTI", "Energy"),
    "BZ=F": ("Brent Crude Oil", "Energy"),
    "NG=F": ("Natural Gas", "Energy"),
    "GC=F": ("Gold", "Precious Metals"),
    "SI=F": ("Silver", "Precious Metals"),
    "PL=F": ("Platinum", "Precious Metals"),
    "HG=F": ("Copper", "Base Metals"),
    "ZC=F": ("Corn", "Agriculture"),
    "ZS=F": ("Soybeans", "Agriculture"),
    "ZW=F": ("Wheat", "Agriculture"),
    "CT=F": ("Cotton", "Agriculture"),
    "KC=F": ("Coffee", "Agriculture"),
    "SB=F": ("Sugar", "Agriculture"),
    "LE=F": ("Live Cattle", "Livestock"),
    "GLD": ("SPDR Gold Shares ETF", "Precious Metals"),
    "SLV": ("iShares Silver Trust ETF", "Precious Metals"),
    "USO": ("United States Oil Fund", "Energy"),
    "DBA": ("Invesco DB Agriculture Fund", "Agriculture"),
}

CRYPTO_TOP_20 = {
    "BTC-USD": ("Bitcoin", "Layer 1"),
    "ETH-USD": ("Ethereum", "Layer 1 / Smart Contracts"),
    "BNB-USD": ("Binance Coin", "Exchange"),
    "XRP-USD": ("Ripple", "Payments"),
    "SOL-USD": ("Solana", "Layer 1 / Smart Contracts"),
    "ADA-USD": ("Cardano", "Layer 1 / Smart Contracts"),
    "DOGE-USD": ("Dogecoin", "Meme / Payments"),
    "TRX-USD": ("TRON", "Layer 1"),
    "AVAX-USD": ("Avalanche", "Layer 1 / Smart Contracts"),
    "DOT-USD": ("Polkadot", "Interoperability"),
    "MATIC-USD": ("Polygon", "Layer 2"),
    "LINK-USD": ("Chainlink", "Oracle"),
    "SHIB-USD": ("Shiba Inu", "Meme"),
    "UNI-USD": ("Uniswap", "DeFi / DEX"),
    "LTC-USD": ("Litecoin", "Payments"),
    "ATOM-USD": ("Cosmos", "Interoperability"),
    "XLM-USD": ("Stellar", "Payments"),
    "NEAR-USD": ("NEAR Protocol", "Layer 1"),
    "APT-USD": ("Aptos", "Layer 1"),
    "ARB-USD": ("Arbitrum", "Layer 2"),
}

FX_MAJORS = {
    "EURUSD=X": ("Euro / US Dollar", "G10"),
    "GBPUSD=X": ("British Pound / US Dollar", "G10"),
    "USDJPY=X": ("US Dollar / Japanese Yen", "G10"),
    "USDCHF=X": ("US Dollar / Swiss Franc", "G10"),
    "AUDUSD=X": ("Australian Dollar / US Dollar", "G10"),
    "USDCAD=X": ("US Dollar / Canadian Dollar", "G10"),
    "NZDUSD=X": ("New Zealand Dollar / US Dollar", "G10"),
    "EURGBP=X": ("Euro / British Pound", "G10 Cross"),
    "EURJPY=X": ("Euro / Japanese Yen", "G10 Cross"),
    "GBPJPY=X": ("British Pound / Japanese Yen", "G10 Cross"),
    "USDMXN=X": ("US Dollar / Mexican Peso", "EM"),
    "USDBRL=X": ("US Dollar / Brazilian Real", "EM"),
    "USDCNY=X": ("US Dollar / Chinese Yuan", "EM"),
    "USDINR=X": ("US Dollar / Indian Rupee", "EM"),
    "USDTRY=X": ("US Dollar / Turkish Lira", "EM"),
    "USDZAR=X": ("US Dollar / South African Rand", "EM"),
    "DX-Y.NYB": ("US Dollar Index (DXY)", "Index"),
}

GLOBAL_INDICES = {
    "^GSPC": ("S&P 500", "US"),
    "^DJI": ("Dow Jones Industrial Average", "US"),
    "^IXIC": ("NASDAQ Composite", "US"),
    "^RUT": ("Russell 2000", "US"),
    "^VIX": ("CBOE Volatility Index", "US"),
    "^FTSE": ("FTSE 100", "UK"),
    "^GDAXI": ("DAX", "Germany"),
    "^FCHI": ("CAC 40", "France"),
    "^N225": ("Nikkei 225", "Japan"),
    "^HSI": ("Hang Seng Index", "Hong Kong"),
    "000001.SS": ("Shanghai Composite", "China"),
    "^STOXX50E": ("Euro Stoxx 50", "Europe"),
    "^BSESN": ("BSE Sensex", "India"),
    "^AXJO": ("S&P/ASX 200", "Australia"),
    "^KS11": ("KOSPI", "South Korea"),
    "^GSPTSE": ("S&P/TSX Composite", "Canada"),
}

SECTOR_ETFS = {
    "XLE": ("Energy Select Sector SPDR", GICSSector.ENERGY),
    "XLB": ("Materials Select Sector SPDR", GICSSector.MATERIALS),
    "XLI": ("Industrials Select Sector SPDR", GICSSector.INDUSTRIALS),
    "XLY": ("Consumer Discretionary Select Sector SPDR", GICSSector.CONSUMER_DISCRETIONARY),
    "XLP": ("Consumer Staples Select Sector SPDR", GICSSector.CONSUMER_STAPLES),
    "XLV": ("Health Care Select Sector SPDR", GICSSector.HEALTH_CARE),
    "XLF": ("Financials Select Sector SPDR", GICSSector.FINANCIALS),
    "XLK": ("Technology Select Sector SPDR", GICSSector.INFORMATION_TECHNOLOGY),
    "XLC": ("Communication Services Select Sector SPDR", GICSSector.COMMUNICATION_SERVICES),
    "XLU": ("Utilities Select Sector SPDR", GICSSector.UTILITIES),
    "XLRE": ("Real Estate Select Sector SPDR", GICSSector.REAL_ESTATE),
    "SPY": ("SPDR S&P 500 ETF Trust", None),
    "QQQ": ("Invesco QQQ Trust", None),
    "IWM": ("iShares Russell 2000 ETF", None),
    "EEM": ("iShares MSCI Emerging Markets ETF", None),
    "EFA": ("iShares MSCI EAFE ETF", None),
    "VTI": ("Vanguard Total Stock Market ETF", None),
    "ARKK": ("ARK Innovation ETF", None),
}


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Detect the asset class of a given symbol based on known universe mappings.

    Classification Logic:
        1. Check against hardcoded universe dictionaries
        2. Use suffix heuristics (=F for futures, -USD for crypto, =X for FX)
        3. Check for index prefix (^)
        4. Default to EQUITY for unrecognized symbols

    Args:
        symbol: Ticker symbol string

    Returns:
        AssetClass enum value
    """
    if symbol in SP500_TOP_50:
        return AssetClass.EQUITY
    if symbol in BOND_ETFS:
        return AssetClass.FIXED_INCOME
    if symbol in COMMODITY_FUTURES:
        return AssetClass.COMMODITY
    if symbol in CRYPTO_TOP_20:
        return AssetClass.CRYPTO
    if symbol in FX_MAJORS:
        return AssetClass.FX
    if symbol in SECTOR_ETFS:
        return AssetClass.ETF
    if symbol in GLOBAL_INDICES:
        return AssetClass.INDEX

    # Heuristic-based detection from symbol patterns
    if symbol.endswith("=F"):
        return AssetClass.COMMODITY
    if symbol.endswith("-USD") and len(symbol.split("-")[0]) <= 5:
        return AssetClass.CRYPTO
    if symbol.endswith("=X"):
        return AssetClass.FX
    if symbol.startswith("^"):
        return AssetClass.INDEX
    if symbol.endswith((".SS", ".SZ", ".L", ".TO", ".AX", ".HK")):
        return AssetClass.EQUITY

    return AssetClass.EQUITY


def classify_by_gics(symbols: Optional[List[str]] = None) -> Dict[GICSSector, List[str]]:
    """
    Classify equity symbols by GICS sector.

    GICS (Global Industry Classification Standard) is maintained by MSCI and S&P.
    4-level hierarchy: Sector -> Industry Group -> Industry -> Sub-Industry

    Args:
        symbols: List of ticker symbols. If None, uses full S&P 500 top 50.

    Returns:
        Dictionary mapping GICSSector to list of symbols in that sector.
    """
    if symbols is None:
        symbols = list(SP500_TOP_50.keys())

    classification: Dict[GICSSector, List[str]] = {sector: [] for sector in GICSSector}

    for symbol in symbols:
        if symbol in SP500_TOP_50:
            _, sector, _ = SP500_TOP_50[symbol]
            classification[sector].append(symbol)
        else:
            # For sector ETFs, map them to their sector
            if symbol in SECTOR_ETFS:
                _, sector = SECTOR_ETFS[symbol]
                if sector is not None:
                    classification[sector].append(symbol)

    # Remove empty sectors
    return {k: v for k, v in classification.items() if v}


def get_equity_universe() -> List[AssetInfo]:
    """
    Return full equity universe with GICS classification.

    Returns:
        List of AssetInfo objects for all tracked equities.
    """
    equities = []
    for symbol, (name, sector, industry_group) in SP500_TOP_50.items():
        equities.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.EQUITY,
            gics_sector=sector,
            gics_industry_group=industry_group,
            country="US",
            currency="USD",
            exchange="NYSE/NASDAQ",
        ))
    return equities


def get_bond_universe() -> List[AssetInfo]:
    """
    Return fixed income universe (bond ETFs as proxies).

    Bond Pricing: P = sum_{t=1}^{N} C/(1+y)^t + F/(1+y)^N
    Duration: D = -dP/dy * (1/P)
    Modified Duration: D_mod = D / (1 + y/m)
    Convexity: C = d^2P/dy^2 * (1/P)
    DV01 = -D_mod * P * 0.0001

    Returns:
        List of AssetInfo objects for all tracked bond ETFs.
    """
    bonds = []
    for symbol, (name, bond_type) in BOND_ETFS.items():
        bonds.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.FIXED_INCOME,
            gics_sector=None,
            gics_industry_group=bond_type,
            country="US",
            currency="USD",
            exchange="NYSE/NASDAQ",
        ))
    return bonds


def get_commodity_universe() -> List[AssetInfo]:
    """
    Return commodity universe (futures and commodity ETFs).

    Futures Pricing: F = S * e^{(r - y + u) * T}
        S = spot, r = risk-free, y = convenience yield, u = storage cost
    Contango: F > S (storage costs dominate)
    Backwardation: F < S (convenience yield dominates)
    Roll yield = (F_near - F_far) / F_far

    Returns:
        List of AssetInfo objects for all tracked commodities.
    """
    commodities = []
    for symbol, (name, category) in COMMODITY_FUTURES.items():
        commodities.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.COMMODITY,
            gics_sector=None,
            gics_industry_group=category,
            country="Global",
            currency="USD",
            exchange="NYMEX/COMEX/CBOT",
        ))
    return commodities


def get_crypto_universe() -> List[AssetInfo]:
    """
    Return crypto universe (top 20 by market cap).

    Crypto-specific metrics:
        NVT Ratio (Network Value to Transactions): NVT = Market Cap / Daily Tx Volume
        MVRV Ratio: MVRV = Market Value / Realized Value
        Stock-to-Flow: S2F = Stock / Annual Production, Price ~ e^{a + b * ln(S2F)}
        Hash Rate correlation: higher hash rate -> higher security -> price support

    Returns:
        List of AssetInfo objects for all tracked cryptocurrencies.
    """
    cryptos = []
    for symbol, (name, category) in CRYPTO_TOP_20.items():
        cryptos.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.CRYPTO,
            gics_sector=None,
            gics_industry_group=category,
            country="Global",
            currency="USD",
            exchange="Crypto",
        ))
    return cryptos


def get_fx_universe() -> List[AssetInfo]:
    """
    Return FX universe (G10 majors and EM pairs).

    FX Pricing Models:
        Covered Interest Rate Parity: F/S = (1 + r_d) / (1 + r_f)
        Purchasing Power Parity: S_t = S_0 * (P_d / P_f)
        Carry Trade Return: R = (r_d - r_f) + (S_t - S_0) / S_0
        Real Effective Exchange Rate: REER = product of bilateral_rate * trade_weight

    Returns:
        List of AssetInfo objects for all tracked FX pairs.
    """
    fx_pairs = []
    for symbol, (name, category) in FX_MAJORS.items():
        base_ccy = symbol[:3] if "=X" in symbol else "USD"
        fx_pairs.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.FX,
            gics_sector=None,
            gics_industry_group=category,
            country="Global",
            currency=base_ccy,
            exchange="Interbank",
        ))
    return fx_pairs


def get_full_universe() -> List[AssetInfo]:
    """
    Return the complete multi-asset universe.

    Aggregates all asset classes into a single list for cross-asset analysis.
    Total coverage: ~170+ instruments across 7 asset classes.

    Portfolio Theory (Markowitz):
        E[R_p] = sum(w_i * E[R_i])
        sigma_p^2 = sum_i sum_j w_i * w_j * sigma_i * sigma_j * rho_ij
        Efficient frontier: min sigma_p^2 s.t. E[R_p] = target, sum(w_i) = 1

    Returns:
        List of AssetInfo objects for the entire universe.
    """
    universe = []
    universe.extend(get_equity_universe())
    universe.extend(get_bond_universe())
    universe.extend(get_commodity_universe())
    universe.extend(get_crypto_universe())
    universe.extend(get_fx_universe())

    # Add indices
    for symbol, (name, country) in GLOBAL_INDICES.items():
        universe.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.INDEX,
            country=country,
            currency="USD",
            exchange="Various",
        ))

    # Add sector/thematic ETFs
    for symbol, (name, sector) in SECTOR_ETFS.items():
        universe.append(AssetInfo(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.ETF,
            gics_sector=sector,
            country="US",
            currency="USD",
            exchange="NYSE/NASDAQ",
        ))

    return universe


def get_historical(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data from OpenBB Platform.

    Uses obb.equity.price.historical, obb.crypto.price.historical, etc.
    based on auto-detected asset class.

    Technical indicators computable from OHLCV:
        SMA(n) = (1/n) * sum_{i=0}^{n-1} P_{t-i}
        EMA(n) = alpha * P_t + (1 - alpha) * EMA_{t-1}, alpha = 2/(n+1)
        RSI(n) = 100 - 100/(1 + RS), RS = avg_gain / avg_loss over n periods
        MACD = EMA(12) - EMA(26), Signal = EMA(9, MACD)
        Bollinger Bands: upper = SMA(20) + 2*sigma, lower = SMA(20) - 2*sigma
        ATR(n) = EMA(n, TR), TR = max(H-L, |H-C_prev|, |L-C_prev|)

    Args:
        symbol: Ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval ('1d', '1h', '1w', '1m')

    Returns:
        DataFrame with columns: open, high, low, close, volume, date
    """
    from openbb import obb

    asset_class = detect_asset_class(symbol)

    try:
        if asset_class == AssetClass.CRYPTO:
            result = obb.crypto.price.historical(
                symbol=symbol.replace("-USD", ""),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        elif asset_class == AssetClass.FX:
            clean_symbol = symbol.replace("=X", "")
            result = obb.currency.price.historical(
                symbol=clean_symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        elif asset_class == AssetClass.COMMODITY and symbol.endswith("=F"):
            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        elif asset_class == AssetClass.INDEX:
            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        else:
            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

        df = result.to_dataframe()
        return df

    except Exception as e:
        raise RuntimeError(f"OpenBB data fetch failed for {symbol}: {e}")


def get_fundamentals(symbol: str) -> Dict:
    """
    Fetch fundamental data from OpenBB Platform for a given equity symbol.

    Fundamental Ratios:
        P/E = Price / EPS
        P/B = Price / Book Value per Share
        P/S = Price / Sales per Share
        EV/EBITDA = Enterprise Value / EBITDA
        ROE = Net Income / Shareholders Equity
        ROA = Net Income / Total Assets
        ROIC = NOPAT / Invested Capital
        Debt/Equity = Total Debt / Total Equity
        Current Ratio = Current Assets / Current Liabilities
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        Free Cash Flow Yield = FCF / Market Cap
        Dividend Yield = Annual Dividend / Price

    DuPont Decomposition:
        ROE = Net Margin * Asset Turnover * Equity Multiplier
            = (NI/Sales) * (Sales/Assets) * (Assets/Equity)

    Args:
        symbol: Equity ticker symbol

    Returns:
        Dictionary with fundamental data fields.
    """
    from openbb import obb

    fundamentals = {}

    try:
        # Income statement
        income = obb.equity.fundamental.income(symbol=symbol, period="annual", limit=2)
        income_df = income.to_dataframe()
        if not income_df.empty:
            latest = income_df.iloc[0]
            fundamentals["revenue"] = float(latest.get("revenue", 0))
            fundamentals["ebit"] = float(latest.get("ebit", 0))
            fundamentals["net_income"] = float(latest.get("net_income", 0))
            fundamentals["gross_profit"] = float(latest.get("gross_profit", 0))
            if len(income_df) > 1:
                prev = income_df.iloc[1]
                fundamentals["prev_revenue"] = float(prev.get("revenue", 0))
                fundamentals["prev_net_income"] = float(prev.get("net_income", 0))
    except Exception:
        pass

    try:
        # Balance sheet
        balance = obb.equity.fundamental.balance(symbol=symbol, period="annual", limit=2)
        balance_df = balance.to_dataframe()
        if not balance_df.empty:
            latest = balance_df.iloc[0]
            fundamentals["total_assets"] = float(latest.get("total_assets", 0))
            fundamentals["total_liabilities"] = float(latest.get("total_liabilities", 0))
            fundamentals["total_equity"] = float(latest.get("total_stockholders_equity", 0))
            fundamentals["current_assets"] = float(latest.get("total_current_assets", 0))
            fundamentals["current_liabilities"] = float(latest.get("total_current_liabilities", 0))
            fundamentals["retained_earnings"] = float(latest.get("retained_earnings", 0))
            fundamentals["long_term_debt"] = float(latest.get("long_term_debt", 0))
            fundamentals["working_capital"] = (
                fundamentals.get("current_assets", 0) - fundamentals.get("current_liabilities", 0)
            )
            if fundamentals["current_liabilities"] > 0:
                fundamentals["current_ratio"] = (
                    fundamentals["current_assets"] / fundamentals["current_liabilities"]
                )
            else:
                fundamentals["current_ratio"] = 0.0
    except Exception:
        pass

    try:
        # Cash flow
        cashflow = obb.equity.fundamental.cash(symbol=symbol, period="annual", limit=1)
        cf_df = cashflow.to_dataframe()
        if not cf_df.empty:
            latest = cf_df.iloc[0]
            fundamentals["cfo"] = float(latest.get("operating_cash_flow", 0))
            fundamentals["capex"] = float(latest.get("capital_expenditure", 0))
            fundamentals["fcf"] = fundamentals.get("cfo", 0) - abs(fundamentals.get("capex", 0))
    except Exception:
        pass

    try:
        # Key metrics / ratios
        metrics = obb.equity.fundamental.metrics(symbol=symbol, period="annual", limit=1)
        metrics_df = metrics.to_dataframe()
        if not metrics_df.empty:
            latest = metrics_df.iloc[0]
            fundamentals["market_cap"] = float(latest.get("market_cap", 0))
            fundamentals["pe_ratio"] = float(latest.get("pe_ratio", 0))
            fundamentals["pb_ratio"] = float(latest.get("pb_ratio", 0))
            fundamentals["dividend_yield"] = float(latest.get("dividend_yield", 0))
            fundamentals["roe"] = float(latest.get("roe", 0))
            fundamentals["roa"] = float(latest.get("roa", 0))
            fundamentals["shares_outstanding"] = float(latest.get("shares_outstanding", 0))
    except Exception:
        pass

    # Compute derived metrics
    ta = fundamentals.get("total_assets", 1)
    if ta > 0:
        fundamentals["asset_turnover"] = fundamentals.get("revenue", 0) / ta
        gp = fundamentals.get("gross_profit", 0)
        rev = fundamentals.get("revenue", 1)
        fundamentals["gross_margin"] = gp / rev if rev > 0 else 0.0
        fundamentals["net_margin"] = fundamentals.get("net_income", 0) / rev if rev > 0 else 0.0

    return fundamentals


def get_macro_data(indicator: str = "gdp", country: str = "united_states") -> pd.DataFrame:
    """
    Fetch macroeconomic data from OpenBB Platform.

    Key Macro Indicators:
        GDP Growth: g = (GDP_t - GDP_{t-1}) / GDP_{t-1}
        Inflation (CPI): pi = (CPI_t - CPI_{t-12}) / CPI_{t-12}
        Unemployment Rate: U = Unemployed / Labor Force
        Taylor Rule: i = r* + pi + 0.5*(pi - pi*) + 0.5*(y - y*)
        Yield Curve: spread = yield_10y - yield_2y (inversion predicts recession)
        Okun's Law: DeltaU ≈ -0.5 * (g - g*) where g* ≈ 2-3%
        Phillips Curve: pi = pi_e - beta * (U - U*) + supply_shock
        Misery Index: MI = Unemployment Rate + Inflation Rate
        Purchasing Managers Index: PMI > 50 = expansion, < 50 = contraction

    Args:
        indicator: Macro indicator name ('gdp', 'cpi', 'unemployment', 'interest_rate', 'pmi')
        country: Country name for data

    Returns:
        DataFrame with macro time series data.
    """
    from openbb import obb

    indicator_map = {
        "gdp": lambda: obb.economy.gdp.nominal(country=country),
        "cpi": lambda: obb.economy.cpi(country=country),
        "unemployment": lambda: obb.economy.unemployment(country=country),
        "interest_rate": lambda: obb.economy.fred_series(symbol="FEDFUNDS"),
        "pmi": lambda: obb.economy.fred_series(symbol="MANEMP"),
        "yield_curve": lambda: obb.economy.fred_series(symbol="T10Y2Y"),
        "consumer_sentiment": lambda: obb.economy.fred_series(symbol="UMCSENT"),
        "housing_starts": lambda: obb.economy.fred_series(symbol="HOUST"),
        "industrial_production": lambda: obb.economy.fred_series(symbol="INDPRO"),
        "retail_sales": lambda: obb.economy.fred_series(symbol="RSAFS"),
    }

    try:
        fetcher = indicator_map.get(indicator.lower())
        if fetcher is None:
            raise ValueError(f"Unknown indicator: {indicator}. Available: {list(indicator_map.keys())}")
        result = fetcher()
        return result.to_dataframe()
    except Exception as e:
        raise RuntimeError(f"OpenBB macro data fetch failed for {indicator}: {e}")
