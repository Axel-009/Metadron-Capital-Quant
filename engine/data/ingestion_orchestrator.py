"""Data Ingestion Orchestrator for Metadron Capital.

Continuous multi-asset data ingestion across all permitted asset classes.
Routes ingested data to engine layers L1-L7 and the equity pooling tool.

HARD CONSTRAINTS:
    - NO CRYPTO — explicitly excluded from all ingestion paths
    - Equities: US S&P 1500 universe + London FTSE 100 ONLY
    - Commodities: Major ETFs only (price reference, cyclical patterns, trade signals)
    - Fixed Income: G10+India+Japan sovereign, US corporate only, major structured benchmarks
    - Currencies: G10 + INR + JPY
    - Econometrics: FRED macro series
    - SEC Filings: Major updates only (10-K, 10-Q, 8-K material events)
    - Options: Selected securities opportunistically for alpha maximization
    - Futures: Beta management within the beta corridor

All data sourced via OpenBB (34+ providers). try/except on ALL external imports.
System runs degraded, never broken.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

logger = logging.getLogger("metadron.scan.ingestion")

# ---------------------------------------------------------------------------
# Graceful imports from engine.data.openbb_data
# ---------------------------------------------------------------------------
try:
    from engine.data.openbb_data import (
        get_prices,
        get_adj_close,
        get_returns,
        get_fundamentals,
        get_bulk_fundamentals,
        get_fred_series,
        get_treasury_rates,
        get_credit_spreads,
        get_fed_balance_sheet,
        get_monetary_data,
        get_sofr_rate,
        get_company_filings,
        get_insider_trading,
        get_options_chains,
        get_etf_holdings,
        get_economic_calendar,
        get_company_news,
        get_macro_data_enriched,
        get_data_source_status,
        FRED_SERIES,
        _obb,
        _openbb_available,
    )
    _DATA_MODULE_AVAILABLE = True
    logger.info("OpenBB data module loaded for ingestion orchestrator")
except ImportError:
    _DATA_MODULE_AVAILABLE = False
    _obb = None
    _openbb_available = False
    FRED_SERIES = {}
    logger.warning(
        "engine.data.openbb_data unavailable — ingestion will return empty frames"
    )

    # Stub functions so calls never crash
    def get_prices(*a, **kw): return pd.DataFrame()
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_fundamentals(*a, **kw): return {}
    def get_bulk_fundamentals(*a, **kw): return pd.DataFrame()
    def get_fred_series(*a, **kw): return pd.DataFrame()
    def get_treasury_rates(*a, **kw): return pd.DataFrame()
    def get_credit_spreads(*a, **kw): return pd.DataFrame()
    def get_fed_balance_sheet(*a, **kw): return pd.DataFrame()
    def get_monetary_data(*a, **kw): return pd.DataFrame()
    def get_sofr_rate(*a, **kw): return pd.DataFrame()
    def get_company_filings(*a, **kw): return pd.DataFrame()
    def get_insider_trading(*a, **kw): return pd.DataFrame()
    def get_options_chains(*a, **kw): return pd.DataFrame()
    def get_etf_holdings(*a, **kw): return pd.DataFrame()
    def get_economic_calendar(*a, **kw): return pd.DataFrame()
    def get_company_news(*a, **kw): return pd.DataFrame()
    def get_macro_data_enriched(*a, **kw): return {}
    def get_data_source_status(*a, **kw): return {"openbb_available": False}

try:
    from engine.data.universe_engine import (
        UniverseEngine,
        SECTOR_ETFS,
        COMMODITY_ETFS,
        FIXED_INCOME_ETFS,
        INDEX_ETFS,
        INTERNATIONAL_ETFS,
        ALL_ETFS,
    )
    _UNIVERSE_AVAILABLE = True
except ImportError:
    _UNIVERSE_AVAILABLE = False
    SECTOR_ETFS = {}
    COMMODITY_ETFS = {}
    FIXED_INCOME_ETFS = {}
    INDEX_ETFS = {}
    INTERNATIONAL_ETFS = {}
    ALL_ETFS = []

try:
    from engine.data.cross_asset_universe import (
        SP500_TICKERS,
        SP400_TICKERS,
        SP600_TICKERS,
        get_all_static_tickers,
    )
    _CROSS_ASSET_AVAILABLE = True
except ImportError:
    _CROSS_ASSET_AVAILABLE = False
    SP500_TICKERS = []
    SP400_TICKERS = []
    SP600_TICKERS = []
    def get_all_static_tickers(): return []


# ============================================================================
# ASSET CLASS ENUMS
# ============================================================================

class AssetClassType(str, Enum):
    """All permitted asset classes for Metadron Capital. NO CRYPTO."""
    EQUITIES_US = "EQUITIES_US"
    EQUITIES_FTSE = "EQUITIES_FTSE"
    COMMODITIES = "COMMODITIES"
    INDICES = "INDICES"
    FIXED_INCOME_SOVEREIGN = "FIXED_INCOME_SOVEREIGN"
    FIXED_INCOME_CORPORATE = "FIXED_INCOME_CORPORATE"
    FIXED_INCOME_STRUCTURED = "FIXED_INCOME_STRUCTURED"
    CURRENCIES = "CURRENCIES"
    ECONOMETRICS = "ECONOMETRICS"
    SEC_FILINGS = "SEC_FILINGS"
    OPTIONS = "OPTIONS"
    FUTURES = "FUTURES"


class IngestionFrequency(str, Enum):
    """Ingestion cadence tiers."""
    REAL_TIME = "REAL_TIME"
    ONE_MIN = "1MIN"
    FIVE_MIN = "5MIN"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class EngineLayer(str, Enum):
    """Engine layer routing targets."""
    L1_DATA = "L1_DATA"
    L2_SIGNALS = "L2_SIGNALS"
    L3_ML = "L3_ML"
    L4_PORTFOLIO = "L4_PORTFOLIO"
    L5_EXECUTION = "L5_EXECUTION"
    L6_AGENTS = "L6_AGENTS"
    L7_HFT = "L7_HFT"


class DataQualityStatus(str, Enum):
    """Data quality check outcomes."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    STALE = "STALE"
    MISSING = "MISSING"


# ============================================================================
# EXCLUDED ASSET CLASSES — HARD BLOCK
# ============================================================================

EXCLUDED_ASSET_CLASSES = frozenset({
    "CRYPTO", "CRYPTOCURRENCY", "BITCOIN", "ETHEREUM", "ALTCOIN",
    "DEFI", "NFT", "STABLECOIN", "DIGITAL_ASSET",
})

CRYPTO_TICKER_PATTERNS = frozenset({
    "BTC", "ETH", "SOL", "ADA", "DOT", "DOGE", "SHIB", "XRP",
    "AVAX", "MATIC", "LINK", "UNI", "AAVE", "BNB", "USDT", "USDC",
})


def _is_crypto(ticker: str) -> bool:
    """Return True if a ticker looks like a crypto asset. Hard block."""
    t = ticker.upper().strip()
    if t in CRYPTO_TICKER_PATTERNS:
        return True
    if "-USD" in t or "-BTC" in t or "-ETH" in t:
        return True
    if t.endswith("USDT") or t.endswith("BUSD"):
        return True
    return False


# ============================================================================
# UNIVERSE DEFINITIONS
# ============================================================================

# --- FTSE 100 constituents (London Stock Exchange, as of March 2026) ---
FTSE_100_TICKERS = sorted(set([
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L",
    "REL.L", "DGE.L", "LSEG.L", "AAL.L", "GLEN.L", "NG.L", "CPG.L",
    "AHT.L", "RKT.L", "CRH.L", "EXPN.L", "BAE.L", "LLOY.L",
    "VOD.L", "PRU.L", "BARC.L", "STAN.L", "ANTO.L", "SSE.L",
    "BATS.L", "IMB.L", "ABF.L", "BKG.L", "SGRO.L", "LAND.L",
    "SVT.L", "WPP.L", "RR.L", "TSCO.L", "SBRY.L", "MNDI.L",
    "NWG.L", "LGEN.L", "AVIVA.L", "AV.L", "PHNX.L", "SN.L",
    "BNZL.L", "HLMA.L", "SMDS.L", "INF.L", "IHG.L", "WTB.L",
    "RTO.L", "BRBY.L", "FRAS.L", "JD.L", "CTEC.L", "EDV.L",
    "WEIR.L", "SPX.L", "SMIN.L", "RS1.L", "DARK.L", "PSON.L",
    "FRES.L", "KGF.L", "HSBC.L", "ENT.L", "ICP.L", "ITRK.L",
    "III.L", "BNKE.L", "HLN.L", "AUTO.L", "PSN.L", "TW.L",
    "BME.L", "SDR.L", "MGGT.L", "JMAT.L", "CRDA.L", "HIK.L",
    "DCC.L", "SMTH.L", "EVR.L", "FLTR.L", "FCIT.L", "SMT.L",
    "NXT.L", "OCDO.L", "MNG.L", "PERS.L", "BDEV.L",
    "ADM.L", "IAG.L", "MRO.L", "BA.L", "UU.L", "RMV.L",
    "POLY.L", "STJ.L", "CCH.L",
]))

# --- Commodity ETFs (price reference, cyclical patterns, global trade signals) ---
COMMODITY_ETF_UNIVERSE = {
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil (WTI)",
    "UNG": "Natural Gas",
    "DBA": "Agriculture",
    "DBC": "Commodities Broad",
    "PDBC": "Commodities Optimum Yield",
    "COPX": "Copper Miners (Copper proxy)",
    "WEAT": "Wheat",
    "CORN": "Corn",
}

# --- Major Index ETFs (benchmarking, constituent reference, monthly rebalancing) ---
INDEX_ETF_UNIVERSE = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones Industrial",
    "VT": "Total World",
    "EFA": "MSCI EAFE (Developed ex-US)",
    "EEM": "MSCI Emerging Markets",
    "ISF.L": "iShares Core FTSE 100 (LSE)",
}

# --- Fixed Income ---

# Sovereign: G10 countries + India + Japan
SOVEREIGN_BOND_UNIVERSE = {
    "TLT": "US Treasury 20Y+",
    "IEF": "US Treasury 7-10Y",
    "SHY": "US Treasury 1-3Y",
    "TIP": "US TIPS",
    "SHV": "US Treasury Short",
    "IGLT.L": "UK Gilts",
    "IBGS.L": "Euro Govt Short",
    "SEGA.L": "Euro Aggregate Govt",
    "BNDX": "Intl Bond (ex-US, includes Japan/G10)",
    "IGOV": "Intl Treasury (G10 Sovereign)",
    "BWX": "SPDR Intl Treasury (G10 Sovereign)",
    "EMLC": "EM Local Currency Bond (includes India)",
    "LEMB": "iShares EM Local Govt (India weight)",
}

# Corporate: US credit bonds ONLY (IG and HY)
CORPORATE_BOND_UNIVERSE = {
    "LQD": "iShares IG Corporate (US)",
    "HYG": "iShares HY Corporate (US)",
    "JNK": "SPDR HY Corporate (US)",
    "VCIT": "Vanguard Intermediate-Term Corp (US IG)",
    "VCSH": "Vanguard Short-Term Corp (US IG)",
    "BNDX": "Vanguard Total Intl Bond",
}

# Structured Products: Major benchmarks only
STRUCTURED_PRODUCT_UNIVERSE = {
    "MBB": "iShares MBS",
    "VMBS": "Vanguard MBS",
}

# --- Currencies: G10 + India (INR) + Japan (JPY) ---
CURRENCY_UNIVERSE = {
    "UUP": "US Dollar Bullish (DXY proxy)",
    "FXE": "Euro",
    "FXB": "British Pound",
    "FXY": "Japanese Yen",
    "FXA": "Australian Dollar",
    "FXC": "Canadian Dollar",
    "FXF": "Swiss Franc",
    "FXS": "Swedish Krona",
}

# G10 currencies + INR + JPY — FRED series IDs for exchange rates
CURRENCY_FRED_SERIES = {
    "DEXUSEU": "USD/EUR",
    "DEXUSUK": "USD/GBP",
    "DEXJPUS": "JPY/USD",
    "DEXUSAL": "USD/AUD",
    "DEXCAUS": "CAD/USD",
    "DEXSZUS": "CHF/USD",
    "DEXNOUS": "NOK/USD",
    "DEXSDUS": "SEK/USD",
    "DEXUSNZ": "USD/NZD",
    "DEXINUS": "INR/USD",
}

# --- Econometrics: FRED macro data ---
ECONOMETRIC_FRED_SERIES = {
    "GDP": "Gross Domestic Product (Nominal)",
    "GDPC1": "Real GDP",
    "A191RL1Q225SBEA": "Real GDP Growth Rate",
    "CPIAUCSL": "CPI All Urban Consumers",
    "CPILFESL": "Core CPI (ex Food & Energy)",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE",
    "PPIACO": "PPI All Commodities",
    "PAYEMS": "Total Nonfarm Payrolls (NFP)",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Jobless Claims",
    "CCSA": "Continued Claims",
    "MANEMP": "Manufacturing Employment (ISM proxy)",
    "INDPRO": "Industrial Production Index",
    "TCU": "Capacity Utilization",
    "NAPM": "ISM Manufacturing PMI",
    "NAPMNOI": "ISM New Orders",
    "UMCSENT": "U Mich Consumer Sentiment",
    "CSCICP03USM665S": "Consumer Confidence (OECD)",
    "RSAFS": "Retail Sales",
    "PCE": "Personal Consumption Expenditures",
    "HOUST": "Housing Starts",
    "PERMIT": "Building Permits",
    "HSN1F": "New Home Sales",
    "EXHOSLUSM495S": "Existing Home Sales",
    "CSUSHPISA": "Case-Shiller Home Price Index",
    "BOPGSTB": "Trade Balance",
    "IMPGS": "Imports of Goods & Services",
    "EXPGS": "Exports of Goods & Services",
    "M2SL": "M2 Money Supply",
    "M2V": "Velocity of M2",
    "BOGMBASE": "Monetary Base",
    "WALCL": "Fed Total Assets (Balance Sheet)",
    "RRPONTSYD": "ON-RRP Balance",
    "WTREGEN": "Treasury General Account",
    "FEDFUNDS": "Federal Funds Rate",
    "DFF": "Fed Funds Effective Daily",
    "DGS2": "2Y Treasury Yield",
    "DGS5": "5Y Treasury Yield",
    "DGS10": "10Y Treasury Yield",
    "DGS30": "30Y Treasury Yield",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    "DTB3": "3-Month T-Bill",
    "SOFR": "SOFR Rate",
    "BAMLH0A0HYM2": "HY OAS Spread",
    "BAMLC0A4CBBB": "BBB OAS Spread",
    "TEDRATE": "TED Spread",
    "VIXCLS": "VIX (FRED)",
    "DTWEXBGS": "Trade-Weighted Dollar Index",
    "DCOILWTICO": "WTI Crude Oil",
    "GOLDAMGBD228NLBM": "Gold (London PM Fix)",
}

SEC_FILING_TYPES = ["10-K", "10-Q", "8-K"]

FUTURES_UNIVERSE = {
    "ES=F": "E-mini S&P 500",
    "NQ=F": "E-mini NASDAQ 100",
    "YM=F": "E-mini Dow Jones",
    "RTY=F": "E-mini Russell 2000",
    "VX=F": "VIX Futures",
    "ZN=F": "10-Year Treasury Note",
    "ZB=F": "30-Year Treasury Bond",
    "ZF=F": "5-Year Treasury Note",
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
}

OPTIONS_DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "GS", "XOM", "UNH",
]


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

@dataclass
class DataQualityReport:
    """Result of a data quality check on an ingested dataset."""
    asset_class: str = ""
    timestamp: str = ""
    status: str = DataQualityStatus.PASS.value
    row_count: int = 0
    column_count: int = 0
    null_pct: float = 0.0
    stale_threshold_hours: float = 24.0
    last_data_timestamp: str = ""
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _check_data_quality(
    df: pd.DataFrame,
    asset_class: str,
    stale_hours: float = 24.0,
) -> DataQualityReport:
    """Run quality checks on an ingested DataFrame.

    Checks:
        1. Non-empty (row count > 0)
        2. Null percentage below threshold (< 30%)
        3. Data freshness (last timestamp within stale_hours)
        4. No duplicate indices
    """
    report = DataQualityReport(
        asset_class=asset_class,
        timestamp=datetime.utcnow().isoformat(),
        stale_threshold_hours=stale_hours,
    )

    if df is None or df.empty:
        report.status = DataQualityStatus.MISSING.value
        report.errors.append("DataFrame is empty or None")
        return report

    report.row_count = len(df)
    report.column_count = len(df.columns)

    # Null check
    total_cells = df.size
    null_cells = int(df.isnull().sum().sum()) if total_cells > 0 else 0
    report.null_pct = (null_cells / total_cells * 100) if total_cells > 0 else 0.0

    if report.null_pct > 50.0:
        report.status = DataQualityStatus.FAIL.value
        report.errors.append(
            f"Null percentage {report.null_pct:.1f}% exceeds 50% threshold"
        )
    elif report.null_pct > 30.0:
        report.status = DataQualityStatus.WARN.value
        report.warnings.append(
            f"Null percentage {report.null_pct:.1f}% exceeds 30% threshold"
        )

    # Staleness check
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            last_ts = df.index.max()
            report.last_data_timestamp = str(last_ts)
            age_hours = (
                datetime.utcnow() - last_ts.to_pydatetime().replace(tzinfo=None)
            ).total_seconds() / 3600
            if age_hours > stale_hours:
                if report.status == DataQualityStatus.PASS.value:
                    report.status = DataQualityStatus.STALE.value
                report.warnings.append(
                    f"Data is {age_hours:.1f}h old (threshold: {stale_hours}h)"
                )
    except Exception:
        pass

    # Duplicate index check
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        report.warnings.append(f"{dup_count} duplicate index entries")

    if not report.errors and not report.warnings:
        report.status = DataQualityStatus.PASS.value

    return report


# ============================================================================
# ROUTING MAP: Asset Class -> Engine Layers
# ============================================================================

ASSET_CLASS_ROUTING = {
    AssetClassType.EQUITIES_US: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L3_ML,
        EngineLayer.L4_PORTFOLIO, EngineLayer.L7_HFT,
    ],
    AssetClassType.EQUITIES_FTSE: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L3_ML,
        EngineLayer.L4_PORTFOLIO,
    ],
    AssetClassType.COMMODITIES: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS,
    ],
    AssetClassType.INDICES: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L4_PORTFOLIO,
    ],
    AssetClassType.FIXED_INCOME_SOVEREIGN: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS,
    ],
    AssetClassType.FIXED_INCOME_CORPORATE: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L3_ML,
    ],
    AssetClassType.FIXED_INCOME_STRUCTURED: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS,
    ],
    AssetClassType.CURRENCIES: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS,
    ],
    AssetClassType.ECONOMETRICS: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L3_ML,
        EngineLayer.L4_PORTFOLIO,
    ],
    AssetClassType.SEC_FILINGS: [
        EngineLayer.L1_DATA, EngineLayer.L2_SIGNALS, EngineLayer.L6_AGENTS,
    ],
    AssetClassType.OPTIONS: [
        EngineLayer.L1_DATA, EngineLayer.L5_EXECUTION, EngineLayer.L7_HFT,
    ],
    AssetClassType.FUTURES: [
        EngineLayer.L1_DATA, EngineLayer.L4_PORTFOLIO, EngineLayer.L5_EXECUTION,
    ],
}


# ============================================================================
# INGESTION SCHEDULER
# ============================================================================

INGESTION_SCHEDULE = {
    (AssetClassType.EQUITIES_US, IngestionFrequency.ONE_MIN): True,
    (AssetClassType.EQUITIES_US, IngestionFrequency.FIVE_MIN): True,
    (AssetClassType.EQUITIES_US, IngestionFrequency.DAILY): True,
    (AssetClassType.EQUITIES_FTSE, IngestionFrequency.FIVE_MIN): True,
    (AssetClassType.EQUITIES_FTSE, IngestionFrequency.DAILY): True,
    (AssetClassType.COMMODITIES, IngestionFrequency.FIVE_MIN): True,
    (AssetClassType.COMMODITIES, IngestionFrequency.DAILY): True,
    (AssetClassType.INDICES, IngestionFrequency.ONE_MIN): True,
    (AssetClassType.INDICES, IngestionFrequency.DAILY): True,
    (AssetClassType.INDICES, IngestionFrequency.MONTHLY): True,
    (AssetClassType.FIXED_INCOME_SOVEREIGN, IngestionFrequency.HOURLY): True,
    (AssetClassType.FIXED_INCOME_SOVEREIGN, IngestionFrequency.DAILY): True,
    (AssetClassType.FIXED_INCOME_CORPORATE, IngestionFrequency.HOURLY): True,
    (AssetClassType.FIXED_INCOME_CORPORATE, IngestionFrequency.DAILY): True,
    (AssetClassType.FIXED_INCOME_STRUCTURED, IngestionFrequency.DAILY): True,
    (AssetClassType.CURRENCIES, IngestionFrequency.FIVE_MIN): True,
    (AssetClassType.CURRENCIES, IngestionFrequency.HOURLY): True,
    (AssetClassType.CURRENCIES, IngestionFrequency.DAILY): True,
    (AssetClassType.ECONOMETRICS, IngestionFrequency.DAILY): True,
    (AssetClassType.ECONOMETRICS, IngestionFrequency.WEEKLY): True,
    (AssetClassType.ECONOMETRICS, IngestionFrequency.MONTHLY): True,
    (AssetClassType.SEC_FILINGS, IngestionFrequency.WEEKLY): True,
    (AssetClassType.SEC_FILINGS, IngestionFrequency.MONTHLY): True,
    (AssetClassType.OPTIONS, IngestionFrequency.ONE_MIN): True,
    (AssetClassType.OPTIONS, IngestionFrequency.FIVE_MIN): True,
    (AssetClassType.FUTURES, IngestionFrequency.ONE_MIN): True,
    (AssetClassType.FUTURES, IngestionFrequency.FIVE_MIN): True,
}


@dataclass
class IngestionState:
    """Tracks the state of the ingestion orchestrator."""
    is_running: bool = False
    started_at: str = ""
    last_heartbeat: str = ""
    cycle_count: int = 0
    total_records_ingested: int = 0
    errors_count: int = 0
    last_error: str = ""
    last_ingestion: dict = field(default_factory=dict)
    quality_reports: dict = field(default_factory=dict)
    data_store: dict = field(default_factory=dict)
    pending_routes: list = field(default_factory=list)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class DataIngestionOrchestrator:
    """Continuous multi-asset data ingestion orchestrator for Metadron Capital.

    Ingests data across all permitted asset classes (NO CRYPTO), validates
    quality, routes to appropriate engine layers (L1-L7), and feeds into
    the equity pooling tool.

    All data sourced via OpenBB (34+ providers: FRED, SEC, Polygon, FMP,
    Intrinio, CBOE, ECB, OECD, etc.).

    Usage::

        orchestrator = DataIngestionOrchestrator()
        orchestrator.run_continuous_loop()  # blocking

        # Or single-shot ingestion:
        equities_df = orchestrator.ingest_equities()
        macro_df = orchestrator.ingest_econometrics()
    """

    def __init__(
        self,
        universe_engine=None,
        heartbeat_seconds: float = 60.0,
        enable_ftse: bool = True,
        options_watchlist=None,
        log_level: int = logging.INFO,
    ):
        """Initialize the data ingestion orchestrator.

        Args:
            universe_engine: Optional pre-loaded UniverseEngine instance.
            heartbeat_seconds: Main loop heartbeat interval in seconds.
            enable_ftse: Whether to include FTSE 100 equities.
            options_watchlist: Override list of tickers for options ingestion.
            log_level: Logging level for the orchestrator logger.
        """
        self._state = IngestionState()
        self._heartbeat_seconds = heartbeat_seconds
        self._enable_ftse = enable_ftse
        self._options_watchlist = options_watchlist or list(OPTIONS_DEFAULT_WATCHLIST)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._universe_engine = universe_engine
        self._us_tickers: list = []
        self._ftse_tickers: list = list(FTSE_100_TICKERS)

        logger.setLevel(log_level)

        self._engine_callbacks: dict = {layer.value: [] for layer in EngineLayer}
        self._pooling_callback = None

        logger.info(
            "DataIngestionOrchestrator initialized | heartbeat=%ds | ftse=%s | "
            "options_watchlist=%d tickers | openbb=%s",
            heartbeat_seconds, enable_ftse, len(self._options_watchlist),
            _openbb_available,
        )

    # ------------------------------------------------------------------
    # Universe loading
    # ------------------------------------------------------------------

    def _ensure_universe(self) -> None:
        """Ensure US equity universe is loaded."""
        if self._us_tickers:
            return
        if _CROSS_ASSET_AVAILABLE:
            self._us_tickers = list(get_all_static_tickers())
            logger.info(
                "US equity universe loaded from cross_asset_universe: %d tickers",
                len(self._us_tickers),
            )
        elif self._universe_engine is not None:
            try:
                secs = self._universe_engine.get_all()
                self._us_tickers = [s.ticker for s in secs]
                logger.info(
                    "US equity universe loaded from UniverseEngine: %d tickers",
                    len(self._us_tickers),
                )
            except Exception as e:
                logger.warning("UniverseEngine.get_all() failed: %s", e)
        if not self._us_tickers:
            self._us_tickers = list(SP500_TICKERS) if SP500_TICKERS else [
                "SPY", "QQQ", "IWM",
            ]
            logger.warning(
                "Using minimal equity universe fallback: %d tickers",
                len(self._us_tickers),
            )

    # ------------------------------------------------------------------
    # Registration hooks
    # ------------------------------------------------------------------

    def register_engine_callback(self, layer, callback) -> None:
        """Register a callback to receive routed data for a specific engine layer.

        The callback signature should be:
            callback(asset_class: str, data: pd.DataFrame, metadata: dict) -> None
        """
        layer_val = layer.value if isinstance(layer, EngineLayer) else str(layer)
        if layer_val not in self._engine_callbacks:
            self._engine_callbacks[layer_val] = []
        self._engine_callbacks[layer_val].append(callback)
        logger.debug("Registered callback for %s", layer_val)

    def register_pooling_callback(self, callback) -> None:
        """Register the equity pooling tool callback.

        Signature: callback(data: dict[str, pd.DataFrame]) -> None
        """
        self._pooling_callback = callback
        logger.debug("Registered equity pooling callback")

    # ------------------------------------------------------------------
    # CRYPTO GUARD
    # ------------------------------------------------------------------

    def _filter_crypto(self, tickers: list) -> list:
        """Remove any crypto tickers. Hard constraint: NO CRYPTO."""
        clean = [t for t in tickers if not _is_crypto(t)]
        removed = len(tickers) - len(clean)
        if removed > 0:
            logger.warning(
                "CRYPTO GUARD: blocked %d crypto ticker(s) from ingestion", removed,
            )
        return clean

    # ------------------------------------------------------------------
    # EQUITIES INGESTION
    # ------------------------------------------------------------------

    def ingest_equities(self, interval: str = "1d", lookback_days: int = 5) -> dict:
        """Ingest equity price data for the full US S&P 1500 + FTSE 100 universe.

        Args:
            interval: Bar interval ('1d', '1h', '5m', '1m').
            lookback_days: Number of days of history to fetch.

        Returns:
            Dict with keys 'us' and 'ftse', each containing a price DataFrame.
        """
        self._ensure_universe()
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        result = {"us": pd.DataFrame(), "ftse": pd.DataFrame()}

        # --- US Equities (S&P 1500) ---
        us_tickers = self._filter_crypto(self._us_tickers)
        logger.info(
            "Ingesting US equities | %d tickers | interval=%s | lookback=%dd",
            len(us_tickers), interval, lookback_days,
        )
        try:
            batch_size = 50
            frames = []
            for i in range(0, len(us_tickers), batch_size):
                batch = us_tickers[i:i + batch_size]
                try:
                    df = get_prices(batch, start=start, interval=interval)
                    if not df.empty:
                        frames.append(df)
                except Exception as e:
                    logger.warning(
                        "US equity batch %d-%d failed: %s", i, i + len(batch), e,
                    )
            if frames:
                result["us"] = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
                result["us"] = result["us"].loc[:, ~result["us"].columns.duplicated()]
        except Exception as e:
            logger.error("US equity ingestion failed: %s", e)
            self._state.errors_count += 1
            self._state.last_error = f"US equities: {e}"

        qr = _check_data_quality(result["us"], AssetClassType.EQUITIES_US.value)
        self._state.quality_reports[AssetClassType.EQUITIES_US.value] = qr
        logger.info("US equities: %d rows, quality=%s", qr.row_count, qr.status)

        # --- FTSE 100 ---
        if self._enable_ftse and self._ftse_tickers:
            ftse_tickers = self._filter_crypto(self._ftse_tickers)
            logger.info(
                "Ingesting FTSE 100 | %d tickers | interval=%s",
                len(ftse_tickers), interval,
            )
            try:
                ftse_df = get_prices(ftse_tickers, start=start, interval=interval)
                if not ftse_df.empty:
                    result["ftse"] = ftse_df
            except Exception as e:
                logger.warning("FTSE 100 ingestion failed: %s", e)
                self._state.errors_count += 1

            qr_ftse = _check_data_quality(
                result["ftse"], AssetClassType.EQUITIES_FTSE.value,
            )
            self._state.quality_reports[AssetClassType.EQUITIES_FTSE.value] = qr_ftse
            logger.info(
                "FTSE 100: %d rows, quality=%s", qr_ftse.row_count, qr_ftse.status,
            )

        self._state.last_ingestion[AssetClassType.EQUITIES_US.value] = (
            datetime.utcnow().isoformat()
        )
        self._state.total_records_ingested += sum(
            len(df) for df in result.values() if not df.empty
        )

        self.route_to_engines(AssetClassType.EQUITIES_US, result["us"])
        if not result["ftse"].empty:
            self.route_to_engines(AssetClassType.EQUITIES_FTSE, result["ftse"])
        self._feed_pooling(result)

        return result

    # ------------------------------------------------------------------
    # COMMODITIES INGESTION
    # ------------------------------------------------------------------

    def ingest_commodities(self, interval: str = "1d", lookback_days: int = 30) -> pd.DataFrame:
        """Ingest major commodity ETF prices.

        Used for: price reference, cyclical patterns, global trade signals.
        NOT for direct trading — macro analysis only.
        """
        tickers = self._filter_crypto(list(COMMODITY_ETF_UNIVERSE.keys()))
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info("Ingesting commodities | %d ETFs | interval=%s", len(tickers), interval)

        df = pd.DataFrame()
        try:
            df = get_prices(tickers, start=start, interval=interval)
        except Exception as e:
            logger.error("Commodity ingestion failed: %s", e)
            self._state.errors_count += 1
            self._state.last_error = f"Commodities: {e}"

        qr = _check_data_quality(df, AssetClassType.COMMODITIES.value)
        self._state.quality_reports[AssetClassType.COMMODITIES.value] = qr
        self._state.last_ingestion[AssetClassType.COMMODITIES.value] = datetime.utcnow().isoformat()
        logger.info("Commodities: %d rows, quality=%s", qr.row_count, qr.status)

        self.route_to_engines(AssetClassType.COMMODITIES, df)
        return df

    # ------------------------------------------------------------------
    # INDICES INGESTION
    # ------------------------------------------------------------------

    def ingest_indices(self, interval: str = "1d", lookback_days: int = 30) -> pd.DataFrame:
        """Ingest major index ETF prices for benchmarking and rebalancing."""
        tickers = self._filter_crypto(list(INDEX_ETF_UNIVERSE.keys()))
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info("Ingesting indices | %d ETFs | interval=%s", len(tickers), interval)

        df = pd.DataFrame()
        try:
            df = get_prices(tickers, start=start, interval=interval)
        except Exception as e:
            logger.error("Index ingestion failed: %s", e)
            self._state.errors_count += 1
            self._state.last_error = f"Indices: {e}"

        qr = _check_data_quality(df, AssetClassType.INDICES.value)
        self._state.quality_reports[AssetClassType.INDICES.value] = qr
        self._state.last_ingestion[AssetClassType.INDICES.value] = datetime.utcnow().isoformat()
        logger.info("Indices: %d rows, quality=%s", qr.row_count, qr.status)

        self.route_to_engines(AssetClassType.INDICES, df)
        return df

    # ------------------------------------------------------------------
    # FIXED INCOME INGESTION
    # ------------------------------------------------------------------

    def ingest_fixed_income(self, lookback_days: int = 90) -> dict:
        """Ingest fixed income data across all sub-classes.

        Sub-classes:
            - Sovereign: G10 + India + Japan (via ETFs + FRED yields)
            - Corporate: US credit bonds only (IG via LQD/VCIT/VCSH, HY via HYG/JNK)
            - Structured: Major benchmarks (MBS via MBB/VMBS)

        Returns:
            Dict with keys 'sovereign', 'corporate', 'structured', 'yields', 'spreads'.
        """
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        result = {
            "sovereign": pd.DataFrame(),
            "corporate": pd.DataFrame(),
            "structured": pd.DataFrame(),
            "yields": pd.DataFrame(),
            "spreads": pd.DataFrame(),
        }

        logger.info("Ingesting fixed income | sovereign + corporate + structured")

        try:
            sov_tickers = self._filter_crypto(list(SOVEREIGN_BOND_UNIVERSE.keys()))
            df = get_prices(sov_tickers, start=start)
            if not df.empty:
                result["sovereign"] = df
        except Exception as e:
            logger.warning("Sovereign bond ETF ingestion failed: %s", e)

        try:
            yields = get_treasury_rates(start=start)
            if not yields.empty:
                result["yields"] = yields
        except Exception as e:
            logger.warning("Treasury yield ingestion failed: %s", e)

        try:
            corp_tickers = self._filter_crypto(list(CORPORATE_BOND_UNIVERSE.keys()))
            df = get_prices(corp_tickers, start=start)
            if not df.empty:
                result["corporate"] = df
        except Exception as e:
            logger.warning("Corporate bond ingestion failed: %s", e)

        try:
            spreads = get_credit_spreads(start=start)
            if not spreads.empty:
                result["spreads"] = spreads
        except Exception as e:
            logger.warning("Credit spread ingestion failed: %s", e)

        try:
            struct_tickers = self._filter_crypto(list(STRUCTURED_PRODUCT_UNIVERSE.keys()))
            df = get_prices(struct_tickers, start=start)
            if not df.empty:
                result["structured"] = df
        except Exception as e:
            logger.warning("Structured product ingestion failed: %s", e)

        for sub_class, df in result.items():
            ac_name = f"FIXED_INCOME_{sub_class.upper()}"
            qr = _check_data_quality(df, ac_name, stale_hours=48.0)
            self._state.quality_reports[ac_name] = qr

        now_str = datetime.utcnow().isoformat()
        self._state.last_ingestion[AssetClassType.FIXED_INCOME_SOVEREIGN.value] = now_str
        self._state.last_ingestion[AssetClassType.FIXED_INCOME_CORPORATE.value] = now_str
        self._state.last_ingestion[AssetClassType.FIXED_INCOME_STRUCTURED.value] = now_str

        total_rows = sum(len(df) for df in result.values() if not df.empty)
        self._state.total_records_ingested += total_rows
        logger.info("Fixed income: %d total rows across %d sub-classes", total_rows, len(result))

        self.route_to_engines(AssetClassType.FIXED_INCOME_SOVEREIGN, result["sovereign"])
        self.route_to_engines(AssetClassType.FIXED_INCOME_CORPORATE, result["corporate"])
        self.route_to_engines(AssetClassType.FIXED_INCOME_STRUCTURED, result["structured"])

        return result

    # ------------------------------------------------------------------
    # CURRENCIES INGESTION
    # ------------------------------------------------------------------

    def ingest_currencies(self, lookback_days: int = 90) -> dict:
        """Ingest currency data for G10 + India (INR) + Japan (JPY).

        Sources:
            - CurrencyShares ETFs (FXE, FXB, FXY, FXA, FXC, FXF, FXS, UUP)
            - FRED exchange rate series for G10 + INR
        """
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        result = {"etf_prices": pd.DataFrame(), "fred_rates": pd.DataFrame()}

        logger.info("Ingesting currencies | G10 + INR + JPY")

        try:
            fx_tickers = self._filter_crypto(list(CURRENCY_UNIVERSE.keys()))
            df = get_prices(fx_tickers, start=start)
            if not df.empty:
                result["etf_prices"] = df
        except Exception as e:
            logger.warning("Currency ETF ingestion failed: %s", e)

        try:
            fred_ids = list(CURRENCY_FRED_SERIES.keys())
            df = get_fred_series(fred_ids, start=start)
            if not df.empty:
                result["fred_rates"] = df
        except Exception as e:
            logger.warning("Currency FRED rates ingestion failed: %s", e)

        qr = _check_data_quality(result["etf_prices"], AssetClassType.CURRENCIES.value)
        self._state.quality_reports[AssetClassType.CURRENCIES.value] = qr
        self._state.last_ingestion[AssetClassType.CURRENCIES.value] = datetime.utcnow().isoformat()

        total_rows = sum(len(df) for df in result.values() if not df.empty)
        self._state.total_records_ingested += total_rows
        logger.info("Currencies: %d total rows", total_rows)

        self.route_to_engines(AssetClassType.CURRENCIES, result["etf_prices"])
        return result

    # ------------------------------------------------------------------
    # ECONOMETRICS INGESTION
    # ------------------------------------------------------------------

    def ingest_econometrics(self, lookback_years: int = 5) -> dict:
        """Ingest macro-economic data from FRED.

        Series include: GDP, CPI, PPI, NFP, ISM, PMI, Consumer Confidence,
        Retail Sales, Housing Starts, Industrial Production, Trade Balance,
        M2, Velocity, and 40+ additional FRED series.
        """
        start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

        logger.info(
            "Ingesting econometrics | %d FRED series | lookback=%dy",
            len(ECONOMETRIC_FRED_SERIES), lookback_years,
        )

        result = {
            "macro_series": pd.DataFrame(),
            "treasury_yields": pd.DataFrame(),
            "credit_spreads": pd.DataFrame(),
            "fed_balance_sheet": pd.DataFrame(),
            "monetary": pd.DataFrame(),
        }

        try:
            fred_ids = list(ECONOMETRIC_FRED_SERIES.keys())
            batch_size = 10
            frames = {}
            for i in range(0, len(fred_ids), batch_size):
                batch = fred_ids[i:i + batch_size]
                try:
                    df = get_fred_series(batch, start=start)
                    if not df.empty:
                        for col in df.columns:
                            frames[col] = df[col]
                except Exception as e:
                    logger.debug("FRED batch %d-%d failed: %s", i, i + len(batch), e)
            if frames:
                result["macro_series"] = pd.DataFrame(frames)
        except Exception as e:
            logger.error("FRED macro series ingestion failed: %s", e)
            self._state.errors_count += 1

        try:
            enriched = get_macro_data_enriched(start=start)
            if isinstance(enriched, dict):
                for key in ("treasury_yields", "credit_spreads", "fed_balance_sheet", "monetary"):
                    if key in enriched and not enriched[key].empty:
                        result[key] = enriched[key]
        except Exception as e:
            logger.warning("Enriched macro data failed: %s", e)

        qr = _check_data_quality(
            result["macro_series"], AssetClassType.ECONOMETRICS.value, stale_hours=72.0,
        )
        self._state.quality_reports[AssetClassType.ECONOMETRICS.value] = qr
        self._state.last_ingestion[AssetClassType.ECONOMETRICS.value] = datetime.utcnow().isoformat()

        total_rows = sum(len(df) for df in result.values() if not df.empty)
        self._state.total_records_ingested += total_rows
        n_series = len(result["macro_series"].columns) if not result["macro_series"].empty else 0
        logger.info("Econometrics: %d total rows, %d series", total_rows, n_series)

        self.route_to_engines(AssetClassType.ECONOMETRICS, result["macro_series"])
        return result

    # ------------------------------------------------------------------
    # SEC FILINGS INGESTION
    # ------------------------------------------------------------------

    def ingest_sec_filings(self, tickers=None, filing_types=None, limit_per_ticker: int = 5) -> dict:
        """Ingest SEC filings for universe securities.

        Only major updates: 10-K (annual), 10-Q (quarterly), 8-K (material events).
        Designed for monthly tracking cadence, not daily bulk download.
        """
        if tickers is None:
            tickers = list(SP500_TICKERS)[:100] if SP500_TICKERS else ["AAPL", "MSFT", "GOOGL"]
        tickers = self._filter_crypto(tickers)

        if filing_types is None:
            filing_types = list(SEC_FILING_TYPES)

        logger.info(
            "Ingesting SEC filings | %d tickers | types=%s | limit=%d/ticker",
            len(tickers), filing_types, limit_per_ticker,
        )

        result = {}
        for ftype in filing_types:
            frames = []
            for ticker in tickers:
                try:
                    df = get_company_filings(ticker, filing_type=ftype, limit=limit_per_ticker)
                    if not df.empty:
                        df["ticker"] = ticker
                        frames.append(df)
                except Exception as e:
                    logger.debug("SEC filing %s for %s failed: %s", ftype, ticker, e)
                time.sleep(0.1)

            result[ftype] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            logger.info(
                "SEC %s: %d filings from %d tickers",
                ftype, len(result[ftype]) if not result[ftype].empty else 0,
                len([f for f in frames if not f.empty]),
            )

        self._state.last_ingestion[AssetClassType.SEC_FILINGS.value] = datetime.utcnow().isoformat()
        total = sum(len(df) for df in result.values() if not df.empty)
        self._state.total_records_ingested += total

        combined = pd.concat(list(result.values()), ignore_index=True) if result else pd.DataFrame()
        self.route_to_engines(AssetClassType.SEC_FILINGS, combined)
        return result

    # ------------------------------------------------------------------
    # OPTIONS CHAIN INGESTION
    # ------------------------------------------------------------------

    def ingest_options_chain(self, tickers=None) -> dict:
        """Ingest options chain data for selected securities.

        Used opportunistically for alpha maximization. Tickers are selected
        from the AlphaOptimizer top picks or a configured watchlist.
        """
        if tickers is None:
            tickers = list(self._options_watchlist)
        tickers = self._filter_crypto(tickers)

        logger.info("Ingesting options chains | %d tickers", len(tickers))

        result = {}
        for ticker in tickers:
            try:
                df = get_options_chains(ticker)
                if not df.empty:
                    result[ticker] = df
                    logger.debug("Options %s: %d contracts", ticker, len(df))
            except Exception as e:
                logger.debug("Options chain for %s failed: %s", ticker, e)
            time.sleep(0.2)

        self._state.last_ingestion[AssetClassType.OPTIONS.value] = datetime.utcnow().isoformat()
        total = sum(len(df) for df in result.values())
        self._state.total_records_ingested += total
        logger.info("Options: %d chains, %d total contracts", len(result), total)

        for ticker, df in result.items():
            self.route_to_engines(
                AssetClassType.OPTIONS, df,
                metadata={"ticker": ticker, "type": "options_chain"},
            )
        return result

    # ------------------------------------------------------------------
    # FUTURES INGESTION
    # ------------------------------------------------------------------

    def ingest_futures(self, interval: str = "1d", lookback_days: int = 30) -> pd.DataFrame:
        """Ingest futures data for beta management within the beta corridor.

        Futures tracked: ES, NQ, YM, RTY (equity index), VX (volatility),
        ZN, ZB, ZF (treasury), GC, CL (commodity).
        """
        tickers = self._filter_crypto(list(FUTURES_UNIVERSE.keys()))
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info("Ingesting futures | %d contracts | interval=%s", len(tickers), interval)

        df = pd.DataFrame()
        try:
            df = get_prices(tickers, start=start, interval=interval)
        except Exception as e:
            logger.error("Futures ingestion failed: %s", e)
            self._state.errors_count += 1
            self._state.last_error = f"Futures: {e}"

        qr = _check_data_quality(df, AssetClassType.FUTURES.value)
        self._state.quality_reports[AssetClassType.FUTURES.value] = qr
        self._state.last_ingestion[AssetClassType.FUTURES.value] = datetime.utcnow().isoformat()
        if not df.empty:
            self._state.total_records_ingested += len(df)
        logger.info("Futures: %d rows, quality=%s", qr.row_count, qr.status)

        self.route_to_engines(AssetClassType.FUTURES, df)
        return df

    # ------------------------------------------------------------------
    # ENGINE ROUTING
    # ------------------------------------------------------------------

    def route_to_engines(self, asset_class, data, metadata=None) -> None:
        """Route ingested data to the appropriate engine layers.

        Routing is determined by the ASSET_CLASS_ROUTING map. Each registered
        callback for a target layer receives the data along with metadata.

        Args:
            asset_class: The AssetClassType of the ingested data.
            data: The ingested DataFrame.
            metadata: Optional metadata dict (ticker, type, etc.).
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return

        target_layers = ASSET_CLASS_ROUTING.get(asset_class, [])
        meta = metadata or {}
        meta["asset_class"] = asset_class.value if hasattr(asset_class, "value") else str(asset_class)
        meta["ingested_at"] = datetime.utcnow().isoformat()
        meta["row_count"] = len(data) if isinstance(data, pd.DataFrame) else 0

        for layer in target_layers:
            layer_val = layer.value if hasattr(layer, "value") else str(layer)
            callbacks = self._engine_callbacks.get(layer_val, [])
            for cb in callbacks:
                try:
                    cb(meta["asset_class"], data, meta)
                except Exception as e:
                    logger.error(
                        "Engine callback failed | layer=%s | asset=%s | error=%s",
                        layer_val, meta["asset_class"], e,
                    )

        store_key = meta["asset_class"]
        with self._lock:
            self._state.data_store[store_key] = {
                "data": data,
                "metadata": meta,
                "timestamp": datetime.utcnow().isoformat(),
            }

        logger.debug(
            "Routed %s to %d layers (%d callbacks total)",
            meta["asset_class"], len(target_layers),
            sum(len(self._engine_callbacks.get(
                l.value if hasattr(l, "value") else str(l), [],
            )) for l in target_layers),
        )

    def _feed_pooling(self, equity_data: dict) -> None:
        """Feed ingested equity data into the equity pooling tool."""
        if self._pooling_callback is None:
            return
        try:
            self._pooling_callback(equity_data)
            logger.debug("Fed equity data to pooling tool")
        except Exception as e:
            logger.error("Equity pooling feed failed: %s", e)

    # ------------------------------------------------------------------
    # SCHEDULED INGESTION CYCLE
    # ------------------------------------------------------------------

    def _should_run(self, asset_class, frequency) -> bool:
        """Check if a given (asset_class, frequency) pair should run now."""
        if not INGESTION_SCHEDULE.get((asset_class, frequency), False):
            return False
        ac_val = asset_class.value if hasattr(asset_class, "value") else str(asset_class)
        last_str = self._state.last_ingestion.get(ac_val)
        if not last_str:
            return True
        try:
            last_dt = datetime.fromisoformat(last_str)
        except (ValueError, TypeError):
            return True
        now = datetime.utcnow()
        elapsed = (now - last_dt).total_seconds()
        interval_map = {
            IngestionFrequency.REAL_TIME: 1,
            IngestionFrequency.ONE_MIN: 60,
            IngestionFrequency.FIVE_MIN: 300,
            IngestionFrequency.HOURLY: 3600,
            IngestionFrequency.DAILY: 86400,
            IngestionFrequency.WEEKLY: 604800,
            IngestionFrequency.MONTHLY: 2592000,
        }
        required_interval = interval_map.get(frequency, 86400)
        return elapsed >= required_interval

    def run_cycle(self, frequency) -> dict:
        """Run a single ingestion cycle at the specified frequency.

        Checks all asset classes against the schedule and ingests those
        that are due.
        """
        cycle_start = time.time()
        self._state.cycle_count += 1
        freq_val = frequency.value if hasattr(frequency, "value") else str(frequency)
        summary = {
            "cycle": self._state.cycle_count,
            "frequency": freq_val,
            "started_at": datetime.utcnow().isoformat(),
            "ingested": [],
            "skipped": [],
            "errors": [],
        }

        logger.info("=== INGESTION CYCLE %d | frequency=%s ===", self._state.cycle_count, freq_val)

        tasks = {
            IngestionFrequency.ONE_MIN: [
                (AssetClassType.EQUITIES_US, lambda: self.ingest_equities(interval="1m", lookback_days=1)),
                (AssetClassType.INDICES, lambda: self.ingest_indices(interval="1m", lookback_days=1)),
                (AssetClassType.OPTIONS, lambda: self.ingest_options_chain()),
                (AssetClassType.FUTURES, lambda: self.ingest_futures(interval="1m", lookback_days=1)),
            ],
            IngestionFrequency.FIVE_MIN: [
                (AssetClassType.EQUITIES_US, lambda: self.ingest_equities(interval="5m", lookback_days=1)),
                (AssetClassType.EQUITIES_FTSE, lambda: self.ingest_equities(interval="5m", lookback_days=1)),
                (AssetClassType.COMMODITIES, lambda: self.ingest_commodities(interval="5m", lookback_days=1)),
                (AssetClassType.CURRENCIES, lambda: self.ingest_currencies(lookback_days=5)),
            ],
            IngestionFrequency.HOURLY: [
                (AssetClassType.FIXED_INCOME_SOVEREIGN, lambda: self.ingest_fixed_income(lookback_days=5)),
                (AssetClassType.FIXED_INCOME_CORPORATE, lambda: self.ingest_fixed_income(lookback_days=5)),
                (AssetClassType.CURRENCIES, lambda: self.ingest_currencies(lookback_days=5)),
            ],
            IngestionFrequency.DAILY: [
                (AssetClassType.EQUITIES_US, lambda: self.ingest_equities(interval="1d", lookback_days=5)),
                (AssetClassType.EQUITIES_FTSE, lambda: self.ingest_equities(interval="1d", lookback_days=5)),
                (AssetClassType.COMMODITIES, lambda: self.ingest_commodities(interval="1d", lookback_days=30)),
                (AssetClassType.INDICES, lambda: self.ingest_indices(interval="1d", lookback_days=30)),
                (AssetClassType.FIXED_INCOME_SOVEREIGN, lambda: self.ingest_fixed_income(lookback_days=90)),
                (AssetClassType.FIXED_INCOME_CORPORATE, lambda: self.ingest_fixed_income(lookback_days=90)),
                (AssetClassType.FIXED_INCOME_STRUCTURED, lambda: self.ingest_fixed_income(lookback_days=90)),
                (AssetClassType.CURRENCIES, lambda: self.ingest_currencies(lookback_days=90)),
                (AssetClassType.ECONOMETRICS, lambda: self.ingest_econometrics(lookback_years=2)),
                (AssetClassType.FUTURES, lambda: self.ingest_futures(interval="1d", lookback_days=30)),
            ],
            IngestionFrequency.WEEKLY: [
                (AssetClassType.SEC_FILINGS, lambda: self.ingest_sec_filings()),
                (AssetClassType.ECONOMETRICS, lambda: self.ingest_econometrics(lookback_years=5)),
            ],
            IngestionFrequency.MONTHLY: [
                (AssetClassType.INDICES, lambda: self.ingest_indices(interval="1d", lookback_days=365)),
                (AssetClassType.SEC_FILINGS, lambda: self.ingest_sec_filings(limit_per_ticker=10)),
                (AssetClassType.ECONOMETRICS, lambda: self.ingest_econometrics(lookback_years=10)),
            ],
        }

        for asset_class, task_fn in tasks.get(frequency, []):
            ac_val = asset_class.value if hasattr(asset_class, "value") else str(asset_class)
            if not self._should_run(asset_class, frequency):
                summary["skipped"].append(ac_val)
                continue
            try:
                task_fn()
                summary["ingested"].append(ac_val)
            except Exception as e:
                logger.error("Ingestion failed for %s: %s", ac_val, e)
                summary["errors"].append({"asset_class": ac_val, "error": str(e)})
                self._state.errors_count += 1
                self._state.last_error = f"{ac_val}: {e}"

        elapsed = time.time() - cycle_start
        summary["elapsed_seconds"] = round(elapsed, 2)
        logger.info(
            "=== CYCLE %d COMPLETE | %.1fs | ingested=%d | skipped=%d | errors=%d ===",
            self._state.cycle_count, elapsed,
            len(summary["ingested"]), len(summary["skipped"]), len(summary["errors"]),
        )
        return summary

    # ------------------------------------------------------------------
    # CONTINUOUS LOOP
    # ------------------------------------------------------------------

    def run_continuous_loop(self) -> None:
        """Main continuous ingestion loop.

        Runs indefinitely (until stop() is called), cycling through
        ingestion frequencies based on elapsed time:

            - Every 1 minute: 1MIN cycle
            - Every 5 minutes: 5MIN cycle
            - Every hour: HOURLY cycle
            - Every day: DAILY cycle
            - Every week: WEEKLY cycle
            - Every month: MONTHLY cycle

        The loop heartbeat is configurable via heartbeat_seconds (default 60s).
        """
        self._state.is_running = True
        self._state.started_at = datetime.utcnow().isoformat()
        self._stop_event.clear()

        logger.info(
            "========================================================\n"
            "  METADRON CAPITAL — DATA INGESTION ORCHESTRATOR STARTED\n"
            "  Heartbeat: %ds | OpenBB: %s | FTSE: %s\n"
            "  NO CRYPTO — hard exclusion enforced\n"
            "========================================================",
            self._heartbeat_seconds, _openbb_available, self._enable_ftse,
        )

        last_run = {}

        # Initial full daily cycle
        try:
            self.run_cycle(IngestionFrequency.DAILY)
            last_run[IngestionFrequency.DAILY.value] = datetime.utcnow()
        except Exception as e:
            logger.error("Initial daily cycle failed: %s", e)

        while not self._stop_event.is_set():
            self._state.last_heartbeat = datetime.utcnow().isoformat()
            now = datetime.utcnow()

            try:
                frequency_intervals = {
                    IngestionFrequency.ONE_MIN: timedelta(minutes=1),
                    IngestionFrequency.FIVE_MIN: timedelta(minutes=5),
                    IngestionFrequency.HOURLY: timedelta(hours=1),
                    IngestionFrequency.DAILY: timedelta(days=1),
                    IngestionFrequency.WEEKLY: timedelta(weeks=1),
                    IngestionFrequency.MONTHLY: timedelta(days=30),
                }

                for freq, interval in frequency_intervals.items():
                    last = last_run.get(freq.value)
                    if last is None or (now - last) >= interval:
                        try:
                            self.run_cycle(freq)
                            last_run[freq.value] = now
                        except Exception as e:
                            logger.error("Cycle %s failed: %s", freq.value, e)
                            self._state.errors_count += 1

            except Exception as e:
                logger.error("Continuous loop iteration error: %s", e)
                self._state.errors_count += 1

            self._stop_event.wait(timeout=self._heartbeat_seconds)

        self._state.is_running = False
        logger.info(
            "Ingestion orchestrator stopped | cycles=%d | records=%d | errors=%d",
            self._state.cycle_count, self._state.total_records_ingested,
            self._state.errors_count,
        )

    def stop(self) -> None:
        """Signal the continuous loop to stop gracefully."""
        logger.info("Stop signal received — shutting down ingestion orchestrator")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # DATA ACCESS
    # ------------------------------------------------------------------

    def get_latest(self, asset_class):
        """Retrieve the latest ingested data for an asset class."""
        ac_val = asset_class.value if hasattr(asset_class, "value") else str(asset_class)
        with self._lock:
            entry = self._state.data_store.get(ac_val)
            if entry:
                return entry.get("data")
        return None

    def get_quality_report(self, asset_class: str):
        """Retrieve the latest quality report for an asset class."""
        return self._state.quality_reports.get(asset_class)

    # ------------------------------------------------------------------
    # STATUS & DIAGNOSTICS
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return comprehensive orchestrator status."""
        return {
            "is_running": self._state.is_running,
            "started_at": self._state.started_at,
            "last_heartbeat": self._state.last_heartbeat,
            "cycle_count": self._state.cycle_count,
            "total_records_ingested": self._state.total_records_ingested,
            "errors_count": self._state.errors_count,
            "last_error": self._state.last_error,
            "openbb_available": _openbb_available,
            "data_module_available": _DATA_MODULE_AVAILABLE,
            "universe_available": _UNIVERSE_AVAILABLE,
            "us_tickers_loaded": len(self._us_tickers),
            "ftse_tickers_loaded": len(self._ftse_tickers),
            "ftse_enabled": self._enable_ftse,
            "options_watchlist_size": len(self._options_watchlist),
            "last_ingestion": dict(self._state.last_ingestion),
            "quality_summary": {
                k: v.status for k, v in self._state.quality_reports.items()
            },
            "data_store_keys": list(self._state.data_store.keys()),
            "registered_callbacks": {
                layer: len(cbs) for layer, cbs in self._engine_callbacks.items()
            },
            "excluded_assets": sorted(EXCLUDED_ASSET_CLASSES),
            "asset_classes_active": [ac.value for ac in AssetClassType],
        }

    def summary(self) -> str:
        """Return a human-readable summary of orchestrator state."""
        s = self.status()
        lines = [
            "=" * 70,
            "  METADRON CAPITAL — DATA INGESTION ORCHESTRATOR STATUS",
            "=" * 70,
            f"  Running:            {s['is_running']}",
            f"  Cycles completed:   {s['cycle_count']}",
            f"  Records ingested:   {s['total_records_ingested']:,}",
            f"  Errors:             {s['errors_count']}",
            f"  OpenBB available:   {s['openbb_available']}",
            f"  US tickers:         {s['us_tickers_loaded']:,}",
            f"  FTSE 100 tickers:   {s['ftse_tickers_loaded']}",
            f"  FTSE enabled:       {s['ftse_enabled']}",
            f"  Options watchlist:  {s['options_watchlist_size']}",
            "",
            "  ASSET CLASS QUALITY:",
        ]
        for ac, qstatus in s.get("quality_summary", {}).items():
            lines.append(f"    {ac:<35} {qstatus}")
        lines.append("")
        lines.append("  LAST INGESTION:")
        for ac, ts in s.get("last_ingestion", {}).items():
            lines.append(f"    {ac:<35} {ts}")
        lines.append("")
        lines.append(f"  EXCLUDED: {', '.join(sorted(EXCLUDED_ASSET_CLASSES))}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

_ORCHESTRATOR_INSTANCE = None


def get_orchestrator(**kwargs):
    """Get or create the singleton DataIngestionOrchestrator instance."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        _ORCHESTRATOR_INSTANCE = DataIngestionOrchestrator(**kwargs)
    return _ORCHESTRATOR_INSTANCE


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    orchestrator = DataIngestionOrchestrator(heartbeat_seconds=60)
    print(orchestrator.summary())

    logger.info("Running single-shot ingestion test...")
    orchestrator.ingest_equities(lookback_days=2)
    orchestrator.ingest_commodities(lookback_days=5)
    orchestrator.ingest_indices(lookback_days=5)
    orchestrator.ingest_fixed_income(lookback_days=30)
    orchestrator.ingest_currencies(lookback_days=30)
    orchestrator.ingest_econometrics(lookback_years=1)
    orchestrator.ingest_futures(lookback_days=5)

    print("\n" + orchestrator.summary())
