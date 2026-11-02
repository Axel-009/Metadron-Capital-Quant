"""Universal Data Pooling Engine — Cross-asset data aggregation and routing.

Upgrade of the EquityLinkedGICPooling concept to cover ALL asset classes in the
Metadron Capital platform. Aggregates ingested data from every asset class and
routes it to the correct layers in the architecture.

Layer Architecture:
    L1 Data           Market data + factor research
    L2 Signals        Strategy library + regime classifier
    L3 Intelligence   Multi-agent decision engine
    L4 Portfolio      Institutional tracker + orchestration
    L5 Infrastructure ML serving + GPU inference
    L6 Agents         Multi-agent orchestration
    L7 HFT/Execution  Technical strategies + order matching

Asset Classes Pooled:
    1. Equities      US S&P 1500 + FTSE 100 -> L1, L2, L3
    2. Commodities   Major ETFs (GLD, SLV, USO, ...) -> L1, L2 cyclical patterns
    3. Indices       SPY, QQQ, IWM, DIA, VT, EFA, EEM -> L1 benchmarks, L4 rebalancing
    4. Fixed Income  G10+India+Japan sovereign, US corporate IG/HY -> L2, MetadronCube
    5. Currencies    G10+INR+JPY -> L2 macro, MetadronCube FX layer
    6. Econometrics  FRED data (GDP, CPI, M2, etc.) -> L2 MacroEngine, MetadronCube
    7. SEC Filings   10-K, 10-Q, 8-K -> L2 event engine, L3 intelligence
    8. Options       Selected securities -> L7 execution, OptionsEngine
    9. Futures       Index futures for beta management -> L4 BetaCorridor

Feeds downstream engines:
    - MetadronCube LiquidityTensor + FedPlumbingLayer
    - MacroEngine econometric inputs
    - SecurityAnalysisEngine fundamental data
    - OptionsEngine options chain data
    - BetaCorridor index futures data

All data via OpenBB -- sole data source.
try/except on ALL external imports -- system runs degraded, never broken.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Graceful external imports
# ---------------------------------------------------------------------------
try:
    from .yahoo_data import (
        get_adj_close, get_returns, get_prices,
        get_fundamentals, get_bulk_fundamentals,
    )
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_prices(*a, **kw): return pd.DataFrame()
    def get_fundamentals(*a, **kw): return {}
    def get_bulk_fundamentals(*a, **kw): return {}

try:
    from .openbb_data import get_macro_data, get_fred_series
except ImportError:
    def get_macro_data(*a, **kw): return pd.DataFrame()
    def get_fred_series(*a, **kw): return pd.DataFrame()

try:
    from .cross_asset_universe import (
        SP500_TICKERS, SP400_TICKERS, SP600_TICKERS, EXTRA_TICKERS,
        SECTOR_MAP, get_sector_for_ticker,
    )
except ImportError:
    SP500_TICKERS = []
    SP400_TICKERS = []
    SP600_TICKERS = []
    EXTRA_TICKERS = []
    SECTOR_MAP = {}
    def get_sector_for_ticker(t): return ""

try:
    from .universe_engine import GICPoolingEngine, Security, GICS_SECTORS
except ImportError:
    GICPoolingEngine = None
    Security = None
    GICS_SECTORS = {}

try:
    from ..signals.metadron_cube import LiquidityTensor, FedPlumbingLayer
except ImportError:
    LiquidityTensor = None
    FedPlumbingLayer = None

try:
    from ..signals.macro_engine import MacroEngine, MacroSnapshot
except ImportError:
    MacroEngine = None
    MacroSnapshot = None

try:
    from ..signals.security_analysis_engine import SecurityAnalysisEngine
except ImportError:
    SecurityAnalysisEngine = None

try:
    from ..execution.options_engine import OptionsEngine
except ImportError:
    OptionsEngine = None

try:
    from ..portfolio.beta_corridor import BetaCorridor
except ImportError:
    BetaCorridor = None

logger = logging.getLogger("metadron.scan.data_pool")


# =============================================================================
# Constants — Asset class ticker universes
# =============================================================================

class AssetClass(str, Enum):
    """All asset classes tracked by the universal pooling engine."""
    EQUITIES = "EQUITIES"
    COMMODITIES = "COMMODITIES"
    INDICES = "INDICES"
    FIXED_INCOME = "FIXED_INCOME"
    CURRENCIES = "CURRENCIES"
    ECONOMETRICS = "ECONOMETRICS"
    SEC_FILINGS = "SEC_FILINGS"
    OPTIONS = "OPTIONS"
    FUTURES = "FUTURES"


class LayerID(str, Enum):
    """Platform architecture layers."""
    L1_DATA = "L1_DATA"
    L2_SIGNALS = "L2_SIGNALS"
    L3_INTELLIGENCE = "L3_INTELLIGENCE"
    L4_PORTFOLIO = "L4_PORTFOLIO"
    L5_INFRASTRUCTURE = "L5_INFRASTRUCTURE"
    L6_AGENTS = "L6_AGENTS"
    L7_HFT_EXECUTION = "L7_HFT_EXECUTION"


class MaturityBucket(str, Enum):
    """Fixed income maturity bucketing (GIC-style)."""
    SHORT = "SHORT"          # 0-2 years
    INTERMEDIATE = "INTERMEDIATE"  # 2-5 years
    MEDIUM = "MEDIUM"        # 5-10 years
    LONG = "LONG"            # 10-20 years
    ULTRA_LONG = "ULTRA_LONG"  # 20-30+ years


class CreditTier(str, Enum):
    """Fixed income credit quality tier."""
    SOVEREIGN = "SOVEREIGN"
    INVESTMENT_GRADE = "INVESTMENT_GRADE"
    HIGH_YIELD = "HIGH_YIELD"
    STRUCTURED = "STRUCTURED"


class FIGeography(str, Enum):
    """Fixed income geographic classification."""
    US = "US"
    G10 = "G10"
    INDIA = "INDIA"
    JAPAN = "JAPAN"
    EMERGING = "EMERGING"


# ---------------------------------------------------------------------------
# Ticker universes per asset class
# ---------------------------------------------------------------------------

COMMODITY_ETFS = {
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Crude Oil",
    "UNG": "Natural Gas",
    "DBA": "Agriculture",
    "DBC": "Commodity Index",
    "COPX": "Copper Miners",
    "WEAT": "Wheat",
    "CORN": "Corn",
}

INDEX_ETFS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "VT": "Total World",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
}

# FTSE 100 — representative large caps (ADRs and London-listed)
FTSE_100_REPRESENTATIVE = [
    "SHEL", "AZN", "HSBC", "UL", "BP", "RIO", "GSK", "BHP",
    "BTI", "DEO", "LIN", "NVS", "VOD",
]

# Fixed income ETF proxies by maturity bucket and credit tier
FI_ETF_UNIVERSE = {
    # US Treasuries
    ("US", "SOVEREIGN", "SHORT"): ["SHV", "BIL", "SHY"],
    ("US", "SOVEREIGN", "INTERMEDIATE"): ["IEI", "VGIT"],
    ("US", "SOVEREIGN", "MEDIUM"): ["IEF", "GOVT"],
    ("US", "SOVEREIGN", "LONG"): ["TLH"],
    ("US", "SOVEREIGN", "ULTRA_LONG"): ["TLT", "EDV", "ZROZ"],
    # US Corporate
    ("US", "INVESTMENT_GRADE", "INTERMEDIATE"): ["LQD", "VCIT", "IGIB"],
    ("US", "HIGH_YIELD", "INTERMEDIATE"): ["HYG", "JNK", "USHY"],
    # Structured / securitized
    ("US", "STRUCTURED", "INTERMEDIATE"): ["MBB", "VMBS"],
    # International sovereign
    ("G10", "SOVEREIGN", "INTERMEDIATE"): ["BWX", "IGOV"],
    ("JAPAN", "SOVEREIGN", "MEDIUM"): ["BNDX"],
    ("INDIA", "SOVEREIGN", "MEDIUM"): ["LEMB"],
    ("EMERGING", "SOVEREIGN", "INTERMEDIATE"): ["EMB", "VWOB"],
}

# Flatten all FI tickers for data fetching
ALL_FI_TICKERS = sorted({t for tickers in FI_ETF_UNIVERSE.values() for t in tickers})

# G10 currency pairs + INR + JPY
FX_PAIRS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    "USDCHF=X": "USD/CHF",
    "USDSEK=X": "USD/SEK",
    "USDNOK=X": "USD/NOK",
    "USDINR=X": "USD/INR",
}

# FRED econometric series IDs
FRED_SERIES = {
    "GDP": "Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index",
    "M2SL": "M2 Money Supply",
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Spread",
    "SOFR": "Secured Overnight Financing Rate",
    "WALCL": "Fed Total Assets",
    "RRPONTSYD": "ON-RRP Balance",
    "WTREGEN": "Treasury General Account",
    "BAMLH0A0HYM2": "HY OAS Spread",
    "BAMLC0A4CBBB": "BBB Spread",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "INDPRO": "Industrial Production",
    "UMCSENT": "Consumer Sentiment",
    "VIXCLS": "VIX Close",
    "DCOILWTICO": "WTI Crude Oil",
    "DEXUSEU": "EUR/USD Exchange Rate",
}

# SEC filing types
SEC_FILING_TYPES = ["10-K", "10-Q", "8-K"]

# Futures proxies (ETF-based for paper trading)
FUTURES_PROXIES = {
    "ES=F": ("S&P 500 E-mini", "SPY"),
    "NQ=F": ("NASDAQ E-mini", "QQQ"),
    "YM=F": ("Dow E-mini", "DIA"),
    "RTY=F": ("Russell 2000 E-mini", "IWM"),
}

# Asset class -> layer routing map
ASSET_LAYER_ROUTING = {
    AssetClass.EQUITIES: [LayerID.L1_DATA, LayerID.L2_SIGNALS, LayerID.L3_INTELLIGENCE],
    AssetClass.COMMODITIES: [LayerID.L1_DATA, LayerID.L2_SIGNALS],
    AssetClass.INDICES: [LayerID.L1_DATA, LayerID.L4_PORTFOLIO],
    AssetClass.FIXED_INCOME: [LayerID.L2_SIGNALS],
    AssetClass.CURRENCIES: [LayerID.L2_SIGNALS],
    AssetClass.ECONOMETRICS: [LayerID.L2_SIGNALS],
    AssetClass.SEC_FILINGS: [LayerID.L2_SIGNALS, LayerID.L3_INTELLIGENCE],
    AssetClass.OPTIONS: [LayerID.L7_HFT_EXECUTION],
    AssetClass.FUTURES: [LayerID.L4_PORTFOLIO],
}

# Data staleness thresholds (seconds)
STALENESS_THRESHOLDS = {
    AssetClass.EQUITIES: 300,       # 5 min during market hours
    AssetClass.COMMODITIES: 300,
    AssetClass.INDICES: 60,         # 1 min for benchmarks
    AssetClass.FIXED_INCOME: 900,   # 15 min
    AssetClass.CURRENCIES: 120,     # 2 min
    AssetClass.ECONOMETRICS: 86400, # 1 day for macro
    AssetClass.SEC_FILINGS: 3600,   # 1 hour
    AssetClass.OPTIONS: 60,         # 1 min for greeks
    AssetClass.FUTURES: 60,         # 1 min for execution
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class AssetPartition:
    """A single asset class partition within the universal pool."""
    asset_class: AssetClass
    tickers: list = field(default_factory=list)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    record_count: int = 0
    is_stale: bool = True

    def age_seconds(self) -> float:
        """Seconds since last update."""
        if self.last_updated is None:
            return float("inf")
        return (datetime.now() - self.last_updated).total_seconds()

    def check_staleness(self) -> bool:
        """Check if this partition's data is stale."""
        threshold = STALENESS_THRESHOLDS.get(self.asset_class, 300)
        self.is_stale = self.age_seconds() > threshold
        return self.is_stale


@dataclass
class FIPartition:
    """Fixed income partition with maturity bucketing."""
    geography: FIGeography = FIGeography.US
    credit_tier: CreditTier = CreditTier.SOVEREIGN
    maturity_bucket: MaturityBucket = MaturityBucket.INTERMEDIATE
    tickers: list = field(default_factory=list)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    yield_level: float = 0.0
    spread_to_treasury: float = 0.0
    duration_estimate: float = 0.0
    last_updated: Optional[datetime] = None


@dataclass
class SectorAllocation:
    """Sector-level allocation signal."""
    sector: str = ""
    weight: float = 0.0
    momentum_1m: float = 0.0
    momentum_3m: float = 0.0
    momentum_12m: float = 0.0
    relative_strength: float = 0.0
    ticker_count: int = 0
    overweight: bool = False


@dataclass
class FreshnessReport:
    """Data freshness/staleness report across all partitions."""
    timestamp: str = ""
    total_partitions: int = 0
    fresh_count: int = 0
    stale_count: int = 0
    partition_details: dict = field(default_factory=dict)
    oldest_partition: str = ""
    oldest_age_seconds: float = 0.0


@dataclass
class LayerPayload:
    """Data payload routed to a specific architecture layer."""
    layer_id: LayerID = LayerID.L1_DATA
    asset_classes: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    timestamp: str = ""
    record_count: int = 0


@dataclass
class PoolSummary:
    """Summary of the entire universal data pool."""
    timestamp: str = ""
    total_tickers: int = 0
    total_records: int = 0
    asset_class_counts: dict = field(default_factory=dict)
    fresh_partitions: int = 0
    stale_partitions: int = 0
    sector_breakdown: dict = field(default_factory=dict)
    fi_maturity_breakdown: dict = field(default_factory=dict)


# =============================================================================
# Universal Data Pool
# =============================================================================

class UniversalDataPool:
    """Cross-asset data aggregation and routing engine.

    Pools all ingested data into a unified DataPool with asset class partitions
    and routes data to the correct layers in the Metadron Capital architecture.

    Inherits the GIC pooling methodology and extends it to cover equities,
    commodities, indices, fixed income, currencies, econometrics, SEC filings,
    options, and futures.
    """

    def __init__(self, lookback_days: int = 252, fetch_on_init: bool = False):
        """Initialise the universal data pool.

        Args:
            lookback_days: Default lookback period for historical data.
            fetch_on_init: If True, immediately fetch all data on construction.
        """
        self._lookback_days = lookback_days
        self._start_date = (
            datetime.now() - timedelta(days=lookback_days + 30)
        ).strftime("%Y-%m-%d")

        # Partitions keyed by AssetClass
        self._partitions: dict[AssetClass, AssetPartition] = {}

        # Fixed income sub-partitions keyed by (geography, credit, maturity)
        self._fi_partitions: dict[tuple, FIPartition] = {}

        # Cross-asset correlation matrix
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_timestamp: Optional[datetime] = None

        # GIC pooling engine for equity sector pools
        self._gic_engine = GICPoolingEngine() if GICPoolingEngine else None

        # Downstream engine references (lazy-bound)
        self._liquidity_tensor: Optional[Any] = None
        self._fed_plumbing: Optional[Any] = None
        self._macro_engine: Optional[Any] = None
        self._security_analysis: Optional[Any] = None
        self._options_engine: Optional[Any] = None
        self._beta_corridor: Optional[Any] = None

        # Initialise empty partitions for every asset class
        for ac in AssetClass:
            self._partitions[ac] = AssetPartition(asset_class=ac)

        logger.info(
            "UniversalDataPool initialised — %d asset classes, lookback=%d days",
            len(AssetClass), lookback_days,
        )

        if fetch_on_init:
            self.pool_all()

    # -----------------------------------------------------------------------
    # Downstream engine binding
    # -----------------------------------------------------------------------

    def bind_engines(
        self,
        liquidity_tensor: Optional[Any] = None,
        fed_plumbing: Optional[Any] = None,
        macro_engine: Optional[Any] = None,
        security_analysis: Optional[Any] = None,
        options_engine: Optional[Any] = None,
        beta_corridor: Optional[Any] = None,
    ):
        """Bind downstream engine instances for routing."""
        self._liquidity_tensor = liquidity_tensor
        self._fed_plumbing = fed_plumbing
        self._macro_engine = macro_engine
        self._security_analysis = security_analysis
        self._options_engine = options_engine
        self._beta_corridor = beta_corridor
        logger.info("Downstream engines bound: %s", ", ".join(
            name for name, eng in [
                ("LiquidityTensor", liquidity_tensor),
                ("FedPlumbingLayer", fed_plumbing),
                ("MacroEngine", macro_engine),
                ("SecurityAnalysisEngine", security_analysis),
                ("OptionsEngine", options_engine),
                ("BetaCorridor", beta_corridor),
            ] if eng is not None
        ))

    # -----------------------------------------------------------------------
    # Primary pooling method
    # -----------------------------------------------------------------------

    def pool_all(self) -> PoolSummary:
        """Aggregate data from ALL asset classes into the unified pool.

        Fetches market data via OpenBB for each asset class, partitions it,
        computes returns, and populates metadata. Returns a summary.
        """
        logger.info("=== UniversalDataPool: pool_all() — aggregating all asset classes ===")
        ts = datetime.now()

        self._pool_equities()
        self._pool_commodities()
        self._pool_indices()
        self._pool_fixed_income()
        self._pool_currencies()
        self._pool_econometrics()
        self._pool_sec_filings()
        self._pool_options()
        self._pool_futures()

        # Compute cross-asset correlation after all data is pooled
        self._compute_cross_asset_correlation_internal()

        # Build sector pools via GIC engine
        self._build_sector_pools()

        elapsed = (datetime.now() - ts).total_seconds()
        summary = self._build_summary()
        logger.info(
            "pool_all() complete in %.2fs — %d tickers, %d records, %d/%d fresh",
            elapsed, summary.total_tickers, summary.total_records,
            summary.fresh_partitions,
            summary.fresh_partitions + summary.stale_partitions,
        )
        return summary

    # -----------------------------------------------------------------------
    # Individual asset class pooling
    # -----------------------------------------------------------------------

    def _pool_equities(self):
        """Pool US S&P 1500 + FTSE 100 representative equities."""
        logger.info("Pooling equities (S&P 1500 + FTSE 100)...")
        all_equity_tickers = list(set(
            SP500_TICKERS + SP400_TICKERS + SP600_TICKERS
            + EXTRA_TICKERS + FTSE_100_REPRESENTATIVE
        ))
        partition = self._partitions[AssetClass.EQUITIES]
        partition.tickers = all_equity_tickers

        try:
            prices = get_adj_close(all_equity_tickers, start=self._start_date)
            if not prices.empty:
                partition.data = prices
                partition.returns = prices.pct_change().dropna()
                partition.record_count = prices.shape[0] * prices.shape[1]
                partition.last_updated = datetime.now()
                partition.is_stale = False
                # Metadata: sector breakdown
                sector_counts: dict[str, int] = {}
                for t in all_equity_tickers:
                    sec = get_sector_for_ticker(t) if callable(get_sector_for_ticker) else ""
                    if sec:
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1
                partition.metadata["sector_counts"] = sector_counts
                partition.metadata["ticker_count"] = len(all_equity_tickers)
                partition.metadata["columns_fetched"] = prices.shape[1]
                logger.info(
                    "Equities pooled: %d tickers, %d with data, %d records",
                    len(all_equity_tickers), prices.shape[1], partition.record_count,
                )
            else:
                logger.warning("Equities: no data returned from OpenBB")
        except Exception as e:
            logger.error("Equities pooling failed: %s", e)

    def _pool_commodities(self):
        """Pool commodity ETFs for cyclical pattern analysis."""
        logger.info("Pooling commodities (9 ETFs)...")
        tickers = list(COMMODITY_ETFS.keys())
        partition = self._partitions[AssetClass.COMMODITIES]
        partition.tickers = tickers
        partition.metadata["etf_names"] = dict(COMMODITY_ETFS)

        try:
            prices = get_adj_close(tickers, start=self._start_date)
            if not prices.empty:
                partition.data = prices
                partition.returns = prices.pct_change().dropna()
                partition.record_count = prices.shape[0] * prices.shape[1]
                partition.last_updated = datetime.now()
                partition.is_stale = False
                logger.info("Commodities pooled: %d ETFs, %d records", prices.shape[1], partition.record_count)
            else:
                logger.warning("Commodities: no data returned")
        except Exception as e:
            logger.error("Commodities pooling failed: %s", e)

    def _pool_indices(self):
        """Pool major index ETFs for benchmarking and rebalancing."""
        logger.info("Pooling indices (7 ETFs)...")
        tickers = list(INDEX_ETFS.keys())
        partition = self._partitions[AssetClass.INDICES]
        partition.tickers = tickers
        partition.metadata["index_names"] = dict(INDEX_ETFS)

        try:
            prices = get_adj_close(tickers, start=self._start_date)
            if not prices.empty:
                partition.data = prices
                partition.returns = prices.pct_change().dropna()
                partition.record_count = prices.shape[0] * prices.shape[1]
                partition.last_updated = datetime.now()
                partition.is_stale = False
                logger.info("Indices pooled: %d ETFs, %d records", prices.shape[1], partition.record_count)
            else:
                logger.warning("Indices: no data returned")
        except Exception as e:
            logger.error("Indices pooling failed: %s", e)

    def _pool_fixed_income(self):
        """Pool fixed income ETFs with maturity bucketing and credit tiering."""
        logger.info("Pooling fixed income (%d ETFs across maturity/credit buckets)...", len(ALL_FI_TICKERS))
        partition = self._partitions[AssetClass.FIXED_INCOME]
        partition.tickers = list(ALL_FI_TICKERS)

        # Fetch all FI ETF prices in one call
        try:
            prices = get_adj_close(ALL_FI_TICKERS, start=self._start_date)
        except Exception as e:
            logger.error("Fixed income price fetch failed: %s", e)
            prices = pd.DataFrame()

        if not prices.empty:
            partition.data = prices
            partition.returns = prices.pct_change().dropna()
            partition.record_count = prices.shape[0] * prices.shape[1]
            partition.last_updated = datetime.now()
            partition.is_stale = False

        # Build sub-partitions by (geography, credit, maturity)
        duration_estimates = {
            MaturityBucket.SHORT: 1.0,
            MaturityBucket.INTERMEDIATE: 3.5,
            MaturityBucket.MEDIUM: 7.0,
            MaturityBucket.LONG: 15.0,
            MaturityBucket.ULTRA_LONG: 25.0,
        }

        self._fi_partitions.clear()
        for (geo, credit, maturity), tickers in FI_ETF_UNIVERSE.items():
            fi_geo = FIGeography(geo)
            fi_credit = CreditTier(credit)
            fi_maturity = MaturityBucket(maturity)
            key = (fi_geo, fi_credit, fi_maturity)

            fi_part = FIPartition(
                geography=fi_geo,
                credit_tier=fi_credit,
                maturity_bucket=fi_maturity,
                tickers=list(tickers),
                duration_estimate=duration_estimates.get(fi_maturity, 5.0),
                last_updated=datetime.now() if not prices.empty else None,
            )

            # Extract sub-partition data
            if not prices.empty:
                available = [t for t in tickers if t in prices.columns]
                if available:
                    fi_part.data = prices[available]

            self._fi_partitions[key] = fi_part

        partition.metadata["bucket_count"] = len(self._fi_partitions)
        partition.metadata["total_fi_tickers"] = len(ALL_FI_TICKERS)
        logger.info(
            "Fixed income pooled: %d tickers, %d maturity/credit buckets",
            len(ALL_FI_TICKERS), len(self._fi_partitions),
        )

    def _pool_currencies(self):
        """Pool G10 + INR + JPY currency pairs."""
        logger.info("Pooling currencies (%d pairs)...", len(FX_PAIRS))
        tickers = list(FX_PAIRS.keys())
        partition = self._partitions[AssetClass.CURRENCIES]
        partition.tickers = tickers
        partition.metadata["pair_names"] = dict(FX_PAIRS)

        try:
            prices = get_adj_close(tickers, start=self._start_date)
            if not prices.empty:
                partition.data = prices
                partition.returns = prices.pct_change().dropna()
                partition.record_count = prices.shape[0] * prices.shape[1]
                partition.last_updated = datetime.now()
                partition.is_stale = False
                logger.info("Currencies pooled: %d pairs, %d records", prices.shape[1], partition.record_count)
            else:
                logger.warning("Currencies: no data returned")
        except Exception as e:
            logger.error("Currencies pooling failed: %s", e)

    def _pool_econometrics(self):
        """Pool FRED econometric data series."""
        logger.info("Pooling econometrics (%d FRED series)...", len(FRED_SERIES))
        partition = self._partitions[AssetClass.ECONOMETRICS]
        partition.tickers = list(FRED_SERIES.keys())
        partition.metadata["series_descriptions"] = dict(FRED_SERIES)

        frames = {}
        for series_id, description in FRED_SERIES.items():
            try:
                df = get_fred_series(series_id)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Use the first numeric column as the series value
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        frames[series_id] = df[numeric_cols[0]]
                elif isinstance(df, pd.Series) and not df.empty:
                    frames[series_id] = df
            except Exception as e:
                logger.debug("FRED series %s (%s) failed: %s", series_id, description, e)

        if frames:
            combined = pd.DataFrame(frames)
            partition.data = combined
            partition.record_count = combined.shape[0] * combined.shape[1]
            partition.last_updated = datetime.now()
            partition.is_stale = False
            partition.metadata["series_fetched"] = list(frames.keys())
            logger.info("Econometrics pooled: %d/%d series", len(frames), len(FRED_SERIES))
        else:
            logger.warning("Econometrics: no FRED data returned")

    def _pool_sec_filings(self):
        """Pool SEC filing metadata for event-driven analysis.

        SEC filings are stored as metadata references (ticker, filing type,
        date) rather than full document content. The actual filing analysis
        is delegated to the EventDrivenEngine and SecurityAnalysisEngine.
        """
        logger.info("Pooling SEC filings metadata (types: %s)...", ", ".join(SEC_FILING_TYPES))
        partition = self._partitions[AssetClass.SEC_FILINGS]
        # Use top 100 by market cap for filing scans
        top_tickers = SP500_TICKERS[:100] if SP500_TICKERS else []
        partition.tickers = top_tickers
        partition.metadata["filing_types"] = list(SEC_FILING_TYPES)
        partition.metadata["scan_universe_size"] = len(top_tickers)

        # SEC filing data is event-based, not time-series.
        # We store a reference frame with columns: ticker, filing_type, last_filing_date
        filing_records = []
        for ticker in top_tickers:
            for ftype in SEC_FILING_TYPES:
                filing_records.append({
                    "ticker": ticker,
                    "filing_type": ftype,
                    "last_filing_date": None,  # populated by EventDrivenEngine
                    "processed": False,
                })

        if filing_records:
            partition.data = pd.DataFrame(filing_records)
            partition.record_count = len(filing_records)
            partition.last_updated = datetime.now()
            partition.is_stale = False

        logger.info(
            "SEC filings pooled: %d tickers x %d filing types = %d entries",
            len(top_tickers), len(SEC_FILING_TYPES), len(filing_records),
        )

    def _pool_options(self):
        """Pool options chain data for selected securities.

        Options data is routed to the OptionsEngine (L7). Fetches are triggered
        on demand for specific tickers rather than pre-pooled, so this method
        initialises the partition structure with the target ticker list.
        """
        logger.info("Pooling options (selected securities)...")
        # Options tracked for top index ETFs + high-vol names
        options_tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
        partition = self._partitions[AssetClass.OPTIONS]
        partition.tickers = options_tickers
        partition.metadata["chain_type"] = "on_demand"
        partition.metadata["target_tickers"] = options_tickers
        partition.last_updated = datetime.now()
        partition.is_stale = False
        partition.record_count = len(options_tickers)
        logger.info("Options partition initialised: %d tickers (on-demand chain fetch)", len(options_tickers))

    def _pool_futures(self):
        """Pool index futures proxies for beta management.

        Uses ETF proxies (SPY, QQQ, DIA, IWM) as futures stand-ins for the
        trade log environment. Real futures data is fetched when available.
        """
        logger.info("Pooling futures (%d contracts)...", len(FUTURES_PROXIES))
        partition = self._partitions[AssetClass.FUTURES]
        # Use ETF proxies for trade log
        proxy_tickers = [proxy for _, proxy in FUTURES_PROXIES.values()]
        partition.tickers = list(FUTURES_PROXIES.keys())
        partition.metadata["futures_contracts"] = {k: v[0] for k, v in FUTURES_PROXIES.items()}
        partition.metadata["etf_proxies"] = {k: v[1] for k, v in FUTURES_PROXIES.items()}

        try:
            prices = get_adj_close(proxy_tickers, start=self._start_date)
            if not prices.empty:
                partition.data = prices
                partition.returns = prices.pct_change().dropna()
                partition.record_count = prices.shape[0] * prices.shape[1]
                partition.last_updated = datetime.now()
                partition.is_stale = False
                logger.info("Futures pooled (via ETF proxies): %d records", partition.record_count)
            else:
                logger.warning("Futures: no proxy data returned")
        except Exception as e:
            logger.error("Futures pooling failed: %s", e)

    # -----------------------------------------------------------------------
    # Sector pool construction (GIC-style)
    # -----------------------------------------------------------------------

    def _build_sector_pools(self):
        """Build sector-level pools using the GIC pooling methodology."""
        if self._gic_engine is None:
            logger.warning("GICPoolingEngine not available — skipping sector pools")
            return

        equity_partition = self._partitions[AssetClass.EQUITIES]
        if equity_partition.data.empty:
            return

        # Group tickers by GICS sector
        sector_groups: dict[str, list[str]] = {}
        for ticker in equity_partition.tickers:
            sector = get_sector_for_ticker(ticker) if callable(get_sector_for_ticker) else ""
            if sector:
                sector_groups.setdefault(sector, []).append(ticker)

        # Create a GIC pool for each sector
        for sector_name, tickers in sector_groups.items():
            pool_name = f"SECTOR_{sector_name.upper().replace(' ', '_')}"
            available_tickers = [t for t in tickers if t in equity_partition.data.columns]
            if available_tickers:
                self._gic_engine.create_pool(pool_name, available_tickers)

        logger.info("Built %d sector GIC pools", len(sector_groups))

    # -----------------------------------------------------------------------
    # Cross-asset correlation
    # -----------------------------------------------------------------------

    def _compute_cross_asset_correlation_internal(self):
        """Compute cross-asset correlation matrix from pooled return data."""
        frames = {}
        label_map = {
            AssetClass.EQUITIES: "EQ",
            AssetClass.COMMODITIES: "CMDTY",
            AssetClass.INDICES: "IDX",
            AssetClass.FIXED_INCOME: "FI",
            AssetClass.CURRENCIES: "FX",
        }

        for ac, prefix in label_map.items():
            partition = self._partitions[ac]
            if not partition.returns.empty:
                # Use equal-weighted portfolio return as representative
                ew_return = partition.returns.mean(axis=1)
                if not ew_return.empty:
                    frames[prefix] = ew_return

        if len(frames) >= 2:
            combined = pd.DataFrame(frames).dropna()
            if len(combined) >= 20:
                self._correlation_matrix = combined.corr()
                self._correlation_timestamp = datetime.now()
                logger.info(
                    "Cross-asset correlation computed: %dx%d matrix",
                    self._correlation_matrix.shape[0],
                    self._correlation_matrix.shape[1],
                )
            else:
                logger.warning("Insufficient overlapping data for correlation matrix")
        else:
            logger.warning("Need at least 2 asset classes with data for correlation")

    # -----------------------------------------------------------------------
    # Public API methods
    # -----------------------------------------------------------------------

    def get_equity_pool(self) -> dict:
        """Return sector-grouped equity data.

        Returns:
            Dict with keys: 'all_data' (DataFrame), 'all_returns' (DataFrame),
            'sector_groups' (dict of sector -> ticker list),
            'sector_pool_returns' (dict of sector -> Series),
            'metadata' (dict).
        """
        partition = self._partitions[AssetClass.EQUITIES]
        partition.check_staleness()

        # Build sector grouping
        sector_groups: dict[str, list[str]] = {}
        for ticker in partition.tickers:
            sector = get_sector_for_ticker(ticker) if callable(get_sector_for_ticker) else ""
            if sector:
                sector_groups.setdefault(sector, []).append(ticker)

        # Compute sector-level equal-weighted returns
        sector_returns: dict[str, pd.Series] = {}
        if not partition.returns.empty:
            for sector_name, tickers in sector_groups.items():
                available = [t for t in tickers if t in partition.returns.columns]
                if available:
                    sector_returns[sector_name] = partition.returns[available].mean(axis=1)

        result = {
            "all_data": partition.data,
            "all_returns": partition.returns,
            "sector_groups": sector_groups,
            "sector_pool_returns": sector_returns,
            "metadata": partition.metadata,
            "is_stale": partition.is_stale,
            "last_updated": partition.last_updated,
        }
        logger.debug("get_equity_pool: %d tickers, %d sectors", len(partition.tickers), len(sector_groups))
        return result

    def get_fi_pool(self) -> dict:
        """Return fixed income data bucketed by maturity, credit, and geography.

        Returns:
            Dict with keys: 'all_data' (DataFrame), 'all_returns' (DataFrame),
            'buckets' (dict of (geo, credit, maturity) -> FIPartition),
            'maturity_breakdown' (dict), 'credit_breakdown' (dict),
            'geography_breakdown' (dict).
        """
        partition = self._partitions[AssetClass.FIXED_INCOME]
        partition.check_staleness()

        # Build breakdowns
        maturity_breakdown: dict[str, list[str]] = {}
        credit_breakdown: dict[str, list[str]] = {}
        geography_breakdown: dict[str, list[str]] = {}

        for (geo, credit, maturity), fi_part in self._fi_partitions.items():
            maturity_breakdown.setdefault(maturity.value, []).extend(fi_part.tickers)
            credit_breakdown.setdefault(credit.value, []).extend(fi_part.tickers)
            geography_breakdown.setdefault(geo.value, []).extend(fi_part.tickers)

        result = {
            "all_data": partition.data,
            "all_returns": partition.returns,
            "buckets": dict(self._fi_partitions),
            "maturity_breakdown": maturity_breakdown,
            "credit_breakdown": credit_breakdown,
            "geography_breakdown": geography_breakdown,
            "is_stale": partition.is_stale,
            "last_updated": partition.last_updated,
        }
        logger.debug("get_fi_pool: %d tickers, %d buckets", len(partition.tickers), len(self._fi_partitions))
        return result

    def get_macro_pool(self) -> dict:
        """Return econometrics + currency data for MacroEngine consumption.

        Returns:
            Dict with keys: 'econometrics' (DataFrame), 'currencies' (DataFrame),
            'fx_returns' (DataFrame), 'fred_series' (list), 'fx_pairs' (list),
            'velocity_inputs' (dict), 'fed_plumbing_inputs' (dict).
        """
        econ_partition = self._partitions[AssetClass.ECONOMETRICS]
        fx_partition = self._partitions[AssetClass.CURRENCIES]
        econ_partition.check_staleness()
        fx_partition.check_staleness()

        # Extract velocity inputs (GDP, M2, FEDFUNDS)
        velocity_inputs = {}
        if not econ_partition.data.empty:
            for key in ["GDP", "M2SL", "FEDFUNDS", "CPIAUCSL"]:
                if key in econ_partition.data.columns:
                    series = econ_partition.data[key].dropna()
                    if not series.empty:
                        velocity_inputs[key] = float(series.iloc[-1])

        # Extract Fed plumbing inputs (WALCL, RRPONTSYD, WTREGEN, SOFR)
        fed_plumbing_inputs = {}
        if not econ_partition.data.empty:
            for key in ["WALCL", "RRPONTSYD", "WTREGEN", "SOFR", "DGS10", "DGS2", "BAMLH0A0HYM2"]:
                if key in econ_partition.data.columns:
                    series = econ_partition.data[key].dropna()
                    if not series.empty:
                        fed_plumbing_inputs[key] = float(series.iloc[-1])

        result = {
            "econometrics": econ_partition.data,
            "currencies": fx_partition.data,
            "fx_returns": fx_partition.returns,
            "fred_series": list(FRED_SERIES.keys()),
            "fx_pairs": list(FX_PAIRS.keys()),
            "velocity_inputs": velocity_inputs,
            "fed_plumbing_inputs": fed_plumbing_inputs,
            "econ_is_stale": econ_partition.is_stale,
            "fx_is_stale": fx_partition.is_stale,
        }
        logger.debug(
            "get_macro_pool: %d econ series, %d FX pairs, %d velocity inputs",
            len(econ_partition.data.columns) if not econ_partition.data.empty else 0,
            len(fx_partition.tickers),
            len(velocity_inputs),
        )
        return result

    def get_commodity_signals(self) -> dict:
        """Return commodity cyclical pattern signals.

        Computes rolling momentum, mean reversion z-scores, and inter-commodity
        correlations for cyclical pattern detection.

        Returns:
            Dict with keys: 'prices' (DataFrame), 'returns' (DataFrame),
            'momentum_21d' (dict), 'momentum_63d' (dict),
            'zscore_21d' (dict), 'correlations' (DataFrame),
            'cyclical_regime' (str).
        """
        partition = self._partitions[AssetClass.COMMODITIES]
        partition.check_staleness()

        momentum_21d = {}
        momentum_63d = {}
        zscore_21d = {}
        correlations = pd.DataFrame()
        cyclical_regime = "NEUTRAL"

        if not partition.returns.empty and len(partition.returns) >= 63:
            for col in partition.returns.columns:
                r = partition.returns[col].dropna()
                if len(r) >= 63:
                    momentum_21d[col] = float(r.iloc[-21:].sum())
                    momentum_63d[col] = float(r.iloc[-63:].sum())
                    # Z-score: (current 21d return - mean) / std
                    rolling_21 = r.rolling(21).sum()
                    if len(rolling_21.dropna()) >= 10:
                        mean_val = float(rolling_21.mean())
                        std_val = float(rolling_21.std())
                        if std_val > 1e-8:
                            zscore_21d[col] = float((rolling_21.iloc[-1] - mean_val) / std_val)

            # Inter-commodity correlation matrix
            if partition.returns.shape[1] >= 2:
                correlations = partition.returns.iloc[-63:].corr()

            # Determine cyclical regime from aggregate commodity momentum
            avg_mom = np.mean(list(momentum_63d.values())) if momentum_63d else 0.0
            if avg_mom > 0.05:
                cyclical_regime = "EXPANSION"
            elif avg_mom < -0.05:
                cyclical_regime = "CONTRACTION"
            else:
                cyclical_regime = "NEUTRAL"

        result = {
            "prices": partition.data,
            "returns": partition.returns,
            "momentum_21d": momentum_21d,
            "momentum_63d": momentum_63d,
            "zscore_21d": zscore_21d,
            "correlations": correlations,
            "cyclical_regime": cyclical_regime,
            "etf_names": COMMODITY_ETFS,
            "is_stale": partition.is_stale,
        }
        logger.debug("get_commodity_signals: regime=%s, %d ETFs", cyclical_regime, len(partition.tickers))
        return result

    def get_cross_asset_correlation(self) -> dict:
        """Return the full cross-asset correlation matrix.

        Returns:
            Dict with keys: 'matrix' (DataFrame), 'timestamp' (datetime),
            'asset_classes' (list), 'regime_signal' (str).
        """
        if self._correlation_matrix is None:
            self._compute_cross_asset_correlation_internal()

        # Derive regime signal from equity-FI correlation
        regime_signal = "NORMAL"
        if self._correlation_matrix is not None and "EQ" in self._correlation_matrix.columns:
            if "FI" in self._correlation_matrix.columns:
                eq_fi_corr = self._correlation_matrix.loc["EQ", "FI"]
                if eq_fi_corr > 0.3:
                    regime_signal = "RISK_ON"  # stocks and bonds moving together (unusual)
                elif eq_fi_corr < -0.3:
                    regime_signal = "FLIGHT_TO_QUALITY"  # normal inverse relationship strong
                else:
                    regime_signal = "NORMAL"

        result = {
            "matrix": self._correlation_matrix if self._correlation_matrix is not None else pd.DataFrame(),
            "timestamp": self._correlation_timestamp,
            "asset_classes": list(self._correlation_matrix.columns) if self._correlation_matrix is not None else [],
            "regime_signal": regime_signal,
        }
        return result

    def get_benchmark_returns(self) -> dict:
        """Return index returns for benchmarking.

        Returns:
            Dict with keys: 'daily_returns' (DataFrame), 'cumulative' (DataFrame),
            'ytd' (dict), 'mtd' (dict), '1m' (dict), '3m' (dict),
            'annualized_vol' (dict).
        """
        partition = self._partitions[AssetClass.INDICES]
        partition.check_staleness()

        daily_returns = partition.returns
        cumulative = pd.DataFrame()
        ytd = {}
        mtd = {}
        ret_1m = {}
        ret_3m = {}
        annualized_vol = {}

        if not daily_returns.empty and len(daily_returns) >= 21:
            cumulative = (1 + daily_returns).cumprod() - 1

            now = datetime.now()
            for col in daily_returns.columns:
                r = daily_returns[col].dropna()
                if r.empty:
                    continue

                # 1-month return
                if len(r) >= 21:
                    ret_1m[col] = float((1 + r.iloc[-21:]).prod() - 1)

                # 3-month return
                if len(r) >= 63:
                    ret_3m[col] = float((1 + r.iloc[-63:]).prod() - 1)

                # YTD return
                try:
                    year_start = datetime(now.year, 1, 1)
                    ytd_mask = r.index >= pd.Timestamp(year_start)
                    ytd_r = r[ytd_mask]
                    if not ytd_r.empty:
                        ytd[col] = float((1 + ytd_r).prod() - 1)
                except Exception:
                    pass

                # MTD return
                try:
                    month_start = datetime(now.year, now.month, 1)
                    mtd_mask = r.index >= pd.Timestamp(month_start)
                    mtd_r = r[mtd_mask]
                    if not mtd_r.empty:
                        mtd[col] = float((1 + mtd_r).prod() - 1)
                except Exception:
                    pass

                # Annualized volatility
                if len(r) >= 21:
                    annualized_vol[col] = float(r.iloc[-252:].std() * np.sqrt(252))

        result = {
            "daily_returns": daily_returns,
            "cumulative": cumulative,
            "ytd": ytd,
            "mtd": mtd,
            "1m": ret_1m,
            "3m": ret_3m,
            "annualized_vol": annualized_vol,
            "index_names": INDEX_ETFS,
            "is_stale": partition.is_stale,
        }
        logger.debug("get_benchmark_returns: %d indices", len(daily_returns.columns) if not daily_returns.empty else 0)
        return result

    def get_liquidity_inputs(self) -> dict:
        """Return data for MetadronCube's LiquidityTensor and FedPlumbingLayer.

        Extracts the FRED series needed by FedPlumbingLayer (SOFR, WALCL, TGA,
        ON-RRP) and LiquidityTensor (M2, credit spreads, reserves) from the
        econometrics and fixed income pools.

        Returns:
            Dict with keys: 'fed_plumbing' (dict), 'liquidity_tensor' (dict),
            'hy_spread' (float), 'ig_spread' (float), 'yield_curve' (dict),
            'reserve_data' (dict), 'fx_stress' (dict).
        """
        econ_data = self._partitions[AssetClass.ECONOMETRICS].data
        fi_data = self._partitions[AssetClass.FIXED_INCOME].data
        fx_data = self._partitions[AssetClass.CURRENCIES].data

        # Fed plumbing inputs
        fed_plumbing = {}
        plumbing_keys = ["SOFR", "WALCL", "WTREGEN", "RRPONTSYD", "FEDFUNDS"]
        if not econ_data.empty:
            for key in plumbing_keys:
                if key in econ_data.columns:
                    series = econ_data[key].dropna()
                    if not series.empty:
                        fed_plumbing[key] = float(series.iloc[-1])
                        # Include delta (change) for flow signals
                        if len(series) >= 2:
                            fed_plumbing[f"{key}_delta"] = float(series.iloc[-1] - series.iloc[-2])

        # Liquidity tensor inputs
        liquidity_tensor = {}
        lt_keys = ["M2SL", "BAMLH0A0HYM2", "BAMLC0A4CBBB", "DGS10", "DGS2", "VIXCLS"]
        if not econ_data.empty:
            for key in lt_keys:
                if key in econ_data.columns:
                    series = econ_data[key].dropna()
                    if not series.empty:
                        liquidity_tensor[key] = float(series.iloc[-1])

        # HY and IG spreads
        hy_spread = liquidity_tensor.get("BAMLH0A0HYM2", 0.0)
        ig_spread = liquidity_tensor.get("BAMLC0A4CBBB", 0.0)

        # Yield curve
        yield_curve = {}
        if "DGS10" in liquidity_tensor and "DGS2" in liquidity_tensor:
            yield_curve["10y"] = liquidity_tensor["DGS10"]
            yield_curve["2y"] = liquidity_tensor["DGS2"]
            yield_curve["spread_10y2y"] = liquidity_tensor["DGS10"] - liquidity_tensor["DGS2"]
            yield_curve["inverted"] = yield_curve["spread_10y2y"] < 0

        # Reserve data
        reserve_data = {}
        if "WALCL" in fed_plumbing:
            reserve_data["total_assets"] = fed_plumbing["WALCL"]
        if "RRPONTSYD" in fed_plumbing:
            reserve_data["on_rrp"] = fed_plumbing["RRPONTSYD"]
        if "WTREGEN" in fed_plumbing:
            reserve_data["tga"] = fed_plumbing["WTREGEN"]
        # Net liquidity proxy = Fed assets - TGA - ON-RRP
        if all(k in reserve_data for k in ["total_assets", "on_rrp", "tga"]):
            reserve_data["net_liquidity"] = (
                reserve_data["total_assets"]
                - reserve_data["tga"]
                - reserve_data["on_rrp"]
            )

        # FX stress indicators
        fx_stress = {}
        if not fx_data.empty:
            fx_returns = self._partitions[AssetClass.CURRENCIES].returns
            if not fx_returns.empty and len(fx_returns) >= 21:
                # DXY proxy: average USD strength across pairs
                usd_long_cols = [c for c in fx_returns.columns if c.startswith("USD")]
                usd_short_cols = [c for c in fx_returns.columns if not c.startswith("USD")]
                if usd_long_cols:
                    fx_stress["usd_strength_21d"] = float(fx_returns[usd_long_cols].mean(axis=1).iloc[-21:].sum())
                if usd_short_cols:
                    fx_stress["usd_weakness_21d"] = float(fx_returns[usd_short_cols].mean(axis=1).iloc[-21:].sum())
                # FX volatility
                fx_stress["fx_vol_21d"] = float(fx_returns.std(axis=1).iloc[-21:].mean() * np.sqrt(252))

        result = {
            "fed_plumbing": fed_plumbing,
            "liquidity_tensor": liquidity_tensor,
            "hy_spread": hy_spread,
            "ig_spread": ig_spread,
            "yield_curve": yield_curve,
            "reserve_data": reserve_data,
            "fx_stress": fx_stress,
        }
        logger.debug(
            "get_liquidity_inputs: %d plumbing keys, %d tensor keys, net_liq=%.0f",
            len(fed_plumbing), len(liquidity_tensor),
            reserve_data.get("net_liquidity", 0),
        )
        return result

    def route_to_layer(self, layer_id: LayerID) -> LayerPayload:
        """Route pooled data to a specific architecture layer.

        Collects all asset class data that maps to the requested layer and
        packages it into a LayerPayload.

        Args:
            layer_id: Target layer (L1 through L7).

        Returns:
            LayerPayload with aggregated data for the layer.
        """
        if isinstance(layer_id, str):
            try:
                layer_id = LayerID(layer_id)
            except ValueError:
                logger.error("Invalid layer_id: %s", layer_id)
                return LayerPayload(layer_id=LayerID.L1_DATA)

        payload = LayerPayload(
            layer_id=layer_id,
            timestamp=datetime.now().isoformat(),
        )

        total_records = 0
        for ac, layers in ASSET_LAYER_ROUTING.items():
            if layer_id in layers:
                partition = self._partitions[ac]
                payload.asset_classes.append(ac.value)
                payload.data[ac.value] = {
                    "data": partition.data,
                    "returns": partition.returns,
                    "tickers": partition.tickers,
                    "metadata": partition.metadata,
                    "is_stale": partition.is_stale,
                    "last_updated": partition.last_updated.isoformat() if partition.last_updated else None,
                }
                total_records += partition.record_count

        payload.record_count = total_records

        # Layer-specific enrichment
        if layer_id == LayerID.L2_SIGNALS:
            # Add macro pool and liquidity inputs for MetadronCube/MacroEngine
            payload.data["macro_pool"] = self.get_macro_pool()
            payload.data["liquidity_inputs"] = self.get_liquidity_inputs()
            payload.data["commodity_signals"] = self.get_commodity_signals()
            # Feed bound MacroEngine if available
            if self._macro_engine is not None:
                try:
                    payload.data["macro_engine_snapshot"] = self._macro_engine.get_snapshot()
                except Exception as e:
                    logger.debug("MacroEngine snapshot failed: %s", e)

        elif layer_id == LayerID.L3_INTELLIGENCE:
            # Add equity fundamentals and SEC filing data
            payload.data["equity_pool"] = self.get_equity_pool()
            payload.data["cross_asset_correlation"] = self.get_cross_asset_correlation()
            # Feed bound SecurityAnalysisEngine if available
            if self._security_analysis is not None:
                try:
                    payload.data["security_analysis_available"] = True
                except Exception as e:
                    logger.debug("SecurityAnalysisEngine check failed: %s", e)

        elif layer_id == LayerID.L4_PORTFOLIO:
            # Add benchmark returns and futures for BetaCorridor
            payload.data["benchmark_returns"] = self.get_benchmark_returns()
            payload.data["fi_pool"] = self.get_fi_pool()
            # Feed bound BetaCorridor if available
            if self._beta_corridor is not None:
                try:
                    futures_data = self._partitions[AssetClass.FUTURES]
                    payload.data["beta_corridor_futures"] = {
                        "prices": futures_data.data,
                        "returns": futures_data.returns,
                    }
                except Exception as e:
                    logger.debug("BetaCorridor data failed: %s", e)

        elif layer_id == LayerID.L7_HFT_EXECUTION:
            # Add options data for OptionsEngine
            payload.data["options_tickers"] = self._partitions[AssetClass.OPTIONS].tickers
            # Feed bound OptionsEngine if available
            if self._options_engine is not None:
                try:
                    payload.data["options_engine_available"] = True
                except Exception as e:
                    logger.debug("OptionsEngine check failed: %s", e)

        logger.info(
            "route_to_layer(%s): %d asset classes, %d records",
            layer_id.value, len(payload.asset_classes), payload.record_count,
        )
        return payload

    def get_data_freshness(self) -> FreshnessReport:
        """Generate a staleness report across all partitions.

        Returns:
            FreshnessReport with per-partition freshness details.
        """
        report = FreshnessReport(
            timestamp=datetime.now().isoformat(),
            total_partitions=len(self._partitions),
        )

        oldest_name = ""
        oldest_age = 0.0

        for ac, partition in self._partitions.items():
            partition.check_staleness()
            age = partition.age_seconds()
            threshold = STALENESS_THRESHOLDS.get(ac, 300)

            detail = {
                "last_updated": partition.last_updated.isoformat() if partition.last_updated else "NEVER",
                "age_seconds": age if age != float("inf") else -1,
                "threshold_seconds": threshold,
                "is_stale": partition.is_stale,
                "record_count": partition.record_count,
                "ticker_count": len(partition.tickers),
            }
            report.partition_details[ac.value] = detail

            if partition.is_stale:
                report.stale_count += 1
            else:
                report.fresh_count += 1

            if age > oldest_age and age != float("inf"):
                oldest_age = age
                oldest_name = ac.value
            elif age == float("inf") and not oldest_name:
                oldest_name = ac.value
                oldest_age = -1

        report.oldest_partition = oldest_name
        report.oldest_age_seconds = oldest_age

        logger.info(
            "Freshness report: %d/%d fresh, oldest=%s (%.0fs)",
            report.fresh_count, report.total_partitions,
            report.oldest_partition, report.oldest_age_seconds,
        )
        return report

    def compute_sector_allocation_signals(self) -> list[SectorAllocation]:
        """Compute sector allocation signals from pooled equity data.

        Combines sector momentum (1m, 3m, 12m), relative strength vs SPY,
        and breadth to produce overweight/underweight signals for each
        GICS sector.

        Returns:
            List of SectorAllocation objects sorted by composite score.
        """
        equity_pool = self.get_equity_pool()
        sector_groups = equity_pool.get("sector_groups", {})
        sector_returns = equity_pool.get("sector_pool_returns", {})
        all_returns = equity_pool.get("all_returns", pd.DataFrame())

        if not sector_returns or all_returns.empty:
            logger.warning("Insufficient data for sector allocation signals")
            return []

        # SPY benchmark
        benchmark = pd.Series(dtype=float)
        idx_partition = self._partitions[AssetClass.INDICES]
        if not idx_partition.returns.empty and "SPY" in idx_partition.returns.columns:
            benchmark = idx_partition.returns["SPY"]

        allocations = []
        for sector_name, sector_r in sector_returns.items():
            if sector_r.empty or len(sector_r) < 21:
                continue

            alloc = SectorAllocation(
                sector=sector_name,
                ticker_count=len(sector_groups.get(sector_name, [])),
            )

            # Momentum calculations
            if len(sector_r) >= 21:
                alloc.momentum_1m = float(sector_r.iloc[-21:].sum())
            if len(sector_r) >= 63:
                alloc.momentum_3m = float(sector_r.iloc[-63:].sum())
            if len(sector_r) >= 252:
                alloc.momentum_12m = float(sector_r.iloc[-252:].sum())

            # Relative strength vs SPY
            if not benchmark.empty and len(benchmark) >= 63 and len(sector_r) >= 63:
                sector_cum = float((1 + sector_r.iloc[-63:]).prod() - 1)
                bench_cum = float((1 + benchmark.iloc[-63:]).prod() - 1)
                alloc.relative_strength = sector_cum - bench_cum

            # Composite score: 50% 3m momentum + 30% relative strength + 20% 1m momentum
            alloc.weight = (
                0.50 * alloc.momentum_3m
                + 0.30 * alloc.relative_strength
                + 0.20 * alloc.momentum_1m
            )

            allocations.append(alloc)

        # Sort by composite weight descending
        allocations.sort(key=lambda a: a.weight, reverse=True)

        # Mark top 3 as overweight
        for i, alloc in enumerate(allocations):
            alloc.overweight = i < 3

        logger.info(
            "Sector allocation signals: %d sectors, top 3: %s",
            len(allocations),
            ", ".join(a.sector for a in allocations[:3]),
        )
        return allocations

    # -----------------------------------------------------------------------
    # Partition access helpers
    # -----------------------------------------------------------------------

    def get_partition(self, asset_class: AssetClass) -> AssetPartition:
        """Direct access to a specific asset class partition."""
        return self._partitions.get(asset_class, AssetPartition(asset_class=asset_class))

    def get_fi_partition(
        self,
        geography: FIGeography,
        credit_tier: CreditTier,
        maturity_bucket: MaturityBucket,
    ) -> Optional[FIPartition]:
        """Direct access to a specific fixed income sub-partition."""
        return self._fi_partitions.get((geography, credit_tier, maturity_bucket))

    def get_all_tickers(self) -> list[str]:
        """Return all tickers across all asset classes (deduplicated)."""
        all_tickers = set()
        for partition in self._partitions.values():
            all_tickers.update(partition.tickers)
        return sorted(all_tickers)

    def get_pool_summary(self) -> PoolSummary:
        """Return a summary of the entire pool state."""
        return self._build_summary()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_summary(self) -> PoolSummary:
        """Build a PoolSummary from current partition state."""
        summary = PoolSummary(
            timestamp=datetime.now().isoformat(),
        )

        total_tickers = set()
        for ac, partition in self._partitions.items():
            partition.check_staleness()
            count = len(partition.tickers)
            summary.asset_class_counts[ac.value] = count
            total_tickers.update(partition.tickers)
            summary.total_records += partition.record_count
            if partition.is_stale:
                summary.stale_partitions += 1
            else:
                summary.fresh_partitions += 1

        summary.total_tickers = len(total_tickers)

        # Sector breakdown from equity partition
        eq_meta = self._partitions[AssetClass.EQUITIES].metadata
        summary.sector_breakdown = eq_meta.get("sector_counts", {})

        # FI maturity breakdown
        for (geo, credit, maturity), fi_part in self._fi_partitions.items():
            key = f"{geo.value}_{credit.value}_{maturity.value}"
            summary.fi_maturity_breakdown[key] = len(fi_part.tickers)

        return summary

    # -----------------------------------------------------------------------
    # String representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        fresh = sum(1 for p in self._partitions.values() if not p.check_staleness())
        total = len(self._partitions)
        tickers = len(self.get_all_tickers())
        return (
            f"UniversalDataPool(tickers={tickers}, "
            f"partitions={total}, fresh={fresh}/{total}, "
            f"fi_buckets={len(self._fi_partitions)})"
        )
