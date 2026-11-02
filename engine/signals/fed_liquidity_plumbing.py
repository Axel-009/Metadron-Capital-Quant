"""Fed Liquidity Plumbing Engine & Cube Liquidity Tensor.

Extends MetadronCube Layers 0-2 with deeper plumbing:
    - Full Fed balance sheet decomposition (WALCL, SOMA, MBS, Treasuries)
    - Reserve distribution vector: Fed → PD → GSIB → Shadow Banks → Markets
    - Sector flow matrix: money velocity per GICS sector (follows the money)
    - Credit impulse: rate of change of new credit
    - Collateral velocity: rehypothecation chain length
    - Dealer balance sheet capacity: PD leverage room
    - Money velocity tracker: V = GDP/M2, sector-level absorption

Data: OpenBB/FRED (sole data source). All FRED series codes specified.
try/except on ALL external imports — system runs degraded, never broken.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import deque
from enum import Enum

logger = logging.getLogger("metadron.signals.fed_liquidity")

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------
try:
    from ..data.openbb_data import get_fred_series, get_fed_balance_sheet
    _data_available = True
except ImportError:
    _data_available = False
    logger.warning("openbb_data import failed — Fed plumbing runs in degraded mode")

    def get_fred_series(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fed_balance_sheet(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


try:
    from ..data.universe_engine import GICS_SECTORS
except ImportError:
    GICS_SECTORS = {
        10: "Energy", 15: "Materials", 20: "Industrials",
        25: "Consumer Discretionary", 30: "Consumer Staples",
        35: "Health Care", 40: "Financials",
        45: "Information Technology", 50: "Communication Services",
        55: "Utilities", 60: "Real Estate",
    }

# ---------------------------------------------------------------------------
# FRB (Federal Reserve Bank) Python client — intelligence_platform/FRB
# Provides direct FRED API access as alternative/supplement to OpenBB FRED.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    import sys as _sys
    _frb_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
                     / "intelligence_platform" / "FRB")
    if _frb_root not in _sys.path:
        _sys.path.insert(0, _frb_root)
    from fred import Fred as FRBFredClient
    FRB_AVAILABLE = True
except (ImportError, FileNotFoundError, Exception):
    FRBFredClient = None
    FRB_AVAILABLE = False
    logger.info("FRB FRED client unavailable — using OpenBB FRED data only")

# ---------------------------------------------------------------------------
# FRED Series Catalog — all plumbing inputs
# ---------------------------------------------------------------------------
FRED_PLUMBING_SERIES = {
    # Fed Balance Sheet
    "WALCL": "Fed Total Assets (Weekly)",
    "WSHOSHO": "SOMA Holdings (Treasuries)",
    "WSHOMCB": "SOMA Holdings (MBS)",
    # Reserves & Repo
    "RRPONTSYD": "ON-RRP Balance (Daily)",
    "WTREGEN": "Treasury General Account",
    "TOTRESNS": "Total Reserves (Monthly)",
    "EXCSRESNS": "Excess Reserves (Monthly)",
    # Rates
    "SOFR": "Secured Overnight Financing Rate",
    "DFF": "Fed Funds Effective Rate",
    "FEDFUNDS": "Federal Funds Target Rate",
    # Money Supply & Velocity
    "M2SL": "M2 Money Supply",
    "M2V": "Velocity of M2 Money Stock",
    "GDP": "Gross Domestic Product (Nominal)",
    # Credit
    "BAMLH0A0HYM2": "ICE BofA US HY OAS",
    "BAMLC0A0CM": "ICE BofA US Corporate OAS",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10YIE": "10Y Breakeven Inflation",
    "DPCREDIT": "Domestic Private Credit (Proxy)",
    "BUSLOANS": "Commercial & Industrial Loans",
    "REALLN": "Real Estate Loans",
}

# Sector ETF proxies for flow tracking (GICS code → ETF)
SECTOR_ETFS = {
    10: "XLE", 15: "XLB", 20: "XLI", 25: "XLY", 30: "XLP",
    35: "XLV", 40: "XLF", 45: "XLK", 50: "XLC", 55: "XLU", 60: "XLRE",
}

# Sector sensitivity to liquidity (empirical weights)
# Higher = more sensitive to reserve/liquidity changes
SECTOR_LIQUIDITY_BETA = {
    10: 0.60,   # Energy — commodity cycle, moderate liquidity sensitivity
    15: 0.55,   # Materials — similar to energy
    20: 0.65,   # Industrials — capex-driven, credit-sensitive
    25: 0.80,   # Consumer Disc — highly credit/liquidity sensitive
    30: 0.25,   # Consumer Staples — defensive, low sensitivity
    35: 0.40,   # Health Care — moderate
    40: 0.90,   # Financials — direct exposure to reserves/rates
    45: 0.75,   # Info Tech — growth, discount rate sensitive
    50: 0.50,   # Comm Services — mixed
    55: 0.35,   # Utilities — rate sensitive but defensive
    60: 0.85,   # Real Estate — highly rate/credit sensitive
}

# Sector credit sensitivity (how much credit impulse affects each sector)
SECTOR_CREDIT_BETA = {
    10: 0.45, 15: 0.50, 20: 0.70, 25: 0.85, 30: 0.20,
    35: 0.35, 40: 0.95, 45: 0.55, 50: 0.40, 55: 0.60, 60: 0.90,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VelocityRegime(str, Enum):
    """Money velocity regime."""
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    STABLE = "STABLE"


class LiquidityRegime(str, Enum):
    """Aggregate liquidity regime."""
    FLOOD = "FLOOD"         # Ample liquidity, risk-on
    NORMAL = "NORMAL"       # Balanced
    TIGHT = "TIGHT"         # Draining, caution
    DRAIN = "DRAIN"         # Active drain, risk-off


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FedBalanceSheet:
    """Snapshot of Fed balance sheet components (all in $B)."""
    walcl: float = 7500.0           # Total assets
    soma_treasuries: float = 4800.0 # SOMA Treasury holdings
    soma_mbs: float = 2400.0        # SOMA MBS holdings
    reserves: float = 3200.0        # Total bank reserves
    excess_reserves: float = 3000.0 # Excess reserves above requirements
    on_rrp: float = 500.0           # ON-RRP balance
    tga: float = 700.0              # Treasury General Account
    timestamp: str = ""


@dataclass
class ReserveDistribution:
    """How reserves flow through the financial system.

    Fed → Primary Dealers → GSIBs → Shadow Banks → Markets.
    Each field is a fraction of total reserves at that node.
    """
    fed_to_pd: float = 0.0          # Flow from Fed to primary dealers
    pd_to_gsib: float = 0.0         # Flow from PDs to GSIBs
    gsib_to_shadow: float = 0.0     # Flow from GSIBs to shadow banking
    shadow_to_market: float = 0.0   # Flow from shadow banks to markets
    net_market_liquidity: float = 0.0  # Net arriving at equity/credit markets
    bottleneck: str = ""             # Where the flow is constrained


@dataclass
class SectorFlowAllocation:
    """Sector allocation signal derived from money flow analysis."""
    sector_scores: Dict[int, float] = field(default_factory=dict)     # GICS code → score [-1,+1]
    sector_weights: Dict[int, float] = field(default_factory=dict)    # GICS code → weight [0,1]
    overweight: List[str] = field(default_factory=list)               # Overweight sector names
    underweight: List[str] = field(default_factory=list)              # Underweight sector names
    flow_regime: str = "NEUTRAL"
    timestamp: str = ""


@dataclass
class CreditImpulseState:
    """Credit impulse: rate of change of new credit."""
    impulse: float = 0.0             # Overall credit impulse [-1,+1]
    business_loans_delta: float = 0.0
    real_estate_loans_delta: float = 0.0
    consumer_credit_delta: float = 0.0
    impulse_trend: str = "STABLE"    # EXPANDING / CONTRACTING / STABLE


@dataclass
class CollateralChain:
    """Collateral velocity — rehypothecation chain depth."""
    velocity: float = 2.5           # Average rehypothecation length
    dealer_capacity: float = 0.0    # PD leverage room [-1,+1]
    collateral_scarcity: float = 0.0  # Scarcity signal [-1,+1]
    srf_usage: float = 0.0         # Standing Repo Facility usage signal


@dataclass
class DrainWarning:
    """Early warning of liquidity drain."""
    warning_level: int = 0          # 0=none, 1=watch, 2=caution, 3=warning, 4=critical
    warning_label: str = "NONE"
    triggers: List[str] = field(default_factory=list)
    tga_draining: bool = False
    rrp_draining: bool = False
    reserves_falling: bool = False
    spread_widening: bool = False
    recommended_beta_adj: float = 0.0  # Suggested beta adjustment


@dataclass
class LiquidityTensorOutput:
    """Full Cube Liquidity Tensor output."""
    liquidity_score: float = 0.0     # Aggregate L(t) in [-1,+1]
    regime: LiquidityRegime = LiquidityRegime.NORMAL
    balance_sheet: FedBalanceSheet = field(default_factory=FedBalanceSheet)
    reserve_distribution: ReserveDistribution = field(default_factory=ReserveDistribution)
    sector_allocation: SectorFlowAllocation = field(default_factory=SectorFlowAllocation)
    credit_impulse: CreditImpulseState = field(default_factory=CreditImpulseState)
    collateral: CollateralChain = field(default_factory=CollateralChain)
    drain_warning: DrainWarning = field(default_factory=DrainWarning)
    velocity: float = 0.0
    velocity_regime: VelocityRegime = VelocityRegime.STABLE
    timestamp: str = ""


# ---------------------------------------------------------------------------
# FedLiquidityPlumbing
# ---------------------------------------------------------------------------
class FedLiquidityPlumbing:
    """Tracks all Fed plumbing data — balance sheet, reserves, repo, rates.

    Extends MetadronCube Layer 0 (FedPlumbingLayer) with full decomposition:
        1. Fed Balance Sheet: WALCL, SOMA, MBS, Treasuries
        2. Reserves: bank reserves at Fed, excess reserves
        3. ON-RRP: money market fund parking
        4. TGA: Treasury cash balance
        5. SRF: Standing Repo Facility
        6. SOFR/Fed Funds spread
        7. Primary Dealer positions (estimated from GSIB reserve share)
        8. GSIB reserve requirements
        9. Discount window usage (proxied from spread stress)
       10. Term funding programs
    """

    # Series pulled on each update
    _CORE_SERIES = [
        "WALCL", "WSHOSHO", "RRPONTSYD", "WTREGEN",
        "SOFR", "DFF", "FEDFUNDS",
        "TOTRESNS", "EXCSRESNS",
        "BAMLH0A0HYM2", "BAMLC0A0CM",
        "T10Y2Y", "T10YIE",
    ]

    _CREDIT_SERIES = ["BUSLOANS", "REALLN"]
    _MONEY_SERIES = ["M2SL", "M2V", "GDP"]

    # Weights for aggregate liquidity score
    _SCORE_WEIGHTS = {
        "balance_sheet": 0.20,
        "reserves": 0.20,
        "rrp": 0.15,
        "tga": 0.10,
        "sofr_spread": 0.15,
        "credit_spread": 0.10,
        "yield_curve": 0.10,
    }

    # Historical norms for z-score calculation ($B)
    _NORMS = {
        "walcl_mean": 7800.0, "walcl_std": 800.0,
        "reserves_mean": 3200.0, "reserves_std": 500.0,
        "rrp_mean": 800.0, "rrp_std": 600.0,
        "tga_mean": 600.0, "tga_std": 250.0,
        "sofr_dff_spread_mean": 0.01, "sofr_dff_spread_std": 0.05,
        "hy_oas_mean": 4.0, "hy_oas_std": 1.5,
    }

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._balance_sheet = FedBalanceSheet()
        self._history: Dict[str, deque] = {
            k: deque(maxlen=lookback_days) for k in [
                "walcl", "reserves", "rrp", "tga", "sofr", "dff",
                "hy_oas", "ig_oas", "t10y2y", "breakeven",
                "busloans", "realln", "m2", "gdp", "m2v",
            ]
        }
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._last_update: Optional[datetime] = None
        self._liquidity_score: float = 0.0

    # ----- public API -----

    def update(self) -> FedBalanceSheet:
        """Pull latest FRED data and update all internal state.

        Returns the latest FedBalanceSheet snapshot.
        """
        start = (datetime.now() - timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")

        # Fetch core plumbing series
        core_df = get_fred_series(self._CORE_SERIES, start)
        credit_df = get_fred_series(self._CREDIT_SERIES, start)
        money_df = get_fred_series(self._MONEY_SERIES, start)

        self._raw_data = {"core": core_df, "credit": credit_df, "money": money_df}

        # Update balance sheet snapshot from latest values
        self._balance_sheet = self._build_balance_sheet(core_df)
        self._balance_sheet.timestamp = datetime.now().isoformat()

        # Append to history
        self._append_history(core_df, credit_df, money_df)

        # Recompute aggregate score
        self._liquidity_score = self._compute_aggregate_score()

        self._last_update = datetime.now()
        logger.info(
            f"Fed plumbing updated: WALCL={self._balance_sheet.walcl:.0f}B, "
            f"reserves={self._balance_sheet.reserves:.0f}B, "
            f"RRP={self._balance_sheet.on_rrp:.0f}B, "
            f"TGA={self._balance_sheet.tga:.0f}B, "
            f"L(t)={self._liquidity_score:+.3f}"
        )
        return self._balance_sheet

    def get_reserve_distribution(self) -> ReserveDistribution:
        """Compute where reserves are flowing: Fed → PD → GSIB → Shadow → Markets.

        Uses reserve levels, RRP changes, and spread dynamics to estimate
        the flow cascade through the financial system.
        """
        rd = ReserveDistribution()

        reserves = self._balance_sheet.reserves
        excess = self._balance_sheet.excess_reserves
        rrp = self._balance_sheet.on_rrp
        tga = self._balance_sheet.tga

        # Total distributable reserves = excess reserves - RRP parking - TGA drain
        distributable = max(excess - rrp * 0.001 - tga * 0.001, 0)
        total = max(reserves, 1.0)

        # Fed → PD: primary dealers get first access via repo/reverse repo
        # High RRP = money parked at Fed, not flowing to PDs
        rrp_drag = np.clip(rrp / 2000.0, 0, 1)
        rd.fed_to_pd = float(np.clip(1.0 - rrp_drag * 0.6, 0, 1))

        # PD → GSIB: dealers pass reserves to GSIBs via interbank
        # TGA buildup = Treasury draining from system
        tga_drag = np.clip((tga - 400) / 600.0, 0, 1)
        rd.pd_to_gsib = float(np.clip(rd.fed_to_pd * (1.0 - tga_drag * 0.4), 0, 1))

        # GSIB → Shadow: GSIBs lend to shadow banking (hedge funds, MMFs)
        # Tighter spreads = more willing to lend
        hy_oas_hist = list(self._history["hy_oas"])
        spread_tightness = 0.5
        if hy_oas_hist:
            current_spread = hy_oas_hist[-1]
            spread_tightness = float(np.clip(1.0 - (current_spread - 3.0) / 4.0, 0, 1))
        rd.gsib_to_shadow = float(np.clip(rd.pd_to_gsib * spread_tightness * 0.8, 0, 1))

        # Shadow → Market: final flow to equity/credit markets
        # Depends on risk appetite proxied by yield curve and vol
        curve_hist = list(self._history["t10y2y"])
        curve_signal = 0.5
        if curve_hist:
            curve_signal = float(np.clip((curve_hist[-1] + 1.0) / 3.0, 0, 1))
        rd.shadow_to_market = float(np.clip(rd.gsib_to_shadow * curve_signal, 0, 1))

        # Net market liquidity = end-of-chain flow
        rd.net_market_liquidity = rd.shadow_to_market

        # Identify bottleneck
        flows = {
            "Fed→PD": rd.fed_to_pd,
            "PD→GSIB": rd.pd_to_gsib,
            "GSIB→Shadow": rd.gsib_to_shadow,
            "Shadow→Market": rd.shadow_to_market,
        }
        rd.bottleneck = min(flows, key=flows.get)

        return rd

    def get_liquidity_score(self) -> float:
        """Return aggregate L(t) in [-1, +1].

        Positive = ample liquidity (risk-on), negative = tight (risk-off).
        """
        return self._liquidity_score

    def get_liquidity_regime(self) -> LiquidityRegime:
        """Map liquidity score to regime."""
        s = self._liquidity_score
        if s > 0.4:
            return LiquidityRegime.FLOOD
        elif s > -0.1:
            return LiquidityRegime.NORMAL
        elif s > -0.5:
            return LiquidityRegime.TIGHT
        else:
            return LiquidityRegime.DRAIN

    def get_drain_warning(self) -> DrainWarning:
        """Early warning system for liquidity drain.

        Monitors:
        - TGA rebuild (Treasury issuance drains reserves)
        - RRP collapse or spike
        - Reserve depletion
        - Credit spread blowout
        - SOFR/FF spread stress
        """
        dw = DrainWarning()
        triggers = []

        # TGA: rising TGA drains reserves
        tga_hist = list(self._history["tga"])
        if len(tga_hist) >= 5:
            tga_delta = tga_hist[-1] - np.mean(tga_hist[-5:])
            if tga_delta > 50:
                dw.tga_draining = True
                triggers.append(f"TGA rising +${tga_delta:.0f}B (draining reserves)")

        # RRP: collapsing RRP can be either sign — context matters
        rrp_hist = list(self._history["rrp"])
        if len(rrp_hist) >= 10:
            rrp_delta_pct = (rrp_hist[-1] - np.mean(rrp_hist[-10:])) / max(np.mean(rrp_hist[-10:]), 1)
            if rrp_delta_pct > 0.3:
                dw.rrp_draining = True
                triggers.append(f"RRP surging +{rrp_delta_pct*100:.0f}% (money parking at Fed)")
            elif rrp_delta_pct < -0.4 and self._balance_sheet.on_rrp < 100:
                # RRP near zero — no buffer left
                triggers.append("RRP near exhaustion — liquidity buffer depleted")

        # Reserves: falling reserves
        res_hist = list(self._history["reserves"])
        if len(res_hist) >= 5:
            res_delta = res_hist[-1] - np.mean(res_hist[-5:])
            if res_delta < -100:
                dw.reserves_falling = True
                triggers.append(f"Reserves falling ${res_delta:.0f}B")

        # Credit spreads: widening
        hy_hist = list(self._history["hy_oas"])
        if len(hy_hist) >= 10:
            hy_z = (hy_hist[-1] - np.mean(hy_hist[-10:])) / max(np.std(hy_hist[-10:]), 0.1)
            if hy_z > 1.5:
                dw.spread_widening = True
                triggers.append(f"HY OAS z-score +{hy_z:.1f} (spreads blowing out)")

        # SOFR/FF spread stress
        sofr_hist = list(self._history["sofr"])
        dff_hist = list(self._history["dff"])
        if sofr_hist and dff_hist:
            spread = sofr_hist[-1] - dff_hist[-1]
            if abs(spread) > 0.10:
                triggers.append(f"SOFR-FF spread {spread:+.2f}% (repo stress)")

        # Aggregate warning level
        dw.triggers = triggers
        n = len(triggers)
        if n == 0:
            dw.warning_level = 0
            dw.warning_label = "NONE"
            dw.recommended_beta_adj = 0.0
        elif n == 1:
            dw.warning_level = 1
            dw.warning_label = "WATCH"
            dw.recommended_beta_adj = -0.05
        elif n == 2:
            dw.warning_level = 2
            dw.warning_label = "CAUTION"
            dw.recommended_beta_adj = -0.10
        elif n == 3:
            dw.warning_level = 3
            dw.warning_label = "WARNING"
            dw.recommended_beta_adj = -0.20
        else:
            dw.warning_level = 4
            dw.warning_label = "CRITICAL"
            dw.recommended_beta_adj = -0.35

        return dw

    def get_sector_flow_allocation(self) -> SectorFlowAllocation:
        """Derive sector allocation from money flows.

        Combines:
        - Reserve distribution (liquidity reaching markets)
        - Credit impulse direction (which sectors benefit from credit growth)
        - Yield curve signal (cyclical vs defensive rotation)
        - Sector liquidity betas (sensitivity to reserve changes)
        """
        alloc = SectorFlowAllocation()
        alloc.timestamp = datetime.now().isoformat()

        # Get flow components
        rd = self.get_reserve_distribution()
        net_liq = rd.net_market_liquidity  # [0, 1]
        liq_score = self._liquidity_score  # [-1, +1]

        # Credit impulse direction
        credit_impulse = self._compute_credit_impulse_raw()

        # Yield curve signal: positive = steepening = cyclicals, negative = inversion = defensives
        curve_hist = list(self._history["t10y2y"])
        curve_signal = curve_hist[-1] if curve_hist else 0.0
        curve_normalized = float(np.clip(curve_signal / 2.0, -1, 1))

        # Score each sector
        scores = {}
        for gics_code in GICS_SECTORS:
            liq_beta = SECTOR_LIQUIDITY_BETA.get(gics_code, 0.5)
            credit_beta = SECTOR_CREDIT_BETA.get(gics_code, 0.5)

            # Liquidity component: sectors with high liq beta benefit when liquidity floods
            liq_component = liq_score * liq_beta

            # Credit component: sectors with high credit beta benefit from credit impulse
            credit_component = credit_impulse * credit_beta

            # Curve component: cyclicals benefit from steepening
            is_cyclical = gics_code in (10, 15, 20, 25, 40)
            is_defensive = gics_code in (30, 35, 55)
            if is_cyclical:
                curve_component = curve_normalized * 0.3
            elif is_defensive:
                curve_component = -curve_normalized * 0.2
            else:
                curve_component = 0.0

            # Reserve flow component
            flow_component = (net_liq - 0.5) * liq_beta * 0.5

            # Composite score
            score = float(np.clip(
                0.35 * liq_component +
                0.25 * credit_component +
                0.20 * curve_component +
                0.20 * flow_component,
                -1, 1
            ))
            scores[gics_code] = score

        alloc.sector_scores = scores

        # Convert scores to weights (softmax-style)
        alloc.sector_weights = self._scores_to_weights(scores)

        # Determine overweight / underweight
        equal_weight = 1.0 / len(GICS_SECTORS)
        for gics_code, w in alloc.sector_weights.items():
            name = GICS_SECTORS.get(gics_code, f"Sector {gics_code}")
            if w > equal_weight * 1.3:
                alloc.overweight.append(name)
            elif w < equal_weight * 0.7:
                alloc.underweight.append(name)

        # Flow regime
        if liq_score > 0.3 and credit_impulse > 0.1:
            alloc.flow_regime = "RISK_ON"
        elif liq_score < -0.3 or credit_impulse < -0.2:
            alloc.flow_regime = "RISK_OFF"
        elif abs(liq_score) < 0.15:
            alloc.flow_regime = "NEUTRAL"
        else:
            alloc.flow_regime = "TRANSITIONING"

        return alloc

    def get_balance_sheet(self) -> FedBalanceSheet:
        """Return current Fed balance sheet snapshot."""
        return self._balance_sheet

    def set_manual_balances(
        self,
        walcl: Optional[float] = None,
        reserves: Optional[float] = None,
        rrp: Optional[float] = None,
        tga: Optional[float] = None,
    ):
        """Manually set balance sheet values (for testing or overrides)."""
        if walcl is not None:
            self._balance_sheet.walcl = walcl
        if reserves is not None:
            self._balance_sheet.reserves = reserves
            self._history["reserves"].append(reserves)
        if rrp is not None:
            self._balance_sheet.on_rrp = rrp
            self._history["rrp"].append(rrp)
        if tga is not None:
            self._balance_sheet.tga = tga
            self._history["tga"].append(tga)
        self._liquidity_score = self._compute_aggregate_score()

    # ----- internal methods -----

    def _build_balance_sheet(self, df: pd.DataFrame) -> FedBalanceSheet:
        """Build FedBalanceSheet from FRED DataFrame."""
        bs = FedBalanceSheet()

        def _latest(col: str, default: float) -> float:
            if df is not None and not df.empty and col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    return float(vals.iloc[-1])
            return default

        # WALCL is in millions, convert to billions
        walcl_raw = _latest("WALCL", 7500000.0)
        bs.walcl = walcl_raw / 1000.0 if walcl_raw > 50000 else walcl_raw

        soma_raw = _latest("WSHOSHO", 4800000.0)
        bs.soma_treasuries = soma_raw / 1000.0 if soma_raw > 50000 else soma_raw

        rrp_raw = _latest("RRPONTSYD", 500000.0)
        bs.on_rrp = rrp_raw / 1000.0 if rrp_raw > 50000 else rrp_raw

        tga_raw = _latest("WTREGEN", 700000.0)
        bs.tga = tga_raw / 1000.0 if tga_raw > 50000 else tga_raw

        reserves_raw = _latest("TOTRESNS", 3200.0)
        bs.reserves = reserves_raw

        excess_raw = _latest("EXCSRESNS", 3000.0)
        bs.excess_reserves = excess_raw

        return bs

    def _append_history(
        self,
        core_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        money_df: pd.DataFrame,
    ):
        """Append latest values to rolling history deques."""
        def _append(hist_key: str, df: pd.DataFrame, col: str, scale: float = 1.0):
            if df is not None and not df.empty and col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    self._history[hist_key].append(float(vals.iloc[-1]) * scale)

        # Core series
        _append("walcl", core_df, "WALCL", 0.001 if self._is_millions(core_df, "WALCL") else 1.0)
        _append("reserves", core_df, "TOTRESNS")
        _append("rrp", core_df, "RRPONTSYD", 0.001 if self._is_millions(core_df, "RRPONTSYD") else 1.0)
        _append("tga", core_df, "WTREGEN", 0.001 if self._is_millions(core_df, "WTREGEN") else 1.0)
        _append("sofr", core_df, "SOFR")
        _append("dff", core_df, "DFF")
        _append("hy_oas", core_df, "BAMLH0A0HYM2")
        _append("ig_oas", core_df, "BAMLC0A0CM")
        _append("t10y2y", core_df, "T10Y2Y")
        _append("breakeven", core_df, "T10YIE")

        # Credit series
        _append("busloans", credit_df, "BUSLOANS")
        _append("realln", credit_df, "REALLN")

        # Money series
        _append("m2", money_df, "M2SL")
        _append("gdp", money_df, "GDP")
        _append("m2v", money_df, "M2V")

    def _is_millions(self, df: pd.DataFrame, col: str) -> bool:
        """Detect if a FRED series is reported in millions (needs /1000 for $B)."""
        if df is None or df.empty or col not in df.columns:
            return False
        vals = df[col].dropna()
        if vals.empty:
            return False
        return float(vals.iloc[-1]) > 50000

    def _compute_aggregate_score(self) -> float:
        """Compute aggregate liquidity score L(t) in [-1, +1]."""
        components = {}

        # Balance sheet: expansion = positive, contraction = negative
        walcl_hist = list(self._history["walcl"])
        if len(walcl_hist) >= 2:
            walcl_z = (walcl_hist[-1] - self._NORMS["walcl_mean"]) / self._NORMS["walcl_std"]
            components["balance_sheet"] = float(np.clip(walcl_z / 2.0, -1, 1))
        else:
            components["balance_sheet"] = 0.0

        # Reserves: higher = more liquidity
        res_hist = list(self._history["reserves"])
        if res_hist:
            res_z = (res_hist[-1] - self._NORMS["reserves_mean"]) / self._NORMS["reserves_std"]
            components["reserves"] = float(np.clip(res_z / 2.0, -1, 1))
        else:
            components["reserves"] = 0.0

        # RRP: higher = money parked at Fed = LESS market liquidity (invert)
        rrp_hist = list(self._history["rrp"])
        if rrp_hist:
            rrp_z = (rrp_hist[-1] - self._NORMS["rrp_mean"]) / self._NORMS["rrp_std"]
            components["rrp"] = float(np.clip(-rrp_z / 2.0, -1, 1))
        else:
            components["rrp"] = 0.0

        # TGA: higher = Treasury hoarding cash = drain (invert)
        tga_hist = list(self._history["tga"])
        if tga_hist:
            tga_z = (tga_hist[-1] - self._NORMS["tga_mean"]) / self._NORMS["tga_std"]
            components["tga"] = float(np.clip(-tga_z / 2.0, -1, 1))
        else:
            components["tga"] = 0.0

        # SOFR/FF spread: wider = funding stress (invert)
        sofr_hist = list(self._history["sofr"])
        dff_hist = list(self._history["dff"])
        if sofr_hist and dff_hist:
            spread = sofr_hist[-1] - dff_hist[-1]
            spread_z = (spread - self._NORMS["sofr_dff_spread_mean"]) / self._NORMS["sofr_dff_spread_std"]
            components["sofr_spread"] = float(np.clip(-spread_z / 2.0, -1, 1))
        else:
            components["sofr_spread"] = 0.0

        # Credit spreads: wider = tighter conditions (invert)
        hy_hist = list(self._history["hy_oas"])
        if hy_hist:
            hy_z = (hy_hist[-1] - self._NORMS["hy_oas_mean"]) / self._NORMS["hy_oas_std"]
            components["credit_spread"] = float(np.clip(-hy_z / 2.0, -1, 1))
        else:
            components["credit_spread"] = 0.0

        # Yield curve: positive slope = expansionary
        curve_hist = list(self._history["t10y2y"])
        if curve_hist:
            components["yield_curve"] = float(np.clip(curve_hist[-1] / 2.0, -1, 1))
        else:
            components["yield_curve"] = 0.0

        # Weighted sum
        score = sum(
            self._SCORE_WEIGHTS.get(k, 0) * v
            for k, v in components.items()
        )
        return float(np.clip(score, -1, 1))

    def _compute_credit_impulse_raw(self) -> float:
        """Compute raw credit impulse from loan growth."""
        bl_hist = list(self._history["busloans"])
        rl_hist = list(self._history["realln"])

        impulse = 0.0
        n = 0
        if len(bl_hist) >= 2:
            bl_delta = (bl_hist[-1] - bl_hist[-2]) / max(abs(bl_hist[-2]), 1)
            impulse += bl_delta
            n += 1
        if len(rl_hist) >= 2:
            rl_delta = (rl_hist[-1] - rl_hist[-2]) / max(abs(rl_hist[-2]), 1)
            impulse += rl_delta
            n += 1

        if n > 0:
            impulse /= n
        return float(np.clip(impulse * 10, -1, 1))  # Scale small pct changes

    @staticmethod
    def _scores_to_weights(scores: Dict[int, float]) -> Dict[int, float]:
        """Convert sector scores to portfolio weights via shifted softmax."""
        if not scores:
            return {}
        codes = sorted(scores.keys())
        raw = np.array([scores[c] for c in codes])
        # Shift to positive domain and exponentiate (temperature=2)
        shifted = np.exp((raw + 1.0) / 2.0)
        total = shifted.sum()
        if total <= 0:
            equal = 1.0 / len(codes)
            return {c: equal for c in codes}
        weights = shifted / total
        return {c: float(w) for c, w in zip(codes, weights)}


# ---------------------------------------------------------------------------
# CubeLiquidityTensor
# ---------------------------------------------------------------------------
class CubeLiquidityTensor:
    """Multi-dimensional liquidity view — the full tensor.

    Axes:
        Axis 0: Time (rolling window)
        Axis 1: Flow nodes (Fed, PD, GSIB, Shadow, Market)
        Axis 2: Asset classes / GICS sectors
        Axis 3: Instruments (reserves, credit, collateral, rates)

    Computes:
        1. Reserve Distribution Vector
        2. Sector Flow Matrix (money velocity per GICS sector)
        3. Credit Impulse
        4. Collateral Velocity (rehypothecation chain)
        5. Dealer Balance Sheet Capacity
    """

    _FLOW_NODES = ["Fed", "PrimaryDealer", "GSIB", "ShadowBank", "Market"]
    _INSTRUMENT_AXES = ["reserves", "credit", "collateral", "rates"]

    def __init__(self, plumbing: Optional[FedLiquidityPlumbing] = None):
        self._plumbing = plumbing or FedLiquidityPlumbing()
        self._tensor: Optional[np.ndarray] = None
        self._sector_velocity: Dict[int, float] = {}
        self._credit_impulse = CreditImpulseState()
        self._collateral = CollateralChain()

    @property
    def plumbing(self) -> FedLiquidityPlumbing:
        return self._plumbing

    def compute_tensor(self) -> np.ndarray:
        """Build the full 4D liquidity tensor.

        Shape: (T, 5, 11, 4) where:
            T = time steps (1 for current snapshot, expandable)
            5 = flow nodes
            11 = GICS sectors
            4 = instrument axes
        """
        n_nodes = len(self._FLOW_NODES)
        n_sectors = len(GICS_SECTORS)
        n_instruments = len(self._INSTRUMENT_AXES)

        tensor = np.zeros((1, n_nodes, n_sectors, n_instruments))

        # Get flow distribution
        rd = self._plumbing.get_reserve_distribution()
        flows = [rd.fed_to_pd, rd.pd_to_gsib, rd.gsib_to_shadow, rd.shadow_to_market, rd.net_market_liquidity]

        # Get credit impulse
        self._credit_impulse = self._compute_credit_impulse()

        # Get collateral chain
        self._collateral = self._compute_collateral_chain()

        # Fill tensor
        gics_codes = sorted(GICS_SECTORS.keys())
        for i_node, flow_val in enumerate(flows):
            for j_sector, gics_code in enumerate(gics_codes):
                liq_beta = SECTOR_LIQUIDITY_BETA.get(gics_code, 0.5)
                credit_beta = SECTOR_CREDIT_BETA.get(gics_code, 0.5)

                # Axis 0 (reserves): flow * sector liquidity beta
                tensor[0, i_node, j_sector, 0] = flow_val * liq_beta

                # Axis 1 (credit): credit impulse * sector credit beta
                tensor[0, i_node, j_sector, 1] = self._credit_impulse.impulse * credit_beta * flow_val

                # Axis 2 (collateral): collateral velocity scaled by node position
                decay = 1.0 - (i_node / n_nodes) * 0.3  # Decays along chain
                tensor[0, i_node, j_sector, 2] = self._collateral.velocity / 5.0 * decay * liq_beta

                # Axis 3 (rates): SOFR/FF spread impact scaled by sector rate sensitivity
                rate_sens = 0.7 if gics_code in (40, 55, 60) else 0.3
                sofr_hist = list(self._plumbing._history["sofr"])
                dff_hist = list(self._plumbing._history["dff"])
                if sofr_hist and dff_hist:
                    rate_signal = float(np.clip(-(sofr_hist[-1] - dff_hist[-1]) / 0.5, -1, 1))
                else:
                    rate_signal = 0.0
                tensor[0, i_node, j_sector, 3] = rate_signal * rate_sens

        self._tensor = tensor

        # Compute sector velocity from the market node (last row)
        self._sector_velocity = {}
        for j, gics_code in enumerate(gics_codes):
            # Sector velocity = sum across instrument axes at market node
            self._sector_velocity[gics_code] = float(tensor[0, -1, j, :].sum())

        logger.info(f"Liquidity tensor computed: shape={tensor.shape}")
        return tensor

    def get_sector_velocity(self) -> Dict[int, float]:
        """Return money velocity per GICS sector.

        Higher velocity = more liquidity flowing into the sector.
        Returns dict of {GICS code: velocity score}.
        """
        if not self._sector_velocity:
            self.compute_tensor()
        return self._sector_velocity

    def get_credit_impulse(self) -> CreditImpulseState:
        """Return rate of change of credit (new lending momentum)."""
        if self._credit_impulse.impulse == 0.0 and self._tensor is None:
            self._credit_impulse = self._compute_credit_impulse()
        return self._credit_impulse

    def get_collateral_chain(self) -> CollateralChain:
        """Return collateral velocity / rehypothecation depth.

        Higher velocity = collateral being reused more times = more leverage.
        Lower velocity = collateral hoarding = tighter conditions.
        """
        if self._collateral.velocity == 2.5 and self._tensor is None:
            self._collateral = self._compute_collateral_chain()
        return self._collateral

    def trace_money_flow(self) -> Dict[str, float]:
        """End-to-end flow tracing: follow the money from Fed to each sector.

        Returns dict mapping each sector name to the net flow arriving there.
        """
        rd = self._plumbing.get_reserve_distribution()
        alloc = self._plumbing.get_sector_flow_allocation()

        result = {}
        for gics_code, name in GICS_SECTORS.items():
            liq_beta = SECTOR_LIQUIDITY_BETA.get(gics_code, 0.5)
            # Net flow = market liquidity * sector weight * liquidity beta
            sector_weight = alloc.sector_weights.get(gics_code, 1.0 / len(GICS_SECTORS))
            net_flow = rd.net_market_liquidity * sector_weight * liq_beta
            result[name] = float(np.clip(net_flow, -1, 1))

        return result

    def get_allocation_signal(self) -> SectorFlowAllocation:
        """Sector allocation recommendation combining all tensor dimensions.

        Merges:
        - Liquidity flow (where reserves are going)
        - Credit impulse (which sectors benefit from lending)
        - Collateral velocity (leverage capacity)
        - Sector velocity (money velocity per sector)
        """
        # Ensure tensor is computed
        if self._tensor is None:
            self.compute_tensor()

        alloc = SectorFlowAllocation()
        alloc.timestamp = datetime.now().isoformat()

        # Base allocation from plumbing
        base_alloc = self._plumbing.get_sector_flow_allocation()

        # Adjust with tensor velocity
        velocity = self.get_sector_velocity()
        credit = self.get_credit_impulse()
        collateral = self.get_collateral_chain()

        scores = {}
        for gics_code in GICS_SECTORS:
            base_score = base_alloc.sector_scores.get(gics_code, 0.0)
            vel_score = velocity.get(gics_code, 0.0)
            credit_adj = credit.impulse * SECTOR_CREDIT_BETA.get(gics_code, 0.5) * 0.2
            collateral_adj = (collateral.dealer_capacity * 0.1) if gics_code == 40 else 0.0

            composite = float(np.clip(
                0.40 * base_score +
                0.30 * vel_score +
                0.20 * credit_adj +
                0.10 * collateral_adj,
                -1, 1
            ))
            scores[gics_code] = composite

        alloc.sector_scores = scores
        alloc.sector_weights = self._plumbing._scores_to_weights(scores)

        # Over/underweight
        equal_weight = 1.0 / len(GICS_SECTORS)
        for gics_code, w in alloc.sector_weights.items():
            name = GICS_SECTORS.get(gics_code, f"Sector {gics_code}")
            if w > equal_weight * 1.3:
                alloc.overweight.append(name)
            elif w < equal_weight * 0.7:
                alloc.underweight.append(name)

        # Regime
        liq_score = self._plumbing.get_liquidity_score()
        if liq_score > 0.3 and credit.impulse > 0:
            alloc.flow_regime = "RISK_ON"
        elif liq_score < -0.3 or credit.impulse < -0.2:
            alloc.flow_regime = "RISK_OFF"
        else:
            alloc.flow_regime = "NEUTRAL"

        return alloc

    # ----- internal -----

    def _compute_credit_impulse(self) -> CreditImpulseState:
        """Compute credit impulse from business + real estate loan growth."""
        ci = CreditImpulseState()

        bl_hist = list(self._plumbing._history["busloans"])
        rl_hist = list(self._plumbing._history["realln"])

        deltas = []
        if len(bl_hist) >= 2:
            ci.business_loans_delta = (bl_hist[-1] - bl_hist[-2]) / max(abs(bl_hist[-2]), 1)
            deltas.append(ci.business_loans_delta)
        if len(rl_hist) >= 2:
            ci.real_estate_loans_delta = (rl_hist[-1] - rl_hist[-2]) / max(abs(rl_hist[-2]), 1)
            deltas.append(ci.real_estate_loans_delta)

        if deltas:
            raw_impulse = np.mean(deltas)
            ci.impulse = float(np.clip(raw_impulse * 10, -1, 1))
        else:
            ci.impulse = 0.0

        # Trend
        if len(bl_hist) >= 5:
            recent = np.mean(bl_hist[-3:])
            older = np.mean(bl_hist[-5:-2]) if len(bl_hist) >= 5 else bl_hist[-3]
            if recent > older * 1.01:
                ci.impulse_trend = "EXPANDING"
            elif recent < older * 0.99:
                ci.impulse_trend = "CONTRACTING"
            else:
                ci.impulse_trend = "STABLE"

        return ci

    def _compute_collateral_chain(self) -> CollateralChain:
        """Estimate collateral velocity and dealer capacity.

        Collateral velocity (rehypothecation length) is estimated from:
        - Credit spread tightness (tight spreads → more re-use)
        - Reserve abundance (more reserves → less need for rehyp)
        - Fed facility usage (SRF usage → collateral scarcity)
        """
        cc = CollateralChain()

        # Base rehypothecation length ~2.5x (pre-GFC was ~3.0, post-GFC ~2.0-2.5)
        base_velocity = 2.5

        # Credit spread adjustment: tighter spreads → more collateral reuse
        hy_hist = list(self._plumbing._history["hy_oas"])
        spread_adj = 0.0
        if hy_hist:
            current = hy_hist[-1]
            # Below 3.5% = tight, above 5% = wide
            spread_adj = float(np.clip((3.5 - current) / 3.0, -0.5, 0.5))

        # Reserve adjustment: ample reserves → less need for rehyp (reduces velocity)
        res_hist = list(self._plumbing._history["reserves"])
        reserve_adj = 0.0
        if res_hist:
            res_z = (res_hist[-1] - 3200) / 500
            reserve_adj = float(np.clip(-res_z * 0.2, -0.3, 0.3))

        cc.velocity = float(np.clip(base_velocity + spread_adj + reserve_adj, 1.0, 4.0))

        # Dealer capacity: how much balance sheet room PDs have
        # Proxied from reserve level and spread conditions
        reserve_signal = float(np.clip((res_hist[-1] - 2800) / 800, -1, 1)) if res_hist else 0.0
        spread_signal = float(np.clip((4.0 - hy_hist[-1]) / 3.0, -1, 1)) if hy_hist else 0.0
        cc.dealer_capacity = float(np.clip(0.5 * reserve_signal + 0.5 * spread_signal, -1, 1))

        # Collateral scarcity: high RRP + tight spreads = collateral is scarce
        rrp_hist = list(self._plumbing._history["rrp"])
        if rrp_hist and hy_hist:
            rrp_signal = float(np.clip(rrp_hist[-1] / 1500.0, 0, 1))
            cc.collateral_scarcity = float(np.clip(rrp_signal - spread_signal * 0.5, -1, 1))

        # SRF usage proxy: SOFR/FF spread stress indicates need for SRF
        sofr_hist = list(self._plumbing._history["sofr"])
        dff_hist = list(self._plumbing._history["dff"])
        if sofr_hist and dff_hist:
            srf_proxy = abs(sofr_hist[-1] - dff_hist[-1])
            cc.srf_usage = float(np.clip(srf_proxy / 0.20, 0, 1))

        return cc


# ---------------------------------------------------------------------------
# MoneyVelocityTracker
# ---------------------------------------------------------------------------
class MoneyVelocityTracker:
    """GDP/M2 velocity tracking — Fisher equation V = GDP/M2.

    Money velocity is a leading indicator for inflation/deflation.
    Tracks:
    - Aggregate velocity (V = GDP / M2)
    - Velocity regime (accelerating / decelerating / stable)
    - Sector-level absorption (where new money is going)
    - Flow distribution (equities vs bonds vs alternatives)

    FRED series: M2SL, M2V, GDP
    """

    # Regime thresholds (annualized velocity change)
    _ACCEL_THRESHOLD = 0.02   # +2% YoY → accelerating
    _DECEL_THRESHOLD = -0.02  # -2% YoY → decelerating

    # Sector absorption coefficients (how much each sector absorbs M2 growth)
    # Higher = absorbs more new money when M2 expands
    _SECTOR_ABSORPTION = {
        10: 0.08,   # Energy — moderate absorber
        15: 0.05,   # Materials — low
        20: 0.10,   # Industrials — capex absorber
        25: 0.14,   # Consumer Disc — high (consumer spending)
        30: 0.06,   # Consumer Staples — low (defensive)
        35: 0.09,   # Health Care — moderate
        40: 0.18,   # Financials — highest (direct conduit)
        45: 0.15,   # Info Tech — high (growth magnet)
        50: 0.07,   # Comm Services — moderate
        55: 0.04,   # Utilities — low
        60: 0.04,   # Real Estate — low (rate-sensitive, absorbs via mortgages)
    }

    def __init__(self, plumbing: Optional[FedLiquidityPlumbing] = None):
        self._plumbing = plumbing or FedLiquidityPlumbing()
        self._velocity_history: deque = deque(maxlen=60)
        self._m2_history: deque = deque(maxlen=60)
        self._gdp_history: deque = deque(maxlen=60)

    def compute_velocity(self) -> float:
        """Compute current money velocity V = GDP / M2.

        Uses FRED M2V directly if available, otherwise computes from GDP and M2.
        Returns the velocity value (typically 1.0-2.0 in modern era).
        """
        m2v_hist = list(self._plumbing._history.get("m2v", []))
        if m2v_hist:
            v = m2v_hist[-1]
            self._velocity_history.append(v)
            return float(v)

        # Fallback: compute from GDP / M2
        gdp_hist = list(self._plumbing._history.get("gdp", []))
        m2_hist = list(self._plumbing._history.get("m2", []))

        if gdp_hist and m2_hist:
            gdp = gdp_hist[-1]
            m2 = m2_hist[-1]
            if m2 > 0:
                # GDP is quarterly annualized, M2 is monthly — both in billions
                v = gdp / m2
                self._velocity_history.append(v)
                self._m2_history.append(m2)
                self._gdp_history.append(gdp)
                return float(v)

        # Default: ~1.2 (post-COVID average)
        return 1.2

    def get_velocity_regime(self) -> VelocityRegime:
        """Determine if velocity is accelerating, decelerating, or stable.

        Uses rolling change in velocity over recent observations.
        """
        v_hist = list(self._velocity_history)
        if len(v_hist) < 4:
            return VelocityRegime.STABLE

        # Compute annualized rate of change
        recent = np.mean(v_hist[-2:])
        older = np.mean(v_hist[-4:-2])
        if older <= 0:
            return VelocityRegime.STABLE

        pct_change = (recent - older) / older

        if pct_change > self._ACCEL_THRESHOLD:
            return VelocityRegime.ACCELERATING
        elif pct_change < self._DECEL_THRESHOLD:
            return VelocityRegime.DECELERATING
        else:
            return VelocityRegime.STABLE

    def get_sector_absorption(self) -> Dict[str, float]:
        """Determine which sectors are absorbing the most liquidity.

        Combines:
        - Sector absorption coefficients (structural)
        - Current M2 growth rate (cyclical)
        - Velocity regime (directional)

        Returns dict of {sector_name: absorption_score} in [-1, +1].
        Positive = sector is absorbing liquidity (bullish flow).
        Negative = sector is losing liquidity (bearish flow).
        """
        m2_hist = list(self._plumbing._history.get("m2", []))
        m2_growth = 0.0
        if len(m2_hist) >= 2:
            m2_growth = (m2_hist[-1] - m2_hist[-2]) / max(abs(m2_hist[-2]), 1)

        velocity = self.compute_velocity()
        regime = self.get_velocity_regime()

        # Regime multiplier: accelerating velocity amplifies sector effects
        regime_mult = {
            VelocityRegime.ACCELERATING: 1.3,
            VelocityRegime.STABLE: 1.0,
            VelocityRegime.DECELERATING: 0.7,
        }.get(regime, 1.0)

        result = {}
        for gics_code, name in GICS_SECTORS.items():
            base_absorption = self._SECTOR_ABSORPTION.get(gics_code, 0.05)

            # Flow = absorption coefficient * M2 growth * regime multiplier
            # Scale by velocity level (higher velocity = money turning over faster)
            flow = base_absorption * m2_growth * regime_mult * velocity * 100
            score = float(np.clip(flow, -1, 1))
            result[name] = score

        return result

    def get_flow_distribution(self) -> Dict[str, float]:
        """Where new money is going: equities vs bonds vs alternatives.

        Returns approximate distribution based on:
        - Credit spreads (tight → equities, wide → bonds/safety)
        - Yield curve (steep → equities, inverted → bonds)
        - Money velocity (rising → equities, falling → cash/bonds)
        """
        liq_score = self._plumbing.get_liquidity_score()
        velocity = self.compute_velocity()
        regime = self.get_velocity_regime()

        # Base allocation shifts with liquidity regime
        equities = 0.50
        bonds = 0.30
        alternatives = 0.10
        cash = 0.10

        # Adjustments
        if regime == VelocityRegime.ACCELERATING:
            equities += 0.10
            bonds -= 0.05
            cash -= 0.05
        elif regime == VelocityRegime.DECELERATING:
            equities -= 0.10
            bonds += 0.05
            cash += 0.05

        if liq_score > 0.3:
            equities += 0.05
            alternatives += 0.05
            bonds -= 0.05
            cash -= 0.05
        elif liq_score < -0.3:
            equities -= 0.10
            bonds += 0.05
            cash += 0.05

        # Normalize
        total = equities + bonds + alternatives + cash
        if total > 0:
            equities /= total
            bonds /= total
            alternatives /= total
            cash /= total

        return {
            "equities": round(float(equities), 4),
            "bonds": round(float(bonds), 4),
            "alternatives": round(float(alternatives), 4),
            "cash": round(float(cash), 4),
        }

    def get_inflation_signal(self) -> float:
        """Velocity-based inflation signal in [-1, +1].

        Rising velocity → inflationary pressure (+).
        Falling velocity → deflationary pressure (-).
        """
        v_hist = list(self._velocity_history)
        if len(v_hist) < 2:
            return 0.0

        # Rate of change
        recent = v_hist[-1]
        older = v_hist[0] if len(v_hist) < 4 else np.mean(v_hist[:len(v_hist)//2])
        if older <= 0:
            return 0.0

        pct_change = (recent - older) / older

        # Breakeven inflation confirmation
        be_hist = list(self._plumbing._history.get("breakeven", []))
        be_signal = 0.0
        if be_hist:
            # 10Y breakeven: 2.0% = neutral, >2.5% = inflationary, <1.5% = deflationary
            be_signal = float(np.clip((be_hist[-1] - 2.0) / 1.5, -1, 1))

        # Combine velocity change and breakeven
        velocity_signal = float(np.clip(pct_change * 10, -1, 1))
        return float(np.clip(0.6 * velocity_signal + 0.4 * be_signal, -1, 1))


# ---------------------------------------------------------------------------
# Convenience: full tensor output builder
# ---------------------------------------------------------------------------
def build_liquidity_tensor_output(
    plumbing: Optional[FedLiquidityPlumbing] = None,
) -> LiquidityTensorOutput:
    """Build a complete LiquidityTensorOutput snapshot.

    Orchestrates FedLiquidityPlumbing, CubeLiquidityTensor, and
    MoneyVelocityTracker into a single output object.
    """
    plumbing = plumbing or FedLiquidityPlumbing()

    tensor = CubeLiquidityTensor(plumbing)
    velocity_tracker = MoneyVelocityTracker(plumbing)

    # Compute all components
    tensor.compute_tensor()

    out = LiquidityTensorOutput()
    out.liquidity_score = plumbing.get_liquidity_score()
    out.regime = plumbing.get_liquidity_regime()
    out.balance_sheet = plumbing.get_balance_sheet()
    out.reserve_distribution = plumbing.get_reserve_distribution()
    out.sector_allocation = tensor.get_allocation_signal()
    out.credit_impulse = tensor.get_credit_impulse()
    out.collateral = tensor.get_collateral_chain()
    out.drain_warning = plumbing.get_drain_warning()
    out.velocity = velocity_tracker.compute_velocity()
    out.velocity_regime = velocity_tracker.get_velocity_regime()
    out.timestamp = datetime.now().isoformat()

    return out
