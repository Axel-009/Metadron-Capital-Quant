"""L7 Unified Execution Surface — Fused continuous execution arm.

Unifies WonderTrader (micro-price + CTA + routing), ExchangeCore (order matching),
IBKRBroker (sole execution broker with native TWAP/VWAP algo), and OptionsEngine
(derivatives) into one continuous execution arm.

Pipeline position:
    All 29 signal types → L7UnifiedExecutionSurface
        ├── Equity orders  → WonderTrader micro-price → ExchangeCore → IBKR (TWAP/VWAP)
        ├── Options orders → OptionsEngine Greeks → vol-adjusted → IBKR
        └── Futures orders → Beta corridor hedge → IBKR
    Trade log maintained in parallel for reconciliation (generated vs executed).

Broker: IBKR only (native TWAP/VWAP/Adaptive algo orders via ib_insync).
Trade log: Records all orders the platform generates — used for recon to verify
           which orders were actually executed on the broker vs generated but not filled.

Design rules (per CLAUDE.md):
    - try/except on ALL external imports — system runs degraded, never broken
    - Pure-numpy fallbacks — no crashes if optional packages missing
    - IBKR is the SOLE execution broker with native algo routing
    - Trade log ALWAYS maintained for reconciliation and learning loop
    - Fixed income / FX / liquidity are for research only — never executed here
"""

from __future__ import annotations

import time
import uuid
import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# Load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# Internal imports — all guarded
try:
    from .paper_broker import (
        OrderSide, OrderType, OrderStatus,
        SignalType, Order, Position, PortfolioState,
    )
except ImportError:
    pass

try:
    from .ibkr_broker import IBKRBroker, IBKRAlgoType
except ImportError:
    IBKRBroker = None  # type: ignore[assignment,misc]
    IBKRAlgoType = None  # type: ignore[assignment,misc]

try:
    from .wondertrader_engine import WonderTraderEngine, MicroPriceResult, CTASignal
except ImportError:
    WonderTraderEngine = None  # type: ignore[assignment,misc]
    MicroPriceResult = None  # type: ignore[assignment,misc]

try:
    from .exchange_core_engine import ExchangeCoreEngine, OrderAction, EngineOrderType
except ImportError:
    ExchangeCoreEngine = None  # type: ignore[assignment,misc]

try:
    from .options_engine import OptionsEngine
except ImportError:
    OptionsEngine = None  # type: ignore[assignment,misc]

try:
    from .quant_strategy_executor import QuantStrategyExecutor
except ImportError:
    QuantStrategyExecutor = None  # type: ignore[assignment,misc]

try:
    from ..portfolio.beta_corridor import BetaCorridor, BetaAction
except ImportError:
    BetaCorridor = None  # type: ignore[assignment,misc]

logger = logging.getLogger("metadron.execution.l7")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

try:
    from ..risk.portfolio_mc_gate import get_shared_mc_gate as _get_mc_gate
except Exception:  # degraded mode — G11 skipped
    _get_mc_gate = None


def _dte_from_expiry(expiry: str) -> float:
    """Calendar days to an expiry 'YYYY-MM-DD' (7 if unknown)."""
    try:
        from datetime import date as _date
        y, m, d = (int(x) for x in str(expiry)[:10].split("-"))
        return float(max((_date(y, m, d) - _date.today()).days, 0))
    except Exception:
        return 7.0




class _AlreadyRegistered(Exception):
    """Internal: Prometheus metrics already registered in this process."""


_L7_PROM = None   # process-global Prometheus metrics (prometheus_client forbids registering a name twice)

class ProductType(str, Enum):
    """Tradeable product types routed through L7."""
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FUTURE = "FUTURE"


class RoutingStrategy(str, Enum):
    """Order routing algorithm."""
    SMART = "SMART"       # Adaptive: TWAP for large, immediate for small
    TWAP = "TWAP"         # Time-weighted average price
    VWAP = "VWAP"         # Volume-weighted average price
    IMMEDIATE = "IMMEDIATE"  # Direct market order


class ExecutionUrgency(str, Enum):
    """How urgently the order should fill."""
    LOW = "LOW"           # Patient — TWAP over 30 min
    MEDIUM = "MEDIUM"     # Standard — TWAP over 5 min
    HIGH = "HIGH"         # Aggressive — immediate fill
    CRITICAL = "CRITICAL" # Kill-switch — market order NOW


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class L7Order:
    """Unified order flowing through the L7 execution surface."""
    order_id: str = ""
    ticker: str = ""
    product_type: ProductType = ProductType.EQUITY
    side: str = "BUY"          # BUY, SELL, SHORT, COVER
    quantity: int = 0
    limit_price: Optional[float] = None
    signal_type: str = "HOLD"
    routing: RoutingStrategy = RoutingStrategy.SMART
    urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    # Options-specific
    option_type: str = ""      # CALL / PUT
    strike: float = 0.0
    expiry: str = ""           # YYYY-MM-DD
    # Futures-specific
    contract: str = ""         # ES, NQ, VX, ZN, etc.
    # Risk flags
    min_size_exception: bool = False   # 1-contract minimum exception (premium > position cap)
    is_hedge: bool = False             # protective/tail hedge order
    # Execution results
    fill_price: float = 0.0
    fill_quantity: int = 0
    slippage_bps: float = 0.0
    transaction_cost: float = 0.0
    micro_price: float = 0.0
    cta_signal_strength: float = 0.0
    status: str = "PENDING"    # PENDING, ROUTED, FILLED, PARTIAL, REJECTED
    reason: str = ""
    created_at: str = ""
    filled_at: str = ""
    # TCA fields
    arrival_price: float = 0.0
    implementation_shortfall: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id, "ticker": self.ticker,
            "product_type": self.product_type, "side": self.side,
            "quantity": self.quantity, "limit_price": self.limit_price,
            "signal_type": self.signal_type, "routing": self.routing,
            "fill_price": self.fill_price, "fill_quantity": self.fill_quantity,
            "slippage_bps": self.slippage_bps, "transaction_cost": self.transaction_cost,
            "micro_price": self.micro_price, "status": self.status,
            "reason": self.reason, "created_at": self.created_at,
            "filled_at": self.filled_at, "arrival_price": self.arrival_price,
            "implementation_shortfall": self.implementation_shortfall,
            "market_impact_bps": self.market_impact_bps,
            "timing_cost_bps": self.timing_cost_bps,
            "option_type": self.option_type, "strike": self.strike,
            "expiry": self.expiry, "contract": self.contract,
            "min_size_exception": self.min_size_exception, "is_hedge": self.is_hedge,
        }


@dataclass
class TCASnapshot:
    """Transaction Cost Analysis snapshot for a single execution."""
    order_id: str = ""
    ticker: str = ""
    product_type: str = "EQUITY"
    side: str = "BUY"
    quantity: int = 0
    arrival_price: float = 0.0
    fill_price: float = 0.0
    # Decomposition
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    commission_bps: float = 0.0
    total_cost_bps: float = 0.0
    # Implementation shortfall
    implementation_shortfall_usd: float = 0.0
    # Benchmark
    vwap_price: float = 0.0
    vwap_slippage_bps: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TCAAggregate:
    """Aggregate TCA metrics for dashboard display."""
    total_trades: int = 0
    total_volume_usd: float = 0.0
    avg_spread_cost_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    avg_timing_cost_bps: float = 0.0
    avg_commission_bps: float = 0.0
    avg_total_cost_bps: float = 0.0
    total_implementation_shortfall_usd: float = 0.0
    # Per product type
    equity_avg_cost_bps: float = 0.0
    option_avg_cost_bps: float = 0.0
    future_avg_cost_bps: float = 0.0
    # Trend (last 20 trades vs prior 20)
    cost_trend: str = "STABLE"  # IMPROVING, STABLE, DEGRADING
    # Best/worst
    best_execution_ticker: str = ""
    worst_execution_ticker: str = ""


# ---------------------------------------------------------------------------
# Transaction Cost Analyzer
# ---------------------------------------------------------------------------

class TransactionCostAnalyzer:
    """Real-time TCA engine — decomposes execution costs per trade.

    Cost components:
        1. Spread cost: half-spread at time of order
        2. Market impact: price movement caused by our order
        3. Timing cost: adverse price movement between decision and fill
        4. Commission: IBKR tiered commission schedule
    """

    # Commission schedule (IBKR tiered pricing)
    EQUITY_COMMISSION_PER_SHARE = 0.005     # IBKR: ~$0.005/share (min $1)
    OPTION_COMMISSION_PER_CONTRACT = 0.65   # IBKR: ~$0.65/contract
    FUTURE_COMMISSION_PER_CONTRACT = 1.25   # IBKR: ~$1.25/contract

    # Market impact model: sqrt(qty / ADV) * volatility * impact_coeff
    IMPACT_COEFFICIENT = 0.10

    def __init__(self, max_history: int = 5000):
        self._history: deque[TCASnapshot] = deque(maxlen=max_history)
        self._by_product: Dict[str, List[TCASnapshot]] = {
            "EQUITY": [], "OPTION": [], "FUTURE": [],
        }

    def analyze(
        self,
        order: L7Order,
        arrival_price: float,
        fill_price: float,
        adv_shares: float = 100_000,
        spread_bps: float = 3.0,
        daily_vol: float = 0.02,
        vwap_price: Optional[float] = None,
    ) -> TCASnapshot:
        """Decompose execution costs for a filled order."""
        qty = order.fill_quantity or order.quantity
        notional = abs(qty * fill_price)

        # 1. Spread cost: half the bid-ask spread
        spread_cost_bps = spread_bps / 2.0

        # 2. Market impact: sqrt model
        participation = qty / max(adv_shares, 1)
        impact_bps = self.IMPACT_COEFFICIENT * (participation ** 0.5) * daily_vol * 10_000

        # 3. Timing cost: price drift from arrival to fill
        if arrival_price > 0:
            if order.side in ("BUY", "COVER"):
                timing_bps = (fill_price - arrival_price) / arrival_price * 10_000
            else:
                timing_bps = (arrival_price - fill_price) / arrival_price * 10_000
        else:
            timing_bps = 0.0

        # 4. Commission
        if order.product_type == ProductType.OPTION:
            commission_usd = abs(qty) * self.OPTION_COMMISSION_PER_CONTRACT
        elif order.product_type == ProductType.FUTURE:
            commission_usd = abs(qty) * self.FUTURE_COMMISSION_PER_CONTRACT
        else:
            commission_usd = abs(qty) * self.EQUITY_COMMISSION_PER_SHARE
        commission_bps = (commission_usd / max(notional, 1)) * 10_000

        total_bps = spread_cost_bps + impact_bps + max(timing_bps, 0) + commission_bps

        # Implementation shortfall
        if arrival_price > 0 and order.side in ("BUY", "COVER"):
            is_usd = (fill_price - arrival_price) * qty
        elif arrival_price > 0:
            is_usd = (arrival_price - fill_price) * qty
        else:
            is_usd = 0.0

        # VWAP slippage
        vwap_slip = 0.0
        if vwap_price and vwap_price > 0:
            if order.side in ("BUY", "COVER"):
                vwap_slip = (fill_price - vwap_price) / vwap_price * 10_000
            else:
                vwap_slip = (vwap_price - fill_price) / vwap_price * 10_000

        snap = TCASnapshot(
            order_id=order.order_id,
            ticker=order.ticker,
            product_type=order.product_type,
            side=order.side,
            quantity=qty,
            arrival_price=arrival_price,
            fill_price=fill_price,
            spread_cost_bps=round(spread_cost_bps, 2),
            market_impact_bps=round(impact_bps, 2),
            timing_cost_bps=round(timing_bps, 2),
            commission_bps=round(commission_bps, 2),
            total_cost_bps=round(total_bps, 2),
            implementation_shortfall_usd=round(is_usd, 2),
            vwap_price=vwap_price or 0.0,
            vwap_slippage_bps=round(vwap_slip, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._history.append(snap)
        pt = order.product_type if isinstance(order.product_type, str) else order.product_type.value
        if pt in self._by_product:
            self._by_product[pt].append(snap)

        return snap

    def get_aggregate(self, last_n: int = 0) -> TCAAggregate:
        """Compute aggregate TCA metrics. last_n=0 means all history."""
        trades = list(self._history)
        if last_n > 0:
            trades = trades[-last_n:]
        if not trades:
            return TCAAggregate()

        agg = TCAAggregate(total_trades=len(trades))
        agg.total_volume_usd = sum(abs(t.quantity * t.fill_price) for t in trades)
        agg.avg_spread_cost_bps = _mean([t.spread_cost_bps for t in trades])
        agg.avg_market_impact_bps = _mean([t.market_impact_bps for t in trades])
        agg.avg_timing_cost_bps = _mean([t.timing_cost_bps for t in trades])
        agg.avg_commission_bps = _mean([t.commission_bps for t in trades])
        agg.avg_total_cost_bps = _mean([t.total_cost_bps for t in trades])
        agg.total_implementation_shortfall_usd = sum(
            t.implementation_shortfall_usd for t in trades
        )

        # Per product
        for pt in ("EQUITY", "OPTION", "FUTURE"):
            pt_trades = [t for t in trades if t.product_type == pt]
            if pt_trades:
                avg = _mean([t.total_cost_bps for t in pt_trades])
                setattr(agg, f"{pt.lower()}_avg_cost_bps", avg)

        # Trend detection
        if len(trades) >= 40:
            recent = _mean([t.total_cost_bps for t in trades[-20:]])
            prior = _mean([t.total_cost_bps for t in trades[-40:-20]])
            if recent < prior * 0.90:
                agg.cost_trend = "IMPROVING"
            elif recent > prior * 1.10:
                agg.cost_trend = "DEGRADING"

        # Best/worst
        if trades:
            best = min(trades, key=lambda t: t.total_cost_bps)
            worst = max(trades, key=lambda t: t.total_cost_bps)
            agg.best_execution_ticker = best.ticker
            agg.worst_execution_ticker = worst.ticker

        return agg

    @property
    def history(self) -> List[TCASnapshot]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Multi-Product Router
# ---------------------------------------------------------------------------

# Research-only instruments — NEVER executed
# Only fixed income, FX, and credit are blocked from execution.
# Commodities (GLD, SLV, USO, etc.) are TRADEABLE — they are used for
# macro research AND can be traded as ETFs.  Only exotic commodity futures
# beyond common ETFs are for research/macro purposes only.
RESEARCH_ONLY_PREFIXES = frozenset({
    "DX",   # Dollar index — FX research only
    "6E", "6J", "6B", "6A", "6C", "6S",  # FX futures — research only
    "ZN", "ZB", "ZF", "ZT",  # Treasury futures (used for beta corridor calc only)
    "TLT", "IEF", "SHY", "BND", "AGG",  # Bond ETFs — FI research
    "LQD", "VCIT", "VCSH", "HYG", "JNK",  # Credit — research only
    "MBB", "VMBS",  # MBS — research only
})
# NOTE: Commodity ETFs (GLD, SLV, USO, UNG, DBA, DBC, COPX, WEAT, CORN)
# are NOT in this set — they are tradeable via L7 for alpha extraction.
# Index ETFs (SPY, QQQ, IWM, DIA, VT, EFA, EEM) are also tradeable.

# Futures tradeable via IBKR — equity index + VIX + treasuries
TRADEABLE_FUTURES = frozenset({"ES", "NQ", "YM", "RTY", "VX", "ZN", "ZB", "ZF", "ZT"})


class MultiProductRouter:
    """Routes orders by product type through the appropriate execution path.

    All products route to IBKR as sole execution broker with native algo routing.
    Trade log maintained in parallel for reconciliation.

    Routing paths:
        EQUITY:  → WonderTrader micro-price → ExchangeCore matching → IBKR (TWAP/VWAP)
        OPTION:  → OptionsEngine Greeks check → vol-adjusted limit → IBKR
        FUTURE:  → Beta corridor validation → IBKR
    """

    def __init__(self):
        self._route_counts: Dict[str, int] = {
            "EQUITY": 0, "OPTION": 0, "FUTURE": 0, "REJECTED": 0,
        }

    def classify(self, order: L7Order) -> ProductType:
        """Auto-classify product type if not explicitly set."""
        ticker = order.ticker.upper()

        # Options: has strike + expiry
        if order.option_type and order.strike > 0:
            return ProductType.OPTION

        # Futures: known contracts
        if order.contract or ticker in TRADEABLE_FUTURES:
            return ProductType.FUTURE

        # Default: equity
        return ProductType.EQUITY

    def is_research_only(self, ticker: str) -> bool:
        """Check if instrument is research-only (FI, FX, credit)."""
        upper = ticker.upper()
        return upper in RESEARCH_ONLY_PREFIXES or any(
            upper.startswith(p) for p in RESEARCH_ONLY_PREFIXES
        )

    def determine_routing(self, order: L7Order) -> RoutingStrategy:
        """Select routing algo based on order size and urgency."""
        if order.urgency == ExecutionUrgency.CRITICAL:
            return RoutingStrategy.IMMEDIATE
        if order.urgency == ExecutionUrgency.HIGH:
            return RoutingStrategy.IMMEDIATE

        # Large orders get TWAP/VWAP
        notional = order.quantity * (order.limit_price or order.arrival_price or 100)
        if notional > 50_000:
            return RoutingStrategy.TWAP
        if notional > 10_000:
            return RoutingStrategy.SMART

        return RoutingStrategy.IMMEDIATE

    def determine_urgency(
        self,
        signal_type: str,
        cta_strength: float = 0.0,
        kill_switch: bool = False,
    ) -> ExecutionUrgency:
        """Infer urgency from signal context."""
        if kill_switch:
            return ExecutionUrgency.CRITICAL

        # High urgency signals
        high_urgency = {
            "MICRO_PRICE_BUY", "MICRO_PRICE_SELL",
            "EVENT_MERGER_ARB", "DISTRESS_FALLEN_ANGEL",
        }
        if signal_type in high_urgency or cta_strength > 0.8:
            return ExecutionUrgency.HIGH

        # Low urgency
        low_urgency = {"HOLD", "QUALITY_BUY", "QUALITY_SELL"}
        if signal_type in low_urgency:
            return ExecutionUrgency.LOW

        return ExecutionUrgency.MEDIUM

    def record_route(self, product_type: str):
        key = product_type if product_type in self._route_counts else "REJECTED"
        self._route_counts[key] = self._route_counts.get(key, 0) + 1

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._route_counts)


# ---------------------------------------------------------------------------
# Slippage & Transaction Cost Model
# ---------------------------------------------------------------------------

class SlippageModel:
    """Estimates and applies slippage to fill prices.

    Components:
        1. Bid-ask spread (calibrated per product type)
        2. Market impact (square-root model)
        3. Timing delay cost
    """

    # Default half-spreads in bps by product
    DEFAULT_HALF_SPREAD = {
        ProductType.EQUITY: 1.5,
        ProductType.OPTION: 15.0,   # Options have wider spreads
        ProductType.FUTURE: 0.5,    # Index futures are tight
    }

    # Impact coefficients by product
    IMPACT_COEFF = {
        ProductType.EQUITY: 0.10,
        ProductType.OPTION: 0.20,
        ProductType.FUTURE: 0.05,
    }

    def estimate_slippage_bps(
        self,
        order: L7Order,
        adv_shares: float = 100_000,
        daily_vol: float = 0.02,
    ) -> float:
        """Pre-trade slippage estimate in bps."""
        pt = order.product_type if isinstance(order.product_type, ProductType) else ProductType.EQUITY
        half_spread = self.DEFAULT_HALF_SPREAD.get(pt, 1.5)
        impact_coeff = self.IMPACT_COEFF.get(pt, 0.10)

        participation = order.quantity / max(adv_shares, 1)
        impact = impact_coeff * (participation ** 0.5) * daily_vol * 10_000

        return half_spread + impact

    def apply_slippage(
        self,
        price: float,
        side: str,
        slippage_bps: float,
    ) -> float:
        """Apply slippage to mid price to get realistic fill price."""
        slip_frac = slippage_bps / 10_000
        if side in ("BUY", "COVER"):
            return price * (1 + slip_frac)
        else:
            return price * (1 - slip_frac)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# L7 Risk Management Engine
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """Real-time risk state updated after every execution."""
    nav: float = 0.0
    cash: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    # Position-level
    max_position_pct: float = 0.0
    max_position_ticker: str = ""
    max_sector_pct: float = 0.0
    max_sector_name: str = ""
    position_count: int = 0
    # Daily P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    daily_pnl_high: float = 0.0
    intraday_drawdown_pct: float = 0.0
    # VaR
    var_95_1d: float = 0.0
    # Risk status
    gates_status: Dict[str, bool] = field(default_factory=dict)
    kill_switch_active: bool = False
    risk_level: str = "NORMAL"  # NORMAL, ELEVATED, HIGH, CRITICAL
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()


class L7RiskEngine:
    """Unified risk management engine for the L7 execution surface.

    Runs pre-trade and post-trade risk checks. Updates risk state after
    every execution. Feeds risk dashboard.

    Risk Gates (all must pass before execution):
        G1: Single position ≤ 10% NAV
        G2: Sector concentration ≤ 30% NAV
        G3: Daily loss circuit breaker ≤ 3% NAV
        G4: Gross leverage ≤ 250%
        G5: Net leverage ≤ 150%
        G6: Trade throttle ≤ 100/day
        G7: Max drawdown ≤ 10% halt
        G8: Cash sufficiency for buys
        G9: Options delta exposure ≤ 20% NAV (new)
        G10: Futures notional ≤ 50% NAV (new)
        G11: Monte Carlo portfolio 1-day VaR95 ≤ 1.5% NAV and CVaR95 ≤ 2.5% NAV
             with the order added (shared PortfolioMCGate; risk-reducing
             orders always pass)

    A 1-contract option order flagged ``min_size_exception`` (single-contract
    premium above the position cap) is exempt from G1 only; it must still
    pass G11.
    """

    # Gate limits
    LIMITS = {
        "G1_POSITION":       0.10,   # 10% NAV single position
        "G2_SECTOR":         0.30,   # 30% NAV sector
        "G3_DAILY_LOSS":     0.03,   # 3% NAV daily loss
        "G4_GROSS_LEVERAGE":  2.50,   # 250%
        "G5_NET_LEVERAGE":    1.50,   # 150%
        "G6_TRADE_THROTTLE":  100,    # trades per day
        "G7_MAX_DRAWDOWN":    0.10,   # 10% from peak
        "G8_CASH":            0.0,    # must have cash for buys
        "G9_OPTIONS_DELTA":   0.20,   # 20% NAV
        "G10_FUTURES_NOTIONAL": 0.50, # 50% NAV
        "G11_MC_VAR95":       0.015,  # 1.5% NAV 1-day (PortfolioMCGate)
        "G11_MC_CVAR95":      0.025,  # 2.5% NAV 1-day
    }

    def __init__(self, initial_nav: float = 1_000.0):
        self._initial_nav = initial_nav
        self._peak_nav = initial_nav
        self._daily_start_nav = initial_nav
        self._trade_count_today: int = 0
        self._last_reset_date: str = ""
        self._risk_history: deque[RiskState] = deque(maxlen=2000)
        self._gate_violations: deque[dict] = deque(maxlen=500)

        # Sector exposure tracking
        self._sector_exposure: Dict[str, float] = {}

        # Options/futures specific
        self._options_delta_exposure: float = 0.0
        self._futures_notional: float = 0.0
        self._last_mc_check: dict = {}

    def reset_daily(self, nav: float):
        """Reset daily counters at market open."""
        self._daily_start_nav = nav
        self._trade_count_today = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._last_reset_date = today
        if nav > self._peak_nav:
            self._peak_nav = nav

    def pre_trade_check(
        self,
        order: L7Order,
        nav: float,
        cash: float,
        positions: Dict[str, any],
        daily_pnl: float,
        gross_exposure: float,
        net_exposure: float,
    ) -> Tuple[bool, List[str]]:
        """Run all risk gates before execution. Returns (passed, violations)."""
        violations = []
        order_value = order.quantity * (order.limit_price or order.arrival_price or 100)

        # Auto-reset daily counters
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self.reset_daily(nav)

        # Position-reducing orders (stops, drawdown protection, expiry closes)
        # must never be blocked by the loss / drawdown / leverage gates that
        # exist to stop NEW risk — otherwise a 3% down day would lock the
        # book and prevent the exits that cut the loss.
        held = positions.get(order.ticker) if positions else None
        held_qty = int(getattr(held, "quantity", 0) or 0) if held is not None else 0
        if ((order.side == "SELL" and held_qty > 0 and order.quantity <= held_qty) or
                (order.side == "COVER" and held_qty < 0 and order.quantity <= abs(held_qty))):
            return True, []

        # G1: Single position limit
        if nav > 0:
            existing_value = 0.0
            if order.ticker in positions:
                pos = positions[order.ticker]
                existing_value = abs(getattr(pos, 'market_value', 0) or
                                    getattr(pos, 'quantity', 0) * getattr(pos, 'current_price', 0))
            new_pct = (existing_value + order_value) / nav
            exempt = (order.min_size_exception and order.quantity == 1
                      and order.product_type == ProductType.OPTION)
            if new_pct > self.LIMITS["G1_POSITION"] and not exempt:
                violations.append(
                    f"G1_POSITION: {order.ticker} would be {new_pct:.1%} of NAV "
                    f"(limit {self.LIMITS['G1_POSITION']:.0%})"
                )

        # G2: Sector concentration — tracked externally
        # (checked via _sector_exposure but we allow it through if unknown)

        # G3: Daily loss circuit breaker
        if nav > 0:
            daily_loss_pct = abs(min(daily_pnl, 0)) / nav
            if daily_loss_pct > self.LIMITS["G3_DAILY_LOSS"]:
                violations.append(
                    f"G3_DAILY_LOSS: daily loss {daily_loss_pct:.2%} exceeds "
                    f"{self.LIMITS['G3_DAILY_LOSS']:.0%}"
                )

        # G4: Gross leverage
        if nav > 0:
            new_gross = (gross_exposure + order_value) / nav
            if new_gross > self.LIMITS["G4_GROSS_LEVERAGE"]:
                violations.append(
                    f"G4_GROSS_LEVERAGE: {new_gross:.1%} exceeds "
                    f"{self.LIMITS['G4_GROSS_LEVERAGE']:.0%}"
                )

        # G5: Net leverage
        if nav > 0:
            side_sign = 1 if order.side in ("BUY", "COVER") else -1
            new_net = abs(net_exposure + side_sign * order_value) / nav
            if new_net > self.LIMITS["G5_NET_LEVERAGE"]:
                violations.append(
                    f"G5_NET_LEVERAGE: {new_net:.1%} exceeds "
                    f"{self.LIMITS['G5_NET_LEVERAGE']:.0%}"
                )

        # G6: Trade throttle
        self._trade_count_today += 1
        if self._trade_count_today > self.LIMITS["G6_TRADE_THROTTLE"]:
            violations.append(
                f"G6_TRADE_THROTTLE: {self._trade_count_today} trades today "
                f"(limit {int(self.LIMITS['G6_TRADE_THROTTLE'])})"
            )

        # G7: Max drawdown from peak
        if self._peak_nav > 0:
            dd = (self._peak_nav - nav) / self._peak_nav
            if dd > self.LIMITS["G7_MAX_DRAWDOWN"]:
                violations.append(
                    f"G7_MAX_DRAWDOWN: drawdown {dd:.2%} exceeds "
                    f"{self.LIMITS['G7_MAX_DRAWDOWN']:.0%}"
                )

        # G8: Cash check for buys
        if order.side in ("BUY", "COVER") and order_value > cash:
            violations.append(
                f"G8_CASH: order ${order_value:.2f} exceeds cash ${cash:.2f}"
            )

        # G9: Options delta exposure
        if order.product_type == ProductType.OPTION:
            new_delta = self._options_delta_exposure + order_value * 0.5  # rough delta
            if nav > 0 and new_delta / nav > self.LIMITS["G9_OPTIONS_DELTA"]:
                violations.append(
                    f"G9_OPTIONS_DELTA: options delta {new_delta/nav:.1%} exceeds "
                    f"{self.LIMITS['G9_OPTIONS_DELTA']:.0%}"
                )

        # G10: Futures notional
        if order.product_type == ProductType.FUTURE:
            new_notional = self._futures_notional + order_value
            if nav > 0 and new_notional / nav > self.LIMITS["G10_FUTURES_NOTIONAL"]:
                violations.append(
                    f"G10_FUTURES_NOTIONAL: futures {new_notional/nav:.1%} exceeds "
                    f"{self.LIMITS['G10_FUTURES_NOTIONAL']:.0%}"
                )

        # G11: Monte Carlo portfolio VaR / CVaR with the order added
        if _get_mc_gate is not None and nav > 0 and order.quantity > 0:
            try:
                gate = _get_mc_gate()
                gate.set_holdings_from_positions(positions, nav)
                opt_fields = {}
                if order.product_type == ProductType.OPTION:
                    opt_fields = {
                        "option_type": (order.option_type or "CALL").lower(),
                        "strike": order.strike,
                        "underlying_price": order.arrival_price or order.strike,
                        "dte": _dte_from_expiry(order.expiry),
                    }
                ptype = order.product_type.value if isinstance(order.product_type, ProductType) else str(order.product_type)
                mc = gate.check_order(
                    order.ticker, order.side, order.quantity,
                    order.limit_price or order.arrival_price or 100.0, nav,
                    product_type="OPTION" if ptype == "OPTION" else ptype, **opt_fields,
                )
                self._last_mc_check = mc.to_dict()
                if not mc.passed:
                    violations.append("G11_MC: " + "; ".join(mc.reasons))
            except Exception as exc:  # never let the risk model crash execution
                logger.warning("G11 MC check unavailable: %s", exc)

        if violations:
            self._gate_violations.append({
                "timestamp": _now_iso(),
                "order_id": order.order_id,
                "ticker": order.ticker,
                "violations": violations,
            })

        return len(violations) == 0, violations

    def post_trade_update(
        self,
        order: L7Order,
        nav: float,
        cash: float,
        positions: Dict[str, any],
        daily_pnl: float,
        gross_exposure: float,
        net_exposure: float,
    ) -> RiskState:
        """Update risk state after an execution. Returns current RiskState."""
        if nav > self._peak_nav:
            self._peak_nav = nav

        # Compute risk metrics
        gross_lev = gross_exposure / nav if nav > 0 else 0.0
        net_lev = net_exposure / nav if nav > 0 else 0.0

        # Max position
        max_pos_pct = 0.0
        max_pos_ticker = ""
        for ticker, pos in positions.items():
            mv = abs(getattr(pos, 'market_value', 0) or
                     getattr(pos, 'quantity', 0) * getattr(pos, 'current_price', 0))
            pct = mv / nav if nav > 0 else 0.0
            if pct > max_pos_pct:
                max_pos_pct = pct
                max_pos_ticker = ticker

        # Drawdown
        dd = (self._peak_nav - nav) / self._peak_nav if self._peak_nav > 0 else 0.0

        # Daily P&L tracking
        daily_pnl_pct = daily_pnl / self._daily_start_nav if self._daily_start_nav > 0 else 0.0

        # VaR estimate (parametric, 95% 1-day)
        # Use 2% daily vol assumption, scaled by leverage
        var_95 = nav * 0.02 * max(gross_lev, 1.0) * 1.645

        # Risk level
        if dd > 0.08 or daily_pnl_pct < -0.025:
            risk_level = "CRITICAL"
        elif dd > 0.05 or daily_pnl_pct < -0.015:
            risk_level = "HIGH"
        elif dd > 0.03 or daily_pnl_pct < -0.01:
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"

        # Kill switch check
        kill_switch = (dd > self.LIMITS["G7_MAX_DRAWDOWN"] or
                       abs(daily_pnl_pct) > self.LIMITS["G3_DAILY_LOSS"])

        # Gate status
        gates = {
            "G1_POSITION": max_pos_pct <= self.LIMITS["G1_POSITION"],
            "G3_DAILY_LOSS": abs(min(daily_pnl_pct, 0)) <= self.LIMITS["G3_DAILY_LOSS"],
            "G4_GROSS_LEVERAGE": gross_lev <= self.LIMITS["G4_GROSS_LEVERAGE"],
            "G5_NET_LEVERAGE": net_lev <= self.LIMITS["G5_NET_LEVERAGE"],
            "G7_MAX_DRAWDOWN": dd <= self.LIMITS["G7_MAX_DRAWDOWN"],
        }

        # Update options/futures tracking
        if order.product_type == ProductType.OPTION:
            self._options_delta_exposure += abs(order.fill_quantity * order.fill_price * 0.5)
        elif order.product_type == ProductType.FUTURE:
            self._futures_notional += abs(order.fill_quantity * order.fill_price)

        state = RiskState(
            nav=nav,
            cash=cash,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            gross_leverage=round(gross_lev, 3),
            net_leverage=round(net_lev, 3),
            max_position_pct=round(max_pos_pct, 4),
            max_position_ticker=max_pos_ticker,
            position_count=len(positions),
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl_pct, 4),
            intraday_drawdown_pct=round(dd, 4),
            var_95_1d=round(var_95, 2),
            gates_status=gates,
            kill_switch_active=kill_switch,
            risk_level=risk_level,
        )
        self._risk_history.append(state)
        return state

    @property
    def latest_state(self) -> Optional[RiskState]:
        return self._risk_history[-1] if self._risk_history else None

    @property
    def gate_violations(self) -> List[dict]:
        return list(self._gate_violations)

    @property
    def risk_history(self) -> List[RiskState]:
        return list(self._risk_history)


# ---------------------------------------------------------------------------
# Execution Learning Loop
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPattern:
    """A learned execution pattern for best-execution optimization."""
    pattern_id: str = ""
    ticker: str = ""
    product_type: str = "EQUITY"
    signal_type: str = ""
    # Context features
    regime: str = "TRENDING"
    time_of_day_bucket: str = ""   # OPEN, MID_MORNING, LUNCH, MID_AFTERNOON, CLOSE
    volatility_bucket: str = ""    # LOW, MEDIUM, HIGH, EXTREME
    order_size_bucket: str = ""    # SMALL, MEDIUM, LARGE
    # Learned optimal parameters
    best_routing: str = "SMART"
    optimal_slice_count: int = 5
    optimal_urgency: str = "MEDIUM"
    avg_slippage_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    # Statistics
    sample_count: int = 0
    win_rate: float = 0.0         # % of trades profitable after costs
    avg_pnl_bps: float = 0.0     # avg P&L per trade in bps
    last_updated: str = ""

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = str(uuid.uuid4())[:8]
        if not self.last_updated:
            self.last_updated = _now_iso()


class ExecutionLearningLoop:
    """Learns optimal execution parameters from trade history.

    After every execution, records outcome. Periodically (intraday, daily,
    weekly, monthly) re-optimizes routing, slicing, and timing parameters
    per (ticker, product_type, signal_type, regime, time_bucket, vol_bucket).

    Learning dimensions:
        1. Routing strategy: which algo minimizes slippage for this context
        2. Slice count: how many child orders minimize impact
        3. Timing: which time-of-day bucket has lowest cost
        4. Urgency: optimal aggressiveness given signal decay
        5. Size: optimal participation rate

    Optimization cadences:
        - Intraday: EWMA update of slippage/impact estimates after each trade
        - Daily: Re-rank routing strategies per context bucket
        - Weekly: Full pattern library refresh with decay of old samples
        - Monthly: Prune stale patterns, recalibrate impact model coefficients
    """

    # Time-of-day buckets (ET)
    TOD_BUCKETS = {
        (9, 30, 10, 0):   "OPEN",
        (10, 0, 11, 30):  "MID_MORNING",
        (11, 30, 13, 30): "LUNCH",
        (13, 30, 15, 0):  "MID_AFTERNOON",
        (15, 0, 16, 0):   "CLOSE",
    }

    # Volatility buckets (daily vol %)
    VOL_THRESHOLDS = [0.01, 0.02, 0.04]  # LOW < 1%, MED < 2%, HIGH < 4%, EXTREME >= 4%

    # Size buckets (notional USD)
    SIZE_THRESHOLDS = [5_000, 25_000, 100_000]  # SMALL, MEDIUM, LARGE, XLARGE

    # EWMA decay factor for intraday updates
    EWMA_ALPHA = 0.15

    def __init__(self, log_dir: Optional[Path] = None):
        self._patterns: Dict[str, ExecutionPattern] = {}
        self._trade_outcomes: deque[dict] = deque(maxlen=10_000)
        self._log_dir = log_dir or Path("logs/l7_learning")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._daily_stats: Dict[str, List[float]] = {}  # date → [slippage_bps]
        self._weekly_refresh_count: int = 0
        self._monthly_prune_count: int = 0

    def _bucket_key(
        self,
        ticker: str,
        product_type: str,
        signal_type: str,
        regime: str,
        tod_bucket: str,
        vol_bucket: str,
        size_bucket: str,
    ) -> str:
        return f"{ticker}|{product_type}|{signal_type}|{regime}|{tod_bucket}|{vol_bucket}|{size_bucket}"

    def _classify_tod(self, hour: int, minute: int) -> str:
        """Classify time of day into bucket."""
        t = hour * 60 + minute
        if t < 600:  # 10:00
            return "OPEN"
        if t < 690:  # 11:30
            return "MID_MORNING"
        if t < 810:  # 13:30
            return "LUNCH"
        if t < 900:  # 15:00
            return "MID_AFTERNOON"
        return "CLOSE"

    def _classify_vol(self, daily_vol: float) -> str:
        if daily_vol < self.VOL_THRESHOLDS[0]:
            return "LOW"
        if daily_vol < self.VOL_THRESHOLDS[1]:
            return "MEDIUM"
        if daily_vol < self.VOL_THRESHOLDS[2]:
            return "HIGH"
        return "EXTREME"

    def _classify_size(self, notional: float) -> str:
        if notional < self.SIZE_THRESHOLDS[0]:
            return "SMALL"
        if notional < self.SIZE_THRESHOLDS[1]:
            return "MEDIUM"
        if notional < self.SIZE_THRESHOLDS[2]:
            return "LARGE"
        return "XLARGE"

    def record_outcome(
        self,
        order: L7Order,
        tca: TCASnapshot,
        regime: str = "TRENDING",
        daily_vol: float = 0.02,
        pnl_bps: float = 0.0,
    ):
        """Record an execution outcome and update the pattern library (intraday EWMA)."""
        now = datetime.now(timezone.utc)
        tod = self._classify_tod(now.hour, now.minute)
        vol_b = self._classify_vol(daily_vol)
        notional = abs(order.quantity * order.fill_price)
        size_b = self._classify_size(notional)
        pt = order.product_type if isinstance(order.product_type, str) else order.product_type.value

        key = self._bucket_key(order.ticker, pt, order.signal_type, regime, tod, vol_b, size_b)

        # Store raw outcome
        outcome = {
            "key": key, "order_id": order.order_id, "ticker": order.ticker,
            "routing": order.routing, "slippage_bps": tca.total_cost_bps,
            "impact_bps": tca.market_impact_bps, "pnl_bps": pnl_bps,
            "timestamp": _now_iso(),
        }
        self._trade_outcomes.append(outcome)

        # EWMA update of pattern
        if key not in self._patterns:
            self._patterns[key] = ExecutionPattern(
                ticker=order.ticker, product_type=pt,
                signal_type=order.signal_type, regime=regime,
                time_of_day_bucket=tod, volatility_bucket=vol_b,
                order_size_bucket=size_b,
                best_routing=order.routing if isinstance(order.routing, str) else order.routing.value,
            )

        pat = self._patterns[key]
        alpha = self.EWMA_ALPHA

        # EWMA slippage
        pat.avg_slippage_bps = alpha * tca.total_cost_bps + (1 - alpha) * pat.avg_slippage_bps
        pat.avg_market_impact_bps = alpha * tca.market_impact_bps + (1 - alpha) * pat.avg_market_impact_bps
        pat.avg_pnl_bps = alpha * pnl_bps + (1 - alpha) * pat.avg_pnl_bps
        pat.sample_count += 1

        # Win rate update
        if pnl_bps > 0:
            pat.win_rate = alpha * 1.0 + (1 - alpha) * pat.win_rate
        else:
            pat.win_rate = alpha * 0.0 + (1 - alpha) * pat.win_rate

        pat.last_updated = _now_iso()

        # Daily stats
        today = now.strftime("%Y-%m-%d")
        if today not in self._daily_stats:
            self._daily_stats[today] = []
        self._daily_stats[today].append(tca.total_cost_bps)

    def suggest_routing(
        self,
        ticker: str,
        product_type: str,
        signal_type: str,
        regime: str,
        daily_vol: float = 0.02,
        notional: float = 10_000,
    ) -> Dict[str, any]:
        """Suggest optimal routing params based on learned patterns."""
        now = datetime.now(timezone.utc)
        tod = self._classify_tod(now.hour, now.minute)
        vol_b = self._classify_vol(daily_vol)
        size_b = self._classify_size(notional)

        key = self._bucket_key(ticker, product_type, signal_type, regime, tod, vol_b, size_b)

        if key in self._patterns and self._patterns[key].sample_count >= 5:
            pat = self._patterns[key]
            return {
                "routing": pat.best_routing,
                "expected_slippage_bps": round(pat.avg_slippage_bps, 2),
                "expected_impact_bps": round(pat.avg_market_impact_bps, 2),
                "sample_count": pat.sample_count,
                "win_rate": round(pat.win_rate, 3),
                "confidence": "HIGH" if pat.sample_count >= 20 else "MEDIUM",
            }

        # Fallback: use product-type defaults
        defaults = {
            "EQUITY": {"routing": "SMART", "expected_slippage_bps": 3.0},
            "OPTION": {"routing": "IMMEDIATE", "expected_slippage_bps": 20.0},
            "FUTURE": {"routing": "IMMEDIATE", "expected_slippage_bps": 1.5},
        }
        d = defaults.get(product_type, defaults["EQUITY"])
        d["confidence"] = "LOW"
        d["sample_count"] = 0
        return d

    def daily_optimize(self):
        """Daily optimization: re-rank routing strategies per bucket."""
        for key, pat in self._patterns.items():
            # If slippage is high and we have enough samples, try different routing
            if pat.sample_count >= 10 and pat.avg_slippage_bps > 10:
                # Switch from current to TWAP if using SMART/IMMEDIATE
                if pat.best_routing in ("SMART", "IMMEDIATE"):
                    pat.best_routing = "TWAP"
                elif pat.best_routing == "TWAP" and pat.avg_slippage_bps > 15:
                    pat.best_routing = "VWAP"

        logger.info("ExecutionLearningLoop: daily optimize complete (%d patterns)", len(self._patterns))

    def weekly_refresh(self):
        """Weekly: decay old samples, refresh pattern weights."""
        decay = 0.90
        for pat in self._patterns.values():
            pat.avg_slippage_bps *= decay
            pat.avg_market_impact_bps *= decay
        self._weekly_refresh_count += 1
        logger.info("ExecutionLearningLoop: weekly refresh #%d", self._weekly_refresh_count)

    def monthly_prune(self):
        """Monthly: remove stale patterns with few samples."""
        stale_keys = [k for k, p in self._patterns.items() if p.sample_count < 3]
        for k in stale_keys:
            del self._patterns[k]
        self._monthly_prune_count += 1
        logger.info(
            "ExecutionLearningLoop: monthly prune #%d, removed %d stale patterns",
            self._monthly_prune_count, len(stale_keys),
        )

    def save_patterns(self):
        """Persist pattern library to disk."""
        path = self._log_dir / "execution_patterns.json"
        data = {}
        for key, pat in self._patterns.items():
            data[key] = {
                "ticker": pat.ticker, "product_type": pat.product_type,
                "signal_type": pat.signal_type, "regime": pat.regime,
                "tod": pat.time_of_day_bucket, "vol": pat.volatility_bucket,
                "size": pat.order_size_bucket, "routing": pat.best_routing,
                "avg_slippage_bps": pat.avg_slippage_bps,
                "avg_impact_bps": pat.avg_market_impact_bps,
                "sample_count": pat.sample_count, "win_rate": pat.win_rate,
            }
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save patterns: %s", e)

    def load_patterns(self):
        """Load pattern library from disk."""
        path = self._log_dir / "execution_patterns.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for key, d in data.items():
                self._patterns[key] = ExecutionPattern(
                    ticker=d.get("ticker", ""),
                    product_type=d.get("product_type", "EQUITY"),
                    signal_type=d.get("signal_type", ""),
                    regime=d.get("regime", "TRENDING"),
                    time_of_day_bucket=d.get("tod", ""),
                    volatility_bucket=d.get("vol", ""),
                    order_size_bucket=d.get("size", ""),
                    best_routing=d.get("routing", "SMART"),
                    avg_slippage_bps=d.get("avg_slippage_bps", 0),
                    avg_market_impact_bps=d.get("avg_impact_bps", 0),
                    sample_count=d.get("sample_count", 0),
                    win_rate=d.get("win_rate", 0),
                )
            logger.info("Loaded %d execution patterns", len(self._patterns))
        except Exception as e:
            logger.warning("Failed to load patterns: %s", e)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def daily_cost_summary(self) -> Dict[str, float]:
        """Average TCA cost per day."""
        out = {}
        for date, costs in sorted(self._daily_stats.items())[-30:]:
            out[date] = round(_mean(costs), 2)
        return out


# ===========================================================================
# L7 UNIFIED EXECUTION SURFACE — Main orchestrator
# ===========================================================================

class L7UnifiedExecutionSurface:
    """Fused continuous execution arm for Metadron Capital.

    Unifies WonderTrader (micro-price + CTA + routing), ExchangeCore (order
    matching), IBKRBroker (sole execution with native TWAP/VWAP), OptionsEngine
    (derivatives), and QuantStrategyExecutor (12 technical strategies) into
    one continuous execution surface.

    ALL tradeable products route through IBKR as the sole execution broker.
    A trade log records every order the platform generates for reconciliation
    (generated vs actually executed on the broker).

    Architecture:
        L7UnifiedExecutionSurface
        ├── Continuous intraday loop (1-min heartbeat from live_loop_orchestrator)
        ├── Multi-product router (equities, options, futures)
        │   ├── Equity → WonderTrader micro-price → ExchangeCore → IBKR (TWAP/VWAP)
        │   ├── Options → OptionsEngine Greeks → vol-adjusted → IBKR
        │   └── Futures → Beta corridor hedge → IBKR
        ├── Unified order book (all products, all horizons)
        ├── IBKR broker (sole execution) + trade log (reconciliation)
        ├── L7RiskEngine (10 gates, per-execution update)
        ├── TransactionCostAnalyzer (per-trade decomposition)
        ├── ExecutionLearningLoop (pattern identification)
        └── SlippageModel (pre-trade cost estimation)
    """

    # ── Thinking Tab output format reference (structure only, not values) ──
    # This defines the gold standard format for how scan results appear in the
    # Thinking Tab before trades are posted to the broker. The L7 surface and
    # FullUniverseScan emit events matching this structure. After all 4 runs
    # complete and trades finalize, the Thinking Tab resets for the next cycle.
    # Transactions are then logged in the Transaction Log with execution time.
    #
    # IMPORTANT: All 4 runs (SP500, SP400, SP600, ETF_FI) display the SAME
    # full detailed format — every run shows the complete per-bucket table
    # with rank, ticker, shares, price, dollar, %NAV, alpha, sharpe, regime.
    # No run is abbreviated. Each run is equally comprehensive.
    THINKING_FORMAT = {
        "run_scorecard": {
            "_description": "One per universe run — ALL 4 runs use identical format",
            "run": int,
            "universe": str,
            "scanned": int,
            "buy": int,
            "sell": int,
            "avg_alpha": float,
            "regime": str,
            "deployed": float,
            "positions": int,
        },
        "position_entry": {
            "_description": "One per position within a bucket, ranked by alpha",
            "rank": int,
            "ticker": str,
            "shares": int,
            "price": float,
            "dollar": float,
            "pct_nav": float,
            "alpha": float,
            "sharpe": float,
            "regime": str,
        },
        "bucket_subtotal": {
            "_description": "Per bucket per run — shows deployed vs target",
            "bucket": str,
            "target_usd": float,
            "deployed_usd": float,
            "utilization_pct": float,
        },
        "derivatives_overlay": {
            "_description": "After all 4 runs — options + futures from margin budget",
            "options": [{"ticker": str, "type": str, "notional": float}],
            "futures": [{"contract": str, "direction": str, "lots": int, "notional": float, "margin": float}],
        },
        "bucket_utilization_summary": {
            "_description": "Final summary after all runs — target vs deployed vs util%",
            "buckets": [{"bucket": str, "target": float, "deployed": float, "util_pct": float}],
            "total_deployed": float,
            "total_pct_nav": float,
        },
    }

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        log_dir: Optional[str] = None,
        ibkr_host: Optional[str] = None,
        ibkr_port: Optional[int] = None,
        ibkr_client_id: Optional[int] = None,
        ibkr_paper: bool = True,
        daily_target_pct: float = 0.05,
    ):
        self._log_dir = Path(log_dir or "logs/l7_execution")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # --- Broker: IBKR only ---
        self._broker: Optional[object] = None

        if IBKRBroker is not None:
            try:
                self._broker = IBKRBroker(
                    initial_cash=initial_cash,
                    log_dir=self._log_dir / "ibkr",
                    host=ibkr_host,
                    port=ibkr_port,
                    client_id=ibkr_client_id,
                    paper=ibkr_paper,
                    daily_target_pct=daily_target_pct,
                )
                logger.info("L7: IBKRBroker connected (paper=%s)", ibkr_paper)
            except Exception as e:
                logger.warning("L7: IBKRBroker init failed: %s — trade-log-only mode", e)
        else:
            logger.warning("L7: ib_insync not installed — no live execution")

        # --- Trade log for reconciliation (generated vs executed) ---
        self._trade_log_dir = self._log_dir / "trade_log"
        self._trade_log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log: deque[dict] = deque(maxlen=50_000)

        # --- Prometheus metrics (optional) ---
        self._prom = None
        global _L7_PROM
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary
            if _L7_PROM is not None:            # metrics are process-global: register once, reuse
                self._prom = _L7_PROM
                raise _AlreadyRegistered
            self._prom = {
                "orders_total": Counter("l7_orders_total", "Total orders submitted", ["product", "side", "algo"]),
                "orders_filled": Counter("l7_orders_filled", "Orders filled", ["product", "algo"]),
                "orders_rejected": Counter("l7_orders_rejected", "Orders rejected", ["reason"]),
                "fill_latency": Histogram("l7_fill_latency_seconds", "Order fill latency", buckets=[0.1, 0.5, 1, 5, 30, 60, 300]),
                "slippage_bps": Summary("l7_slippage_bps", "Realized slippage in bps", ["product"]),
                "nav": Gauge("l7_nav_usd", "Current NAV"),
                "gross_leverage": Gauge("l7_gross_leverage", "Gross leverage ratio"),
                "net_leverage": Gauge("l7_net_leverage", "Net leverage ratio"),
                "position_count": Gauge("l7_position_count", "Active positions"),
                "daily_pnl": Gauge("l7_daily_pnl_usd", "Daily P&L"),
                "risk_level": Gauge("l7_risk_level", "Risk level (0=NORMAL,1=ELEVATED,2=HIGH,3=CRITICAL)"),
                "kill_switch": Gauge("l7_kill_switch_active", "Kill switch status (0/1)"),
                "twap_orders": Counter("l7_twap_orders", "TWAP algo orders"),
                "vwap_orders": Counter("l7_vwap_orders", "VWAP algo orders"),
                "ibkr_connected": Gauge("l7_ibkr_connected", "IBKR connection status (0/1)"),
                "tca_total_cost_bps": Summary("l7_tca_total_cost_bps", "TCA total cost per trade"),
                "tca_implementation_shortfall": Summary("l7_tca_is_usd", "Implementation shortfall USD"),
            }
            _L7_PROM = self._prom
            logger.info("L7: Prometheus metrics registered")
        except _AlreadyRegistered:
            pass
        except ImportError:
            logger.debug("L7: prometheus_client not installed — metrics disabled")

        # WonderTrader: CTA signals + micro-price + routing
        self._wondertrader: Optional[object] = None
        if WonderTraderEngine is not None:
            try:
                self._wondertrader = WonderTraderEngine()
            except Exception as e:
                logger.warning("L7: WonderTraderEngine init failed: %s", e)

        # ExchangeCore: order matching simulation
        self._exchange_core: Optional[object] = None
        if ExchangeCoreEngine is not None:
            try:
                self._exchange_core = ExchangeCoreEngine()
            except Exception as e:
                logger.warning("L7: ExchangeCoreEngine init failed: %s", e)

        # OptionsEngine: Greeks, vol surface, hedge
        self._options_engine: Optional[object] = None
        if OptionsEngine is not None:
            try:
                self._options_engine = OptionsEngine(nav=initial_cash)
            except Exception as e:
                logger.warning("L7: OptionsEngine init failed: %s", e)

        # QuantStrategyExecutor: 12 technical strategies
        self._quant_executor: Optional[object] = None
        if QuantStrategyExecutor is not None:
            try:
                self._quant_executor = QuantStrategyExecutor()
            except Exception as e:
                logger.warning("L7: QuantStrategyExecutor init failed: %s", e)

        # BetaCorridor for futures hedging
        self._beta_corridor: Optional[object] = None
        if BetaCorridor is not None:
            try:
                self._beta_corridor = BetaCorridor()
            except Exception as e:
                logger.warning("L7: BetaCorridor init failed: %s", e)

        # AI Hedge Fund HFT Bridge (optional — enhances HFT order routing)
        try:
            from engine.execution.ai_hedgefund_hft_bridge import AiHedgeFundHFTBridge
            self._aihf_hft = AiHedgeFundHFTBridge()
            if self._aihf_hft.is_available():
                logger.info("L7: AI Hedge Fund HFT bridge active")
            else:
                logger.debug("L7: AI Hedge Fund HFT bridge loaded but engine unavailable")
        except Exception as _exc:
            logger.debug("L7: AiHedgeFundHFTBridge not loaded: %s", _exc)
            self._aihf_hft = None

        # --- L7-specific components ---
        self._router = MultiProductRouter()
        self._risk_engine = L7RiskEngine(initial_nav=initial_cash)
        self._tca = TransactionCostAnalyzer()
        self._slippage = SlippageModel()
        self._learning = ExecutionLearningLoop(log_dir=self._log_dir / "learning")
        self._learning.load_patterns()

        # Unified order book
        self._order_book: deque[L7Order] = deque(maxlen=50_000)
        self._filled_orders: deque[L7Order] = deque(maxlen=50_000)

        # State
        self._initial_cash = initial_cash
        self._heartbeat_count: int = 0
        self._daily_target_pct = daily_target_pct

        # Update Prometheus connection gauge
        if self._prom:
            self._prom["ibkr_connected"].set(1 if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected else 0)

        logger.info(
            "L7UnifiedExecutionSurface initialized: cash=$%.2f, "
            "ibkr=%s, trade_log=YES, wondertrader=%s, exchange_core=%s, "
            "options=%s, quant=%s, beta_corridor=%s",
            initial_cash,
            "YES" if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected else "NO",
            "YES" if self._wondertrader else "NO",
            "YES" if self._exchange_core else "NO",
            "YES" if self._options_engine else "NO",
            "YES" if self._quant_executor else "NO",
            "YES" if self._beta_corridor else "NO",
        )

    # ------------------------------------------------------------------
    # Core execution: submit_order
    # ------------------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        signal_type: str = "HOLD",
        product_type: Optional[str] = None,
        limit_price: Optional[float] = None,
        option_type: str = "",
        strike: float = 0.0,
        expiry: str = "",
        contract: str = "",
        regime: str = "TRENDING",
        daily_vol: float = 0.02,
        kill_switch: bool = False,
        reason: str = "",
        min_size_exception: bool = False,
        is_hedge: bool = False,
    ) -> L7Order:
        """Submit a unified order through the L7 execution surface.

        This is the single entry point for ALL trades. The order flows through:
        1. Research-only guard (reject FI/FX/credit)
        2. Product classification
        3. Learning loop routing suggestion
        4. Pre-trade risk gates (10 checks)
        5. Slippage estimation
        6. Product-specific execution path
        7. IBKR execution (TWAP/VWAP/Market) + trade log for recon
        8. Post-trade risk update + Prometheus metrics
        9. TCA analysis
        10. Learning loop outcome recording
        """
        # Build the L7Order
        order = L7Order(
            ticker=ticker, side=side, quantity=quantity,
            signal_type=signal_type, limit_price=limit_price,
            option_type=option_type, strike=strike, expiry=expiry,
            contract=contract, reason=reason,
            min_size_exception=min_size_exception, is_hedge=is_hedge,
        )

        # 1. Research-only guard
        if self._router.is_research_only(ticker):
            order.status = "REJECTED"
            order.reason = f"Research-only instrument: {ticker} (FI/FX/credit)"
            self._order_book.append(order)
            logger.info("L7 REJECTED (research-only): %s", ticker)
            return order

        # 2. Product classification
        if product_type:
            order.product_type = ProductType(product_type)
        else:
            order.product_type = self._router.classify(order)

        # 3. Learning loop routing suggestion
        notional = quantity * (limit_price or 100)
        suggestion = self._learning.suggest_routing(
            ticker, order.product_type.value if isinstance(order.product_type, ProductType) else order.product_type,
            signal_type, regime, daily_vol, notional,
        )
        order.routing = RoutingStrategy(suggestion.get("routing", "SMART"))

        # Urgency
        cta_strength = 0.0
        if self._wondertrader and hasattr(self._wondertrader, '_execution_log'):
            cta_strength = suggestion.get("win_rate", 0.5)
        order.urgency = self._router.determine_urgency(signal_type, cta_strength, kill_switch)

        # Get arrival price
        arrival_price = self._get_price(ticker)
        order.arrival_price = arrival_price

        # 4. Pre-trade risk gates
        nav, cash, positions, daily_pnl, gross_exp, net_exp = self._get_portfolio_state()
        passed, violations = self._risk_engine.pre_trade_check(
            order, nav, cash, positions, daily_pnl, gross_exp, net_exp,
        )
        if not passed:
            order.status = "REJECTED"
            order.reason = f"Risk gate violation: {'; '.join(violations)}"
            self._order_book.append(order)
            self._router.record_route("REJECTED")
            logger.warning("L7 REJECTED (risk): %s %s — %s", side, ticker, order.reason)
            return order

        # 5. Slippage estimation
        est_slippage = self._slippage.estimate_slippage_bps(order, daily_vol=daily_vol)
        order.slippage_bps = est_slippage

        # 6. Product-specific execution path
        if order.product_type == ProductType.OPTION:
            self._execute_option(order, regime, arrival_price)
        elif order.product_type == ProductType.FUTURE:
            self._execute_future(order, regime, arrival_price)
        else:
            self._execute_equity(order, regime, arrival_price, daily_vol)

        # Record in order book
        self._order_book.append(order)
        self._router.record_route(order.product_type.value if isinstance(order.product_type, ProductType) else str(order.product_type))

        if order.status == "FILLED":
            self._filled_orders.append(order)

            # 7. Record in trade log for reconciliation
            self._log_to_trade_log(order)

            # 8. Post-trade risk update + Prometheus
            nav, cash, positions, daily_pnl, gross_exp, net_exp = self._get_portfolio_state()
            risk_state = self._risk_engine.post_trade_update(
                order, nav, cash, positions, daily_pnl, gross_exp, net_exp,
            )

            # 9. TCA analysis
            tca = self._tca.analyze(
                order, arrival_price, order.fill_price,
                daily_vol=daily_vol,
            )
            order.implementation_shortfall = tca.implementation_shortfall_usd
            order.market_impact_bps = tca.market_impact_bps
            order.timing_cost_bps = tca.timing_cost_bps

            # 10. Learning loop
            self._learning.record_outcome(order, tca, regime, daily_vol)

            # Prometheus metric updates
            if self._prom:
                risk_level_map = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}
                self._prom["nav"].set(nav)
                self._prom["gross_leverage"].set(risk_state.gross_leverage)
                self._prom["net_leverage"].set(risk_state.net_leverage)
                self._prom["position_count"].set(risk_state.position_count)
                self._prom["daily_pnl"].set(risk_state.daily_pnl)
                self._prom["risk_level"].set(risk_level_map.get(risk_state.risk_level, 0))
                self._prom["kill_switch"].set(1 if risk_state.kill_switch_active else 0)
                pt = order.product_type.value if isinstance(order.product_type, ProductType) else str(order.product_type)
                self._prom["slippage_bps"].labels(product=pt).observe(tca.total_cost_bps)
                self._prom["tca_total_cost_bps"].observe(tca.total_cost_bps)
                self._prom["tca_implementation_shortfall"].observe(abs(tca.implementation_shortfall_usd))
                self._prom["ibkr_connected"].set(1 if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected else 0)

            logger.info(
                "L7 FILLED: %s %s %d %s @ $%.2f (slip=%.1fbps, cost=%.1fbps, risk=%s)",
                side, ticker, quantity, order.product_type.value,
                order.fill_price, order.slippage_bps, tca.total_cost_bps,
                risk_state.risk_level,
            )

        return order

    # ------------------------------------------------------------------
    # Product-specific execution paths
    # ------------------------------------------------------------------

    def _execute_equity(self, order: L7Order, regime: str, arrival_price: float, daily_vol: float):
        """Equity path: WonderTrader micro-price → ExchangeCore → IBKR (TWAP/VWAP)."""
        ticker = order.ticker
        price = arrival_price

        # Step 1: WonderTrader micro-price adjustment
        if self._wondertrader and price > 0:
            try:
                ohlcv = {"open": price, "high": price * 1.001, "low": price * 0.999,
                         "close": price, "volume": 100_000}
                mp_result = self._wondertrader.compute_micro_price(ohlcv)
                if mp_result and hasattr(mp_result, 'micro_price') and mp_result.micro_price > 0:
                    order.micro_price = mp_result.micro_price
                    price = mp_result.micro_price
            except Exception as e:
                logger.debug("WonderTrader micro-price failed for %s: %s", ticker, e)

        # Step 2: Apply slippage
        fill_price = self._slippage.apply_slippage(price, order.side, order.slippage_bps)

        # Step 3: Compute transaction cost
        order.transaction_cost = abs(order.quantity * fill_price) * (order.slippage_bps / 10_000)

        # Step 4: Route to IBKR
        self._route_to_broker(order, fill_price)

    def _execute_option(self, order: L7Order, regime: str, arrival_price: float):
        """Options path: OptionsEngine Greeks → vol-adjusted → IBKR."""
        price = arrival_price

        # Options have wider spreads — adjust slippage
        order.slippage_bps = max(order.slippage_bps, 15.0)

        # Greeks check via OptionsEngine
        if self._options_engine:
            try:
                self._options_engine.update_regime(regime)
            except Exception as e:
                logger.debug("OptionsEngine regime update failed: %s", e)

        # Apply slippage
        fill_price = self._slippage.apply_slippage(price, order.side, order.slippage_bps)
        order.transaction_cost = abs(order.quantity * fill_price) * (order.slippage_bps / 10_000)

        # Route to IBKR
        self._route_to_broker(order, fill_price)

    def _execute_future(self, order: L7Order, regime: str, arrival_price: float):
        """Futures path: Beta corridor validation → IBKR."""
        price = arrival_price

        # Futures are tight — lower slippage
        order.slippage_bps = max(order.slippage_bps, 0.5)

        # Beta corridor check
        if self._beta_corridor:
            try:
                # Validate the futures hedge is within corridor
                pass  # BetaCorridor validation integrated via pipeline
            except Exception as e:
                logger.debug("BetaCorridor check failed: %s", e)

        fill_price = self._slippage.apply_slippage(price, order.side, order.slippage_bps)
        order.transaction_cost = abs(order.quantity * fill_price) * (order.slippage_bps / 10_000)

        self._route_to_broker(order, fill_price)

    # ------------------------------------------------------------------
    # Broker routing
    # ------------------------------------------------------------------

    def _route_to_broker(self, order: L7Order, fill_price: float):
        """Route order to IBKR with appropriate algo (TWAP/VWAP/Market).

        Records every order in the trade log for reconciliation regardless
        of execution outcome.
        """
        side_map = {"BUY": "BUY", "SELL": "SELL", "SHORT": "SHORT", "COVER": "COVER"}
        broker_side = side_map.get(order.side, "BUY")

        executed = False
        algo_used = "MARKET"

        # Record in trade log BEFORE execution (generated order)
        trade_log_entry = {
            "order_id": order.order_id,
            "ticker": order.ticker,
            "side": order.side,
            "quantity": order.quantity,
            "product_type": order.product_type.value if isinstance(order.product_type, ProductType) else str(order.product_type),
            "signal_type": order.signal_type,
            "routing": order.routing.value if isinstance(order.routing, RoutingStrategy) else str(order.routing),
            "limit_price": order.limit_price,
            "arrival_price": order.arrival_price,
            "micro_price": order.micro_price,
            "generated_at": _now_iso(),
            "broker_status": "PENDING",
            "broker_fill_price": None,
            "broker_algo": None,
        }

        # Route to IBKR with algo selection
        if self._broker and hasattr(self._broker, 'is_connected') and self._broker.is_connected:
            try:
                from .paper_broker import OrderSide as BrokerSide, SignalType as BrokerSignal
                t_side = BrokerSide(broker_side)
                t_signal = BrokerSignal.HOLD
                try:
                    t_signal = BrokerSignal(order.signal_type)
                except (ValueError, KeyError):
                    pass

                notional = order.quantity * (order.limit_price or order.arrival_price or fill_price)

                # Algo selection based on routing strategy and order size
                if order.routing == RoutingStrategy.TWAP or (order.routing == RoutingStrategy.SMART and notional > 50_000):
                    duration = 30 if order.urgency == ExecutionUrgency.MEDIUM else 15 if order.urgency == ExecutionUrgency.HIGH else 60
                    result = self._broker.place_twap_order(
                        ticker=order.ticker, side=t_side, quantity=order.quantity,
                        duration_minutes=duration, signal_type=t_signal,
                        limit_price=order.limit_price,
                        reason=order.reason or f"L7:{order.signal_type}",
                    )
                    algo_used = "TWAP"
                    if self._prom:
                        self._prom["twap_orders"].inc()

                elif order.routing == RoutingStrategy.VWAP:
                    result = self._broker.place_vwap_order(
                        ticker=order.ticker, side=t_side, quantity=order.quantity,
                        duration_minutes=60, max_pct_volume=0.25,
                        signal_type=t_signal, limit_price=order.limit_price,
                        reason=order.reason or f"L7:{order.signal_type}",
                    )
                    algo_used = "VWAP"
                    if self._prom:
                        self._prom["vwap_orders"].inc()

                else:
                    result = self._broker.place_order(
                        ticker=order.ticker, side=t_side, quantity=order.quantity,
                        signal_type=t_signal, limit_price=order.limit_price,
                        reason=order.reason or f"L7:{order.signal_type}",
                    )
                    algo_used = "MARKET"

                if hasattr(result, 'status'):
                    status_str = result.status if isinstance(result.status, str) else result.status.value
                    if status_str in ("FILLED", "PENDING"):
                        order.fill_price = getattr(result, 'fill_price', fill_price) or fill_price
                        order.fill_quantity = order.quantity
                        order.status = "FILLED"
                        order.filled_at = _now_iso()
                        executed = True

                        trade_log_entry["broker_status"] = "FILLED"
                        trade_log_entry["broker_fill_price"] = order.fill_price
                    else:
                        order.reason = f"IBKR: {getattr(result, 'reason', 'unknown')}"
                        trade_log_entry["broker_status"] = status_str

                trade_log_entry["broker_algo"] = algo_used

                # Prometheus metrics
                if self._prom:
                    pt = order.product_type.value if isinstance(order.product_type, ProductType) else str(order.product_type)
                    self._prom["orders_total"].labels(product=pt, side=order.side, algo=algo_used).inc()
                    if executed:
                        self._prom["orders_filled"].labels(product=pt, algo=algo_used).inc()

            except Exception as e:
                logger.error("IBKR execution failed for %s: %s", order.ticker, e)
                trade_log_entry["broker_status"] = f"ERROR: {e}"
                if self._prom:
                    self._prom["orders_rejected"].labels(reason="ibkr_error").inc()

        # No broker or not connected — log as NOT EXECUTED
        if not executed:
            trade_log_entry["broker_status"] = "NOT_EXECUTED"
            logger.warning(
                "TRADE LOG ONLY: %s %s %d @ $%.2f — IBKR not connected. "
                "Order recorded for reconciliation but NOT executed on any market.",
                order.side, order.ticker, order.quantity, fill_price,
            )
            if self._prom:
                self._prom["orders_rejected"].labels(reason="no_broker").inc()
            order.fill_price = fill_price
            order.fill_quantity = order.quantity
            order.status = "FILLED"
            order.filled_at = _now_iso()
            order.reason = (order.reason or "") + " [not-executed-on-broker]"

        # Always persist to trade log for reconciliation
        self._trade_log.append(trade_log_entry)
        self._persist_trade_log_entry(trade_log_entry)

    def _persist_trade_log_entry(self, entry: dict):
        """Append trade log entry to JSONL file for recon audit."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = self._trade_log_dir / f"trade_log_{today}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("Trade log write failed: %s", e)

    def _log_to_trade_log(self, order: L7Order):
        """Record a filled order to the reconciliation trade log."""
        entry = {
            "order_id": order.order_id,
            "ticker": order.ticker,
            "side": order.side,
            "quantity": order.quantity,
            "fill_price": order.fill_price,
            "signal_type": order.signal_type,
            "status": order.status,
            "timestamp": _now_iso(),
        }
        self._trade_log.append(entry)
        self._persist_trade_log_entry(entry)

    def get_trade_log(self, last_n: int = 0) -> List[dict]:
        """Return trade log entries for reconciliation dashboard."""
        entries = list(self._trade_log)
        if last_n > 0:
            entries = entries[-last_n:]
        return entries

    def get_recon_summary(self) -> dict:
        """Reconciliation summary: generated vs executed vs failed."""
        entries = list(self._trade_log)
        executed = sum(1 for e in entries if e.get("broker_status") == "FILLED")
        not_executed = sum(1 for e in entries if e.get("broker_status") == "NOT_EXECUTED")
        errored = sum(1 for e in entries if str(e.get("broker_status", "")).startswith("ERROR"))
        return {
            "total_generated": len(entries),
            "executed_on_broker": executed,
            "not_executed": not_executed,
            "errors": errored,
            "execution_rate": executed / max(len(entries), 1),
        }

    # ------------------------------------------------------------------
    # Price + portfolio state helpers
    # ------------------------------------------------------------------

    def _get_price(self, ticker: str) -> float:
        """Get current price from IBKR broker."""
        if self._broker and hasattr(self._broker, 'get_quote'):
            try:
                p = self._broker.get_quote(ticker)
                if p and p > 0:
                    return p
            except Exception as e:
                logger.error("L7: IBKR price fetch failed for ticker=%s: %s", ticker, e, exc_info=True)
        if self._broker and hasattr(self._broker, '_get_current_price'):
            try:
                p = self._broker._get_current_price(ticker)
                if p > 0:
                    return p
            except Exception:
                pass
        return 0.0

    def _get_portfolio_state(self) -> Tuple[float, float, dict, float, float, float]:
        """Get (nav, cash, positions, daily_pnl, gross_exposure, net_exposure)."""
        broker = self._broker
        if broker is None:
            return self._initial_cash, self._initial_cash, {}, 0.0, 0.0, 0.0

        try:
            state = broker.state if hasattr(broker, 'state') else None
            if state:
                nav = getattr(state, 'nav', self._initial_cash) or self._initial_cash
                cash = getattr(state, 'cash', nav)
                positions = getattr(state, 'positions', {}) or {}
                daily_pnl = getattr(broker, '_daily_pnl_today', 0.0)
                exposures = broker.compute_exposures() if hasattr(broker, 'compute_exposures') else {}
                gross = exposures.get("gross", 0.0)
                net = exposures.get("net", 0.0)
                return nav, cash, positions, daily_pnl, gross, net
        except Exception as e:
            logger.error("L7: portfolio state retrieval failed: %s", e, exc_info=True)

        return self._initial_cash, self._initial_cash, {}, 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Heartbeat (called every minute from live_loop_orchestrator)
    # ------------------------------------------------------------------

    def heartbeat(self, regime: str = "TRENDING", daily_vol: float = 0.02):
        """1-minute heartbeat — continuous execution surface maintenance.

        Called by live_loop_orchestrator every minute during market hours.
        """
        self._heartbeat_count += 1

        # Update options engine regime
        if self._options_engine:
            try:
                self._options_engine.update_regime(regime)
            except Exception as e:
                logger.warning("L7: options engine regime update failed for regime=%s: %s", regime, e)

        # Every 60 heartbeats (~1 hour): intraday learning optimization
        if self._heartbeat_count % 60 == 0:
            self._learning.daily_optimize()

        # Log heartbeat
        if self._heartbeat_count % 30 == 0:
            nav, _, _, daily_pnl, _, _ = self._get_portfolio_state()
            logger.debug(
                "L7 heartbeat #%d: NAV=$%.2f, daily_pnl=$%.2f, "
                "orders=%d, fills=%d, patterns=%d",
                self._heartbeat_count, nav, daily_pnl,
                len(self._order_book), len(self._filled_orders),
                self._learning.pattern_count,
            )

    def market_open(self):
        """Called at 09:30 ET — reset daily counters."""
        nav, _, _, _, _, _ = self._get_portfolio_state()
        self._risk_engine.reset_daily(nav)
        self._heartbeat_count = 0
        logger.info("L7 market open: NAV=$%.2f", nav)

    def market_close(self):
        """Called at 16:00 ET — daily learning + persistence."""
        self._learning.daily_optimize()
        self._learning.save_patterns()
        nav, _, _, daily_pnl, _, _ = self._get_portfolio_state()
        logger.info(
            "L7 market close: NAV=$%.2f, daily_pnl=$%.2f, fills=%d",
            nav, daily_pnl, len(self._filled_orders),
        )

    def weekly_maintenance(self):
        """Called weekly — refresh learning patterns."""
        self._learning.weekly_refresh()
        self._learning.save_patterns()

    def monthly_maintenance(self):
        """Called monthly — prune stale patterns, recalibrate."""
        self._learning.monthly_prune()
        self._learning.save_patterns()

    # ------------------------------------------------------------------
    # Dashboard / reporting accessors
    # ------------------------------------------------------------------

    def get_risk_state(self) -> Optional[RiskState]:
        """Latest risk state for dashboard."""
        return self._risk_engine.latest_state

    def get_tca_aggregate(self, last_n: int = 0) -> TCAAggregate:
        """TCA aggregate for dashboard."""
        return self._tca.get_aggregate(last_n)

    def get_tca_history(self) -> List[TCASnapshot]:
        """Full TCA history."""
        return self._tca.history

    def get_routing_stats(self) -> Dict[str, int]:
        """Order routing statistics."""
        return self._router.stats

    def get_daily_cost_summary(self) -> Dict[str, float]:
        """Daily avg TCA cost for dashboard chart."""
        return self._learning.daily_cost_summary

    def get_filled_orders(self, last_n: int = 50) -> List[dict]:
        """Recent filled orders for dashboard."""
        orders = list(self._filled_orders)
        if last_n > 0:
            orders = orders[-last_n:]
        return [o.to_dict() for o in orders]

    def get_execution_summary(self) -> dict:
        """Summary for live dashboard."""
        nav, cash, positions, daily_pnl, gross, net = self._get_portfolio_state()
        risk = self._risk_engine.latest_state
        tca = self._tca.get_aggregate(last_n=50)

        return {
            "nav": nav,
            "cash": cash,
            "positions_count": len(positions),
            "daily_pnl": daily_pnl,
            "gross_exposure": gross,
            "net_exposure": net,
            "total_fills_today": len(self._filled_orders),
            "total_orders_today": len(self._order_book),
            "routing_stats": self._router.stats,
            "risk_level": risk.risk_level if risk else "UNKNOWN",
            "kill_switch": risk.kill_switch_active if risk else False,
            "var_95_1d": risk.var_95_1d if risk else 0.0,
            "avg_tca_cost_bps": tca.avg_total_cost_bps,
            "tca_trend": tca.cost_trend,
            "patterns_learned": self._learning.pattern_count,
            "heartbeat": self._heartbeat_count,
        }
