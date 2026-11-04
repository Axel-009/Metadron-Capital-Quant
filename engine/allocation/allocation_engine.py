"""Metadron Capital — Allocation Engine.

Implements portfolio allocation rules, position sizing, bucket management,
beta corridor gating, and kill switch enforcement.

All rules apply identically in backtest mode (backtest=True flag).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

try:
    from engine.execution.options_engine import get_options_sizer, OptionsSizer
except ImportError:
    get_options_sizer = None
    OptionsSizer = None

try:
    from engine.utils.money import D, money, to_float, safe_div
except ImportError:
    from decimal import Decimal as D  # type: ignore
    def money(v): return round(float(v), 2)  # type: ignore
    def to_float(v): return float(v)  # type: ignore
    def safe_div(n, d, default=0): return n / d if d != 0 else default  # type: ignore

logger = logging.getLogger("metadron.decision.allocation")

# ─── Prometheus instrumentation ──────────────────────────────────
_prom_alloc = None

def _get_prom_alloc():
    global _prom_alloc
    if _prom_alloc is not None:
        return _prom_alloc
    try:
        from engine.bridges.prometheus_metrics import get_metrics
        _prom_alloc = get_metrics()
    except Exception:
        _prom_alloc = {}
    return _prom_alloc


class BucketType(str, Enum):
    IG_EQUITY = "IG_EQUITY"                # 40% — investment grade single name
    HY_EQUITY = "HY_EQUITY"                # 10% — high yield equities
    DISTRESSED_EQUITY = "DISTRESSED_EQUITY" # 10% — fallen angels, recovery plays
    TLTW_CASHFLOW = "TLTW_CASHFLOW"        # 15% — TLTW / monthly cash payout ETF (DRIP)
    FI_MACRO = "FI_MACRO"                  # 5%  — fixed income + macro signals
    EVENT_DRIVEN_CVR = "EVENT_DRIVEN_CVR"  # 10% — event-driven + contingent value rights
    # Equity subtotal: 90%
    OPTIONS_IG = "OPTIONS_IG"              # Options overlay (notional, funded by margin)
    OPTIONS_HY = "OPTIONS_HY"
    OPTIONS_DISTRESSED = "OPTIONS_DISTRESSED"
    FUTURES_BETA = "FUTURES_BETA"          # Futures beta corridor (ES/NQ, funded by margin)
    # Levered overlay: 25% options notional + beta corridor futures
    MARGIN = "MARGIN"                      # 8% — initial margin for options + futures
    MONEY_MARKET = "MONEY_MARKET"          # 2% — dry powder (HARD FLOOR)


class BetaCorridorLevel(str, Enum):
    HIGH = "HIGH"
    NEUTRAL = "NEUTRAL"
    LOW = "LOW"


class CyclePhase(str, Enum):
    SCANNING = "SCANNING"
    AGGREGATING = "AGGREGATING"
    EXECUTING = "EXECUTING"
    RISK_CHECK = "RISK_CHECK"
    COOLDOWN = "COOLDOWN"


class InstrumentType(str, Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    ETF = "ETF"
    FIXED_INCOME = "FIXED_INCOME"
    DERIVATIVE = "DERIVATIVE"


@dataclass
class AllocationRules:
    """Allocation rules aligned with Architecture DNA.

    Capital structure (100% of NAV):
        Equity positions (90%):
            IG Equities:       40%  — investment grade single name
            HY Equities:       10%  — high yield equities (BB-B rated)
            Distressed Equity: 10%  — fallen angels, recovery plays
            TLTW/Cashflow ETF: 15%  — monthly payout vehicle (DRIP reinvest)
            FI/Macro:           5%  — fixed income + macro signals
            Event/CVR:         10%  — event-driven + contingent value rights
        Margin (8%):
            Covers 25% options notional + beta corridor futures (ES/NQ)
        Money Market (2%):
            Hard floor dry powder — NEVER breached

    Leverage overlay (funded by margin):
        Options: 25% notional exposure (IG + HY + Distressed)
        Futures: Beta corridor positions (EMini or equiv)

    Hard rule: If margin required > 8%, reduce IG by 5% (IG → 35%).

    OWNER UPDATE (Nov 2026, used by algorithm/decision.py): option premium 10% of NAV (IG 4 · HY 4 · distressed 2),
    money market removed, no IG cut for margin — futures initial margin is sized to the smart-beta corridor.
    """
    max_drawdown_kill_switch: float = 0.10
    # ── Equity sleeves (90% total) ────────────────────────────────
    ig_equity_pct: float = 0.40             # 40% — IG single name equities
    hy_equity_pct: float = 0.10             # 10% — HY equities (separate from distressed)
    distressed_equity_pct: float = 0.10     # 10% — fallen angels, recovery plays
    tltw_cashflow_pct: float = 0.15         # 15% — TLTW or cash payout ETF (DRIP)
    fi_macro_pct: float = 0.05              # 5%  — fixed income + macro signals
    event_driven_cvr_pct: float = 0.10      # 10% — event-driven + CVR
    # ── Levered overlay (funded by margin) ────────────────────────
    options_notional_pct: float = 0.10      # 10% of NAV in option PREMIUM (owner, Nov 2026; was 25% notional)
    options_ig_pct: float = 0.04            #    4% IG options premium
    options_hy_pct: float = 0.04            #    4% HY options premium
    options_distressed_pct: float = 0.02    #    2% distressed options premium
    futures_beta_pct: float = 0.00          # PHASE 2 (owner, Nov 2026): MES overlay off; phase-2 cap 2.0x NAV
    # ── Margin + dry powder ───────────────────────────────────────
    margin_pct: float = 0.08               # 8% — IM for options + futures
    margin_breach_threshold: float = 0.08  # If margin > 8%, reduce IG by 5%
    ig_reduction_on_margin_breach: float = 0.0   # owner: no IG cut — futures margin sized to the corridor
    money_market_pct: float = 0.0           # owner: money-market floor removed (was 2%)
    # ── Behavioral rules ──────────────────────────────────────────
    drip_rule: bool = True                  # Reinvest TLTW distributions
    alpha_primary_goal: bool = True
    cube_kill_switch_override: bool = True  # Honor MetadronCube kill-switch
    # ── Profit-taking rule ────────────────────────────────────────
    profit_take_threshold: float = 0.20     # 20% aggregate P&L → liquidate overlays, lock gains
    profit_take_overlays_only: bool = True  # Only liquidate options + futures (not equity core)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "max_drawdown_kill_switch": self.max_drawdown_kill_switch,
            "ig_equity_pct": self.ig_equity_pct,
            "hy_equity_pct": self.hy_equity_pct,
            "distressed_equity_pct": self.distressed_equity_pct,
            "tltw_cashflow_pct": self.tltw_cashflow_pct,
            "fi_macro_pct": self.fi_macro_pct,
            "event_driven_cvr_pct": self.event_driven_cvr_pct,
            "options_notional_pct": self.options_notional_pct,
            "options_ig_pct": self.options_ig_pct,
            "options_hy_pct": self.options_hy_pct,
            "options_distressed_pct": self.options_distressed_pct,
            "futures_beta_pct": self.futures_beta_pct,
            "margin_pct": self.margin_pct,
            "margin_breach_threshold": self.margin_breach_threshold,
            "ig_reduction_on_margin_breach": self.ig_reduction_on_margin_breach,
            "money_market_pct": self.money_market_pct,
            "drip_rule": self.drip_rule,
            "alpha_primary_goal": self.alpha_primary_goal,
            "cube_kill_switch_override": self.cube_kill_switch_override,
            "timestamp": self.timestamp,
        }


@dataclass
class ScanSignal:
    ticker: str
    signal_type: str
    instrument_type: str
    confidence: float
    alpha_score: float
    regime_context: str
    bucket: str = ""
    universe: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "instrument_type": self.instrument_type,
            "confidence": self.confidence,
            "alpha_score": self.alpha_score,
            "regime_context": self.regime_context,
            "bucket": self.bucket,
            "universe": self.universe,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class PositionAllocation:
    ticker: str
    bucket: str
    instrument_type: str
    position_size: float
    dollar_amount: float
    confidence: float
    alpha_score: float
    signal_type: str
    regime_context: str
    timestamp: str = ""
    # Options sizing (populated by OptionsSizer: BS + MC + Kelly)
    options_contracts: int = 0
    options_bs_price: float = 0.0
    options_mc_price: float = 0.0
    options_kelly_fraction: float = 0.0
    options_edge_bps: float = 0.0
    options_delta: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "bucket": self.bucket,
            "instrument_type": self.instrument_type,
            "position_size": self.position_size,
            "dollar_amount": round(self.dollar_amount, 2),
            "confidence": self.confidence,
            "alpha_score": self.alpha_score,
            "signal_type": self.signal_type,
            "regime_context": self.regime_context,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if self.options_contracts > 0:
            d["options_contracts"] = self.options_contracts
            d["options_bs_price"] = self.options_bs_price
            d["options_mc_price"] = self.options_mc_price
            d["options_kelly_fraction"] = self.options_kelly_fraction
            d["options_edge_bps"] = self.options_edge_bps
            d["options_delta"] = self.options_delta
        return d


@dataclass
class AllocationSlate:
    positions: list = field(default_factory=list)
    # Equity sleeves (90%)
    total_ig_equity: float = 0.0
    total_hy_equity: float = 0.0
    total_distressed_equity: float = 0.0
    total_tltw_cashflow: float = 0.0
    total_fi_macro: float = 0.0
    total_event_cvr: float = 0.0
    # Levered overlay
    total_options_notional: float = 0.0
    total_futures_beta: float = 0.0
    # Margin + dry powder
    total_margin: float = 0.0
    total_money_market: float = 0.0
    # State
    margin_breach: bool = False             # True if margin > 8% threshold
    ig_reduced: bool = False                # True if IG was reduced due to margin breach
    kill_switch_triggered: bool = False
    beta_corridor: str = "NEUTRAL"
    leverage_multiplier: float = 1.0
    cycle_number: int = 0
    phase: str = "SCANNING"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.positions],
            "bucket_utilization": {
                "ig_equity": round(self.total_ig_equity, 4),
                "hy_equity": round(self.total_hy_equity, 4),
                "distressed_equity": round(self.total_distressed_equity, 4),
                "tltw_cashflow": round(self.total_tltw_cashflow, 4),
                "fi_macro": round(self.total_fi_macro, 4),
                "event_driven_cvr": round(self.total_event_cvr, 4),
                "options_notional": round(self.total_options_notional, 4),
                "futures_beta": round(self.total_futures_beta, 4),
                "margin": round(self.total_margin, 4),
                "money_market": round(self.total_money_market, 4),
            },
            "margin_breach": self.margin_breach,
            "ig_reduced": self.ig_reduced,
            "kill_switch_triggered": self.kill_switch_triggered,
            "beta_corridor": self.beta_corridor,
            "leverage_multiplier": self.leverage_multiplier,
            "cycle_number": self.cycle_number,
            "phase": self.phase,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ─── Kill Switch Monitor ──────────────────────────────────────────

class KillSwitchMonitor:
    """Monitors drawdown and enforces the kill switch when breached.

    The kill switch is a one-way latch: once triggered it does NOT
    auto-reset.  An operator must call request_reset() followed by
    confirm_reset() (or the legacy reset() alias) to clear it.
    """

    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold
        self._triggered: bool = False
        self._triggered_at: Optional[str] = None
        self.pending_reset: bool = False
        self._reset_requested_timestamp: Optional[str] = None

    # ── state queries ─────────────────────────────────────────────

    @property
    def triggered(self) -> bool:
        return self._triggered

    def is_triggered(self) -> bool:
        return self._triggered

    # ── trigger ───────────────────────────────────────────────────

    def check(self, drawdown: float) -> bool:
        """Evaluate drawdown; latch kill switch if threshold breached."""
        if drawdown >= self.threshold and not self._triggered:
            self._triggered = True
            self._triggered_at = datetime.now(timezone.utc).isoformat()
            self.pending_reset = True  # awaits manual operator action
            logger.critical(
                "KILL SWITCH TRIGGERED: drawdown=%.4f >= threshold=%.4f at %s",
                drawdown,
                self.threshold,
                self._triggered_at,
            )
        return self._triggered

    # ── operator reset flow ───────────────────────────────────────

    def request_reset(self) -> None:
        """Signal operator intent to reset.

        Sets pending_reset=True and records the request timestamp.
        Does NOT clear the triggered latch — confirm_reset() must
        be called to complete the reset.
        """
        self.pending_reset = True
        self._reset_requested_timestamp = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Kill switch reset REQUESTED by operator at %s (still triggered until confirmed)",
            self._reset_requested_timestamp,
        )

    def confirm_reset(self) -> None:
        """Complete the operator reset cycle.

        Clears both triggered and pending_reset flags and logs the event.
        """
        prev_triggered_at = self._triggered_at
        self._triggered = False
        self._triggered_at = None
        self.pending_reset = False
        confirmed_at = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Kill switch RESET CONFIRMED by operator at %s (was triggered at %s)",
            confirmed_at,
            prev_triggered_at,
        )
        # Emit Prometheus metric
        try:
            metrics = _get_prom_alloc()
            if "kill_switch_reset_total" in metrics:
                metrics["kill_switch_reset_total"].inc()
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(0)
        except Exception:
            pass

    def reset(self) -> None:
        """Backward-compatible alias for confirm_reset()."""
        self.confirm_reset()

    # ── status ────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "triggered": self._triggered,
            "triggered_at": self._triggered_at,
            "threshold": self.threshold,
            "pending_reset": self.pending_reset,
            "reset_requested_at": self._reset_requested_timestamp,
        }


# ─── Allocation Engine ────────────────────────────────────────────

class AllocationEngine:
    """Core allocation engine — applies rules, sizes positions, enforces limits."""

    def __init__(
        self,
        rules: Optional[AllocationRules] = None,
        backtest: bool = False,
        cycle_number: int = 0,
        nav: float = 1_000_000.0,
    ):
        self.rules = rules or AllocationRules()
        self.backtest = backtest
        self.cycle_number = cycle_number
        self._nav: float = nav
        self.kill_switch = KillSwitchMonitor(
            threshold=self.rules.max_drawdown_kill_switch
        )
        self._beta_corridor: BetaCorridorLevel = BetaCorridorLevel.NEUTRAL
        self._leverage_multiplier: float = 1.0
        self._margin_breached: bool = False     # True when margin > threshold
        self._cube_kill_switch: bool = False     # True when MetadronCube kill-switch active

        # Per-bucket utilization accumulators (fraction of total capital)
        self._utilization: dict[str, float] = {
            BucketType.IG_EQUITY: 0.0,
            BucketType.HY_EQUITY: 0.0,
            BucketType.DISTRESSED_EQUITY: 0.0,
            BucketType.TLTW_CASHFLOW: 0.0,
            BucketType.FI_MACRO: 0.0,
            BucketType.EVENT_DRIVEN_CVR: 0.0,
            BucketType.OPTIONS_IG: 0.0,
            BucketType.OPTIONS_HY: 0.0,
            BucketType.OPTIONS_DISTRESSED: 0.0,
            BucketType.FUTURES_BETA: 0.0,
            BucketType.MARGIN: 0.0,
            BucketType.MONEY_MARKET: 0.0,
        }

    # ── bucket utilization ────────────────────────────────────────

    def reset_cycle_utilization(self) -> None:
        """Zero all bucket utilization accumulators.

        Called automatically at the start of each apply_rules() call
        to prevent cross-cycle accumulation (bucket accumulation bug fix).
        """
        for key in self._utilization:
            self._utilization[key] = 0.0
        logger.debug("Allocation bucket utilization reset for new cycle.")
        try:
            metrics = _get_prom_alloc()
            if "allocation_cycle_reset_total" in metrics:
                metrics["allocation_cycle_reset_total"].inc()
        except Exception:
            pass

    # ── beta corridor ─────────────────────────────────────────────

    def set_beta_corridor(self, level: BetaCorridorLevel) -> None:
        self._beta_corridor = level
        if level == BetaCorridorLevel.HIGH:
            self._leverage_multiplier = 0.5
        elif level == BetaCorridorLevel.LOW:
            self._leverage_multiplier = 1.25
        else:
            self._leverage_multiplier = 1.0
        logger.info(
            "Beta corridor set to %s → leverage_multiplier=%.2f",
            level,
            self._leverage_multiplier,
        )

    # ── core rules application ────────────────────────────────────

    def apply_rules(
        self,
        signals: list,
        total_capital: float,
        drawdown: float = 0.0,
        beta_corridor: Optional[BetaCorridorLevel] = None,
    ) -> "AllocationSlate":
        """Apply allocation rules to a list of ScanSignals.

        Resets bucket utilization at the start of each call to prevent
        cross-cycle accumulation (bucket accumulation bug fix).
        """
        # ── FIX: reset utilization at start of every cycle ────────
        self.reset_cycle_utilization()

        if beta_corridor is not None:
            self.set_beta_corridor(beta_corridor)

        self.cycle_number += 1
        slate = AllocationSlate(
            cycle_number=self.cycle_number,
            beta_corridor=self._beta_corridor.value,
            leverage_multiplier=self._leverage_multiplier,
            phase=CyclePhase.RISK_CHECK.value,
        )

        # ── Per-cycle rule header ─────────────────────────────────
        logger.info(
            "[AllocationEngine] ── Cycle #%d Rule Check ──────────────────────────────",
            self.cycle_number,
        )
        logger.info(
            "[AllocationEngine]   NAV: $%.0f  |  Drawdown: %.2f%%  |  Kill-switch threshold: %.0f%%",
            total_capital,
            drawdown * 100,
            self.rules.max_drawdown_kill_switch * 100,
        )
        logger.info(
            "[AllocationEngine]   Beta corridor: %s  |  Leverage multiplier: %.2fx  |  Cube kill-switch: %s",
            self._beta_corridor.value,
            self._leverage_multiplier,
            self._cube_kill_switch,
        )
        logger.info(
            "[AllocationEngine]   Bucket caps (pct of NAV):"
            "  IG=%.0f%%  HY=%.0f%%  Dist=%.0f%%  TLTW=%.0f%%"
            "  FI=%.0f%%  CVR=%.0f%%  Opts=%.0f%%  Margin=%.0f%%  MM=%.0f%%",
            self.rules.ig_equity_pct * 100,
            self.rules.hy_equity_pct * 100,
            self.rules.distressed_equity_pct * 100,
            self.rules.tltw_cashflow_pct * 100,
            self.rules.fi_macro_pct * 100,
            self.rules.event_driven_cvr_pct * 100,
            self.rules.options_notional_pct * 100,
            self.rules.margin_pct * 100,
            self.rules.money_market_pct * 100,
        )
        logger.info(
            "[AllocationEngine]   DRIP rule: %s  |  Alpha primary goal: %s  |  Signals in: %d",
            self.rules.drip_rule,
            self.rules.alpha_primary_goal,
            len(signals),
        )

        # Kill switch check — portfolio-level 20% drawdown
        if self.kill_switch.check(drawdown):
            logger.critical(
                "[AllocationEngine]   PORTFOLIO KILL SWITCH ACTIVE — drawdown %.2f%% >= threshold %.0f%% "
                "— LIQUIDATE ALL POSITIONS. Operator reset required.",
                drawdown * 100,
                self.rules.max_drawdown_kill_switch * 100,
            )
            return self.validate_against_kill_switch(slate, total_capital)

        # MetadronCube kill-switch override (HY OAS +35bp & VIX flat & breadth <50%)
        if self._cube_kill_switch and self.rules.cube_kill_switch_override:
            logger.critical(
                "[AllocationEngine]   CUBE KILL SWITCH ACTIVE "
                "— HY OAS / VIX / breadth matrix triggered — ALL equity trades BLOCKED"
            )
            return self.validate_against_kill_switch(slate, total_capital)

        slate.phase = CyclePhase.AGGREGATING.value

        rejected_cap: list[str] = []
        for sig in signals:
            alloc = self._size_position(sig, total_capital)
            if alloc is None:
                continue
            if not self._fits_in_bucket(alloc):
                rejected_cap.append(
                    f"{sig.ticker} (bucket={alloc.bucket} "
                    f"used={self._utilization.get(alloc.bucket, 0.0)*100:.1f}% "
                    f"cap={self._bucket_cap(alloc.bucket)*100:.0f}%)"
                )
                logger.info(
                    "[AllocationEngine]   REJECTED %s — bucket %s full "
                    "(utilization=%.1f%% cap=%.0f%% size_requested=%.2f%%)",
                    sig.ticker,
                    alloc.bucket,
                    self._utilization.get(alloc.bucket, 0.0) * 100,
                    (self._bucket_cap(alloc.bucket) or 0.0) * 100,
                    alloc.position_size * 100,
                )
                continue
            self._apply_utilization(alloc, total_capital)
            slate.positions.append(alloc)
            logger.info(
                "[AllocationEngine]   ACCEPTED %s — bucket=%s size=%.2f%% ($%.0f) "
                "confidence=%.2f alpha=%.4f",
                alloc.ticker,
                alloc.bucket,
                alloc.position_size * 100,
                alloc.dollar_amount,
                alloc.confidence,
                alloc.alpha_score,
            )

        # ── Margin breach check ───────────────────────────────────
        # If margin required for options + futures > threshold, reduce IG by 5%
        options_margin = (
            self._utilization[BucketType.OPTIONS_IG]
            + self._utilization[BucketType.OPTIONS_HY]
            + self._utilization[BucketType.OPTIONS_DISTRESSED]
        )
        futures_margin = self._utilization[BucketType.FUTURES_BETA]
        estimated_margin = (options_margin + futures_margin) * 0.35  # ~35% of notional as IM
        if estimated_margin > self.rules.margin_breach_threshold:
            self._margin_breached = True
            logger.warning(
                "[AllocationEngine]   MARGIN BREACH: estimated IM=%.2f%% > threshold=%.0f%% "
                "— reducing IG cap by %.0f%% (%.0f%% → %.0f%%)",
                estimated_margin * 100,
                self.rules.margin_breach_threshold * 100,
                self.rules.ig_reduction_on_margin_breach * 100,
                self.rules.ig_equity_pct * 100,
                (self.rules.ig_equity_pct - self.rules.ig_reduction_on_margin_breach) * 100,
            )
        else:
            self._margin_breached = False

        # ── Money market hard floor enforcement ──────────────────
        # 2% money market is NEVER breached — always reserve this
        self._utilization[BucketType.MONEY_MARKET] = max(
            self._utilization[BucketType.MONEY_MARKET],
            self.rules.money_market_pct,
        )

        # Populate slate totals from utilization
        slate.total_ig_equity = self._utilization[BucketType.IG_EQUITY]
        slate.total_hy_equity = self._utilization[BucketType.HY_EQUITY]
        slate.total_distressed_equity = self._utilization[BucketType.DISTRESSED_EQUITY]
        slate.total_tltw_cashflow = self._utilization[BucketType.TLTW_CASHFLOW]
        slate.total_fi_macro = self._utilization[BucketType.FI_MACRO]
        slate.total_event_cvr = self._utilization[BucketType.EVENT_DRIVEN_CVR]
        slate.total_options_notional = options_margin
        slate.total_futures_beta = futures_margin
        slate.total_margin = estimated_margin
        slate.total_money_market = self._utilization[BucketType.MONEY_MARKET]
        slate.margin_breach = self._margin_breached
        slate.ig_reduced = self._margin_breached
        slate.phase = CyclePhase.EXECUTING.value
        slate.timestamp = datetime.now(timezone.utc).isoformat()

        # ── Per-cycle rule summary ────────────────────────────────
        logger.info(
            "[AllocationEngine] ── Cycle #%d Summary ───────────────────────────────────",
            self.cycle_number,
        )
        logger.info(
            "[AllocationEngine]   Result: %d/%d signals → slate  |  %d rejected by bucket caps",
            len(slate.positions),
            len(signals),
            len(rejected_cap),
        )
        logger.info(
            "[AllocationEngine]   Bucket utilization after cycle:"
            "  IG=%.1f%%  HY=%.1f%%  Dist=%.1f%%  TLTW=%.1f%%"
            "  FI=%.1f%%  CVR=%.1f%%  Opts=%.1f%%  Futures=%.1f%%  Margin=%.1f%%  MM=%.1f%%",
            slate.total_ig_equity * 100,
            slate.total_hy_equity * 100,
            slate.total_distressed_equity * 100,
            slate.total_tltw_cashflow * 100,
            slate.total_fi_macro * 100,
            slate.total_event_cvr * 100,
            slate.total_options_notional * 100,
            slate.total_futures_beta * 100,
            slate.total_margin * 100,
            slate.total_money_market * 100,
        )
        total_deployed = sum(p.position_size for p in slate.positions)
        total_dollars = sum(p.dollar_amount for p in slate.positions)
        logger.info(
            "[AllocationEngine]   Total deployed: %.2f%% of NAV = $%.0f  |  Margin breach: %s",
            total_deployed * 100,
            total_dollars,
            self._margin_breached,
        )
        logger.info(
            "[AllocationEngine] ────────────────────────────────────────────────────────",
        )
        return slate

    # ── position sizing ───────────────────────────────────────────

    def _size_position(
        self, sig: ScanSignal, total_capital: float
    ) -> Optional[PositionAllocation]:
        """Compute position size and dollar amount for a signal (Decimal precision).

        For options: uses OptionsSizer (Black-Scholes + Monte Carlo + Kelly)
        to compute proper contract quantity instead of falling through to qty=1.
        """
        bucket = sig.bucket or self._infer_bucket(sig)
        cap = self._bucket_cap(bucket)
        if cap is None or cap <= 0:
            return None

        d_conf = D(sig.confidence)
        d_cap = D(cap)
        d_lev = D(self._leverage_multiplier)
        base_size = min(d_conf * d_cap * d_lev, d_cap)
        dollar_amount = to_float(money(base_size * D(total_capital)))

        # Options: BS + MC + Kelly sizing — ONLY if mispricing detected
        options_sizing = None
        if bucket in (BucketType.OPTIONS_IG, BucketType.OPTIONS_HY, BucketType.OPTIONS_DISTRESSED):
            options_sizing = self._size_option_contracts(sig, dollar_amount, total_capital)
            if options_sizing is None:
                return None

        alloc = PositionAllocation(
            ticker=sig.ticker,
            bucket=bucket,
            instrument_type=sig.instrument_type,
            position_size=round(to_float(base_size), 6),
            dollar_amount=dollar_amount,
            confidence=sig.confidence,
            alpha_score=sig.alpha_score,
            signal_type=sig.signal_type,
            regime_context=sig.regime_context,
        )

        if options_sizing:
            alloc.options_contracts = options_sizing.get("contracts", 1)
            alloc.options_bs_price = options_sizing.get("bs_price", 0)
            alloc.options_mc_price = options_sizing.get("mc_price", 0)
            alloc.options_kelly_fraction = options_sizing.get("kelly_fraction", 0)
            alloc.options_edge_bps = options_sizing.get("edge_bps", 0)
            alloc.options_delta = options_sizing.get("delta", 0)
            alloc.dollar_amount = options_sizing.get("total_cost", dollar_amount)

        return alloc

    def _size_option_contracts(
        self, sig: ScanSignal, budget_dollars: float, nav: float,
    ) -> Optional[dict]:
        """Use OptionsSizer (BS + MC + Kelly) — ONLY allocates on detected mispricing.

        Requires market_price on the signal for edge assessment. Options without
        a market price or without sufficient mispricing edge are rejected.
        """
        if get_options_sizer is None:
            return None
        try:
            sizer = get_options_sizer()
            spot = getattr(sig, "spot_price", 0) or getattr(sig, "price", 0) or 0
            vol = getattr(sig, "implied_vol", 0) or getattr(sig, "volatility", 0) or 0.25
            market_price = getattr(sig, "market_price", 0) or getattr(sig, "option_price", 0) or 0
            strike = getattr(sig, "strike", None)
            signal_type = (sig.signal_type or "").upper()

            if spot <= 0:
                logger.debug("OptionsSizer: %s skipped — no spot price", sig.ticker)
                return None

            if market_price <= 0:
                logger.debug("OptionsSizer: %s skipped — no market price for mispricing detection", sig.ticker)
                return None

            is_call = "PUT" not in signal_type
            otm_pct = 0.05
            expiry_days = 30

            if "DISTRESSED" in signal_type or "FALLEN" in signal_type:
                otm_pct = 0.10
                expiry_days = 45
            elif "HY" in signal_type:
                otm_pct = 0.07

            result = sizer.size_from_signal(
                ticker=sig.ticker,
                spot=spot,
                vol=vol,
                nav=nav,
                budget_dollars=budget_dollars,
                market_price=market_price,
                is_call=is_call,
                strike=strike,
                otm_pct=otm_pct,
                expiry_days=expiry_days,
            )

            if result.get("rejected"):
                logger.info(
                    "OptionsSizer REJECTED %s: %s (edge=%.0fbps)",
                    sig.ticker, result.get("reject_reason", ""), result.get("edge_bps", 0),
                )
                return None

            logger.info(
                "OptionsSizer: %s → %d contracts (fair=$%.2f vs mkt=$%.2f, edge=%.0fbps, Kelly=%.4f)",
                sig.ticker, result["contracts"], result["fair_value"],
                result["market_price"], result["edge_bps"], result["kelly_fraction"],
            )
            return result
        except Exception as e:
            logger.warning("OptionsSizer failed for %s: %s", sig.ticker, e)
            return None

    def _infer_bucket(self, sig: ScanSignal) -> str:
        itype = sig.instrument_type.upper() if sig.instrument_type else ""
        signal_type = (sig.signal_type or "").upper()
        regime = (sig.regime_context or "").upper()
        ticker = (sig.ticker or "").upper()

        # Options overlay — route by credit quality context
        if itype == InstrumentType.OPTION:
            if "DISTRESSED" in regime or "DISTRESSED" in signal_type:
                return BucketType.OPTIONS_DISTRESSED
            if "HY" in regime or "HY" in signal_type:
                return BucketType.OPTIONS_HY
            return BucketType.OPTIONS_IG

        # Futures — beta corridor instruments
        if itype == InstrumentType.FUTURE or itype == InstrumentType.DERIVATIVE:
            return BucketType.FUTURES_BETA

        # TLTW / cash payout ETF
        if ticker in ("TLTW", "JEPI", "JEPQ", "DIVO", "XYLD", "QYLD", "RYLD"):
            return BucketType.TLTW_CASHFLOW

        # Fixed income
        if itype == InstrumentType.FIXED_INCOME:
            if "CVR" in signal_type or "EVENT" in signal_type:
                return BucketType.EVENT_DRIVEN_CVR
            return BucketType.FI_MACRO

        # ETFs — route to TLTW cashflow or FI depending on type
        if itype == InstrumentType.ETF:
            # Income ETFs go to TLTW cashflow bucket
            if ticker in ("TLT", "IEF", "SHY", "AGG", "BND", "MBB", "VMBS", "TIPS", "MUB"):
                return BucketType.FI_MACRO
            if ticker in ("GLD", "SLV", "USO", "UNG", "DBA", "DBC", "COPX", "WEAT", "CORN"):
                return BucketType.TLTW_CASHFLOW
            return BucketType.TLTW_CASHFLOW

        # Event-driven / CVR signals
        if "CVR" in signal_type or "EVENT" in signal_type or "MERGER" in signal_type:
            return BucketType.EVENT_DRIVEN_CVR

        # Distressed equity
        if "DISTRESSED" in regime or "FALLEN_ANGEL" in signal_type:
            return BucketType.DISTRESSED_EQUITY

        # High yield equity
        if "HY" in regime:
            return BucketType.HY_EQUITY

        # Default: investment grade equity
        return BucketType.IG_EQUITY

    def _bucket_cap(self, bucket: str) -> Optional[float]:
        # If margin breach active, IG cap reduced by ig_reduction_on_margin_breach
        ig_cap = self.rules.ig_equity_pct
        if self._margin_breached:
            ig_cap = self.rules.ig_equity_pct - self.rules.ig_reduction_on_margin_breach

        caps = {
            BucketType.IG_EQUITY: ig_cap,
            BucketType.HY_EQUITY: self.rules.hy_equity_pct,
            BucketType.DISTRESSED_EQUITY: self.rules.distressed_equity_pct,
            BucketType.TLTW_CASHFLOW: self.rules.tltw_cashflow_pct,
            BucketType.FI_MACRO: self.rules.fi_macro_pct,
            BucketType.EVENT_DRIVEN_CVR: self.rules.event_driven_cvr_pct,
            BucketType.OPTIONS_IG: self.rules.options_ig_pct,
            BucketType.OPTIONS_HY: self.rules.options_hy_pct,
            BucketType.OPTIONS_DISTRESSED: self.rules.options_distressed_pct,
            BucketType.FUTURES_BETA: self.rules.futures_beta_pct,
            BucketType.MARGIN: self.rules.margin_pct + (
                self.rules.ig_reduction_on_margin_breach if self._margin_breached else 0.0
            ),
            BucketType.MONEY_MARKET: self.rules.money_market_pct,
        }
        return caps.get(bucket)

    # ── bucket fits check ─────────────────────────────────────────

    def _fits_in_bucket(self, alloc: PositionAllocation) -> bool:
        cap = self._bucket_cap(alloc.bucket)
        if cap is None:
            return False
        current = self._utilization.get(alloc.bucket, 0.0)
        return (current + alloc.position_size) <= cap

    def _apply_utilization(
        self, alloc: PositionAllocation, total_capital: float
    ) -> None:
        self._utilization[alloc.bucket] = (
            self._utilization.get(alloc.bucket, 0.0) + alloc.position_size
        )

    # ── kill switch enforcement ───────────────────────────────────

    def validate_against_kill_switch(
        self, slate: AllocationSlate, total_capital: float
    ) -> AllocationSlate:
        """Build a halt slate when kill switch is active.

        Emits Prometheus counters/gauges for observability.
        """
        halt_slate = AllocationSlate(
            positions=[],
            kill_switch_triggered=True,
            beta_corridor=self._beta_corridor.value,
            leverage_multiplier=0.0,
            cycle_number=self.cycle_number,
            phase=CyclePhase.COOLDOWN.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.critical(
            "KILL SWITCH ACTIVE — all allocations halted (cycle=%d)",
            self.cycle_number,
        )

        # ── Prometheus instrumentation ────────────────────────────
        try:
            metrics = _get_prom_alloc()
            if "kill_switch_triggered_total" in metrics:
                metrics["kill_switch_triggered_total"].inc()
            # New gauge: 1 = triggered/awaiting reset, 0 = clear
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(
                    1 if self.kill_switch.pending_reset else 0
                )
        except Exception:
            pass

        return halt_slate

    # ── introspection ─────────────────────────────────────────────

    def get_utilization(self) -> dict:
        return {k: round(v, 6) for k, v in self._utilization.items()}

    def get_rules(self) -> dict:
        return self.rules.to_dict()

    def get_kill_switch_status(self) -> dict:
        return self.kill_switch.status()

    # ── nav property ──────────────────────────────────────────────

    @property
    def nav(self) -> float:
        """Current NAV used for position sizing."""
        return self._nav

    @nav.setter
    def nav(self, value: float) -> None:
        self._nav = float(value)

    # ── classify_opportunity ──────────────────────────────────────

    # ── cube kill-switch integration ─────────────────────────────

    def set_cube_kill_switch(self, active: bool) -> None:
        """Set the MetadronCube kill-switch state.

        When active (HY OAS +35bp & VIX flat/inverted & breadth <50%),
        all equity trades are blocked until the cube clears the condition.
        """
        prev = self._cube_kill_switch
        self._cube_kill_switch = active
        if active and not prev:
            logger.critical(
                "[AllocationEngine] Cube kill-switch ACTIVATED — "
                "HY OAS / VIX term / breadth matrix triggered"
            )
        elif not active and prev:
            logger.info("[AllocationEngine] Cube kill-switch CLEARED")

    # ── position-level drawdown exit ─────────────────────────────

    def check_position_drawdown(
        self, positions: list, cost_basis: dict, current_prices: dict,
    ) -> list:
        """Check individual position drawdowns and flag exits.

        If any position is down >= max_drawdown_kill_switch (20%) from
        cost basis, return it for liquidation.

        Parameters
        ----------
        positions : list
            Current open positions (ticker, qty, etc.)
        cost_basis : dict
            {ticker: avg_cost_per_share}
        current_prices : dict
            {ticker: current_price}

        Returns
        -------
        list
            Tickers that must be liquidated (position down >= 20%).
        """
        liquidate = []
        threshold = self.rules.max_drawdown_kill_switch

        for pos in positions:
            ticker = pos.get("ticker") or getattr(pos, "ticker", "")
            if not ticker:
                continue
            cost = cost_basis.get(ticker, 0.0)
            price = current_prices.get(ticker, 0.0)
            if cost <= 0 or price <= 0:
                continue
            drawdown = (cost - price) / cost
            if drawdown >= threshold:
                logger.warning(
                    "[AllocationEngine] POSITION EXIT: %s down %.1f%% "
                    "(cost=%.2f current=%.2f threshold=%.0f%%) — LIQUIDATE",
                    ticker, drawdown * 100, cost, price, threshold * 100,
                )
                liquidate.append(ticker)
        return liquidate

    # ── Profit-taking: 3-layer P&L check → liquidate overlays ────

    def check_profit_take(
        self,
        nav: float,
        initial_nav: float,
        ibkr_pnl: float = 0.0,
        futures_pnl: float = 0.0,
        equities_pnl: float = 0.0,
        options_pnl: float = 0.0,
    ) -> dict:
        """3-layer profit-take check with per-product-class breakdown.

        Layer 1 (IBKR-specific): If IBKR broker P&L > 20% →
            liquidate OPTIONS only, re-run. IBKR handles equities + options.
        Layer 2 (Futures-specific): If futures/Rithmic P&L > 20% →
            liquidate FUTURES only, re-run. Futures on trade log (Rithmic later).
        Layer 3 (Aggregate): If total P&L > 20% →
            liquidate ALL overlays (options + futures), re-run.

        Equity core positions are always retained. Only overlays liquidated.

        Parameters
        ----------
        nav : float
            Current total portfolio NAV
        initial_nav : float
            Starting NAV (beginning of day or initial capital)
        ibkr_pnl : float
            P&L from IBKR broker (equities + options combined)
        futures_pnl : float
            P&L from trade log / Rithmic (futures only)
        equities_pnl : float
            P&L from equity positions only (for reporting)
        options_pnl : float
            P&L from options positions only (for reporting)

        Returns
        -------
        dict with:
            pnl_breakdown: {total, equities, options, futures, total_pct}
            triggered: bool
            trigger_layer: str — "ibkr", "futures", "aggregate", or "none"
            liquidate: list of bucket names to liquidate
            action: str — "PROFIT_TAKE_OPTIONS", "PROFIT_TAKE_FUTURES",
                         "PROFIT_TAKE_ALL_OVERLAYS", or "HOLD"
        """
        if initial_nav <= 0:
            return {
                "pnl_breakdown": {"total": 0, "equities": 0, "options": 0, "futures": 0, "total_pct": 0},
                "triggered": False, "trigger_layer": "none", "liquidate": [], "action": "HOLD",
            }

        total_pnl = nav - initial_nav
        total_pct = total_pnl / initial_nav
        threshold = self.rules.profit_take_threshold

        # P&L breakdown (always reported)
        breakdown = {
            "total": round(total_pnl, 2),
            "total_pct": round(total_pct, 4),
            "equities": round(equities_pnl, 2),
            "options": round(options_pnl, 2),
            "futures": round(futures_pnl, 2),
            "ibkr_combined": round(ibkr_pnl, 2),
        }

        result = {
            "pnl_breakdown": breakdown,
            "triggered": False,
            "trigger_layer": "none",
            "liquidate": [],
            "action": "HOLD",
            "threshold": threshold,
        }

        options_buckets = [
            BucketType.OPTIONS_IG.value,
            BucketType.OPTIONS_HY.value,
            BucketType.OPTIONS_DISTRESSED.value,
        ]
        futures_buckets = [BucketType.FUTURES_BETA.value]
        all_overlay_buckets = options_buckets + futures_buckets

        # ── Layer 3 (checked first — aggregate overrides individual) ──
        if total_pct >= threshold:
            result["triggered"] = True
            result["trigger_layer"] = "aggregate"
            result["liquidate"] = all_overlay_buckets
            result["action"] = "PROFIT_TAKE_ALL_OVERLAYS"
            logger.critical(
                "[AllocationEngine] PROFIT TAKE — AGGREGATE: total P&L %.1f%% >= %.0f%% "
                "— liquidating ALL overlays (options + futures). Re-run on next scan.",
                total_pct * 100, threshold * 100,
            )
            return result

        # ── Layer 1 (IBKRBroker: equities + options) ──
        if initial_nav > 0 and ibkr_pnl / initial_nav >= threshold:
            result["triggered"] = True
            result["trigger_layer"] = "ibkr"
            result["liquidate"] = options_buckets
            result["action"] = "PROFIT_TAKE_OPTIONS"
            logger.critical(
                "[AllocationEngine] PROFIT TAKE — IBKR: IBKR P&L $%.0f (%.1f%%) >= %.0f%% "
                "— liquidating OPTIONS. Equities retained. Re-run on next scan.",
                ibkr_pnl, (ibkr_pnl / initial_nav) * 100, threshold * 100,
            )
            return result

        # ── Layer 2 (Futures: trade log / Rithmic) ──
        if initial_nav > 0 and futures_pnl / initial_nav >= threshold:
            result["triggered"] = True
            result["trigger_layer"] = "futures"
            result["liquidate"] = futures_buckets
            result["action"] = "PROFIT_TAKE_FUTURES"
            logger.critical(
                "[AllocationEngine] PROFIT TAKE — FUTURES: Futures P&L $%.0f (%.1f%%) >= %.0f%% "
                "— liquidating FUTURES. Options + equities retained. Re-run on next scan.",
                futures_pnl, (futures_pnl / initial_nav) * 100, threshold * 100,
            )
            return result

        return result

    def classify_opportunity(self, sig: "ScanSignal") -> str:
        """Classify a ScanSignal into its allocation bucket.

        Used by FullUniverseScan to emit bucket labels on the Thinking Tab SSE.
        Mirrors _infer_bucket() but public-facing and accepts a ScanSignal.
        """
        itype = (sig.instrument_type or "").upper()
        signal_type = (sig.signal_type or "").upper()
        regime = (sig.regime_context or "").upper()
        ticker = (sig.ticker or "").upper()

        if itype == InstrumentType.OPTION:
            if "HY" in regime or "DISTRESSED" in regime:
                return BucketType.OPTIONS_DISTRESSED
            if "HY" in signal_type:
                return BucketType.OPTIONS_HY
            return BucketType.OPTIONS_IG
        if itype in (InstrumentType.FUTURE, InstrumentType.DERIVATIVE):
            return BucketType.FUTURES_BETA
        if ticker in ("TLTW", "JEPI", "JEPQ", "DIVO", "XYLD", "QYLD", "RYLD"):
            return BucketType.TLTW_CASHFLOW
        if itype == InstrumentType.FIXED_INCOME:
            if "CVR" in signal_type or "EVENT" in signal_type:
                return BucketType.EVENT_DRIVEN_CVR
            return BucketType.FI_MACRO
        if itype == InstrumentType.ETF:
            return BucketType.TLTW_CASHFLOW
        # Equity paths
        if "DISTRESSED" in regime or "DISTRESS" in signal_type:
            return BucketType.DISTRESSED_EQUITY
        if "HY" in regime or "HY" in signal_type:
            return BucketType.HY_EQUITY
        if "CVR" in signal_type or "EVENT" in signal_type:
            return BucketType.EVENT_DRIVEN_CVR
        return BucketType.IG_EQUITY

    # ── aggregate_runs ────────────────────────────────────────────

    def aggregate_runs(self, *slates: "AllocationSlate") -> "AllocationSlate":
        """Merge 4 universe-run slates into a single de-duplicated slate.

        Rules:
        - If any run triggered the kill switch → aggregate is halted
        - Duplicate tickers across runs are merged by taking the highest
          confidence score (same ticker may appear in SP500 + SP400)
        - Bucket utilization is recomputed from scratch on the merged set
        - Leverage multiplier is taken from the last (most recent) slate
        """
        if not slates:
            return AllocationSlate(cycle_number=self.cycle_number)

        # Kill switch propagation — any triggered run halts the aggregate
        if any(s.kill_switch_triggered for s in slates):
            halt = AllocationSlate(
                kill_switch_triggered=True,
                cycle_number=self.cycle_number,
                phase=CyclePhase.COOLDOWN.value,
                beta_corridor=slates[-1].beta_corridor,
                leverage_multiplier=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            logger.critical(
                "[AllocationEngine] aggregate_runs: kill switch in one or more run slates — halting aggregate"
            )
            return halt

        # Merge positions — deduplicate by ticker, keep highest confidence
        seen: dict[str, PositionAllocation] = {}
        for slate in slates:
            for pos in slate.positions:
                existing = seen.get(pos.ticker)
                if existing is None or pos.confidence > existing.confidence:
                    seen[pos.ticker] = pos

        # Re-enforce bucket caps on the merged set
        self.reset_cycle_utilization()
        accepted: list[PositionAllocation] = []
        for pos in sorted(seen.values(), key=lambda p: p.confidence, reverse=True):
            if self._fits_in_bucket(pos):
                self._apply_utilization(pos, self._nav)
                accepted.append(pos)
            else:
                logger.info(
                    "[AllocationEngine] aggregate_runs: REJECTED %s — bucket %s full after merge",
                    pos.ticker, pos.bucket,
                )

        merged = AllocationSlate(
            positions=accepted,
            kill_switch_triggered=False,
            beta_corridor=slates[-1].beta_corridor,
            leverage_multiplier=slates[-1].leverage_multiplier,
            cycle_number=self.cycle_number,
            phase=CyclePhase.AGGREGATING.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        merged.total_ig_equity = self._utilization[BucketType.IG_EQUITY]
        merged.total_hy_equity = self._utilization[BucketType.HY_EQUITY]
        merged.total_distressed_equity = self._utilization[BucketType.DISTRESSED_EQUITY]
        merged.total_tltw_cashflow = self._utilization[BucketType.TLTW_CASHFLOW]
        merged.total_fi_macro = self._utilization[BucketType.FI_MACRO]
        merged.total_event_cvr = self._utilization[BucketType.EVENT_DRIVEN_CVR]
        merged.total_options_notional = (
            self._utilization[BucketType.OPTIONS_IG]
            + self._utilization[BucketType.OPTIONS_HY]
            + self._utilization[BucketType.OPTIONS_DISTRESSED]
        )
        merged.total_futures_beta = self._utilization[BucketType.FUTURES_BETA]
        merged.total_margin = self._utilization[BucketType.MARGIN]
        merged.total_money_market = self._utilization[BucketType.MONEY_MARKET]

        total_input = sum(len(s.positions) for s in slates)
        logger.info(
            "[AllocationEngine] aggregate_runs: %d runs → %d input positions → %d accepted after dedup+caps",
            len(slates), total_input, len(accepted),
        )
        return merged

    # ── update_rules ──────────────────────────────────────────────

    def update_rules(self, updates: dict) -> AllocationRules:
        """Apply a partial rules update dict and return the updated rules.

        Called by the allocation API router's POST /rules endpoint.
        Only updates fields present in the dict — all others unchanged.
        """
        for key, value in updates.items():
            if hasattr(self.rules, key):
                setattr(self.rules, key, value)
                logger.info("[AllocationEngine] Rule updated: %s = %s", key, value)
            else:
                logger.warning("[AllocationEngine] update_rules: unknown field %s — skipped", key)
        # Re-sync kill switch threshold if max_drawdown_kill_switch changed
        if "max_drawdown_kill_switch" in updates:
            self.kill_switch.threshold = self.rules.max_drawdown_kill_switch
        return self.rules

    # ── get_status ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return full engine status for the allocation /status endpoint.

        Includes bucket utilization, kill switch state, beta corridor,
        leverage multiplier, rules summary, and current NAV.
        """
        return {
            "nav": self._nav,
            "beta_corridor": {
                "corridor": self._beta_corridor.value,
                "leverage_multiplier": self._leverage_multiplier,
                "beta": 1.0,  # live beta injected by BetaCorridor component
            },
            "kill_switch": self.kill_switch.status(),
            "bucket_utilization": self.get_utilization(),
            "rules": self.rules.to_dict(),
            "cycle_number": self.cycle_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── apply_rules backtest kwarg passthrough ────────────────────

    def _apply_backtest_flag(self, backtest: bool) -> None:
        """Sync the backtest flag — called by universe_scan.run_universe()."""
        self.backtest = backtest
