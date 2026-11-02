"""ConvictionOverride — High-conviction signal bypass for position sizing.

Allows signals that pass ALL 8 eligibility gates to bypass normal position
sizing constraints and take larger positions (1.5x to 3.0x normal).

Gates (all must pass):
    1. Signal strength > 2.5 standard deviations
    2. ML consensus > 80% (at least 4/5 ML tiers agree)
    3. Macro regime alignment (signal direction matches regime)
    4. Sector momentum positive (longs) or negative (shorts)
    5. Liquidity sufficient (avg daily volume > $10M)
    6. No earnings within 5 trading days
    7. Correlation to existing positions < 0.7
    8. Risk budget available (portfolio drawdown < 5%)

Tiers:
    CONTROLLED  1.5x  max 3% NAV
    AGGRESSIVE  2.0x  max 5% NAV
    MAXIMUM     3.0x  max 8% NAV

Audit trail: every override decision gets UUID + JSONL logging.
"""

import uuid
import json
import time
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

try:
    from ..data.yahoo_data import get_adj_close, get_returns, get_market_stats
except ImportError:
    get_adj_close = None
    get_returns = None
    get_market_stats = None

try:
    from ..signals.macro_engine import MacroSnapshot, MarketRegime
except ImportError:
    MacroSnapshot = None
    MarketRegime = None

try:
    from ..execution.paper_broker import PaperBroker, Position, SignalType
except ImportError:
    PaperBroker = None
    Position = None
    SignalType = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNAL_STRENGTH_THRESHOLD = 2.5       # Standard deviations
ML_CONSENSUS_THRESHOLD = 0.80         # 80% agreement (8/10 tiers)
ML_TIER_COUNT = 10                    # MLVoteEnsemble tiers T1..T10
MIN_ML_AGREEMENT = 8                  # Minimum tiers that must agree (abstentions don't count)
LIQUIDITY_FLOOR_USD = 10_000_000      # $10M average daily volume
EARNINGS_BLACKOUT_DAYS = 5            # Trading days before earnings
CORRELATION_CAP = 0.70                # Max correlation to existing positions
DRAWDOWN_GATE = 0.05                  # Portfolio drawdown must be < 5%
MAX_ACTIVE_OVERRIDES = 3              # Portfolio-level cap on active overrides
OVERRIDE_EXPIRY_DAYS = 5              # Trading days before override expires
OVERRIDE_BUDGET_PCT = 0.15            # 15% of NAV reserved for overrides
DRAWDOWN_REDUCTION_THRESHOLD = 0.04   # Start reducing at 4% drawdown
DRAWDOWN_HARD_STOP = 0.08             # Hard stop at 8% drawdown

# Default audit log directory
AUDIT_LOG_DIR = Path("logs/conviction_overrides")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ConvictionTier(str, Enum):
    """Conviction override tiers with increasing position size multipliers."""
    CONTROLLED = "CONTROLLED"
    AGGRESSIVE = "AGGRESSIVE"
    MAXIMUM = "MAXIMUM"


class GateResult(str, Enum):
    """Individual gate pass/fail."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class OverrideStatus(str, Enum):
    """Lifecycle status of an override."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"
    REDUCED = "REDUCED"
    STOPPED_OUT = "STOPPED_OUT"


class SignalDirection(str, Enum):
    """Signal direction for regime alignment check."""
    LONG = "LONG"
    SHORT = "SHORT"


# ---------------------------------------------------------------------------
# Tier configurations
# ---------------------------------------------------------------------------
TIER_CONFIG = {
    ConvictionTier.CONTROLLED: {
        "size_multiplier": 1.5,
        "max_nav_pct": 0.03,
        "min_signal_strength": 2.5,
        "min_ml_consensus": 0.80,
        "description": "Controlled override: 1.5x size, max 3% NAV",
    },
    ConvictionTier.AGGRESSIVE: {
        "size_multiplier": 2.0,
        "max_nav_pct": 0.05,
        "min_signal_strength": 3.0,
        "min_ml_consensus": 0.85,
        "description": "Aggressive override: 2.0x size, max 5% NAV",
    },
    ConvictionTier.MAXIMUM: {
        "size_multiplier": 3.0,
        "max_nav_pct": 0.08,
        "min_signal_strength": 3.5,
        "min_ml_consensus": 0.90,
        "description": "Maximum override: 3.0x size, max 8% NAV",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GateEvaluation:
    """Result of a single eligibility gate check."""
    gate_number: int
    gate_name: str
    result: GateResult = GateResult.FAIL
    value: float = 0.0
    threshold: float = 0.0
    detail: str = ""
    timestamp: str = ""

    def passed(self) -> bool:
        return self.result == GateResult.PASS

    def to_dict(self) -> dict:
        return {
            "gate_number": self.gate_number,
            "gate_name": self.gate_name,
            "result": self.result.value,
            "value": round(self.value, 6),
            "threshold": round(self.threshold, 6),
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class OverrideDecision:
    """Full override decision with gate-by-gate results and audit trail."""
    override_id: str = ""
    ticker: str = ""
    direction: SignalDirection = SignalDirection.LONG
    signal_strength: float = 0.0
    ml_consensus: float = 0.0
    tier: Optional[ConvictionTier] = None
    approved: bool = False
    gates: list = field(default_factory=list)
    gates_passed: int = 0
    gates_total: int = 8
    normal_position_size: float = 0.0
    override_position_size: float = 0.0
    max_position_size: float = 0.0
    size_multiplier: float = 1.0
    reason: str = ""
    timestamp: str = ""
    expiry_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "override_id": self.override_id,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "signal_strength": round(self.signal_strength, 4),
            "ml_consensus": round(self.ml_consensus, 4),
            "tier": self.tier.value if self.tier else None,
            "approved": self.approved,
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
            "gates": [g.to_dict() for g in self.gates],
            "normal_position_size": round(self.normal_position_size, 2),
            "override_position_size": round(self.override_position_size, 2),
            "max_position_size": round(self.max_position_size, 2),
            "size_multiplier": round(self.size_multiplier, 2),
            "reason": self.reason,
            "timestamp": self.timestamp,
            "expiry_timestamp": self.expiry_timestamp,
        }


@dataclass
class ActiveOverride:
    """Tracks a currently active conviction override."""
    override_id: str = ""
    ticker: str = ""
    direction: SignalDirection = SignalDirection.LONG
    tier: ConvictionTier = ConvictionTier.CONTROLLED
    status: OverrideStatus = OverrideStatus.ACTIVE
    entry_price: float = 0.0
    current_price: float = 0.0
    position_size: float = 0.0
    original_size: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_timestamp: str = ""
    expiry_timestamp: str = ""
    last_update: str = ""
    reduction_count: int = 0
    peak_price: float = 0.0
    trough_price: float = 0.0

    def is_expired(self) -> bool:
        """Check if override has expired based on trading day count."""
        if not self.expiry_timestamp:
            return False
        try:
            expiry = datetime.fromisoformat(self.expiry_timestamp)
            return datetime.now() >= expiry
        except (ValueError, TypeError):
            return False

    def compute_pnl(self) -> float:
        """Compute unrealized P&L."""
        if self.direction == SignalDirection.LONG:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.position_size
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.position_size
        return self.unrealized_pnl

    def to_dict(self) -> dict:
        return {
            "override_id": self.override_id,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "tier": self.tier.value,
            "status": self.status.value,
            "entry_price": round(self.entry_price, 4),
            "current_price": round(self.current_price, 4),
            "position_size": round(self.position_size, 2),
            "original_size": round(self.original_size, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "entry_timestamp": self.entry_timestamp,
            "expiry_timestamp": self.expiry_timestamp,
            "last_update": self.last_update,
            "reduction_count": self.reduction_count,
            "peak_price": round(self.peak_price, 4),
            "trough_price": round(self.trough_price, 4),
        }


@dataclass
class OverrideBudget:
    """Budget tracking for conviction overrides, separate from normal risk."""
    total_budget_pct: float = OVERRIDE_BUDGET_PCT
    used_budget_pct: float = 0.0
    remaining_budget_pct: float = OVERRIDE_BUDGET_PCT
    active_count: int = 0
    max_active: int = MAX_ACTIVE_OVERRIDES
    total_override_exposure: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None

    def has_capacity(self) -> bool:
        """Check if there is room for another override."""
        return (
            self.active_count < self.max_active
            and self.remaining_budget_pct > 0.01
        )

    def to_dict(self) -> dict:
        return {
            "total_budget_pct": round(self.total_budget_pct, 4),
            "used_budget_pct": round(self.used_budget_pct, 4),
            "remaining_budget_pct": round(self.remaining_budget_pct, 4),
            "active_count": self.active_count,
            "max_active": self.max_active,
            "total_override_exposure": round(self.total_override_exposure, 2),
        }


@dataclass
class OverridePerformance:
    """Aggregate performance tracking for conviction overrides."""
    total_overrides: int = 0
    approved_overrides: int = 0
    rejected_overrides: int = 0
    active_overrides: int = 0
    expired_overrides: int = 0
    closed_overrides: int = 0
    stopped_out: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_override: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    best_override_pnl: float = 0.0
    worst_override_pnl: float = 0.0
    avg_holding_period_days: float = 0.0
    tier_breakdown: dict = field(default_factory=dict)

    def update_stats(self, closed_overrides: list):
        """Recalculate aggregate stats from closed overrides."""
        if not closed_overrides:
            return
        pnls = [o.realized_pnl + o.unrealized_pnl for o in closed_overrides]
        self.total_pnl = sum(pnls)
        self.win_count = sum(1 for p in pnls if p > 0)
        self.loss_count = sum(1 for p in pnls if p <= 0)
        total = self.win_count + self.loss_count
        self.win_rate = self.win_count / total if total > 0 else 0.0
        self.avg_pnl_per_override = self.total_pnl / len(pnls) if pnls else 0.0
        self.best_override_pnl = max(pnls) if pnls else 0.0
        self.worst_override_pnl = min(pnls) if pnls else 0.0

        # Tier breakdown
        tier_pnl = {}
        tier_count = {}
        for o in closed_overrides:
            t = o.tier.value
            tier_pnl[t] = tier_pnl.get(t, 0.0) + o.realized_pnl + o.unrealized_pnl
            tier_count[t] = tier_count.get(t, 0) + 1
        self.tier_breakdown = {
            t: {"count": tier_count.get(t, 0), "total_pnl": round(tier_pnl.get(t, 0.0), 2)}
            for t in [ct.value for ct in ConvictionTier]
        }

    def to_dict(self) -> dict:
        return {
            "total_overrides": self.total_overrides,
            "approved_overrides": self.approved_overrides,
            "rejected_overrides": self.rejected_overrides,
            "active_overrides": self.active_overrides,
            "expired_overrides": self.expired_overrides,
            "closed_overrides": self.closed_overrides,
            "stopped_out": self.stopped_out,
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl_per_override": round(self.avg_pnl_per_override, 2),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_rate, 4),
            "best_override_pnl": round(self.best_override_pnl, 2),
            "worst_override_pnl": round(self.worst_override_pnl, 2),
            "avg_holding_period_days": round(self.avg_holding_period_days, 2),
            "tier_breakdown": self.tier_breakdown,
        }


# ---------------------------------------------------------------------------
# Eligibility Gate Evaluators
# ---------------------------------------------------------------------------
class EligibilityGates:
    """Evaluates all 8 eligibility gates for conviction override.

    (These 8 override-eligibility gates are separate from the DecisionMatrix
    4-gate entry logic.)
    """

    def __init__(self):
        self._gate_names = {
            1: "signal_strength",
            2: "ml_consensus",
            3: "macro_regime_alignment",
            4: "sector_momentum",
            5: "liquidity_sufficient",
            6: "no_earnings_blackout",
            7: "correlation_check",
            8: "risk_budget_available",
        }

    def evaluate_all(
        self,
        ticker: str,
        direction: SignalDirection,
        signal_strength: float,
        ml_votes: dict,
        macro_snapshot: Optional[object] = None,
        sector_momentum: float = 0.0,
        avg_daily_volume_usd: float = 0.0,
        days_to_earnings: Optional[int] = None,
        correlation_to_portfolio: float = 0.0,
        portfolio_drawdown: float = 0.0,
    ) -> list:
        """Run all 8 eligibility gates and return list of GateEvaluation."""
        now = datetime.now().isoformat()
        gates = []

        # Gate 1: Signal strength
        gates.append(self._gate_signal_strength(signal_strength, now))

        # Gate 2: ML consensus
        gates.append(self._gate_ml_consensus(ml_votes, direction, now))

        # Gate 3: Macro regime alignment
        gates.append(self._gate_macro_alignment(direction, macro_snapshot, now))

        # Gate 4: Sector momentum
        gates.append(self._gate_sector_momentum(direction, sector_momentum, now))

        # Gate 5: Liquidity
        gates.append(self._gate_liquidity(avg_daily_volume_usd, now))

        # Gate 6: Earnings blackout
        gates.append(self._gate_earnings_blackout(days_to_earnings, now))

        # Gate 7: Correlation
        gates.append(self._gate_correlation(correlation_to_portfolio, now))

        # Gate 8: Risk budget
        gates.append(self._gate_risk_budget(portfolio_drawdown, now))

        return gates

    def _gate_signal_strength(self, strength: float, ts: str) -> GateEvaluation:
        """Gate 1: Signal strength must exceed 2.5 standard deviations."""
        gate = GateEvaluation(
            gate_number=1,
            gate_name="signal_strength",
            value=strength,
            threshold=SIGNAL_STRENGTH_THRESHOLD,
            timestamp=ts,
        )
        if strength > SIGNAL_STRENGTH_THRESHOLD:
            gate.result = GateResult.PASS
            gate.detail = f"Signal {strength:.2f}σ exceeds threshold {SIGNAL_STRENGTH_THRESHOLD}σ"
        else:
            gate.result = GateResult.FAIL
            gate.detail = f"Signal {strength:.2f}σ below threshold {SIGNAL_STRENGTH_THRESHOLD}σ"
        return gate

    def _gate_ml_consensus(
        self, ml_votes: dict, direction: SignalDirection, ts: str
    ) -> GateEvaluation:
        """Gate 2: ML consensus — at least MIN_ML_AGREEMENT (8) of the
        ML_TIER_COUNT (10) MLVoteEnsemble tiers must agree with the direction.

        The denominator is always the full 10 tiers: a tier that is missing
        or votes 0 (abstains) counts as NOT agreeing, so a partial vote dict
        cannot inflate consensus.
        """
        gate = GateEvaluation(
            gate_number=2,
            gate_name="ml_consensus",
            threshold=ML_CONSENSUS_THRESHOLD,
            timestamp=ts,
        )
        if not ml_votes:
            gate.result = GateResult.FAIL
            gate.detail = "No ML votes available"
            gate.value = 0.0
            return gate

        total_tiers = max(len(ml_votes), ML_TIER_COUNT)
        if len(ml_votes) == 0:
            gate.result = GateResult.FAIL
            gate.detail = "Empty ML vote dict"
            gate.value = 0.0
            return gate

        # Count tiers agreeing with direction
        if direction == SignalDirection.LONG:
            agreeing = sum(1 for v in ml_votes.values() if v > 0)
        else:
            agreeing = sum(1 for v in ml_votes.values() if v < 0)

        consensus = agreeing / total_tiers
        gate.value = consensus

        if agreeing >= MIN_ML_AGREEMENT and consensus >= ML_CONSENSUS_THRESHOLD:
            gate.result = GateResult.PASS
            gate.detail = f"ML consensus {consensus:.0%} ({agreeing}/{total_tiers} tiers agree)"
        else:
            gate.result = GateResult.FAIL
            gate.detail = (f"ML consensus {consensus:.0%} ({agreeing}/{total_tiers}) below "
                           f"{MIN_ML_AGREEMENT}/{ML_TIER_COUNT} tiers")
        return gate

    def _gate_macro_alignment(
        self, direction: SignalDirection, macro: Optional[object], ts: str
    ) -> GateEvaluation:
        """Gate 3: Signal direction must align with macro regime."""
        gate = GateEvaluation(
            gate_number=3,
            gate_name="macro_regime_alignment",
            threshold=1.0,
            timestamp=ts,
        )
        if macro is None:
            gate.result = GateResult.SKIP
            gate.detail = "No macro snapshot available; gate skipped (treated as pass)"
            gate.value = 1.0
            # Treat skip as pass for graceful degradation
            gate.result = GateResult.PASS
            return gate

        regime = getattr(macro, "regime", None)
        if regime is None:
            gate.result = GateResult.PASS
            gate.detail = "No regime attribute on macro snapshot"
            gate.value = 1.0
            return gate

        regime_val = regime.value if hasattr(regime, "value") else str(regime)

        # Bull/Transition regimes align with LONG; Bear/Stress align with SHORT
        bullish_regimes = {"BULL", "TRANSITION"}
        bearish_regimes = {"BEAR", "STRESS", "CRASH"}

        if direction == SignalDirection.LONG and regime_val in bullish_regimes:
            gate.result = GateResult.PASS
            gate.value = 1.0
            gate.detail = f"LONG signal aligns with {regime_val} regime"
        elif direction == SignalDirection.SHORT and regime_val in bearish_regimes:
            gate.result = GateResult.PASS
            gate.value = 1.0
            gate.detail = f"SHORT signal aligns with {regime_val} regime"
        else:
            gate.result = GateResult.FAIL
            gate.value = 0.0
            gate.detail = f"{direction.value} signal misaligned with {regime_val} regime"
        return gate

    def _gate_sector_momentum(
        self, direction: SignalDirection, momentum: float, ts: str
    ) -> GateEvaluation:
        """Gate 4: Sector momentum must be positive for longs, negative for shorts."""
        gate = GateEvaluation(
            gate_number=4,
            gate_name="sector_momentum",
            value=momentum,
            threshold=0.0,
            timestamp=ts,
        )
        if direction == SignalDirection.LONG and momentum > 0:
            gate.result = GateResult.PASS
            gate.detail = f"Sector momentum {momentum:+.4f} positive (LONG aligned)"
        elif direction == SignalDirection.SHORT and momentum < 0:
            gate.result = GateResult.PASS
            gate.detail = f"Sector momentum {momentum:+.4f} negative (SHORT aligned)"
        else:
            gate.result = GateResult.FAIL
            gate.detail = (
                f"Sector momentum {momentum:+.4f} misaligned with {direction.value}"
            )
        return gate

    def _gate_liquidity(self, avg_volume_usd: float, ts: str) -> GateEvaluation:
        """Gate 5: Average daily dollar volume must exceed $10M."""
        gate = GateEvaluation(
            gate_number=5,
            gate_name="liquidity_sufficient",
            value=avg_volume_usd,
            threshold=LIQUIDITY_FLOOR_USD,
            timestamp=ts,
        )
        if avg_volume_usd >= LIQUIDITY_FLOOR_USD:
            gate.result = GateResult.PASS
            gate.detail = f"Avg daily volume ${avg_volume_usd/1e6:.1f}M >= ${LIQUIDITY_FLOOR_USD/1e6:.0f}M"
        else:
            gate.result = GateResult.FAIL
            gate.detail = f"Avg daily volume ${avg_volume_usd/1e6:.1f}M < ${LIQUIDITY_FLOOR_USD/1e6:.0f}M"
        return gate

    def _gate_earnings_blackout(
        self, days_to_earnings: Optional[int], ts: str
    ) -> GateEvaluation:
        """Gate 6: No earnings within 5 trading days."""
        gate = GateEvaluation(
            gate_number=6,
            gate_name="no_earnings_blackout",
            threshold=EARNINGS_BLACKOUT_DAYS,
            timestamp=ts,
        )
        if days_to_earnings is None:
            # If we cannot determine earnings date, pass with caveat
            gate.result = GateResult.PASS
            gate.value = float(EARNINGS_BLACKOUT_DAYS + 1)
            gate.detail = "Earnings date unknown; gate passed with caveat"
            return gate

        gate.value = float(days_to_earnings)
        if days_to_earnings > EARNINGS_BLACKOUT_DAYS:
            gate.result = GateResult.PASS
            gate.detail = f"Earnings in {days_to_earnings} days > {EARNINGS_BLACKOUT_DAYS} day blackout"
        else:
            gate.result = GateResult.FAIL
            gate.detail = f"Earnings in {days_to_earnings} days within {EARNINGS_BLACKOUT_DAYS} day blackout"
        return gate

    def _gate_correlation(self, correlation: float, ts: str) -> GateEvaluation:
        """Gate 7: Correlation to existing positions must be < 0.7."""
        gate = GateEvaluation(
            gate_number=7,
            gate_name="correlation_check",
            value=correlation,
            threshold=CORRELATION_CAP,
            timestamp=ts,
        )
        if correlation < CORRELATION_CAP:
            gate.result = GateResult.PASS
            gate.detail = f"Correlation {correlation:.3f} < cap {CORRELATION_CAP}"
        else:
            gate.result = GateResult.FAIL
            gate.detail = f"Correlation {correlation:.3f} >= cap {CORRELATION_CAP}"
        return gate

    def _gate_risk_budget(self, drawdown: float, ts: str) -> GateEvaluation:
        """Gate 8: Portfolio drawdown must be < 5%."""
        gate = GateEvaluation(
            gate_number=8,
            gate_name="risk_budget_available",
            value=drawdown,
            threshold=DRAWDOWN_GATE,
            timestamp=ts,
        )
        if drawdown < DRAWDOWN_GATE:
            gate.result = GateResult.PASS
            gate.detail = f"Portfolio drawdown {drawdown:.2%} < gate {DRAWDOWN_GATE:.0%}"
        else:
            gate.result = GateResult.FAIL
            gate.detail = f"Portfolio drawdown {drawdown:.2%} >= gate {DRAWDOWN_GATE:.0%}"
        return gate


# ---------------------------------------------------------------------------
# Conviction Tier Classifier
# ---------------------------------------------------------------------------
class TierClassifier:
    """Determines which conviction tier a signal qualifies for."""

    def classify(
        self,
        signal_strength: float,
        ml_consensus: float,
        gates_passed: int,
        gates_total: int = 8,
    ) -> Optional[ConvictionTier]:
        """Classify into CONTROLLED, AGGRESSIVE, or MAXIMUM tier.

        All 8 eligibility gates must pass. Tier is determined by signal strength
        and ML consensus thresholds.
        """
        if gates_passed < gates_total:
            return None

        # Check from highest tier down
        for tier in [ConvictionTier.MAXIMUM, ConvictionTier.AGGRESSIVE, ConvictionTier.CONTROLLED]:
            config = TIER_CONFIG[tier]
            if (signal_strength >= config["min_signal_strength"]
                    and ml_consensus >= config["min_ml_consensus"]):
                return tier

        # All gates passed but doesn't meet minimum tier thresholds
        return None

    def get_size_multiplier(self, tier: ConvictionTier) -> float:
        """Return position size multiplier for the given tier."""
        return TIER_CONFIG[tier]["size_multiplier"]

    def get_max_nav_pct(self, tier: ConvictionTier) -> float:
        """Return maximum NAV percentage for the given tier."""
        return TIER_CONFIG[tier]["max_nav_pct"]

    def compute_override_size(
        self,
        tier: ConvictionTier,
        normal_size: float,
        nav: float,
    ) -> float:
        """Compute the override position size, capped by tier NAV limit."""
        config = TIER_CONFIG[tier]
        multiplied_size = normal_size * config["size_multiplier"]
        max_size = nav * config["max_nav_pct"]
        return min(multiplied_size, max_size)


# ---------------------------------------------------------------------------
# Correlation Analyzer
# ---------------------------------------------------------------------------
class CorrelationAnalyzer:
    """Computes correlation between a candidate ticker and existing positions."""

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days

    def compute_max_correlation(
        self,
        ticker: str,
        existing_tickers: list,
    ) -> float:
        """Return the maximum pairwise correlation between ticker and
        any existing position. Returns 0.0 if no positions or no data.
        """
        if not existing_tickers:
            return 0.0

        all_tickers = [ticker] + list(existing_tickers)
        start = (pd.Timestamp.now() - pd.Timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")

        try:
            if get_returns is None:
                return 0.0
            returns = get_returns(all_tickers, start=start)
            if returns.empty or ticker not in returns.columns:
                return 0.0

            corr_matrix = returns.corr()
            if ticker not in corr_matrix.index:
                return 0.0

            correlations = []
            for et in existing_tickers:
                if et in corr_matrix.columns and et != ticker:
                    c = abs(corr_matrix.loc[ticker, et])
                    if not np.isnan(c):
                        correlations.append(c)

            return max(correlations) if correlations else 0.0
        except Exception:
            return 0.0

    def compute_override_cross_correlation(
        self,
        active_overrides: list,
        candidate_ticker: str,
    ) -> float:
        """Compute max correlation between candidate and all active overrides."""
        override_tickers = [o.ticker for o in active_overrides if o.ticker != candidate_ticker]
        if not override_tickers:
            return 0.0
        return self.compute_max_correlation(candidate_ticker, override_tickers)


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------
class AuditLogger:
    """JSONL audit trail for all override decisions."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or AUDIT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / f"overrides_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log_decision(self, decision: OverrideDecision):
        """Write a single override decision to the JSONL audit file."""
        entry = decision.to_dict()
        entry["_log_type"] = "DECISION"
        entry["_log_ts"] = datetime.now().isoformat()
        self._write_entry(entry)

    def log_activation(self, override: ActiveOverride):
        """Log when an override becomes active."""
        entry = override.to_dict()
        entry["_log_type"] = "ACTIVATION"
        entry["_log_ts"] = datetime.now().isoformat()
        self._write_entry(entry)

    def log_expiry(self, override: ActiveOverride):
        """Log when an override expires."""
        entry = override.to_dict()
        entry["_log_type"] = "EXPIRY"
        entry["_log_ts"] = datetime.now().isoformat()
        self._write_entry(entry)

    def log_reduction(self, override: ActiveOverride, reason: str):
        """Log when an override position is reduced."""
        entry = override.to_dict()
        entry["_log_type"] = "REDUCTION"
        entry["_log_ts"] = datetime.now().isoformat()
        entry["reduction_reason"] = reason
        self._write_entry(entry)

    def log_close(self, override: ActiveOverride, reason: str):
        """Log when an override is closed out."""
        entry = override.to_dict()
        entry["_log_type"] = "CLOSE"
        entry["_log_ts"] = datetime.now().isoformat()
        entry["close_reason"] = reason
        self._write_entry(entry)

    def log_performance_snapshot(self, performance: OverridePerformance):
        """Log a periodic performance snapshot."""
        entry = performance.to_dict()
        entry["_log_type"] = "PERFORMANCE_SNAPSHOT"
        entry["_log_ts"] = datetime.now().isoformat()
        self._write_entry(entry)

    def _write_entry(self, entry: dict):
        """Append a JSON line to the audit file."""
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Never crash on logging failure

    def read_audit_log(self, date_str: Optional[str] = None) -> list:
        """Read back audit entries for a given date."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        path = self.log_dir / f"overrides_{date_str}.jsonl"
        entries = []
        if path.exists():
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except Exception:
                pass
        return entries


# ---------------------------------------------------------------------------
# Override Manager
# ---------------------------------------------------------------------------
class OverrideManager:
    """Central manager for conviction override lifecycle.

    Tracks active overrides, enforces portfolio-level limits,
    handles time-decay expiry, and integrates with risk controls.
    """

    def __init__(
        self,
        nav: float = 1_000_000.0,
        max_active: int = MAX_ACTIVE_OVERRIDES,
        expiry_days: int = OVERRIDE_EXPIRY_DAYS,
        override_budget_pct: float = OVERRIDE_BUDGET_PCT,
        log_dir: Optional[Path] = None,
    ):
        self.nav = nav
        self.max_active = max_active
        self.expiry_days = expiry_days
        self.override_budget_pct = override_budget_pct

        self._gates = EligibilityGates()
        self._classifier = TierClassifier()
        self._correlator = CorrelationAnalyzer()
        self._logger = AuditLogger(log_dir=log_dir)

        self._active_overrides: dict[str, ActiveOverride] = {}
        self._closed_overrides: list[ActiveOverride] = []
        self._decision_history: list[OverrideDecision] = []
        self._performance = OverridePerformance()
        self._budget = OverrideBudget(
            total_budget_pct=override_budget_pct,
            max_active=max_active,
        )
        self._peak_nav = nav

    # --- Core evaluation ---------------------------------------------------

    def evaluate_override(
        self,
        ticker: str,
        direction: SignalDirection,
        signal_strength: float,
        ml_votes: dict,
        normal_position_size: float,
        macro_snapshot: Optional[object] = None,
        sector_momentum: float = 0.0,
        avg_daily_volume_usd: float = 0.0,
        days_to_earnings: Optional[int] = None,
        existing_position_tickers: Optional[list] = None,
        portfolio_drawdown: float = 0.0,
    ) -> OverrideDecision:
        """Evaluate whether a signal qualifies for conviction override.

        Runs all 8 eligibility gates, classifies tier, computes override size,
        and logs the decision.
        """
        decision = OverrideDecision(
            override_id=str(uuid.uuid4()),
            ticker=ticker,
            direction=direction,
            signal_strength=signal_strength,
            normal_position_size=normal_position_size,
            timestamp=datetime.now().isoformat(),
        )

        # Compute correlation to existing positions
        all_existing = list(existing_position_tickers or [])
        all_existing.extend([o.ticker for o in self._active_overrides.values()])
        all_existing = list(set(t for t in all_existing if t != ticker))
        correlation = self._correlator.compute_max_correlation(ticker, all_existing)

        # Cross-correlation with other active overrides
        override_corr = self._correlator.compute_override_cross_correlation(
            list(self._active_overrides.values()), ticker,
        )
        effective_correlation = max(correlation, override_corr)

        # Run all 8 eligibility gates
        gates = self._gates.evaluate_all(
            ticker=ticker,
            direction=direction,
            signal_strength=signal_strength,
            ml_votes=ml_votes,
            macro_snapshot=macro_snapshot,
            sector_momentum=sector_momentum,
            avg_daily_volume_usd=avg_daily_volume_usd,
            days_to_earnings=days_to_earnings,
            correlation_to_portfolio=effective_correlation,
            portfolio_drawdown=portfolio_drawdown,
        )
        decision.gates = gates
        decision.gates_passed = sum(1 for g in gates if g.passed())
        decision.gates_total = len(gates)

        # Extract ML consensus from gate evaluation
        ml_gate = next((g for g in gates if g.gate_number == 2), None)
        decision.ml_consensus = ml_gate.value if ml_gate else 0.0

        # Check budget capacity
        if not self._budget.has_capacity():
            decision.approved = False
            decision.reason = (
                f"Override budget exhausted: {self._budget.active_count}/{self._budget.max_active} "
                f"active, {self._budget.remaining_budget_pct:.1%} remaining"
            )
            self._log_and_record(decision)
            return decision

        # Classify tier
        tier = self._classifier.classify(
            signal_strength=signal_strength,
            ml_consensus=decision.ml_consensus,
            gates_passed=decision.gates_passed,
            gates_total=decision.gates_total,
        )

        if tier is None:
            decision.approved = False
            failed_gates = [g for g in gates if not g.passed()]
            if failed_gates:
                details = "; ".join(f"G{g.gate_number}:{g.detail}" for g in failed_gates)
                decision.reason = f"Gates failed: {details}"
            else:
                decision.reason = "All gates passed but signal/consensus below minimum tier thresholds"
            self._log_and_record(decision)
            return decision

        # Compute override position size
        decision.tier = tier
        decision.size_multiplier = self._classifier.get_size_multiplier(tier)
        decision.override_position_size = self._classifier.compute_override_size(
            tier=tier,
            normal_size=normal_position_size,
            nav=self.nav,
        )
        decision.max_position_size = self.nav * self._classifier.get_max_nav_pct(tier)

        # Check against remaining budget
        override_nav_pct = decision.override_position_size / self.nav if self.nav > 0 else 1.0
        if override_nav_pct > self._budget.remaining_budget_pct:
            # Scale down to fit within remaining budget
            decision.override_position_size = self.nav * self._budget.remaining_budget_pct
            decision.reason = f"Size scaled to fit remaining budget ({self._budget.remaining_budget_pct:.1%})"

        # Compute expiry
        expiry_dt = datetime.now() + timedelta(days=self.expiry_days * 1.5)
        decision.expiry_timestamp = expiry_dt.isoformat()

        decision.approved = True
        if not decision.reason:
            decision.reason = (
                f"Override approved: {tier.value} tier, {decision.size_multiplier}x size, "
                f"all {decision.gates_passed}/{decision.gates_total} gates passed"
            )

        self._log_and_record(decision)
        return decision

    def activate_override(
        self,
        decision: OverrideDecision,
        entry_price: float,
    ) -> Optional[ActiveOverride]:
        """Activate an approved override, creating an ActiveOverride record."""
        if not decision.approved or decision.tier is None:
            return None

        if len(self._active_overrides) >= self.max_active:
            return None

        override = ActiveOverride(
            override_id=decision.override_id,
            ticker=decision.ticker,
            direction=decision.direction,
            tier=decision.tier,
            status=OverrideStatus.ACTIVE,
            entry_price=entry_price,
            current_price=entry_price,
            position_size=decision.override_position_size,
            original_size=decision.override_position_size,
            entry_timestamp=datetime.now().isoformat(),
            expiry_timestamp=decision.expiry_timestamp,
            last_update=datetime.now().isoformat(),
            peak_price=entry_price,
            trough_price=entry_price,
        )

        self._active_overrides[override.override_id] = override
        self._update_budget()
        self._logger.log_activation(override)
        self._performance.approved_overrides += 1
        self._performance.active_overrides = len(self._active_overrides)
        self._performance.total_overrides += 1

        return override

    # --- Lifecycle management -----------------------------------------------

    def update_prices(self, price_map: dict):
        """Update current prices for all active overrides.

        price_map: {ticker: current_price}
        """
        now = datetime.now().isoformat()
        for oid, override in list(self._active_overrides.items()):
            if override.ticker in price_map:
                price = price_map[override.ticker]
                override.current_price = price
                override.last_update = now
                override.peak_price = max(override.peak_price, price)
                override.trough_price = min(override.trough_price, price)
                override.compute_pnl()

    def check_expiries(self) -> list:
        """Check and expire overrides that have exceeded their time limit."""
        expired = []
        for oid, override in list(self._active_overrides.items()):
            if override.is_expired():
                override.status = OverrideStatus.EXPIRED
                override.compute_pnl()
                self._logger.log_expiry(override)
                self._closed_overrides.append(override)
                del self._active_overrides[oid]
                expired.append(override)
                self._performance.expired_overrides += 1

        if expired:
            self._update_budget()
            self._performance.active_overrides = len(self._active_overrides)
            self._performance.update_stats(self._closed_overrides)

        return expired

    def close_override(self, override_id: str, reason: str = "manual") -> Optional[ActiveOverride]:
        """Manually close an active override."""
        if override_id not in self._active_overrides:
            return None

        override = self._active_overrides.pop(override_id)
        override.status = OverrideStatus.CLOSED
        override.compute_pnl()
        self._logger.log_close(override, reason)
        self._closed_overrides.append(override)
        self._update_budget()
        self._performance.closed_overrides += 1
        self._performance.active_overrides = len(self._active_overrides)
        self._performance.update_stats(self._closed_overrides)

        return override

    def check_drawdown_protection(self, current_nav: float) -> list:
        """Auto-reduce or close overrides if portfolio drawdown exceeds thresholds.

        Returns list of overrides that were reduced or stopped out.
        """
        affected = []
        self._peak_nav = max(self._peak_nav, current_nav)
        drawdown = 1.0 - (current_nav / self._peak_nav) if self._peak_nav > 0 else 0.0

        if drawdown < DRAWDOWN_REDUCTION_THRESHOLD:
            return affected

        for oid, override in list(self._active_overrides.items()):
            if drawdown >= DRAWDOWN_HARD_STOP:
                # Hard stop: close everything
                override.status = OverrideStatus.STOPPED_OUT
                override.compute_pnl()
                self._logger.log_close(override, f"Hard stop: drawdown {drawdown:.2%}")
                self._closed_overrides.append(override)
                del self._active_overrides[oid]
                affected.append(override)
                self._performance.stopped_out += 1
            elif drawdown >= DRAWDOWN_REDUCTION_THRESHOLD and override.reduction_count == 0:
                # Soft reduction: cut position by 50% — ONCE per override
                # (previously this halved the position again on every call)
                reduction_factor = 0.5
                old_size = override.position_size
                override.position_size *= reduction_factor
                override.reduction_count += 1
                override.status = OverrideStatus.REDUCED
                override.last_update = datetime.now().isoformat()
                self._logger.log_reduction(
                    override,
                    f"Drawdown {drawdown:.2%} >= {DRAWDOWN_REDUCTION_THRESHOLD:.0%}; "
                    f"reduced {old_size:.0f} -> {override.position_size:.0f}",
                )
                affected.append(override)

        if affected:
            self._update_budget()
            self._performance.active_overrides = len(self._active_overrides)
            self._performance.update_stats(self._closed_overrides)

        return affected

    # --- Budget management --------------------------------------------------

    def _update_budget(self):
        """Recalculate override budget usage."""
        total_exposure = sum(o.position_size for o in self._active_overrides.values())
        used_pct = total_exposure / self.nav if self.nav > 0 else 0.0

        self._budget.active_count = len(self._active_overrides)
        self._budget.used_budget_pct = used_pct
        self._budget.remaining_budget_pct = max(0.0, self._budget.total_budget_pct - used_pct)
        self._budget.total_override_exposure = total_exposure

    def update_nav(self, new_nav: float):
        """Update NAV for dynamic sizing."""
        self.nav = new_nav
        self._peak_nav = max(self._peak_nav, new_nav)
        self._update_budget()

    # --- Logging helpers ---------------------------------------------------

    def _log_and_record(self, decision: OverrideDecision):
        """Log decision and add to history."""
        self._logger.log_decision(decision)
        self._decision_history.append(decision)
        self._performance.total_overrides += 1
        if not decision.approved:
            self._performance.rejected_overrides += 1

    # --- Queries ------------------------------------------------------------

    def get_active_overrides(self) -> list:
        """Return all currently active overrides."""
        return list(self._active_overrides.values())

    def get_active_override(self, override_id: str) -> Optional[ActiveOverride]:
        """Return a specific active override by ID."""
        return self._active_overrides.get(override_id)

    def get_closed_overrides(self) -> list:
        """Return all closed/expired overrides."""
        return list(self._closed_overrides)

    def get_decision_history(self) -> list:
        """Return all override decisions (approved and rejected)."""
        return list(self._decision_history)

    def get_budget(self) -> OverrideBudget:
        """Return current override budget state."""
        return self._budget

    def get_performance(self) -> OverridePerformance:
        """Return aggregate performance metrics."""
        self._performance.update_stats(self._closed_overrides)
        return self._performance

    def get_summary(self) -> dict:
        """Return a complete summary of override state."""
        return {
            "budget": self._budget.to_dict(),
            "performance": self._performance.to_dict(),
            "active_overrides": [o.to_dict() for o in self._active_overrides.values()],
            "active_count": len(self._active_overrides),
            "closed_count": len(self._closed_overrides),
            "decision_count": len(self._decision_history),
            "nav": self.nav,
            "peak_nav": self._peak_nav,
        }

    def log_performance_snapshot(self):
        """Write a performance snapshot to the audit log."""
        self._performance.update_stats(self._closed_overrides)
        self._logger.log_performance_snapshot(self._performance)


# ---------------------------------------------------------------------------
# Convenience: compute signal strength in standard deviations
# ---------------------------------------------------------------------------
def compute_signal_strength(
    ticker: str,
    signal_value: float,
    lookback_days: int = 252,
) -> float:
    """Compute how many standard deviations a signal is from the mean.

    Uses historical returns to establish the distribution, then measures
    the signal value relative to that distribution.
    """
    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    try:
        if get_returns is None:
            return 0.0
        returns = get_returns(ticker, start=start)
        if returns.empty:
            return 0.0
        if isinstance(returns, pd.DataFrame):
            r = returns.iloc[:, 0].dropna()
        else:
            r = returns.dropna()
        if len(r) < 20:
            return 0.0
        mu = r.mean()
        sigma = r.std()
        if sigma == 0 or np.isnan(sigma):
            return 0.0
        return abs(signal_value - mu) / sigma
    except Exception:
        return 0.0


def estimate_avg_daily_volume(ticker: str, lookback_days: int = 30) -> float:
    """Estimate average daily dollar volume for a ticker.

    Uses price * volume over the lookback period.
    Returns dollar volume estimate.
    """
    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")
    try:
        if get_adj_close is None:
            return 0.0
        prices = get_adj_close(ticker, start=start)
        if prices.empty:
            return 0.0
        # Use price as proxy; actual volume would need OpenBB volume data
        if isinstance(prices, pd.DataFrame):
            latest_price = float(prices.iloc[-1].iloc[0])
        else:
            latest_price = float(prices.iloc[-1])

        # Heuristic: large caps ~$500M+ daily, mid caps ~$50M-500M
        # Without volume data, use price-based heuristic
        if latest_price > 200:
            return 100_000_000.0  # Likely liquid large cap
        elif latest_price > 50:
            return 50_000_000.0
        elif latest_price > 10:
            return 20_000_000.0
        else:
            return 5_000_000.0
    except Exception:
        return 0.0
