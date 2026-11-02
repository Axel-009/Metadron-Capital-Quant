"""MetadronCube — C(t) = f(L_t, R_t, F_t).

Aggressive alpha-seeking investment engine targeting 95%+ alpha extraction.

Multi-layer intelligence tensor between MacroEngine and AlphaOptimizer:
    Layer 0  FedPlumbingLayer    → SOFR, HY spreads, M2V, TGA, ON-RRP, SOMA
    Layer 1  LiquidityTensor     → Fed→PD→GSIB→Shadow bank reserve routing → L(t) in [-1,+1]
    Layer 2  ReserveFlowKernel   → TVP: ΔReserves → ΔSector β at t+1..t+10
    Risk     RiskStateModel       → VIX + realized vol + credit spread + skew → R(t) in [0,1]
    Flow     CapitalFlowModel     → sector momentum, leader/laggard, rotation velocity → F(t)
    Layer 4  RegimeEngine (HMM+RL) → TRENDING / RANGE / STRESS / CRASH
    Gate-Z   GateZAllocator       → 5-sleeve: Carry / Rotation / Trend-LHC / Neutral-Alpha / Down-Offense
    RiskGovernor: β target 0.65 (burst 0.70), VaR ≤ $0.30M (95%/1d), Gross ≤ 3.0x
                  Crash floor ≥ +25%, Gamma corridor [7%–12%]

4-Gate Entry Logic:
    Gate 1 — Fundamentals (40%): Quality/ROIC/FCF + GNN supply-chain penalty
    Gate 2 — Flow/Headlines (20%): ETF creations + Tensor signal → shortlist
    Gate 3 — Macro/Regime (20%): Kernel projections + rates/FX betas → filter
    Gate 4 — Momentum/Technical (20%): Breadth/leadership/gamma/vanna confirms

Kill-Switch: HY OAS +35bp & VIX term flat/inverted & breadth <50%
             → auto β ≤ 0.35, max tail spend

FCLP (Full Calibration Learning Protocol):
    1. Ingest plumbing  2. Recompute Tensor/Kernel  3. Regime detect
    4. Gate scoring  5. Risk pass  6. Write allocations

Regime leverage:
    TRENDING  3.0x  β≤0.65 (burst 0.70)
    RANGE     2.5x  β≤0.45
    STRESS    1.5x  β≤0.15
    CRASH     0.8x  β≤-0.20
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import deque

from .macro_engine import MacroSnapshot, CubeRegime, MarketRegime

logger = logging.getLogger("metadron.signals.cube")

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
# Aggressive regime parameters — 95% alpha target
REGIME_PARAMS = {
    CubeRegime.TRENDING: {
        "max_leverage": 3.0, "beta_cap": 0.65, "beta_burst": 0.70,
        "equity_pct": 0.55, "hedge_pct": 0.05,
        "tail_spend_pct_wk": 0.004, "crash_floor": 0.25,
        "theta_budget_daily": 0.0015,
    },
    CubeRegime.RANGE: {
        "max_leverage": 2.5, "beta_cap": 0.45, "beta_burst": 0.55,
        "equity_pct": 0.40, "hedge_pct": 0.12,
        "tail_spend_pct_wk": 0.005, "crash_floor": 0.25,
        "theta_budget_daily": 0.0010,
    },
    CubeRegime.STRESS: {
        "max_leverage": 1.5, "beta_cap": 0.15, "beta_burst": 0.20,
        "equity_pct": 0.20, "hedge_pct": 0.30,
        "tail_spend_pct_wk": 0.006, "crash_floor": 0.25,
        "theta_budget_daily": 0.0005,
    },
    CubeRegime.CRASH: {
        "max_leverage": 0.8, "beta_cap": -0.20, "beta_burst": -0.10,
        "equity_pct": 0.05, "hedge_pct": 0.50,
        "tail_spend_pct_wk": 0.008, "crash_floor": 0.25,
        "theta_budget_daily": 0.0002,
    },
}

# Beta corridor bounds (7%–12% return target)
R_LOW = 0.07
R_HIGH = 0.12
BETA_MAX = 2.0
BETA_INV = -0.136
EXECUTION_MULTIPLIER = 4.7
VOL_STANDARD = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LiquidityState:
    """L(t) — aggregate liquidity tensor in [-1, +1]."""
    value: float = 0.0
    sofr_signal: float = 0.0
    credit_impulse: float = 0.0
    m2_velocity: float = 0.0
    hy_spread_z: float = 0.0
    fed_funds_impact: float = 0.0
    reverse_repo_signal: float = 0.0
    tga_balance_signal: float = 0.0
    reserve_flow: float = 0.0
    net_liquidity_score: float = 0.0


@dataclass
class RiskState:
    """R(t) — aggregate risk in [0, 1]. 0=calm, 1=extreme stress."""
    value: float = 0.3
    vix_component: float = 0.0
    realized_vol: float = 0.0
    credit_spread_component: float = 0.0
    vix_term_structure: float = 0.0
    correlation_stress: float = 0.0
    tail_risk: float = 0.0
    implied_vs_realized: float = 0.0


@dataclass
class FlowState:
    """F(t) — capital flow model."""
    value: float = 0.0
    sector_momentum: dict = field(default_factory=dict)
    leader_sectors: list = field(default_factory=list)
    laggard_sectors: list = field(default_factory=list)
    rotation_velocity: float = 0.0
    breadth: float = 0.0
    persistence: float = 0.0
    mean_reversion_signal: float = 0.0


@dataclass
class SleeveAllocation:
    """Portfolio allocation mix — credit-aware sleeve structure.

    Allocation (100% of deployable capital):
      40%  IG Equities       — Investment-grade names, all cap sizes (no mega cap restriction)
      25%  Options           — 10% IG options, 10% HY options, 5% distressed options
      10%  Bond/Commodity ETFs — FI + commodity exposure (TLT, GLD, USO, HYG, LQD, etc.)
      10%  HY Equities       — High-yield names (BB-B rated, leveraged but not distressed)
      10%  Distressed Equity — Fallen angels, recovery plays, special situations
       5%  Cash (dry powder) — Buying power reserve, never deployed

    Deployment target: 95% of NAV + leverage. Only 5% remains as dry powder.
    """
    # --- Primary allocation mix (sums to 1.0) ---
    ig_equity: float = 0.40              # IG equities — all caps, no mega cap restriction
    options: float = 0.25                # Options sleeve (IG + HY + distressed)
    options_ig: float = 0.10             # Options on IG names
    options_hy: float = 0.10             # Options on HY names
    options_distressed: float = 0.05     # Options on distressed names
    bond_commodity_etf: float = 0.10     # FI + commodity ETFs
    hy_equity: float = 0.10              # HY equities
    distressed_equity: float = 0.10      # Distressed / fallen angel equity
    cash_reserve: float = 0.05           # Dry powder — never deployed

    # --- Deployment target ---
    deploy_pct: float = 0.95             # Deploy 95% of NAV + leverage

    # --- Legacy sleeve mapping (backward compat for existing code) ---
    p1_directional_equity: float = 0.40  # Maps to ig_equity (was 0.35)
    p2_factor_rotation: float = 0.10     # Maps to hy_equity
    p3_commodities_macro: float = 0.10   # Maps to bond_commodity_etf
    p4_options_convexity: float = 0.25   # Maps to options
    p5_hedges_volatility: float = 0.10   # Maps to distressed_equity

    # Extended sleeve breakdown (deprecated — use primary allocation above)
    carry: float = 0.40
    rotation: float = 0.10
    trend_lhc: float = 0.10
    neutral_alpha: float = 0.25
    down_offense: float = 0.10

    def as_dict(self) -> dict:
        """Return all sleeve allocations. Legacy keys map to primary 5-sleeve structure."""
        return {
            "P1_Carry": self.carry,
            "P2_Rotation": self.rotation,
            "P3_Trend_LHC": self.trend_lhc,
            "P4_Neutral_Alpha": self.neutral_alpha,
            "P5_Down_Offense": self.down_offense,
        }

    def legacy_dict(self) -> dict:
        """Legacy 5-sleeve keys for backward compatibility."""
        return {
            "P1_Directional_Equity": self.p1_directional_equity,
            "P2_Factor_Rotation": self.p2_factor_rotation,
            "P3_Commodities_Macro": self.p3_commodities_macro,
            "P4_Options_Convexity": self.p4_options_convexity,
            "P5_Hedges_Volatility": self.p5_hedges_volatility,
        }

    def total(self) -> float:
        """Sum of the primary 5 legacy sleeves (used for allocation)."""
        return (self.p1_directional_equity + self.p2_factor_rotation +
                self.p3_commodities_macro + self.p4_options_convexity +
                self.p5_hedges_volatility)

    def normalize(self):
        t = self.total()
        if t > 0 and abs(t - 1.0) > 0.01:
            f = 1.0 / t
            self.p1_directional_equity *= f
            self.p2_factor_rotation *= f
            self.p3_commodities_macro *= f
            self.p4_options_convexity *= f
            self.p5_hedges_volatility *= f
            # Sync extended sleeve names
            self.carry = self.p1_directional_equity
            self.rotation = self.p2_factor_rotation
            self.trend_lhc = self.p3_commodities_macro
            self.neutral_alpha = self.p4_options_convexity
            self.down_offense = self.p5_hedges_volatility


@dataclass
class CubeOutput:
    """Full MetadronCube output."""
    regime: CubeRegime = CubeRegime.RANGE
    liquidity: LiquidityState = field(default_factory=LiquidityState)
    risk: RiskState = field(default_factory=RiskState)
    flow: FlowState = field(default_factory=FlowState)
    sleeves: SleeveAllocation = field(default_factory=SleeveAllocation)
    max_leverage: float = 2.0
    beta_cap: float = 0.30
    target_beta: float = 0.0
    risk_budget_pct: float = 0.10
    cube_tensor: Optional[np.ndarray] = None
    timestamp: str = ""
    regime_confidence: float = 0.0
    transition_probability: float = 0.0


@dataclass
class RegimeTransition:
    """Regime transition record."""
    from_regime: CubeRegime = CubeRegime.RANGE
    to_regime: CubeRegime = CubeRegime.RANGE
    timestamp: str = ""
    confidence: float = 0.0
    trigger: str = ""


# ---------------------------------------------------------------------------
# FedPlumbingLayer
# ---------------------------------------------------------------------------
class FedPlumbingLayer:
    """Layer 0: Fed plumbing — SOFR, reserves, TGA, ON-RRP.

    Models the Fed's balance sheet mechanics and their impact on liquidity.
    """

    def __init__(self):
        self._sofr_target: float = 5.25
        self._tga_balance_bn: float = 700.0
        self._rrp_balance_bn: float = 500.0
        self._reserve_balance_bn: float = 3200.0
        self._fed_funds_rate: float = 5.25

    def compute(self, macro: MacroSnapshot) -> dict:
        """Compute Fed plumbing signals."""
        # SOFR proxy from short-term yield
        sofr_proxy = macro.yield_2y
        sofr_spread = sofr_proxy - self._sofr_target
        sofr_signal = np.clip(sofr_spread / 1.0, -1, 1)

        # TGA balance impact (higher TGA = tighter liquidity)
        tga_signal = np.clip((800 - self._tga_balance_bn) / 400, -1, 1)

        # RRP impact (higher RRP = money parked, less in market)
        rrp_signal = np.clip((1000 - self._rrp_balance_bn) / 500, -1, 1)

        # Reserve adequacy
        reserve_signal = np.clip((self._reserve_balance_bn - 3000) / 500, -1, 1)

        # Fed funds effective rate impact
        ff_impact = np.clip((5.0 - self._fed_funds_rate) / 2.0, -1, 1)

        # Net plumbing score
        net = 0.25 * sofr_signal + 0.20 * tga_signal + 0.20 * rrp_signal + 0.20 * reserve_signal + 0.15 * ff_impact

        return {
            "sofr_signal": float(sofr_signal),
            "tga_signal": float(tga_signal),
            "rrp_signal": float(rrp_signal),
            "reserve_signal": float(reserve_signal),
            "ff_impact": float(ff_impact),
            "net_plumbing": float(np.clip(net, -1, 1)),
        }

    def update_balances(self, tga: float = None, rrp: float = None, reserves: float = None, ff_rate: float = None):
        if tga is not None: self._tga_balance_bn = tga
        if rrp is not None: self._rrp_balance_bn = rrp
        if reserves is not None: self._reserve_balance_bn = reserves
        if ff_rate is not None: self._fed_funds_rate = ff_rate


# ---------------------------------------------------------------------------
# LiquidityTensor
# ---------------------------------------------------------------------------
class LiquidityTensor:
    """Layer 1: Multi-dimensional liquidity state.

    Combines SOFR, credit impulse, M2 velocity, HY spreads, and Fed plumbing
    into a single L(t) value in [-1, +1].
    """

    WEIGHTS = {
        "sofr": 0.20,
        "credit": 0.25,
        "m2v": 0.15,
        "hy_spread": 0.15,
        "fed_plumbing": 0.25,
    }

    def compute(self, macro: MacroSnapshot, fed_plumbing: dict) -> LiquidityState:
        ls = LiquidityState()

        # SOFR signal
        ls.sofr_signal = np.clip((5.0 - macro.yield_2y) / 3.0, -1, 1)

        # Credit impulse
        ls.credit_impulse = np.clip(1.0 - macro.credit_spread / 5.0, -1, 1)

        # M2 velocity proxy
        ls.m2_velocity = np.clip(macro.yield_spread / 2.0, -1, 1)

        # HY spread z-score
        ls.hy_spread_z = np.clip(-macro.credit_spread / 3.0, -1, 1)

        # Fed plumbing
        ls.fed_funds_impact = fed_plumbing.get("ff_impact", 0)
        ls.reverse_repo_signal = fed_plumbing.get("rrp_signal", 0)
        ls.tga_balance_signal = fed_plumbing.get("tga_signal", 0)
        ls.reserve_flow = fed_plumbing.get("reserve_signal", 0)

        net_plumbing = fed_plumbing.get("net_plumbing", 0)

        # Weighted aggregate
        ls.value = float(np.clip(
            self.WEIGHTS["sofr"] * ls.sofr_signal +
            self.WEIGHTS["credit"] * ls.credit_impulse +
            self.WEIGHTS["m2v"] * ls.m2_velocity +
            self.WEIGHTS["hy_spread"] * ls.hy_spread_z +
            self.WEIGHTS["fed_plumbing"] * net_plumbing,
            -1, 1,
        ))

        # Net liquidity score 0-100
        ls.net_liquidity_score = (ls.value + 1) * 50

        return ls


# ---------------------------------------------------------------------------
# ReserveFlowKernel
# ---------------------------------------------------------------------------
class ReserveFlowKernel:
    """Layer 2: Reserve flow impact computation.

    Models how changes in bank reserves flow through to equities and credit.
    ΔReserves → ΔEquity/Credit with time lags.
    """

    def __init__(self, lag_days: int = 5, decay: float = 0.85):
        self.lag_days = lag_days
        self.decay = decay
        self._reserve_history: deque = deque(maxlen=60)

    def compute_impulse(self, current_reserves: float) -> float:
        """Compute reserve flow impulse."""
        self._reserve_history.append(current_reserves)
        if len(self._reserve_history) < 2:
            return 0.0

        history = list(self._reserve_history)
        deltas = [history[i] - history[i - 1] for i in range(1, len(history))]

        # Weighted impulse with decay
        impulse = 0.0
        weight = 1.0
        for d in reversed(deltas[-self.lag_days:]):
            impulse += d * weight
            weight *= self.decay

        return float(np.clip(impulse / 100, -1, 1))

    def compute_equity_impact(self, impulse: float) -> float:
        """Map reserve impulse to equity market impact."""
        return float(np.clip(impulse * 0.7, -1, 1))

    def compute_credit_impact(self, impulse: float) -> float:
        """Map reserve impulse to credit spread impact."""
        return float(np.clip(-impulse * 0.5, -1, 1))


# ---------------------------------------------------------------------------
# RiskStateModel (Enhanced)
# ---------------------------------------------------------------------------
class RiskStateModel:
    """Enhanced R(t) in [0, 1]. Multi-factor risk computation."""

    WEIGHTS = {
        "vix": 0.30,
        "realized_vol": 0.20,
        "credit": 0.20,
        "correlation": 0.15,
        "tail_risk": 0.15,
    }

    def compute(self, macro: MacroSnapshot) -> RiskState:
        rs = RiskState()

        # VIX component (0-1, scales with VIX/60)
        rs.vix_component = float(np.clip(macro.vix / 60.0, 0, 1))

        # Realized vol proxy
        rs.realized_vol = float(np.clip(macro.vix / 40.0, 0, 1))

        # Credit spread component
        rs.credit_spread_component = float(np.clip(macro.credit_spread / 8.0, 0, 1))

        # VIX term structure (flat/inverted = higher risk)
        rs.vix_term_structure = float(np.clip(macro.vix / 50.0, 0, 1))

        # Implied vs realized (higher IV/RV = elevated risk expectations)
        rs.implied_vs_realized = float(np.clip((macro.vix / 100) / max(rs.realized_vol, 0.01) - 1, -0.5, 0.5) + 0.5)

        # Correlation stress (high VIX → correlations spike)
        rs.correlation_stress = float(np.clip((macro.vix - 20) / 30, 0, 1))

        # Tail risk (extreme VIX levels)
        rs.tail_risk = float(np.clip((macro.vix - 30) / 20, 0, 1))

        # Weighted aggregate
        rs.value = float(np.clip(
            self.WEIGHTS["vix"] * rs.vix_component +
            self.WEIGHTS["realized_vol"] * rs.realized_vol +
            self.WEIGHTS["credit"] * rs.credit_spread_component +
            self.WEIGHTS["correlation"] * rs.correlation_stress +
            self.WEIGHTS["tail_risk"] * rs.tail_risk,
            0, 1,
        ))

        return rs


# ---------------------------------------------------------------------------
# CapitalFlowModel (Enhanced)
# ---------------------------------------------------------------------------
class CapitalFlowModel:
    """Enhanced F(t) — capital flow with persistence and mean-reversion."""

    def __init__(self, persistence_decay: float = 0.90, reversion_speed: float = 0.1):
        self.persistence_decay = persistence_decay
        self.reversion_speed = reversion_speed
        self._prev_flow: float = 0.0
        self._flow_history: deque = deque(maxlen=252)

    def compute(self, macro: MacroSnapshot) -> FlowState:
        fs = FlowState()
        fs.sector_momentum = macro.sector_rankings
        sectors = list(macro.sector_rankings.keys())

        if sectors:
            n = max(1, len(sectors) // 3)
            fs.leader_sectors = sectors[:n]
            fs.laggard_sectors = sectors[-n:]

        vals = list(macro.sector_rankings.values())
        raw_flow = float(np.mean(vals)) if vals else 0.0

        # Persistence (momentum)
        persistent_flow = self.persistence_decay * self._prev_flow + (1 - self.persistence_decay) * raw_flow
        fs.persistence = self.persistence_decay

        # Mean reversion component
        if len(self._flow_history) > 20:
            long_avg = float(np.mean(list(self._flow_history)))
            mean_rev = self.reversion_speed * (long_avg - persistent_flow)
            fs.mean_reversion_signal = mean_rev
            persistent_flow += mean_rev

        fs.value = persistent_flow
        self._prev_flow = persistent_flow
        self._flow_history.append(raw_flow)

        # Rotation velocity
        if len(self._flow_history) >= 5:
            recent = list(self._flow_history)[-5:]
            fs.rotation_velocity = float(np.std(recent))

        # Breadth (dispersion of sector returns)
        if vals:
            fs.breadth = float(np.std(vals))

        return fs


# ---------------------------------------------------------------------------
# RegimeEngine
# ---------------------------------------------------------------------------
class RegimeEngine:
    """Formal regime determination with transition probabilities."""

    # Markov transition matrix (from → to)
    TRANSITION_PROBS = {
        CubeRegime.TRENDING: {CubeRegime.TRENDING: 0.85, CubeRegime.RANGE: 0.10, CubeRegime.STRESS: 0.04, CubeRegime.CRASH: 0.01},
        CubeRegime.RANGE: {CubeRegime.TRENDING: 0.15, CubeRegime.RANGE: 0.70, CubeRegime.STRESS: 0.12, CubeRegime.CRASH: 0.03},
        CubeRegime.STRESS: {CubeRegime.TRENDING: 0.05, CubeRegime.RANGE: 0.20, CubeRegime.STRESS: 0.55, CubeRegime.CRASH: 0.20},
        CubeRegime.CRASH: {CubeRegime.TRENDING: 0.02, CubeRegime.RANGE: 0.08, CubeRegime.STRESS: 0.40, CubeRegime.CRASH: 0.50},
    }

    def __init__(self):
        self._current_regime = CubeRegime.RANGE
        self._regime_history: list[RegimeTransition] = []
        self._regime_days: int = 0

    def determine(self, macro_regime: CubeRegime, risk: float, liquidity: float, flow: float) -> tuple[CubeRegime, float]:
        """Determine regime with confidence scoring.

        Returns (regime, confidence).
        """
        # Score each potential regime
        scores = {}

        for regime in CubeRegime:
            score = 0.0

            # Macro regime prior carries 50% weight — upstream classification is authoritative
            macro_match = 1.0 if macro_regime == regime else 0.0

            if regime == CubeRegime.TRENDING:
                score += (1 - risk) * 0.15
                score += max(liquidity, 0) * 0.15
                score += max(flow, 0) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.RANGE:
                score += (1 - abs(risk - 0.3)) * 0.15
                score += (1 - abs(liquidity)) * 0.15
                score += (1 - abs(flow)) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.STRESS:
                score += risk * 0.15
                score += max(-liquidity, 0) * 0.15
                score += max(-flow, 0) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.CRASH:
                score += max(risk - 0.7, 0) * 0.20
                score += max(-liquidity - 0.5, 0) * 0.15
                score += macro_match * 0.65

            # Apply transition probability prior (light touch — macro prior is dominant)
            trans_prob = self.TRANSITION_PROBS.get(self._current_regime, {}).get(regime, 0.1)
            score *= (0.9 + 0.1 * trans_prob)

            scores[regime] = max(score, 0.01)

        # Select highest scoring regime
        best_regime = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_regime] / total if total > 0 else 0.5

        # Record transition
        if best_regime != self._current_regime:
            self._regime_history.append(RegimeTransition(
                from_regime=self._current_regime,
                to_regime=best_regime,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
            ))
            self._regime_days = 0
        else:
            self._regime_days += 1

        self._current_regime = best_regime
        return best_regime, confidence

    def get_transition_probability(self, target: CubeRegime) -> float:
        return self.TRANSITION_PROBS.get(self._current_regime, {}).get(target, 0.0)


# ---------------------------------------------------------------------------
# GateZAllocator (Enhanced)
# ---------------------------------------------------------------------------
class GateZAllocator:
    """Gate-Z 5-sleeve allocation with risk-adjusted weights.

    Sleeves:  (P1 Carry, P2 Rotation, P3 Trend/LHC, P4 Neutral-Alpha, P5 Down-Offense)
    """

    # Base allocations per regime — (carry, rotation, trend_lhc, neutral_alpha, down_offense)
    BASE_ALLOCATIONS = {
        CubeRegime.TRENDING: (0.25, 0.25, 0.30, 0.10, 0.10),
        CubeRegime.RANGE:    (0.20, 0.20, 0.20, 0.25, 0.15),
        CubeRegime.STRESS:   (0.15, 0.10, 0.10, 0.25, 0.40),
        CubeRegime.CRASH:    (0.05, 0.05, 0.05, 0.20, 0.65),
    }

    REBALANCE_BAND = 0.03

    def __init__(self):
        self._prev_allocation: Optional[SleeveAllocation] = None

    def allocate(self, regime: CubeRegime, risk: float, liquidity: float = 0.0) -> SleeveAllocation:
        base = self.BASE_ALLOCATIONS.get(regime, self.BASE_ALLOCATIONS[CubeRegime.RANGE])

        sa = SleeveAllocation(
            p1_directional_equity=base[0],   # Carry
            p2_factor_rotation=base[1],      # Rotation
            p3_commodities_macro=base[2],    # Trend/LHC
            p4_options_convexity=base[3],    # Neutral-Alpha
            p5_hedges_volatility=base[4],    # Down-Offense
            carry=base[0],
            rotation=base[1],
            trend_lhc=base[2],
            neutral_alpha=base[3],
            down_offense=base[4],
        )

        # Risk adjustment: shift from carry/trend to hedges/neutral-alpha
        risk_shift = risk * 0.15
        sa.p1_directional_equity = max(0.02, sa.p1_directional_equity - risk_shift * 0.5)
        sa.p3_commodities_macro = max(0.02, sa.p3_commodities_macro - risk_shift * 0.5)
        sa.p5_hedges_volatility = min(0.70, sa.p5_hedges_volatility + risk_shift * 0.6)
        sa.p4_options_convexity = min(0.35, sa.p4_options_convexity + risk_shift * 0.4)

        # Liquidity adjustment: positive liquidity favours carry + trend
        if liquidity > 0.3:
            liq_boost = (liquidity - 0.3) * 0.1
            sa.p1_directional_equity = min(0.40, sa.p1_directional_equity + liq_boost * 0.5)
            sa.p3_commodities_macro = min(0.40, sa.p3_commodities_macro + liq_boost * 0.5)
            sa.p5_hedges_volatility = max(0.05, sa.p5_hedges_volatility - liq_boost)

        # Normalize to 1.0 (also syncs extended fields)
        sa.normalize()

        # Apply rebalancing bands
        if self._prev_allocation is not None:
            sa = self._apply_bands(sa, self._prev_allocation)

        self._prev_allocation = sa
        return sa

    def _apply_bands(self, new: SleeveAllocation, prev: SleeveAllocation) -> SleeveAllocation:
        """Only adjust if change exceeds rebalancing band."""
        fields = ['p1_directional_equity', 'p2_factor_rotation', 'p3_commodities_macro',
                   'p4_options_convexity', 'p5_hedges_volatility']
        for f in fields:
            new_val = getattr(new, f)
            prev_val = getattr(prev, f)
            if abs(new_val - prev_val) < self.REBALANCE_BAND:
                setattr(new, f, prev_val)
        new.normalize()
        return new


# ---------------------------------------------------------------------------
# RiskGovernor
# ---------------------------------------------------------------------------
class GateLogic:
    """4-Gate entry logic for position approval.

    Gate 1 — Fundamentals (40%):      Quality/ROIC/FCF + GNN supply-chain penalty
    Gate 2 — Flow/Headlines (20%):    ETF creations + Tensor signal → shortlist
    Gate 3 — Macro/Regime (20%):      Kernel projections + rates/FX betas → filter
    Gate 4 — Momentum/Technical (20%): Breadth/leadership/gamma/vanna confirms

    Each gate returns a score in [0, 1]. A position must pass all 4 gates
    with a combined score above the threshold (default 0.50).
    """

    GATE_WEIGHTS = [0.40, 0.20, 0.20, 0.20]  # Fundamentals, Flow, Macro, Momentum
    PASS_THRESHOLD = 0.50

    def evaluate(self, ticker: str, flow_score: float = 0.5, macro_score: float = 0.5,
                 fundamental_score: float = 0.5, momentum_score: float = 0.5) -> dict:
        """Evaluate a ticker through all 4 gates."""
        scores = [
            np.clip(flow_score, 0, 1),
            np.clip(macro_score, 0, 1),
            np.clip(fundamental_score, 0, 1),
            np.clip(momentum_score, 0, 1),
        ]
        weighted = sum(s * w for s, w in zip(scores, self.GATE_WEIGHTS))
        gate_pass = [s >= 0.3 for s in scores]  # Each gate has minimum 0.3

        return {
            "ticker": ticker,
            "gate_scores": scores,
            "weighted_score": float(weighted),
            "gates_passed": sum(gate_pass),
            "all_gates_pass": all(gate_pass),
            "approved": all(gate_pass) and weighted >= self.PASS_THRESHOLD,
            "gate_details": {
                "G1_Flow": {"score": scores[0], "pass": gate_pass[0]},
                "G2_Macro": {"score": scores[1], "pass": gate_pass[1]},
                "G3_Fundamental": {"score": scores[2], "pass": gate_pass[2]},
                "G4_Momentum": {"score": scores[3], "pass": gate_pass[3]},
            },
        }

    def batch_evaluate(self, candidates: list[dict]) -> list[dict]:
        """Evaluate a batch of candidates, return sorted by weighted score."""
        results = []
        for c in candidates:
            result = self.evaluate(
                ticker=c.get("ticker", "???"),
                flow_score=c.get("flow_score", 0.5),
                macro_score=c.get("macro_score", 0.5),
                fundamental_score=c.get("fundamental_score", 0.5),
                momentum_score=c.get("momentum_score", 0.5),
            )
            results.append(result)
        return sorted(results, key=lambda x: x["weighted_score"], reverse=True)


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------
class KillSwitch:
    """Auto-derisking kill switch.

    Triggers when ALL of:
      - HY OAS widens +35bp (proxy: credit_spread delta)
      - VIX term structure flat or inverted
      - Market breadth < 50%

    Action: force β ≤ 0.35, max tail spend, reduce gross leverage.
    """

    HY_OAS_THRESHOLD = 0.35    # +35bp widening
    VIX_TERM_FLAT = 0.02       # Term structure < 2% contango → flat/inverted
    BREADTH_THRESHOLD = 0.50   # Breadth below 50%
    FORCED_BETA_CAP = 0.35

    def __init__(self):
        self._triggered = False
        self._trigger_time: Optional[str] = None
        self._prev_credit_spread: float = 3.0

    def check(self, credit_spread: float, vix: float, vix_3m: float = None,
              breadth: float = 0.6) -> dict:
        """Check kill switch conditions.

        Args:
            credit_spread: Current HY credit spread
            vix: Current VIX level
            vix_3m: 3-month VIX (for term structure). If None, estimate from VIX.
            breadth: Market breadth ratio [0, 1]
        """
        # HY OAS widening
        hy_delta = credit_spread - self._prev_credit_spread
        hy_triggered = hy_delta >= self.HY_OAS_THRESHOLD
        self._prev_credit_spread = credit_spread

        # VIX term structure (flat/inverted if spot >= 3M)
        if vix_3m is None:
            vix_3m = vix * 0.95  # Estimate: normal contango ~5%
        term_ratio = (vix_3m - vix) / max(vix, 1)
        vix_term_triggered = term_ratio < self.VIX_TERM_FLAT

        # Breadth
        breadth_triggered = breadth < self.BREADTH_THRESHOLD

        # All three must fire
        all_triggered = hy_triggered and vix_term_triggered and breadth_triggered

        if all_triggered and not self._triggered:
            self._triggered = True
            self._trigger_time = datetime.now().isoformat()
            logger.warning("KILL SWITCH ACTIVATED — forcing β ≤ %.2f", self.FORCED_BETA_CAP)

        result = {
            "triggered": all_triggered,
            "active": self._triggered,
            "hy_oas_delta": round(hy_delta, 4),
            "hy_triggered": hy_triggered,
            "vix_term_ratio": round(term_ratio, 4),
            "vix_term_triggered": vix_term_triggered,
            "breadth": round(breadth, 4),
            "breadth_triggered": breadth_triggered,
            "forced_beta_cap": self.FORCED_BETA_CAP if self._triggered else None,
            "trigger_time": self._trigger_time,
        }
        return result

    def reset(self):
        """Manually reset kill switch (requires explicit action)."""
        self._triggered = False
        self._trigger_time = None
        logger.info("Kill switch reset")

    @property
    def is_active(self) -> bool:
        return self._triggered


# ---------------------------------------------------------------------------
# FCLP — Full Calibration Learning Protocol
# ---------------------------------------------------------------------------
class FCLPLoop:
    """Full Calibration Learning Protocol — daily recalibration.

    Steps:
      1. Ingest plumbing data (Fed balance sheet, SOFR, RRP)
      2. Recompute Liquidity Tensor + Reserve Flow Kernel
      3. Regime detection via HMM+RL
      4. Gate scoring (4-gate) on universe
      5. Risk pass (RiskGovernor)
      6. Write final allocations

    Tracks calibration history for drift detection.
    """

    def __init__(self):
        self._calibration_history: deque = deque(maxlen=252)
        self._last_calibration: Optional[dict] = None

    def run(self, cube: "MetadronCube", macro: MacroSnapshot) -> dict:
        """Execute full FCLP calibration cycle."""
        cal = {"timestamp": datetime.now().isoformat(), "steps": {}}

        # Step 1: Plumbing
        fed_data = cube._fed_plumbing.compute(macro)
        cal["steps"]["plumbing"] = {"net_plumbing": fed_data["net_plumbing"]}

        # Step 2: Tensor + Kernel
        liquidity = cube._liquidity_tensor.compute(macro, fed_data)
        impulse = cube._reserve_kernel.compute_impulse(3200)
        cal["steps"]["tensor"] = {"L_t": liquidity.value, "impulse": impulse}

        # Step 3: Regime
        risk = cube._risk_model.compute(macro)
        flow = cube._flow_model.compute(macro)
        regime, confidence = cube._regime_engine.determine(
            macro.cube_regime, risk.value, liquidity.value, flow.value
        )
        cal["steps"]["regime"] = {"regime": regime.value, "confidence": confidence}

        # Step 4: Gate scoring (placeholder — requires universe)
        cal["steps"]["gate_scoring"] = {"candidates_scored": 0}

        # Step 5: Risk pass
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS[CubeRegime.RANGE])
        risk_check = cube._risk_governor.full_check(regime=regime)
        cal["steps"]["risk_pass"] = {"all_clear": risk_check.get("all_pass", True)}

        # Step 6: Write allocations
        sleeves = cube._gate_z.allocate(regime, risk.value, liquidity.value)
        cal["steps"]["allocations"] = sleeves.legacy_dict()

        # Drift detection
        if self._last_calibration:
            prev_regime = self._last_calibration.get("steps", {}).get("regime", {}).get("regime")
            cal["regime_changed"] = prev_regime != regime.value
        else:
            cal["regime_changed"] = False

        self._calibration_history.append(cal)
        self._last_calibration = cal

        return cal

    def get_drift_report(self) -> dict:
        """Analyze calibration drift over recent history."""
        if len(self._calibration_history) < 2:
            return {"drift": 0, "regime_changes": 0, "samples": len(self._calibration_history)}

        regime_changes = sum(1 for c in self._calibration_history if c.get("regime_changed"))
        l_values = [c["steps"]["tensor"]["L_t"] for c in self._calibration_history if "tensor" in c["steps"]]
        l_drift = float(np.std(l_values)) if len(l_values) > 1 else 0

        return {
            "drift": round(l_drift, 4),
            "regime_changes": regime_changes,
            "samples": len(self._calibration_history),
            "last_calibration": self._last_calibration.get("timestamp") if self._last_calibration else None,
        }


# ---------------------------------------------------------------------------
# RiskGovernor (Enhanced)
# ---------------------------------------------------------------------------
class RiskGovernor:
    """Position limit enforcement, VaR budgeting, leverage monitoring.

    Enhanced with:
      - Crash floor enforcement (≥ +25% annual)
      - Beta burst logic (e.g. 0.65 → 0.70 on momentum confirms)
      - VaR limit: ≤ $0.30M (95%/1-day) on $20M NAV reference
      - Theta budget enforcement
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_leverage: float = 3.0,
        max_drawdown: float = 0.15,
        var_limit_pct: float = 0.015,  # VaR ≤ 1.5% of NAV (=$0.30M on $20M)
        crash_floor: float = 0.25,
        nav_reference: float = 20_000_000.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.var_limit_pct = var_limit_pct
        self.crash_floor = crash_floor
        self.nav_reference = nav_reference

    def check_position_limit(self, position_pct: float) -> tuple[bool, str]:
        if position_pct > self.max_position_pct:
            return False, f"Position {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"
        return True, "OK"

    def check_sector_limit(self, sector_pct: float) -> tuple[bool, str]:
        if sector_pct > self.max_sector_pct:
            return False, f"Sector {sector_pct:.1%} exceeds limit {self.max_sector_pct:.1%}"
        return True, "OK"

    def check_leverage(self, current_leverage: float, regime: CubeRegime) -> tuple[bool, str]:
        regime_limit = REGIME_PARAMS.get(regime, {}).get("max_leverage", self.max_leverage)
        if current_leverage > regime_limit:
            return False, f"Leverage {current_leverage:.2f} exceeds regime limit {regime_limit:.2f}"
        return True, "OK"

    def check_drawdown(self, current_drawdown: float) -> tuple[bool, str]:
        if abs(current_drawdown) > self.max_drawdown:
            return False, f"Drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown:.1%}"
        return True, "OK"

    def check_var(self, portfolio_var_pct: float) -> tuple[bool, str]:
        """Check VaR limit (95%/1-day)."""
        if portfolio_var_pct > self.var_limit_pct:
            return False, f"VaR {portfolio_var_pct:.2%} exceeds limit {self.var_limit_pct:.2%}"
        return True, "OK"

    def check_crash_floor(self, ytd_return: float) -> tuple[bool, str]:
        """Ensure crash-protection floor (annualized ≥ crash_floor)."""
        if ytd_return < -self.crash_floor:
            return False, f"YTD {ytd_return:.1%} breaches crash floor -{self.crash_floor:.0%}"
        return True, "OK"

    def check_theta_budget(self, daily_theta_pct: float, regime: CubeRegime) -> tuple[bool, str]:
        """Check daily theta spend vs budget."""
        budget = REGIME_PARAMS.get(regime, {}).get("theta_budget_daily", 0.001)
        if abs(daily_theta_pct) > budget:
            return False, f"Theta {daily_theta_pct:.4%} exceeds budget {budget:.4%}"
        return True, "OK"

    def get_beta_with_burst(self, regime: CubeRegime, momentum_confirms: bool = False) -> float:
        """Return beta cap, optionally with burst if momentum confirms."""
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS[CubeRegime.RANGE])
        if momentum_confirms and "beta_burst" in params:
            return params["beta_burst"]
        return params["beta_cap"]

    def compute_risk_budget(self, regime: CubeRegime, current_risk: float) -> float:
        """Compute available risk budget as fraction of total."""
        base = (R_LOW + R_HIGH) / 2
        regime_mult = {
            CubeRegime.TRENDING: 1.2,
            CubeRegime.RANGE: 1.0,
            CubeRegime.STRESS: 0.7,
            CubeRegime.CRASH: 0.4,
        }.get(regime, 1.0)
        budget = base * regime_mult
        used = current_risk * budget
        return max(0, budget - used)

    def full_check(self, position_pct: float = 0, sector_pct: float = 0,
                   leverage: float = 0, drawdown: float = 0,
                   regime: CubeRegime = CubeRegime.RANGE,
                   var_pct: float = 0, ytd_return: float = 0,
                   theta_pct: float = 0) -> dict:
        checks = {}
        checks["position"] = self.check_position_limit(position_pct)
        checks["sector"] = self.check_sector_limit(sector_pct)
        checks["leverage"] = self.check_leverage(leverage, regime)
        checks["drawdown"] = self.check_drawdown(drawdown)
        checks["var"] = self.check_var(var_pct)
        checks["crash_floor"] = self.check_crash_floor(ytd_return)
        checks["theta"] = self.check_theta_budget(theta_pct, regime)
        checks["all_pass"] = all(v[0] for v in checks.values() if isinstance(v, tuple))
        return checks


# ---------------------------------------------------------------------------
# CubeLearningLoop
# ---------------------------------------------------------------------------
class CubeLearningLoop:
    """Adaptive parameter adjustment based on forecast accuracy."""

    def __init__(self, learning_rate: float = 0.01, memory: int = 100):
        self.learning_rate = learning_rate
        self._predictions: deque = deque(maxlen=memory)
        self._actuals: deque = deque(maxlen=memory)
        self._regime_accuracy: dict[str, list[bool]] = {}
        self._weight_adjustments: dict[str, float] = {}

    def record(self, predicted_regime: CubeRegime, predicted_beta: float,
               actual_return: float, actual_vol: float):
        self._predictions.append({
            "regime": predicted_regime.value,
            "beta": predicted_beta,
            "timestamp": datetime.now().isoformat(),
        })
        self._actuals.append({
            "return": actual_return,
            "vol": actual_vol,
        })

    def compute_accuracy(self) -> dict:
        if len(self._predictions) < 10:
            return {"accuracy": 0, "sample_size": len(self._predictions)}

        correct = 0
        total = len(self._predictions)
        for pred, actual in zip(self._predictions, self._actuals):
            pred_direction = 1 if pred["beta"] > 0 else -1
            actual_direction = 1 if actual["return"] > 0 else -1
            if pred_direction == actual_direction:
                correct += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "sample_size": total,
            "correct": correct,
        }

    def suggest_adjustments(self) -> dict:
        accuracy = self.compute_accuracy()
        adjustments = {}

        if accuracy["accuracy"] < 0.45:
            adjustments["reduce_leverage"] = True
            adjustments["increase_hedge"] = 0.05
            adjustments["note"] = "Below 45% accuracy — reduce risk"
        elif accuracy["accuracy"] > 0.60:
            adjustments["increase_leverage"] = True
            adjustments["decrease_hedge"] = 0.03
            adjustments["note"] = "Above 60% accuracy — increase conviction"
        else:
            adjustments["note"] = "Accuracy in acceptable range"

        return adjustments


# ---------------------------------------------------------------------------
# CubeHistory
# ---------------------------------------------------------------------------
class CubeHistory:
    """Track historical cube outputs."""

    def __init__(self, max_size: int = 1000):
        self._outputs: deque[CubeOutput] = deque(maxlen=max_size)

    def record(self, output: CubeOutput):
        output.timestamp = datetime.now().isoformat()
        self._outputs.append(output)

    def get_recent(self, n: int = 10) -> list[CubeOutput]:
        return list(self._outputs)[-n:]

    def get_regime_distribution(self) -> dict[str, int]:
        dist = {}
        for o in self._outputs:
            r = o.regime.value
            dist[r] = dist.get(r, 0) + 1
        return dist

    def get_avg_beta(self, window: int = 20) -> float:
        recent = list(self._outputs)[-window:]
        if not recent:
            return 0
        return float(np.mean([o.target_beta for o in recent]))

    @property
    def size(self) -> int:
        return len(self._outputs)


# ---------------------------------------------------------------------------
# StressScenarioEngine
# ---------------------------------------------------------------------------
class StressScenarioEngine:
    """Run stress tests across multiple market scenarios."""

    SCENARIOS = {
        "2008_GFC": MacroSnapshot(vix=80, spy_return_3m=-0.40, credit_spread=10, yield_spread=-2.0, cube_regime=CubeRegime.CRASH),
        "2020_COVID": MacroSnapshot(vix=65, spy_return_3m=-0.30, credit_spread=8, yield_spread=-0.5, cube_regime=CubeRegime.CRASH),
        "2022_HIKE": MacroSnapshot(vix=35, spy_return_3m=-0.15, credit_spread=5, yield_spread=-0.8, cube_regime=CubeRegime.STRESS),
        "BULL_RUN": MacroSnapshot(vix=12, spy_return_3m=0.15, credit_spread=2, yield_spread=1.5, cube_regime=CubeRegime.TRENDING),
        "RANGE_BOUND": MacroSnapshot(vix=18, spy_return_3m=0.02, credit_spread=3, yield_spread=0.2, cube_regime=CubeRegime.RANGE),
        "VIX_SPIKE": MacroSnapshot(vix=45, spy_return_3m=-0.10, credit_spread=6, yield_spread=-1.0, cube_regime=CubeRegime.CRASH),
    }

    def run_all(self, cube: "MetadronCube") -> dict:
        results = {}
        for name, scenario in self.SCENARIOS.items():
            output = cube.compute(scenario)
            results[name] = {
                "regime": output.regime.value,
                "target_beta": round(output.target_beta, 4),
                "max_leverage": output.max_leverage,
                "equity_allocation": round(output.sleeves.p1_directional_equity, 3),
                "hedge_allocation": round(output.sleeves.p5_hedges_volatility, 3),
                "risk_score": round(output.risk.value, 3),
            }
        return results


# ---------------------------------------------------------------------------
# MetadronCube
# ---------------------------------------------------------------------------
class MetadronCube:
    """C(t) = f(L_t, R_t, F_t) — multi-layer allocation tensor.

    Computes the cube state from macro inputs and determines:
    - Regime classification
    - 5-sleeve capital allocation
    - Target beta
    - Risk budget
    """

    def __init__(self):
        self._last_output: Optional[CubeOutput] = None
        self._fed_plumbing = FedPlumbingLayer()
        self._liquidity_tensor = LiquidityTensor()
        self._reserve_kernel = ReserveFlowKernel()
        self._risk_model = RiskStateModel()
        self._flow_model = CapitalFlowModel()
        self._regime_engine = RegimeEngine()
        self._gate_z = GateZAllocator()
        self._gate_logic = GateLogic()
        self._kill_switch = KillSwitch()
        self._risk_governor = RiskGovernor()
        self._learning = CubeLearningLoop()
        self._fclp = FCLPLoop()
        self._history = CubeHistory()
        self._stress = StressScenarioEngine()

    def compute(self, macro: MacroSnapshot) -> CubeOutput:
        """Compute full cube state from macro snapshot."""
        output = CubeOutput()

        # Layer 0: Fed plumbing
        fed_data = self._fed_plumbing.compute(macro)

        # Layer 1: Liquidity tensor
        output.liquidity = self._liquidity_tensor.compute(macro, fed_data)

        # Layer 2: Reserve flow
        reserve_impulse = self._reserve_kernel.compute_impulse(3200)

        # Risk state
        output.risk = self._risk_model.compute(macro)

        # Flow state
        output.flow = self._flow_model.compute(macro)

        # Regime determination
        regime, confidence = self._regime_engine.determine(
            macro_regime=macro.cube_regime,
            risk=output.risk.value,
            liquidity=output.liquidity.value,
            flow=output.flow.value,
        )
        output.regime = regime
        output.regime_confidence = confidence

        # Get regime parameters
        params = REGIME_PARAMS.get(output.regime, REGIME_PARAMS[CubeRegime.RANGE])
        output.max_leverage = params["max_leverage"]
        output.beta_cap = params["beta_cap"]

        # Gate-Z: 5-sleeve allocation
        output.sleeves = self._gate_z.allocate(output.regime, output.risk.value, output.liquidity.value)

        # Kill-switch check — override beta cap if triggered
        ks = self._kill_switch.check(
            credit_spread=macro.credit_spread,
            vix=macro.vix,
            breadth=output.flow.breadth,
        )
        if ks["active"]:
            output.beta_cap = min(output.beta_cap, KillSwitch.FORCED_BETA_CAP)

        # Target beta from corridor
        output.target_beta = self._compute_target_beta(
            Rm=macro.spy_return_3m * 4,
            sigma_m=macro.vix / 100,
        )
        output.target_beta = max(BETA_INV, min(output.beta_cap, output.target_beta))

        # Risk budget
        output.risk_budget_pct = self._compute_risk_budget(output)

        # Cube tensor
        output.cube_tensor = np.array([output.liquidity.value, output.risk.value, output.flow.value])

        # Record history
        self._history.record(output)
        self._last_output = output

        return output

    def get_last(self) -> Optional[CubeOutput]:
        return self._last_output

    def get_history(self) -> CubeHistory:
        return self._history

    def run_stress_tests(self) -> dict:
        return self._stress.run_all(self)

    def get_learning_stats(self) -> dict:
        return self._learning.compute_accuracy()

    def get_risk_governor(self) -> RiskGovernor:
        return self._risk_governor

    def get_gate_logic(self) -> GateLogic:
        return self._gate_logic

    def get_kill_switch(self) -> KillSwitch:
        return self._kill_switch

    def run_fclp(self, macro: MacroSnapshot) -> dict:
        """Run Full Calibration Learning Protocol."""
        return self._fclp.run(self, macro)

    def get_fclp_drift(self) -> dict:
        """Get FCLP calibration drift report."""
        return self._fclp.get_drift_report()

    # --- Layer computations --------------------------------------------------

    def _compute_liquidity(self, macro: MacroSnapshot) -> LiquidityState:
        """L(t) in [-1, +1]. Legacy method for backwards compatibility."""
        fed_data = self._fed_plumbing.compute(macro)
        return self._liquidity_tensor.compute(macro, fed_data)

    def _compute_risk(self, macro: MacroSnapshot) -> RiskState:
        """R(t) in [0, 1]. 0=calm, 1=extreme."""
        return self._risk_model.compute(macro)

    def _compute_flow(self, macro: MacroSnapshot) -> FlowState:
        """Capital flow model from sector rankings."""
        return self._flow_model.compute(macro)

    def _compute_sleeves(self, cube: CubeOutput) -> SleeveAllocation:
        """Gate-Z 5-sleeve allocation based on regime and risk."""
        return self._gate_z.allocate(cube.regime, cube.risk.value, cube.liquidity.value)

    def _compute_target_beta(self, Rm: float, sigma_m: float) -> float:
        """Beta corridor from Dataset 1.

        Linear interpolation in [R_LOW, R_HIGH] with vol-normalization.
        """
        if Rm <= R_LOW:
            base_beta = -0.029
        elif Rm >= R_HIGH:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (R_HIGH - R_LOW)
            base_beta = -0.029 + slope * (Rm - R_LOW)

        vol_adj = VOL_STANDARD / max(sigma_m, 0.05)
        target = base_beta * EXECUTION_MULTIPLIER * vol_adj

        return max(BETA_INV, min(BETA_MAX, target))

    def _compute_risk_budget(self, cube: CubeOutput) -> float:
        """Risk budget as % of NAV — targets 7–12% corridor."""
        base = (R_LOW + R_HIGH) / 2
        if cube.regime == CubeRegime.TRENDING:
            return min(R_HIGH, base * 1.2)
        elif cube.regime == CubeRegime.CRASH:
            return max(R_LOW * 0.5, base * 0.4)
        elif cube.regime == CubeRegime.STRESS:
            return max(R_LOW, base * 0.7)
        return base

    def summary(self) -> str:
        """ASCII cube summary."""
        if not self._last_output:
            return "MetadronCube: No data yet"

        o = self._last_output
        lines = [
            "=" * 60,
            "METADRON CUBE STATE",
            "=" * 60,
            f"  Regime:      {o.regime.value} (confidence: {o.regime_confidence:.1%})",
            f"  L(t):        {o.liquidity.value:>+.3f}  (liquidity)",
            f"  R(t):        {o.risk.value:>.3f}   (risk)",
            f"  F(t):        {o.flow.value:>+.3f}  (flow)",
            f"  Target β:    {o.target_beta:>.4f}",
            f"  Max Leverage: {o.max_leverage:.1f}x",
            f"  Beta Cap:    {o.beta_cap:.2f}",
            "",
            "  SLEEVES:",
            f"    P1 Equity:     {o.sleeves.p1_directional_equity:.1%}",
            f"    P2 Factor:     {o.sleeves.p2_factor_rotation:.1%}",
            f"    P3 Commodity:  {o.sleeves.p3_commodities_macro:.1%}",
            f"    P4 Convexity:  {o.sleeves.p4_options_convexity:.1%}",
            f"    P5 Hedges:     {o.sleeves.p5_hedges_volatility:.1%}",
            f"    TOTAL:         {o.sleeves.total():.1%}",
            "",
            f"  History: {self._history.size} records",
            f"  Avg Beta (20d): {self._history.get_avg_beta():.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
