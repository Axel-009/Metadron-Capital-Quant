"""Agent Scorecard & Management System — 25 agents across 4 tiers.

Complete agent orchestration layer for Metadron Capital:
    - 12 investor persona agents
    - 6 analytical agents
    - 7 engine agents
    - 4-tier hierarchy with promotion/demotion
    - Weekly scoring: 40% accuracy + 30% Sharpe + 30% hit rate
    - Signal aggregation, consensus voting, conflict resolution
    - Historical performance tracking with rolling Sharpe computation
    - Agent coordination with weight adjustment
"""

import json
import math
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    from ..signals.macro_engine import CubeRegime
except ImportError:
    class CubeRegime(str, Enum):
        TRENDING = "TRENDING"
        RANGE = "RANGE"
        STRESS = "STRESS"
        CRASH = "CRASH"


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHARPE_NORM_CAP = 3.0
PROMOTION_TOP_WEEKS = 4
DEMOTION_BOTTOM_WEEKS = 2
MAX_WEEKLY_HISTORY = 104  # 2 years of weekly data
ROLLING_SHARPE_WINDOW = 52
ANNUALIZATION_FACTOR = 52  # weekly -> annual


# ---------------------------------------------------------------------------
# Tier Hierarchy
# ---------------------------------------------------------------------------
class AgentTierLevel(str, Enum):
    """Agent hierarchy tiers. Matches the sector_bots AgentTier."""
    TIER_1_GENERAL = "TIER_1_General"
    TIER_2_CAPTAIN = "TIER_2_Captain"
    TIER_3_LIEUTENANT = "TIER_3_Lieutenant"
    TIER_4_RECRUIT = "TIER_4_Recruit"


TIER_ORDER = [
    AgentTierLevel.TIER_1_GENERAL,
    AgentTierLevel.TIER_2_CAPTAIN,
    AgentTierLevel.TIER_3_LIEUTENANT,
    AgentTierLevel.TIER_4_RECRUIT,
]

TIER_THRESHOLDS = {
    AgentTierLevel.TIER_1_GENERAL: {"min_sharpe": 2.0, "min_accuracy": 0.80},
    AgentTierLevel.TIER_2_CAPTAIN: {"min_sharpe": 1.5, "min_accuracy": 0.55},
    AgentTierLevel.TIER_3_LIEUTENANT: {"min_sharpe": 1.0, "min_accuracy": 0.50},
    AgentTierLevel.TIER_4_RECRUIT: {"min_sharpe": -999, "min_accuracy": 0.0},
}


# ---------------------------------------------------------------------------
# Agent Categories
# ---------------------------------------------------------------------------
class AgentCategory(str, Enum):
    INVESTOR_PERSONA = "investor_persona"
    ANALYTICAL = "analytical"
    ENGINE = "engine"


# ---------------------------------------------------------------------------
# Agent Registry — 25 agents
# ---------------------------------------------------------------------------
AGENT_REGISTRY = {
    # --- 12 Investor Personas ---
    "ValueHunter": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Deep value investor seeking undervalued securities with strong fundamentals",
        "strategy": "value",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": ["Financials", "Energy", "Industrials"],
        "style": "contrarian",
    },
    "MomentumRider": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Trend-following agent that rides momentum across sectors",
        "strategy": "momentum",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": ["Information Technology", "Consumer Discretionary"],
        "style": "trend_following",
    },
    "VolatilityTrader": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Options and volatility surface specialist",
        "strategy": "volatility",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "market_neutral",
    },
    "MacroStrategist": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Top-down macro-driven allocation specialist",
        "strategy": "macro",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": ["Energy", "Materials", "Utilities"],
        "style": "top_down",
    },
    "GrowthSeeker": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Growth-at-any-price investor focused on revenue acceleration",
        "strategy": "growth",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": ["Information Technology", "Health Care", "Communication Services"],
        "style": "growth",
    },
    "DividendCollector": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Income-focused agent targeting high dividend yield and sustainability",
        "strategy": "income",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": ["Utilities", "Real Estate", "Consumer Staples"],
        "style": "income",
    },
    "MeanReversionBot": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Statistical arbitrage agent trading mean-reversion patterns",
        "strategy": "mean_reversion",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "statistical_arbitrage",
    },
    "EventDrivenTrader": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Catalyst-driven agent focused on earnings, M&A, and restructuring",
        "strategy": "event_driven",
        "default_tier": AgentTierLevel.TIER_4_RECRUIT,
        "sector_bias": ["Health Care", "Information Technology"],
        "style": "event_driven",
    },
    "SentimentReader": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "NLP-driven agent analyzing news, social media, and earnings calls",
        "strategy": "sentiment",
        "default_tier": AgentTierLevel.TIER_4_RECRUIT,
        "sector_bias": [],
        "style": "sentiment",
    },
    "TechAnalyst": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Pure technical analysis agent using chart patterns and indicators",
        "strategy": "technical",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "technical",
    },
    "QualityScreener": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Quality factor investor filtering on ROE, margins, and balance sheet",
        "strategy": "quality",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": ["Health Care", "Information Technology", "Consumer Staples"],
        "style": "quality_factor",
    },
    "SmallCapHunter": {
        "category": AgentCategory.INVESTOR_PERSONA,
        "description": "Small-cap specialist seeking alpha in less-covered names",
        "strategy": "small_cap",
        "default_tier": AgentTierLevel.TIER_4_RECRUIT,
        "sector_bias": ["Industrials", "Health Care", "Information Technology"],
        "style": "small_cap",
    },

    # --- 6 Analytical Agents ---
    "RiskAnalyst": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Portfolio risk monitoring and VaR/CVaR computation",
        "strategy": "risk_management",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": [],
        "style": "risk_overlay",
    },
    "CorrelationTracker": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Cross-asset correlation monitoring and regime detection",
        "strategy": "correlation",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "correlation_analysis",
    },
    "LiquidityMonitor": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Market microstructure and liquidity condition analyzer",
        "strategy": "liquidity",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "liquidity_analysis",
    },
    "RegimeDetector": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Hidden Markov Model regime detection and transition probability",
        "strategy": "regime",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": [],
        "style": "regime_detection",
    },
    "FlowAnalyst": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Fund flow and institutional positioning tracker",
        "strategy": "flow",
        "default_tier": AgentTierLevel.TIER_4_RECRUIT,
        "sector_bias": [],
        "style": "flow_analysis",
    },
    "VolSurfaceMapper": {
        "category": AgentCategory.ANALYTICAL,
        "description": "Implied volatility surface analysis and skew monitoring",
        "strategy": "vol_surface",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "vol_surface_analysis",
    },

    # --- 7 Engine Agents ---
    "MacroBot": {
        "category": AgentCategory.ENGINE,
        "description": "GMTF money velocity engine and macro regime signals",
        "strategy": "macro_engine",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": [],
        "style": "macro_systematic",
    },
    "CubeBot": {
        "category": AgentCategory.ENGINE,
        "description": "MetadronCube C(t) regime classification and allocation engine",
        "strategy": "cube_regime",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": [],
        "style": "regime_allocation",
    },
    "AlphaBot": {
        "category": AgentCategory.ENGINE,
        "description": "ML walk-forward alpha optimizer and signal generator",
        "strategy": "ml_alpha",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "ml_systematic",
    },
    "ExecutionBot": {
        "category": AgentCategory.ENGINE,
        "description": "Trade execution optimizer and slippage minimizer",
        "strategy": "execution",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "execution_optimization",
    },
    "BetaBot": {
        "category": AgentCategory.ENGINE,
        "description": "Beta corridor manager ensuring 7-12% return target range",
        "strategy": "beta_management",
        "default_tier": AgentTierLevel.TIER_2_CAPTAIN,
        "sector_bias": [],
        "style": "beta_corridor",
    },
    "HedgeBot": {
        "category": AgentCategory.ENGINE,
        "description": "Dynamic hedging agent managing tail risk and portfolio protection",
        "strategy": "hedging",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "hedge_overlay",
    },
    "RebalanceBot": {
        "category": AgentCategory.ENGINE,
        "description": "Portfolio rebalancing optimizer with tax-loss harvesting awareness",
        "strategy": "rebalancing",
        "default_tier": AgentTierLevel.TIER_3_LIEUTENANT,
        "sector_bias": [],
        "style": "rebalance_optimization",
    },
}

# Verify we have exactly 25 agents
if len(AGENT_REGISTRY) != 25:
    logger.warning("Expected 25 agents in registry, got %d — registry may be incomplete", len(AGENT_REGISTRY))


# ---------------------------------------------------------------------------
# Performance Record
# ---------------------------------------------------------------------------
@dataclass
class PerformanceRecord:
    """Single performance observation for an agent."""
    timestamp: str = ""
    weekly_return: float = 0.0
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0
    composite_score: float = 0.0
    signals_generated: int = 0
    signals_correct: int = 0
    regime: str = "RANGE"


@dataclass
class AgentProfile:
    """Complete profile for a single agent."""
    name: str = ""
    category: str = AgentCategory.INVESTOR_PERSONA
    description: str = ""
    strategy: str = ""
    style: str = ""
    sector_bias: list = field(default_factory=list)

    # Current state
    tier: str = AgentTierLevel.TIER_4_RECRUIT
    is_active: bool = True
    weight: float = 1.0  # Signal weight (adjusted by performance)

    # Scoring
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0
    composite_score: float = 0.0

    # Streak tracking
    consecutive_top_weeks: int = 0
    consecutive_bottom_weeks: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Performance history
    history: list = field(default_factory=list)
    weekly_returns: list = field(default_factory=list)
    total_signals: int = 0
    correct_signals: int = 0

    # Sector specialization
    sector_scores: dict = field(default_factory=dict)

    def rolling_sharpe(self, window: int = ROLLING_SHARPE_WINDOW) -> float:
        """Compute rolling annualized Sharpe ratio from weekly returns."""
        if len(self.weekly_returns) < 4:
            return 0.0
        recent = self.weekly_returns[-window:]
        if len(recent) < 2:
            return 0.0
        mean_ret = statistics.mean(recent)
        std_ret = statistics.stdev(recent)
        if std_ret == 0:
            return 0.0
        return (mean_ret * math.sqrt(ANNUALIZATION_FACTOR)) / std_ret

    def update_streak(self, is_win: bool):
        """Update win/loss streak counters."""
        if is_win:
            self.win_streak += 1
            self.loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.win_streak)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)


@dataclass
class AgentSignal:
    """Signal generated by an agent."""
    agent_name: str = ""
    ticker: str = ""
    direction: str = "HOLD"   # BUY / SELL / HOLD
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    regime: str = "RANGE"
    sector: str = ""
    weight: float = 1.0  # Effective weight (agent weight * confidence)


@dataclass
class ConsensusResult:
    """Result of consensus voting across agents."""
    ticker: str = ""
    direction: str = "HOLD"
    consensus_score: float = 0.0
    votes_buy: int = 0
    votes_sell: int = 0
    votes_hold: int = 0
    total_votes: int = 0
    agreement_pct: float = 0.0
    weighted_confidence: float = 0.0
    participating_agents: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    resolution_method: str = ""


# ---------------------------------------------------------------------------
# Agent Performance Database (in-memory)
# ---------------------------------------------------------------------------
class AgentPerformanceDB:
    """In-memory database tracking historical agent performance.

    Stores weekly snapshots, computes rolling statistics,
    and maintains sector specialization scores.
    """

    def __init__(self):
        self._records: Dict[str, List[PerformanceRecord]] = defaultdict(list)
        self._sector_accuracy: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(self, agent_name: str, record: PerformanceRecord):
        """Add a performance record for an agent."""
        self._records[agent_name].append(record)
        # Trim to max history
        if len(self._records[agent_name]) > MAX_WEEKLY_HISTORY:
            self._records[agent_name] = self._records[agent_name][-MAX_WEEKLY_HISTORY:]

    def record_sector_prediction(
        self, agent_name: str, sector: str, was_correct: bool
    ):
        """Track per-sector prediction accuracy."""
        self._sector_accuracy[agent_name][sector].append(was_correct)
        # Trim per-sector history
        if len(self._sector_accuracy[agent_name][sector]) > MAX_WEEKLY_HISTORY:
            self._sector_accuracy[agent_name][sector] = (
                self._sector_accuracy[agent_name][sector][-MAX_WEEKLY_HISTORY:]
            )

    def get_sector_accuracy(self, agent_name: str, sector: str) -> float:
        """Get agent's accuracy for a specific sector."""
        preds = self._sector_accuracy.get(agent_name, {}).get(sector, [])
        if not preds:
            return 0.0
        return sum(preds) / len(preds)

    def get_sector_specialization_scores(self, agent_name: str) -> Dict[str, float]:
        """Get accuracy scores across all sectors for an agent."""
        sectors = self._sector_accuracy.get(agent_name, {})
        result = {}
        for sector, preds in sectors.items():
            if preds:
                result[sector] = sum(preds) / len(preds)
        return result

    def get_rolling_sharpe(
        self, agent_name: str, window: int = ROLLING_SHARPE_WINDOW
    ) -> float:
        """Compute rolling Sharpe from stored performance records."""
        records = self._records.get(agent_name, [])
        if len(records) < 4:
            return 0.0
        recent = records[-window:]
        returns = [r.weekly_return for r in recent]
        if len(returns) < 2:
            return 0.0
        mean_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret * math.sqrt(ANNUALIZATION_FACTOR)) / std_ret

    def get_history(
        self, agent_name: str, last_n: Optional[int] = None
    ) -> List[PerformanceRecord]:
        """Retrieve performance history for an agent."""
        records = self._records.get(agent_name, [])
        if last_n is not None:
            return records[-last_n:]
        return records

    def get_cumulative_return(self, agent_name: str) -> float:
        """Compute cumulative return from weekly returns."""
        records = self._records.get(agent_name, [])
        if not records:
            return 0.0
        cumulative = 1.0
        for r in records:
            cumulative *= (1 + r.weekly_return)
        return cumulative - 1.0

    def get_max_drawdown(self, agent_name: str) -> float:
        """Compute maximum drawdown from weekly returns."""
        records = self._records.get(agent_name, [])
        if not records:
            return 0.0
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in records:
            cumulative *= (1 + r.weekly_return)
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def summary(self, agent_name: str) -> dict:
        """Full statistical summary for an agent."""
        records = self._records.get(agent_name, [])
        if not records:
            return {"agent": agent_name, "records": 0}
        returns = [r.weekly_return for r in records]
        accuracies = [r.accuracy for r in records]
        return {
            "agent": agent_name,
            "records": len(records),
            "avg_return": statistics.mean(returns) if returns else 0.0,
            "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
            "rolling_sharpe": self.get_rolling_sharpe(agent_name),
            "cumulative_return": self.get_cumulative_return(agent_name),
            "max_drawdown": self.get_max_drawdown(agent_name),
            "avg_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
            "sector_scores": self.get_sector_specialization_scores(agent_name),
        }


# ---------------------------------------------------------------------------
# Full Agent Scorecard Manager
# ---------------------------------------------------------------------------
class FullAgentScorecard:
    """Comprehensive agent scorecard and management system.

    Manages 25 agents across 4 tiers with weekly scoring,
    promotion/demotion, performance tracking, and signal coordination.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/agent_scorecard")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.agents: Dict[str, AgentProfile] = {}
        self.perf_db = AgentPerformanceDB()
        self._signal_buffer: Dict[str, List[AgentSignal]] = defaultdict(list)
        self._consensus_history: List[ConsensusResult] = []

        self._initialize_agents()

    def _initialize_agents(self):
        """Create all 25 agent profiles from registry."""
        for name, config in AGENT_REGISTRY.items():
            profile = AgentProfile(
                name=name,
                category=config["category"],
                description=config["description"],
                strategy=config["strategy"],
                style=config["style"],
                sector_bias=config.get("sector_bias", []),
                tier=config["default_tier"],
                is_active=True,
                weight=self._initial_weight(config["default_tier"]),
            )
            self.agents[name] = profile

    @staticmethod
    def _initial_weight(tier: AgentTierLevel) -> float:
        """Assign initial signal weight based on tier."""
        weights = {
            AgentTierLevel.TIER_1_GENERAL: 2.0,
            AgentTierLevel.TIER_2_CAPTAIN: 1.5,
            AgentTierLevel.TIER_3_LIEUTENANT: 1.0,
            AgentTierLevel.TIER_4_RECRUIT: 0.5,
        }
        return weights.get(tier, 0.5)

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------
    def _assign_tier(self, profile: AgentProfile) -> AgentTierLevel:
        """Assign tier based on current metrics and streak rules."""
        sharpe = profile.sharpe
        accuracy = profile.accuracy

        # Base tier from thresholds
        if sharpe > 2.0 and accuracy > 0.80:
            base_tier = AgentTierLevel.TIER_1_GENERAL
        elif sharpe > 1.5 and accuracy > 0.55:
            base_tier = AgentTierLevel.TIER_2_CAPTAIN
        elif sharpe > 1.0 and accuracy > 0.50:
            base_tier = AgentTierLevel.TIER_3_LIEUTENANT
        else:
            base_tier = AgentTierLevel.TIER_4_RECRUIT

        # Promotion override: 4 consecutive top weeks -> GENERAL
        if profile.consecutive_top_weeks >= PROMOTION_TOP_WEEKS:
            base_tier = AgentTierLevel.TIER_1_GENERAL

        # Demotion override: 2 consecutive bottom weeks -> demote one tier
        if profile.consecutive_bottom_weeks >= DEMOTION_BOTTOM_WEEKS:
            if base_tier != AgentTierLevel.TIER_4_RECRUIT:
                idx = TIER_ORDER.index(base_tier)
                if idx < len(TIER_ORDER) - 1:
                    base_tier = TIER_ORDER[idx + 1]

        return base_tier

    def _update_weight(self, profile: AgentProfile):
        """Adjust signal weight based on tier and recent performance."""
        tier_weights = {
            AgentTierLevel.TIER_1_GENERAL: 2.0,
            AgentTierLevel.TIER_2_CAPTAIN: 1.5,
            AgentTierLevel.TIER_3_LIEUTENANT: 1.0,
            AgentTierLevel.TIER_4_RECRUIT: 0.5,
        }
        base_weight = tier_weights.get(
            AgentTierLevel(profile.tier), 0.5
        )

        # Adjust by recent accuracy (boost high accuracy, penalize low)
        accuracy_factor = 0.5 + profile.accuracy  # [0.5, 1.5]

        # Adjust by win streak (small bonus for hot streaks)
        streak_bonus = min(0.3, profile.win_streak * 0.05)

        profile.weight = round(base_weight * accuracy_factor + streak_bonus, 3)

    # ------------------------------------------------------------------
    # Weekly scoring
    # ------------------------------------------------------------------
    def update_agent_score(
        self,
        agent_name: str,
        accuracy: float,
        sharpe: float,
        hit_rate: float,
        weekly_return: float = 0.0,
        signals_generated: int = 0,
        signals_correct: int = 0,
        is_top_this_week: bool = False,
        is_bottom_this_week: bool = False,
        regime: str = "RANGE",
    ):
        """Update weekly score for a single agent.

        Computes composite score, assigns tier, updates streaks,
        records performance, and adjusts signal weight.
        """
        if agent_name not in self.agents:
            logger.warning(f"Unknown agent: {agent_name}")
            return

        profile = self.agents[agent_name]
        profile.accuracy = accuracy
        profile.sharpe = sharpe
        profile.hit_rate = hit_rate

        # Composite score: 40% accuracy + 30% sharpe_normalized + 30% hit_rate
        sharpe_normalized = min(sharpe / SHARPE_NORM_CAP, 1.0)
        profile.composite_score = (
            0.40 * accuracy + 0.30 * sharpe_normalized + 0.30 * hit_rate
        )

        # Track signals
        profile.total_signals += signals_generated
        profile.correct_signals += signals_correct

        # Weekly returns history
        profile.weekly_returns.append(weekly_return)
        if len(profile.weekly_returns) > MAX_WEEKLY_HISTORY:
            profile.weekly_returns = profile.weekly_returns[-MAX_WEEKLY_HISTORY:]

        # Consecutive week tracking
        if is_top_this_week:
            profile.consecutive_top_weeks += 1
            profile.consecutive_bottom_weeks = 0
        elif is_bottom_this_week:
            profile.consecutive_bottom_weeks += 1
            profile.consecutive_top_weeks = 0

        # Update win/loss streak based on weekly return
        profile.update_streak(weekly_return > 0)

        # Assign tier
        profile.tier = self._assign_tier(profile)

        # Adjust signal weight
        self._update_weight(profile)

        # Record to performance database
        record = PerformanceRecord(
            timestamp=datetime.now().isoformat(),
            weekly_return=weekly_return,
            accuracy=accuracy,
            sharpe=sharpe,
            hit_rate=hit_rate,
            composite_score=profile.composite_score,
            signals_generated=signals_generated,
            signals_correct=signals_correct,
            regime=regime,
        )
        profile.history.append(record)
        if len(profile.history) > MAX_WEEKLY_HISTORY:
            profile.history = profile.history[-MAX_WEEKLY_HISTORY:]

        self.perf_db.record(agent_name, record)

    def batch_update_scores(self, scores: Dict[str, dict]):
        """Update scores for multiple agents at once.

        scores: {agent_name: {accuracy, sharpe, hit_rate, weekly_return, ...}}
        """
        # Determine top and bottom performers this week
        composites = {}
        for name, data in scores.items():
            acc = data.get("accuracy", 0.0)
            sh = data.get("sharpe", 0.0)
            hr = data.get("hit_rate", 0.0)
            sh_norm = min(sh / SHARPE_NORM_CAP, 1.0)
            composites[name] = 0.40 * acc + 0.30 * sh_norm + 0.30 * hr

        if composites:
            sorted_agents = sorted(composites.items(), key=lambda x: x[1], reverse=True)
            top_name = sorted_agents[0][0]
            bottom_name = sorted_agents[-1][0]
        else:
            top_name = bottom_name = None

        for name, data in scores.items():
            self.update_agent_score(
                agent_name=name,
                accuracy=data.get("accuracy", 0.0),
                sharpe=data.get("sharpe", 0.0),
                hit_rate=data.get("hit_rate", 0.0),
                weekly_return=data.get("weekly_return", 0.0),
                signals_generated=data.get("signals_generated", 0),
                signals_correct=data.get("signals_correct", 0),
                is_top_this_week=(name == top_name),
                is_bottom_this_week=(name == bottom_name),
                regime=data.get("regime", "RANGE"),
            )

    # ------------------------------------------------------------------
    # Leaderboard and reporting
    # ------------------------------------------------------------------
    def get_leaderboard(self) -> List[Tuple[str, AgentProfile]]:
        """Return agents sorted by composite score descending."""
        return sorted(
            self.agents.items(),
            key=lambda x: x[1].composite_score,
            reverse=True,
        )

    def get_tier_members(self, tier: AgentTierLevel) -> List[str]:
        """Return all agents in a given tier."""
        return [
            name for name, p in self.agents.items()
            if p.tier == tier
        ]

    def get_agents_by_category(self, category: AgentCategory) -> List[str]:
        """Return agent names for a given category."""
        return [
            name for name, p in self.agents.items()
            if p.category == category
        ]

    def get_active_agents(self) -> List[str]:
        """Return names of all active agents."""
        return [name for name, p in self.agents.items() if p.is_active]

    def deactivate_agent(self, name: str):
        """Deactivate an agent (stop receiving signals from it)."""
        if name in self.agents:
            self.agents[name].is_active = False

    def activate_agent(self, name: str):
        """Activate a previously deactivated agent."""
        if name in self.agents:
            self.agents[name].is_active = True

    def print_leaderboard(self) -> str:
        """ASCII formatted leaderboard with Rank, tiers, and scores.

        Returns a formatted string containing rankings for all agents.
        """
        lines = []
        header_width = 105
        lines.append("=" * header_width)
        lines.append(f"{'METADRON CAPITAL — AGENT LEADERBOARD':^{header_width}}")
        lines.append(f"{'Generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M'):^{header_width}}")
        lines.append("=" * header_width)

        lines.append(
            f"| {'Rank':>4} | {'Agent':<22} | {'Category':<18} | "
            f"{'Score':>7} | {'Sharpe':>7} | {'Acc':>6} | {'Hit':>6} | {'Tier':<20} |"
        )
        lines.append("-" * header_width)

        for i, (name, profile) in enumerate(self.get_leaderboard()):
            tier_str = profile.tier if isinstance(profile.tier, str) else profile.tier.value
            cat_str = profile.category if isinstance(profile.category, str) else profile.category.value
            # Truncate category for display
            cat_display = cat_str[:18]
            lines.append(
                f"| {i + 1:>4} | {name:<22} | {cat_display:<18} | "
                f"{profile.composite_score:>7.4f} | {profile.sharpe:>7.2f} | "
                f"{profile.accuracy:>5.1%} | {profile.hit_rate:>5.1%} | {tier_str:<20} |"
            )

        lines.append("=" * header_width)

        # Tier summary
        lines.append("")
        lines.append("Tier Summary:")
        for tier in TIER_ORDER:
            members = self.get_tier_members(tier)
            tier_str = tier.value if isinstance(tier, Enum) else tier
            lines.append(f"  {tier_str}: {len(members)} agents")

        return "\n".join(lines)

    def print_tier_report(self) -> str:
        """Detailed report organized by tier."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{'AGENT TIER REPORT':^80}")
        lines.append("=" * 80)

        for tier in TIER_ORDER:
            tier_str = tier.value if isinstance(tier, Enum) else tier
            members = self.get_tier_members(tier)
            lines.append("")
            lines.append(f"--- {tier_str} ({len(members)} agents) ---")

            if not members:
                lines.append("  (no agents)")
                continue

            for name in sorted(members):
                p = self.agents[name]
                lines.append(
                    f"  {name:<22} | Score: {p.composite_score:.4f} | "
                    f"Sharpe: {p.sharpe:.2f} | Acc: {p.accuracy:.1%} | "
                    f"W/L: {p.win_streak}/{p.loss_streak} | Wt: {p.weight:.2f}"
                )

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Signal aggregation and consensus
    # ------------------------------------------------------------------
    def submit_signal(self, signal: AgentSignal):
        """Submit a signal from an agent into the buffer."""
        if signal.agent_name not in self.agents:
            logger.warning(f"Signal from unknown agent: {signal.agent_name}")
            return
        profile = self.agents[signal.agent_name]
        if not profile.is_active:
            return
        # Apply agent weight to signal
        signal.weight = profile.weight * signal.confidence
        self._signal_buffer[signal.ticker].append(signal)

    def clear_signal_buffer(self):
        """Clear all buffered signals."""
        self._signal_buffer.clear()

    def get_consensus(self, ticker: str) -> ConsensusResult:
        """Compute consensus vote for a given ticker.

        Aggregates signals from all agents that submitted for this ticker.
        Uses weighted voting where weights depend on agent tier and confidence.

        Resolution methods:
        - "unanimous": All agents agree
        - "supermajority": >75% weighted agreement
        - "majority": >50% weighted agreement
        - "tie_break_by_generals": Generals break ties
        - "abstain": No clear consensus
        """
        signals = self._signal_buffer.get(ticker, [])
        if not signals:
            return ConsensusResult(ticker=ticker, direction="HOLD", resolution_method="no_signals")

        votes_buy = 0.0
        votes_sell = 0.0
        votes_hold = 0.0
        count_buy = 0
        count_sell = 0
        count_hold = 0
        participants = []
        conflicts = []

        for sig in signals:
            w = sig.weight
            participants.append(sig.agent_name)

            if sig.direction == "BUY":
                votes_buy += w
                count_buy += 1
            elif sig.direction == "SELL":
                votes_sell += w
                count_sell += 1
            else:
                votes_hold += w
                count_hold += 1

        total_weight = votes_buy + votes_sell + votes_hold
        total_votes = count_buy + count_sell + count_hold

        if total_weight == 0:
            return ConsensusResult(
                ticker=ticker, direction="HOLD",
                total_votes=total_votes,
                participating_agents=participants,
                resolution_method="zero_weight",
            )

        # Determine direction by weighted vote
        buy_pct = votes_buy / total_weight
        sell_pct = votes_sell / total_weight
        hold_pct = votes_hold / total_weight

        # Check for conflicts (BUY and SELL both significant)
        if buy_pct > 0.20 and sell_pct > 0.20:
            # Identify conflicting agents
            buy_agents = [s.agent_name for s in signals if s.direction == "BUY"]
            sell_agents = [s.agent_name for s in signals if s.direction == "SELL"]
            conflicts = [
                f"BUY agents: {', '.join(buy_agents)}",
                f"SELL agents: {', '.join(sell_agents)}",
            ]

        # Resolution
        max_pct = max(buy_pct, sell_pct, hold_pct)

        if buy_pct == sell_pct and buy_pct >= hold_pct:
            # Tie between BUY and SELL -> generals break tie
            direction, method = self._tie_break_by_generals(signals)
        elif max_pct > 0.75:
            method = "supermajority"
            direction = "BUY" if buy_pct == max_pct else ("SELL" if sell_pct == max_pct else "HOLD")
        elif max_pct > 0.50:
            method = "majority"
            direction = "BUY" if buy_pct == max_pct else ("SELL" if sell_pct == max_pct else "HOLD")
        else:
            # No clear majority -> generals break tie or abstain
            direction, method = self._tie_break_by_generals(signals)

        # Calculate agreement percentage
        if direction == "BUY":
            agreement = buy_pct
        elif direction == "SELL":
            agreement = sell_pct
        else:
            agreement = hold_pct

        result = ConsensusResult(
            ticker=ticker,
            direction=direction,
            consensus_score=round(max_pct, 4),
            votes_buy=count_buy,
            votes_sell=count_sell,
            votes_hold=count_hold,
            total_votes=total_votes,
            agreement_pct=round(agreement * 100, 1),
            weighted_confidence=round(total_weight / total_votes if total_votes > 0 else 0.0, 4),
            participating_agents=participants,
            conflicts=conflicts,
            resolution_method=method,
        )
        self._consensus_history.append(result)
        return result

    def _tie_break_by_generals(
        self, signals: List[AgentSignal]
    ) -> Tuple[str, str]:
        """Break ties using only GENERAL-tier agents' votes.

        If no generals voted, default to HOLD (abstain).
        """
        general_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        has_generals = False

        for sig in signals:
            if sig.agent_name in self.agents:
                profile = self.agents[sig.agent_name]
                if profile.tier == AgentTierLevel.TIER_1_GENERAL:
                    general_votes[sig.direction] += sig.weight
                    has_generals = True

        if not has_generals:
            return "HOLD", "abstain"

        best = max(general_votes, key=general_votes.get)
        return best, "tie_break_by_generals"

    def get_all_consensus(self) -> Dict[str, ConsensusResult]:
        """Compute consensus for all tickers in the signal buffer."""
        results = {}
        for ticker in self._signal_buffer:
            results[ticker] = self.get_consensus(ticker)
        return results

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------
    def detect_conflicts(self) -> Dict[str, List[str]]:
        """Detect tickers where agents have conflicting signals.

        Returns {ticker: [conflict descriptions]}.
        """
        conflicts = {}
        for ticker, signals in self._signal_buffer.items():
            directions = set(s.direction for s in signals)
            if "BUY" in directions and "SELL" in directions:
                buy_agents = [s.agent_name for s in signals if s.direction == "BUY"]
                sell_agents = [s.agent_name for s in signals if s.direction == "SELL"]
                conflicts[ticker] = [
                    f"BUY: {', '.join(buy_agents)}",
                    f"SELL: {', '.join(sell_agents)}",
                ]
        return conflicts

    def resolve_conflict(
        self, ticker: str, method: str = "weighted_vote"
    ) -> str:
        """Resolve a conflict for a given ticker.

        Methods:
        - "weighted_vote": Highest weighted direction wins
        - "generals_only": Only GENERAL tier votes count
        - "highest_confidence": Single highest confidence signal wins
        - "conservative": Default to HOLD if conflicted
        """
        signals = self._signal_buffer.get(ticker, [])
        if not signals:
            return "HOLD"

        if method == "weighted_vote":
            result = self.get_consensus(ticker)
            return result.direction

        elif method == "generals_only":
            general_signals = [
                s for s in signals
                if s.agent_name in self.agents
                and self.agents[s.agent_name].tier == AgentTierLevel.TIER_1_GENERAL
            ]
            if not general_signals:
                return "HOLD"
            direction_weights = defaultdict(float)
            for s in general_signals:
                direction_weights[s.direction] += s.weight
            return max(direction_weights, key=direction_weights.get)

        elif method == "highest_confidence":
            best_signal = max(signals, key=lambda s: s.confidence)
            return best_signal.direction

        elif method == "conservative":
            return "HOLD"

        else:
            return "HOLD"

    # ------------------------------------------------------------------
    # Regime-aware agent selection
    # ------------------------------------------------------------------
    def get_regime_specialists(self, regime: CubeRegime) -> List[str]:
        """Return agents best suited for the current market regime.

        Different regimes favor different agent types:
        - TRENDING: MomentumRider, GrowthSeeker, TechAnalyst
        - RANGE: MeanReversionBot, ValueHunter, DividendCollector
        - STRESS: RiskAnalyst, HedgeBot, MacroStrategist
        - CRASH: RiskAnalyst, HedgeBot, VolatilityTrader
        """
        regime_map = {
            CubeRegime.TRENDING: [
                "MomentumRider", "GrowthSeeker", "TechAnalyst", "AlphaBot",
                "MacroBot", "CubeBot",
            ],
            CubeRegime.RANGE: [
                "MeanReversionBot", "ValueHunter", "DividendCollector",
                "QualityScreener", "CorrelationTracker",
            ],
            CubeRegime.STRESS: [
                "RiskAnalyst", "HedgeBot", "MacroStrategist",
                "VolatilityTrader", "LiquidityMonitor", "BetaBot",
            ],
            CubeRegime.CRASH: [
                "RiskAnalyst", "HedgeBot", "VolatilityTrader",
                "MacroStrategist", "BetaBot", "LiquidityMonitor",
            ],
        }
        specialists = regime_map.get(regime, [])
        # Filter to active agents only
        return [name for name in specialists if name in self.agents and self.agents[name].is_active]

    def adjust_weights_for_regime(self, regime: CubeRegime):
        """Boost weights for regime-specialist agents, reduce others."""
        specialists = set(self.get_regime_specialists(regime))

        for name, profile in self.agents.items():
            if name in specialists:
                profile.weight *= 1.25  # 25% boost for regime specialists
            else:
                profile.weight *= 0.85  # 15% reduction for non-specialists

            # Clamp weights
            profile.weight = round(max(0.1, min(3.0, profile.weight)), 3)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        """Persist full scorecard state to JSON."""
        week = datetime.now().strftime("%Y%W")

        # Save individual agent scores
        agent_data = {}
        for name, profile in self.agents.items():
            agent_data[name] = {
                "category": profile.category if isinstance(profile.category, str) else profile.category.value,
                "tier": profile.tier if isinstance(profile.tier, str) else profile.tier.value,
                "accuracy": profile.accuracy,
                "sharpe": profile.sharpe,
                "hit_rate": profile.hit_rate,
                "composite_score": profile.composite_score,
                "weight": profile.weight,
                "consecutive_top_weeks": profile.consecutive_top_weeks,
                "consecutive_bottom_weeks": profile.consecutive_bottom_weeks,
                "win_streak": profile.win_streak,
                "loss_streak": profile.loss_streak,
                "total_signals": profile.total_signals,
                "correct_signals": profile.correct_signals,
                "is_active": profile.is_active,
            }
        (self.log_dir / f"agents_{week}.json").write_text(
            json.dumps(agent_data, indent=2)
        )

        # Save leaderboard
        lb = []
        for i, (name, profile) in enumerate(self.get_leaderboard()):
            tier_str = profile.tier if isinstance(profile.tier, str) else profile.tier.value
            lb.append({
                "rank": i + 1,
                "agent": name,
                "score": profile.composite_score,
                "tier": tier_str,
                "sharpe": profile.sharpe,
                "accuracy": profile.accuracy,
            })
        (self.log_dir / "leaderboard.json").write_text(
            json.dumps(lb, indent=2)
        )

    def load(self, week: Optional[str] = None):
        """Load scorecard state from JSON."""
        if week is None:
            week = datetime.now().strftime("%Y%W")
        filepath = self.log_dir / f"agents_{week}.json"
        if not filepath.exists():
            logger.warning(f"No saved state for week {week}")
            return

        data = json.loads(filepath.read_text())
        for name, values in data.items():
            if name in self.agents:
                profile = self.agents[name]
                profile.accuracy = values.get("accuracy", 0.0)
                profile.sharpe = values.get("sharpe", 0.0)
                profile.hit_rate = values.get("hit_rate", 0.0)
                profile.composite_score = values.get("composite_score", 0.0)
                profile.weight = values.get("weight", 1.0)
                profile.tier = values.get("tier", AgentTierLevel.TIER_4_RECRUIT)
                profile.is_active = values.get("is_active", True)
                profile.consecutive_top_weeks = values.get("consecutive_top_weeks", 0)
                profile.consecutive_bottom_weeks = values.get("consecutive_bottom_weeks", 0)
                profile.win_streak = values.get("win_streak", 0)
                profile.loss_streak = values.get("loss_streak", 0)
                profile.total_signals = values.get("total_signals", 0)
                profile.correct_signals = values.get("correct_signals", 0)

    # ------------------------------------------------------------------
    # Statistics and analysis
    # ------------------------------------------------------------------
    def get_system_stats(self) -> dict:
        """Aggregate statistics across all agents."""
        total = len(self.agents)
        active = len(self.get_active_agents())
        tier_counts = {tier.value: len(self.get_tier_members(tier)) for tier in TIER_ORDER}
        category_counts = {
            cat.value: len(self.get_agents_by_category(cat))
            for cat in AgentCategory
        }

        composites = [p.composite_score for p in self.agents.values() if p.composite_score > 0]
        avg_composite = statistics.mean(composites) if composites else 0.0
        avg_accuracy = statistics.mean([p.accuracy for p in self.agents.values()]) if self.agents else 0.0
        avg_sharpe = statistics.mean([p.sharpe for p in self.agents.values()]) if self.agents else 0.0

        return {
            "total_agents": total,
            "active_agents": active,
            "tier_distribution": tier_counts,
            "category_distribution": category_counts,
            "avg_composite_score": round(avg_composite, 4),
            "avg_accuracy": round(avg_accuracy, 4),
            "avg_sharpe": round(avg_sharpe, 2),
            "total_signals_system": sum(p.total_signals for p in self.agents.values()),
            "total_correct_system": sum(p.correct_signals for p in self.agents.values()),
            "consensus_history_size": len(self._consensus_history),
        }

    def get_agent_rankings_by_metric(
        self, metric: str = "composite_score", ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """Rank agents by a specific metric.

        Available metrics: composite_score, accuracy, sharpe, hit_rate,
        weight, win_streak, total_signals.
        """
        valid_metrics = {
            "composite_score", "accuracy", "sharpe", "hit_rate",
            "weight", "win_streak", "loss_streak", "total_signals",
        }
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Choose from: {valid_metrics}")

        rankings = [
            (name, getattr(profile, metric, 0.0))
            for name, profile in self.agents.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings

    def get_top_agents(self, n: int = 5) -> List[Tuple[str, AgentProfile]]:
        """Return top N agents by composite score."""
        return self.get_leaderboard()[:n]

    def get_bottom_agents(self, n: int = 5) -> List[Tuple[str, AgentProfile]]:
        """Return bottom N agents by composite score."""
        return self.get_leaderboard()[-n:]

    def get_agent_profile(self, name: str) -> Optional[AgentProfile]:
        """Get full profile for a single agent."""
        return self.agents.get(name)

    def print_agent_detail(self, name: str) -> str:
        """Detailed report for a single agent."""
        profile = self.agents.get(name)
        if not profile:
            return f"Agent '{name}' not found."

        tier_str = profile.tier if isinstance(profile.tier, str) else profile.tier.value
        cat_str = profile.category if isinstance(profile.category, str) else profile.category.value

        lines = [
            "=" * 60,
            f"Agent: {name}",
            "=" * 60,
            f"  Category:    {cat_str}",
            f"  Strategy:    {profile.strategy}",
            f"  Style:       {profile.style}",
            f"  Description: {profile.description}",
            f"  Tier:        {tier_str}",
            f"  Active:      {profile.is_active}",
            f"  Weight:      {profile.weight:.3f}",
            "",
            "  Performance:",
            f"    Composite Score: {profile.composite_score:.4f}",
            f"    Accuracy:        {profile.accuracy:.1%}",
            f"    Sharpe:          {profile.sharpe:.2f}",
            f"    Hit Rate:        {profile.hit_rate:.1%}",
            f"    Rolling Sharpe:  {profile.rolling_sharpe():.2f}",
            "",
            "  Streaks:",
            f"    Win Streak:              {profile.win_streak}",
            f"    Loss Streak:             {profile.loss_streak}",
            f"    Max Win Streak:          {profile.max_win_streak}",
            f"    Max Loss Streak:         {profile.max_loss_streak}",
            f"    Consecutive Top Weeks:   {profile.consecutive_top_weeks}",
            f"    Consecutive Bottom Weeks:{profile.consecutive_bottom_weeks}",
            "",
            "  Signal Stats:",
            f"    Total Signals:    {profile.total_signals}",
            f"    Correct Signals:  {profile.correct_signals}",
            "",
            "  Sector Bias: {0}".format(", ".join(profile.sector_bias) if profile.sector_bias else "None"),
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_consensus_report(self) -> str:
        """Report of all recent consensus results."""
        if not self._consensus_history:
            return "No consensus history available."

        lines = []
        lines.append("=" * 90)
        lines.append(f"{'CONSENSUS VOTING REPORT':^90}")
        lines.append("=" * 90)
        lines.append(
            f"| {'Ticker':<8} | {'Direction':^9} | {'Score':>6} | "
            f"{'B/S/H':^9} | {'Agree':>6} | {'Method':<25} |"
        )
        lines.append("-" * 90)

        for result in self._consensus_history[-20:]:  # Last 20
            bsh = f"{result.votes_buy}/{result.votes_sell}/{result.votes_hold}"
            lines.append(
                f"| {result.ticker:<8} | {result.direction:^9} | "
                f"{result.consensus_score:>6.2f} | {bsh:^9} | "
                f"{result.agreement_pct:>5.1f}% | {result.resolution_method:<25} |"
            )

        lines.append("=" * 90)
        return "\n".join(lines)
