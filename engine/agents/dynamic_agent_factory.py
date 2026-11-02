"""Dynamic Agent Factory — Create and enforce agents on-the-fly.

Integrates the Paul Plugin (Pattern Awareness & Unified Learning) with
Ruflo-agents orchestration to dynamically create, configure, and enforce
agents across the Metadron Capital intelligence platform.

Capabilities:
    - Spawn new agents from templates (sector bots, research bots, personas)
    - Clone and specialize existing agents for sub-sector or ticker focus
    - Enforce performance thresholds via GSD gradient monitoring
    - Auto-demote/promote agents based on real-time gradient signals
    - Dynamic rebalancing of agent weights via Paul pattern matching
    - Registry of all active agents with lifecycle management

Integration points:
    - GSDPlugin (gradient-driven enforcement)
    - PaulPlugin (pattern-driven learning)
    - AgentLearningWrapper (pre/post decision hooks)
    - AgentScorecard (tier management)
    - AgentMonitor (performance tracking)
    - SectorBots, ResearchBots, InvestorPersonas (concrete agents)
"""

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent templates
# ---------------------------------------------------------------------------
class AgentTemplate(str, Enum):
    """Available agent templates for dynamic creation."""
    SECTOR_BOT = "sector_bot"
    RESEARCH_BOT = "research_bot"
    INVESTOR_PERSONA = "investor_persona"
    SPECIALIST = "specialist"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class AgentLifecycleState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    PROBATION = "PROBATION"
    TERMINATED = "TERMINATED"


class EnforcementAction(str, Enum):
    """Actions the enforcement engine can take on an agent."""
    NONE = "NONE"
    WARN = "WARN"
    REDUCE_WEIGHT = "REDUCE_WEIGHT"
    SUSPEND = "SUSPEND"
    TERMINATE = "TERMINATE"
    PROMOTE = "PROMOTE"
    BOOST_WEIGHT = "BOOST_WEIGHT"


# ---------------------------------------------------------------------------
# Agent specification
# ---------------------------------------------------------------------------
@dataclass
class AgentSpec:
    """Full specification for creating a dynamic agent."""
    agent_id: str = ""
    name: str = ""
    template: str = AgentTemplate.CUSTOM
    focus: str = ""                  # sector, ticker, strategy focus
    description: str = ""

    # Capabilities
    signal_engines: list = field(default_factory=list)  # which engines to listen to
    allowed_directions: list = field(default_factory=lambda: ["BUY", "SELL", "HOLD"])
    max_position_size: float = 0.05  # 5% max per-position
    max_daily_signals: int = 50
    min_confidence: float = 0.3

    # Enforcement thresholds
    min_accuracy: float = 0.45
    min_sharpe: float = 0.5
    max_drawdown: float = -0.15
    probation_threshold: float = 0.35  # accuracy below this → probation
    termination_threshold: float = 0.25  # accuracy below this → terminate

    # Learning config
    learning_rate: float = 0.01
    gradient_sensitivity: float = 1.0
    pattern_replay_enabled: bool = True
    max_pattern_age_days: int = 90

    # Metadata
    created_at: str = ""
    created_by: str = "factory"
    parent_agent_id: str = ""  # if cloned from another agent
    tags: list = field(default_factory=list)


@dataclass
class DynamicAgent:
    """A dynamically created agent with full lifecycle management."""
    spec: AgentSpec = field(default_factory=AgentSpec)
    state: str = AgentLifecycleState.INITIALIZING
    weight: float = 1.0
    tier: str = "TIER_4_Recruit"

    # Runtime stats
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_realized: float = 0.0

    # Enforcement
    warnings_issued: int = 0
    suspensions: int = 0
    last_enforcement_action: str = EnforcementAction.NONE
    last_enforcement_time: str = ""

    # Learning
    gsd_confidence: float = 0.5
    pattern_matches_used: int = 0
    gradient_trend: str = "neutral"

    # History
    pnl_history: list = field(default_factory=list)
    enforcement_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enforcement rules
# ---------------------------------------------------------------------------
@dataclass
class EnforcementRule:
    """A rule that triggers enforcement actions on agents."""
    rule_id: str = ""
    name: str = ""
    condition: str = ""           # Python expression evaluated against agent
    action: str = EnforcementAction.NONE
    cooldown_minutes: int = 60    # min time between enforcement actions
    enabled: bool = True


DEFAULT_ENFORCEMENT_RULES = [
    EnforcementRule(
        rule_id="accuracy_warn",
        name="Low accuracy warning",
        condition="accuracy < min_accuracy and total_signals >= 10",
        action=EnforcementAction.WARN,
        cooldown_minutes=60,
    ),
    EnforcementRule(
        rule_id="accuracy_reduce",
        name="Accuracy below threshold — reduce weight",
        condition="accuracy < probation_threshold and total_signals >= 20",
        action=EnforcementAction.REDUCE_WEIGHT,
        cooldown_minutes=120,
    ),
    EnforcementRule(
        rule_id="accuracy_suspend",
        name="Accuracy critical — suspend agent",
        condition="accuracy < termination_threshold and total_signals >= 30",
        action=EnforcementAction.SUSPEND,
        cooldown_minutes=240,
    ),
    EnforcementRule(
        rule_id="drawdown_suspend",
        name="Max drawdown breached — suspend",
        condition="max_drawdown_realized < max_drawdown",
        action=EnforcementAction.SUSPEND,
        cooldown_minutes=480,
    ),
    EnforcementRule(
        rule_id="gradient_warn",
        name="Gradient declining — warn",
        condition="gsd_confidence < 0.3 and total_signals >= 10",
        action=EnforcementAction.WARN,
        cooldown_minutes=60,
    ),
    EnforcementRule(
        rule_id="elite_promote",
        name="Elite performance — promote",
        condition="accuracy > 0.80 and sharpe_ratio > 2.0 and total_signals >= 50",
        action=EnforcementAction.PROMOTE,
        cooldown_minutes=1440,
    ),
    EnforcementRule(
        rule_id="strong_boost",
        name="Strong performer — boost weight",
        condition="accuracy > 0.65 and sharpe_ratio > 1.5 and total_signals >= 30",
        action=EnforcementAction.BOOST_WEIGHT,
        cooldown_minutes=720,
    ),
]


# ---------------------------------------------------------------------------
# Dynamic Agent Factory
# ---------------------------------------------------------------------------
class DynamicAgentFactory:
    """Factory for creating, managing, and enforcing dynamic agents.

    Integrates with GSD + Paul plugins to provide gradient-driven
    enforcement and pattern-aware learning for all agents.

    Usage:
        from intelligence_platform.plugins.gsd_paul_plugin import (
            GSDPlugin, PaulPlugin, AgentLearningWrapper,
        )

        gsd = GSDPlugin()
        paul = PaulPlugin()
        wrapper = AgentLearningWrapper(gsd, paul)

        factory = DynamicAgentFactory(gsd=gsd, paul=paul, wrapper=wrapper)

        # Create a sector specialist
        spec = AgentSpec(
            name="Tech Alpha Hunter",
            template=AgentTemplate.SPECIALIST,
            focus="Information Technology",
            signal_engines=["macro", "cube", "alpha_optimizer"],
        )
        agent = factory.create_agent(spec)

        # Clone an existing agent with modifications
        clone = factory.clone_agent(agent.spec.agent_id, focus="Semiconductors")

        # Enforce all agents
        actions = factory.enforce_all()
    """

    def __init__(
        self,
        gsd=None,
        paul=None,
        wrapper=None,
        enforcement_rules: Optional[list] = None,
        log_dir: Optional[Path] = None,
    ):
        self._lock = threading.RLock()
        self._gsd = gsd
        self._paul = paul
        self._wrapper = wrapper
        self._log_dir = log_dir or Path("logs/agent_factory")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Agent registry: agent_id -> DynamicAgent
        self._agents: dict[str, DynamicAgent] = {}

        # Index by template type
        self._template_index: dict[str, list[str]] = defaultdict(list)

        # Index by focus (sector, ticker, etc.)
        self._focus_index: dict[str, list[str]] = defaultdict(list)

        # Enforcement rules
        self._rules = enforcement_rules or DEFAULT_ENFORCEMENT_RULES

        # Enforcement cooldown tracking: (agent_id, rule_id) -> last_action_time
        self._cooldowns: dict[tuple, datetime] = {}

        # Counters
        self._total_created = 0
        self._total_terminated = 0

    # --- Agent creation -----------------------------------------------------

    def create_agent(self, spec: AgentSpec) -> DynamicAgent:
        """Create a new dynamic agent from a specification.

        Args:
            spec: AgentSpec defining the agent's capabilities and thresholds.

        Returns:
            Created DynamicAgent instance.
        """
        if not spec.agent_id:
            spec.agent_id = f"dyn_{spec.template}_{uuid.uuid4().hex[:8]}"
        spec.created_at = spec.created_at or datetime.now().isoformat()

        agent = DynamicAgent(
            spec=spec,
            state=AgentLifecycleState.ACTIVE,
        )

        # Register with GSD + Paul via wrapper
        if self._wrapper is not None:
            self._wrapper.attach_to_agent(agent)

        with self._lock:
            self._agents[spec.agent_id] = agent
            self._template_index[spec.template].append(spec.agent_id)
            if spec.focus:
                self._focus_index[spec.focus].append(spec.agent_id)
            self._total_created += 1

        self._log_event("agent_created", {
            "agent_id": spec.agent_id,
            "name": spec.name,
            "template": spec.template,
            "focus": spec.focus,
        })

        logger.info(
            "DynamicAgentFactory: created agent %s (template=%s, focus=%s)",
            spec.agent_id, spec.template, spec.focus,
        )
        return agent

    def clone_agent(
        self,
        source_agent_id: str,
        name: Optional[str] = None,
        focus: Optional[str] = None,
        **overrides,
    ) -> Optional[DynamicAgent]:
        """Clone an existing agent with optional modifications.

        Args:
            source_agent_id: ID of agent to clone.
            name: Override name for the clone.
            focus: Override focus for the clone.
            **overrides: Additional AgentSpec field overrides.

        Returns:
            New DynamicAgent clone, or None if source not found.
        """
        with self._lock:
            source = self._agents.get(source_agent_id)
            if source is None:
                logger.warning("Cannot clone: agent %s not found", source_agent_id)
                return None

        # Copy spec
        spec_dict = asdict(source.spec)
        spec_dict["agent_id"] = ""  # will be auto-generated
        spec_dict["parent_agent_id"] = source_agent_id
        spec_dict["created_at"] = ""
        spec_dict["created_by"] = "clone"

        if name:
            spec_dict["name"] = name
        if focus:
            spec_dict["focus"] = focus
        spec_dict.update(overrides)

        new_spec = AgentSpec(**spec_dict)
        return self.create_agent(new_spec)

    def create_sector_specialist(
        self,
        sector: str,
        name: Optional[str] = None,
    ) -> DynamicAgent:
        """Convenience: create a sector-focused specialist agent."""
        spec = AgentSpec(
            name=name or f"{sector} Specialist",
            template=AgentTemplate.SECTOR_BOT,
            focus=sector,
            signal_engines=[
                "macro", "cube", "security_analysis", "alpha_optimizer",
                "decision_matrix",
            ],
            tags=["sector", sector.lower().replace(" ", "_")],
        )
        return self.create_agent(spec)

    def create_ticker_specialist(
        self,
        ticker: str,
        sector: str = "",
        name: Optional[str] = None,
    ) -> DynamicAgent:
        """Convenience: create a ticker-focused specialist agent."""
        spec = AgentSpec(
            name=name or f"{ticker} Deep Analyst",
            template=AgentTemplate.SPECIALIST,
            focus=ticker,
            signal_engines=[
                "security_analysis", "pattern_discovery", "event_driven",
                "alpha_optimizer", "distress",
            ],
            max_position_size=0.03,
            min_confidence=0.4,
            tags=["ticker", ticker, sector.lower().replace(" ", "_")],
        )
        return self.create_agent(spec)

    def create_strategy_agent(
        self,
        strategy: str,
        name: Optional[str] = None,
    ) -> DynamicAgent:
        """Convenience: create a strategy-focused agent."""
        strategy_engines = {
            "momentum": ["macro", "cube", "alpha_optimizer"],
            "mean_reversion": ["stat_arb", "alpha_optimizer"],
            "event_driven": ["event_driven", "cvr", "distress"],
            "social": ["social", "pattern_discovery"],
            "macro": ["macro", "cube", "security_analysis"],
        }
        spec = AgentSpec(
            name=name or f"{strategy.title()} Strategy Agent",
            template=AgentTemplate.SPECIALIST,
            focus=strategy,
            signal_engines=strategy_engines.get(
                strategy, ["alpha_optimizer", "decision_matrix"]
            ),
            tags=["strategy", strategy],
        )
        return self.create_agent(spec)

    # --- Agent lifecycle management -----------------------------------------

    def suspend_agent(self, agent_id: str, reason: str = "") -> bool:
        """Suspend an agent (stop generating signals)."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            agent.state = AgentLifecycleState.SUSPENDED
            agent.suspensions += 1
            agent.enforcement_log.append({
                "action": "SUSPEND",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
        self._log_event("agent_suspended", {
            "agent_id": agent_id, "reason": reason,
        })
        return True

    def reactivate_agent(self, agent_id: str) -> bool:
        """Reactivate a suspended agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None or agent.state not in (
                AgentLifecycleState.SUSPENDED,
                AgentLifecycleState.PROBATION,
            ):
                return False
            agent.state = AgentLifecycleState.ACTIVE
            agent.enforcement_log.append({
                "action": "REACTIVATE",
                "timestamp": datetime.now().isoformat(),
            })
        return True

    def terminate_agent(self, agent_id: str, reason: str = "") -> bool:
        """Permanently terminate an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            agent.state = AgentLifecycleState.TERMINATED
            agent.enforcement_log.append({
                "action": "TERMINATE",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
            self._total_terminated += 1
        self._log_event("agent_terminated", {
            "agent_id": agent_id, "reason": reason,
        })
        return True

    # --- Enforcement engine -------------------------------------------------

    def enforce_agent(self, agent_id: str) -> list[dict]:
        """Run enforcement rules against a single agent.

        Returns list of actions taken.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None or agent.state == AgentLifecycleState.TERMINATED:
                return []

        # Refresh GSD confidence
        if self._gsd is not None:
            agent.gsd_confidence = self._gsd.get_gradient_confidence(agent_id)

        actions_taken = []
        now = datetime.now()

        for rule in self._rules:
            if not rule.enabled:
                continue

            # Check cooldown
            cooldown_key = (agent_id, rule.rule_id)
            last_action = self._cooldowns.get(cooldown_key)
            if last_action is not None:
                elapsed = (now - last_action).total_seconds() / 60
                if elapsed < rule.cooldown_minutes:
                    continue

            # Evaluate condition
            if self._evaluate_rule(agent, rule):
                action = self._execute_enforcement(agent, rule)
                if action:
                    self._cooldowns[cooldown_key] = now
                    actions_taken.append(action)

        return actions_taken

    def enforce_all(self) -> dict[str, list[dict]]:
        """Run enforcement rules against all active agents.

        Returns dict of agent_id -> actions taken.
        """
        with self._lock:
            active_ids = [
                aid for aid, a in self._agents.items()
                if a.state in (
                    AgentLifecycleState.ACTIVE,
                    AgentLifecycleState.PROBATION,
                )
            ]

        results = {}
        for aid in active_ids:
            actions = self.enforce_agent(aid)
            if actions:
                results[aid] = actions

        if results:
            self._log_event("enforcement_sweep", {
                "agents_checked": len(active_ids),
                "agents_actioned": len(results),
                "total_actions": sum(len(a) for a in results.values()),
            })

        return results

    def _evaluate_rule(self, agent: DynamicAgent, rule: EnforcementRule) -> bool:
        """Evaluate a rule condition against an agent's current state."""
        try:
            # Build evaluation context from agent + spec
            ctx = {
                "accuracy": agent.accuracy,
                "sharpe_ratio": agent.sharpe_ratio,
                "total_signals": agent.total_signals,
                "total_pnl": agent.total_pnl,
                "max_drawdown_realized": agent.max_drawdown_realized,
                "gsd_confidence": agent.gsd_confidence,
                "warnings_issued": agent.warnings_issued,
                "weight": agent.weight,
                # From spec thresholds
                "min_accuracy": agent.spec.min_accuracy,
                "min_sharpe": agent.spec.min_sharpe,
                "max_drawdown": agent.spec.max_drawdown,
                "probation_threshold": agent.spec.probation_threshold,
                "termination_threshold": agent.spec.termination_threshold,
            }
            return bool(eval(rule.condition, {"__builtins__": {}}, ctx))  # noqa: S307
        except Exception as e:
            logger.debug("Rule evaluation failed: %s — %s", rule.rule_id, e)
            return False

    def _execute_enforcement(
        self, agent: DynamicAgent, rule: EnforcementRule,
    ) -> Optional[dict]:
        """Execute an enforcement action on an agent."""
        action = rule.action
        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "action": action,
            "agent_id": agent.spec.agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        if action == EnforcementAction.WARN:
            agent.warnings_issued += 1
            agent.last_enforcement_action = action
            logger.info("ENFORCEMENT WARNING: %s — %s", agent.spec.agent_id, rule.name)

        elif action == EnforcementAction.REDUCE_WEIGHT:
            old_weight = agent.weight
            agent.weight = max(0.1, agent.weight * 0.7)
            agent.state = AgentLifecycleState.PROBATION
            agent.last_enforcement_action = action
            result["old_weight"] = old_weight
            result["new_weight"] = agent.weight
            logger.info(
                "ENFORCEMENT REDUCE_WEIGHT: %s — %.2f → %.2f",
                agent.spec.agent_id, old_weight, agent.weight,
            )

        elif action == EnforcementAction.SUSPEND:
            self.suspend_agent(agent.spec.agent_id, reason=rule.name)

        elif action == EnforcementAction.TERMINATE:
            self.terminate_agent(agent.spec.agent_id, reason=rule.name)

        elif action == EnforcementAction.PROMOTE:
            old_tier = agent.tier
            agent.tier = self._compute_promotion_tier(agent)
            agent.last_enforcement_action = action
            result["old_tier"] = old_tier
            result["new_tier"] = agent.tier
            logger.info(
                "ENFORCEMENT PROMOTE: %s — %s → %s",
                agent.spec.agent_id, old_tier, agent.tier,
            )

        elif action == EnforcementAction.BOOST_WEIGHT:
            old_weight = agent.weight
            agent.weight = min(2.0, agent.weight * 1.3)
            agent.last_enforcement_action = action
            result["old_weight"] = old_weight
            result["new_weight"] = agent.weight

        else:
            return None

        agent.last_enforcement_time = datetime.now().isoformat()
        agent.enforcement_log.append(result)

        self._log_event("enforcement_action", result)
        return result

    def _compute_promotion_tier(self, agent: DynamicAgent) -> str:
        """Compute the appropriate tier for an agent based on performance."""
        if agent.accuracy > 0.85 and agent.sharpe_ratio > 2.5:
            return "TIER_0_Director"
        elif agent.accuracy > 0.80 and agent.sharpe_ratio > 2.0:
            return "TIER_1_General"
        elif agent.accuracy > 0.55 and agent.sharpe_ratio > 1.5:
            return "TIER_2_Captain"
        elif agent.accuracy > 0.50 and agent.sharpe_ratio > 1.0:
            return "TIER_3_Lieutenant"
        return "TIER_4_Recruit"

    # --- Signal processing --------------------------------------------------

    def record_signal_outcome(
        self,
        agent_id: str,
        outcome: dict,
    ) -> Optional[dict]:
        """Record a signal outcome for a dynamic agent.

        Updates accuracy, PnL, drawdown, and triggers enforcement check.
        Also updates GSD + Paul via the learning wrapper.

        Args:
            agent_id: Agent identifier.
            outcome: Dict with realized_pnl, was_correct, etc.

        Returns:
            Dict with updated stats and any enforcement actions.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return None

            was_correct = outcome.get("was_correct", False)
            pnl = outcome.get("realized_pnl", 0.0)

            agent.total_signals += 1
            if was_correct:
                agent.correct_signals += 1
            agent.accuracy = agent.correct_signals / max(agent.total_signals, 1)
            agent.total_pnl += pnl
            agent.pnl_history.append(pnl)

            # Update Sharpe (rolling)
            if len(agent.pnl_history) >= 5 and np is not None:
                returns = np.array(agent.pnl_history[-52:], dtype=np.float64)
                mean_r = float(np.mean(returns))
                std_r = float(np.std(returns))
                agent.sharpe_ratio = (mean_r / (std_r + 1e-8)) * (52 ** 0.5)

            # Update max drawdown
            if agent.pnl_history and np is not None:
                cumulative = np.cumsum(agent.pnl_history)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - peak) / (np.abs(peak) + 1e-8)
                agent.max_drawdown_realized = float(np.min(drawdown))

        # Update via learning wrapper (GSD + Paul)
        if self._wrapper is not None:
            self._wrapper.post_decision_hook(agent_id, outcome)

        # Run enforcement check
        actions = self.enforce_agent(agent_id)

        return {
            "agent_id": agent_id,
            "accuracy": agent.accuracy,
            "total_pnl": agent.total_pnl,
            "sharpe_ratio": agent.sharpe_ratio,
            "enforcement_actions": actions,
        }

    def get_pre_decision_enrichment(
        self,
        agent_id: str,
        market_state: dict,
    ) -> dict:
        """Get Paul + GSD enrichment before an agent makes a decision.

        Args:
            agent_id: Agent identifier.
            market_state: Current market state.

        Returns:
            Enrichment dict from AgentLearningWrapper.
        """
        if self._wrapper is not None:
            return self._wrapper.pre_decision_hook(agent_id, market_state)
        return {"agent_id": agent_id, "gradient_confidence": 0.5}

    # --- Registry queries ---------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[DynamicAgent]:
        """Get a specific agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_active_agents(self) -> list[DynamicAgent]:
        """Get all active agents."""
        with self._lock:
            return [
                a for a in self._agents.values()
                if a.state == AgentLifecycleState.ACTIVE
            ]

    def get_agents_by_template(self, template: str) -> list[DynamicAgent]:
        """Get agents filtered by template type."""
        with self._lock:
            ids = self._template_index.get(template, [])
            return [self._agents[aid] for aid in ids if aid in self._agents]

    def get_agents_by_focus(self, focus: str) -> list[DynamicAgent]:
        """Get agents filtered by focus (sector, ticker, strategy)."""
        with self._lock:
            ids = self._focus_index.get(focus, [])
            return [self._agents[aid] for aid in ids if aid in self._agents]

    def get_weighted_consensus(
        self,
        agent_ids: Optional[list[str]] = None,
    ) -> dict:
        """Compute weighted consensus across agents.

        Weights are based on agent.weight * tier_multiplier.

        Returns:
            Dict with consensus_score, direction, contributing_agents.
        """
        tier_multipliers = {
            "TIER_0_Director": 2.5,
            "TIER_1_General": 2.0,
            "TIER_2_Captain": 1.5,
            "TIER_3_Lieutenant": 1.0,
            "TIER_4_Recruit": 0.5,
        }

        with self._lock:
            if agent_ids:
                agents = [
                    self._agents[aid] for aid in agent_ids
                    if aid in self._agents
                    and self._agents[aid].state == AgentLifecycleState.ACTIVE
                ]
            else:
                agents = [
                    a for a in self._agents.values()
                    if a.state == AgentLifecycleState.ACTIVE
                ]

        if not agents:
            return {"consensus_score": 0.0, "direction": "NEUTRAL", "agents": 0}

        total_weight = 0.0
        weighted_sum = 0.0
        for agent in agents:
            tier_mult = tier_multipliers.get(agent.tier, 1.0)
            w = agent.weight * tier_mult * agent.gsd_confidence
            # Use accuracy as a proxy for directional signal
            signal = (agent.accuracy - 0.5) * 2  # map [0,1] → [-1,1]
            weighted_sum += signal * w
            total_weight += w

        if total_weight < 1e-8:
            return {"consensus_score": 0.0, "direction": "NEUTRAL", "agents": len(agents)}

        consensus = weighted_sum / total_weight
        direction = "BULLISH" if consensus > 0.1 else "BEARISH" if consensus < -0.1 else "NEUTRAL"

        return {
            "consensus_score": float(consensus),
            "direction": direction,
            "agents": len(agents),
            "total_weight": float(total_weight),
        }

    def get_registry_summary(self) -> dict:
        """Get comprehensive summary of the agent registry."""
        with self._lock:
            state_counts = defaultdict(int)
            tier_counts = defaultdict(int)
            template_counts = defaultdict(int)

            for agent in self._agents.values():
                state_counts[agent.state] += 1
                tier_counts[agent.tier] += 1
                template_counts[agent.spec.template] += 1

            active_agents = [
                a for a in self._agents.values()
                if a.state == AgentLifecycleState.ACTIVE
            ]

        avg_accuracy = 0.0
        avg_sharpe = 0.0
        if active_agents:
            avg_accuracy = sum(a.accuracy for a in active_agents) / len(active_agents)
            avg_sharpe = sum(a.sharpe_ratio for a in active_agents) / len(active_agents)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self._agents),
            "total_created": self._total_created,
            "total_terminated": self._total_terminated,
            "state_distribution": dict(state_counts),
            "tier_distribution": dict(tier_counts),
            "template_distribution": dict(template_counts),
            "active_count": len(active_agents),
            "avg_accuracy": float(avg_accuracy),
            "avg_sharpe": float(avg_sharpe),
        }

    # --- Logging -----------------------------------------------------------

    def _log_event(self, event_type: str, data: dict):
        """Persist an event to JSONL log."""
        import json
        log_file = self._log_dir / f"factory_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("Factory log write failed: %s", e)
