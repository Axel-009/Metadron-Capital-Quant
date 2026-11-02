"""Enforcement Engine — Dynamic agent governance and compliance.

Bridges Ruflo-agents orchestration capabilities with the Metadron Capital
intelligence platform. Provides real-time enforcement of agent behavior,
performance thresholds, risk limits, and learning compliance.

The enforcement engine operates at two levels:
    1. **Individual enforcement**: Per-agent rules (accuracy, Sharpe, drawdown)
    2. **Collective enforcement**: Cross-agent coordination (consensus drift,
       herding detection, concentration risk, signal redundancy)

Integration with Ruflo-agents:
    - Uses GSDPlugin gradient tracking for proactive enforcement (detect
      degradation before it impacts P&L)
    - Uses PaulPlugin pattern matching to enforce consistency (agents should
      learn from historical patterns, not ignore them)
    - Connects to AgentScorecard for tier-based weight management
    - Feeds enforcement events into LearningLoop for feedback

Enforcement cadence:
    - Per-signal: After every trade outcome
    - Periodic: Every 15 minutes during market hours
    - Sweep: Daily EOD comprehensive enforcement review
"""

import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collective enforcement thresholds
# ---------------------------------------------------------------------------
HERDING_THRESHOLD = 0.85       # >85% agents same direction = herding risk
CONCENTRATION_MAX = 0.20       # No single agent should drive >20% of portfolio
SIGNAL_REDUNDANCY_MIN = 3      # Min unique signal engines across active agents
CONSENSUS_DRIFT_MAX = 0.30     # Max allowed shift in consensus per hour
GRADIENT_ALIGNMENT_MIN = 0.20  # Min cross-engine alignment for confidence


@dataclass
class EnforcementEvent:
    """Record of an enforcement action."""
    event_id: str = ""
    timestamp: str = ""
    agent_id: str = ""
    event_type: str = ""       # individual | collective | sweep
    rule_triggered: str = ""
    action_taken: str = ""
    details: dict = field(default_factory=dict)
    severity: str = "INFO"     # INFO | WARNING | CRITICAL


@dataclass
class CollectiveState:
    """Snapshot of collective agent behavior."""
    timestamp: str = ""
    active_agents: int = 0
    consensus_score: float = 0.0
    herding_risk: float = 0.0
    concentration_risk: float = 0.0
    signal_diversity: int = 0
    gradient_alignment: float = 0.0
    total_pnl: float = 0.0
    avg_accuracy: float = 0.0


class EnforcementEngine:
    """Central enforcement engine for the Metadron Capital agent fleet.

    Monitors individual and collective agent behavior, enforces performance
    thresholds, detects anomalies, and takes corrective actions.

    Usage:
        from engine.agents.dynamic_agent_factory import DynamicAgentFactory
        from intelligence_platform.plugins.gsd_paul_plugin import (
            GSDPlugin, PaulPlugin, AgentLearningWrapper,
        )

        gsd = GSDPlugin()
        paul = PaulPlugin()
        wrapper = AgentLearningWrapper(gsd, paul)
        factory = DynamicAgentFactory(gsd=gsd, paul=paul, wrapper=wrapper)

        enforcement = EnforcementEngine(factory=factory, gsd=gsd, paul=paul)

        # Run periodic enforcement (every 15 min)
        enforcement.run_periodic_enforcement()

        # Run daily sweep (EOD)
        enforcement.run_daily_sweep()
    """

    def __init__(
        self,
        factory=None,
        gsd=None,
        paul=None,
        log_dir: Optional[Path] = None,
    ):
        self._lock = threading.RLock()
        self._factory = factory
        self._gsd = gsd
        self._paul = paul
        self._log_dir = log_dir or Path("logs/enforcement")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Event history
        self._events: deque = deque(maxlen=5000)

        # Consensus tracking for drift detection
        self._consensus_history: deque = deque(maxlen=100)

        # Enforcement stats
        self._total_enforcements = 0
        self._total_warnings = 0
        self._total_suspensions = 0
        self._total_terminations = 0

    # --- Individual enforcement ---------------------------------------------

    def enforce_on_signal(self, agent_id: str, outcome: dict) -> dict:
        """Post-signal enforcement check.

        Called after every signal outcome. Checks individual thresholds
        and triggers corrective actions if needed.

        Args:
            agent_id: Agent identifier.
            outcome: Signal outcome dict.

        Returns:
            Enforcement result with any actions taken.
        """
        if self._factory is None:
            return {"status": "no_factory"}

        result = self._factory.record_signal_outcome(agent_id, outcome)
        if result and result.get("enforcement_actions"):
            for action in result["enforcement_actions"]:
                self._record_event(EnforcementEvent(
                    event_id=f"sig_{datetime.now().strftime('%H%M%S')}_{agent_id[:8]}",
                    timestamp=datetime.now().isoformat(),
                    agent_id=agent_id,
                    event_type="individual",
                    rule_triggered=action.get("rule_id", ""),
                    action_taken=action.get("action", ""),
                    details=action,
                    severity="WARNING" if action.get("action") in (
                        "WARN", "REDUCE_WEIGHT"
                    ) else "CRITICAL",
                ))

        return result or {}

    # --- Collective enforcement ---------------------------------------------

    def compute_collective_state(self) -> CollectiveState:
        """Analyze collective behavior of all active agents.

        Computes herding risk, concentration, signal diversity, and
        consensus metrics.
        """
        if self._factory is None:
            return CollectiveState(timestamp=datetime.now().isoformat())

        agents = self._factory.get_active_agents()
        if not agents:
            return CollectiveState(timestamp=datetime.now().isoformat())

        state = CollectiveState(
            timestamp=datetime.now().isoformat(),
            active_agents=len(agents),
        )

        # Consensus
        consensus = self._factory.get_weighted_consensus()
        state.consensus_score = consensus.get("consensus_score", 0.0)

        # Herding risk: how many agents agree on direction
        directions = []
        for agent in agents:
            signal = (agent.accuracy - 0.5) * 2
            directions.append("BULLISH" if signal > 0.1 else
                            "BEARISH" if signal < -0.1 else "NEUTRAL")

        if directions:
            max_direction_count = max(
                directions.count("BULLISH"),
                directions.count("BEARISH"),
                directions.count("NEUTRAL"),
            )
            state.herding_risk = max_direction_count / len(directions)

        # Concentration risk: max single agent weight
        total_weight = sum(a.weight for a in agents)
        if total_weight > 0:
            max_concentration = max(a.weight / total_weight for a in agents)
            state.concentration_risk = max_concentration

        # Signal diversity: unique signal engines across all agents
        all_engines = set()
        for agent in agents:
            all_engines.update(agent.spec.signal_engines)
        state.signal_diversity = len(all_engines)

        # Gradient alignment from GSD
        if self._gsd is not None:
            alignment = self._gsd.get_cross_engine_alignment()
            state.gradient_alignment = alignment.get("alignment_score", 0.0)

        # Aggregate stats
        if agents:
            state.total_pnl = sum(a.total_pnl for a in agents)
            state.avg_accuracy = sum(a.accuracy for a in agents) / len(agents)

        # Track for drift detection
        with self._lock:
            self._consensus_history.append({
                "timestamp": state.timestamp,
                "consensus": state.consensus_score,
            })

        return state

    def check_collective_violations(self) -> list[EnforcementEvent]:
        """Check for collective enforcement violations.

        Returns list of enforcement events for any detected issues.
        """
        state = self.compute_collective_state()
        violations = []

        # Herding check
        if state.herding_risk > HERDING_THRESHOLD:
            violations.append(EnforcementEvent(
                event_id=f"herd_{datetime.now().strftime('%H%M%S')}",
                timestamp=state.timestamp,
                event_type="collective",
                rule_triggered="herding_detected",
                action_taken="ALERT",
                details={
                    "herding_risk": state.herding_risk,
                    "threshold": HERDING_THRESHOLD,
                    "active_agents": state.active_agents,
                },
                severity="WARNING",
            ))

        # Concentration check
        if state.concentration_risk > CONCENTRATION_MAX:
            violations.append(EnforcementEvent(
                event_id=f"conc_{datetime.now().strftime('%H%M%S')}",
                timestamp=state.timestamp,
                event_type="collective",
                rule_triggered="concentration_exceeded",
                action_taken="REBALANCE",
                details={
                    "concentration_risk": state.concentration_risk,
                    "max_allowed": CONCENTRATION_MAX,
                },
                severity="WARNING",
            ))

        # Signal diversity check
        if state.signal_diversity < SIGNAL_REDUNDANCY_MIN:
            violations.append(EnforcementEvent(
                event_id=f"div_{datetime.now().strftime('%H%M%S')}",
                timestamp=state.timestamp,
                event_type="collective",
                rule_triggered="low_signal_diversity",
                action_taken="RECOMMEND_NEW_AGENTS",
                details={
                    "signal_diversity": state.signal_diversity,
                    "min_required": SIGNAL_REDUNDANCY_MIN,
                },
                severity="INFO",
            ))

        # Consensus drift check
        drift = self._compute_consensus_drift()
        if abs(drift) > CONSENSUS_DRIFT_MAX:
            violations.append(EnforcementEvent(
                event_id=f"drift_{datetime.now().strftime('%H%M%S')}",
                timestamp=state.timestamp,
                event_type="collective",
                rule_triggered="consensus_drift",
                action_taken="ALERT",
                details={
                    "drift": drift,
                    "max_allowed": CONSENSUS_DRIFT_MAX,
                },
                severity="WARNING",
            ))

        for v in violations:
            self._record_event(v)

        return violations

    def _compute_consensus_drift(self) -> float:
        """Compute consensus drift over the last hour."""
        with self._lock:
            history = list(self._consensus_history)

        if len(history) < 2:
            return 0.0

        recent = history[-1]["consensus"]
        older = history[0]["consensus"]
        return recent - older

    # --- Periodic enforcement -----------------------------------------------

    def run_periodic_enforcement(self) -> dict:
        """Run periodic enforcement (every 15 minutes during market hours).

        Checks:
        1. Individual agent enforcement (via factory)
        2. Collective behavior violations
        3. GSD gradient health
        4. Paul pattern compliance

        Returns:
            Summary of enforcement results.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "type": "periodic",
        }

        # 1. Individual enforcement sweep
        if self._factory is not None:
            individual_results = self._factory.enforce_all()
            results["individual_actions"] = sum(
                len(a) for a in individual_results.values()
            )
            results["agents_actioned"] = len(individual_results)

        # 2. Collective checks
        collective_violations = self.check_collective_violations()
        results["collective_violations"] = len(collective_violations)

        # 3. GSD gradient health
        if self._gsd is not None:
            gsd_state = self._gsd.log_gradient_state()
            results["gsd_total_updates"] = gsd_state.get("total_updates", 0)
            results["cross_engine_alignment"] = gsd_state.get(
                "cross_engine_alignment", 0.0
            )

        # 4. Paul pattern compliance
        if self._paul is not None:
            paul_state = self._paul.log_learning_state()
            results["paul_total_patterns"] = paul_state.get("total_patterns", 0)
            results["paul_total_matches"] = paul_state.get("total_matches", 0)

            # Prune stale patterns
            pruned = self._paul.prune_stale_patterns()
            results["paul_patterns_pruned"] = pruned

        self._total_enforcements += 1
        self._log_event("periodic_enforcement", results)

        return results

    def run_daily_sweep(self) -> dict:
        """Run comprehensive daily enforcement sweep (EOD).

        More thorough than periodic — includes:
        1. Full agent performance review
        2. Tier promotion/demotion
        3. Pattern evolution for regime changes
        4. Agent fleet optimization recommendations
        5. Learning state persistence

        Returns:
            Comprehensive sweep results.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "type": "daily_sweep",
        }

        # Run periodic first
        periodic = self.run_periodic_enforcement()
        results["periodic"] = periodic

        # Agent fleet summary
        if self._factory is not None:
            results["fleet_summary"] = self._factory.get_registry_summary()

            # Identify underperformers for review
            active = self._factory.get_active_agents()
            underperformers = [
                {
                    "agent_id": a.spec.agent_id,
                    "name": a.spec.name,
                    "accuracy": a.accuracy,
                    "sharpe": a.sharpe_ratio,
                    "total_pnl": a.total_pnl,
                }
                for a in active
                if a.accuracy < 0.40 and a.total_signals >= 20
            ]
            results["underperformers"] = underperformers

            # Identify top performers
            top_performers = sorted(
                active,
                key=lambda a: a.accuracy * a.sharpe_ratio,
                reverse=True,
            )[:5]
            results["top_performers"] = [
                {
                    "agent_id": a.spec.agent_id,
                    "name": a.spec.name,
                    "accuracy": a.accuracy,
                    "sharpe": a.sharpe_ratio,
                    "tier": a.tier,
                }
                for a in top_performers
            ]

        # Paul pattern library state
        if self._paul is not None:
            results["pattern_library"] = self._paul.log_learning_state()

            # Serialize pattern library for persistence
            self._paul.serialize_library()

        # GSD comprehensive state
        if self._gsd is not None:
            results["gradient_state"] = self._gsd.log_gradient_state()

        # Enforcement stats
        results["enforcement_stats"] = {
            "total_enforcements": self._total_enforcements,
            "total_warnings": self._total_warnings,
            "total_suspensions": self._total_suspensions,
            "total_terminations": self._total_terminations,
        }

        self._log_event("daily_sweep", results)
        return results

    # --- Rebalancing --------------------------------------------------------

    def rebalance_agent_weights(self) -> dict:
        """Rebalance agent weights based on performance and gradient signals.

        Uses GSD gradient confidence and Paul pattern success rates to
        dynamically adjust agent weights.

        Returns:
            Dict of agent_id -> new_weight.
        """
        if self._factory is None:
            return {}

        agents = self._factory.get_active_agents()
        if not agents:
            return {}

        weight_updates = {}
        for agent in agents:
            old_weight = agent.weight

            # Base weight from accuracy
            accuracy_weight = max(0.1, agent.accuracy)

            # GSD gradient modifier
            gsd_modifier = 1.0
            if self._gsd is not None:
                gsd_conf = self._gsd.get_gradient_confidence(agent.spec.agent_id)
                gsd_modifier = 0.5 + gsd_conf  # [0.5, 1.5]

            # Sharpe modifier
            sharpe_modifier = 1.0
            if agent.sharpe_ratio > 1.5:
                sharpe_modifier = 1.3
            elif agent.sharpe_ratio < 0.5 and agent.total_signals >= 20:
                sharpe_modifier = 0.7

            # Compute new weight
            new_weight = accuracy_weight * gsd_modifier * sharpe_modifier
            new_weight = max(0.1, min(2.0, new_weight))

            agent.weight = new_weight
            if abs(new_weight - old_weight) > 0.05:
                weight_updates[agent.spec.agent_id] = {
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "accuracy": agent.accuracy,
                    "gsd_confidence": gsd_modifier - 0.5,
                    "sharpe": agent.sharpe_ratio,
                }

        if weight_updates:
            self._log_event("weight_rebalance", {
                "agents_updated": len(weight_updates),
                "updates": weight_updates,
            })

        return weight_updates

    # --- Auto-spawning based on patterns ------------------------------------

    def auto_spawn_agents(self, market_state: dict) -> list[str]:
        """Automatically spawn new specialist agents based on pattern insights.

        If Paul identifies strong patterns in a sector/ticker that no
        existing agent covers, spawn a specialist.

        Args:
            market_state: Current market state.

        Returns:
            List of newly created agent IDs.
        """
        if self._factory is None or self._paul is None:
            return []

        new_agents = []

        # Find strong patterns without matching agents
        top_patterns = self._paul.get_unified_library(
            regime=market_state.get("regime"),
            min_success_rate=0.7,
            limit=10,
        )

        # Get covered sectors/tickers
        active = self._factory.get_active_agents()
        covered_focuses = {a.spec.focus for a in active if a.spec.focus}

        for pattern in top_patterns:
            # Check if sector/ticker is already covered
            if pattern.sector and pattern.sector not in covered_focuses:
                agent = self._factory.create_sector_specialist(pattern.sector)
                new_agents.append(agent.spec.agent_id)
                covered_focuses.add(pattern.sector)

            if pattern.ticker and pattern.ticker not in covered_focuses:
                if pattern.match_count >= 3 and pattern.success_rate >= 0.75:
                    agent = self._factory.create_ticker_specialist(
                        pattern.ticker, sector=pattern.sector,
                    )
                    new_agents.append(agent.spec.agent_id)
                    covered_focuses.add(pattern.ticker)

        if new_agents:
            self._log_event("auto_spawn", {
                "agents_spawned": len(new_agents),
                "agent_ids": new_agents,
                "regime": market_state.get("regime", ""),
            })

        return new_agents

    # --- Event tracking -----------------------------------------------------

    def _record_event(self, event: EnforcementEvent):
        """Record an enforcement event."""
        with self._lock:
            self._events.append(event)

            if event.severity == "WARNING":
                self._total_warnings += 1
            elif event.severity == "CRITICAL":
                if "suspend" in event.action_taken.lower():
                    self._total_suspensions += 1
                elif "terminate" in event.action_taken.lower():
                    self._total_terminations += 1

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get recent enforcement events."""
        with self._lock:
            events = list(self._events)
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "agent_id": e.agent_id,
                "type": e.event_type,
                "rule": e.rule_triggered,
                "action": e.action_taken,
                "severity": e.severity,
            }
            for e in events[-limit:]
        ]

    def _log_event(self, event_type: str, data: dict):
        """Persist event to JSONL log."""
        log_file = self._log_dir / f"enforcement_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("Enforcement log write failed: %s", e)
