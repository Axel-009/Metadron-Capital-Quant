"""Paul Orchestrator — Full integration of GSD + Paul across all 34+ agents.

Wires the GSD (Gradient Signal Dynamics) and Paul (Pattern Awareness &
Unified Learning) plugins into every agent in the Metadron Capital platform:
    - 11 GICS sector bots
    - 11 GICS research bots
    - 12 investor persona agents
    - Dynamic agents spawned by the factory

Responsibilities:
    1. Initialize GSD + Paul plugins with platform-specific config
    2. Attach learning wrappers to all existing agents
    3. Route signal outcomes through the learning pipeline
    4. Manage pattern evolution on regime changes
    5. Provide unified learning state for monitoring
    6. Coordinate with DynamicAgentFactory for dynamic agent enforcement
    7. Feed enrichment data into the 10-tier MLVoteEnsemble

Usage in live_loop_orchestrator.py:
    orchestrator = PaulOrchestrator()
    orchestrator.initialize()
    orchestrator.attach_all_platform_agents(sector_bots, research_bots, personas)

    # Before each decision cycle
    enrichments = orchestrator.enrich_all_agents(market_state)

    # After each execution
    orchestrator.process_outcomes(outcomes)

    # On regime change
    orchestrator.on_regime_change(old_regime, new_regime, confidence)

    # EOD
    orchestrator.end_of_day_summary()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False


class PaulOrchestrator:
    """Master orchestrator connecting Paul + GSD to the entire agent fleet.

    This is the single entry point for all learning and enforcement
    operations across the platform.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self._log_dir = log_dir or Path("logs")
        self._data_dir = data_dir or Path("data/paul_patterns")
        self._initialized = False

        # These are set during initialize()
        self._gsd = None
        self._paul = None
        self._wrapper = None
        self._factory = None
        self._enforcement = None

        # Agent ID mapping: name -> agent_id for wrapper tracking
        self._agent_id_map: dict[str, str] = {}

        # Track all attached agents by category
        self._sector_bots: list = []
        self._research_bots: list = []
        self._personas: list = []
        self._dynamic_agents: list = []

    def initialize(self) -> dict:
        """Initialize all Paul + GSD subsystems.

        Creates the GSD plugin, Paul plugin, AgentLearningWrapper,
        DynamicAgentFactory, and EnforcementEngine.

        Returns:
            Initialization status dict.
        """
        status = {"timestamp": datetime.now().isoformat()}

        try:
            from intelligence_platform.plugins.gsd_paul_plugin import (
                GSDPlugin,
                PaulPlugin,
                AgentLearningWrapper,
            )

            self._gsd = GSDPlugin(log_dir=self._log_dir / "gsd_plugin")
            self._paul = PaulPlugin(
                log_dir=self._log_dir / "paul_plugin",
                data_dir=self._data_dir,
            )
            self._wrapper = AgentLearningWrapper(
                self._gsd, self._paul,
            )

            status["gsd"] = "initialized"
            status["paul"] = "initialized"
            status["wrapper"] = "initialized"
            logger.info("PaulOrchestrator: GSD + Paul + Wrapper initialized")

        except ImportError as e:
            logger.warning("PaulOrchestrator: GSD/Paul plugin import failed: %s", e)
            status["gsd"] = f"import_error: {e}"
            status["paul"] = f"import_error: {e}"
            status["wrapper"] = "unavailable"

        # Initialize factory
        try:
            from .dynamic_agent_factory import DynamicAgentFactory
            self._factory = DynamicAgentFactory(
                gsd=self._gsd,
                paul=self._paul,
                wrapper=self._wrapper,
                log_dir=self._log_dir / "agent_factory",
            )
            status["factory"] = "initialized"
        except ImportError as e:
            logger.warning("PaulOrchestrator: Factory import failed: %s", e)
            status["factory"] = f"import_error: {e}"

        # Initialize enforcement engine
        try:
            from .enforcement_engine import EnforcementEngine
            self._enforcement = EnforcementEngine(
                factory=self._factory,
                gsd=self._gsd,
                paul=self._paul,
                log_dir=self._log_dir / "enforcement",
            )
            status["enforcement"] = "initialized"
        except ImportError as e:
            logger.warning("PaulOrchestrator: Enforcement import failed: %s", e)
            status["enforcement"] = f"import_error: {e}"

        self._initialized = True
        status["status"] = "ready"
        logger.info("PaulOrchestrator: fully initialized")
        return status

    # --- Agent attachment ---------------------------------------------------

    def attach_all_platform_agents(
        self,
        sector_bots: Optional[list] = None,
        research_bots: Optional[list] = None,
        personas: Optional[list] = None,
    ) -> dict:
        """Attach GSD + Paul learning to all platform agents.

        Args:
            sector_bots: List of SectorBot instances (11 GICS bots).
            research_bots: List of ResearchBot instances (11 GICS bots).
            personas: List of investor persona agents (12 personas).

        Returns:
            Attachment summary.
        """
        if not self._initialized:
            self.initialize()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "sector_bots": 0,
            "research_bots": 0,
            "personas": 0,
            "total_attached": 0,
        }

        # Attach sector bots
        if sector_bots and self._wrapper:
            for bot in sector_bots:
                agent_id = self._attach_agent(bot, "sector_bot")
                if agent_id:
                    self._sector_bots.append(bot)
                    summary["sector_bots"] += 1

        # Attach research bots
        if research_bots and self._wrapper:
            for bot in research_bots:
                agent_id = self._attach_agent(bot, "research_bot")
                if agent_id:
                    self._research_bots.append(bot)
                    summary["research_bots"] += 1

        # Attach personas
        if personas and self._wrapper:
            for persona in personas:
                agent_id = self._attach_agent(persona, "persona")
                if agent_id:
                    self._personas.append(persona)
                    summary["personas"] += 1

        summary["total_attached"] = (
            summary["sector_bots"]
            + summary["research_bots"]
            + summary["personas"]
        )

        logger.info(
            "PaulOrchestrator: attached %d agents (%d sector, %d research, %d personas)",
            summary["total_attached"],
            summary["sector_bots"],
            summary["research_bots"],
            summary["personas"],
        )

        return summary

    def _attach_agent(self, agent: Any, category: str) -> Optional[str]:
        """Attach a single agent and register it."""
        if self._wrapper is None:
            return None

        try:
            agent_id = self._wrapper.attach_to_agent(agent)
            name = (
                getattr(agent, "name", None)
                or getattr(agent, "bot_name", None)
                or getattr(agent, "agent_id", None)
                or str(id(agent))
            )
            self._agent_id_map[name] = agent_id
            return agent_id
        except Exception as e:
            logger.warning("Failed to attach agent: %s", e)
            return None

    # --- Pre-decision enrichment -------------------------------------------

    def enrich_all_agents(self, market_state: dict) -> dict[str, dict]:
        """Get Paul + GSD enrichment for all attached agents.

        Called before each decision cycle. Returns per-agent enrichment
        data that can be used to modulate confidence and direction.

        Args:
            market_state: Current market state with signals, regime, etc.

        Returns:
            Dict of agent_id -> enrichment data.
        """
        if self._wrapper is None:
            return {}

        enrichments = {}
        for name, agent_id in self._agent_id_map.items():
            try:
                enrichment = self._wrapper.pre_decision_hook(agent_id, market_state)
                enrichments[agent_id] = enrichment
            except Exception as e:
                logger.debug("Enrichment failed for %s: %s", name, e)

        return enrichments

    def enrich_agent(
        self, agent_name_or_id: str, market_state: dict,
    ) -> dict:
        """Get enrichment for a specific agent."""
        agent_id = self._agent_id_map.get(agent_name_or_id, agent_name_or_id)
        if self._wrapper is None:
            return {"agent_id": agent_id, "gradient_confidence": 0.5}
        return self._wrapper.pre_decision_hook(agent_id, market_state)

    # --- Outcome processing ------------------------------------------------

    def process_outcomes(self, outcomes: list[dict]) -> list[dict]:
        """Process a batch of signal outcomes through the learning pipeline.

        Each outcome updates:
        1. GSD gradient profile for the agent
        2. Paul pattern library (stores new patterns)
        3. Enforcement checks
        4. LearningLoop integration

        Args:
            outcomes: List of outcome dicts, each with agent_id/agent_name
                and standard outcome fields.

        Returns:
            List of processing results.
        """
        results = []
        for outcome in outcomes:
            agent_name = outcome.get("agent_name", "")
            agent_id = outcome.get("agent_id", "")

            # Resolve agent_id from name
            if not agent_id and agent_name:
                agent_id = self._agent_id_map.get(agent_name, agent_name)

            if not agent_id:
                continue

            # Process through wrapper (GSD + Paul)
            if self._wrapper is not None:
                try:
                    result = self._wrapper.post_decision_hook(agent_id, outcome)
                    results.append(result)
                except Exception as e:
                    logger.debug("Outcome processing failed for %s: %s", agent_id, e)

            # Process through enforcement engine
            if self._enforcement is not None:
                self._enforcement.enforce_on_signal(agent_id, outcome)

        return results

    def process_single_outcome(
        self, agent_name_or_id: str, outcome: dict,
    ) -> dict:
        """Process a single outcome for one agent."""
        agent_id = self._agent_id_map.get(agent_name_or_id, agent_name_or_id)
        outcome["agent_id"] = agent_id
        results = self.process_outcomes([outcome])
        return results[0] if results else {}

    # --- Regime change handling ---------------------------------------------

    def on_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float = 0.8,
    ) -> dict:
        """Handle market regime change across the entire agent fleet.

        Updates:
        1. Paul pattern evolution (decay patterns from old regime)
        2. GSD gradient reset for regime-sensitive agents
        3. Dynamic agent auto-spawn for new regime opportunities
        4. Weight rebalancing

        Args:
            old_regime: Previous regime (TRENDING/RANGE/STRESS/CRASH).
            new_regime: New regime.
            confidence: Confidence in the regime transition.

        Returns:
            Regime change processing results.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
        }

        # Evolve Paul patterns
        if self._paul is not None:
            affected = self._paul.evolve_patterns({
                "old_regime": old_regime,
                "new_regime": new_regime,
                "transition_confidence": confidence,
            })
            results["patterns_evolved"] = affected

        # Auto-spawn agents for new regime opportunities
        if self._enforcement is not None:
            new_agents = self._enforcement.auto_spawn_agents({
                "regime": new_regime,
            })
            results["agents_spawned"] = len(new_agents)

        # Rebalance weights
        if self._enforcement is not None:
            weight_updates = self._enforcement.rebalance_agent_weights()
            results["weights_rebalanced"] = len(weight_updates)

        logger.info(
            "PaulOrchestrator: regime change %s → %s (confidence=%.2f, "
            "patterns_evolved=%d)",
            old_regime, new_regime, confidence,
            results.get("patterns_evolved", 0),
        )

        return results

    # --- Periodic operations -----------------------------------------------

    def run_periodic_check(self) -> dict:
        """Run periodic enforcement and learning check (every 15 min).

        Returns:
            Enforcement + learning status.
        """
        results = {"timestamp": datetime.now().isoformat()}

        if self._enforcement is not None:
            results["enforcement"] = self._enforcement.run_periodic_enforcement()

        if self._gsd is not None:
            results["gsd_state"] = self._gsd.log_gradient_state()

        return results

    def end_of_day_summary(self) -> dict:
        """Generate comprehensive end-of-day learning summary.

        Includes:
        1. Daily enforcement sweep
        2. Agent fleet performance summary
        3. Pattern library state
        4. Gradient analysis
        5. Learning recommendations

        Returns:
            Comprehensive EOD summary.
        """
        summary = {"timestamp": datetime.now().isoformat()}

        # Daily sweep
        if self._enforcement is not None:
            summary["enforcement_sweep"] = self._enforcement.run_daily_sweep()

        # Agent learning summaries
        if self._wrapper is not None:
            all_summaries = self._wrapper.get_all_agent_summaries()
            summary["agent_count"] = len(all_summaries)

            # Aggregate learning metrics
            if all_summaries:
                confidences = [
                    s.get("gsd", {}).get("gradient_confidence", 0.5)
                    for s in all_summaries.values()
                ]
                summary["avg_gradient_confidence"] = (
                    sum(confidences) / len(confidences) if confidences else 0.5
                )

                accuracies = [
                    s.get("performance", {}).get("accuracy", 0.0)
                    for s in all_summaries.values()
                    if s.get("performance", {}).get("total_decisions", 0) > 0
                ]
                if accuracies:
                    summary["avg_accuracy"] = sum(accuracies) / len(accuracies)

        # Factory registry
        if self._factory is not None:
            summary["factory_registry"] = self._factory.get_registry_summary()

        # Paul pattern library persistence
        if self._paul is not None:
            lib_path = self._paul.serialize_library()
            summary["pattern_library_saved"] = str(lib_path)

        logger.info("PaulOrchestrator: EOD summary generated")
        return summary

    # --- Dynamic agent management ------------------------------------------

    def create_dynamic_agent(self, spec_dict: dict) -> Optional[str]:
        """Create a dynamic agent via the factory.

        Args:
            spec_dict: Dict of AgentSpec fields.

        Returns:
            Agent ID if created, None otherwise.
        """
        if self._factory is None:
            return None

        from .dynamic_agent_factory import AgentSpec
        spec = AgentSpec(**{
            k: v for k, v in spec_dict.items()
            if hasattr(AgentSpec, k)
        })
        agent = self._factory.create_agent(spec)
        self._dynamic_agents.append(agent)
        return agent.spec.agent_id

    def spawn_sector_specialist(self, sector: str) -> Optional[str]:
        """Spawn a specialist for a specific GICS sector."""
        if self._factory is None:
            return None
        agent = self._factory.create_sector_specialist(sector)
        self._dynamic_agents.append(agent)
        return agent.spec.agent_id

    def spawn_ticker_specialist(
        self, ticker: str, sector: str = "",
    ) -> Optional[str]:
        """Spawn a specialist for a specific ticker."""
        if self._factory is None:
            return None
        agent = self._factory.create_ticker_specialist(ticker, sector)
        self._dynamic_agents.append(agent)
        return agent.spec.agent_id

    # --- MLVoteEnsemble integration ----------------------------------------

    def get_ensemble_adjustments(self, market_state: dict) -> dict:
        """Get confidence adjustments for the 10-tier MLVoteEnsemble.

        Based on GSD gradient analysis and Paul pattern matching,
        provides per-tier confidence adjustments.

        Args:
            market_state: Current market state.

        Returns:
            Dict of tier_name -> confidence_adjustment (-0.3 to +0.3).
        """
        # Per-engine gradients map to ensemble tiers (available regardless of GSD/Paul state)
        tier_engine_map = {
            "T1_Neural": "ml_ensemble",
            "T2_Momentum": "alpha_optimizer",
            "T3_Vol_Regime": "cube",
            "T4_Monte_Carlo": "alpha_optimizer",
            "T5_Quality": "security_analysis",
            "T6_Social": "social",
            "T7_Distress": "distress",
            "T8_Event": "event_driven",
            "T9_CVR": "cvr",
            "T10_Credit": "security_analysis",
        }

        # Pre-populate neutral adjustments so Paul boosts apply even when GSD is unavailable
        adjustments = {tier: 0.0 for tier in tier_engine_map}

        if self._gsd is not None:
            # Cross-engine alignment affects overall confidence
            alignment = self._gsd.get_cross_engine_alignment()
            alignment_score = alignment.get("alignment_score", 0.0)

            for tier, engine in tier_engine_map.items():
                grad_info = self._gsd.compute_signal_gradient(engine)
                stability = grad_info.get("stability", 0.5)
                momentum = grad_info.get("momentum", 0.0)

                # Stable, positive momentum → boost confidence
                # Unstable or negative momentum → reduce
                adj = (stability - 0.5) * 0.2 + momentum * 0.1
                adj = max(-0.3, min(0.3, adj))

                # Global alignment modifier
                adj *= (0.5 + alignment_score)

                adjustments[tier] = round(adj, 4)

        # Paul pattern modifier
        if self._paul is not None:
            matches = self._paul.match_pattern(market_state, top_k=3)
            if matches and matches[0].combined_score > 0.7:
                top = matches[0]
                if top.pattern and top.pattern.was_successful:
                    # Boost tiers that the pattern relied on
                    for engine in (top.pattern.signal_engines_active or []):
                        for tier, eng in tier_engine_map.items():
                            if eng == engine and tier in adjustments:
                                adjustments[tier] = min(
                                    0.3, adjustments[tier] + 0.05
                                )

        return adjustments

    # --- Status and diagnostics -------------------------------------------

    def status(self) -> dict:
        """Get comprehensive orchestrator status."""
        return {
            "initialized": self._initialized,
            "gsd_active": self._gsd is not None,
            "paul_active": self._paul is not None,
            "wrapper_active": self._wrapper is not None,
            "factory_active": self._factory is not None,
            "enforcement_active": self._enforcement is not None,
            "attached_agents": len(self._agent_id_map),
            "sector_bots": len(self._sector_bots),
            "research_bots": len(self._research_bots),
            "personas": len(self._personas),
            "dynamic_agents": len(self._dynamic_agents),
        }

    # --- agent_skills bridge --------------------------------------------------

    def dispatch_skill_analysis(
        self, skill_name: str, parameters: dict,
    ) -> dict:
        """Dispatch an analysis task via agent_skills when available.

        Args:
            skill_name: Name of the skill to run.
            parameters: Skill-specific parameters dict.

        Returns:
            Skill execution result or empty dict if unavailable.
        """
        if not AGENT_SKILLS_AVAILABLE:
            logger.debug("agent_skills not available; skipping skill dispatch")
            return {}
        try:
            result = test_skill(skill_name, parameters)
            return result if isinstance(result, dict) else {"output": result}
        except Exception as e:
            logger.warning("Skill dispatch failed for %s: %s", skill_name, e)
            return {}
