"""GSD Workflow Bridge — Get Shit Done integration for Metadron Capital.

Bridges the GSD (Get Shit Done) meta-prompting and context engineering
framework with the Metadron Capital intelligence platform. GSD provides
structured workflow orchestration, agent coordination patterns, and
persistent state management that complement the platform's existing
GSD Plugin (Gradient Signal Dynamics) and Paul Plugin.

GSD Capabilities Integrated:
    1. **State Management** — File-based persistent state (.planning/)
       for tracking experiment phases, model lifecycle, and strategy evolution
    2. **Phase Orchestration** — Structured research → plan → execute → verify
       pipeline for alpha development and signal engine evolution
    3. **Agent Coordination** — Wave-based parallel execution patterns for
       multi-agent decision coordination
    4. **Checkpoint System** — Gate-based progression with human verification
       for high-conviction trade decisions
    5. **Context Engineering** — Fresh context per agent to prevent drift,
       with structured handoff artifacts
    6. **Artifact Tracking** — Requirements, roadmaps, verification docs
       for audit trail and compliance

Integration with existing systems:
    - GSDPlugin (gradient dynamics) — metrics tracking feeds into GSD state
    - PaulPlugin (pattern library) — pattern discoveries stored as GSD phases
    - DynamicAgentFactory — agent creation follows GSD research → plan → execute
    - EnforcementEngine — enforcement events logged as GSD state updates
    - LearningLoop — outcome feedback integrated with GSD progression
    - LiveLoopOrchestrator — GSD orchestration available during learning phase

Source: https://github.com/gsd-build/get-shit-done.git (v1.26.0)
Location: intelligence_platform/get-shit-done/
"""

import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# GSD source location within Metadron Capital
GSD_ROOT = Path(__file__).parent.parent / "get-shit-done"
GSD_TOOLS = GSD_ROOT / "get-shit-done" / "bin" / "gsd-tools.cjs"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GSDPhase:
    """A GSD workflow phase for financial strategy development."""
    phase_id: str = ""
    name: str = ""
    description: str = ""
    status: str = "pending"        # pending | researching | planning | executing | verifying | complete
    phase_type: str = ""           # research | alpha_dev | backtest | deployment | monitoring
    requirements: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    completed_at: str = ""


@dataclass
class GSDPlan:
    """An execution plan within a GSD phase."""
    plan_id: str = ""
    phase_id: str = ""
    name: str = ""
    tasks: list = field(default_factory=list)
    wave: int = 1                  # Parallel execution wave
    depends_on: list = field(default_factory=list)
    subsystem: str = ""            # signal_engine | ml_model | agent | execution | monitoring
    status: str = "pending"
    started_at: str = ""
    completed_at: str = ""
    summary: str = ""


@dataclass
class GSDTask:
    """A single task within a GSD plan."""
    task_id: str = ""
    name: str = ""
    task_type: str = "auto"        # auto | checkpoint:human-verify | checkpoint:decision
    files: list = field(default_factory=list)
    action: str = ""
    verify: str = ""
    done_criteria: str = ""
    status: str = "pending"
    outcome: str = ""


@dataclass
class GSDState:
    """Persistent state for GSD workflow within Metadron Capital."""
    timestamp: str = ""
    current_phase: int = 0
    current_plan: int = 0
    status: str = "idle"           # idle | researching | planning | executing | verifying
    total_phases: int = 0
    completed_phases: int = 0
    decisions: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    session_info: dict = field(default_factory=dict)


@dataclass
class StrategyExperiment:
    """An alpha/strategy development experiment tracked by GSD."""
    experiment_id: str = ""
    name: str = ""
    hypothesis: str = ""
    signal_engines: list = field(default_factory=list)
    target_alpha: float = 0.0
    backtest_period: str = ""
    phases: list = field(default_factory=list)  # List of GSDPhase
    status: str = "proposed"       # proposed | active | backtesting | validated | deployed | retired
    results: dict = field(default_factory=dict)
    created_at: str = ""


# ---------------------------------------------------------------------------
# GSD Workflow Bridge
# ---------------------------------------------------------------------------
class GSDWorkflowBridge:
    """Bridges GSD workflow orchestration with Metadron Capital.

    Provides structured workflow management for:
    - Alpha strategy development (research → backtest → deploy)
    - Signal engine evolution (feature engineering cycles)
    - Agent development lifecycle (design → create → train → enforce)
    - Model training pipeline (data → features → train → validate)
    - Portfolio strategy updates (quarterly thesis refresh)

    Usage:
        bridge = GSDWorkflowBridge()

        # Create a new strategy experiment
        exp = bridge.create_experiment(
            name="Macro Momentum Alpha",
            hypothesis="Fed liquidity changes predict sector rotation 5-10 days ahead",
            signal_engines=["macro", "fed_liquidity", "cube"],
            target_alpha=0.05,
        )

        # Research phase
        bridge.start_phase(exp.experiment_id, "research", {
            "domain": "macro_momentum",
            "data_sources": ["FRED", "OpenBB"],
        })

        # Plan phase
        bridge.start_phase(exp.experiment_id, "alpha_dev", {
            "features": ["M2V_gradient", "WALCL_change", "yield_curve_slope"],
            "model_type": "walk_forward_ml",
        })

        # Execute backtest
        bridge.start_phase(exp.experiment_id, "backtest", {
            "period": "2020-2025",
            "benchmark": "SPY",
        })

        # Record results
        bridge.record_experiment_results(exp.experiment_id, {
            "sharpe": 2.1,
            "alpha": 0.06,
            "max_drawdown": -0.08,
            "win_rate": 0.62,
        })
    """

    def __init__(
        self,
        planning_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        self._planning_dir = planning_dir or Path("data/gsd_planning")
        self._planning_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir or Path("logs/gsd_workflow")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Experiment registry
        self._experiments: dict[str, StrategyExperiment] = {}

        # State persistence
        self._state = GSDState(timestamp=datetime.now().isoformat())
        self._load_state()

        # GSD tools availability
        self._gsd_tools_available = GSD_TOOLS.exists()
        if self._gsd_tools_available:
            logger.info("GSD tools found at %s", GSD_TOOLS)
        else:
            logger.info("GSD tools not found (JS runtime); using Python-native state management")

    # --- Experiment lifecycle -----------------------------------------------

    def create_experiment(
        self,
        name: str,
        hypothesis: str,
        signal_engines: Optional[list] = None,
        target_alpha: float = 0.0,
        backtest_period: str = "",
    ) -> StrategyExperiment:
        """Create a new strategy experiment with GSD tracking.

        Returns:
            StrategyExperiment instance.
        """
        import uuid
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"

        experiment = StrategyExperiment(
            experiment_id=exp_id,
            name=name,
            hypothesis=hypothesis,
            signal_engines=signal_engines or [],
            target_alpha=target_alpha,
            backtest_period=backtest_period,
            status="proposed",
            created_at=datetime.now().isoformat(),
        )

        self._experiments[exp_id] = experiment

        # Create experiment directory
        exp_dir = self._planning_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write PROJECT.md (GSD artifact)
        project_md = exp_dir / "PROJECT.md"
        project_md.write_text(
            f"# {name}\n\n"
            f"## Hypothesis\n{hypothesis}\n\n"
            f"## Signal Engines\n"
            + "\n".join(f"- {e}" for e in (signal_engines or []))
            + f"\n\n## Target Alpha\n{target_alpha:.2%}\n\n"
            f"## Backtest Period\n{backtest_period or 'TBD'}\n\n"
            f"## Status\n{experiment.status}\n\n"
            f"## Created\n{experiment.created_at}\n"
        )

        # Write ROADMAP.md skeleton
        roadmap_md = exp_dir / "ROADMAP.md"
        roadmap_md.write_text(
            f"# Roadmap: {name}\n\n"
            "## Phase 1: Research\n"
            f"- Domain research for: {', '.join(signal_engines or [])}\n"
            "- Data availability assessment\n"
            "- Literature review\n\n"
            "## Phase 2: Feature Engineering\n"
            "- Signal feature extraction\n"
            "- Feature importance analysis\n"
            "- Walk-forward window design\n\n"
            "## Phase 3: Model Development\n"
            "- Model selection\n"
            "- Hyperparameter optimization\n"
            "- Cross-validation\n\n"
            "## Phase 4: Backtesting\n"
            f"- Period: {backtest_period or 'TBD'}\n"
            "- Monte Carlo simulation\n"
            "- Regime-conditional analysis\n\n"
            "## Phase 5: Deployment\n"
            "- Integration with MLVoteEnsemble\n"
            "- Live paper trading validation\n"
            "- Performance monitoring\n"
        )

        self._persist_state()
        self._log_event("experiment_created", {
            "experiment_id": exp_id,
            "name": name,
            "hypothesis": hypothesis,
        })

        logger.info("GSD experiment created: %s — %s", exp_id, name)
        return experiment

    def start_phase(
        self,
        experiment_id: str,
        phase_type: str,
        config: Optional[dict] = None,
    ) -> Optional[GSDPhase]:
        """Start a new phase within an experiment.

        Args:
            experiment_id: Experiment to add phase to.
            phase_type: research | alpha_dev | backtest | deployment | monitoring
            config: Phase-specific configuration.

        Returns:
            Created GSDPhase.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            logger.warning("Experiment %s not found", experiment_id)
            return None

        phase_num = len(exp.phases) + 1
        phase = GSDPhase(
            phase_id=f"phase_{phase_num:02d}",
            name=f"Phase {phase_num}: {phase_type}",
            description=json.dumps(config or {}, indent=2),
            status="executing",
            phase_type=phase_type,
            created_at=datetime.now().isoformat(),
        )

        exp.phases.append(phase)
        if exp.status == "proposed":
            exp.status = "active"

        # Create phase directory with GSD artifacts
        phase_dir = self._planning_dir / experiment_id / f"phases" / phase.phase_id
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Write CONTEXT.md
        (phase_dir / "CONTEXT.md").write_text(
            f"# {phase.name}\n\n"
            f"## Type\n{phase_type}\n\n"
            f"## Configuration\n```json\n{json.dumps(config or {}, indent=2)}\n```\n\n"
            f"## Started\n{phase.created_at}\n"
        )

        self._persist_state()
        self._log_event("phase_started", {
            "experiment_id": experiment_id,
            "phase_id": phase.phase_id,
            "phase_type": phase_type,
        })

        return phase

    def complete_phase(
        self,
        experiment_id: str,
        phase_id: str,
        results: Optional[dict] = None,
    ) -> bool:
        """Mark a phase as complete with results."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return False

        for phase in exp.phases:
            if phase.phase_id == phase_id:
                phase.status = "complete"
                phase.completed_at = datetime.now().isoformat()
                if results:
                    phase.metrics = results

                # Write SUMMARY.md
                phase_dir = (
                    self._planning_dir / experiment_id / "phases" / phase_id
                )
                phase_dir.mkdir(parents=True, exist_ok=True)
                (phase_dir / "SUMMARY.md").write_text(
                    f"# {phase.name} — Summary\n\n"
                    f"## Status\nComplete\n\n"
                    f"## Results\n```json\n{json.dumps(results or {}, indent=2)}\n```\n\n"
                    f"## Completed\n{phase.completed_at}\n"
                )

                self._persist_state()
                return True

        return False

    def record_experiment_results(
        self,
        experiment_id: str,
        results: dict,
    ) -> bool:
        """Record final experiment results (backtest metrics, alpha, etc.)."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return False

        exp.results = results
        exp.status = "validated" if results.get("sharpe", 0) > 1.0 else "retired"

        # Write RESULTS.md
        results_path = self._planning_dir / experiment_id / "RESULTS.md"
        results_path.write_text(
            f"# Results: {exp.name}\n\n"
            f"## Status\n{exp.status}\n\n"
            f"## Metrics\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in results.items())
            + f"\n\n## Hypothesis\n{exp.hypothesis}\n\n"
            f"## Validated\n{datetime.now().isoformat()}\n"
        )

        self._persist_state()
        self._log_event("experiment_results", {
            "experiment_id": experiment_id,
            "status": exp.status,
            "results": results,
        })

        return True

    # --- Wave-based parallel coordination ----------------------------------

    def create_parallel_wave(
        self,
        plans: list[dict],
    ) -> list[list[dict]]:
        """Group plans into parallel execution waves based on dependencies.

        Uses GSD's wave-based parallelization pattern: independent plans
        run in parallel within the same wave, dependent plans wait for
        prior waves.

        Args:
            plans: List of plan dicts with 'id' and 'depends_on' fields.

        Returns:
            List of waves, each wave is a list of plans that can execute
            in parallel.
        """
        # Build dependency graph
        plan_map = {p["id"]: p for p in plans}
        remaining = set(plan_map.keys())
        completed = set()
        waves = []

        while remaining:
            # Find plans whose dependencies are all completed
            wave = []
            for pid in list(remaining):
                plan = plan_map[pid]
                deps = set(plan.get("depends_on", []))
                if deps.issubset(completed):
                    wave.append(plan)

            if not wave:
                # Circular dependency — break by adding remaining as final wave
                wave = [plan_map[pid] for pid in remaining]
                logger.warning("Circular dependency detected, forcing remaining plans into final wave")

            for plan in wave:
                remaining.discard(plan["id"])
                completed.add(plan["id"])

            waves.append(wave)

        return waves

    # --- GSD State management (Python-native) -------------------------------

    def get_state(self) -> GSDState:
        """Get current GSD workflow state."""
        return self._state

    def update_state(self, updates: dict) -> GSDState:
        """Update GSD state fields."""
        for key, value in updates.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._state.timestamp = datetime.now().isoformat()
        self._persist_state()
        return self._state

    def record_decision(self, decision: str, rationale: str = "") -> None:
        """Record a workflow decision in state."""
        self._state.decisions.append({
            "decision": decision,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        })
        self._persist_state()

    def record_metric(
        self,
        metric_name: str,
        value: float,
        context: Optional[dict] = None,
    ) -> None:
        """Record a metric in GSD state (analogous to gsd-tools state record-metric)."""
        if "metrics_history" not in self._state.metrics:
            self._state.metrics["metrics_history"] = []

        self._state.metrics["metrics_history"].append({
            "name": metric_name,
            "value": value,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        })

        # Keep latest value for quick access
        self._state.metrics[metric_name] = value
        self._persist_state()

    # --- GSD Tools integration (when Node.js available) --------------------

    def call_gsd_tools(self, *args) -> Optional[dict]:
        """Call gsd-tools.cjs CLI if Node.js is available.

        Falls back to None if Node.js or gsd-tools not available.
        """
        if not self._gsd_tools_available:
            return None

        try:
            result = subprocess.run(
                ["node", str(GSD_TOOLS)] + list(args),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._planning_dir),
            )
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"raw": result.stdout.strip()}
            else:
                logger.debug("gsd-tools error: %s", result.stderr)
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("gsd-tools call failed: %s", e)
            return None

    # --- Integration with platform components ------------------------------

    def integrate_with_learning_loop(
        self,
        signal_outcomes: list[dict],
    ) -> dict:
        """Feed LearningLoop signal outcomes into GSD state tracking.

        Records key metrics from signal outcomes as GSD metrics,
        enabling structured review of alpha performance over time.
        """
        if not signal_outcomes:
            return {"recorded": 0}

        total_pnl = sum(o.get("realized_pnl", 0.0) for o in signal_outcomes)
        correct = sum(1 for o in signal_outcomes if o.get("was_correct", False))
        accuracy = correct / len(signal_outcomes) if signal_outcomes else 0.0

        self.record_metric("signal_accuracy", accuracy, {
            "total_signals": len(signal_outcomes),
            "correct": correct,
        })
        self.record_metric("total_pnl", total_pnl)

        return {
            "recorded": len(signal_outcomes),
            "accuracy": accuracy,
            "total_pnl": total_pnl,
        }

    def integrate_with_gradient_state(
        self,
        gsd_plugin_state: dict,
    ) -> None:
        """Record GSD Plugin (Gradient Signal Dynamics) state as metrics."""
        if not gsd_plugin_state:
            return

        alignment = gsd_plugin_state.get("cross_engine_alignment", 0.0)
        self.record_metric("cross_engine_alignment", alignment)
        self.record_metric("gsd_total_updates", gsd_plugin_state.get("total_updates", 0))

    def integrate_with_paul_state(
        self,
        paul_state: dict,
    ) -> None:
        """Record Paul Plugin state as metrics."""
        if not paul_state:
            return

        self.record_metric("paul_total_patterns", paul_state.get("total_patterns", 0))
        self.record_metric("paul_total_matches", paul_state.get("total_matches", 0))

        sr_stats = paul_state.get("success_rate_stats", {})
        if sr_stats:
            self.record_metric("paul_avg_success_rate", sr_stats.get("mean", 0.0))

    # --- Experiment queries ------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Optional[StrategyExperiment]:
        """Get a specific experiment."""
        return self._experiments.get(experiment_id)

    def get_active_experiments(self) -> list[StrategyExperiment]:
        """Get all active experiments."""
        return [e for e in self._experiments.values() if e.status == "active"]

    def get_validated_experiments(self) -> list[StrategyExperiment]:
        """Get experiments that passed validation (Sharpe > 1.0)."""
        return [e for e in self._experiments.values() if e.status == "validated"]

    def get_all_experiments(self) -> dict[str, StrategyExperiment]:
        """Get all experiments."""
        return dict(self._experiments)

    # --- Summary and reporting ---------------------------------------------

    def get_workflow_summary(self) -> dict:
        """Get comprehensive workflow summary."""
        exp_by_status = defaultdict(int)
        for exp in self._experiments.values():
            exp_by_status[exp.status] += 1

        return {
            "timestamp": datetime.now().isoformat(),
            "state": {
                "status": self._state.status,
                "current_phase": self._state.current_phase,
                "completed_phases": self._state.completed_phases,
                "decisions": len(self._state.decisions),
                "blockers": len(self._state.blockers),
            },
            "experiments": {
                "total": len(self._experiments),
                "by_status": dict(exp_by_status),
            },
            "metrics": {
                k: v for k, v in self._state.metrics.items()
                if k != "metrics_history"
            },
            "gsd_tools_available": self._gsd_tools_available,
        }

    # --- Persistence -------------------------------------------------------

    def _persist_state(self):
        """Save state to disk."""
        state_file = self._planning_dir / "STATE.json"
        try:
            from dataclasses import asdict
            data = {
                "state": asdict(self._state),
                "experiments": {
                    eid: asdict(exp) for eid, exp in self._experiments.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug("GSD state persistence failed: %s", e)

    def _load_state(self):
        """Load state from disk."""
        state_file = self._planning_dir / "STATE.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                data = json.load(f)

            # Restore state
            state_data = data.get("state", {})
            for key, value in state_data.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

            # Restore experiments
            for eid, exp_data in data.get("experiments", {}).items():
                phases = []
                for p_data in exp_data.pop("phases", []):
                    phases.append(GSDPhase(**{
                        k: v for k, v in p_data.items()
                        if hasattr(GSDPhase, k)
                    }))
                exp = StrategyExperiment(**{
                    k: v for k, v in exp_data.items()
                    if hasattr(StrategyExperiment, k)
                })
                exp.phases = phases
                self._experiments[eid] = exp

            logger.info("GSD state loaded: %d experiments", len(self._experiments))
        except Exception as e:
            logger.debug("GSD state load failed: %s", e)

    def _log_event(self, event_type: str, data: dict):
        """Persist event to JSONL log."""
        log_file = self._log_dir / f"gsd_workflow_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("GSD workflow log write failed: %s", e)
