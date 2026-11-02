"""GSD & Paul Plugin — Gradient Signal Dynamics + Pattern Awareness & Unified Learning.

Facilitates dynamic learning loops for agents that are constantly learning
throughout the trading day. Two complementary systems:

**GSD (Gradient Signal Dynamics)**:
    Tracks signal gradients over time to identify momentum, convergence,
    divergence, and decay across all 12 signal engines. Agents learn from
    gradient dynamics (rate-of-change of signal quality), not just outcomes.
    Each agent carries a GSD profile tracking gradient history, confidence
    adjustments, cross-agent alignment, and adaptive learning rates.

**Paul (Pattern Awareness & Unified Learning)**:
    Named after the concept of pattern recognition in markets. Stores
    successful trade patterns with full market context, matches current
    state to historical patterns, evolves patterns as regimes change,
    and maintains a unified pattern library shared across all 34 agents
    (11 sector bots + 11 research bots + 12 investor personas).

Integration points:
    - LearningLoop (engine/monitoring/learning_loop.py) — feedback channel
    - Agent Scorecard (engine/agents/agent_scorecard.py) — performance tiers
    - MLVoteEnsemble (10 tiers) — gradient-adjusted confidence
    - All signal engines — gradient tracking per engine
    - 34 agents — GSD profiles + pattern replay

Learning flow:
    Market State → GSD gradient analysis → confidence adjustment
                → Paul pattern matching → enriched decision context
                → Agent decision → Outcome
                → GSD profile update + Paul pattern storage
                → Adaptive learning rate recalibration
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GSD_HISTORY_MAXLEN = 100          # Gradient observations per agent
PATTERN_LIBRARY_MAXLEN = 5000     # Max stored patterns
PATTERN_STALE_DAYS = 90           # Prune patterns older than this
CONVERGENCE_THRESHOLD = 0.70      # Multi-signal alignment threshold
DIVERGENCE_THRESHOLD = 0.30       # Signal conflict threshold
MIN_GRADIENT_SAMPLES = 5          # Min samples before gradient computation
GRADIENT_DECAY_HALFLIFE = 20.0    # Default decay half-life in observations
BASE_LEARNING_RATE = 0.01         # Default agent learning rate
LR_STABILITY_WINDOW = 20         # Window for LR stability estimation
PATTERN_MATCH_TOP_K = 10         # Return top K matching patterns
CONTEXT_SIMILARITY_THRESHOLD = 0.5  # Min context similarity for match

# Signal engines from LearningLoop
SIGNAL_ENGINES = [
    "macro", "cube", "security_analysis", "pattern_discovery",
    "social", "distress", "cvr", "event_driven",
    "alpha_optimizer", "decision_matrix", "hft_technical",
    "ml_ensemble",
]


# ---------------------------------------------------------------------------
# Data classes — GSD
# ---------------------------------------------------------------------------
@dataclass
class GradientObservation:
    """Single gradient observation for a signal engine."""
    timestamp: str = ""
    engine: str = ""
    signal_value: float = 0.0
    gradient: float = 0.0           # First derivative of signal
    second_gradient: float = 0.0    # Second derivative (acceleration)
    confidence: float = 0.0
    regime: str = ""


@dataclass
class AgentGSDProfile:
    """GSD profile carried by each agent."""
    agent_id: str = ""
    gradient_history: list = field(default_factory=list)  # Last N GradientObservations
    gradient_weighted_confidence: float = 0.5
    cross_agent_alignment: float = 0.0
    learning_rate: float = BASE_LEARNING_RATE
    total_updates: int = 0
    cumulative_gradient_score: float = 0.0
    last_updated: str = ""

    # Per-engine gradient tracking
    engine_gradients: dict = field(default_factory=dict)  # engine -> list of floats


@dataclass
class ConvergenceEvent:
    """Recorded when multiple signals converge."""
    timestamp: str = ""
    engines_aligned: list = field(default_factory=list)
    alignment_score: float = 0.0
    direction: str = ""             # BULLISH / BEARISH / NEUTRAL
    ticker: str = ""
    regime: str = ""


@dataclass
class DivergenceEvent:
    """Recorded when signals conflict."""
    timestamp: str = ""
    bullish_engines: list = field(default_factory=list)
    bearish_engines: list = field(default_factory=list)
    divergence_score: float = 0.0
    ticker: str = ""
    regime: str = ""


# ---------------------------------------------------------------------------
# Data classes — Paul
# ---------------------------------------------------------------------------
@dataclass
class TradePattern:
    """A stored trade pattern with full context."""
    pattern_id: str = ""
    timestamp: str = ""
    ticker: str = ""
    sector: str = ""
    direction: str = ""             # BUY / SELL / SHORT / COVER
    entry_price: float = 0.0
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    sharpe_contribution: float = 0.0
    holding_period_days: int = 0

    # Market context at time of trade
    regime: str = ""                # TRENDING / RANGE / STRESS / CRASH
    volatility: float = 0.0
    liquidity_score: float = 0.0
    vix_level: float = 0.0
    spread_bps: float = 0.0

    # Agent consensus at time of trade
    consensus_score: float = 0.0
    num_agents_agreed: int = 0
    vote_score: float = 0.0

    # Signal context
    signal_engines_active: list = field(default_factory=list)
    signal_strengths: dict = field(default_factory=dict)

    # Outcome tracking
    was_successful: bool = False
    subsequent_5d_return: float = 0.0
    subsequent_20d_return: float = 0.0
    pattern_repeated: bool = False
    repeat_count: int = 0

    # Pattern features (for matching)
    feature_vector: list = field(default_factory=list)

    # Metadata
    created_at: str = ""
    last_matched: str = ""
    match_count: int = 0
    success_rate: float = 0.0
    regime_at_creation: str = ""


@dataclass
class PatternMatch:
    """Result of pattern matching against current state."""
    pattern_id: str = ""
    similarity_score: float = 0.0
    context_score: float = 0.0
    combined_score: float = 0.0
    pattern: Optional[TradePattern] = None
    applicable: bool = False


@dataclass
class LearningState:
    """Snapshot of the combined GSD + Paul learning state."""
    timestamp: str = ""
    total_agents_tracked: int = 0
    total_patterns_stored: int = 0
    avg_gradient_confidence: float = 0.0
    cross_engine_alignment: float = 0.0
    top_patterns_by_success: list = field(default_factory=list)
    regime_pattern_counts: dict = field(default_factory=dict)
    stale_patterns_pruned: int = 0
    total_gsd_updates: int = 0
    total_paul_matches: int = 0


# ---------------------------------------------------------------------------
# GSD Plugin — Gradient Signal Dynamics
# ---------------------------------------------------------------------------
class GSDPlugin:
    """Tracks signal gradients across all engines to drive agent learning.

    Instead of learning from binary outcomes (correct/incorrect), GSD
    captures the rate-of-change of signal quality over time. This allows
    agents to anticipate signal degradation, detect convergence/divergence
    early, and adjust confidence dynamically.

    Thread-safe: all mutable state guarded by a reentrant lock.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self._lock = threading.RLock()
        self.log_dir = log_dir or Path("logs/gsd_plugin")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Per-engine signal history: engine -> deque of (timestamp, value)
        self._signal_history: dict[str, deque] = {
            eng: deque(maxlen=GSD_HISTORY_MAXLEN) for eng in SIGNAL_ENGINES
        }

        # Per-agent GSD profiles
        self._agent_profiles: dict[str, AgentGSDProfile] = {}

        # Convergence / divergence event logs
        self._convergence_events: deque = deque(maxlen=500)
        self._divergence_events: deque = deque(maxlen=500)

        # Cross-engine correlation cache
        self._cross_engine_corr: dict[tuple, float] = {}
        self._cross_engine_alignment: float = 0.0

        # Counters
        self._total_updates = 0

    # --- Core gradient computation -----------------------------------------

    def compute_signal_gradient(
        self,
        engine: str,
        signal_history: Optional[list[float]] = None,
    ) -> dict[str, float]:
        """Compute gradient (first + second derivative) of signal over time.

        Args:
            engine: Signal engine name.
            signal_history: Optional explicit signal history. If None,
                uses the internally tracked history for this engine.

        Returns:
            Dict with keys: gradient, second_gradient, momentum,
            decay_rate, stability.
        """
        if np is None:
            return {"gradient": 0.0, "second_gradient": 0.0,
                    "momentum": 0.0, "decay_rate": 0.0, "stability": 0.0}

        with self._lock:
            if signal_history is not None:
                values = np.array(signal_history, dtype=np.float64)
            elif engine in self._signal_history:
                raw = list(self._signal_history[engine])
                if not raw:
                    return {"gradient": 0.0, "second_gradient": 0.0,
                            "momentum": 0.0, "decay_rate": 0.0, "stability": 0.0}
                values = np.array([v for _, v in raw], dtype=np.float64)
            else:
                return {"gradient": 0.0, "second_gradient": 0.0,
                        "momentum": 0.0, "decay_rate": 0.0, "stability": 0.0}

        if len(values) < MIN_GRADIENT_SAMPLES:
            return {"gradient": 0.0, "second_gradient": 0.0,
                    "momentum": 0.0, "decay_rate": 0.0, "stability": 0.0}

        # First derivative: finite differences
        grad = np.diff(values)
        current_gradient = float(grad[-1]) if len(grad) > 0 else 0.0

        # Second derivative: acceleration of gradient
        if len(grad) >= 2:
            grad2 = np.diff(grad)
            second_gradient = float(grad2[-1])
        else:
            second_gradient = 0.0

        # Momentum: exponentially weighted moving average of gradient
        alpha = 2.0 / (min(len(grad), 20) + 1)
        ewma = 0.0
        for g in grad:
            ewma = alpha * g + (1.0 - alpha) * ewma
        momentum = float(ewma)

        # Decay rate: fit exponential decay to absolute signal strength
        abs_vals = np.abs(values)
        if len(abs_vals) >= MIN_GRADIENT_SAMPLES and abs_vals[0] > 1e-10:
            # Log-linear regression for decay estimation
            nonzero_mask = abs_vals > 1e-10
            if np.sum(nonzero_mask) >= MIN_GRADIENT_SAMPLES:
                log_vals = np.log(abs_vals[nonzero_mask])
                x = np.arange(len(log_vals), dtype=np.float64)
                if len(x) >= 2:
                    # Linear regression: log(y) = a + b*x => decay_rate = b
                    x_mean = x.mean()
                    y_mean = log_vals.mean()
                    ss_xx = np.sum((x - x_mean) ** 2)
                    if ss_xx > 1e-10:
                        decay_rate = float(
                            np.sum((x - x_mean) * (log_vals - y_mean)) / ss_xx
                        )
                    else:
                        decay_rate = 0.0
                else:
                    decay_rate = 0.0
            else:
                decay_rate = 0.0
        else:
            decay_rate = 0.0

        # Stability: inverse of gradient variance (higher = more stable)
        grad_std = float(np.std(grad)) if len(grad) > 1 else 0.0
        stability = 1.0 / (1.0 + grad_std)

        return {
            "gradient": current_gradient,
            "second_gradient": second_gradient,
            "momentum": momentum,
            "decay_rate": decay_rate,
            "stability": stability,
        }

    def record_signal(self, engine: str, value: float, timestamp: Optional[str] = None):
        """Record a signal value for an engine."""
        ts = timestamp or datetime.now().isoformat()
        with self._lock:
            if engine not in self._signal_history:
                self._signal_history[engine] = deque(maxlen=GSD_HISTORY_MAXLEN)
            self._signal_history[engine].append((ts, value))

    # --- Convergence / divergence detection --------------------------------

    def detect_convergence(
        self, all_signals: dict[str, float],
    ) -> Optional[ConvergenceEvent]:
        """Detect when multiple signal engines align in direction.

        Args:
            all_signals: Dict of engine_name -> signal_value.
                Positive values = bullish, negative = bearish.

        Returns:
            ConvergenceEvent if alignment exceeds threshold, else None.
        """
        if not all_signals or len(all_signals) < 2:
            return None

        values = np.array(list(all_signals.values()), dtype=np.float64)
        signs = np.sign(values)

        # Alignment: fraction of engines agreeing on direction
        bullish_count = int(np.sum(signs > 0))
        bearish_count = int(np.sum(signs < 0))
        total = len(values)

        max_aligned = max(bullish_count, bearish_count)
        alignment_score = max_aligned / total

        if alignment_score >= CONVERGENCE_THRESHOLD:
            direction = "BULLISH" if bullish_count >= bearish_count else "BEARISH"
            aligned_engines = [
                eng for eng, val in all_signals.items()
                if (direction == "BULLISH" and val > 0) or
                   (direction == "BEARISH" and val < 0)
            ]
            event = ConvergenceEvent(
                timestamp=datetime.now().isoformat(),
                engines_aligned=aligned_engines,
                alignment_score=float(alignment_score),
                direction=direction,
            )
            with self._lock:
                self._convergence_events.append(event)
            self._log_event("convergence", asdict(event))
            return event
        return None

    def detect_divergence(
        self, all_signals: dict[str, float],
    ) -> Optional[DivergenceEvent]:
        """Detect when signals conflict (some bullish, some bearish).

        Args:
            all_signals: Dict of engine_name -> signal_value.

        Returns:
            DivergenceEvent if divergence exceeds threshold, else None.
        """
        if not all_signals or len(all_signals) < 2:
            return None

        values = np.array(list(all_signals.values()), dtype=np.float64)
        signs = np.sign(values)

        bullish_engines = [e for e, v in all_signals.items() if v > 0]
        bearish_engines = [e for e, v in all_signals.items() if v < 0]

        if not bullish_engines or not bearish_engines:
            return None

        # Divergence: how evenly split are the engines?
        # Max divergence at 50/50 split
        total = len(values)
        min_side = min(len(bullish_engines), len(bearish_engines))
        divergence_score = (2.0 * min_side) / total  # 1.0 = perfect split

        if divergence_score >= DIVERGENCE_THRESHOLD:
            event = DivergenceEvent(
                timestamp=datetime.now().isoformat(),
                bullish_engines=bullish_engines,
                bearish_engines=bearish_engines,
                divergence_score=float(divergence_score),
            )
            with self._lock:
                self._divergence_events.append(event)
            self._log_event("divergence", asdict(event))
            return event
        return None

    # --- Agent GSD profile management --------------------------------------

    def get_gradient_confidence(self, agent_id: str) -> float:
        """Get gradient-adjusted confidence for an agent.

        The confidence is modulated by:
        1. Recent gradient stability (stable gradients = higher confidence)
        2. Cross-agent alignment (agreement boosts confidence)
        3. Signal momentum direction (strengthening signals = higher confidence)

        Returns:
            Confidence in [0.0, 1.0].
        """
        with self._lock:
            profile = self._agent_profiles.get(agent_id)
            if profile is None:
                return 0.5  # neutral default

            return float(np.clip(profile.gradient_weighted_confidence, 0.0, 1.0))

    def update_agent_gsd_profile(
        self,
        agent_id: str,
        outcome: dict,
    ) -> AgentGSDProfile:
        """Update an agent's GSD profile based on a trade outcome.

        Args:
            agent_id: Agent identifier.
            outcome: Dict with keys: realized_pnl, was_correct, signal_engine,
                signal_value, regime, confidence.

        Returns:
            Updated AgentGSDProfile.
        """
        with self._lock:
            if agent_id not in self._agent_profiles:
                self._agent_profiles[agent_id] = AgentGSDProfile(
                    agent_id=agent_id,
                    last_updated=datetime.now().isoformat(),
                )
            profile = self._agent_profiles[agent_id]

            engine = outcome.get("signal_engine", "unknown")
            signal_value = outcome.get("signal_value", 0.0)
            was_correct = outcome.get("was_correct", False)
            realized_pnl = outcome.get("realized_pnl", 0.0)
            regime = outcome.get("regime", "")
            confidence = outcome.get("confidence", 0.5)

            # Compute gradient for the engine that generated this signal
            grad_info = self.compute_signal_gradient(engine)
            gradient = grad_info["gradient"]
            stability = grad_info["stability"]

            # Record observation
            obs = GradientObservation(
                timestamp=datetime.now().isoformat(),
                engine=engine,
                signal_value=signal_value,
                gradient=gradient,
                second_gradient=grad_info["second_gradient"],
                confidence=confidence,
                regime=regime,
            )

            # Keep last GSD_HISTORY_MAXLEN observations
            profile.gradient_history.append(asdict(obs))
            if len(profile.gradient_history) > GSD_HISTORY_MAXLEN:
                profile.gradient_history = profile.gradient_history[-GSD_HISTORY_MAXLEN:]

            # Track per-engine gradients
            if engine not in profile.engine_gradients:
                profile.engine_gradients[engine] = []
            profile.engine_gradients[engine].append(gradient)
            if len(profile.engine_gradients[engine]) > GSD_HISTORY_MAXLEN:
                profile.engine_gradients[engine] = \
                    profile.engine_gradients[engine][-GSD_HISTORY_MAXLEN:]

            # Update gradient-weighted confidence
            # Correct outcomes on strong gradients boost confidence more
            lr = profile.learning_rate
            outcome_signal = 1.0 if was_correct else -1.0
            gradient_weight = stability * (1.0 + abs(gradient))
            adjustment = lr * outcome_signal * gradient_weight

            profile.gradient_weighted_confidence = float(np.clip(
                profile.gradient_weighted_confidence + adjustment, 0.0, 1.0
            ))

            # Update cumulative gradient score
            profile.cumulative_gradient_score += gradient * outcome_signal
            profile.total_updates += 1
            profile.last_updated = datetime.now().isoformat()

            # Recompute adaptive learning rate
            profile.learning_rate = self._compute_lr(profile)

            self._total_updates += 1

        self._log_event("gsd_update", {
            "agent_id": agent_id,
            "engine": engine,
            "gradient": gradient,
            "stability": stability,
            "was_correct": was_correct,
            "new_confidence": profile.gradient_weighted_confidence,
            "learning_rate": profile.learning_rate,
        })

        return profile

    def get_cross_engine_alignment(self) -> dict[str, Any]:
        """Compute alignment score across all signal engines.

        Measures how correlated engine outputs are over their recent
        history. High alignment suggests strong market conviction;
        low alignment suggests uncertainty.

        Returns:
            Dict with alignment_score, correlation_matrix, and
            per-pair correlations.
        """
        if np is None:
            return {"alignment_score": 0.0, "correlations": {}}

        with self._lock:
            # Collect recent values for each engine
            engine_values = {}
            for eng, history in self._signal_history.items():
                vals = [v for _, v in history]
                if len(vals) >= MIN_GRADIENT_SAMPLES:
                    engine_values[eng] = vals

        if len(engine_values) < 2:
            return {"alignment_score": 0.0, "correlations": {}}

        # Align all series to the same length (shortest)
        min_len = min(len(v) for v in engine_values.values())
        engines = sorted(engine_values.keys())
        matrix = np.array(
            [engine_values[eng][-min_len:] for eng in engines],
            dtype=np.float64,
        )

        # Compute pairwise Pearson correlations
        n_engines = len(engines)
        corr_matrix = np.corrcoef(matrix)

        # Handle NaN from constant series
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper triangle (excluding diagonal)
        pair_corrs = {}
        total_corr = 0.0
        n_pairs = 0
        for i in range(n_engines):
            for j in range(i + 1, n_engines):
                pair_key = f"{engines[i]}|{engines[j]}"
                corr_val = float(corr_matrix[i, j])
                pair_corrs[pair_key] = corr_val
                total_corr += abs(corr_val)
                n_pairs += 1

        alignment_score = total_corr / max(n_pairs, 1)

        with self._lock:
            self._cross_engine_alignment = alignment_score
            self._cross_engine_corr = {
                (engines[i], engines[j]): float(corr_matrix[i, j])
                for i in range(n_engines)
                for j in range(i + 1, n_engines)
            }

        return {
            "alignment_score": float(alignment_score),
            "correlations": pair_corrs,
            "n_engines": n_engines,
            "n_pairs": n_pairs,
        }

    def compute_adaptive_learning_rate(self, agent_id: str) -> float:
        """Compute dynamic learning rate for an agent based on recent
        gradient stability.

        Stable gradients (low variance) → lower LR (fine-tuning mode).
        Volatile gradients → higher LR (rapid adaptation mode).

        Returns:
            Learning rate in [0.001, 0.1].
        """
        with self._lock:
            profile = self._agent_profiles.get(agent_id)
            if profile is None:
                return BASE_LEARNING_RATE
            return self._compute_lr(profile)

    def _compute_lr(self, profile: AgentGSDProfile) -> float:
        """Internal learning rate computation from profile gradient history."""
        if np is None or len(profile.gradient_history) < MIN_GRADIENT_SAMPLES:
            return BASE_LEARNING_RATE

        # Extract recent gradients
        recent = profile.gradient_history[-LR_STABILITY_WINDOW:]
        grads = np.array([obs["gradient"] for obs in recent], dtype=np.float64)

        if len(grads) < 2:
            return BASE_LEARNING_RATE

        grad_std = float(np.std(grads))
        grad_mean_abs = float(np.mean(np.abs(grads)))

        # High volatility → higher LR (need to adapt fast)
        # Low volatility → lower LR (fine-tuning)
        # Scaled sigmoid: maps std to [0.001, 0.1]
        lr_min, lr_max = 0.001, 0.1
        # Normalize: std of ~0 → lr_min, std of ~1+ → lr_max
        t = min(grad_std / (grad_mean_abs + 1e-8), 3.0) / 3.0
        lr = lr_min + (lr_max - lr_min) * t

        return float(np.clip(lr, lr_min, lr_max))

    # --- Logging -----------------------------------------------------------

    def log_gradient_state(self) -> dict:
        """Generate comprehensive gradient state log.

        Returns:
            Dict summarizing current GSD state across all engines and agents.
        """
        with self._lock:
            engine_gradients = {}
            for eng in SIGNAL_ENGINES:
                grad_info = self.compute_signal_gradient(eng)
                history_len = len(self._signal_history.get(eng, []))
                engine_gradients[eng] = {
                    **grad_info,
                    "history_length": history_len,
                }

            agent_summaries = {}
            for agent_id, profile in self._agent_profiles.items():
                agent_summaries[agent_id] = {
                    "confidence": profile.gradient_weighted_confidence,
                    "learning_rate": profile.learning_rate,
                    "total_updates": profile.total_updates,
                    "cumulative_score": profile.cumulative_gradient_score,
                    "n_observations": len(profile.gradient_history),
                }

        state = {
            "timestamp": datetime.now().isoformat(),
            "engine_gradients": engine_gradients,
            "agent_summaries": agent_summaries,
            "cross_engine_alignment": self._cross_engine_alignment,
            "total_convergence_events": len(self._convergence_events),
            "total_divergence_events": len(self._divergence_events),
            "total_updates": self._total_updates,
        }

        self._log_event("gradient_state", state)
        return state

    def _log_event(self, event_type: str, data: dict):
        """Persist a learning event to JSONL log."""
        log_file = self.log_dir / f"gsd_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("GSD log write failed: %s", e)


# ---------------------------------------------------------------------------
# Paul Plugin — Pattern Awareness & Unified Learning
# ---------------------------------------------------------------------------
class PaulPlugin:
    """Pattern recognition and unified learning across all agents.

    Stores successful trade patterns with full market context, then
    matches current market states against this library. Patterns evolve
    as market regimes change and stale patterns are pruned.

    Thread-safe: all mutable state guarded by a reentrant lock.
    """

    def __init__(self, log_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self._lock = threading.RLock()
        self.log_dir = log_dir or Path("logs/paul_plugin")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir or Path("data/paul_patterns")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Pattern library: pattern_id -> TradePattern
        self._patterns: dict[str, TradePattern] = {}

        # Index by regime for fast lookup
        self._regime_index: dict[str, list[str]] = defaultdict(list)

        # Index by sector
        self._sector_index: dict[str, list[str]] = defaultdict(list)

        # Match counters
        self._total_matches = 0
        self._total_replays = 0

        # Load persisted patterns
        self._load_patterns()

    # --- Pattern storage ---------------------------------------------------

    def store_pattern(self, pattern: TradePattern) -> str:
        """Save a successful trade pattern to the library.

        Args:
            pattern: TradePattern with full context.

        Returns:
            Pattern ID.
        """
        if not pattern.pattern_id:
            pattern.pattern_id = str(uuid.uuid4())[:12]
        pattern.created_at = pattern.created_at or datetime.now().isoformat()
        pattern.regime_at_creation = pattern.regime_at_creation or pattern.regime

        # Compute feature vector if not set
        if not pattern.feature_vector:
            pattern.feature_vector = self._extract_features(pattern)

        with self._lock:
            # Enforce library size limit
            if len(self._patterns) >= PATTERN_LIBRARY_MAXLEN:
                self._evict_weakest_pattern()

            self._patterns[pattern.pattern_id] = pattern

            # Update indices
            if pattern.regime:
                self._regime_index[pattern.regime].append(pattern.pattern_id)
            if pattern.sector:
                self._sector_index[pattern.sector].append(pattern.pattern_id)

        self._log_event("pattern_stored", {
            "pattern_id": pattern.pattern_id,
            "ticker": pattern.ticker,
            "direction": pattern.direction,
            "regime": pattern.regime,
            "realized_pnl": pattern.realized_pnl,
            "was_successful": pattern.was_successful,
        })

        # Persist to disk
        self._persist_pattern(pattern)

        return pattern.pattern_id

    def _extract_features(self, pattern: TradePattern) -> list[float]:
        """Extract a numeric feature vector from a pattern for similarity matching.

        Feature vector components:
        [0] direction_encoded: BUY=1, SELL=-1, SHORT=-1, COVER=1
        [1] normalized_pnl: realized_pnl / (entry_price + 1e-8)
        [2] regime_encoded: TRENDING=1, RANGE=0, STRESS=-0.5, CRASH=-1
        [3] volatility
        [4] liquidity_score
        [5] vix_level / 80 (normalized)
        [6] consensus_score
        [7] vote_score
        [8] holding_period_days / 30 (normalized)
        [9] num_engines_active / len(SIGNAL_ENGINES)
        """
        direction_map = {"BUY": 1.0, "COVER": 1.0, "SELL": -1.0, "SHORT": -1.0}
        regime_map = {"TRENDING": 1.0, "RANGE": 0.0, "STRESS": -0.5, "CRASH": -1.0}

        return [
            direction_map.get(pattern.direction, 0.0),
            pattern.realized_pnl / (abs(pattern.entry_price) + 1e-8),
            regime_map.get(pattern.regime, 0.0),
            pattern.volatility,
            pattern.liquidity_score,
            pattern.vix_level / 80.0,
            pattern.consensus_score,
            pattern.vote_score,
            pattern.holding_period_days / 30.0,
            len(pattern.signal_engines_active) / max(len(SIGNAL_ENGINES), 1),
        ]

    def _evict_weakest_pattern(self):
        """Remove the least valuable pattern to make room.

        Eviction priority: lowest success_rate, then oldest, then
        fewest match_count.
        """
        if not self._patterns:
            return

        weakest_id = min(
            self._patterns,
            key=lambda pid: (
                self._patterns[pid].success_rate,
                self._patterns[pid].match_count,
                -len(self._patterns[pid].created_at or "9999"),  # older first
            ),
        )

        pattern = self._patterns.pop(weakest_id)
        # Clean indices
        if pattern.regime and weakest_id in self._regime_index.get(pattern.regime, []):
            self._regime_index[pattern.regime].remove(weakest_id)
        if pattern.sector and weakest_id in self._sector_index.get(pattern.sector, []):
            self._sector_index[pattern.sector].remove(weakest_id)

    # --- Pattern matching --------------------------------------------------

    def match_pattern(
        self,
        current_state: dict,
        top_k: int = PATTERN_MATCH_TOP_K,
    ) -> list[PatternMatch]:
        """Find patterns most similar to the current market state.

        Args:
            current_state: Dict with keys matching TradePattern fields:
                regime, volatility, liquidity_score, vix_level,
                consensus_score, vote_score, direction, sector.
            top_k: Number of top matches to return.

        Returns:
            List of PatternMatch sorted by combined_score descending.
        """
        if np is None:
            return []

        # Build query feature vector
        query_features = self._state_to_features(current_state)
        query_vec = np.array(query_features, dtype=np.float64)

        matches = []
        with self._lock:
            # Pre-filter by regime if available
            regime = current_state.get("regime", "")
            if regime and regime in self._regime_index:
                candidate_ids = self._regime_index[regime]
            else:
                candidate_ids = list(self._patterns.keys())

            for pid in candidate_ids:
                pattern = self._patterns.get(pid)
                if pattern is None:
                    continue

                if not pattern.feature_vector:
                    pattern.feature_vector = self._extract_features(pattern)

                pattern_vec = np.array(pattern.feature_vector, dtype=np.float64)

                # Ensure vectors are same length
                min_len = min(len(query_vec), len(pattern_vec))
                if min_len == 0:
                    continue

                q = query_vec[:min_len]
                p = pattern_vec[:min_len]

                # Cosine similarity
                dot = float(np.dot(q, p))
                norm_q = float(np.linalg.norm(q))
                norm_p = float(np.linalg.norm(p))
                if norm_q < 1e-10 or norm_p < 1e-10:
                    similarity = 0.0
                else:
                    similarity = dot / (norm_q * norm_p)

                # Context score (regime + sector match bonus)
                context_score = self.get_context_score(pattern, current_state)

                # Combined: 60% similarity + 40% context
                combined = 0.6 * similarity + 0.4 * context_score

                if combined >= CONTEXT_SIMILARITY_THRESHOLD:
                    matches.append(PatternMatch(
                        pattern_id=pid,
                        similarity_score=float(similarity),
                        context_score=float(context_score),
                        combined_score=float(combined),
                        pattern=pattern,
                        applicable=True,
                    ))

        # Sort by combined score
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        top_matches = matches[:top_k]

        # Update match counts
        with self._lock:
            now = datetime.now().isoformat()
            for m in top_matches:
                if m.pattern and m.pattern.pattern_id in self._patterns:
                    self._patterns[m.pattern.pattern_id].match_count += 1
                    self._patterns[m.pattern.pattern_id].last_matched = now
            self._total_matches += len(top_matches)

        if top_matches:
            self._log_event("pattern_match", {
                "query_regime": current_state.get("regime", ""),
                "n_candidates": len(candidate_ids),
                "n_matches": len(top_matches),
                "top_score": top_matches[0].combined_score if top_matches else 0.0,
            })

        return top_matches

    def _state_to_features(self, state: dict) -> list[float]:
        """Convert a market state dict to a feature vector."""
        direction_map = {"BUY": 1.0, "COVER": 1.0, "SELL": -1.0, "SHORT": -1.0}
        regime_map = {"TRENDING": 1.0, "RANGE": 0.0, "STRESS": -0.5, "CRASH": -1.0}

        return [
            direction_map.get(state.get("direction", ""), 0.0),
            0.0,  # pnl not known yet for current state
            regime_map.get(state.get("regime", ""), 0.0),
            state.get("volatility", 0.0),
            state.get("liquidity_score", 0.0),
            state.get("vix_level", 0.0) / 80.0,
            state.get("consensus_score", 0.0),
            state.get("vote_score", 0.0),
            0.0,  # holding period not known yet
            state.get("n_engines_active", 0) / max(len(SIGNAL_ENGINES), 1),
        ]

    # --- Pattern replay ----------------------------------------------------

    def replay_pattern(
        self,
        pattern_id: str,
        context: dict,
    ) -> dict:
        """Replay a stored pattern in the current market context.

        Compares the original pattern context to current context and
        produces an adjusted recommendation with confidence.

        Args:
            pattern_id: ID of pattern to replay.
            context: Current market context dict.

        Returns:
            Dict with direction, confidence, adjustments, and warnings.
        """
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern is None:
                return {"error": f"Pattern {pattern_id} not found", "confidence": 0.0}

        context_score = self.get_context_score(pattern, context)

        # Adjust confidence based on context similarity
        base_confidence = pattern.success_rate
        adjusted_confidence = base_confidence * context_score

        # Detect regime drift
        warnings = []
        current_regime = context.get("regime", "")
        if current_regime and current_regime != pattern.regime:
            warnings.append(
                f"Regime mismatch: pattern={pattern.regime}, current={current_regime}"
            )
            adjusted_confidence *= 0.7  # Penalize regime mismatch

        # Volatility adjustment
        current_vol = context.get("volatility", 0.0)
        if pattern.volatility > 0 and current_vol > 0:
            vol_ratio = current_vol / (pattern.volatility + 1e-8)
            if vol_ratio > 1.5:
                warnings.append(
                    f"Volatility {vol_ratio:.1f}x higher than pattern context"
                )
                adjusted_confidence *= 0.8
            elif vol_ratio < 0.5:
                warnings.append(
                    f"Volatility {vol_ratio:.1f}x lower than pattern context"
                )

        result = {
            "pattern_id": pattern_id,
            "direction": pattern.direction,
            "ticker": pattern.ticker,
            "sector": pattern.sector,
            "confidence": float(np.clip(adjusted_confidence, 0.0, 1.0)),
            "context_score": float(context_score),
            "original_pnl": pattern.realized_pnl,
            "original_regime": pattern.regime,
            "current_regime": current_regime,
            "warnings": warnings,
            "repeat_count": pattern.repeat_count,
            "historical_success_rate": pattern.success_rate,
        }

        with self._lock:
            self._total_replays += 1

        self._log_event("pattern_replay", result)
        return result

    # --- Pattern evolution --------------------------------------------------

    def evolve_patterns(self, regime_change: dict) -> int:
        """Update patterns when market regime changes.

        Adjusts success_rate and feature vectors of stored patterns
        based on the new regime. Patterns that thrived in the old
        regime may be less applicable.

        Args:
            regime_change: Dict with old_regime, new_regime, and
                optional transition_confidence.

        Returns:
            Number of patterns affected.
        """
        old_regime = regime_change.get("old_regime", "")
        new_regime = regime_change.get("new_regime", "")
        transition_confidence = regime_change.get("transition_confidence", 0.8)

        if not old_regime or not new_regime or old_regime == new_regime:
            return 0

        affected = 0

        # Regime distance: how different are the regimes?
        regime_order = {"TRENDING": 3, "RANGE": 2, "STRESS": 1, "CRASH": 0}
        old_rank = regime_order.get(old_regime, 2)
        new_rank = regime_order.get(new_regime, 2)
        regime_distance = abs(old_rank - new_rank) / 3.0  # Normalized to [0, 1]

        # Decay factor: farther regime transitions decay patterns more
        decay_factor = 1.0 - (regime_distance * 0.3 * transition_confidence)

        with self._lock:
            old_regime_patterns = self._regime_index.get(old_regime, [])
            for pid in old_regime_patterns:
                pattern = self._patterns.get(pid)
                if pattern is None:
                    continue
                # Decay success rate for patterns from old regime
                pattern.success_rate *= decay_factor
                # Recompute feature vector with decayed values
                pattern.feature_vector = self._extract_features(pattern)
                affected += 1

        self._log_event("pattern_evolution", {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "regime_distance": regime_distance,
            "decay_factor": decay_factor,
            "patterns_affected": affected,
        })

        return affected

    # --- Unified library ---------------------------------------------------

    def get_unified_library(
        self,
        regime: Optional[str] = None,
        sector: Optional[str] = None,
        min_success_rate: float = 0.0,
        limit: int = 100,
    ) -> list[TradePattern]:
        """Get all patterns sorted by success rate, optionally filtered.

        Args:
            regime: Filter by regime.
            sector: Filter by sector.
            min_success_rate: Minimum success rate filter.
            limit: Max patterns to return.

        Returns:
            List of TradePattern sorted by success_rate descending.
        """
        with self._lock:
            if regime and regime in self._regime_index:
                candidate_ids = set(self._regime_index[regime])
            else:
                candidate_ids = set(self._patterns.keys())

            if sector and sector in self._sector_index:
                sector_ids = set(self._sector_index[sector])
                candidate_ids = candidate_ids & sector_ids

            patterns = []
            for pid in candidate_ids:
                p = self._patterns.get(pid)
                if p and p.success_rate >= min_success_rate:
                    patterns.append(p)

        patterns.sort(key=lambda p: p.success_rate, reverse=True)
        return patterns[:limit]

    # --- Context scoring ---------------------------------------------------

    def get_context_score(self, pattern: TradePattern, current_context: dict) -> float:
        """Score how applicable a pattern is in the current context.

        Considers regime match, volatility similarity, liquidity,
        VIX proximity, and consensus alignment.

        Returns:
            Score in [0.0, 1.0].
        """
        score = 0.0
        weights_total = 0.0

        # Regime match (weight: 3.0)
        w_regime = 3.0
        if pattern.regime == current_context.get("regime", ""):
            score += w_regime * 1.0
        elif pattern.regime and current_context.get("regime"):
            # Partial credit for adjacent regimes
            regime_order = {"TRENDING": 3, "RANGE": 2, "STRESS": 1, "CRASH": 0}
            pr = regime_order.get(pattern.regime, 2)
            cr = regime_order.get(current_context.get("regime", ""), 2)
            distance = abs(pr - cr) / 3.0
            score += w_regime * (1.0 - distance)
        weights_total += w_regime

        # Volatility similarity (weight: 2.0)
        w_vol = 2.0
        p_vol = pattern.volatility
        c_vol = current_context.get("volatility", 0.0)
        if p_vol > 0 and c_vol > 0:
            vol_sim = 1.0 - min(abs(p_vol - c_vol) / (max(p_vol, c_vol) + 1e-8), 1.0)
            score += w_vol * vol_sim
        elif p_vol == 0 and c_vol == 0:
            score += w_vol * 0.5  # neutral
        weights_total += w_vol

        # VIX proximity (weight: 1.5)
        w_vix = 1.5
        p_vix = pattern.vix_level
        c_vix = current_context.get("vix_level", 0.0)
        if p_vix > 0 and c_vix > 0:
            vix_sim = 1.0 - min(abs(p_vix - c_vix) / 40.0, 1.0)
            score += w_vix * vix_sim
        weights_total += w_vix

        # Consensus alignment (weight: 1.5)
        w_cons = 1.5
        p_cons = pattern.consensus_score
        c_cons = current_context.get("consensus_score", 0.0)
        if p_cons != 0 or c_cons != 0:
            cons_sim = 1.0 - min(abs(p_cons - c_cons) / 2.0, 1.0)
            score += w_cons * cons_sim
        weights_total += w_cons

        # Sector match (weight: 2.0)
        w_sector = 2.0
        if pattern.sector and pattern.sector == current_context.get("sector", ""):
            score += w_sector * 1.0
        elif not pattern.sector:
            score += w_sector * 0.5  # no sector info, neutral
        weights_total += w_sector

        if weights_total == 0:
            return 0.0

        return float(score / weights_total)

    # --- Pattern pruning ---------------------------------------------------

    def prune_stale_patterns(self, max_age_days: int = PATTERN_STALE_DAYS) -> int:
        """Remove patterns that are too old and have low success rates.

        Patterns are pruned if:
        1. Older than max_age_days AND success_rate < 0.4
        2. Never matched and older than max_age_days / 2

        Returns:
            Number of patterns pruned.
        """
        now = datetime.now()
        pruned = 0
        to_remove = []

        with self._lock:
            for pid, pattern in self._patterns.items():
                created = pattern.created_at or pattern.timestamp
                if not created:
                    continue
                try:
                    created_dt = datetime.fromisoformat(created)
                except (ValueError, TypeError):
                    continue

                age_days = (now - created_dt).days

                # Rule 1: old + low success
                if age_days > max_age_days and pattern.success_rate < 0.4:
                    to_remove.append(pid)
                    continue

                # Rule 2: never matched + half max age
                if pattern.match_count == 0 and age_days > max_age_days // 2:
                    to_remove.append(pid)
                    continue

            for pid in to_remove:
                pattern = self._patterns.pop(pid, None)
                if pattern:
                    if pattern.regime and pid in self._regime_index.get(pattern.regime, []):
                        self._regime_index[pattern.regime].remove(pid)
                    if pattern.sector and pid in self._sector_index.get(pattern.sector, []):
                        self._sector_index[pattern.sector].remove(pid)
                    pruned += 1

        if pruned > 0:
            self._log_event("patterns_pruned", {
                "count": pruned,
                "max_age_days": max_age_days,
                "remaining_patterns": len(self._patterns),
            })

        return pruned

    # --- Logging -----------------------------------------------------------

    def log_learning_state(self) -> dict:
        """Generate comprehensive Paul learning state log.

        Returns:
            Dict summarizing pattern library state.
        """
        with self._lock:
            regime_counts = {
                regime: len(pids) for regime, pids in self._regime_index.items()
            }
            sector_counts = {
                sector: len(pids) for sector, pids in self._sector_index.items()
            }

            # Top patterns by success rate
            all_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.success_rate,
                reverse=True,
            )
            top_patterns = [
                {
                    "pattern_id": p.pattern_id,
                    "ticker": p.ticker,
                    "direction": p.direction,
                    "regime": p.regime,
                    "success_rate": p.success_rate,
                    "match_count": p.match_count,
                    "realized_pnl": p.realized_pnl,
                }
                for p in all_patterns[:20]
            ]

            # Success rate distribution
            success_rates = [p.success_rate for p in self._patterns.values()]

        if np is not None and success_rates:
            sr_array = np.array(success_rates)
            sr_stats = {
                "mean": float(np.mean(sr_array)),
                "median": float(np.median(sr_array)),
                "std": float(np.std(sr_array)),
                "min": float(np.min(sr_array)),
                "max": float(np.max(sr_array)),
            }
        else:
            sr_stats = {}

        state = {
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(self._patterns),
            "regime_distribution": regime_counts,
            "sector_distribution": sector_counts,
            "top_patterns": top_patterns,
            "success_rate_stats": sr_stats,
            "total_matches": self._total_matches,
            "total_replays": self._total_replays,
        }

        self._log_event("learning_state", state)
        return state

    # --- Persistence -------------------------------------------------------

    def _persist_pattern(self, pattern: TradePattern):
        """Write a single pattern to JSONL file."""
        log_file = self.data_dir / f"patterns_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(asdict(pattern), default=str) + "\n")
        except Exception as e:
            logger.debug("Paul pattern persist failed: %s", e)

    def _load_patterns(self):
        """Load persisted patterns from data directory on startup."""
        if not self.data_dir.exists():
            return

        loaded = 0
        try:
            for pattern_file in sorted(self.data_dir.glob("patterns_*.jsonl")):
                with open(pattern_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            pattern = TradePattern(**{
                                k: v for k, v in data.items()
                                if k in TradePattern.__dataclass_fields__
                            })
                            if pattern.pattern_id:
                                self._patterns[pattern.pattern_id] = pattern
                                if pattern.regime:
                                    self._regime_index[pattern.regime].append(
                                        pattern.pattern_id
                                    )
                                if pattern.sector:
                                    self._sector_index[pattern.sector].append(
                                        pattern.pattern_id
                                    )
                                loaded += 1
                        except (json.JSONDecodeError, TypeError):
                            continue
        except Exception as e:
            logger.warning("Paul pattern load failed: %s", e)

        if loaded > 0:
            logger.info("Paul: loaded %d patterns from disk", loaded)

    def serialize_library(self, filepath: Optional[Path] = None) -> Path:
        """Serialize the entire pattern library to a JSON file.

        Args:
            filepath: Output path. Defaults to data_dir/library_<date>.json.

        Returns:
            Path to the written file.
        """
        filepath = filepath or self.data_dir / f"library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with self._lock:
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_patterns": len(self._patterns),
                "patterns": [asdict(p) for p in self._patterns.values()],
            }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Paul library serialization failed: %s", e)

        return filepath

    def deserialize_library(self, filepath: Path) -> int:
        """Load a serialized pattern library from JSON.

        Args:
            filepath: Path to library JSON file.

        Returns:
            Number of patterns loaded.
        """
        if not filepath.exists():
            logger.warning("Paul library file not found: %s", filepath)
            return 0

        loaded = 0
        try:
            with open(filepath) as f:
                data = json.load(f)

            for p_data in data.get("patterns", []):
                pattern = TradePattern(**{
                    k: v for k, v in p_data.items()
                    if k in TradePattern.__dataclass_fields__
                })
                if pattern.pattern_id:
                    with self._lock:
                        self._patterns[pattern.pattern_id] = pattern
                        if pattern.regime:
                            self._regime_index[pattern.regime].append(
                                pattern.pattern_id
                            )
                        if pattern.sector:
                            self._sector_index[pattern.sector].append(
                                pattern.pattern_id
                            )
                    loaded += 1
        except Exception as e:
            logger.error("Paul library deserialization failed: %s", e)

        return loaded

    def _log_event(self, event_type: str, data: dict):
        """Persist a learning event to JSONL log."""
        log_file = self.log_dir / f"paul_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("Paul log write failed: %s", e)


# ---------------------------------------------------------------------------
# Agent Learning Wrapper — attaches GSD + Paul to any agent
# ---------------------------------------------------------------------------
class AgentLearningWrapper:
    """Wraps any agent with GSD + Paul capabilities.

    Provides pre-decision and post-decision hooks that enrich agent
    decisions with gradient dynamics and pattern awareness, then learn
    from outcomes.

    Usage:
        gsd = GSDPlugin()
        paul = PaulPlugin()
        wrapper = AgentLearningWrapper(gsd, paul)

        # Attach to an agent (sector bot, research bot, persona, etc.)
        wrapper.attach_to_agent(my_agent)

        # Before agent makes a decision
        enrichment = wrapper.pre_decision_hook("agent_001", market_state)

        # After agent makes a decision and outcome is known
        wrapper.post_decision_hook("agent_001", outcome)

        # Get learning report
        summary = wrapper.get_learning_summary("agent_001")
    """

    def __init__(
        self,
        gsd: GSDPlugin,
        paul: PaulPlugin,
        learning_loop: Optional[Any] = None,
    ):
        self._gsd = gsd
        self._paul = paul
        self._learning_loop = learning_loop  # Optional LearningLoop integration
        self._lock = threading.RLock()

        # Track attached agents and their metadata
        self._attached_agents: dict[str, dict] = {}

        # Decision history per agent
        self._decision_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200)
        )

    def attach_to_agent(self, agent: Any) -> str:
        """Attach GSD + Paul learning to an agent.

        Works with any agent that has an `agent_id` or `name` attribute.

        Args:
            agent: Agent object (SectorBot, ResearchBot, InvestorPersona, etc.)

        Returns:
            Agent ID used for tracking.
        """
        # Resolve agent ID from common attributes
        agent_id = (
            getattr(agent, "agent_id", None)
            or getattr(agent, "name", None)
            or getattr(agent, "bot_name", None)
            or str(id(agent))
        )

        with self._lock:
            self._attached_agents[agent_id] = {
                "agent_ref": agent,
                "attached_at": datetime.now().isoformat(),
                "type": type(agent).__name__,
                "pre_decisions": 0,
                "post_decisions": 0,
            }

        # Initialize GSD profile for this agent
        self._gsd.update_agent_gsd_profile(agent_id, {
            "signal_engine": "initialization",
            "signal_value": 0.0,
            "was_correct": True,
            "realized_pnl": 0.0,
            "regime": "",
            "confidence": 0.5,
        })

        logger.info(
            "AgentLearningWrapper: attached to %s (type=%s)",
            agent_id, type(agent).__name__,
        )
        return agent_id

    def pre_decision_hook(
        self,
        agent_id: str,
        market_state: dict,
    ) -> dict:
        """Enrich an agent's decision context with GSD + Paul insights.

        Called BEFORE the agent makes a trade decision. Returns
        additional context the agent can use.

        Args:
            agent_id: Agent identifier.
            market_state: Current market state dict with keys:
                regime, volatility, vix_level, liquidity_score,
                consensus_score, vote_score, signals (dict of engine->value),
                ticker, sector, direction.

        Returns:
            Enrichment dict with gradient_confidence, pattern_matches,
            convergence/divergence events, and recommended adjustments.
        """
        enrichment: dict[str, Any] = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. GSD: gradient confidence for this agent
        gradient_confidence = self._gsd.get_gradient_confidence(agent_id)
        enrichment["gradient_confidence"] = gradient_confidence

        # 2. GSD: check signal convergence/divergence
        signals = market_state.get("signals", {})
        if signals:
            convergence = self._gsd.detect_convergence(signals)
            divergence = self._gsd.detect_divergence(signals)
            enrichment["convergence"] = asdict(convergence) if convergence else None
            enrichment["divergence"] = asdict(divergence) if divergence else None

            # Record signals for gradient tracking
            for eng, val in signals.items():
                self._gsd.record_signal(eng, val)

        # 3. GSD: per-engine gradient info
        engine_gradients = {}
        for eng in signals:
            engine_gradients[eng] = self._gsd.compute_signal_gradient(eng)
        enrichment["engine_gradients"] = engine_gradients

        # 4. GSD: cross-engine alignment
        alignment = self._gsd.get_cross_engine_alignment()
        enrichment["cross_engine_alignment"] = alignment.get("alignment_score", 0.0)

        # 5. Paul: pattern matching
        pattern_matches = self._paul.match_pattern(market_state, top_k=5)
        enrichment["pattern_matches"] = [
            {
                "pattern_id": m.pattern_id,
                "similarity": m.similarity_score,
                "context_score": m.context_score,
                "combined_score": m.combined_score,
                "direction": m.pattern.direction if m.pattern else None,
                "historical_pnl": m.pattern.realized_pnl if m.pattern else None,
                "success_rate": m.pattern.success_rate if m.pattern else None,
            }
            for m in pattern_matches
        ]

        # 6. Compute recommended confidence adjustment
        adjustment = self._compute_confidence_adjustment(
            gradient_confidence, pattern_matches, enrichment.get("convergence"),
            enrichment.get("divergence"),
        )
        enrichment["confidence_adjustment"] = adjustment

        # 7. Adaptive learning rate
        enrichment["learning_rate"] = self._gsd.compute_adaptive_learning_rate(agent_id)

        # Track decision
        with self._lock:
            if agent_id in self._attached_agents:
                self._attached_agents[agent_id]["pre_decisions"] += 1
            self._decision_history[agent_id].append({
                "type": "pre_decision",
                "timestamp": enrichment["timestamp"],
                "market_state_regime": market_state.get("regime", ""),
                "gradient_confidence": gradient_confidence,
                "n_pattern_matches": len(pattern_matches),
                "confidence_adjustment": adjustment,
            })

        return enrichment

    def post_decision_hook(
        self,
        agent_id: str,
        outcome: dict,
    ) -> dict:
        """Learn from a trade outcome, updating GSD profiles and Paul patterns.

        Called AFTER the agent's trade outcome is known.

        Args:
            agent_id: Agent identifier.
            outcome: Dict with keys: ticker, direction, entry_price,
                exit_price, realized_pnl, was_correct, signal_engine,
                signal_value, regime, confidence, volatility,
                liquidity_score, vix_level, consensus_score, vote_score,
                holding_period_days, signal_engines_active,
                signal_strengths, subsequent_5d_return, subsequent_20d_return.

        Returns:
            Dict with updated GSD profile, pattern storage result, and
            new learning rate.
        """
        result: dict[str, Any] = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Update GSD profile
        profile = self._gsd.update_agent_gsd_profile(agent_id, outcome)
        result["gsd_profile"] = {
            "gradient_weighted_confidence": profile.gradient_weighted_confidence,
            "learning_rate": profile.learning_rate,
            "total_updates": profile.total_updates,
            "cumulative_gradient_score": profile.cumulative_gradient_score,
        }

        # 2. If trade was successful, store as a Paul pattern
        was_successful = outcome.get("was_correct", False)
        realized_pnl = outcome.get("realized_pnl", 0.0)

        # Store pattern for any completed trade (successful or not) for learning
        pattern = TradePattern(
            ticker=outcome.get("ticker", ""),
            sector=outcome.get("sector", ""),
            direction=outcome.get("direction", ""),
            entry_price=outcome.get("entry_price", 0.0),
            exit_price=outcome.get("exit_price", 0.0),
            realized_pnl=realized_pnl,
            sharpe_contribution=outcome.get("sharpe_contribution", 0.0),
            holding_period_days=outcome.get("holding_period_days", 0),
            regime=outcome.get("regime", ""),
            volatility=outcome.get("volatility", 0.0),
            liquidity_score=outcome.get("liquidity_score", 0.0),
            vix_level=outcome.get("vix_level", 0.0),
            spread_bps=outcome.get("spread_bps", 0.0),
            consensus_score=outcome.get("consensus_score", 0.0),
            num_agents_agreed=outcome.get("num_agents_agreed", 0),
            vote_score=outcome.get("vote_score", 0.0),
            signal_engines_active=outcome.get("signal_engines_active", []),
            signal_strengths=outcome.get("signal_strengths", {}),
            was_successful=was_successful,
            subsequent_5d_return=outcome.get("subsequent_5d_return", 0.0),
            subsequent_20d_return=outcome.get("subsequent_20d_return", 0.0),
            success_rate=1.0 if was_successful else 0.0,
        )

        pattern_id = self._paul.store_pattern(pattern)
        result["pattern_id"] = pattern_id
        result["pattern_stored"] = True

        # 3. Forward to LearningLoop if connected
        if self._learning_loop is not None:
            try:
                from engine.monitoring.learning_loop import SignalOutcome
                signal_outcome = SignalOutcome(
                    ticker=outcome.get("ticker", ""),
                    signal_engine=outcome.get("signal_engine", ""),
                    signal_type=outcome.get("signal_type", ""),
                    signal_timestamp=outcome.get("signal_timestamp", ""),
                    execution_timestamp=datetime.now().isoformat(),
                    side=outcome.get("direction", ""),
                    entry_price=outcome.get("entry_price", 0.0),
                    exit_price=outcome.get("exit_price", 0.0),
                    realized_pnl=realized_pnl,
                    holding_period_days=outcome.get("holding_period_days", 0),
                    was_correct=was_successful,
                    vote_score=outcome.get("vote_score", 0.0),
                    confidence=outcome.get("confidence", 0.0),
                    regime_at_entry=outcome.get("regime", ""),
                )
                self._learning_loop.record_signal_outcome(signal_outcome)
                result["learning_loop_recorded"] = True
            except Exception as e:
                logger.debug("LearningLoop integration failed: %s", e)
                result["learning_loop_recorded"] = False

        # Track decision
        with self._lock:
            if agent_id in self._attached_agents:
                self._attached_agents[agent_id]["post_decisions"] += 1
            self._decision_history[agent_id].append({
                "type": "post_decision",
                "timestamp": result["timestamp"],
                "was_correct": was_successful,
                "realized_pnl": realized_pnl,
                "pattern_id": pattern_id,
                "new_confidence": profile.gradient_weighted_confidence,
            })

        return result

    def get_learning_summary(self, agent_id: str) -> dict:
        """Generate a comprehensive learning report for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dict with GSD metrics, Paul pattern stats, decision history,
            and performance trends.
        """
        summary: dict[str, Any] = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Agent metadata
        with self._lock:
            agent_meta = self._attached_agents.get(agent_id, {})
            summary["agent_type"] = agent_meta.get("type", "unknown")
            summary["attached_at"] = agent_meta.get("attached_at", "")
            summary["pre_decisions"] = agent_meta.get("pre_decisions", 0)
            summary["post_decisions"] = agent_meta.get("post_decisions", 0)

        # GSD metrics
        gsd_confidence = self._gsd.get_gradient_confidence(agent_id)
        gsd_lr = self._gsd.compute_adaptive_learning_rate(agent_id)
        summary["gsd"] = {
            "gradient_confidence": gsd_confidence,
            "learning_rate": gsd_lr,
        }

        with self._lock:
            profile = self._gsd._agent_profiles.get(agent_id)
            if profile:
                summary["gsd"]["total_updates"] = profile.total_updates
                summary["gsd"]["cumulative_gradient_score"] = profile.cumulative_gradient_score
                summary["gsd"]["cross_agent_alignment"] = profile.cross_agent_alignment
                summary["gsd"]["n_observations"] = len(profile.gradient_history)

                # Engine-level gradient breakdown
                engine_breakdown = {}
                for eng, grads in profile.engine_gradients.items():
                    if grads:
                        arr = np.array(grads, dtype=np.float64) if np is not None else grads
                        if np is not None:
                            engine_breakdown[eng] = {
                                "n_observations": len(grads),
                                "mean_gradient": float(np.mean(arr)),
                                "std_gradient": float(np.std(arr)),
                                "latest_gradient": float(arr[-1]),
                                "trend": "strengthening" if float(np.mean(arr[-5:])) > float(np.mean(arr)) else "weakening",
                            }
                        else:
                            engine_breakdown[eng] = {
                                "n_observations": len(grads),
                                "latest_gradient": grads[-1] if grads else 0.0,
                            }
                summary["gsd"]["engine_breakdown"] = engine_breakdown

        # Decision history analysis
        with self._lock:
            history = list(self._decision_history.get(agent_id, []))

        post_decisions = [d for d in history if d.get("type") == "post_decision"]
        if post_decisions:
            correct = sum(1 for d in post_decisions if d.get("was_correct", False))
            total_pnl = sum(d.get("realized_pnl", 0.0) for d in post_decisions)
            pnl_values = [d.get("realized_pnl", 0.0) for d in post_decisions]

            summary["performance"] = {
                "total_decisions": len(post_decisions),
                "accuracy": correct / len(post_decisions),
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(post_decisions),
                "hit_rate": sum(1 for p in pnl_values if p > 0) / len(pnl_values),
            }

            # Rolling performance (last 20 decisions)
            recent = post_decisions[-20:]
            if recent:
                recent_correct = sum(1 for d in recent if d.get("was_correct", False))
                recent_pnl = sum(d.get("realized_pnl", 0.0) for d in recent)
                summary["performance"]["rolling_accuracy"] = recent_correct / len(recent)
                summary["performance"]["rolling_pnl"] = recent_pnl

            # Confidence trend
            conf_values = [d.get("new_confidence", 0.5) for d in post_decisions
                           if "new_confidence" in d]
            if np is not None and len(conf_values) >= 2:
                arr = np.array(conf_values)
                summary["performance"]["confidence_trend"] = (
                    "improving" if float(np.mean(arr[-5:])) > float(np.mean(arr))
                    else "declining"
                )
        else:
            summary["performance"] = {"total_decisions": 0}

        return summary

    def _compute_confidence_adjustment(
        self,
        gradient_confidence: float,
        pattern_matches: list[PatternMatch],
        convergence: Optional[dict],
        divergence: Optional[dict],
    ) -> float:
        """Compute a confidence adjustment factor based on GSD + Paul signals.

        Returns adjustment in [-0.3, +0.3] to be added to base confidence.
        """
        adjustment = 0.0

        # Gradient confidence contribution: deviation from 0.5 neutral
        adjustment += (gradient_confidence - 0.5) * 0.2

        # Pattern match contribution: strong matches boost confidence
        if pattern_matches:
            top_match = pattern_matches[0]
            if top_match.combined_score > 0.8 and top_match.pattern:
                if top_match.pattern.was_successful:
                    adjustment += 0.1
                else:
                    adjustment -= 0.05

        # Convergence boosts confidence
        if convergence:
            alignment = convergence.get("alignment_score", 0.0)
            adjustment += alignment * 0.1

        # Divergence reduces confidence
        if divergence:
            div_score = divergence.get("divergence_score", 0.0)
            adjustment -= div_score * 0.15

        # Clamp to [-0.3, +0.3]
        return float(np.clip(adjustment, -0.3, 0.3))

    def get_all_agent_summaries(self) -> dict[str, dict]:
        """Get learning summaries for all attached agents."""
        with self._lock:
            agent_ids = list(self._attached_agents.keys())
        return {aid: self.get_learning_summary(aid) for aid in agent_ids}
