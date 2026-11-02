"""LearningLoop — Closed-loop feedback across all engines.

Captures execution outcomes and feeds them back into signal engines,
ML models, agent scoring, and portfolio construction. Every trade
generates a learning signal that propagates through the full stack.

Feedback flow:
    Execution outcome (P&L, fill quality, slippage)
        → Signal accuracy tracking (per engine)
        → ML model retraining signals (feature importance, decay)
        → Agent scorecard updates (promotion/demotion)
        → Regime model calibration (HMM transition priors)
        → Alpha optimizer weight adjustment (walk-forward)
        → Portfolio risk parameter tuning (beta corridor, VaR)

Learning channels:
    1. SIGNAL_ACCURACY   — Was the signal correct? (per engine, per ticker)
    2. EXECUTION_QUALITY — Fill price vs expected, slippage, market impact
    3. REGIME_FEEDBACK   — Did the regime call match realized market behavior?
    4. ALPHA_DECAY       — How quickly did alpha signals decay after generation?
    5. RISK_CALIBRATION  — Did risk limits trigger appropriately?
    6. AGENT_PERFORMANCE — Per-agent accuracy, Sharpe, hit rate updates
    7. CROSS_ASSET_FEEDBACK — Did macro signals correctly predict sector rotation?
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("metadron.learning.loop")


# ---------------------------------------------------------------------------
# Learning record types
# ---------------------------------------------------------------------------
@dataclass
class SignalOutcome:
    """Outcome of a single signal → execution → P&L cycle."""
    ticker: str = ""
    signal_engine: str = ""         # e.g. "macro", "social", "distress", "event"
    signal_type: str = ""           # e.g. "ML_AGENT_BUY", "EVENT_MERGER_ARB"
    signal_timestamp: str = ""
    execution_timestamp: str = ""
    side: str = ""                  # BUY/SELL/SHORT/COVER
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    holding_period_days: int = 0
    was_correct: bool = False       # signal direction matched outcome
    vote_score: float = 0.0         # ML ensemble vote at entry
    confidence: float = 0.0         # ensemble confidence at entry
    regime_at_entry: str = ""       # TRENDING/RANGE/STRESS/CRASH
    alpha_pred_at_entry: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0


@dataclass
class EngineAccuracy:
    """Rolling accuracy tracker for a single engine."""
    engine_name: str = ""
    total_signals: int = 0
    correct_signals: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_signal: float = 0.0
    accuracy: float = 0.0          # correct / total
    sharpe: float = 0.0
    hit_rate: float = 0.0
    avg_holding_period: float = 0.0
    last_updated: str = ""

    # Rolling window (last N signals)
    rolling_accuracy: float = 0.0
    rolling_pnl: float = 0.0


@dataclass
class RegimeFeedback:
    """Feedback on regime classification accuracy."""
    predicted_regime: str = ""
    actual_market_behavior: str = ""  # computed from realized returns
    regime_correct: bool = False
    realized_return_1d: float = 0.0
    realized_return_5d: float = 0.0
    realized_vol_5d: float = 0.0
    timestamp: str = ""


@dataclass
class LearningSnapshot:
    """Complete learning state at a point in time."""
    timestamp: str = ""
    engine_accuracies: dict = field(default_factory=dict)
    regime_accuracy: float = 0.0
    alpha_decay_rate: float = 0.0
    best_engines: list = field(default_factory=list)
    worst_engines: list = field(default_factory=list)
    suggested_weight_adjustments: dict = field(default_factory=dict)
    risk_calibration_score: float = 0.0
    total_learning_events: int = 0


# ---------------------------------------------------------------------------
# Learning Loop Engine
# ---------------------------------------------------------------------------
class LearningLoop:
    """Closed-loop feedback engine for the Metadron Capital platform.

    Captures every execution outcome and feeds it back into:
    - Signal engine accuracy tracking
    - ML model weight adjustments
    - Agent scorecard updates
    - Regime calibration
    - Alpha optimizer walk-forward tuning
    - Risk parameter recalibration
    """

    # Engine names that generate tradeable signals
    SIGNAL_ENGINES = [
        "macro", "cube", "security_analysis", "pattern_discovery",
        "social", "distress", "cvr", "event_driven",
        "alpha_optimizer", "decision_matrix", "hft_technical",
        "ml_ensemble",
    ]

    # Default ML ensemble tier weights (can be adjusted by learning)
    DEFAULT_TIER_WEIGHTS = {
        "T1_neural": 1.0,
        "T2_momentum": 1.2,
        "T3_vol_regime": 0.8,
        "T4_monte_carlo": 0.9,
        "T5_quality": 1.1,
        "T6_social": 1.0,
        "T7_distress": 0.9,
        "T8_event": 1.0,
        "T9_cvr": 0.7,
        "T10_credit_quality": 0.9,
    }

    ROLLING_WINDOW = 100  # Last N signals for rolling metrics

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/learning_loop")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Per-engine outcome tracking
        self._outcomes: dict[str, deque] = {
            eng: deque(maxlen=500) for eng in self.SIGNAL_ENGINES
        }
        self._outcomes["unknown"] = deque(maxlen=500)

        # Aggregate engine accuracy
        self._engine_stats: dict[str, EngineAccuracy] = {
            eng: EngineAccuracy(engine_name=eng) for eng in self.SIGNAL_ENGINES
        }

        # Regime feedback
        self._regime_history: deque = deque(maxlen=500)

        # Alpha decay tracking
        self._alpha_entries: deque = deque(maxlen=500)

        # Learned tier weight adjustments
        self._tier_weights: dict[str, float] = dict(self.DEFAULT_TIER_WEIGHTS)
        self._weights_file = self.log_dir / "tier_weights.json"
        self._load_tier_weights()  # Restore from last session

        # Risk calibration events
        self._risk_events: deque = deque(maxlen=200)

        # Cross-asset feedback (macro signal → sector outcome)
        self._sector_feedback: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Session counter
        self._total_events = 0
        self._retrain_check_interval = 10  # Check retraining every 10 outcomes

        # Reference to alpha optimizer (set externally)
        self._alpha_optimizer = None
        self._alpha_retrain_needed = False

    # --- Tier weight persistence -------------------------------------------

    def _save_tier_weights(self):
        """Persist learned tier weights to disk."""
        try:
            data = {
                "weights": self._tier_weights,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events": self._total_events,
            }
            tmp = self._weights_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2))
            tmp.rename(self._weights_file)
        except Exception as e:
            logger.warning("Failed to save tier weights: %s", e)

    def _load_tier_weights(self):
        """Load learned tier weights from previous session."""
        if not self._weights_file.exists():
            return
        try:
            data = json.loads(self._weights_file.read_text())
            saved_weights = data.get("weights", {})
            # Validate: all values should be positive floats
            for tier, weight in saved_weights.items():
                if isinstance(weight, (int, float)) and weight > 0:
                    self._tier_weights[tier] = float(weight)
            saved_at = data.get("timestamp", "unknown")
            events = data.get("total_events", 0)
            logger.info("Loaded tier weights from %s (%d events, %d tiers)",
                        saved_at, events, len(saved_weights))
        except Exception as e:
            logger.warning("Failed to load tier weights: %s — using defaults", e)

    # --- Record outcomes ----------------------------------------------------

    def _check_alpha_retrain(self):
        """Check if alpha optimizer needs retraining and trigger if needed."""
        if self._alpha_optimizer is None:
            return
        try:
            if hasattr(self._alpha_optimizer, 'should_retrain') and self._alpha_optimizer.should_retrain():
                logger.info("Alpha optimizer retraining threshold reached — triggering retrain")
                # Retraining requires X, y data which comes from the pipeline
                # The actual retrain is triggered in the pipeline's alpha stage
                # Here we just set a flag
                self._alpha_retrain_needed = True
        except Exception as e:
            logger.debug("Alpha retrain check failed: %s", e)

    def record_signal_outcome(self, outcome: SignalOutcome):
        """Record the outcome of a signal → execution cycle.

        This is the primary learning input. Called after a position is
        closed or at EOD for mark-to-market.
        """
        engine = outcome.signal_engine or "unknown"
        if engine not in self._outcomes:
            self._outcomes[engine] = deque(maxlen=500)
            self._engine_stats[engine] = EngineAccuracy(engine_name=engine)

        self._outcomes[engine].append(outcome)
        self._total_events += 1

        # Trigger alpha optimizer retraining if threshold reached
        if self._total_events % self._retrain_check_interval == 0:
            self._check_alpha_retrain()

        # Update engine stats
        stats = self._engine_stats[engine]
        stats.total_signals += 1
        stats.total_pnl += outcome.realized_pnl
        if outcome.was_correct:
            stats.correct_signals += 1
        stats.accuracy = stats.correct_signals / max(stats.total_signals, 1)
        stats.avg_pnl_per_signal = stats.total_pnl / max(stats.total_signals, 1)

        # Rolling accuracy (last N)
        recent = list(self._outcomes[engine])[-self.ROLLING_WINDOW:]
        if recent:
            stats.rolling_accuracy = sum(1 for o in recent if o.was_correct) / len(recent)
            stats.rolling_pnl = sum(o.realized_pnl for o in recent)

        # Hit rate (positive P&L trades)
        profitable = [o for o in self._outcomes[engine] if o.realized_pnl > 0]
        stats.hit_rate = len(profitable) / max(len(self._outcomes[engine]), 1)

        # Average holding period
        holding_periods = [o.holding_period_days for o in self._outcomes[engine]
                          if o.holding_period_days > 0]
        stats.avg_holding_period = (
            sum(holding_periods) / len(holding_periods) if holding_periods else 0
        )

        stats.last_updated = datetime.now().isoformat()

        # Persist
        self._persist_outcome(outcome)

        logger.debug(
            "Learning: %s signal for %s — P&L=$%.2f correct=%s (engine accuracy=%.1f%%)",
            engine, outcome.ticker, outcome.realized_pnl,
            outcome.was_correct, stats.accuracy * 100,
        )

    def record_regime_feedback(self, feedback: RegimeFeedback):
        """Record whether regime classification was correct."""
        self._regime_history.append(feedback)

    def record_risk_event(self, event: dict):
        """Record a risk management event (trigger, near-miss, etc.)."""
        event["timestamp"] = datetime.now().isoformat()
        self._risk_events.append(event)

    def record_sector_feedback(self, sector: str, predicted_direction: str,
                                realized_return: float):
        """Record whether macro-driven sector allocation was correct."""
        correct = (
            (predicted_direction == "OVERWEIGHT" and realized_return > 0) or
            (predicted_direction == "UNDERWEIGHT" and realized_return < 0)
        )
        self._sector_feedback[sector].append({
            "predicted": predicted_direction,
            "realized_return": realized_return,
            "correct": correct,
            "timestamp": datetime.now().isoformat(),
        })

    # --- Compute learning signals -------------------------------------------

    def compute_tier_weight_adjustments(self) -> dict[str, float]:
        """Compute adjusted ML ensemble tier weights based on signal outcomes.

        Engines with higher rolling accuracy get weight boosts.
        Engines with lower accuracy get dampened.

        Returns new tier weights for MLVoteEnsemble.
        """
        adjustments = dict(self.DEFAULT_TIER_WEIGHTS)

        # Map engines to tiers
        engine_tier_map = {
            "ml_ensemble": "T1_neural",
            "alpha_optimizer": "T2_momentum",
            "macro": "T3_vol_regime",
            "cube": "T4_monte_carlo",
            "security_analysis": "T5_quality",
            "social": "T6_social",
            "distress": "T7_distress",
            "event_driven": "T8_event",
            "cvr": "T9_cvr",
            "decision_matrix": "T10_credit_quality",
        }

        for engine, tier in engine_tier_map.items():
            stats = self._engine_stats.get(engine)
            if stats and stats.total_signals >= 10:
                # Adjust weight based on rolling accuracy relative to 50% baseline
                accuracy_delta = stats.rolling_accuracy - 0.50
                # +/- 30% max adjustment
                weight_adj = max(-0.3, min(0.3, accuracy_delta))
                base = self.DEFAULT_TIER_WEIGHTS.get(tier, 1.0)
                adjustments[tier] = base * (1.0 + weight_adj)

        self._tier_weights = adjustments
        self._save_tier_weights()  # Persist learned weights
        return adjustments

    def get_best_engines(self, n: int = 3) -> list[tuple[str, float]]:
        """Return top N engines by rolling accuracy."""
        engines = [
            (name, stats.rolling_accuracy)
            for name, stats in self._engine_stats.items()
            if stats.total_signals >= 5
        ]
        return sorted(engines, key=lambda x: x[1], reverse=True)[:n]

    def get_worst_engines(self, n: int = 3) -> list[tuple[str, float]]:
        """Return bottom N engines by rolling accuracy."""
        engines = [
            (name, stats.rolling_accuracy)
            for name, stats in self._engine_stats.items()
            if stats.total_signals >= 5
        ]
        return sorted(engines, key=lambda x: x[1])[:n]

    def compute_regime_accuracy(self) -> float:
        """Compute rolling accuracy of regime classification."""
        if not self._regime_history:
            return 0.0
        recent = list(self._regime_history)[-self.ROLLING_WINDOW:]
        correct = sum(1 for r in recent if r.regime_correct)
        return correct / len(recent)

    def get_regime_calibration_bias(self) -> dict:
        """
        Check regime accuracy and return calibration recommendation.
        
        If accuracy < 60% over 20+ samples, recommend conservative bias.
        This feeds into MetadronCube to reduce risk when regime detection
        is unreliable.
        """
        accuracy = self.compute_regime_accuracy()
        n_samples = len(self._regime_history)
        
        result = {
            "accuracy": accuracy,
            "n_samples": n_samples,
            "bias": "NORMAL",
            "recommendation": None,
        }
        
        if n_samples < 20:
            return result  # Not enough data
        
        if accuracy < 0.50:
            result["bias"] = "DEFENSIVE"
            result["recommendation"] = (
                f"Regime accuracy {accuracy:.1%} (<50%) — bias Cube toward "
                "DEFENSIVE regime. Reduce leverage, prioritize capital preservation."
            )
            logger.warning("🚨 REGIME CALIBRATION: accuracy %.1%% — biasing DEFENSIVE", accuracy * 100)
        elif accuracy < 0.60:
            result["bias"] = "CONSERVATIVE"
            result["recommendation"] = (
                f"Regime accuracy {accuracy:.1%} (<60%) — bias Cube toward "
                "CONSERVATIVE. Reduce position sizes, widen stop losses."
            )
            logger.warning("REGIME CALIBRATION: accuracy %.1%% — biasing CONSERVATIVE", accuracy * 100)
        
        return result

    def compute_alpha_decay_rate(self) -> float:
        """Estimate how quickly alpha signals decay after generation.

        Returns decay half-life in trading days.
        """
        if len(self._alpha_entries) < 10:
            return 0.0
        # Simple estimate: correlation between holding period and P&L
        periods = np.array([e.holding_period_days for e in self._alpha_entries
                           if e.holding_period_days > 0])
        pnls = np.array([e.realized_pnl for e in self._alpha_entries
                         if e.holding_period_days > 0])
        if len(periods) < 10 or periods.std() == 0:
            return 0.0
        corr = np.corrcoef(periods, pnls)[0, 1]
        # Negative correlation means alpha decays with time
        return float(-corr * 10) if corr < 0 else 0.0

    def compute_risk_calibration_score(self) -> float:
        """Score how well risk limits are calibrated.

        1.0 = perfectly calibrated (limits trigger at right levels)
        0.0 = poorly calibrated (limits too tight or too loose)
        """
        if not self._risk_events:
            return 0.5  # neutral
        recent = list(self._risk_events)[-50:]
        # Count near-misses vs hard stops
        near_misses = sum(1 for e in recent if e.get("type") == "near_miss")
        hard_stops = sum(1 for e in recent if e.get("type") == "hard_stop")
        # Good calibration: ~80% near misses, ~20% hard stops
        total = near_misses + hard_stops
        if total == 0:
            return 0.5
        near_miss_pct = near_misses / total
        return float(1.0 - abs(near_miss_pct - 0.80))

    def get_sector_allocation_accuracy(self) -> dict[str, float]:
        """Return accuracy of macro-driven sector allocation per sector."""
        result = {}
        for sector, history in self._sector_feedback.items():
            if not history:
                continue
            recent = list(history)[-50:]
            correct = sum(1 for e in recent if e.get("correct", False))
            result[sector] = correct / len(recent)
        return result

    # --- Generate learning snapshot -----------------------------------------

    def get_snapshot(self) -> LearningSnapshot:
        """Generate a complete learning snapshot for dashboard/reporting."""
        tier_adj = self.compute_tier_weight_adjustments()
        return LearningSnapshot(
            timestamp=datetime.now().isoformat(),
            engine_accuracies={
                name: {
                    "accuracy": stats.accuracy,
                    "rolling_accuracy": stats.rolling_accuracy,
                    "total_signals": stats.total_signals,
                    "total_pnl": stats.total_pnl,
                    "hit_rate": stats.hit_rate,
                    "avg_pnl": stats.avg_pnl_per_signal,
                }
                for name, stats in self._engine_stats.items()
                if stats.total_signals > 0
            },
            regime_accuracy=self.compute_regime_accuracy(),
            alpha_decay_rate=self.compute_alpha_decay_rate(),
            best_engines=[
                {"engine": e, "accuracy": a}
                for e, a in self.get_best_engines()
            ],
            worst_engines=[
                {"engine": e, "accuracy": a}
                for e, a in self.get_worst_engines()
            ],
            suggested_weight_adjustments=tier_adj,
            risk_calibration_score=self.compute_risk_calibration_score(),
            total_learning_events=self._total_events,
        )

    # --- Apply learning to engines ------------------------------------------

    # Damping parameters (prevent feedback loop oscillation)
    MAX_WEIGHT_CHANGE_PER_CYCLE = 0.05       # Max 5% change per cycle
    OSCILLATION_HISTORY_LEN = 10             # Track last 10 cycles
    OSCILLATION_REVERSAL_THRESHOLD = 3       # 3+ reversals = freeze weight

    def apply_to_ensemble(self, ensemble) -> dict:
        """Apply learned tier weights to MLVoteEnsemble with damping + oscillation detection.

        Damping: max 5% change per cycle to prevent runaway adjustments.
        Oscillation detection: if a weight reverses direction 3+ times in
        10 cycles, freeze it and alert (indicates unstable dynamics).

        Args:
            ensemble: MLVoteEnsemble instance

        Returns:
            Dict of old → new weight changes + frozen tiers.
        """
        if not hasattr(self, "_weight_history"):
            self._weight_history = {}  # {tier: [delta_1, delta_2, ...]}
            self._frozen_tiers = set()

        new_weights = self.compute_tier_weight_adjustments()
        changes = {}
        frozen_now = []

        for tier, target_w in new_weights.items():
            # Skip frozen tiers (oscillation detected)
            if tier in self._frozen_tiers:
                continue

            old_w = ensemble.TIER_WEIGHTS.get(tier, 1.0)
            raw_delta = target_w - old_w

            # Apply damping: cap change at MAX_WEIGHT_CHANGE_PER_CYCLE
            if abs(raw_delta) > self.MAX_WEIGHT_CHANGE_PER_CYCLE:
                capped_delta = self.MAX_WEIGHT_CHANGE_PER_CYCLE * (1 if raw_delta > 0 else -1)
                new_w = old_w + capped_delta
            else:
                new_w = target_w

            # Track in history for oscillation detection
            hist = self._weight_history.setdefault(tier, [])
            delta = new_w - old_w
            hist.append(delta)
            if len(hist) > self.OSCILLATION_HISTORY_LEN:
                hist.pop(0)

            # Count direction reversals in recent history
            reversals = 0
            for i in range(1, len(hist)):
                if hist[i] * hist[i-1] < 0:  # sign flip
                    reversals += 1

            # Freeze if oscillating
            if reversals >= self.OSCILLATION_REVERSAL_THRESHOLD:
                self._frozen_tiers.add(tier)
                frozen_now.append(tier)
                logger.warning(
                    "OSCILLATION DETECTED: tier %s frozen after %d reversals in %d cycles",
                    tier, reversals, len(hist),
                )
                continue

            # Apply change if meaningful
            if abs(new_w - old_w) > 0.01:
                changes[tier] = {"old": old_w, "new": new_w, "capped": abs(raw_delta) > self.MAX_WEIGHT_CHANGE_PER_CYCLE}
                ensemble.TIER_WEIGHTS[tier] = new_w

        if changes:
            logger.info(
                "Learning loop adjusted %d tier weights (damped): %s",
                len(changes),
                {k: f"{v['old']:.2f}→{v['new']:.2f}{' [capped]' if v.get('capped') else ''}"
                 for k, v in changes.items()},
            )
        if frozen_now:
            logger.critical("Tiers frozen due to oscillation: %s", frozen_now)

        return {"changes": changes, "frozen": list(self._frozen_tiers)}

    def get_engine_stats(self, engine: str) -> Optional[EngineAccuracy]:
        """Get accuracy stats for a specific engine."""
        return self._engine_stats.get(engine)

    def get_all_engine_stats(self) -> dict[str, EngineAccuracy]:
        """Get accuracy stats for all engines with data."""
        return {
            name: stats for name, stats in self._engine_stats.items()
            if stats.total_signals > 0
        }

    # --- Persistence --------------------------------------------------------

    def _persist_outcome(self, outcome: SignalOutcome):
        """Persist signal outcome to disk."""
        log_file = self.log_dir / f"outcomes_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(asdict(outcome)) + "\n")
        except Exception:
            pass

    def persist_snapshot(self):
        """Persist full learning snapshot to disk."""
        snapshot = self.get_snapshot()
        snap_file = self.log_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(snap_file, "w") as f:
                json.dump(asdict(snapshot), f, indent=2, default=str)
        except Exception:
            pass

    def load_outcomes(self, date_str: Optional[str] = None) -> int:
        """Load historical outcomes from disk for continuing learning."""
        date_str = date_str or datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"outcomes_{date_str}.jsonl"
        count = 0
        if log_file.exists():
            try:
                with open(log_file) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        outcome = SignalOutcome(**data)
                        engine = outcome.signal_engine or "unknown"
                        if engine not in self._outcomes:
                            self._outcomes[engine] = deque(maxlen=500)
                        self._outcomes[engine].append(outcome)
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to load learning outcomes: {e}")
        return count

    # --- Report generation --------------------------------------------------

    def format_learning_report(self) -> str:
        """Generate ASCII learning report."""
        snapshot = self.get_snapshot()
        lines = [
            "=" * 70,
            "LEARNING LOOP — PERFORMANCE FEEDBACK REPORT",
            f"Timestamp: {snapshot.timestamp}",
            f"Total Learning Events: {snapshot.total_learning_events}",
            "=" * 70,
        ]

        # Engine accuracies
        lines.append("\nENGINE ACCURACY RANKINGS:")
        lines.append(f"  {'Engine':<25} {'Accuracy':>8} {'Rolling':>8} {'Signals':>8} {'P&L':>12} {'Hit%':>6}")
        lines.append("  " + "-" * 69)
        sorted_engines = sorted(
            snapshot.engine_accuracies.items(),
            key=lambda x: x[1].get("rolling_accuracy", 0),
            reverse=True,
        )
        for name, data in sorted_engines:
            lines.append(
                f"  {name:<25} {data['accuracy']:>7.1%} {data['rolling_accuracy']:>7.1%} "
                f"{data['total_signals']:>8} ${data['total_pnl']:>11,.0f} {data['hit_rate']:>5.1%}"
            )

        # Regime accuracy
        lines.append(f"\nREGIME ACCURACY: {snapshot.regime_accuracy:.1%}")
        lines.append(f"ALPHA DECAY RATE: {snapshot.alpha_decay_rate:.2f} (half-life in days)")
        lines.append(f"RISK CALIBRATION: {snapshot.risk_calibration_score:.1%}")

        # Weight adjustments
        if snapshot.suggested_weight_adjustments:
            lines.append("\nSUGGESTED TIER WEIGHT ADJUSTMENTS:")
            for tier, weight in sorted(snapshot.suggested_weight_adjustments.items()):
                default = self.DEFAULT_TIER_WEIGHTS.get(tier, 1.0)
                delta = weight - default
                if abs(delta) > 0.01:
                    direction = "↑" if delta > 0 else "↓"
                    lines.append(f"  {tier:<25} {default:.2f} → {weight:.2f} {direction}")

        # Sector allocation feedback
        sector_acc = self.get_sector_allocation_accuracy()
        if sector_acc:
            lines.append("\nSECTOR ALLOCATION ACCURACY (macro → GICS):")
            for sector, acc in sorted(sector_acc.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {sector:<35} {acc:.1%}")

        lines.append("=" * 70)
        return "\n".join(lines)
