"""
Agent Performance Monitoring System.

4-tier hierarchy for evaluating and ranking agent performance.
Tracks accuracy, risk-adjusted returns, streaks, and memory usage.
Distinct from agent_scorecard.py — this module handles continuous
monitoring, tier promotion/demotion, and composite leaderboard scoring.
"""

import statistics
import math
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class AgentPerformanceTier(Enum):
    """4-tier performance hierarchy for monitored agents."""
    ELITE = "ELITE"
    STRONG = "STRONG"
    DEVELOPING = "DEVELOPING"
    UNDERPERFORM = "UNDERPERFORM"


# ---------------------------------------------------------------------------
# Performance record
# ---------------------------------------------------------------------------

@dataclass
class AgentPerformanceRecord:
    """Snapshot of an agent's evaluated performance metrics."""
    agent_id: str
    agent_name: str
    agent_type: str
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_signal: float = 0.0
    tier: AgentPerformanceTier = AgentPerformanceTier.UNDERPERFORM
    last_evaluated: str = ""
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Lightweight memory profiler backed by tracemalloc."""

    def __init__(self) -> None:
        self._active = False

    def start(self) -> None:
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._active = True
        except Exception:
            self._active = False

    def stop(self) -> None:
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception:
            pass
        self._active = False

    def get_snapshot(self) -> dict:
        if not self._active:
            return {"status": "inactive", "current_mb": 0.0, "peak_mb": 0.0}
        return {
            "status": "active",
            "current_mb": self.get_current_memory(),
            "peak_mb": self.get_peak_memory(),
            "top_allocations": self.get_top_allocations(),
        }

    def get_peak_memory(self) -> float:
        """Return peak memory usage in MB."""
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                _, peak = tracemalloc.get_traced_memory()
                return peak / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get_current_memory(self) -> float:
        """Return current memory usage in MB."""
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, _ = tracemalloc.get_traced_memory()
                return current / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get_top_allocations(self, n: int = 10) -> list[dict]:
        """Return the *n* largest allocation sites."""
        results: list[dict] = []
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                return results
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
            for stat in top_stats[:n]:
                results.append({
                    "file": str(stat.traceback),
                    "size_kb": stat.size / 1024,
                    "count": stat.count,
                })
        except Exception:
            pass
        return results


# ---------------------------------------------------------------------------
# Tier classification helper
# ---------------------------------------------------------------------------

def _classify_tier(sharpe: float, accuracy: float,
                   consecutive_wins: int) -> AgentPerformanceTier:
    """Determine tier from raw metrics."""
    if sharpe > 2.5 and accuracy > 85.0 and consecutive_wins > 10:
        return AgentPerformanceTier.ELITE
    if sharpe > 1.5 and accuracy > 70.0 and consecutive_wins > 5:
        return AgentPerformanceTier.STRONG
    if sharpe > 0.5 and accuracy > 55.0:
        return AgentPerformanceTier.DEVELOPING
    return AgentPerformanceTier.UNDERPERFORM


_TIER_RANK = {
    AgentPerformanceTier.ELITE: 3,
    AgentPerformanceTier.STRONG: 2,
    AgentPerformanceTier.DEVELOPING: 1,
    AgentPerformanceTier.UNDERPERFORM: 0,
}

_HISTORY_WINDOW = 50  # rolling window of recent signal outcomes


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

class AgentMonitor:
    """Continuous performance monitoring with tier promotion/demotion."""

    def __init__(self) -> None:
        self._records: dict[str, AgentPerformanceRecord] = {}
        self._signals: dict[str, list[dict]] = {}
        # Track how many consecutive eval periods an agent qualifies for a tier
        self._promotion_counter: dict[str, int] = {}
        self._demotion_counter: dict[str, int] = {}
        # Elite immunity: number of demotion checks to skip
        self._elite_immunity: dict[str, int] = {}
        self._memory_tracker = MemoryTracker()
        self._memory_tracker.start()

    # ------------------------------------------------------------------
    # Registration & signal recording
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, agent_name: str,
                       agent_type: str) -> None:
        """Register a new agent for monitoring."""
        if agent_id not in self._records:
            self._records[agent_id] = AgentPerformanceRecord(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type=agent_type,
            )
            self._signals[agent_id] = []
            self._promotion_counter[agent_id] = 0
            self._demotion_counter[agent_id] = 0
            self._elite_immunity[agent_id] = 0

    def record_signal(self, agent_id: str, ticker: str, signal: float,
                      actual_outcome: Optional[float] = None) -> None:
        """Record a signal (and optionally its realised outcome)."""
        if agent_id not in self._records:
            raise KeyError(f"Agent '{agent_id}' is not registered.")
        entry = {
            "ticker": ticker,
            "signal": signal,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._signals[agent_id].append(entry)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_agent(self, agent_id: str) -> AgentPerformanceRecord:
        """Compute all performance metrics for a single agent."""
        if agent_id not in self._records:
            raise KeyError(f"Agent '{agent_id}' is not registered.")

        rec = self._records[agent_id]
        signals = self._signals[agent_id]

        # Filter signals that have an actual outcome resolved
        resolved = [s for s in signals if s["actual_outcome"] is not None]

        rec.total_signals = len(resolved)
        if rec.total_signals == 0:
            rec.last_evaluated = datetime.now(timezone.utc).isoformat()
            return rec

        # --- accuracy ---
        correct = sum(
            1 for s in resolved
            if (s["signal"] > 0 and s["actual_outcome"] > 0)
            or (s["signal"] < 0 and s["actual_outcome"] < 0)
            or (s["signal"] == 0 and s["actual_outcome"] == 0)
        )
        rec.correct_signals = correct
        rec.accuracy = (correct / rec.total_signals) * 100.0

        # --- PnL series ---
        pnl_series = [s["actual_outcome"] for s in resolved]
        rec.total_pnl = sum(pnl_series)
        rec.avg_pnl_per_signal = rec.total_pnl / rec.total_signals

        # --- Sharpe ratio (annualised, assuming daily signals) ---
        if len(pnl_series) >= 2:
            mean_ret = statistics.mean(pnl_series)
            std_ret = statistics.stdev(pnl_series)
            rec.sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)
                                if std_ret > 0 else 0.0)
        else:
            rec.sharpe_ratio = 0.0

        # --- Sortino ratio ---
        downside = [r for r in pnl_series if r < 0]
        if len(downside) >= 2:
            downside_std = statistics.stdev(downside)
            mean_ret = statistics.mean(pnl_series)
            rec.sortino_ratio = (mean_ret / downside_std * math.sqrt(252)
                                 if downside_std > 0 else 0.0)
        elif len(downside) == 0:
            rec.sortino_ratio = rec.sharpe_ratio * 1.5  # no downside
        else:
            rec.sortino_ratio = 0.0

        # --- Max drawdown ---
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        rec.max_drawdown = max_dd

        # --- Consecutive wins / losses ---
        wins, losses = 0, 0
        max_wins, max_losses = 0, 0
        for pnl in pnl_series:
            if pnl > 0:
                wins += 1
                losses = 0
            elif pnl < 0:
                losses += 1
                wins = 0
            else:
                wins = 0
                losses = 0
            max_wins = max(max_wins, wins)
            max_losses = max(max_losses, losses)
        rec.consecutive_wins = max_wins
        rec.consecutive_losses = max_losses

        # --- Tier ---
        rec.tier = _classify_tier(
            rec.sharpe_ratio, rec.accuracy, rec.consecutive_wins
        )

        # --- History (rolling window) ---
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": rec.accuracy,
            "sharpe_ratio": rec.sharpe_ratio,
            "tier": rec.tier.value,
            "total_pnl": rec.total_pnl,
        }
        rec.history.append(snapshot)
        if len(rec.history) > _HISTORY_WINDOW:
            rec.history = rec.history[-_HISTORY_WINDOW:]

        rec.last_evaluated = datetime.now(timezone.utc).isoformat()
        return rec

    def evaluate_all(self) -> dict[str, AgentPerformanceRecord]:
        """Evaluate every registered agent and return the results."""
        return {aid: self.evaluate_agent(aid) for aid in self._records}

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_score(rec: AgentPerformanceRecord) -> float:
        """
        40% accuracy + 30% Sharpe + 20% profitability + 10% consistency_bonus.

        accuracy_norm:  correct_signals / total_signals (0-1)
        sharpe_norm:    Sharpe ratio normalised to [0,1] (cap at 3.0)
        profitability:  avg_pnl_per_signal normalised — measures whether
                        correct calls are actually profitable (dynamic metric
                        distinct from raw accuracy).
        consistency:    consecutive_wins / 20 (capped at 1.0)
        """
        accuracy_norm = min(rec.accuracy / 100.0, 1.0)
        sharpe_norm = min(max(rec.sharpe_ratio / 3.0, 0.0), 1.0)
        # Profitability: sigmoid-map avg_pnl_per_signal into [0, 1]
        # Positive avg PnL → >0.5, negative → <0.5
        avg_pnl = rec.avg_pnl_per_signal
        profitability = 1.0 / (1.0 + math.exp(-avg_pnl * 10.0)) if avg_pnl != 0 else 0.5
        consistency_bonus = min(rec.consecutive_wins / 20.0, 1.0)
        return (
            0.40 * accuracy_norm
            + 0.30 * sharpe_norm
            + 0.20 * profitability
            + 0.10 * consistency_bonus
        )

    # ------------------------------------------------------------------
    # Promotion / demotion
    # ------------------------------------------------------------------

    def promote_demote(self) -> list[dict]:
        """
        Check every agent for tier changes.

        Promotion:  meets higher-tier thresholds for >= 4 consecutive evals.
        Demotion:   below current-tier thresholds for >= 2 consecutive evals.
        ELITE agents receive 1 demotion-check immunity.

        Returns a list of change dicts:
            {"agent_id", "agent_name", "from_tier", "to_tier", "direction"}
        """
        changes: list[dict] = []
        for aid, rec in self._records.items():
            candidate_tier = _classify_tier(
                rec.sharpe_ratio, rec.accuracy, rec.consecutive_wins
            )
            candidate_rank = _TIER_RANK[candidate_tier]
            current_rank = _TIER_RANK[rec.tier]

            # --- promotion path ---
            if candidate_rank > current_rank:
                self._promotion_counter[aid] = (
                    self._promotion_counter.get(aid, 0) + 1
                )
                self._demotion_counter[aid] = 0
                if self._promotion_counter[aid] >= 4:
                    old_tier = rec.tier
                    rec.tier = candidate_tier
                    self._promotion_counter[aid] = 0
                    if rec.tier == AgentPerformanceTier.ELITE:
                        self._elite_immunity[aid] = 1
                    changes.append({
                        "agent_id": aid,
                        "agent_name": rec.agent_name,
                        "from_tier": old_tier.value,
                        "to_tier": rec.tier.value,
                        "direction": "promotion",
                    })

            # --- demotion path ---
            elif candidate_rank < current_rank:
                # Elite immunity check
                if rec.tier == AgentPerformanceTier.ELITE:
                    immunity = self._elite_immunity.get(aid, 0)
                    if immunity > 0:
                        self._elite_immunity[aid] = immunity - 1
                        continue

                self._demotion_counter[aid] = (
                    self._demotion_counter.get(aid, 0) + 1
                )
                self._promotion_counter[aid] = 0
                if self._demotion_counter[aid] >= 2:
                    old_tier = rec.tier
                    rec.tier = candidate_tier
                    self._demotion_counter[aid] = 0
                    changes.append({
                        "agent_id": aid,
                        "agent_name": rec.agent_name,
                        "from_tier": old_tier.value,
                        "to_tier": rec.tier.value,
                        "direction": "demotion",
                    })

            # --- stable ---
            else:
                self._promotion_counter[aid] = 0
                self._demotion_counter[aid] = 0

        return changes

    # ------------------------------------------------------------------
    # Leaderboard & distribution
    # ------------------------------------------------------------------

    def get_leaderboard(self) -> list[AgentPerformanceRecord]:
        """Return all agents sorted by composite score (descending)."""
        return sorted(
            self._records.values(),
            key=lambda r: self._composite_score(r),
            reverse=True,
        )

    def get_tier_distribution(self) -> dict:
        """Count of agents in each tier."""
        dist = {t.value: 0 for t in AgentPerformanceTier}
        for rec in self._records.values():
            dist[rec.tier.value] += 1
        return dist

    # ------------------------------------------------------------------
    # Memory report
    # ------------------------------------------------------------------

    def get_memory_report(self) -> dict:
        """Return a memory usage snapshot from the MemoryTracker."""
        return self._memory_tracker.get_snapshot()

    # ------------------------------------------------------------------
    # ASCII report
    # ------------------------------------------------------------------

    def format_monitor_report(self) -> str:
        """Produce a full ASCII report with leaderboard, tiers, and memory."""
        lines: list[str] = []
        sep = "=" * 72

        lines.append(sep)
        lines.append("  AGENT PERFORMANCE MONITOR REPORT")
        lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(sep)

        # --- Leaderboard ---
        lines.append("")
        lines.append("  LEADERBOARD")
        lines.append("  " + "-" * 68)
        lines.append(
            f"  {'Rank':<5} {'Agent':<20} {'Tier':<14} {'Accuracy':>8} "
            f"{'Sharpe':>7} {'PnL':>10} {'Score':>6}"
        )
        lines.append("  " + "-" * 68)

        leaderboard = self.get_leaderboard()
        for rank, rec in enumerate(leaderboard, 1):
            score = self._composite_score(rec)
            lines.append(
                f"  {rank:<5} {rec.agent_name:<20} {rec.tier.value:<14} "
                f"{rec.accuracy:>7.1f}% {rec.sharpe_ratio:>7.2f} "
                f"{rec.total_pnl:>10.2f} {score:>6.3f}"
            )

        if not leaderboard:
            lines.append("  (no agents registered)")

        # --- Tier distribution ---
        lines.append("")
        lines.append("  TIER DISTRIBUTION")
        lines.append("  " + "-" * 40)
        dist = self.get_tier_distribution()
        total = sum(dist.values()) or 1
        for tier_name, count in dist.items():
            bar = "#" * (count * 3)
            pct = count / total * 100
            lines.append(f"  {tier_name:<14} {count:>3}  ({pct:5.1f}%)  {bar}")

        # --- Memory ---
        lines.append("")
        lines.append("  MEMORY USAGE")
        lines.append("  " + "-" * 40)
        mem = self.get_memory_report()
        lines.append(f"  Status:   {mem.get('status', 'unknown')}")
        lines.append(f"  Current:  {mem.get('current_mb', 0.0):.2f} MB")
        lines.append(f"  Peak:     {mem.get('peak_mb', 0.0):.2f} MB")

        top_allocs = mem.get("top_allocations", [])
        if top_allocs:
            lines.append("  Top allocations:")
            for alloc in top_allocs[:5]:
                lines.append(
                    f"    {alloc.get('size_kb', 0):.1f} KB  "
                    f"({alloc.get('count', 0)} blocks)  "
                    f"{alloc.get('file', 'unknown')}"
                )

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)
