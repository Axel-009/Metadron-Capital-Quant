"""Metadron Capital — Full Universe Scan Orchestrator.

Timing Architecture:
- 4 universe runs x 150 seconds each = 600s (10 min) scanning
- 5 minutes aggregation + L7 execution
- Total full cycle: 15 minutes
- 5 minutes backtesting + risk management pass
- Total with risk: 20 minutes
- 3 full cycles/hour through market hours (09:30-16:00 ET)

Emits real-time SSE events for the Thinking Tab as signals are discovered.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Callable, List, Optional

logger = logging.getLogger("metadron.allocation.universe_scan")

# ─── Prometheus instrumentation ──────────────────────────────────
_prom = None

def _get_prom():
    global _prom
    if _prom is not None:
        return _prom
    try:
        from engine.bridges.prometheus_metrics import get_metrics
        _prom = get_metrics()
    except Exception:
        _prom = {}
    return _prom

from .allocation_engine import (
    AllocationEngine, AllocationRules, AllocationSlate,
    BucketType, CyclePhase, InstrumentType, ScanSignal,
)


# ═══════════════════════════════════════════════════════════════════════════
# Scan Configuration
# ═══════════════════════════════════════════════════════════════════════════

HEARTBEAT_SECONDS = 150       # 2.5 minutes per universe run
AGGREGATION_SECONDS = 300     # 5 minutes for aggregation + execution
RISK_PASS_SECONDS = 300       # 5 minutes for backtesting + risk
FULL_CYCLE_SECONDS = HEARTBEAT_SECONDS * 4 + AGGREGATION_SECONDS  # 15 min
TOTAL_CYCLE_SECONDS = FULL_CYCLE_SECONDS + RISK_PASS_SECONDS       # 20 min
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

UNIVERSE_ORDER = ["SP500", "SP400_MIDCAP", "SP600_SMALLCAP", "ETF_FI"]


# ═══════════════════════════════════════════════════════════════════════════
# S&P 1500 Universe (inlined from sp1500_universe.py)
# ═══════════════════════════════════════════════════════════════════════════

_universe_logger = logging.getLogger("metadron.allocation.universe")

# Import from existing cross-asset universe
try:
    from engine.data.cross_asset_universe import (
        SP500_TICKERS, SP400_TICKERS, SP600_TICKERS,
    )
except ImportError:
    try:
        from ..data.cross_asset_universe import (
            SP500_TICKERS, SP400_TICKERS, SP600_TICKERS,
        )
    except ImportError:
        SP500_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH"]
        SP400_TICKERS = ["DECK", "SAIA", "TOST", "FIX", "COOP", "LNTH", "RBC", "WFRD"]
        SP600_TICKERS = ["SPSC", "CALM", "SIG", "BOOT", "ARCH", "CARG", "PTGX"]
        _universe_logger.warning("cross_asset_universe not available — using minimal fallback tickers.")

ETF_TICKERS = [
    # Specified in spec
    "TLTW", "QQQ", "SPY", "IWM", "HYG", "LQD", "TLT", "PDBC", "GLD", "USO",
    # GICS Sector ETFs
    "XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLC", "XLU", "XLRE",
    # Additional broad / factor ETFs
    "DIA", "VTI", "VOO", "MDY", "IJR", "IEMG", "EFA", "VWO",
    # Dividend / Income ETFs
    "VIG", "SCHD", "DVY", "HDV", "JEPI", "JEPQ",
    # Commodity
    "SLV", "DBC", "COPX", "UNG",
]

FI_TICKERS = [
    # Specified in spec
    "TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "BKLN",
    # Additional FI
    "AGG", "BND", "VCIT", "VCSH", "MUB", "TIP", "GOVT", "MBB",
    "FLOT", "SCHO", "SCHR", "IGIB", "IGSB", "USIG",
    # TIPS / Inflation
    "STIP", "SCHP",
    # International FI
    "BNDX", "IAGG",
]

UNIVERSES = {
    "SP500": "S&P 500 Large Cap",
    "SP400_MIDCAP": "S&P 400 MidCap",
    "SP600_SMALLCAP": "S&P 600 SmallCap",
    "ETF_FI": "ETF + Fixed Income",
}


def get_universe(run_name: str) -> List[str]:
    """Return ticker list for a given universe run."""
    if run_name == "SP500":
        return list(SP500_TICKERS)
    elif run_name == "SP400_MIDCAP":
        return list(SP400_TICKERS)
    elif run_name == "SP600_SMALLCAP":
        return list(SP600_TICKERS)
    elif run_name == "ETF_FI":
        combined = list(dict.fromkeys(ETF_TICKERS + FI_TICKERS))
        return combined
    else:
        _universe_logger.warning("Unknown universe: %s — returning empty list.", run_name)
        return []


def get_all_universes() -> dict:
    """Return all universe compositions with counts."""
    return {
        name: {
            "description": desc,
            "ticker_count": len(get_universe(name)),
            "tickers": get_universe(name),
        }
        for name, desc in UNIVERSES.items()
    }


def get_universe_summary() -> dict:
    """Return summary counts for all universes."""
    total = 0
    summary = {}
    for name, desc in UNIVERSES.items():
        count = len(get_universe(name))
        summary[name] = {"description": desc, "count": count}
        total += count
    summary["total_unique"] = total
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Scan Status Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScanRunStatus:
    """Status of a single universe run."""
    universe: str = ""
    description: str = ""
    run_number: int = 0
    started_at: str = ""
    elapsed_seconds: float = 0.0
    heartbeat_total: int = HEARTBEAT_SECONDS
    signals_discovered: int = 0
    completed: bool = False
    favored_names: list = field(default_factory=list)
    trades: list = field(default_factory=list)


@dataclass
class ScanCycleStatus:
    """Status of a full scan cycle (4 runs + aggregation + risk)."""
    cycle_number: int = 0
    phase: str = CyclePhase.SCANNING.value
    current_run: int = 0
    current_universe: str = ""
    elapsed_seconds: float = 0.0
    total_signals: int = 0
    runs: list = field(default_factory=list)
    started_at: str = ""
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "phase": self.phase,
            "current_run": self.current_run,
            "current_universe": self.current_universe,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_signals": self.total_signals,
            "runs": [
                {
                    "universe": r.universe,
                    "description": r.description,
                    "run_number": r.run_number,
                    "elapsed_seconds": round(r.elapsed_seconds, 1),
                    "heartbeat_total": r.heartbeat_total,
                    "signals_discovered": r.signals_discovered,
                    "completed": r.completed,
                }
                for r in self.runs
            ],
            "started_at": self.started_at,
            "completed": self.completed,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Signal Event Queue (for SSE streaming to Thinking Tab)
# ═══════════════════════════════════════════════════════════════════════════

class SignalEventBus:
    """Thread-safe event bus for streaming scan signals to the frontend.

    The Thinking Tab SSE endpoint reads from this bus.
    """

    def __init__(self, max_events: int = 500):
        self._events: list[dict] = []
        self._max_events = max_events
        self._listeners: list[asyncio.Queue] = []

    def emit(self, event: dict):
        """Emit a signal event. Listeners (SSE streams) receive it."""
        event["_emitted_at"] = datetime.now(timezone.utc).isoformat()
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        # Push to all active listeners
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        # Prometheus: track signal generation
        if event.get("type") == "signal_discovered":
            try:
                prom = _get_prom()
                if prom and "signals_generated_total" in prom:
                    prom["signals_generated_total"].inc()
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscription queue for SSE streaming."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscription queue."""
        if q in self._listeners:
            self._listeners.remove(q)

    def recent_events(self, limit: int = 50) -> list[dict]:
        """Return the most recent events."""
        return self._events[-limit:]

    def clear(self):
        self._events.clear()


# Global event bus instance
signal_bus = SignalEventBus()


# ═══════════════════════════════════════════════════════════════════════════
# Full Universe Scan Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class FullUniverseScan:
    """Orchestrates the 4-universe scan cycle.

    Each cycle:
    1. Run 1-4: Scan each universe (150s heartbeat each)
    2. Aggregate results across all runs
    3. Feed to AllocationEngine for sizing
    4. Output to L7 execution surface
    5. Run backtesting + risk management pass

    Emits real-time events for each signal discovered → Thinking Tab SSE.
    """

    def __init__(self, allocation_engine: Optional[AllocationEngine] = None,
                 backtest: bool = False):
        self.engine = allocation_engine or AllocationEngine(backtest=backtest)
        self.backtest = backtest
        self.cycle_count: int = 0
        self.current_status = ScanCycleStatus()
        self._running = False
        self._last_slate: Optional[AllocationSlate] = None
        self._run_slates: list[AllocationSlate] = []

    async def run_universe(self, universe_name: str, run_number: int) -> tuple[ScanRunStatus, AllocationSlate]:
        """Execute a single 150s universe scan with MiroMomentum agent sim."""
        run = ScanRunStatus(
            universe=universe_name,
            description=UNIVERSES.get(universe_name, ""),
            run_number=run_number,
            started_at=datetime.now(timezone.utc).isoformat(),
            heartbeat_total=HEARTBEAT_SECONDS,
        )

        tickers = get_universe(universe_name)
        signals: list[ScanSignal] = []
        start = time.time()

        # Initialize MiroMomentumEngine for this run
        miro_engine = None
        try:
            from engine.signals.social_prediction_engine import MiroMomentumEngine
            miro_engine = MiroMomentumEngine(n_simulations=50, simulation_horizon=15)
        except Exception:
            pass

        # Emit scan start event
        signal_bus.emit({
            "type": "scan_start",
            "universe": universe_name,
            "run_number": run_number,
            "ticker_count": len(tickers),
            "miro_momentum_active": miro_engine is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        scan_interval = max(0.5, HEARTBEAT_SECONDS / max(len(tickers), 1))
        if self.backtest:
            scan_interval = 0.01

        for i, ticker in enumerate(tickers):
            elapsed = time.time() - start
            run.elapsed_seconds = elapsed

            self.current_status.elapsed_seconds = (
                (run_number - 1) * HEARTBEAT_SECONDS + elapsed
            )

            if i % max(1, len(tickers) // 15) == 0:
                instrument_types = [InstrumentType.EQUITY.value]
                if universe_name == "ETF_FI":
                    instrument_types = [InstrumentType.ETF.value, InstrumentType.FIXED_INCOME.value]

                # Use MiroMomentumEngine for real signals when available
                sig_type = "HOLD"
                confidence = 0.5
                alpha_score = 0.0
                regime_ctx = "RANGE"

                if miro_engine is not None:
                    try:
                        miro_sig = miro_engine.get_signal(ticker)
                        sig_type = miro_sig.miro_momentum_signal or "HOLD"
                        confidence = round(miro_sig.consensus_strength, 2) if miro_sig.consensus_strength > 0 else 0.5
                        alpha_score = round(miro_sig.momentum, 4)
                        regime_map = {"trending": "BULL", "mean_reverting": "RANGE", "random_walk": "TRANSITION"}
                        regime_ctx = regime_map.get(miro_sig.regime, "RANGE")
                    except Exception:
                        pass  # Fall through to defaults

                signal = ScanSignal(
                    ticker=ticker,
                    signal_type=sig_type,
                    instrument_type=instrument_types[0],
                    confidence=confidence,
                    alpha_score=alpha_score,
                    regime_context=regime_ctx,
                    universe=universe_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                signals.append(signal)
                run.signals_discovered = len(signals)

                signal_bus.emit({
                    "type": "signal_discovered",
                    "ticker": ticker,
                    "signal_type": signal.signal_type,
                    "instrument_type": signal.instrument_type,
                    "confidence": signal.confidence,
                    "alpha_score": signal.alpha_score,
                    "regime_context": signal.regime_context,
                    "bucket": self.engine.classify_opportunity(signal),
                    "universe": universe_name,
                    "run_number": run_number,
                    "timestamp": signal.timestamp,
                })

            if not self.backtest:
                await asyncio.sleep(min(scan_interval, 1.0))
            else:
                await asyncio.sleep(0)

            if not self.backtest and (time.time() - start) >= HEARTBEAT_SECONDS:
                break

        self.engine._apply_backtest_flag(self.backtest)
        slate = self.engine.apply_rules(signals, total_capital=self.engine.nav)
        run.completed = True
        run.elapsed_seconds = time.time() - start
        run.favored_names = [s.ticker for s in signals if s.alpha_score > 0.02]
        run.trades = [p.to_dict() for p in slate.positions[:10]]

        signal_bus.emit({
            "type": "run_complete",
            "universe": universe_name,
            "run_number": run_number,
            "signals_discovered": len(signals),
            "favored_names": run.favored_names[:20],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return run, slate

    async def run_full_cycle(self, cycle_number: int = 0) -> AllocationSlate:
        """Execute a full scan cycle: 4 runs (async parallel) → aggregate → execute → risk check.

        TIMING TELEMETRY: Tracks per-run duration + total cycle time.
        Warns if total cycle exceeds 5-minute Phase 3 cadence.
        """
        import asyncio as _asyncio
        cycle_start = time.time()

        self.cycle_count = cycle_number or self.cycle_count + 1
        self.current_status = ScanCycleStatus(
            cycle_number=self.cycle_count,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._running = True
        self._run_slates = []

        self.current_status.phase = CyclePhase.SCANNING.value

        # Run 4 universes in parallel via asyncio.gather
        async def _run_with_telemetry(i, universe):
            run_start = time.time()
            signal_bus.emit({
                "type": "phase_update",
                "phase": CyclePhase.SCANNING.value,
                "current_run": i + 1,
                "universe": universe,
                "cycle_number": self.cycle_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            run_status, slate = await self.run_universe(universe, i + 1)
            run_duration = time.time() - run_start
            logger.info("[FullUniverseScan] Run %d (%s) completed in %.1fs",
                        i + 1, universe, run_duration)
            return run_status, slate, run_duration

        try:
            results = await _asyncio.gather(*[
                _run_with_telemetry(i, universe)
                for i, universe in enumerate(UNIVERSE_ORDER)
            ])
            for i, (run_status, slate, run_duration) in enumerate(results):
                self.current_status.current_run = i + 1
                self.current_status.current_universe = UNIVERSE_ORDER[i]
                self.current_status.runs.append(run_status)
                self.current_status.total_signals += run_status.signals_discovered
                self._run_slates.append(slate)
        except Exception as exc:
            logger.warning("[FullUniverseScan] async gather failed, falling back to sequential: %s", exc)
            # Sequential fallback
            for i, universe in enumerate(UNIVERSE_ORDER):
                self.current_status.current_run = i + 1
                self.current_status.current_universe = universe
                run_status, slate = await self.run_universe(universe, i + 1)
            self.current_status.runs.append(run_status)
            self.current_status.total_signals += run_status.signals_discovered
            self._run_slates.append(slate)

        self.current_status.phase = CyclePhase.AGGREGATING.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.AGGREGATING.value,
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        aggregated = self.engine.aggregate_runs(*self._run_slates)
        aggregated.cycle_number = self.cycle_count
        aggregated.phase = CyclePhase.AGGREGATING.value

        if not self.backtest:
            await asyncio.sleep(min(5, AGGREGATION_SECONDS))

        self.current_status.phase = CyclePhase.EXECUTING.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.EXECUTING.value,
            "positions_count": len(aggregated.positions),
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        aggregated = self.engine.validate_against_kill_switch(
            aggregated, self.engine.nav if not aggregated.kill_switch_triggered else self.engine.nav
        )

        if not self.backtest:
            await asyncio.sleep(min(5, AGGREGATION_SECONDS))

        self.current_status.phase = CyclePhase.RISK_CHECK.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.RISK_CHECK.value,
            "kill_switch": aggregated.kill_switch_triggered,
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if not self.backtest:
            await asyncio.sleep(min(5, RISK_PASS_SECONDS))

        self.current_status.phase = CyclePhase.COOLDOWN.value
        self.current_status.completed = True
        aggregated.phase = CyclePhase.COOLDOWN.value

        signal_bus.emit({
            "type": "cycle_complete",
            "cycle_number": self.cycle_count,
            "total_signals": self.current_status.total_signals,
            "positions": len(aggregated.positions),
            "kill_switch": aggregated.kill_switch_triggered,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Prometheus: track scan cycle metrics
        try:
            prom = _get_prom()
            if prom:
                if "scan_run_number" in prom:
                    prom["scan_run_number"].set(self.cycle_count)
                if "universe_scan_progress_pct" in prom:
                    prom["universe_scan_progress_pct"].set(100.0)
                if "scan_cycle_duration_seconds" in prom:
                    prom["scan_cycle_duration_seconds"].observe(self.current_status.elapsed_seconds)
                if "active_universe_size" in prom:
                    total_tickers = sum(len(get_universe(u)) for u in UNIVERSE_ORDER)
                    prom["active_universe_size"].set(total_tickers)
        except Exception:
            pass

        # Cycle-level timing telemetry
        total_duration = time.time() - cycle_start
        CADENCE_SECONDS = 300  # Phase 3 cadence is 5 minutes
        if total_duration > CADENCE_SECONDS:
            logger.warning(
                "[FullUniverseScan] Cycle %d took %.1fs — EXCEEDS %ds cadence. "
                "Consider progressive subset rotation.",
                self.cycle_count, total_duration, CADENCE_SECONDS,
            )
        else:
            logger.info(
                "[FullUniverseScan] Cycle %d completed in %.1fs (within %ds cadence)",
                self.cycle_count, total_duration, CADENCE_SECONDS,
            )
        try:
            aggregated.cycle_duration_seconds = round(total_duration, 2)
        except Exception:
            pass  # AllocationSlate may not have this field — non-critical

        self._last_slate = aggregated
        self._running = False
        return aggregated

    async def run_continuous(self, max_cycles: int = 0):
        """Run continuous scan cycles during market hours."""
        cycle = 0
        while True:
            cycle += 1
            if max_cycles > 0 and cycle > max_cycles:
                break

            await self.run_full_cycle(cycle)

            if not self.backtest:
                await asyncio.sleep(5)

    def get_scan_status(self) -> dict:
        """Return current scan cycle status."""
        return self.current_status.to_dict()

    def get_last_slate(self) -> Optional[dict]:
        """Return last computed allocation slate."""
        if self._last_slate:
            return self._last_slate.to_dict()
        return None

    @property
    def is_running(self) -> bool:
        return self._running
