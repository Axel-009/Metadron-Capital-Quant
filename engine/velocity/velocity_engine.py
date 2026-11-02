"""Metadron Capital — Velocity Engine.

Computes real-time velocity metrics across the trading platform:
- Order flow velocity (trades/sec, volume/sec)
- Signal velocity (signals/sec across universe scan)
- Execution velocity (fill rate, latency)
- Capital deployment velocity ($/min deployed)
- Momentum score (rolling 5-min price momentum)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Optional, Dict, List, Any

logger = logging.getLogger("metadron.velocity")


@dataclass
class VelocitySnapshot:
    """Complete velocity state at a point in time."""
    # Order flow
    trades_per_second: float = 0.0
    volume_per_second: float = 0.0
    total_trades_session: int = 0
    total_volume_session: float = 0.0

    # Signal velocity
    signals_per_second: float = 0.0
    signals_per_cycle: int = 0
    active_signals: int = 0
    scan_cycles_completed: int = 0

    # Execution velocity
    fill_rate_pct: float = 0.0
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    orders_in_flight: int = 0

    # Capital deployment
    capital_deployed_per_min: float = 0.0
    total_capital_deployed: float = 0.0
    deployment_rate_trend: str = "STABLE"  # ACCELERATING / STABLE / DECELERATING
    nav: float = 0.0

    # Momentum
    momentum_score: float = 0.0
    momentum_breadth: float = 0.0  # % of positions with positive momentum
    momentum_leaders: list = field(default_factory=list)
    momentum_laggards: list = field(default_factory=list)

    # Meta
    timestamp: str = ""
    uptime_seconds: float = 0.0
    engine_status: str = "ONLINE"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = d["timestamp"] or datetime.now(timezone.utc).isoformat()
        return d


class VelocityEngine:
    """Real-time velocity metrics engine for Metadron Capital."""

    def __init__(self):
        self._start_time = time.time()
        self._trade_history_cache: List[dict] = []
        self._last_trade_count = 0
        self._last_volume = 0.0
        self._last_capital_deployed = 0.0
        self._last_snapshot_time = time.time()
        self._deployment_history: List[float] = []
        logger.info("VelocityEngine initialized")

    # ─── Order Flow Velocity ─────────────────────────────────────

    def _compute_order_flow(self) -> dict:
        """Compute trades/sec and volume/sec from broker trade history."""
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
            if eng is None:
                try:
                    from engine.api.routers.execution import _get_exec
                    eng = _get_exec()
                except Exception:
                    pass

            if eng is None or not hasattr(eng, "broker"):
                return self._default_order_flow()

            broker = eng.broker
            trades = broker.get_trade_history() if hasattr(broker, "get_trade_history") else []
            total_trades = len(trades)
            total_volume = sum(
                abs(t.get("quantity", 0) * t.get("fill_price", 0))
                if isinstance(t, dict)
                else abs(getattr(t, "quantity", 0) * getattr(t, "fill_price", 0))
                for t in trades
            )

            elapsed = time.time() - self._start_time
            elapsed = max(elapsed, 1.0)

            # Rolling rate over last snapshot interval
            dt = time.time() - self._last_snapshot_time
            dt = max(dt, 1.0)
            new_trades = total_trades - self._last_trade_count
            new_volume = total_volume - self._last_volume

            tps = new_trades / dt if dt > 0 else 0
            vps = new_volume / dt if dt > 0 else 0

            self._last_trade_count = total_trades
            self._last_volume = total_volume

            return {
                "trades_per_second": round(tps, 3),
                "volume_per_second": round(vps, 2),
                "total_trades_session": total_trades,
                "total_volume_session": round(total_volume, 2),
            }
        except Exception as e:
            logger.debug(f"Order flow computation error: {e}")
            return self._default_order_flow()

    def _default_order_flow(self) -> dict:
        return {
            "trades_per_second": 0.0,
            "volume_per_second": 0.0,
            "total_trades_session": 0,
            "total_volume_session": 0.0,
        }

    # ─── Signal Velocity ─────────────────────────────────────────

    def _compute_signal_velocity(self) -> dict:
        """Compute signal firing rate from universe scan state."""
        try:
            from engine.allocation.universe_scan import UniverseScanOrchestrator
            orch = UniverseScanOrchestrator._instance if hasattr(UniverseScanOrchestrator, "_instance") else None
            if orch is not None and hasattr(orch, "status"):
                status = orch.status
                total_signals = status.total_signals if hasattr(status, "total_signals") else 0
                elapsed = status.elapsed_seconds if hasattr(status, "elapsed_seconds") else 1.0
                elapsed = max(elapsed, 1.0)
                cycles = status.cycle_number if hasattr(status, "cycle_number") else 0
                sps = total_signals / elapsed if elapsed > 0 else 0
                spc = total_signals / max(cycles, 1)
                return {
                    "signals_per_second": round(sps, 3),
                    "signals_per_cycle": int(spc),
                    "active_signals": total_signals,
                    "scan_cycles_completed": cycles,
                }
        except Exception as e:
            logger.debug(f"Signal velocity error: {e}")

        # Fallback: check scan status cache
        try:
            from pathlib import Path
            cache = Path(__file__).resolve().parent.parent.parent / "data" / "scan_status_cache.json"
            if cache.exists():
                with open(cache) as f:
                    data = json.load(f)
                total = data.get("total_signals", 0)
                elapsed = data.get("elapsed_seconds", 1.0) or 1.0
                return {
                    "signals_per_second": round(total / elapsed, 3),
                    "signals_per_cycle": total,
                    "active_signals": total,
                    "scan_cycles_completed": data.get("cycle_number", 0),
                }
        except Exception:
            pass

        return {
            "signals_per_second": 0.0,
            "signals_per_cycle": 0,
            "active_signals": 0,
            "scan_cycles_completed": 0,
        }

    # ─── Execution Velocity ──────────────────────────────────────

    def _compute_execution_velocity(self) -> dict:
        """Compute fill rate and latency metrics."""
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
            if eng is None:
                try:
                    from engine.api.routers.execution import _get_exec
                    eng = _get_exec()
                except Exception:
                    pass

            if eng is None or not hasattr(eng, "broker"):
                return self._default_execution()

            broker = eng.broker
            trades = broker.get_trade_history() if hasattr(broker, "get_trade_history") else []
            if not trades:
                return self._default_execution()

            total = len(trades)
            filled = [t for t in trades if (t.get("fill_price", 0) if isinstance(t, dict) else getattr(t, "fill_price", 0)) > 0]
            fill_rate = len(filled) / total if total > 0 else 0

            latencies = []
            for t in filled:
                lat = t.get("latency_ms", 0) if isinstance(t, dict) else getattr(t, "latency_ms", 0)
                if lat and lat > 0:
                    latencies.append(lat)

            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            sorted_lat = sorted(latencies) if latencies else [0]
            median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
            p99_lat = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else (sorted_lat[0] if sorted_lat else 0)

            return {
                "fill_rate_pct": round(fill_rate * 100, 2),
                "avg_latency_ms": round(avg_lat, 2),
                "median_latency_ms": round(median_lat, 2),
                "p99_latency_ms": round(p99_lat, 2),
                "orders_in_flight": 0,
            }
        except Exception as e:
            logger.debug(f"Execution velocity error: {e}")
            return self._default_execution()

    def _default_execution(self) -> dict:
        return {
            "fill_rate_pct": 0.0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "orders_in_flight": 0,
        }

    # ─── Capital Deployment Velocity ─────────────────────────────

    def _compute_capital_deployment(self) -> dict:
        """Compute capital deployment rate from portfolio state."""
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
            if eng is None:
                try:
                    from engine.api.routers.execution import _get_exec
                    eng = _get_exec()
                except Exception:
                    pass

            if eng is None or not hasattr(eng, "broker"):
                return self._default_deployment()

            broker = eng.broker
            summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
            nav = summary.get("nav", 0)
            cash = summary.get("cash", nav)
            deployed = nav - cash

            dt_min = (time.time() - self._last_snapshot_time) / 60.0
            dt_min = max(dt_min, 0.0167)  # At least 1 second
            delta = deployed - self._last_capital_deployed
            rate = delta / dt_min

            self._deployment_history.append(rate)
            if len(self._deployment_history) > 30:
                self._deployment_history = self._deployment_history[-30:]

            # Trend detection
            trend = "STABLE"
            if len(self._deployment_history) >= 5:
                recent = self._deployment_history[-5:]
                older = self._deployment_history[-10:-5] if len(self._deployment_history) >= 10 else self._deployment_history[:5]
                avg_recent = sum(recent) / len(recent)
                avg_older = sum(older) / len(older)
                if avg_recent > avg_older * 1.2:
                    trend = "ACCELERATING"
                elif avg_recent < avg_older * 0.8:
                    trend = "DECELERATING"

            self._last_capital_deployed = deployed

            return {
                "capital_deployed_per_min": round(rate, 2),
                "total_capital_deployed": round(deployed, 2),
                "deployment_rate_trend": trend,
                "nav": round(nav, 2),
            }
        except Exception as e:
            logger.debug(f"Capital deployment error: {e}")
            return self._default_deployment()

    def _default_deployment(self) -> dict:
        return {
            "capital_deployed_per_min": 0.0,
            "total_capital_deployed": 0.0,
            "deployment_rate_trend": "STABLE",
            "nav": 0.0,
        }

    # ─── Momentum Score ──────────────────────────────────────────

    def _compute_momentum(self) -> dict:
        """Compute rolling momentum across portfolio positions."""
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
            if eng is None:
                try:
                    from engine.api.routers.execution import _get_exec
                    eng = _get_exec()
                except Exception:
                    pass

            if eng is None or not hasattr(eng, "broker"):
                return self._default_momentum()

            broker = eng.broker
            positions = broker.get_all_positions()
            if not positions:
                return self._default_momentum()

            scores = []
            leaders = []
            laggards = []

            for ticker, pos in (positions.items() if isinstance(positions, dict) else []):
                pnl_pct = 0
                if isinstance(pos, dict):
                    cost = pos.get("avg_cost", 0)
                    current = pos.get("current_price", 0)
                else:
                    cost = getattr(pos, "avg_cost", 0)
                    current = getattr(pos, "current_price", 0)

                if cost and cost > 0:
                    pnl_pct = ((current - cost) / cost) * 100
                scores.append(pnl_pct)

                entry = {"ticker": ticker, "momentum": round(pnl_pct, 2)}
                if pnl_pct > 0:
                    leaders.append(entry)
                else:
                    laggards.append(entry)

            leaders.sort(key=lambda x: x["momentum"], reverse=True)
            laggards.sort(key=lambda x: x["momentum"])

            avg_score = sum(scores) / len(scores) if scores else 0
            breadth = len([s for s in scores if s > 0]) / len(scores) if scores else 0

            return {
                "momentum_score": round(avg_score, 4),
                "momentum_breadth": round(breadth * 100, 2),
                "momentum_leaders": leaders[:5],
                "momentum_laggards": laggards[:5],
            }
        except Exception as e:
            logger.debug(f"Momentum computation error: {e}")
            return self._default_momentum()

    def _default_momentum(self) -> dict:
        return {
            "momentum_score": 0.0,
            "momentum_breadth": 0.0,
            "momentum_leaders": [],
            "momentum_laggards": [],
        }

    # ─── Aggregate Snapshot ──────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return complete velocity snapshot."""
        now = time.time()
        result = {}
        result.update(self._compute_order_flow())
        result.update(self._compute_signal_velocity())
        result.update(self._compute_execution_velocity())
        result.update(self._compute_capital_deployment())
        result.update(self._compute_momentum())
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["uptime_seconds"] = round(now - self._start_time, 1)
        result["engine_status"] = "ONLINE"
        self._last_snapshot_time = now
        return result

    def get_order_flow(self) -> dict:
        """Return just order flow metrics."""
        data = self._compute_order_flow()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return data

    def get_signal_velocity(self) -> dict:
        """Return just signal velocity metrics."""
        data = self._compute_signal_velocity()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return data

    def get_execution_velocity(self) -> dict:
        """Return just execution velocity metrics."""
        data = self._compute_execution_velocity()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return data

    def get_capital_deployment(self) -> dict:
        """Return just capital deployment metrics."""
        data = self._compute_capital_deployment()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return data

    def get_momentum(self) -> dict:
        """Return just momentum metrics."""
        data = self._compute_momentum()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return data

    async def get_stream_data(self, interval: float = 2.0) -> AsyncGenerator[dict, None]:
        """Yield velocity snapshots at regular intervals for SSE streaming."""
        while True:
            try:
                snapshot = self.get_snapshot()
                yield snapshot
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
            await asyncio.sleep(interval)
