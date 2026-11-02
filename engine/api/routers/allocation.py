"""Metadron Capital — Allocation Engine API endpoints.

REST + SSE endpoints for the allocation engine, scan orchestrator,
and collateral/margin monitoring.

Endpoints:
    GET  /api/allocation/rules      — current allocation rules
    POST /api/allocation/rules      — update allocation rules
    GET  /api/allocation/status     — bucket utilization, kill switch, beta corridor
    GET  /api/allocation/slate      — last computed allocation slate
    GET  /api/scan/status           — current scan cycle status
    GET  /api/scan/thinking         — SSE stream of real-time scan signals
    GET  /api/collateral/status     — margin bucket status
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("metadron.api.allocation")

# ─── Prometheus instrumentation ──────────────────────────────────
_prom_alloc_api = None

def _get_prom_alloc_api():
    global _prom_alloc_api
    if _prom_alloc_api is not None:
        return _prom_alloc_api
    try:
        from engine.bridges.prometheus_metrics import get_metrics
        _prom_alloc_api = get_metrics()
    except Exception:
        _prom_alloc_api = {}
    return _prom_alloc_api

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise RuntimeError("FastAPI is required.")

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None
    logger.warning("sse_starlette not installed — SSE endpoints will return 501.")

# Engine imports — graceful fallback
try:
    from engine.allocation.allocation_engine import (
        AllocationEngine, AllocationRules, BetaCorridorEngine,
        KillSwitchMonitor, AllocationSlate,
    )
    from engine.allocation.universe_scan import (
        FullUniverseScan, signal_bus, ScanCycleStatus,
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning("Allocation engine not available: %s", e)
    ENGINE_AVAILABLE = False


router = APIRouter()

# ═══════════════════════════════════════════════════════════════════════════
# Singleton engine instances
# ═══════════════════════════════════════════════════════════════════════════

_allocation_engine: Optional[Any] = None
_scan_orchestrator: Optional[Any] = None


def _get_engine():
    global _allocation_engine
    if _allocation_engine is None and ENGINE_AVAILABLE:
        _allocation_engine = AllocationEngine(nav=1_000_000)
    return _allocation_engine


def _get_scanner():
    global _scan_orchestrator
    if _scan_orchestrator is None and ENGINE_AVAILABLE:
        _scan_orchestrator = FullUniverseScan(allocation_engine=_get_engine())
    return _scan_orchestrator


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class RulesUpdateRequest(BaseModel):
    max_drawdown_kill_switch: Optional[float] = None
    ig_equity_pct: Optional[float] = None
    hy_equity_pct: Optional[float] = None
    distressed_equity_pct: Optional[float] = None
    tltw_cashflow_pct: Optional[float] = None
    fi_macro_pct: Optional[float] = None
    event_driven_cvr_pct: Optional[float] = None
    options_notional_pct: Optional[float] = None
    margin_pct: Optional[float] = None
    money_market_pct: Optional[float] = None
    drip_rule: Optional[bool] = None


# ═══════════════════════════════════════════════════════════════════════════
# Allocation Rules Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/rules")
async def get_allocation_rules():
    """Return current allocation rules as JSON."""
    engine = _get_engine()
    if engine is None:
        return _fallback_rules()
    return engine.rules.to_dict()


@router.post("/rules")
async def update_allocation_rules(request: RulesUpdateRequest):
    """Update allocation rules from operator instructions."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Allocation engine not available.")

    updates = {k: v for k, v in request.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No rule updates provided.")

    updated_rules = engine.update_rules(updates)
    return {
        "status": "updated",
        "rules": updated_rules.to_dict(),
        "changes": updates,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Allocation Status Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_allocation_status():
    """Return current bucket utilization, kill switch status, beta corridor."""
    engine = _get_engine()
    if engine is None:
        return _fallback_status()
    return engine.get_status()


@router.get("/slate")
async def get_allocation_slate():
    """Return last computed allocation slate."""
    scanner = _get_scanner()
    if scanner is None:
        return _fallback_slate()
    slate = scanner.get_last_slate()
    if slate is None:
        return _fallback_slate()
    return slate


# ═══════════════════════════════════════════════════════════════════════════
# Scan Status Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/scan/status")
async def get_scan_status():
    """Return current scan cycle status."""
    scanner = _get_scanner()
    if scanner is None:
        return _fallback_scan_status()
    return scanner.get_scan_status()


@router.get("/scan/thinking")
async def stream_thinking():
    """SSE stream of real-time scan signals and reasoning.

    Feeds the Thinking Tab with live signal discovery events.
    """
    if EventSourceResponse is None:
        return JSONResponse(
            status_code=501,
            content={"detail": "SSE streaming not available — install sse-starlette."},
        )

    async def event_generator():
        if not ENGINE_AVAILABLE:
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "type": "heartbeat",
                    "message": "Allocation engine not loaded — signals idle.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
            }
            return

        queue = signal_bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield {
                        "event": "thinking",
                        "data": json.dumps(event),
                    }
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    scanner = _get_scanner()
                    status = scanner.get_scan_status() if scanner else {}
                    yield {
                        "event": "thinking",
                        "data": json.dumps({
                            "type": "heartbeat",
                            "scan_status": status,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }),
                    }
        finally:
            signal_bus.unsubscribe(queue)

    return EventSourceResponse(event_generator())


# ═══════════════════════════════════════════════════════════════════════════
# Collateral / Margin Endpoint
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/collateral/status")
async def get_collateral_status():
    """Return margin bucket status: real capital, notional exposure, beta corridor, leverage."""
    engine = _get_engine()
    if engine is None:
        return _fallback_collateral()

    utilization = engine._utilization
    rules = engine.rules
    beta = engine.beta_engine.status()
    kill = engine.kill_switch.status(engine.nav)

    options_total = (
        utilization.get("OPTIONS_IG", 0.0) +
        utilization.get("OPTIONS_HY", 0.0) +
        utilization.get("OPTIONS_DISTRESSED", 0.0)
    )
    margin_real = utilization.get("MARGIN", 0.0)
    total_real_capital = margin_real + options_total * 0.3  # ~30% of notional as real capital

    result = {
        "beta_corridor": beta,
        "margin_bucket": {
            "real_capital_deployed_pct": round(total_real_capital, 4),
            "real_capital_deployed_usd": round(total_real_capital * engine.nav, 2),
            "real_capital_range": [rules.margin_pct, rules.margin_pct + rules.ig_reduction_on_margin_breach],
            "notional_exposure_pct": round(options_total + margin_real, 4),
            "notional_exposure_usd": round((options_total + margin_real) * engine.nav, 2),
        },
        "breakdown": {
            "futures_margin": round(margin_real, 4),
            "options_premium": {
                "ig": round(utilization.get("OPTIONS_IG", 0.0), 4),
                "hy": round(utilization.get("OPTIONS_HY", 0.0), 4),
                "distressed": round(utilization.get("OPTIONS_DISTRESSED", 0.0), 4),
                "total": round(options_total, 4),
            },
            "leverage_multiplier": beta.get("leverage_multiplier", 1.0),
        },
        "kill_switch": kill,
        "utilization_alert": total_real_capital > 0.12,  # Alert at >12%
        "nav": engine.nav,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Prometheus: update margin/collateral metrics
    try:
        prom = _get_prom_alloc_api()
        if prom:
            if "margin_utilization_pct" in prom:
                prom["margin_utilization_pct"].set(round(total_real_capital * 100, 2))
            if "beta_corridor_value" in prom:
                prom["beta_corridor_value"].set(beta.get("beta", 0))
            if "leverage_multiplier" in prom:
                prom["leverage_multiplier"].set(beta.get("leverage_multiplier", 1.0))
            if "drawdown_pct" in prom:
                prom["drawdown_pct"].set(kill.get("current_drawdown", 0) * 100)
            if "collateral_available_usd" in prom:
                deployed_usd = total_real_capital * engine.nav
                prom["collateral_available_usd"].set(engine.nav - deployed_usd)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Fallback responses when engine is not loaded
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_rules() -> dict:
    rules = AllocationRules() if ENGINE_AVAILABLE else {}
    if isinstance(rules, dict):
        return {
            "max_drawdown_kill_switch": 0.10,
            "ig_equity_pct": 0.40,
            "hy_equity_pct": 0.10,
            "distressed_equity_pct": 0.10,
            "tltw_cashflow_pct": 0.15,
            "fi_macro_pct": 0.05,
            "event_driven_cvr_pct": 0.10,
            "options_notional_pct": 0.10,
            "options_ig_pct": 0.04,
            "options_hy_pct": 0.04,
            "options_distressed_pct": 0.02,
            "margin_pct": 0.08,
            "money_market_pct": 0.0,
            "drip_rule": True,
            "alpha_primary_goal": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return rules.to_dict()


def _fallback_status() -> dict:
    return {
        "rules": _fallback_rules(),
        "bucket_utilization": {
            "IG_EQUITY": 0.0, "HY_EQUITY": 0.0,
            "TLTW_CASHFLOW": 0.0, "FI_MACRO": 0.0,
            "EVENT_DRIVEN_CVR": 0.0, "OPTIONS_IG": 0.0,
            "OPTIONS_HY": 0.0, "OPTIONS_DISTRESSED": 0.0,
            "MONEY_MARKET": 0.0, "MARGIN": 0.0,
        },
        "kill_switch": {
            "triggered": False, "high_water_mark": 0.0,
            "current_drawdown": 0.0, "max_drawdown_threshold": 0.20,
            "trigger_timestamp": None, "total_events": 0,
        },
        "beta_corridor": {
            "beta": 1.0, "corridor": "NEUTRAL",
            "leverage_multiplier": 1.0, "history_length": 0,
        },
        "drip_events": 0, "rule_changes": 0,
        "nav": 1000000, "backtest": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_slate() -> dict:
    return {
        "positions": [],
        "bucket_utilization": {
            "ig_equity": 0.0, "hy_equity": 0.0,
            "distressed_equity": 0.0, "tltw_cashflow": 0.0,
            "fi_macro": 0.0, "event_driven_cvr": 0.0,
            "options_notional": 0.0,
            "margin": 0.0, "money_market": 0.0,
        },
        "kill_switch_triggered": False,
        "beta_corridor": "NEUTRAL",
        "leverage_multiplier": 1.0,
        "cycle_number": 0,
        "phase": "IDLE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_scan_status() -> dict:
    return {
        "cycle_number": 0, "phase": "IDLE",
        "current_run": 0, "current_universe": "",
        "elapsed_seconds": 0.0, "total_signals": 0,
        "runs": [], "started_at": "", "completed": False,
    }


def _fallback_collateral() -> dict:
    return {
        "beta_corridor": {
            "beta": 1.0, "corridor": "NEUTRAL",
            "leverage_multiplier": 1.0, "history_length": 0,
        },
        "margin_bucket": {
            "real_capital_deployed_pct": 0.0,
            "real_capital_deployed_usd": 0.0,
            "real_capital_range": [0.05, 0.15],
            "notional_exposure_pct": 0.0,
            "notional_exposure_usd": 0.0,
        },
        "breakdown": {
            "futures_margin": 0.0,
            "options_premium": {"ig": 0.0, "hy": 0.0, "distressed": 0.0, "total": 0.0},
            "leverage_multiplier": 1.0,
        },
        "kill_switch": {
            "triggered": False, "high_water_mark": 0.0,
            "current_drawdown": 0.0, "max_drawdown_threshold": 0.20,
            "trigger_timestamp": None, "total_events": 0,
        },
        "utilization_alert": False,
        "nav": 1000000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Kill Switch Reset Endpoints ────────────────────────────────────────────

@router.post("/kill-switch/request-reset", tags=["Kill Switch"])
async def kill_switch_request_reset():
    """Operator signals intent to reset the kill switch (step 1 of 2).

    Sets pending_reset=True and logs the request timestamp.
    The triggered latch remains active until confirm-reset is called.
    Emits kill_switch_pending_reset gauge = 1 to Prometheus.
    """
    try:
        engine = _get_engine()
        engine.kill_switch.request_reset()
        try:
            metrics = _get_prom_alloc_api()
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(1)
        except Exception:
            pass
        return {
            "status": "reset_requested",
            "pending_reset": True,
            "kill_switch_triggered": engine.kill_switch.triggered,
            "message": (
                "Kill switch reset requested. "
                "Call POST /kill-switch/confirm-reset to complete reset and resume trading."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/kill-switch/confirm-reset", tags=["Kill Switch"])
async def kill_switch_confirm_reset():
    """Operator confirms kill switch reset — clears triggered state (step 2 of 2).

    Clears both triggered and pending_reset flags.
    Engine resumes with fresh allocation target on next cycle (targeting 20%).
    Emits kill_switch_pending_reset gauge = 0 and increments kill_switch_reset_total.
    """
    try:
        engine = _get_engine()
        engine.kill_switch.confirm_reset()
        try:
            metrics = _get_prom_alloc_api()
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(0)
            if "kill_switch_reset_total" in metrics:
                metrics["kill_switch_reset_total"].inc()
        except Exception:
            pass
        return {
            "status": "reset_confirmed",
            "triggered": False,
            "pending_reset": False,
            "message": (
                "Kill switch reset confirmed. "
                "Engine will resume normal allocation on the next cycle targeting 20%."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/kill-switch/status", tags=["Kill Switch"])
async def kill_switch_status():
    """Get current kill switch state including pending_reset flag.

    Returns triggered, pending_reset, triggered_at, threshold, and reset_requested_at.
    This is the canonical indicator for the TECH tab and Grafana dashboard.
    """
    try:
        engine = _get_engine()
        return engine.kill_switch.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
