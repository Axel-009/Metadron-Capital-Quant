"""
Velocity router — VELOCITY tab
Real-time velocity metrics: order flow, signal, execution, capital, momentum.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
import logging
import asyncio
import json

logger = logging.getLogger("metadron-api.velocity")
router = APIRouter()

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from engine.velocity.velocity_engine import VelocityEngine
        _engine = VelocityEngine()
    return _engine


@router.get("/snapshot")
async def velocity_snapshot():
    """Full velocity snapshot — all metrics."""
    try:
        engine = _get_engine()
        return engine.get_snapshot()
    except Exception as e:
        logger.error(f"Velocity snapshot error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/order-flow")
async def order_flow():
    """Order flow velocity — trades/sec, volume/sec."""
    try:
        engine = _get_engine()
        return engine.get_order_flow()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/signal-velocity")
async def signal_velocity():
    """Signal velocity — signals/sec across universe scan."""
    try:
        engine = _get_engine()
        return engine.get_signal_velocity()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/execution-velocity")
async def execution_velocity():
    """Execution velocity — fill rate, latency."""
    try:
        engine = _get_engine()
        return engine.get_execution_velocity()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/capital-deployment")
async def capital_deployment():
    """Capital deployment velocity — $/min deployed."""
    try:
        engine = _get_engine()
        return engine.get_capital_deployment()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/momentum")
async def momentum():
    """Portfolio momentum — rolling momentum score + leaders/laggards."""
    try:
        engine = _get_engine()
        return engine.get_momentum()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/stream")
async def velocity_stream():
    """SSE stream of velocity snapshots every 2 seconds."""
    async def event_generator():
        engine = _get_engine()
        async for snapshot in engine.get_stream_data(interval=2.0):
            data = json.dumps(snapshot)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
