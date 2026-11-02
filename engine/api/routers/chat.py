"""
Chat router — CHAT tab
Bridges frontend API calls to NanoClaw agent, manages Ruflo agent status,
and handles CEO recommendation approval workflow.
"""
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from pydantic import BaseModel
import logging
import asyncio
import json
import time

logger = logging.getLogger("metadron-api.chat")
router = APIRouter()

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

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_nanoclaw_agent = None
_guard = None


def _get_agent():
    global _nanoclaw_agent
    if _nanoclaw_agent is None:
        try:
            from engine.agents.nanoclaw.nanoclaw_agent import NanoClawAgent
            _nanoclaw_agent = NanoClawAgent()
        except Exception as e:
            logger.error(f"Failed to initialize NanoClawAgent: {e}")
            raise
    return _nanoclaw_agent


def _get_guard():
    global _guard
    if _guard is None:
        from engine.agents.nanoclaw.permission_guard import AgentPermissionGuard
        _guard = AgentPermissionGuard()
    return _guard


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class SendMessageRequest(BaseModel):
    message: str


class RufloSendRequest(BaseModel):
    agent_id: str = "all"
    message: str


# ---------------------------------------------------------------------------
# NanoClaw endpoints
# ---------------------------------------------------------------------------
@router.get("/nanoclaw/messages")
async def get_nanoclaw_messages():
    """Return NanoClaw message history."""
    try:
        agent = _get_agent()
        return {"messages": agent.get_history(), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/nanoclaw/send")
async def send_nanoclaw_message(req: SendMessageRequest):
    """Send a message to NanoClaw and stream the response via SSE."""
    try:
        agent = _get_agent()
        prom = _get_prom()
        start_t = time.time()

        # Track message count
        if prom and "nanoclaw_messages_total" in prom:
            prom["nanoclaw_messages_total"].inc()
        if prom and "active_chat_sessions" in prom:
            prom["active_chat_sessions"].set(1)

        async def event_generator():
            try:
                async for token in agent.send_message(req.message):
                    data = json.dumps({"type": "token", "content": token})
                    yield f"data: {data}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                # Track response latency
                if prom and "chat_response_latency_ms" in prom:
                    prom["chat_response_latency_ms"].observe((time.time() - start_t) * 1000)
            except Exception as e:
                logger.error(f"NanoClaw stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/nanoclaw/stream")
async def nanoclaw_stream():
    """SSE stream for NanoClaw status updates."""
    async def event_generator():
        while True:
            try:
                agent = _get_agent()
                ctx = agent.get_system_context()
                data = json.dumps({"type": "status", "context": ctx})
                yield f"data: {data}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            await asyncio.sleep(10)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Ruflo endpoints
# ---------------------------------------------------------------------------
MOCK_RUFLO_AGENTS = [
    {
        "agent_id": "ruflo-alpha",
        "name": "Alpha Scanner",
        "current_task": "Scanning SP500 universe for alpha signals",
        "status": "active",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "signals_processed": 142,
    },
    {
        "agent_id": "ruflo-macro",
        "name": "Macro Sentinel",
        "current_task": "Monitoring FRED data feeds for regime shifts",
        "status": "active",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "signals_processed": 87,
    },
    {
        "agent_id": "ruflo-risk",
        "name": "Risk Watchdog",
        "current_task": "Tracking beta corridor and kill switch status",
        "status": "active",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "signals_processed": 56,
    },
    {
        "agent_id": "ruflo-distress",
        "name": "Distress Hunter",
        "current_task": "Idle — awaiting task delegation",
        "status": "idle",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "signals_processed": 23,
    },
    {
        "agent_id": "ruflo-event",
        "name": "Event Monitor",
        "current_task": "Scanning SEC filings for material events",
        "status": "active",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "signals_processed": 34,
    },
]


@router.get("/ruflo/agents")
async def get_ruflo_agents():
    """List active Ruflo agents and their status."""
    return {"agents": MOCK_RUFLO_AGENTS, "timestamp": datetime.utcnow().isoformat()}


@router.post("/ruflo/send")
async def send_ruflo_message(req: RufloSendRequest):
    """Send a message to a specific Ruflo agent or broadcast to all."""
    target = req.agent_id
    if target == "all":
        responses = []
        for agent in MOCK_RUFLO_AGENTS:
            responses.append({
                "agent_id": agent["agent_id"],
                "name": agent["name"],
                "response": f"[{agent['name']}] Acknowledged: {req.message}. Current task: {agent['current_task']}. Status: {agent['status']}.",
                "timestamp": datetime.utcnow().isoformat(),
            })
        return {"responses": responses, "broadcast": True}

    matched = next((a for a in MOCK_RUFLO_AGENTS if a["agent_id"] == target), None)
    if not matched:
        return JSONResponse(status_code=404, content={"error": f"Ruflo agent '{target}' not found"})

    return {
        "agent_id": matched["agent_id"],
        "name": matched["name"],
        "response": f"[{matched['name']}] Acknowledged: {req.message}. Current task: {matched['current_task']}. Status: {matched['status']}.",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
@router.get("/recommendations")
async def get_recommendations():
    """Return pending CEO recommendations."""
    try:
        agent = _get_agent()
        recs = agent.get_pending_recommendations()
        # Track recommendation count
        prom = _get_prom()
        if prom and "openclaw_recommendations_total" in prom and recs:
            prom["openclaw_recommendations_total"].inc(len(recs))
        return {"recommendations": recs, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/recommendations/{rec_id}/approve")
async def approve_recommendation(rec_id: int):
    """Approve a CEO recommendation (routes to NanoClaw)."""
    try:
        agent = _get_agent()
        result = agent.approve_recommendation(rec_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/recommendations/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int):
    """Dismiss a CEO recommendation."""
    try:
        agent = _get_agent()
        result = agent.dismiss_recommendation(rec_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Agent Permissions
# ---------------------------------------------------------------------------
@router.get("/agent-permissions")
async def get_agent_permissions():
    """Return permission map for all agents."""
    try:
        guard = _get_guard()
        permissions = guard.get_all_permissions()
        # Track permission guard blocks
        prom = _get_prom()
        if prom and "permission_guard_blocks_total" in prom and isinstance(permissions, dict):
            blocks = sum(1 for v in permissions.values()
                        if isinstance(v, dict) and not v.get("allowed", True))
            if blocks > 0:
                prom["permission_guard_blocks_total"].inc(blocks)
        return {
            "permissions": permissions,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
