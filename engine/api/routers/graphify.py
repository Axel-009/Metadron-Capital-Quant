"""Graphify Knowledge Graph — API Router (Security Hardened).

All processes are timestamped + logged. Kill switch bypasses graphify
without interrupting the learning loop (reroutes to LLM review).

Prometheus metrics:
  metadron_graphify_runs_total      — graph generation runs
  metadron_graphify_queries_total   — query count
  metadron_graphify_kill_switch     — 1 if killed, 0 if active
  metadron_graphify_run_duration_s  — last generation duration

Endpoints:
    GET  /graphify/status      — availability, god nodes, metrics
    GET  /graphify/report      — GRAPH_REPORT.md
    GET  /graphify/query?q=    — natural language query
    POST /graphify/generate    — trigger graph regeneration (timestamped)
    GET  /graphify/god-nodes   — highest-degree concept nodes
    POST /graphify/kill        — activate kill switch (bypass graphify)
    POST /graphify/resume      — deactivate kill switch
    GET  /graphify/run-log     — audit trail of all graphify processes
    GET  /graphify/metrics     — Prometheus-format metrics
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

logger = logging.getLogger("metadron-api.graphify")
router = APIRouter()

_graphify = None


def _get_bridge():
    global _graphify
    if _graphify is None:
        from engine.agents.graphify_bridge import GraphifyBridge
        _graphify = GraphifyBridge()
    return _graphify


@router.get("/status")
async def graphify_status():
    """Knowledge graph status, availability, god nodes, and metrics."""
    try:
        bridge = _get_bridge()
        from pathlib import Path
        _REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        graph_path = _REPO_ROOT / "graphify-out" / "graph.json"
        report_path = _REPO_ROOT / "graphify-out" / "GRAPH_REPORT.md"

        god_nodes = bridge.get_god_nodes() if bridge.is_available() else bridge._cached_god_nodes
        metrics = bridge.get_metrics()

        return {
            "available": bridge.is_available(),
            "kill_switch": bridge.is_killed,
            "graph_exists": graph_path.exists(),
            "report_exists": report_path.exists(),
            "graph_size_kb": round(graph_path.stat().st_size / 1024, 1) if graph_path.exists() else 0,
            "god_nodes": god_nodes,
            "god_node_count": len(god_nodes),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/status error: {e}")
        return {"available": False, "error": str(e)}


@router.get("/report")
async def graphify_report():
    """Return the full GRAPH_REPORT.md analysis."""
    try:
        bridge = _get_bridge()
        report = bridge.get_report()
        return {
            "report": report,
            "has_report": bool(report and "No graph report" not in report and "killed" not in report),
            "kill_switch": bridge.is_killed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/report error: {e}")
        return {"report": "", "has_report": False, "error": str(e)}


@router.get("/query")
async def graphify_query(
    q: str = Query(..., description="Natural language question about the codebase"),
):
    """Query the knowledge graph. Timestamped + logged."""
    try:
        bridge = _get_bridge()
        answer = bridge.query(q)
        return {
            "question": q,
            "answer": answer,
            "kill_switch": bridge.is_killed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/query error: {e}")
        return {"question": q, "answer": "", "error": str(e)}


@router.post("/generate")
async def graphify_generate():
    """Trigger graph regeneration. Timestamped + logged + hash-signed."""
    try:
        bridge = _get_bridge()
        result = bridge.generate_graph()
        return {**result, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"graphify/generate error: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/god-nodes")
async def graphify_god_nodes(limit: int = Query(20, ge=1, le=100)):
    """Return highest-degree concept nodes from the knowledge graph."""
    try:
        bridge = _get_bridge()
        nodes = bridge.get_god_nodes()
        return {
            "nodes": nodes[:limit],
            "total": len(nodes),
            "kill_switch": bridge.is_killed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/god-nodes error: {e}")
        return {"nodes": [], "total": 0, "error": str(e)}


@router.post("/kill")
async def graphify_kill():
    """Activate graphify kill switch.

    Bypasses all graphify processing. The learning loop continues —
    any data that would go through graphify is rerouted through
    LLM review directly. Cached god nodes remain accessible.
    No interruption to the live loop.
    """
    try:
        bridge = _get_bridge()
        bridge.activate_kill_switch()
        return {
            "status": "killed",
            "message": "Graphify bypassed — learning loop continues via LLM review",
            "cached_nodes": len(bridge._cached_god_nodes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/kill error: {e}")
        return {"error": str(e)}


@router.post("/resume")
async def graphify_resume():
    """Deactivate graphify kill switch. Resume normal processing."""
    try:
        bridge = _get_bridge()
        bridge.deactivate_kill_switch()
        return {
            "status": "resumed",
            "available": bridge.is_available(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/resume error: {e}")
        return {"error": str(e)}


@router.get("/run-log")
async def graphify_run_log(limit: int = Query(50, ge=1, le=500)):
    """Audit trail of all graphify processes — timestamped + hash-signed."""
    try:
        bridge = _get_bridge()
        entries = bridge.get_run_log(limit)
        return {
            "entries": entries,
            "total": len(entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/run-log error: {e}")
        return {"entries": [], "error": str(e)}


@router.get("/metrics")
async def graphify_metrics():
    """Prometheus-compatible metrics for graphify usage tracking."""
    try:
        bridge = _get_bridge()
        m = bridge.get_metrics()
        # Return in a format Prometheus can scrape
        lines = [
            f'metadron_graphify_kill_switch {1 if m["kill_switch"] else 0}',
            f'metadron_graphify_available {1 if m["available"] else 0}',
            f'metadron_graphify_runs_total {m["total_runs"]}',
            f'metadron_graphify_queries_total {m["total_queries"]}',
            f'metadron_graphify_run_duration_seconds {m["total_run_time_s"]}',
            f'metadron_graphify_cached_nodes {m["cached_god_nodes"]}',
        ]
        return {"metrics": lines, "raw": m}
    except Exception as e:
        return {"error": str(e)}
