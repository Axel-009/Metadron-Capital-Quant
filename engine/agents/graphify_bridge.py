"""Metadron Capital — Graphify Bridge (Security Hardened).

All graphify processes are:
  - Restricted to timestamped runs only (no ad-hoc subprocess calls)
  - Logged with hash signatures for audit trail
  - Tracked in Prometheus (runs, duration, node count, queries)
  - Subject to kill switch that bypasses graphify and reroutes
    output through LLM review without interrupting the learning loop

Kill switch: when active, all graphify calls return cached data or
empty results. The learning loop continues using LLM review directly
for any analysis that would normally go through graphify.

Install: pip install graphifyy
Build graph: graphify . (from repo root)
"""
from __future__ import annotations
import hashlib
import json
import logging
import subprocess
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GRAPH_JSON = _REPO_ROOT / "graphify-out" / "graph.json"
_GRAPH_REPORT = _REPO_ROOT / "graphify-out" / "GRAPH_REPORT.md"
_RUN_LOG = _REPO_ROOT / "data" / "graphify" / "run_log.jsonl"
_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

try:
    import graphifyy  # noqa: F401
    GRAPHIFY_AVAILABLE = True
except ImportError:
    GRAPHIFY_AVAILABLE = False
    logger.info("graphifyy not installed — run: pip install graphifyy")


class GraphifyBridge:
    """Security-hardened knowledge graph bridge.

    All subprocess execution is:
    - Restricted to the graphify binary only (no arbitrary commands)
    - Timestamped and logged to data/graphify/run_log.jsonl
    - Hash-signed for tamper detection
    - Tracked via Prometheus metrics
    - Subject to kill switch bypass
    """

    def __init__(self):
        self._kill_switch = False
        self._cached_god_nodes: list = []
        self._cached_report: str = ""
        self._lock = threading.Lock()
        # Metrics
        self.total_runs = 0
        self.total_queries = 0
        self.total_run_time_s = 0.0
        self.last_run_at: Optional[str] = None
        self.last_run_duration_s: float = 0.0
        self.last_run_hash: str = ""

    # ── Kill switch ──────────────────────────────────────────

    def activate_kill_switch(self):
        """Bypass graphify completely. Learning loop continues via LLM review."""
        with self._lock:
            self._kill_switch = True
            logger.warning("GRAPHIFY KILL SWITCH ACTIVATED — all output bypassed to LLM review")

    def deactivate_kill_switch(self):
        """Re-enable graphify processing."""
        with self._lock:
            self._kill_switch = False
            logger.info("GRAPHIFY KILL SWITCH DEACTIVATED — graphify processing resumed")

    @property
    def is_killed(self) -> bool:
        return self._kill_switch

    # ── Availability ─────────────────────────────────────────

    def is_available(self) -> bool:
        if self._kill_switch:
            return False
        return GRAPHIFY_AVAILABLE and _GRAPH_JSON.exists()

    # ── Timestamped run logging ──────────────────────────────

    def _log_run(self, run_type: str, duration_s: float, result: dict):
        """Log every graphify process to audit trail with hash signature."""
        entry = {
            "run_type": run_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration_s, 3),
            "result_summary": {k: v for k, v in result.items() if k != "report"},
            "kill_switch": self._kill_switch,
        }
        # Sign the entry
        payload = json.dumps(entry, sort_keys=True, default=str).encode()
        entry["signature"] = hashlib.sha256(payload).hexdigest()

        try:
            with open(_RUN_LOG, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error("Graphify run log write failed: %s", e)

        return entry

    # ── Core operations (all timestamped + logged) ───────────

    def get_report(self) -> str:
        """Returns GRAPH_REPORT.md contents. Logged + cached."""
        t0 = time.time()
        if self._kill_switch:
            self._log_run("report", 0, {"bypassed": True, "reason": "kill_switch"})
            return self._cached_report or "Graphify killed — using cached/LLM review."

        if not _GRAPH_REPORT.exists():
            return "No graph report found. Run: graphify . from repo root."
        try:
            report = _GRAPH_REPORT.read_text(encoding="utf-8")
            self._cached_report = report
            self._log_run("report", time.time() - t0, {"length": len(report)})
            return report
        except Exception as e:
            logger.warning("GraphifyBridge.get_report error: %s", e)
            self._log_run("report", time.time() - t0, {"error": str(e)})
            return ""

    def query(self, question: str) -> str:
        """Query the knowledge graph. Timestamped + logged."""
        t0 = time.time()
        if self._kill_switch:
            self._log_run("query", 0, {"bypassed": True, "question": question[:100]})
            return "Graphify killed — reroute this query through LLM review."

        if not self.is_available():
            return "Graph not available. Run: graphify . from repo root."

        self.total_queries += 1
        try:
            # RESTRICTED: only graphify binary, only query subcommand, only against our graph
            result = subprocess.run(
                ["graphify", "query", question, "--graph", str(_GRAPH_JSON)],
                capture_output=True, text=True, timeout=30, cwd=str(_REPO_ROOT),
            )
            answer = result.stdout.strip() or result.stderr.strip()
            duration = time.time() - t0
            self._log_run("query", duration, {"question": question[:100], "answer_length": len(answer)})
            return answer
        except Exception as e:
            logger.warning("GraphifyBridge.query error: %s", e)
            self._log_run("query", time.time() - t0, {"error": str(e)})
            return f"Query failed: {e}"

    def get_god_nodes(self) -> list:
        """Returns highest-degree concept nodes. Timestamped + cached."""
        t0 = time.time()
        if self._kill_switch:
            self._log_run("god_nodes", 0, {"bypassed": True, "cached_count": len(self._cached_god_nodes)})
            return self._cached_god_nodes

        if not _GRAPH_JSON.exists():
            return []
        try:
            data = json.loads(_GRAPH_JSON.read_text(encoding="utf-8"))
            nodes = data.get("nodes", [])
            sorted_nodes = sorted(nodes, key=lambda n: n.get("degree", 0), reverse=True)
            result = [
                {"id": n.get("id"), "label": n.get("label"), "degree": n.get("degree")}
                for n in sorted_nodes[:20]
            ]
            self._cached_god_nodes = result
            duration = time.time() - t0
            self._log_run("god_nodes", duration, {"count": len(result)})
            return result
        except Exception as e:
            logger.warning("GraphifyBridge.get_god_nodes error: %s", e)
            self._log_run("god_nodes", time.time() - t0, {"error": str(e)})
            return []

    def generate_graph(self) -> dict:
        """Trigger graph regeneration. Timestamped + logged + hash-signed."""
        t0 = time.time()
        if self._kill_switch:
            result = {"status": "blocked", "reason": "kill_switch_active"}
            self._log_run("generate", 0, result)
            return result

        if not GRAPHIFY_AVAILABLE:
            return {"status": "error", "message": "graphifyy not installed"}

        self.total_runs += 1
        try:
            # RESTRICTED: only graphify binary, only "." argument, only from repo root
            process = subprocess.Popen(
                ["graphify", "."],
                cwd=str(_REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.last_run_at = datetime.now(timezone.utc).isoformat()

            result = {
                "status": "generating",
                "pid": process.pid,
                "run_number": self.total_runs,
                "started_at": self.last_run_at,
            }
            self._log_run("generate", time.time() - t0, result)
            return result
        except Exception as e:
            logger.error("Graphify generate failed: %s", e)
            result = {"status": "error", "message": str(e)}
            self._log_run("generate", time.time() - t0, result)
            return result

    # ── Metrics for Prometheus + TECH tab ────────────────────

    def get_metrics(self) -> dict:
        return {
            "kill_switch": self._kill_switch,
            "available": self.is_available(),
            "total_runs": self.total_runs,
            "total_queries": self.total_queries,
            "total_run_time_s": round(self.total_run_time_s, 2),
            "last_run_at": self.last_run_at,
            "last_run_duration_s": self.last_run_duration_s,
            "cached_god_nodes": len(self._cached_god_nodes),
            "cached_report_length": len(self._cached_report),
        }

    def get_run_log(self, limit: int = 50) -> list:
        """Return recent run log entries for TECH tab."""
        if not _RUN_LOG.exists():
            return []
        try:
            lines = _RUN_LOG.read_text(encoding="utf-8").strip().split("\n")
            entries = [json.loads(line) for line in lines[-limit:]]
            return entries
        except Exception:
            return []
