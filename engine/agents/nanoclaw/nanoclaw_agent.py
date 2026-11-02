"""NanoClaw Agent — main operator agent for Metadron Capital.

Wraps the Brain Power bridge (Xiaomi Mimo V2 Pro) to provide a
streaming conversational agent with system context awareness and
permission-gated write actions.
"""
import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator

logger = logging.getLogger("metadron-api.nanoclaw.agent")

try:
    from engine.bridges.brain_power import BrainPowerClient
    _brain_power_available = True
except ImportError:
    _brain_power_available = False
    logger.warning("BrainPowerClient not available — NanoClaw will use mock responses")

from engine.agents.nanoclaw.permission_guard import AgentPermissionGuard, WRITE_ACTIONS
from engine.agents.nanoclaw.agent_config import NANOCLAW_ID, OPENCLAW_ID

NANOCLAW_SYSTEM_PROMPT = """You are NanoClaw, the main operator agent for Metadron Capital — a quantitative multi-asset investment platform.

IDENTITY: You are the governing operator agent. You have write access but ONLY when explicitly instructed by the operator (AJ).

YOUR ROLE:
- Debug system issues across all engines (allocation engine, universe scan, L7 execution, fixed income engine, ML models)
- Execute operational instructions from AJ
- Monitor system health
- Coordinate Ruflo agents
- Evaluate and flag CEO (OpenClaw) recommendations for AJ's review — never auto-execute them

SYSTEM CONTEXT:
{system_context}

PERMISSION PROTOCOL:
- Default mode: READ + RECOMMEND
- To execute a write action, AJ must explicitly say so (e.g., "go ahead", "execute", "apply this", "write this")
- Always confirm before writing: "I'll [action]. Shall I proceed?" unless AJ has already confirmed

OPENCLAW/CEO AGENT:
- Read-only research agent
- Its recommendations come to you as flagged items
- Present them to AJ, do not auto-execute
- Format: [CEO RECOMMENDATION - REQUIRES APPROVAL] {{recommendation}}

RUFLO AGENTS:
- Swarm agents you can delegate to
- They report back with status and data
- They cannot execute writes"""


class NanoClawAgent:
    """NanoClaw operator agent with streaming responses and permission-gated actions."""

    def __init__(self):
        self._guard = AgentPermissionGuard()
        self._message_history: list[dict] = []
        self._recommendations: list[dict] = []
        self._recommendation_counter = 0

        if _brain_power_available:
            self._client = BrainPowerClient()
        else:
            self._client = None

    def get_system_context(self) -> dict:
        """Read current system state to give NanoClaw awareness."""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "allocation_engine": "unknown",
            "scan_status": "unknown",
            "l7_status": "unknown",
            "kill_switch": "unknown",
            "beta_corridor": "unknown",
        }

        try:
            import httpx
            base = f"http://127.0.0.1:{os.environ.get('ENGINE_API_PORT', '8001')}"
            with httpx.Client(timeout=5.0) as client:
                try:
                    r = client.get(f"{base}/api/allocation/status")
                    if r.status_code == 200:
                        data = r.json()
                        context["allocation_engine"] = "HEALTHY"
                        context["kill_switch"] = "TRIGGERED" if data.get("kill_switch", {}).get("triggered") else "CLEAR"
                        bc = data.get("beta_corridor", {})
                        context["beta_corridor"] = f"{bc.get('corridor', 'N/A')} (beta={bc.get('beta', 'N/A')}, leverage={bc.get('leverage_multiplier', 'N/A')}x)"
                except Exception:
                    context["allocation_engine"] = "UNREACHABLE"

                try:
                    r = client.get(f"{base}/api/allocation/scan/status")
                    if r.status_code == 200:
                        data = r.json()
                        context["scan_status"] = f"Cycle {data.get('cycle_number', '?')} — {data.get('phase', 'UNKNOWN')}"
                except Exception:
                    context["scan_status"] = "UNREACHABLE"

                try:
                    r = client.get(f"{base}/api/engine/health")
                    if r.status_code == 200:
                        context["l7_status"] = "HEALTHY"
                except Exception:
                    context["l7_status"] = "UNREACHABLE"
        except ImportError:
            pass

        return context

    async def send_message(self, message: str, operator_id: str = "aj") -> AsyncGenerator[str, None]:
        """Send a message to NanoClaw and stream the response token by token."""
        self._message_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": operator_id,
        })

        system_context = self.get_system_context()
        system_prompt = NANOCLAW_SYSTEM_PROMPT.format(
            system_context=json.dumps(system_context, indent=2),
        )

        messages = []
        for msg in self._message_history[-50:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if self._client and _brain_power_available:
            try:
                full_response = ""
                for chunk in self._client.stream_chat(
                    messages=messages,
                    system=system_prompt,
                    max_tokens=4096,
                    temperature=0.3,
                ):
                    full_response += chunk
                    yield chunk

                self._message_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": NANOCLAW_ID,
                })
            except Exception as e:
                error_msg = f"[NanoClaw Error] {str(e)}"
                logger.error(f"NanoClaw Brain Power error: {e}")
                self._message_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": NANOCLAW_ID,
                })
                yield error_msg
        else:
            mock = (
                f"NanoClaw online. Received: \"{message}\"\n\n"
                f"System Status:\n"
                f"  Allocation Engine: {system_context.get('allocation_engine', 'N/A')}\n"
                f"  Scan: {system_context.get('scan_status', 'N/A')}\n"
                f"  L7 Surface: {system_context.get('l7_status', 'N/A')}\n"
                f"  Kill Switch: {system_context.get('kill_switch', 'N/A')}\n"
                f"  Beta Corridor: {system_context.get('beta_corridor', 'N/A')}\n\n"
                f"Awaiting instructions."
            )
            self._message_history.append({
                "role": "assistant",
                "content": mock,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": NANOCLAW_ID,
            })
            for chunk in [mock[i:i + 20] for i in range(0, len(mock), 20)]:
                yield chunk

    def execute_action(self, action: str, params: dict, permission_granted: bool = False) -> dict:
        """Execute a write action only if permission_granted=True."""
        if not permission_granted:
            return {
                "status": "blocked",
                "reason": "Write action requires explicit operator permission.",
                "action": action,
                "params": params,
            }
        self._guard.assert_allowed(action, NANOCLAW_ID)
        return {
            "status": "executed",
            "action": action,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": NANOCLAW_ID,
        }

    def add_recommendation(self, recommendation: dict, source_agent: str = OPENCLAW_ID) -> dict:
        """Add a CEO recommendation to the pending queue."""
        # OPENCLAW: READ ONLY — no write access
        self._recommendation_counter += 1
        rec = {
            "id": self._recommendation_counter,
            "source_agent": source_agent,
            "content": recommendation.get("content", ""),
            "type": "RECOMMENDATION",
            "requires_approval": True,
            "approved": False,
            "dismissed": False,
            "prefix": "[CEO RECOMMENDATION - REQUIRES NANOCLAW APPROVAL]",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._recommendations.append(rec)
        return rec

    def approve_recommendation(self, rec_id: int) -> dict:
        for rec in self._recommendations:
            if rec["id"] == rec_id and not rec["dismissed"]:
                rec["approved"] = True
                rec["approved_at"] = datetime.utcnow().isoformat()
                return {"status": "approved", "recommendation": rec}
        return {"status": "not_found", "id": rec_id}

    def dismiss_recommendation(self, rec_id: int) -> dict:
        for rec in self._recommendations:
            if rec["id"] == rec_id:
                rec["dismissed"] = True
                rec["dismissed_at"] = datetime.utcnow().isoformat()
                return {"status": "dismissed", "recommendation": rec}
        return {"status": "not_found", "id": rec_id}

    def get_pending_recommendations(self) -> list[dict]:
        return [r for r in self._recommendations if not r.get("dismissed", False)]

    def get_history(self) -> list[dict]:
        return list(self._message_history)
