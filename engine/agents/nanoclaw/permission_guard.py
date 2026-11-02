"""Agent Permission Guard — enforces read-only on OpenClaw/CEO and Ruflo agents.

NanoClaw is the ONLY agent with write access, and only with explicit operator instruction.
"""
import logging
from datetime import datetime

logger = logging.getLogger("metadron-api.nanoclaw.permissions")

# ─── Permission Matrix ─────────────────────────────────────────────
AGENT_PERMISSIONS = {
    "nanoclaw": {
        "read_data": True,
        "search": True,
        "analyze": True,
        "recommend": True,
        "write_file": True,
        "execute_trade": True,
        "modify_config": True,
        "update_rules": True,
        "push_code": True,
        "call_l7": True,
        "requires_permission": True,
    },
    "openclaw": {
        "read_data": True,
        "search": True,
        "analyze": True,
        "recommend": True,
        "write_file": False,
        "execute_trade": False,
        "modify_config": False,
        "update_rules": False,
        "push_code": False,
        "call_l7": False,
        "requires_permission": False,
    },
    "ruflo": {
        "read_data": True,
        "search": True,
        "analyze": True,
        "recommend": True,
        "write_file": False,
        "execute_trade": False,
        "modify_config": False,
        "update_rules": False,
        "push_code": False,
        "call_l7": False,
        "requires_permission": False,
    },
}

WRITE_ACTIONS = {"write_file", "execute_trade", "modify_config", "update_rules", "push_code", "call_l7"}


class AgentPermissionGuard:
    """Enforces agent-level permission checks across all Metadron agent interactions."""

    def __init__(self):
        self._audit_log: list[dict] = []

    def check(self, action: str, agent_id: str) -> bool:
        """Check if agent_id is allowed to perform action."""
        perms = AGENT_PERMISSIONS.get(agent_id, {})
        return perms.get(action, False)

    def assert_allowed(self, action: str, agent_id: str):
        """Raise PermissionError if agent_id cannot perform action."""
        if not self.check(action, agent_id):
            self._log_denied(action, agent_id)
            raise PermissionError(
                f"Agent '{agent_id}' is not permitted to perform action '{action}'. "
                f"Only NanoClaw can perform write operations, and only with explicit operator instruction."
            )
        if action in WRITE_ACTIONS:
            self._log_write_attempt(action, agent_id)

    def is_write_action(self, action: str) -> bool:
        return action in WRITE_ACTIONS

    def convert_to_recommendation(self, output: dict, agent_id: str) -> dict:
        """Convert a write action output from a non-NanoClaw agent to a recommendation."""
        # OPENCLAW: READ ONLY — no write access
        output["type"] = "RECOMMENDATION"
        output["requires_approval"] = True
        output["approved"] = False
        output["prefix"] = "[CEO RECOMMENDATION - REQUIRES NANOCLAW APPROVAL]"
        output["source_agent"] = agent_id
        output["timestamp"] = datetime.utcnow().isoformat()
        return output

    def get_agent_permissions(self, agent_id: str) -> dict:
        """Return the permission map for a specific agent."""
        perms = AGENT_PERMISSIONS.get(agent_id, {})
        return {
            "agent_id": agent_id,
            "permissions": perms,
            "can_write": any(perms.get(a, False) for a in WRITE_ACTIONS),
            "requires_permission": perms.get("requires_permission", False),
            "write_actions_blocked": [a for a in WRITE_ACTIONS if not perms.get(a, False)],
            "write_actions_allowed": [a for a in WRITE_ACTIONS if perms.get(a, False)],
        }

    def get_all_permissions(self) -> dict:
        """Return permission maps for all agents."""
        return {aid: self.get_agent_permissions(aid) for aid in AGENT_PERMISSIONS}

    def _log_denied(self, action: str, agent_id: str):
        entry = {
            "event": "permission_denied",
            "agent_id": agent_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._audit_log.append(entry)
        logger.warning(f"PERMISSION DENIED: agent={agent_id} action={action}")

    def _log_write_attempt(self, action: str, agent_id: str):
        entry = {
            "event": "write_attempt",
            "agent_id": agent_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._audit_log.append(entry)
        logger.info(f"WRITE ATTEMPT: agent={agent_id} action={action}")

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)
