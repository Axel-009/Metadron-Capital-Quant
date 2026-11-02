"""Agent identity configuration — NanoClaw, OpenClaw/CEO, Ruflo."""

# ─── Agent Identity Constants ──────────────────────────────────────
NANOCLAW_ID = "nanoclaw"
OPENCLAW_ID = "openclaw"
RUFLO_ID = "ruflo"

# ─── Display Metadata ─────────────────────────────────────────────
AGENT_DISPLAY = {
    NANOCLAW_ID: {
        "name": "NanoClaw",
        "display_name": "NANOCLAW OPERATOR",
        "color": "#00e5c0",
        "role": "Main Operator Agent",
        "description": "Governing operator agent with write access (requires explicit instruction from AJ).",
        "permission_summary": "WRITE (with explicit operator instruction)",
    },
    OPENCLAW_ID: {
        "name": "OpenClaw",
        "display_name": "CEO RESEARCH",
        "color": "#fbbf24",
        "role": "CEO Research Agent",
        "description": "Research-only agent. Produces recommendations that require NanoClaw approval.",
        "permission_summary": "READ + RECOMMEND only — all writes blocked",
    },
    RUFLO_ID: {
        "name": "Ruflo",
        "display_name": "RUFLO SWARM",
        "color": "#4facfe",
        "role": "Swarm Task Agent",
        "description": "Task delegation agents. Report progress and data. No write access.",
        "permission_summary": "READ + TASK REPORT only — all writes blocked",
    },
}

ALL_AGENT_IDS = [NANOCLAW_ID, OPENCLAW_ID, RUFLO_ID]


def get_agent_info(agent_id: str) -> dict:
    """Return full identity, permissions, and display metadata for an agent."""
    from engine.agents.nanoclaw.permission_guard import AgentPermissionGuard

    display = AGENT_DISPLAY.get(agent_id)
    if not display:
        return {"error": f"Unknown agent: {agent_id}"}

    guard = AgentPermissionGuard()
    perms = guard.get_agent_permissions(agent_id)

    return {
        "agent_id": agent_id,
        **display,
        **perms,
    }


def get_all_agents_info() -> list[dict]:
    """Return info for all registered agents."""
    return [get_agent_info(aid) for aid in ALL_AGENT_IDS]
