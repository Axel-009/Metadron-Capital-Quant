"""
MCP (Model Context Protocol) plugin configuration for Metadron agents.
These plugins enhance agent reasoning and documentation access.

Sequential Thinking: structured step-by-step problem solving
Context7: version-specific library documentation (no more hallucinated APIs)
"""
from __future__ import annotations

MCP_PLUGINS: dict = {
    "sequential_thinking": {
        "name": "sequential-thinking",
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "install_cmd": (
            "claude mcp add sequential-thinking -- "
            "npx -y @modelcontextprotocol/server-sequential-thinking"
        ),
        "description": "Structured step-by-step reasoning for complex financial decisions",
        "use_cases": [
            "Multi-step trade analysis",
            "Portfolio rebalancing decisions",
            "Risk cascade evaluation",
            "Signal pipeline debugging",
        ],
        "note": "Install once on your local machine — not a server-side dependency",
    },
    "context7": {
        "name": "context7",
        "package": "@upstash/context7-mcp",
        "install_cmd": (
            "claude mcp add --scope user context7 -- "
            "npx -y @upstash/context7-mcp"
        ),
        "description": "Version-specific library docs for AI agents (IBKR, FastAPI, React, etc.)",
        "use_cases": [
            "IBKR trading API reference",
            "FastAPI endpoint documentation",
            "Pandas/NumPy function signatures",
            "React/TypeScript component APIs",
        ],
        "note": "Install once on your local machine — not a server-side dependency",
    },
}


def get_mcp_status() -> dict:
    """Return current MCP plugin configuration and install instructions."""
    return {
        "plugins": MCP_PLUGINS,
        "install_note": (
            "These are client-side Claude Code MCP servers. "
            "Run each install_cmd from your local terminal, not the production server."
        ),
    }
