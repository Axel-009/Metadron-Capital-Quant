"""One helper for every agent's Claude call (Anthropic Python SDK).

    model        claude-opus-5-5 (design rule 6), adaptive thinking (always on for this model), effort "high"
    fallbacks    server-side refusal fallback ("default" routing) so a declined request is retried on a
                 fallback model inside the same call
    web search   optional server tool (web_search_20260209) for the events agent — pause_turn is resumed
    output       plain text; callers extract and validate JSON themselves (never trusted blindly)
Returns None when the SDK or ANTHROPIC_API_KEY is missing, on API errors, or on a refusal.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

MODEL = os.environ.get("METADRON_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def ask(system: str, prompt: str, web_search: bool = False, effort: str = "high",
        max_tokens: int = 16000, max_searches: int = 8) -> Optional[str]:
    try:
        import anthropic
    except Exception:
        print("[llm] anthropic SDK missing", file=sys.stderr)
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    client = anthropic.Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}] if web_search else []
    messages = [{"role": "user", "content": prompt}]
    text = []
    try:
        for _ in range(6):                                   # resume server-tool turns (pause_turn)
            kwargs = dict(model=MODEL, max_tokens=max_tokens, system=system, messages=messages,
                          output_config={"effort": effort}, betas=[FALLBACK_BETA], fallbacks="default")
            if tools:
                kwargs["tools"] = tools
            resp = client.beta.messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                print(f"[llm] refusal: {getattr(resp, 'stop_details', None)}", file=sys.stderr)
                return None
            text += [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            if resp.stop_reason != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": resp.content}]
    except anthropic.RateLimitError as exc:
        print(f"[llm] rate limited: {exc}", file=sys.stderr)
        return None
    except anthropic.APIStatusError as exc:
        print(f"[llm] API error {exc.status_code}: {exc.message}", file=sys.stderr)
        return None
    except anthropic.APIConnectionError as exc:
        print(f"[llm] connection error: {exc}", file=sys.stderr)
        return None
    return "".join(text) or None


def extract_json(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None
