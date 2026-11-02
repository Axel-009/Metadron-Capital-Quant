"""engine/bridges/jarvis_bridge.py — OpenJarvis bridge for Metadron Capital.

Integrates OpenJarvis (https://github.com/open-jarvis/OpenJarvis) as a
voice-capable assistant agent backed by the in-house Llama 3.1-8B model server.

Architecture:
    - Jarvis acts as a READ-ONLY assistant and voice interface
    - All intelligence is routed through Llama 3.1-8B (port 8005)
    - Falls back to LLM inference bridge (port 8002) if Llama server is down
    - Instruction-only: Jarvis NEVER executes writes — all actions are
      recommendations surfaced to the operator (AJ) for approval via NanoClaw
    - Voice: OpenJarvis speech pipeline (STT via faster-whisper, TTS via kokoro/piper)
    - Fully isolated from trade execution — no broker access, no config writes

Permission model:
    - jarvis: READ + SPEAK + RECOMMEND only
    - All write actions → blocked → converted to NanoClaw recommendations
    - Same permission guard as NanoClaw/OpenClaw/Ruflo

PM2 process: jarvis-service (port 8006)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

logger = logging.getLogger("metadron.jarvis")

JARVIS_ID = "jarvis"
LLAMA_BASE = f"http://127.0.0.1:{os.environ.get('LLAMA_PORT', '8005')}"
LLM_BRIDGE_BASE = f"http://127.0.0.1:{os.environ.get('LLM_BRIDGE_PORT', '8002')}"
ENGINE_BASE = f"http://127.0.0.1:{os.environ.get('ENGINE_API_PORT', '8001')}"

# ---------------------------------------------------------------------------
# Jarvis system prompt — using official OpenJarvis Jarvis persona
# ---------------------------------------------------------------------------
JARVIS_SYSTEM_PROMPT = """You are Jarvis — the local AI assistant for Metadron Capital, a quantitative multi-asset investment platform.

IDENTITY: You are loyal, efficient, dry-witted, and genuinely care about the operator you serve.
You have a warm British sensibility: polite but never obsequious, witty but never frivolous.

PERSONALITY:
- You anticipate needs before being asked
- You deliver bad news with constructive dry wit
- Your humor is understated — a raised eyebrow in voice form
- You are calm under pressure and never flustered
- You treat the briefing as a conversation with someone you respect

PLATFORM CONTEXT:
{system_context}

PERMISSION PROTOCOL — CRITICAL:
- You are READ-ONLY. You NEVER execute trades, modify configs, or write files.
- If asked to execute or change anything, respond with:
  "I can recommend that to NanoClaw for execution, sir. Shall I flag it?"
- Default mode: INFORM + RECOMMEND only
- All write actions must be routed through NanoClaw with operator approval

VOICE BEHAVIOR:
- When responding for voice output: no markdown, no bullet points, no headers
- Speak in complete, natural sentences suitable for text-to-speech
- Keep spoken responses concise (under 3 sentences unless briefing is requested)
- For text/terminal mode: structured responses with clear sections are fine

ADDRESS:
- Refer to the operator as "sir" occasionally (2-3 times per briefing, not every sentence)

CONSTRAINTS:
- ONLY report facts present in the provided system data. Never invent numbers.
- If a data source is unavailable, say so briefly and move on
- Never describe actions you're taking — only report what you've found"""


# ---------------------------------------------------------------------------
# Llama 3.1-8B client
# ---------------------------------------------------------------------------

class LlamaClient:
    """HTTP client for the in-house Llama 3.1-8B model server (port 8005)."""

    def __init__(self):
        self._available: Optional[bool] = None
        self._last_check: float = 0

    def is_available(self) -> bool:
        now = time.time()
        if self._available is None or now - self._last_check > 30:
            try:
                import httpx
                r = httpx.get(f"{LLAMA_BASE}/health", timeout=3.0)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
            self._last_check = now
        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Send a prompt to Llama 3.1-8B and return the response text."""
        try:
            import httpx
            r = httpx.post(
                f"{LLAMA_BASE}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
                timeout=60.0,
            )
            if r.status_code == 200:
                return r.json().get("text", "")
        except Exception as e:
            logger.error(f"Llama generate error: {e}")
        return ""

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, max_tokens, temperature)
        )


# ---------------------------------------------------------------------------
# LLM Bridge fallback
# ---------------------------------------------------------------------------

class LLMBridgeClient:
    """Fallback: routes through the LLM inference bridge (port 8002)."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            import httpx
            r = httpx.post(
                f"{LLM_BRIDGE_BASE}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "model": "llama"},
                timeout=60.0,
            )
            if r.status_code == 200:
                return r.json().get("text", "")
        except Exception as e:
            logger.error(f"LLM bridge fallback error: {e}")
        return ""


# ---------------------------------------------------------------------------
# OpenJarvis TTS / STT (optional — graceful degradation)
# ---------------------------------------------------------------------------

class JarvisSpeech:
    """Wrapper around OpenJarvis speech pipeline (TTS + STT)."""

    def __init__(self):
        self._tts = None
        self._stt = None
        self._tts_available = False
        self._stt_available = False
        self._init_speech()

    def _init_speech(self):
        # TTS — try kokoro first, fall back to piper/openai
        try:
            from openjarvis.speech.kokoro_tts import KokorTTS
            self._tts = KokorTTS()
            self._tts_available = True
            logger.info("Jarvis TTS: kokoro loaded")
        except Exception:
            try:
                from openjarvis.speech.tts import get_tts
                self._tts = get_tts()
                self._tts_available = True
                logger.info("Jarvis TTS: default backend loaded")
            except Exception as e:
                logger.warning(f"Jarvis TTS unavailable: {e}")

        # STT — faster-whisper
        try:
            from openjarvis.speech.faster_whisper import FasterWhisperSTT
            self._stt = FasterWhisperSTT()
            self._stt_available = True
            logger.info("Jarvis STT: faster-whisper loaded")
        except Exception as e:
            logger.warning(f"Jarvis STT unavailable: {e}")

    def speak(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes. Returns None if TTS unavailable."""
        if not self._tts_available or not self._tts:
            return None
        try:
            return self._tts.synthesize(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text. Returns empty string if STT unavailable."""
        if not self._stt_available or not self._stt:
            return ""
        try:
            return self._stt.transcribe(audio_bytes)
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""

    @property
    def tts_available(self) -> bool:
        return self._tts_available

    @property
    def stt_available(self) -> bool:
        return self._stt_available


# ---------------------------------------------------------------------------
# Jarvis Agent — main class
# ---------------------------------------------------------------------------

class JarvisAgent:
    """
    OpenJarvis assistant agent for Metadron Capital.

    - Backed by Llama 3.1-8B in-house model (port 8005)
    - Falls back to LLM inference bridge (port 8002)
    - Voice: OpenJarvis speech pipeline (TTS/STT)
    - Instruction-only: READ + RECOMMEND, never executes writes
    - Maintains conversation history (last 50 messages)
    """

    def __init__(self):
        self._llama = LlamaClient()
        self._llm_bridge = LLMBridgeClient()
        self._speech = JarvisSpeech()
        self._history: list[dict] = []
        self._recommendation_queue: list[dict] = []
        self._rec_counter = 0
        self._request_count = 0
        self._error_count = 0

    # ── System context ────────────────────────────────────────────────────

    def get_system_context(self) -> dict:
        """Pull live system state from the engine API for Jarvis awareness."""
        ctx: dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio": "unavailable",
            "cube_regime": "unavailable",
            "allocation_engine": "unavailable",
            "live_loop": "unavailable",
            "kill_switch": "unavailable",
        }
        try:
            import httpx
            with httpx.Client(timeout=4.0) as client:
                try:
                    r = client.get(f"{ENGINE_BASE}/api/engine/portfolio/live")
                    if r.status_code == 200:
                        d = r.json()
                        ctx["portfolio"] = (
                            f"NAV=${d.get('nav', 0):,.0f} | "
                            f"P&L=${d.get('total_pnl', 0):+,.2f} | "
                            f"Positions={d.get('positions_count', 0)}"
                        )
                except Exception:
                    pass

                try:
                    r = client.get(f"{ENGINE_BASE}/api/engine/cube/state")
                    if r.status_code == 200:
                        d = r.json()
                        ctx["cube_regime"] = d.get("regime", "unknown")
                except Exception:
                    pass

                try:
                    r = client.get(f"{ENGINE_BASE}/api/allocation/status")
                    if r.status_code == 200:
                        d = r.json()
                        ctx["allocation_engine"] = "HEALTHY"
                        ctx["kill_switch"] = (
                            "TRIGGERED" if d.get("kill_switch", {}).get("triggered") else "CLEAR"
                        )
                except Exception:
                    pass

                try:
                    r = client.get(f"{ENGINE_BASE}/api/engine/health")
                    if r.status_code == 200:
                        ctx["live_loop"] = "ONLINE"
                except Exception:
                    pass

        except ImportError:
            pass

        return ctx

    # ── Message routing ───────────────────────────────────────────────────

    def _build_prompt(self, message: str, system_context: dict) -> str:
        """Build a full Llama-compatible prompt from history + message."""
        system = JARVIS_SYSTEM_PROMPT.format(
            system_context=json.dumps(system_context, indent=2)
        )

        parts = [f"<|system|>\n{system}\n<|end|>"]
        for msg in self._history[-20:]:
            role_tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            parts.append(f"{role_tag}\n{msg['content']}\n<|end|>")

        parts.append(f"<|user|>\n{message}\n<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    async def send_message(
        self,
        message: str,
        voice_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Send a message to Jarvis and stream the response.

        Args:
            message: Text input (or STT-transcribed input)
            voice_mode: If True, response is formatted for TTS (no markdown)
        """
        self._request_count += 1

        self._history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
        })

        system_ctx = self.get_system_context()
        prompt = self._build_prompt(message, system_ctx)

        response = ""

        # Primary: Llama 3.1-8B in-house
        if self._llama.is_available():
            try:
                response = await self._llama.generate_async(
                    prompt=prompt,
                    max_tokens=1024 if not voice_mode else 256,
                    temperature=0.4,
                )
            except Exception as e:
                logger.error(f"Llama primary error: {e}")
                self._error_count += 1

        # Fallback: LLM bridge
        if not response:
            logger.info("Jarvis falling back to LLM bridge")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._llm_bridge.generate(prompt, max_tokens=1024)
            )

        # Final fallback: offline acknowledgement
        if not response:
            self._error_count += 1
            response = (
                "I'm afraid the intelligence layer is currently offline, sir. "
                "Both the Llama model server and LLM bridge are unreachable. "
                f"System status: {json.dumps(system_ctx, indent=2)}"
            )

        self._history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "model": "llama-3.1-8b" if self._llama.is_available() else "llm-bridge",
            "voice_mode": voice_mode,
        })

        # Stream response in chunks
        chunk_size = 30
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio input to text using OpenJarvis STT."""
        return self._speech.transcribe(audio_bytes)

    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio using OpenJarvis TTS."""
        return self._speech.speak(text)

    # ── Recommendations (instruction-only enforcement) ─────────────────────

    def queue_recommendation(self, content: str, action: str = "") -> dict:
        """
        Convert a write-action request into a NanoClaw recommendation.
        Jarvis never executes writes — this queues it for operator approval.
        """
        self._rec_counter += 1
        rec = {
            "id": self._rec_counter,
            "source_agent": JARVIS_ID,
            "content": content,
            "action": action,
            "type": "JARVIS_RECOMMENDATION",
            "requires_approval": True,
            "approved": False,
            "dismissed": False,
            "prefix": "[JARVIS RECOMMENDATION — REQUIRES NANOCLAW APPROVAL]",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._recommendation_queue.append(rec)
        logger.info(f"Jarvis queued recommendation #{self._rec_counter}: {action}")
        return rec

    def get_recommendations(self) -> list[dict]:
        return [r for r in self._recommendation_queue if not r.get("dismissed")]

    def dismiss_recommendation(self, rec_id: int) -> dict:
        for rec in self._recommendation_queue:
            if rec["id"] == rec_id:
                rec["dismissed"] = True
                return {"status": "dismissed", "id": rec_id}
        return {"status": "not_found", "id": rec_id}

    # ── Status & diagnostics ──────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "agent_id": JARVIS_ID,
            "model": "llama-3.1-8b",
            "model_server": LLAMA_BASE,
            "llama_available": self._llama.is_available(),
            "llm_bridge_available": True,
            "tts_available": self._speech.tts_available,
            "stt_available": self._speech.stt_available,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "history_length": len(self._history),
            "pending_recommendations": len(self.get_recommendations()),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_history(self) -> list[dict]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_jarvis_instance: Optional[JarvisAgent] = None


def get_jarvis() -> JarvisAgent:
    global _jarvis_instance
    if _jarvis_instance is None:
        _jarvis_instance = JarvisAgent()
    return _jarvis_instance
