"""engine/api/routers/jarvis.py — OpenJarvis API router for Metadron Capital.

Endpoints:
    GET  /jarvis/status           — Agent status, model availability, TTS/STT caps
    GET  /jarvis/messages         — Conversation history
    POST /jarvis/chat             — Send a text message, stream response (SSE)
    POST /jarvis/voice            — Upload audio, transcribe → chat → return text + audio
    POST /jarvis/speak            — Convert text to speech (TTS only)
    DELETE /jarvis/history        — Clear conversation history
    GET  /jarvis/recommendations  — Pending recommendations for NanoClaw review
    POST /jarvis/recommendations/{id}/dismiss — Dismiss a recommendation

All write-action requests from Jarvis are converted to recommendations.
Jarvis is READ + SPEAK + RECOMMEND only — never executes trades or writes.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("metadron-api.jarvis")
router = APIRouter()


def _get_jarvis():
    try:
        from engine.bridges.jarvis_bridge import get_jarvis
        return get_jarvis()
    except Exception as e:
        logger.error(f"Jarvis bridge unavailable: {e}")
        return None


# ── Request / Response models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    voice_mode: bool = False      # True = TTS-optimised response (no markdown)
    operator_id: str = "aj"


class SpeakRequest(BaseModel):
    text: str


class RecommendationRequest(BaseModel):
    content: str
    action: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/status")
async def jarvis_status():
    """Return Jarvis agent status — model availability, TTS/STT, request counts."""
    jarvis = _get_jarvis()
    if not jarvis:
        return {
            "agent_id": "jarvis",
            "status": "offline",
            "reason": "Jarvis bridge failed to initialise",
            "timestamp": datetime.utcnow().isoformat(),
        }
    status = jarvis.get_status()
    status["status"] = "online" if status.get("llama_available") else "degraded"
    return status


@router.get("/messages")
async def get_messages():
    """Return full conversation history with Jarvis."""
    jarvis = _get_jarvis()
    if not jarvis:
        return {"messages": [], "error": "Jarvis offline"}
    return {
        "messages": jarvis.get_history(),
        "count": len(jarvis.get_history()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a message to Jarvis and stream the response via SSE.

    Jarvis is backed by Llama 3.1-8B (port 8005) with LLM bridge fallback.
    Instruction-only: any write-action requests are returned as recommendations.
    """
    jarvis = _get_jarvis()
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis bridge offline")

    async def event_stream():
        try:
            async for chunk in jarvis.send_message(
                message=req.message,
                voice_mode=req.voice_mode,
            ):
                # SSE format
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Jarvis chat stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/voice")
async def voice_chat(audio: UploadFile = File(...)):
    """
    Voice endpoint: upload audio → STT → Jarvis chat → return transcript + response text.

    Requires OpenJarvis speech pipeline (faster-whisper for STT).
    Returns JSON with transcript, response text, and optional TTS audio (base64).
    """
    jarvis = _get_jarvis()
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis bridge offline")

    # Read audio bytes
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # STT: transcribe audio → text
    transcript = jarvis.transcribe_audio(audio_bytes)
    if not transcript:
        return {
            "transcript": "",
            "response": "I'm sorry, sir — I couldn't make out the audio. Could you try again?",
            "tts_audio": None,
            "stt_available": jarvis._speech.stt_available,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Chat: get Jarvis response (voice mode = no markdown)
    response_text = ""
    async for chunk in jarvis.send_message(message=transcript, voice_mode=True):
        response_text += chunk

    # TTS: convert response to speech
    import base64
    tts_audio = None
    audio_b64 = None
    tts_bytes = jarvis.synthesize_speech(response_text)
    if tts_bytes:
        audio_b64 = base64.b64encode(tts_bytes).decode("utf-8")

    return {
        "transcript": transcript,
        "response": response_text,
        "tts_audio_b64": audio_b64,
        "tts_available": jarvis._speech.tts_available,
        "stt_available": jarvis._speech.stt_available,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/speak")
async def speak(req: SpeakRequest):
    """
    TTS-only endpoint: convert text to speech audio.
    Returns audio/wav bytes or 503 if TTS is unavailable.
    """
    jarvis = _get_jarvis()
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis bridge offline")

    if not jarvis._speech.tts_available:
        raise HTTPException(
            status_code=503,
            detail="TTS unavailable — install openjarvis speech: pip install openjarvis[speech]",
        )

    audio_bytes = jarvis.synthesize_speech(req.text)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    return Response(content=audio_bytes, media_type="audio/wav")


@router.delete("/history")
async def clear_history():
    """Clear Jarvis conversation history."""
    jarvis = _get_jarvis()
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis bridge offline")
    jarvis.clear_history()
    return {"status": "cleared", "timestamp": datetime.utcnow().isoformat()}


@router.get("/recommendations")
async def get_recommendations():
    """Return pending Jarvis recommendations queued for NanoClaw review."""
    jarvis = _get_jarvis()
    if not jarvis:
        return {"recommendations": [], "error": "Jarvis offline"}
    recs = jarvis.get_recommendations()
    return {
        "recommendations": recs,
        "count": len(recs),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/recommendations/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int):
    """Dismiss a Jarvis recommendation."""
    jarvis = _get_jarvis()
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis bridge offline")
    result = jarvis.dismiss_recommendation(rec_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")
    return result
