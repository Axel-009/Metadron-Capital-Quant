"""
Metadron Capital — Qwen 2.5-7B Standalone Model Server

Provides a dedicated FastAPI service for Qwen 2.5-7B-Instruct inference on port 8004.
Uses standard HuggingFace transformers (AutoModelForCausalLM + AutoTokenizer).

Runs on cuda:0 (GEX44 single GPU).

The LLM Inference Bridge imports QwenModelManager in-process for
parallel ensemble execution.

Usage:
    python3 -m uvicorn engine.bridges.qwen_model_server:create_app \
        --factory --host 0.0.0.0 --port 8004 --log-level info
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("qwen-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Transformers Availability ──────────────────────────────────────

_transformers_available = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _transformers_available = True
    logger.info("Transformers library: AVAILABLE")
except ImportError:
    logger.warning("Transformers/torch not found — server will start in degraded mode")


# ─── Model Manager ─────────────────────────────────────────────────

class QwenModelManager:
    """Manages lazy-loading and inference for Qwen 2.5-7B-Instruct on cuda:1."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: str = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        self.device: str = os.environ.get("QWEN_DEVICE", "cuda:1")
        self.loaded: bool = False
        self.load_time_s: float = 0.0
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def load(self):
        """Lazy-load the Qwen model on first request."""
        if self.loaded:
            return

        if not _transformers_available:
            raise RuntimeError(
                "Transformers/torch not installed. "
                "Install with: pip install transformers torch"
            )

        logger.info(f"Loading Qwen model: {self.model_id} on {self.device} ...")
        start = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.load_time_s = round(time.time() - start, 2)
            self.loaded = True
            logger.info(f"Qwen model loaded in {self.load_time_s}s on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Run text generation inference.

        Returns:
            {"text": str, "tokens_generated": int, "latency_ms": float}
        """
        self.load()

        start = time.time()
        self.request_count += 1

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            kwargs = {"max_new_tokens": max_tokens}
            if temperature != 1.0:
                kwargs["temperature"] = temperature
                kwargs["do_sample"] = True

            with torch.no_grad():
                output = self.model.generate(**inputs, **kwargs)

            # Decode only the new tokens (skip the input)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            latency_ms = (time.time() - start) * 1000
            self.total_latency_ms += latency_ms

            return {
                "text": text.strip(),
                "tokens_generated": len(new_tokens),
                "latency_ms": round(latency_ms, 1),
            }

        except Exception as e:
            self.error_count += 1
            latency_ms = (time.time() - start) * 1000
            self.total_latency_ms += latency_ms
            logger.error(f"Generation error: {e}")
            raise

    def get_info(self) -> dict:
        """Return model metadata and stats."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "loaded": self.loaded,
            "transformers_available": _transformers_available,
            "load_time_s": self.load_time_s,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ─── FastAPI Service (PM2 managed) ─────────────────────────────────

def create_app():
    """Create FastAPI app for the Qwen model server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Metadron Qwen 2.5-7B Model Server",
        description="Standalone Qwen 2.5-7B-Instruct inference on cuda:1",
        version="1.0.0",
    )

    # CORS restricted to LLM bridge + engine API — set QWEN_CORS_ORIGINS to your public IP at deploy
    _cors_origins = os.environ.get(
        "QWEN_CORS_ORIGINS",
        "http://localhost:8002,http://localhost:8001,http://127.0.0.1:8002,http://127.0.0.1:8001",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    manager = QwenModelManager()

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if _transformers_available else "degraded",
            "service": "qwen-model-server",
            "model_loaded": manager.loaded,
            "model_id": manager.model_id,
            "device": manager.device,
            "transformers_available": _transformers_available,
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        if not _transformers_available:
            raise HTTPException(
                status_code=503,
                detail="Transformers/torch not available. Install with: pip install transformers torch",
            )

        try:
            import asyncio
            result = await asyncio.to_thread(
                manager.generate,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model-info")
    async def model_info():
        return manager.get_info()

    return app


# ─── Entry Point ───────────────────────────────────────────────────

def main():
    """Run the Qwen model server as a PM2-managed service."""
    import uvicorn

    port = int(os.environ.get("QWEN_PORT", "8004"))
    host = os.environ.get("QWEN_HOST", "0.0.0.0")

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting Qwen 2.5-7B Model Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
