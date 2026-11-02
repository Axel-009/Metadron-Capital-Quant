"""Metadron Capital — Llama 3.1-8B Model Server.

Dedicated FastAPI inference server for Meta Llama 3.1-8B-Instruct.
Runs as a PM2-managed service on port 11434, same pattern as
Qwen (7860) and Air-LLM (8003).

Serves as a fast router/classifier model in the intelligence layer.
The LLM Inference Bridge imports LlamaModelManager in-process for
parallel ensemble inference alongside Qwen and Air-LLM.

PM2 manages this as: llama-model-server (port 11434, cuda:0)

Usage:
    python3 engine/bridges/llama_model_server.py
    # or via PM2:
    pm2 start ecosystem.config.cjs --only llama-model-server
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

logger = logging.getLogger("llama-model-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Ensure project root on path
PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))


class LlamaModelManager:
    """Manages lazy-loading and inference for Llama 3.1-8B-Instruct."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: str = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
        self.device: str = os.environ.get("LLAMA_DEVICE", "cuda:0")
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

    def _load_model(self):
        """Lazy-load the model on first inference request."""
        if self.loaded:
            return
        start = time.time()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.loaded = True
            self.load_time_s = time.time() - start
            logger.info(
                f"Llama 3.1-8B loaded in {self.load_time_s:.1f}s on {self.device}"
            )
        except Exception as e:
            self.load_time_s = time.time() - start
            logger.error(f"Failed to load Llama model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Generate text from the Llama model."""
        self._load_model()
        start = time.time()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0.01,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            latency = time.time() - start
            self.request_count += 1
            self.total_latency_ms += latency * 1000
            return {
                "text": text,
                "tokens_generated": len(response_ids),
                "latency_ms": round(latency * 1000, 1),
                "model": self.model_id,
            }
        except Exception as e:
            latency = time.time() - start
            self.error_count += 1
            self.total_latency_ms += latency * 1000
            logger.error(f"Llama generation error: {e}")
            return {"text": "", "error": str(e), "model": self.model_id}


def create_app():
    """Create FastAPI app for the Llama model server."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise RuntimeError("FastAPI required: pip install fastapi[all]")

    app = FastAPI(
        title="Metadron Llama 3.1-8B Model Server",
        description="Dedicated inference server for Llama 3.1-8B-Instruct (router/classifier)",
    )

    # CORS restricted to LLM bridge + engine API
    _cors_origins = os.environ.get(
        "LLAMA_CORS_ORIGINS",
        "http://localhost:8002,http://localhost:8001,http://127.0.0.1:8002,http://127.0.0.1:8001",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    manager = LlamaModelManager()

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": manager.model_id,
            "device": manager.device,
            "loaded": manager.loaded,
            "load_time_s": round(manager.load_time_s, 1),
            "request_count": manager.request_count,
            "error_count": manager.error_count,
            "avg_latency_ms": round(manager.avg_latency_ms, 1),
            "error_rate": round(manager.error_rate, 3),
        }

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: manager.generate(req.prompt, req.max_tokens, req.temperature),
        )
        return result

    @app.get("/model-info")
    async def model_info():
        return {
            "model_id": manager.model_id,
            "device": manager.device,
            "loaded": manager.loaded,
            "request_count": manager.request_count,
            "error_rate": round(manager.error_rate, 3),
        }

    @app.get("/cuda-health")
    async def cuda_health():
        """CUDA memory health check — detect leaks, report VRAM usage."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"cuda_available": False}
            idx = int(manager.device.split(":")[-1]) if ":" in manager.device else 0
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            total = torch.cuda.get_device_properties(idx).total_mem
            return {
                "cuda_available": True,
                "device": manager.device,
                "allocated_mb": round(allocated / 1024**2, 1),
                "reserved_mb": round(reserved / 1024**2, 1),
                "total_mb": round(total / 1024**2, 1),
                "utilization_pct": round(allocated / total * 100, 1) if total > 0 else 0,
                "leak_risk": allocated > 0.90 * total,
            }
        except Exception as e:
            return {"cuda_available": False, "error": str(e)}

    return app


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LLAMA_PORT", "11434"))
    host = os.environ.get("LLAMA_HOST", "0.0.0.0")

    def handle_signal(signum, frame):
        logger.info("Llama model server shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting Llama 3.1-8B Model Server on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
