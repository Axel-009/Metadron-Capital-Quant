"""
Metadron Capital — Unified LLM Inference Bridge (Parallel Ensemble)

All models run simultaneously and constantly in parallel.
Brain Power (Xiaomi Mimo V2 Pro) is the orchestrating intelligence
that receives outputs from all other models and synthesizes, corrects,
or navigates the final decision/output.

Architecture:
    [Llama 3.1-8B]  ─────┐  (Ollama, port 11434, fast router)
    [Qwen 2.5-7B]  ──────┤──► [Brain Power / Xiaomi Mimo] ──► Final Output
    [AI-Newton]    ──────┤       (synthesize / correct / navigate)
    [AlphaOptimizer]─────┤
    [DeepLearning] ──────┘

PM2 manages this as a persistent FastAPI service on port 8002.
"""

import os
import sys
import json
import time
import signal
import logging
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("llm-bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Backend Imports ──────────────────────────────────────────────────

try:
    from engine.bridges.brain_power import BrainPowerClient
    _brain_power_available = True
except ImportError:
    _brain_power_available = False
    logger.warning("BrainPowerClient not available — Brain Power orchestration disabled")

try:
    from engine.bridges.airllm_model_server import AirLLMModelManager
    _airllm_available = True
except ImportError:
    _airllm_available = False
    logger.warning("AirLLMModelManager not available — Air-LLM disabled")

try:
    from engine.bridges.qwen_model_server import QwenModelManager
    _qwen_available = True
except ImportError:
    _qwen_available = False
    logger.warning("QwenModelManager not available — Qwen disabled")

# ─── Prometheus Metrics ──────────────────────────────────────────────

_prom_metrics = None


def _get_prom_metrics():
    """Lazy-load Prometheus metrics to avoid import-time errors."""
    global _prom_metrics
    if _prom_metrics is None:
        try:
            from engine.bridges.prometheus_metrics import get_metrics
            _prom_metrics = get_metrics()
        except Exception:
            _prom_metrics = {}
    return _prom_metrics


# Thread pool for parallel model execution
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-ensemble")


@dataclass
class LLMBackend:
    name: str
    backend_type: str  # "brain_power" | "airllm" | "qwen"
    available: bool = False
    model_id: str = ""
    capabilities: list = field(default_factory=list)
    avg_latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class LLMInferenceBridge:
    """Parallel ensemble — all models run simultaneously.

    Brain Power (Xiaomi Mimo V2 Pro) is the orchestrator that synthesizes
    outputs from Air-LLM and Qwen into a final response. If Brain Power
    API key is not configured, returns merged local model outputs.
    """

    def __init__(self):
        self.backends: dict[str, LLMBackend] = {}
        self._brain_power_client: Optional[BrainPowerClient] = None
        self._airllm_manager: Optional[AirLLMModelManager] = None
        self._qwen_manager: Optional[QwenModelManager] = None
        self._initialize_backends()

    def _initialize_backends(self):
        """Register all backends for parallel ensemble execution."""
        self.backends["brain_power"] = LLMBackend(
            name="Brain Power (Xiaomi Mimo V2 Pro)",
            backend_type="brain_power",
            model_id="xiaomi-mimo-v2-pro",
            capabilities=["text", "reasoning", "code", "analysis", "long_context",
                          "sentiment", "earnings", "sec_filing", "trade_thesis",
                          "narrative", "orchestration", "synthesis"],
        )
        self.backends["airllm"] = LLMBackend(
            name="Llama 3.1-8B (Air-LLM)",
            backend_type="airllm",
            model_id=os.environ.get("AIRLLM_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
            capabilities=["text", "reasoning", "code", "analysis"],
        )
        # Ollama Llama 3.1-8B — fast router/classifier on port 11434
        self.backends["llama"] = LLMBackend(
            name="Llama 3.1-8B-Instruct",
            backend_type="llama",
            model_id=os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
            capabilities=["text", "reasoning", "classification", "routing"],
        )
        self.backends["qwen"] = LLMBackend(
            name="Qwen 2.5-7B-Instruct",
            backend_type="qwen",
            model_id=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
            capabilities=["text", "reasoning", "code", "analysis"],
        )
        self._probe_backends()

    def _probe_backends(self):
        """Check backend availability."""
        metrics = _get_prom_metrics()

        # Brain Power (orchestrator)
        if _brain_power_available:
            self._brain_power_client = BrainPowerClient()
            self.backends["brain_power"].available = True
            if self._brain_power_client.is_stub:
                logger.info("Brain Power orchestrator: STUB MODE (key not configured)")
                if metrics and "brain_power_orchestrating" in metrics:
                    metrics["brain_power_orchestrating"].set(0)
            else:
                logger.info("Brain Power orchestrator: AVAILABLE")
                if metrics and "brain_power_orchestrating" in metrics:
                    metrics["brain_power_orchestrating"].set(1)
                if metrics and "model_online" in metrics:
                    metrics["model_online"].labels(model_name="brain_power", port="api").set(1)
        else:
            logger.warning("Brain Power orchestrator: import failed")
            if metrics and "brain_power_orchestrating" in metrics:
                metrics["brain_power_orchestrating"].set(0)

        # Air-LLM (cuda:0)
        if _airllm_available:
            self._airllm_manager = AirLLMModelManager()
            self.backends["airllm"].available = True
            logger.info("Air-LLM backend: AVAILABLE (cuda:0, lazy-load)")
            if metrics and "model_online" in metrics:
                metrics["model_online"].labels(model_name="air_llm", port="8002").set(1)
        else:
            logger.warning("Air-LLM backend: import failed")

        # Qwen (cuda:1)
        if _qwen_available:
            self._qwen_manager = QwenModelManager()
            self.backends["qwen"].available = True
            logger.info("Qwen backend: AVAILABLE (cuda:1, lazy-load)")
            if metrics and "model_online" in metrics:
                metrics["model_online"].labels(model_name="qwen_2_5_7b", port="8004").set(1)
        else:
            logger.warning("Qwen backend: import failed")

    # ─── Parallel Model Execution ─────────────────────────────────

    async def _run_airllm(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Run Air-LLM inference in thread pool."""
        if not self._airllm_manager:
            return {"text": "", "error": "Air-LLM not available", "backend": "airllm"}

        metrics = _get_prom_metrics()
        start_time = time.time()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self._airllm_manager.generate(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                ),
            )
            latency = time.time() - start_time
            self.backends["airllm"].request_count += 1

            # Prometheus instrumentation
            if metrics:
                if "model_online" in metrics:
                    metrics["model_online"].labels(model_name="air_llm", port="8002").set(1)
                if "model_inference_latency" in metrics:
                    metrics["model_inference_latency"].labels(model_name="air_llm").observe(latency)
                if "model_inference_counter" in metrics:
                    metrics["model_inference_counter"].labels(model_name="air_llm", status="success").inc()

            return {"text": result.get("text", ""), "tokens_used": result.get("tokens_generated", 0), "backend": "airllm", "latency_ms": round(latency * 1000, 1)}
        except Exception as e:
            latency = time.time() - start_time
            self.backends["airllm"].request_count += 1
            self.backends["airllm"].error_count += 1
            logger.error(f"Air-LLM parallel inference error: {e}")

            # Prometheus instrumentation
            if metrics:
                if "model_online" in metrics:
                    metrics["model_online"].labels(model_name="air_llm", port="8002").set(0)
                if "model_inference_latency" in metrics:
                    metrics["model_inference_latency"].labels(model_name="air_llm").observe(latency)
                if "model_inference_counter" in metrics:
                    metrics["model_inference_counter"].labels(model_name="air_llm", status="error").inc()

            return {"text": "", "error": str(e), "backend": "airllm"}

    async def _run_qwen(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Run Qwen inference in thread pool."""
        if not self._qwen_manager:
            return {"text": "", "error": "Qwen not available", "backend": "qwen"}

        metrics = _get_prom_metrics()
        start_time = time.time()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self._qwen_manager.generate(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                ),
            )
            latency = time.time() - start_time
            self.backends["qwen"].request_count += 1

            # Prometheus instrumentation
            if metrics:
                if "model_online" in metrics:
                    metrics["model_online"].labels(model_name="qwen_2_5_7b", port="8004").set(1)
                if "model_inference_latency" in metrics:
                    metrics["model_inference_latency"].labels(model_name="qwen_2_5_7b").observe(latency)
                if "model_inference_counter" in metrics:
                    metrics["model_inference_counter"].labels(model_name="qwen_2_5_7b", status="success").inc()

            return {"text": result.get("text", ""), "tokens_used": result.get("tokens_generated", 0), "backend": "qwen", "latency_ms": round(latency * 1000, 1)}
        except Exception as e:
            latency = time.time() - start_time
            self.backends["qwen"].request_count += 1
            self.backends["qwen"].error_count += 1
            logger.error(f"Qwen parallel inference error: {e}")

            # Prometheus instrumentation
            if metrics:
                if "model_online" in metrics:
                    metrics["model_online"].labels(model_name="qwen_2_5_7b", port="8004").set(0)
                if "model_inference_latency" in metrics:
                    metrics["model_inference_latency"].labels(model_name="qwen_2_5_7b").observe(latency)
                if "model_inference_counter" in metrics:
                    metrics["model_inference_counter"].labels(model_name="qwen_2_5_7b", status="error").inc()

            return {"text": "", "error": str(e), "backend": "qwen"}

    async def _run_llama(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Run Llama 3.1-8B via local FastAPI server (port 8005).

        Same architecture as Air-LLM (8003) and Qwen (8004) — dedicated PM2
        service on the GEX44 intelligence layer.
        """
        llama_url = os.environ.get("LLAMA_URL", "http://localhost:8005")
        start_time = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{llama_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    latency = time.time() - start_time
                    self.backends["llama"].request_count += 1
                    return {
                        "text": data.get("text", ""),
                        "backend": "llama",
                        "model": data.get("model", "llama3.1:8b"),
                        "latency_ms": round(latency * 1000, 1),
                        "tokens_generated": data.get("tokens_generated", 0),
                    }
                else:
                    self.backends["llama"].error_count += 1
                    return {"text": "", "error": f"Llama HTTP {response.status_code}", "backend": "llama"}
        except Exception as e:
            latency = time.time() - start_time
            self.backends["llama"].error_count += 1
            logger.debug(f"Llama inference error: {e}")
            return {"text": "", "error": str(e), "backend": "llama"}

    async def _orchestrate_brain_power(
        self,
        prompt: str,
        model_outputs: dict,
        ml_context: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> dict:
        """Brain Power synthesizes outputs from all parallel models.

        If Brain Power API key is not configured (stub mode), returns
        a merged consensus from the local models.
        """
        metrics = _get_prom_metrics()

        # Build synthesis context for Brain Power
        synthesis_parts = [
            "You are Brain Power (Xiaomi Mimo V2 Pro), the orchestrating intelligence "
            "for Metadron Capital's parallel ensemble system. Multiple models have "
            "simultaneously processed the same request. Your job is to synthesize, "
            "correct, or navigate their outputs into the best final response.\n",
        ]

        if system_prompt:
            synthesis_parts.append(f"Task system prompt: {system_prompt}\n")

        synthesis_parts.append("=== MODEL OUTPUTS ===\n")
        for backend_name, output in model_outputs.items():
            if output.get("error"):
                synthesis_parts.append(f"[{backend_name}] ERROR: {output['error']}\n")
            elif output.get("text"):
                synthesis_parts.append(f"[{backend_name}]\n{output['text']}\n")

        if ml_context:
            synthesis_parts.append("\n=== ML ENGINE CONTEXT ===\n")
            synthesis_parts.append(json.dumps(ml_context, indent=2, default=str))
            synthesis_parts.append("\n")

        synthesis_parts.append(
            "\n=== ORIGINAL PROMPT ===\n"
            f"{prompt}\n\n"
            "Synthesize the above model outputs into a single, authoritative response. "
            "Correct any errors, resolve disagreements, and integrate ML context if provided."
        )

        synthesis_prompt = "\n".join(synthesis_parts)

        # Check if Brain Power is available and not in stub mode
        if (self._brain_power_client and not self._brain_power_client.is_stub):
            start_time = time.time()
            try:
                messages = [{"role": "user", "content": synthesis_prompt}]
                result = await asyncio.to_thread(
                    self._brain_power_client.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = time.time() - start_time
                self.backends["brain_power"].request_count += 1
                usage = result.get("usage", {})

                # Prometheus instrumentation
                if metrics:
                    if "model_online" in metrics:
                        metrics["model_online"].labels(model_name="brain_power", port="api").set(1)
                    if "model_inference_latency" in metrics:
                        metrics["model_inference_latency"].labels(model_name="brain_power").observe(latency)
                    if "model_inference_counter" in metrics:
                        metrics["model_inference_counter"].labels(model_name="brain_power", status="success").inc()
                    if "brain_power_orchestrating" in metrics:
                        metrics["brain_power_orchestrating"].set(1)

                return {
                    "text": result.get("text", ""),
                    "tokens_used": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    "orchestrator": "brain_power",
                    "stub": False,
                }
            except Exception as e:
                latency = time.time() - start_time
                self.backends["brain_power"].request_count += 1
                self.backends["brain_power"].error_count += 1
                logger.error(f"Brain Power orchestration error: {e}")

                # Prometheus instrumentation
                if metrics:
                    if "model_online" in metrics:
                        metrics["model_online"].labels(model_name="brain_power", port="api").set(0)
                    if "model_inference_latency" in metrics:
                        metrics["model_inference_latency"].labels(model_name="brain_power").observe(latency)
                    if "model_inference_counter" in metrics:
                        metrics["model_inference_counter"].labels(model_name="brain_power", status="error").inc()

                # Fall through to local consensus
        else:
            # Stub mode
            if metrics and "brain_power_orchestrating" in metrics:
                metrics["brain_power_orchestrating"].set(0)

        # Stub mode or Brain Power unavailable — merge local model outputs
        return self._build_local_consensus(model_outputs)

    def _build_local_consensus(self, model_outputs: dict) -> dict:
        """Build a merged response from local model outputs when Brain Power is unavailable."""
        valid_outputs = {k: v for k, v in model_outputs.items() if v.get("text") and not v.get("error")}

        if not valid_outputs:
            return {
                "text": "[Ensemble] No model produced output.",
                "tokens_used": 0,
                "orchestrator": "local_consensus",
                "stub": True,
            }

        if len(valid_outputs) == 1:
            name, out = next(iter(valid_outputs.items()))
            return {
                "text": (
                    f"[Brain Power synthesis pending — API key not yet provided. "
                    f"Local model consensus:]\n\n"
                    f"[Source: {name}]\n{out['text']}"
                ),
                "tokens_used": out.get("tokens_used", 0),
                "orchestrator": "local_consensus",
                "stub": True,
            }

        # Multiple outputs — concatenate with headers
        parts = [
            "[Brain Power synthesis pending — API key not yet provided. "
            "Local model consensus:]\n"
        ]
        total_tokens = 0
        for name, out in valid_outputs.items():
            parts.append(f"\n--- [{name}] ---\n{out['text']}")
            total_tokens += out.get("tokens_used", 0)

        return {
            "text": "\n".join(parts),
            "tokens_used": total_tokens,
            "orchestrator": "local_consensus",
            "stub": True,
        }

    # ─── Primary Ensemble Endpoint ────────────────────────────────

    async def ensemble(
        self,
        prompt: str,
        task_type: str = "text",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        ml_context: Optional[dict] = None,
    ) -> dict:
        """Run all models in parallel, Brain Power orchestrates final output.

        This is the primary inference method. All other endpoints route here.

        Args:
            prompt: The user/system prompt.
            task_type: Task classification for metrics.
            max_tokens: Max tokens for generation.
            temperature: Sampling temperature.
            system_prompt: Optional system-level instruction.
            ml_context: Optional dict with alpha_signals, regime, patterns, agent_scores.

        Returns:
            {
                "text": str,
                "orchestrator": str,
                "model_outputs": {backend: {text, tokens_used}},
                "latency_ms": float,
                "task_type": str,
                "ml_context_provided": bool,
                "stub": bool,
            }
        """
        metrics = _get_prom_metrics()
        start_time = time.time()

        # Run Air-LLM and Qwen in parallel
        airllm_task = self._run_airllm(prompt, max_tokens, temperature)
        qwen_task = self._run_qwen(prompt, max_tokens, temperature)

        airllm_result, qwen_result = await asyncio.gather(
            airllm_task, qwen_task, return_exceptions=True
        )

        # Handle exceptions from gather
        if isinstance(airllm_result, Exception):
            airllm_result = {"text": "", "error": str(airllm_result), "backend": "airllm"}
        if isinstance(qwen_result, Exception):
            qwen_result = {"text": "", "error": str(qwen_result), "backend": "qwen"}

        model_outputs = {
            "airllm": airllm_result,
            "qwen": qwen_result,
        }

        # Brain Power orchestrates the final synthesis
        final = await self._orchestrate_brain_power(
            prompt=prompt,
            model_outputs=model_outputs,
            ml_context=ml_context,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Record ensemble synthesis latency
        if metrics and "ensemble_synthesis_latency" in metrics:
            metrics["ensemble_synthesis_latency"].observe((time.time() - start_time))

        # ── Token metering: record per-model usage ──
        try:
            from engine.security.token_meter import get_meter
            meter = get_meter()
            prompt_tokens = len(prompt) // 4  # estimate
            for backend_name, result in model_outputs.items():
                out_tokens = len(result.get("text", "")) // 4
                meter.record(
                    model=backend_name,
                    tokens_in=prompt_tokens,
                    tokens_out=out_tokens,
                    latency_ms=result.get("latency_ms", latency_ms),
                    caller=task_type,
                )
        except Exception:
            pass

        return {
            "text": final.get("text", ""),
            "orchestrator": final.get("orchestrator", "unknown"),
            "model_outputs": {
                k: {"text": v.get("text", "")[:200] + "..." if len(v.get("text", "")) > 200 else v.get("text", ""),
                     "error": v.get("error")}
                for k, v in model_outputs.items()
            },
            "latency_ms": round(latency_ms, 1),
            "tokens_used": final.get("tokens_used", 0),
            "task_type": task_type,
            "ml_context_provided": ml_context is not None,
            "stub": final.get("stub", False),
        }

    # ─── Legacy single-backend infer (routes through ensemble) ────

    async def infer(
        self,
        prompt: str,
        task_type: str = "text",
        preferred_backend: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        images: Optional[list] = None,
        audio: Optional[str] = None,
    ) -> dict:
        """Run inference through the parallel ensemble.

        All requests now route through the ensemble. The preferred_backend
        parameter is kept for API compatibility but all models always run.
        """
        result = await self.ensemble(
            prompt=prompt,
            task_type=task_type,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        # Reshape to legacy response format
        return {
            "text": result.get("text", ""),
            "backend": result.get("orchestrator", "ensemble"),
            "latency_ms": result.get("latency_ms", 0),
            "tokens_used": result.get("tokens_used", 0),
            "task_type": task_type,
            "stub": result.get("stub", False),
            "ensemble": True,
        }

    def get_status(self) -> dict:
        """Return status of all backends."""
        return {
            name: {
                "available": b.available,
                "model_id": b.model_id,
                "role": "orchestrator" if name == "brain_power" else "parallel_model",
                "capabilities": b.capabilities,
                "request_count": b.request_count,
                "error_rate": round(b.error_rate, 3),
                "avg_latency_ms": round(b.avg_latency_ms, 1),
            }
            for name, b in self.backends.items()
        }


# ─── FastAPI Service (PM2 managed) ───────────────────────────────────

def create_app():
    """Create FastAPI app for the LLM inference bridge."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Metadron LLM Inference Bridge — Parallel Ensemble",
        description=(
            "Parallel ensemble: Air-LLM + Qwen run simultaneously, "
            "Brain Power (Xiaomi Mimo V2 Pro) orchestrates final output."
        ),
        version="3.0.0",
    )

    # CORS restricted to platform services — set LLM_BRIDGE_CORS_ORIGINS to your public IP at deploy
    _cors_origins = os.environ.get(
        "LLM_BRIDGE_CORS_ORIGINS",
        "http://localhost:5000,http://localhost:8001,http://127.0.0.1:5000,http://127.0.0.1:8001",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    bridge = LLMInferenceBridge()

    class InferenceRequest(BaseModel):
        prompt: str
        task_type: str = "text"
        preferred_backend: Optional[str] = None
        max_tokens: int = 2048
        temperature: float = 0.3
        system_prompt: Optional[str] = None

    class EnsembleRequest(BaseModel):
        prompt: str
        task_type: str = "text"
        max_tokens: int = 2048
        temperature: float = 0.3
        system_prompt: Optional[str] = None
        ml_context: Optional[dict] = None

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "architecture": "parallel_ensemble",
            "orchestrator": "brain_power",
            "backends": bridge.get_status(),
        }

    @app.post("/ensemble")
    async def ensemble(request: EnsembleRequest):
        """Primary endpoint — parallel ensemble with Brain Power orchestration."""
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type=request.task_type,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            ml_context=request.ml_context,
        )

    @app.post("/infer")
    async def infer(request: InferenceRequest):
        """Legacy endpoint — routes through ensemble internally."""
        return await bridge.infer(
            prompt=request.prompt,
            task_type=request.task_type,
            preferred_backend=request.preferred_backend,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
        )

    @app.get("/backends")
    async def backends():
        return bridge.get_status()

    # ─── Investment-Specific Endpoints (routed through ensemble) ───

    @app.post("/analyze/sentiment")
    async def analyze_sentiment(request: InferenceRequest):
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type="sentiment",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=(
                "You are a financial sentiment analyst. Analyze the following text and return "
                "a JSON object with: sentiment (-1.0 to 1.0), confidence (0-1), key_phrases (list), "
                "market_impact (bearish/neutral/bullish), and reasoning (brief)."
            ),
        )

    @app.post("/analyze/earnings")
    async def analyze_earnings(request: InferenceRequest):
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type="earnings",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=(
                "You are an expert earnings analyst. Analyze the earnings transcript/report and return "
                "a JSON with: revenue_surprise (%), eps_surprise (%), guidance (raised/maintained/lowered), "
                "key_themes (list), risk_factors (list), and overall_signal (bullish/neutral/bearish)."
            ),
        )

    @app.post("/analyze/sec_filing")
    async def analyze_sec_filing(request: InferenceRequest):
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type="sec_filing",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=(
                "You are an SEC filing analyst. Parse the filing and extract: filing_type, "
                "material_changes (list), risk_factors (list with severity), insider_transactions, "
                "related_party_transactions, and investment_implications."
            ),
        )

    @app.post("/generate/trade_thesis")
    async def generate_trade_thesis(request: InferenceRequest):
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type="trade_thesis",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=(
                "You are a quantitative portfolio manager at Metadron Capital. Generate a trade thesis "
                "with: ticker, direction (long/short), conviction (1-10), entry_price, stop_loss, "
                "target_price, timeframe, catalyst, risk_factors, and position_size_suggestion."
            ),
        )

    @app.post("/generate/narrative")
    async def generate_narrative(request: InferenceRequest):
        return await bridge.ensemble(
            prompt=request.prompt,
            task_type="narrative",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=(
                "You are the chief strategist at Metadron Capital. Generate a market narrative "
                "covering: regime_assessment, macro_outlook, sector_rotation, key_risks, "
                "opportunities, and portfolio_positioning. Be concise and actionable."
            ),
        )

    return app


# ─── Entry Point ──────────────────────────────────────────────────────

def main():
    """Run the LLM inference bridge as a PM2-managed service."""
    import uvicorn

    port = int(os.environ.get("LLM_BRIDGE_PORT", "8002"))
    host = os.environ.get("LLM_BRIDGE_HOST", "0.0.0.0")

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting LLM Inference Bridge (Parallel Ensemble) on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
