"""
Metadron Capital — Model Ensemble Status Router

GET /api/models/status — returns online/offline/stub status of all models
in the parallel ensemble architecture.
"""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger("models-router")

try:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("FastAPI required: pip install fastapi")

router = APIRouter()

# Model registry — all models in the Metadron ensemble
MODEL_REGISTRY = [
    {
        "name": "Brain Power (Xiaomi Mimo V2 Pro)",
        "key": "brain_power",
        "port": "API",
        "module": "engine.bridges.brain_power",
        "class_name": "BrainPowerClient",
        "is_orchestrator": True,
    },
    {
        "name": "Air-LLM (Llama-3.1-8B)",
        "key": "air_llm",
        "port": "8003",
        "module": "engine.bridges.airllm_model_server",
        "class_name": "AirLLMModelManager",
    },
    {
        "name": "Qwen 2.5-7B",
        "key": "qwen_2_5_7b",
        "port": "8004",
        "module": "engine.bridges.qwen_model_server",
        "class_name": "QwenModelManager",
    },
    {
        "name": "AI-Newton",
        "key": "ai_newton",
        "port": "in-process",
        "module": "engine.ml.alpha_optimizer",
        "class_name": "AlphaOptimizer",
    },
    {
        "name": "Alpha Optimizer",
        "key": "alpha_optimizer",
        "port": "in-process",
        "module": "engine.ml.alpha_optimizer",
        "class_name": "AlphaOptimizer",
    },
    {
        "name": "Deep Learning Engine (PPO)",
        "key": "deep_learning_engine",
        "port": "in-process",
        "module": "engine.ml.deep_learning_engine",
        "class_name": "DeepLearningEngine",
    },
    {
        "name": "MetadronCube",
        "key": "metadron_cube",
        "port": "in-process",
        "module": "engine.signals.metadron_cube",
        "class_name": "MetadronCube",
    },
]


def _check_bridge_health() -> dict:
    """Call GET http://localhost:8002/health with a 2s timeout."""
    import urllib.request
    import json

    try:
        req = urllib.request.Request("http://localhost:8002/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug("Bridge health check failed: %s", e)
        return {}


def _check_model_status(model_def: dict, bridge_health: dict) -> dict:
    """Determine status of a single model."""
    key = model_def["key"]
    result = {
        "name": model_def["name"],
        "key": key,
        "port": model_def["port"],
        "status": "offline",
        "status_detail": "Not responding",
        "last_latency_ms": None,
    }

    # Check bridge-reported backends
    backends = bridge_health.get("backends", {})

    # Map keys to bridge backend names
    bridge_key_map = {
        "brain_power": "brain_power",
        "air_llm": "airllm",
        "qwen_2_5_7b": "qwen",
    }

    bridge_key = bridge_key_map.get(key)
    if bridge_key and bridge_key in backends:
        backend_info = backends[bridge_key]
        if backend_info.get("available", False):
            # Brain Power has special stub detection
            if key == "brain_power":
                try:
                    from engine.bridges.brain_power import BrainPowerClient
                    client = BrainPowerClient()
                    if client.is_stub:
                        result["status"] = "stub"
                        result["status_detail"] = "Awaiting API key"
                    else:
                        result["status"] = "online"
                        result["status_detail"] = "Ready"
                except Exception:
                    result["status"] = "stub"
                    result["status_detail"] = "Awaiting API key"
            else:
                result["status"] = "online"
                result["status_detail"] = "Ready"
                latency = backend_info.get("avg_latency_ms", 0)
                if latency > 0:
                    result["last_latency_ms"] = round(latency, 1)
        else:
            result["status"] = "offline"
            result["status_detail"] = "Backend unavailable"
        return result

    # For in-process models, try importing
    try:
        import importlib
        mod = importlib.import_module(model_def["module"])
        cls = getattr(mod, model_def["class_name"], None)
        if cls is not None:
            result["status"] = "online"
            if key == "deep_learning_engine":
                result["status_detail"] = "Ready — ensemble advisor wired"
            else:
                result["status_detail"] = "Ready"
        else:
            result["status"] = "offline"
            result["status_detail"] = "Class not found"
    except Exception as e:
        result["status"] = "offline"
        result["status_detail"] = f"Import failed: {str(e)[:60]}"

    return result


@router.get("/status")
async def models_status():
    """GET /api/models/status — returns status of all models in the ensemble."""
    bridge_health = _check_bridge_health()

    models = []
    brain_power_orchestrating = False
    any_offline = False
    all_non_stub_online = True

    for model_def in MODEL_REGISTRY:
        status = _check_model_status(model_def, bridge_health)
        models.append(status)

        if status["key"] == "brain_power":
            brain_power_orchestrating = status["status"] == "online"

        if status["status"] == "offline":
            any_offline = True
            all_non_stub_online = False
        elif status["status"] == "stub":
            pass  # stubs don't count against ensemble_active
        elif status["status"] != "online":
            all_non_stub_online = False

    return {
        "models": models,
        "ensemble_active": all_non_stub_online and not any_offline,
        "brain_power_orchestrating": brain_power_orchestrating,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
