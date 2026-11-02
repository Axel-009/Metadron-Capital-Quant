"""
Metadron Capital — Engine API Bridge Server
Runs on port 8001, proxied by Express on port 5000.
Exposes all engine modules as REST endpoints for the frontend.
"""
import sys
import os
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Ensure engine is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from engine.api.auth import AuthMiddleware

from engine.api.routers import (
    portfolio,
    cube,
    macro,
    signals,
    risk,
    execution,
    agents,
    ml,
    monitoring,
    universe,
    futures,
    quant,
    etf,
    fixed_income,
    archive,
    backtest,
    chat,
    velocity,
    flows,
    flow_runs,
    api_keys,
    models,
    graphify,
    vault,
    security,
    jarvis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
logger = logging.getLogger("metadron-api")

try:
    from engine.api.routers import allocation
    _allocation_available = True
except Exception as _alloc_err:
    _allocation_available = False
    logger.warning("Allocation router unavailable: %s", _alloc_err)
from engine.bridges.prometheus_metrics import create_metrics_router

# Install centralized error handler — captures all engine errors for TECH tab
try:
    from engine.ops.error_logger import install_global_handler
    install_global_handler()
except Exception:
    pass

app = FastAPI(
    title="Metadron Capital Engine API",
    version="1.0.0",
    docs_url="/api/engine/docs",
    openapi_url="/api/engine/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — configurable origins (no more allow_origins=["*"])
# ---------------------------------------------------------------------------
_default_origins = ["http://localhost:5000", "http://localhost:3000", "http://localhost:5173"]
_env = os.getenv("METADRON_ENV", "development").lower()
_cors_origins_raw = os.getenv("METADRON_CORS_ORIGINS", "")
if _cors_origins_raw:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
elif _env == "production":
    _cors_origins = []  # production requires explicit origins
    logger.warning("METADRON_CORS_ORIGINS not set in production — CORS will reject all cross-origin requests")
else:
    _cors_origins = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if os.getenv("METADRON_ENV", "development").lower() == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# Middleware order: outermost runs first.
# RequestID → SecurityHeaders → Auth → CORS (already added above)
app.add_middleware(AuthMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

# ─── Register routers ──────────────────────────────────────
app.include_router(portfolio.router, prefix="/api/engine/portfolio", tags=["Portfolio"])
app.include_router(cube.router, prefix="/api/engine/cube", tags=["MetadronCube"])
app.include_router(macro.router, prefix="/api/engine/macro", tags=["Macro"])
app.include_router(signals.router, prefix="/api/engine/signals", tags=["Signals"])
app.include_router(risk.router, prefix="/api/engine/risk", tags=["Risk"])
app.include_router(execution.router, prefix="/api/engine/execution", tags=["Execution"])
app.include_router(agents.router, prefix="/api/engine/agents", tags=["Agents"])
app.include_router(ml.router, prefix="/api/engine/ml", tags=["ML"])
app.include_router(monitoring.router, prefix="/api/engine/monitoring", tags=["Monitoring"])
app.include_router(universe.router, prefix="/api/engine/universe", tags=["Universe"])
app.include_router(futures.router, prefix="/api/engine/futures", tags=["Futures"])
app.include_router(quant.router, prefix="/api/engine/quant", tags=["Quant"])
app.include_router(etf.router, prefix="/api/engine/etf", tags=["ETF"])
app.include_router(fixed_income.router, prefix="/api/engine/fixed-income", tags=["FixedIncome"])
app.include_router(archive.router, prefix="/api/engine/archive", tags=["Archive"])
app.include_router(backtest.router, prefix="/api/engine/backtest", tags=["Backtest"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(velocity.router, prefix="/api/engine/velocity", tags=["Velocity"])
app.include_router(flows.router, prefix="/api/engine/flows", tags=["Flows"])
app.include_router(flow_runs.router, prefix="/api/engine/flow-runs", tags=["FlowRuns"])
app.include_router(api_keys.router, prefix="/api/engine/api-keys", tags=["ApiKeys"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(graphify.router, prefix="/api/engine/graphify", tags=["Graphify"])
app.include_router(vault.router, prefix="/api/engine/vault", tags=["Vault"])
app.include_router(security.router, prefix="/api/engine/security", tags=["Security"])
app.include_router(jarvis.router, prefix="/api/engine/jarvis", tags=["Jarvis"])
if _allocation_available:
    app.include_router(allocation.router, prefix="/api/allocation", tags=["Allocation"])

# ─── Prometheus metrics (scraped by Contabo monitoring stack) ──────
_metrics_router = create_metrics_router(app)
if _metrics_router:
    app.include_router(_metrics_router, prefix="/api/engine", tags=["Metrics"])


@app.get("/api/engine/health")
async def health():
    """Public health check — minimal info only. No provider fingerprinting."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/engine/health/providers")
async def health_providers():
    """Check all external API provider keys and their live status.

    Returns per-provider: configured (key exists), live (test call succeeded),
    and error details if the test failed. The frontend terminal should poll
    this and surface a visible warning banner if FMP goes stale.
    """
    import os
    results = {}
    overall = "ok"

    # ── FMP (critical — sole equity/fundamental/news data provider) ────
    fmp_key = os.environ.get("FMP_API_KEY", "")
    fmp_status = {"configured": bool(fmp_key), "live": False, "error": None}
    if fmp_key:
        try:
            from engine.data.openbb_data import get_quote
            quotes = get_quote(["SPY"])
            fmp_status["live"] = bool(quotes and len(quotes) > 0)
            if not fmp_status["live"]:
                fmp_status["error"] = "get_quote returned empty — key may be expired or rate-limited"
                overall = "degraded"
        except Exception as e:
            fmp_status["error"] = str(e)
            overall = "degraded"
    else:
        fmp_status["error"] = "FMP_API_KEY not set"
        overall = "critical"
    results["fmp"] = fmp_status

    # ── IBKRBroker (trading — optional if paper-only) ─────────────────────
    ibkr_key = os.environ.get("IBKR_API_KEY", "")
    ibkr_secret = os.environ.get("IBKR_SECRET_KEY", "")
    ibkr_status = {"configured": bool(ibkr_key and ibkr_secret), "live": False, "error": None}
    if ibkr_key and ibkr_secret:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(ibkr_key, ibkr_secret, paper=True)
            acct = client.get_account()
            ibkr_status["live"] = bool(acct)
        except ImportError:
            ibkr_status["error"] = "ib_insync not installed"
        except Exception as e:
            ibkr_status["error"] = str(e)
    else:
        ibkr_status["error"] = "IBKR_API_KEY or IBKR_SECRET_KEY not set"
    results["ibkr"] = ibkr_status

    # ── Anthropic (LLM — optional) ────────────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    results["anthropic"] = {
        "configured": bool(anthropic_key),
        "live": None,  # not tested (costs money)
        "error": None if anthropic_key else "ANTHROPIC_API_KEY not set",
    }

    # ── OpenBB SDK ─────────────────────────────────────────────────────
    openbb_status = {"configured": True, "live": False, "error": None}
    try:
        from engine.data.openbb_data import _openbb_available
        openbb_status["live"] = _openbb_available
        if not _openbb_available:
            openbb_status["error"] = "OpenBB SDK not installed or failed to initialize"
            overall = "degraded"
    except Exception as e:
        openbb_status["error"] = str(e)
    results["openbb"] = openbb_status

    # ── newsfilter.io (optional real-time news) ───────────────────────
    newsfilter_status = {"configured": False, "live": False, "error": None}
    try:
        import socketio  # noqa: F401
        newsfilter_status["configured"] = True
        # Check if NewsEngine has an active connection
        try:
            from engine.signals.news_engine import NewsEngine
            ne = NewsEngine()
            if ne._newsfilter and ne._newsfilter.is_connected:
                newsfilter_status["live"] = True
            else:
                newsfilter_status["error"] = "WebSocket not connected (FMP news fallback active)"
        except Exception:
            newsfilter_status["error"] = "NewsEngine not initialized"
    except ImportError:
        newsfilter_status["error"] = "python-socketio not installed (FMP news fallback active)"
    results["newsfilter"] = newsfilter_status

    return {
        "status": overall,
        "providers": results,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.utcnow().isoformat()},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ENGINE_API_PORT", "8001"))
    logger.info(f"Starting Metadron Engine API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
