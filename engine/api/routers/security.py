"""Security Dashboard Router — Full security subsystem status + controls.

Endpoints:
    GET  /security/status         — Full 6-layer defense status
    GET  /security/ledger         — Recent transaction ledger entries
    GET  /security/tokens         — Token meter status (per-model, hourly)
    GET  /security/tokens/daily   — Daily token breakdown for archive
    POST /security/tokens/override — Manual cap override (4M daily flag)
    POST /security/broker/unfreeze — Unfreeze broker integrity lock
    POST /security/chain/reset     — Reset phase chain after investigation
    GET  /security/heartbeats     — Service heartbeat status
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

logger = logging.getLogger("metadron-api.security")
router = APIRouter()


def _get_security():
    from engine.security.integrity import get_security
    return get_security()


def _get_meter():
    from engine.security.token_meter import get_meter
    return get_meter()


@router.get("/status")
async def security_status():
    """Full 6-layer defense status — phase chain, broker lock, circuit breaker, etc."""
    try:
        sec = _get_security()
        meter = _get_meter()
        result = sec.get_full_status()
        result["token_meter"] = meter.get_status()
        return result
    except Exception as e:
        logger.error(f"security/status error: {e}")
        return {"error": str(e)}


@router.get("/ledger")
async def security_ledger(limit: int = 50):
    """Recent entries from the tamper-evident transaction ledger."""
    try:
        sec = _get_security()
        entries = sec.ledger.get_recent(limit)
        verification = sec.ledger.verify_chain()
        return {
            "entries": entries,
            "chain_integrity": verification,
            "total_entries": len(sec.ledger._entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"security/ledger error: {e}")
        return {"error": str(e)}


@router.get("/tokens")
async def security_tokens():
    """Token meter status — per-model hourly breakdown, anomaly flags."""
    try:
        meter = _get_meter()
        return meter.get_status()
    except Exception as e:
        logger.error(f"security/tokens error: {e}")
        return {"error": str(e)}


@router.get("/tokens/daily")
async def security_tokens_daily(date: str = ""):
    """Full daily token breakdown for archive — hourly per model with callers."""
    try:
        meter = _get_meter()
        return meter.get_daily_breakdown(date or None)
    except Exception as e:
        logger.error(f"security/tokens/daily error: {e}")
        return {"error": str(e)}


@router.post("/tokens/override")
async def security_tokens_override():
    """Manual override of 4M daily token cap. Unlocks inference for rest of day."""
    try:
        meter = _get_meter()
        return meter.override_cap()
    except Exception as e:
        logger.error(f"security/tokens/override error: {e}")
        return {"error": str(e)}


@router.post("/broker/unfreeze")
async def security_broker_unfreeze():
    """Unfreeze broker integrity lock after manual investigation."""
    try:
        sec = _get_security()
        sec.broker_lock.unfreeze()
        return {"status": "unfrozen", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"security/broker/unfreeze error: {e}")
        return {"error": str(e)}


@router.post("/chain/reset")
async def security_chain_reset():
    """Reset phase chain after investigation confirms integrity."""
    try:
        sec = _get_security()
        sec.phase_chain.reset()
        return {"status": "chain_reset", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"security/chain/reset error: {e}")
        return {"error": str(e)}


@router.get("/heartbeats")
async def security_heartbeats():
    """Service heartbeat integrity status."""
    try:
        sec = _get_security()
        return sec.heartbeat.check_integrity()
    except Exception as e:
        logger.error(f"security/heartbeats error: {e}")
        return {"error": str(e)}
