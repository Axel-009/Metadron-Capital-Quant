"""API Vault Router — Manage external API keys from the VAULT tab.

Endpoints:
    GET  /vault/status     — Show all slots (configured/missing, masked values)
    POST /vault/keys       — Set a key (slot + value)
    DELETE /vault/keys     — Remove a key (slot)
    GET  /vault/token      — Get internal proxy token (for Express frontend)
    POST /vault/test       — Test connectivity for a specific slot
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("metadron-api.vault")
router = APIRouter()


class VaultKeySet(BaseModel):
    slot: str
    value: str


class VaultKeyRemove(BaseModel):
    slot: str


def _get_vault():
    from engine.api.vault import get_vault
    return get_vault()


@router.get("/status")
async def vault_status():
    """Return all vault slots with configuration status (values masked)."""
    try:
        vault = _get_vault()
        return vault.get_status()
    except Exception as e:
        logger.error(f"vault/status error: {e}")
        return {"error": str(e)}


@router.post("/keys")
async def vault_set_key(payload: VaultKeySet):
    """Set or update an API key in the vault. Deploys immediately."""
    try:
        vault = _get_vault()
        result = vault.set_key(payload.slot, payload.value)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"vault/keys POST error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/keys")
async def vault_remove_key(payload: VaultKeyRemove):
    """Remove an API key from the vault."""
    try:
        vault = _get_vault()
        return vault.remove_key(payload.slot)
    except Exception as e:
        logger.error(f"vault/keys DELETE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token")
async def vault_internal_token():
    """Return the internal proxy token for Express frontend auth.

    This endpoint is used by the Express server on startup to get the
    token it injects as X-Internal-Token on proxied requests.
    Only accessible from localhost.
    """
    try:
        vault = _get_vault()
        token = vault.get_internal_token()
        return {"token": token, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"vault/token error: {e}")
        return {"error": str(e)}


@router.post("/test")
async def vault_test_key(payload: VaultKeySet):
    """Test connectivity for a specific API key slot.

    Attempts a lightweight API call to verify the key works.
    """
    slot = payload.slot
    value = payload.value

    result = {"slot": slot, "status": "unknown", "timestamp": datetime.now(timezone.utc).isoformat()}

    try:
        if slot == "FMP_API_KEY":
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={value}")
                result["status"] = "valid" if r.status_code == 200 else f"error ({r.status_code})"

        elif slot in ("IBKR_API_KEY", "IBKR_SECRET_KEY"):
            import httpx
            headers = {"APCA-API-KEY-ID": value, "APCA-API-SECRET-KEY": payload.value}
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://paper-api.ibkr.markets/v2/account", headers=headers)
                result["status"] = "valid" if r.status_code == 200 else f"error ({r.status_code})"

        elif slot == "XIAOMI_MIMO_API_KEY":
            result["status"] = "configured" if value else "empty"
            result["note"] = "Xiaomi Mimo V2 Pro — connectivity test not available"

        elif slot == "ZEP_API_KEY":
            result["status"] = "configured" if value else "empty"
            result["note"] = "Zep — connectivity test not available"

        elif slot == "OPENBB_TOKEN":
            result["status"] = "configured" if value else "empty"

        else:
            result["status"] = "no_test_available"

    except Exception as e:
        result["status"] = f"error: {e}"

    return result
