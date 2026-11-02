"""Metadron Capital — API Vault.

Centralized secure storage for all external API keys. Sits on top of the
entire architecture. Keys are persisted to data/vault/api_vault.json and
deployed to engines via os.environ injection on startup.

Authorized APIs:
    1. IBKR           — Trade execution (live + paper via TWS/Gateway)
    2. FMP            — Market data via OpenBB
    3. Zep            — MiroFish agent simulation (knowledge graph)
    4. Xiaomi Mimo V2 — Brain Power LLM orchestrator
    5. OpenBB Token   — OpenBB SDK platform access
    6. Internal Token  — Frontend → Backend proxy auth (auto-generated)

Frontend auth:
    The Express frontend proxies /api/engine/* to FastAPI. Instead of
    exposing API keys in client JavaScript, the frontend injects
    METADRON_INTERNAL_TOKEN as X-Internal-Token header. The backend
    trusts this token for same-host proxy traffic, eliminating the
    need for user-facing API keys in the browser.
"""

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("metadron.vault")

_VAULT_DIR = Path(os.environ.get("METADRON_VAULT_DIR", "data/vault"))
_VAULT_DIR.mkdir(parents=True, exist_ok=True)
_VAULT_FILE = _VAULT_DIR / "api_vault.json"

# ── Known API key slots ──────────────────────────────────────

VAULT_SLOTS = {
    "IBKR_API_KEY": {
        "label": "IBKR API Key",
        "target": "IBKR Broker (live + paper trading)",
        "required": True,
        "category": "execution",
    },
    "IBKR_SECRET_KEY": {
        "label": "IBKR Port",
        "target": "IBKR Broker (TWS/Gateway connection)",
        "required": True,
        "category": "execution",
    },
    "IBKR_PAPER_TRADE": {
        "label": "IBKR Paper Trade Mode",
        "target": "IBKR Broker (True=paper, False=live)",
        "required": True,
        "category": "execution",
        "is_boolean": True,
    },
    "FMP_API_KEY": {
        "label": "FMP (Financial Modeling Prep) API Key",
        "target": "OpenBB data layer (market data, fundamentals, news)",
        "required": True,
        "category": "data",
    },
    "OPENBB_TOKEN": {
        "label": "OpenBB Platform Token",
        "target": "OpenBB SDK (34+ data providers)",
        "required": False,
        "category": "data",
    },
    "ZEP_API_KEY": {
        "label": "Zep API Key",
        "target": "MiroFish agent simulation (knowledge graph)",
        "required": False,
        "category": "intelligence",
    },
    "XIAOMI_MIMO_API_KEY": {
        "label": "Xiaomi Mimo V2 Pro API Key",
        "target": "Brain Power LLM orchestrator (ensemble synthesis)",
        "required": False,
        "category": "intelligence",
    },
    "METADRON_INTERNAL_TOKEN": {
        "label": "Internal Proxy Token (auto-generated)",
        "target": "Frontend → Backend proxy authentication",
        "required": True,
        "category": "system",
        "auto_generate": True,
    },
}


class APIVault:
    """Centralized API key storage and deployment.

    Keys are persisted to data/vault/api_vault.json and injected into
    os.environ so all engines can read them via standard env var access.
    The VAULT tab in the frontend manages keys via REST API.
    """

    def __init__(self):
        self._keys: dict = {}
        self._load()
        self._ensure_internal_token()
        self._deploy_to_env()

    def _load(self):
        """Load vault from disk, bootstrapping from env if no file exists."""
        if _VAULT_FILE.exists():
            try:
                self._keys = json.loads(_VAULT_FILE.read_text(encoding="utf-8"))
                logger.info("API Vault loaded: %d keys", len(self._keys))
            except Exception as e:
                logger.warning("Vault load failed: %s", e)
                self._keys = {}
        else:
            # Bootstrap from environment variables on first run
            for slot in VAULT_SLOTS:
                val = os.environ.get(slot, "")
                if val:
                    self._keys[slot] = val
            if self._keys:
                self._save()
                logger.info("Vault bootstrapped from environment: %d keys", len(self._keys))

    def _save(self):
        """Persist vault to disk."""
        try:
            _VAULT_FILE.write_text(
                json.dumps(self._keys, indent=2, default=str), encoding="utf-8",
            )
        except Exception as e:
            logger.error("Vault save failed: %s", e)

    def _ensure_internal_token(self):
        """Auto-generate METADRON_INTERNAL_TOKEN if not set."""
        if not self._keys.get("METADRON_INTERNAL_TOKEN"):
            self._keys["METADRON_INTERNAL_TOKEN"] = secrets.token_urlsafe(48)
            self._save()
            logger.info("Internal proxy token auto-generated")

    def _deploy_to_env(self):
        """Inject all vault keys into os.environ for engine consumption."""
        deployed = 0
        for key, value in self._keys.items():
            if value:
                os.environ[key] = str(value)
                deployed += 1
        logger.info("Vault deployed %d keys to environment", deployed)

    # ── Public API ────────────────────────────────────────────

    def set_key(self, slot: str, value: str) -> dict:
        """Set or update a key in the vault. Deploys immediately to env."""
        if slot not in VAULT_SLOTS:
            return {"error": f"Unknown slot: {slot}. Valid: {list(VAULT_SLOTS.keys())}"}
        self._keys[slot] = value
        os.environ[slot] = str(value)
        self._save()
        logger.info("Vault key set: %s → %s", slot, VAULT_SLOTS[slot]["target"])
        return {"slot": slot, "status": "set", "target": VAULT_SLOTS[slot]["target"]}

    def remove_key(self, slot: str) -> dict:
        """Remove a key from the vault."""
        if slot in self._keys:
            del self._keys[slot]
            os.environ.pop(slot, None)
            self._save()
            return {"slot": slot, "status": "removed"}
        return {"slot": slot, "status": "not_found"}

    def get_status(self) -> dict:
        """Return vault status. Values are masked — never exposed in full."""
        slots = {}
        for slot, meta in VAULT_SLOTS.items():
            val = self._keys.get(slot, "")
            slots[slot] = {
                **meta,
                "configured": bool(val),
                "masked": (
                    f"{val[:4]}...{val[-4:]}" if val and len(val) > 8
                    else ("***" if val else "")
                ),
            }
        return {
            "total_slots": len(VAULT_SLOTS),
            "configured": sum(1 for s in VAULT_SLOTS if self._keys.get(s)),
            "slots": slots,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_internal_token(self) -> str:
        """Return the internal proxy token for frontend auth."""
        return self._keys.get("METADRON_INTERNAL_TOKEN", "")

    def verify_internal_token(self, token: str) -> bool:
        """Verify a token matches the internal proxy token."""
        expected = self._keys.get("METADRON_INTERNAL_TOKEN", "")
        if not expected:
            return False
        return secrets.compare_digest(token, expected)

    # ── Legacy compatibility ──────────────────────────────────

    @property
    def fmp_api_key(self) -> str:
        return self._keys.get("FMP_API_KEY", "")

    @property
    def ibkr_api_key(self) -> str:
        return self._keys.get("IBKR_API_KEY", "")

    @property
    def ibkr_secret_key(self) -> str:
        return self._keys.get("IBKR_SECRET_KEY", "")

    @property
    def ibkr_base_url(self) -> str:
        paper = self._keys.get("IBKR_PAPER_TRADE", "True")
        if str(paper).lower() in ("true", "1", "yes"):
            return "https://paper-api.ibkr.markets"
        return "https://api.ibkr.markets"

    @property
    def xiaomi_mimo_api_key(self) -> str:
        return self._keys.get("XIAOMI_MIMO_API_KEY", "")


# ── Singleton ────────────────────────────────────────────────

_vault: Optional[APIVault] = None


def get_vault() -> APIVault:
    """Return the singleton vault, creating on first call."""
    global _vault
    if _vault is None:
        _vault = APIVault()
    return _vault


def load_vault() -> APIVault:
    """Legacy alias for get_vault()."""
    return get_vault()
