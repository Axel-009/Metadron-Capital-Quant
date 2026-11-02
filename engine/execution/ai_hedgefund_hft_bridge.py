"""
AI Hedge Fund HFT Bridge — wraps the ai-hedgefund HFT engine for L7 integration.

Provides high-frequency execution capabilities to the L7 Unified Execution Surface.
The underlying HFTEngine lives at intelligence_platform/ai-hedgefund/src/execution/hft_engine.py.
Falls back gracefully if the engine is unavailable.
"""
from __future__ import annotations
import logging
import importlib.util as _ilu
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic load of ai-hedgefund HFT engine
# ---------------------------------------------------------------------------
_HFT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "intelligence_platform" / "ai-hedgefund"
    / "src" / "execution" / "hft_engine.py"
)

try:
    _hft_spec = _ilu.spec_from_file_location("hft_engine", str(_HFT_PATH))
    _hft_mod = _ilu.module_from_spec(_hft_spec)
    _hft_spec.loader.exec_module(_hft_mod)
    HFTEngine = getattr(_hft_mod, "HFTEngine", None)
    _HFTENGINE_AVAILABLE = HFTEngine is not None
except Exception as exc:
    logger.info("ai-hedgefund HFT engine not available: %s", exc)
    HFTEngine = None
    _HFTENGINE_AVAILABLE = False


class AiHedgeFundHFTBridge:
    """
    Adapter between L7UnifiedExecutionSurface and the ai-hedgefund HFT engine.

    Usage:
        bridge = AiHedgeFundHFTBridge()
        if bridge.is_available():
            result = bridge.submit_hft_order(symbol="AAPL", qty=100, side="buy")
    """

    def __init__(self) -> None:
        self._engine: Optional[object] = None
        if _HFTENGINE_AVAILABLE and HFTEngine is not None:
            try:
                self._engine = HFTEngine()
                logger.info("AiHedgeFundHFTBridge: HFTEngine instantiated")
            except Exception as exc:
                logger.warning("HFTEngine init failed: %s", exc)
                self._engine = None

    def is_available(self) -> bool:
        return self._engine is not None

    def submit_hft_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        **kwargs,
    ) -> dict:
        """Submit a high-frequency order via the HFT engine."""
        if not self.is_available():
            return {"status": "unavailable", "reason": "HFT engine not loaded"}
        try:
            method = (
                getattr(self._engine, "submit_order", None)
                or getattr(self._engine, "place_order", None)
            )
            if method is None:
                return {"status": "error", "reason": "No submit_order method on HFTEngine"}
            result = method(
                symbol=symbol, qty=qty, side=side,
                order_type=order_type, time_in_force=time_in_force, **kwargs,
            )
            return result if isinstance(result, dict) else {"status": "ok", "result": str(result)}
        except Exception as exc:
            logger.error("HFT submit_hft_order error: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def get_status(self) -> dict:
        """Return HFT engine status."""
        return {
            "available": self.is_available(),
            "engine_class": type(self._engine).__name__ if self._engine else None,
            "hft_engine_path": str(_HFT_PATH),
            "hft_engine_exists": _HFT_PATH.exists(),
        }
