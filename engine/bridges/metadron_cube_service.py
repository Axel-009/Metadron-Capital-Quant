"""
Metadron Capital — MetadronCube Continuous Service

PM2-managed background service that runs MetadronCube regime detection
in continuous mode. Periodically computes C(t) = f(L,R,F) and caches
the latest cube state for consumption by the API layer and execution
pipeline.

Modes (via METADRON_CUBE_MODE env var):
    continuous — recompute cube state on a configurable interval (default)
    once       — compute once and exit (for testing)

Usage:
    python3 engine/bridges/metadron_cube_service.py
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("metadron-cube-service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Import Cube and Dependencies ──────────────────────────────────

_cube_available = False
_MetadronCube = None
_MacroEngine = None

try:
    from engine.signals.metadron_cube import MetadronCube
    _MetadronCube = MetadronCube
    _cube_available = True
    logger.info("MetadronCube: AVAILABLE")
except ImportError as e:
    logger.warning(f"MetadronCube import failed: {e}")

try:
    from engine.signals.macro_engine import MacroEngine
    _MacroEngine = MacroEngine
    logger.info("MacroEngine: AVAILABLE")
except ImportError as e:
    logger.warning(f"MacroEngine import failed: {e}")


# ─── Cached State ──────────────────────────────────────────────────

_cached_state_path = PLATFORM_ROOT / "data" / "cube_state_cache.json"


# ─── Service ───────────────────────────────────────────────────────

class MetadronCubeService:
    """Runs MetadronCube in a continuous loop with cached state output."""

    # Default compute interval in seconds
    DEFAULT_INTERVAL_MARKET = 60      # During market hours: every 60s
    DEFAULT_INTERVAL_OFFHOURS = 300   # Off-market: every 5 minutes

    MARKET_OPEN_HOUR = 9
    MARKET_CLOSE_HOUR = 16

    def __init__(self):
        self.cube = None
        self.macro_engine = None
        self.running = False
        self.mode = os.environ.get("METADRON_CUBE_MODE", "continuous")
        self.started_at = None
        self.compute_count = 0
        self.error_count = 0
        self.last_regime = None
        self.last_compute_time = None
        self.last_error = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _is_market_hours(self) -> bool:
        now = datetime.now()
        return self.MARKET_OPEN_HOUR <= now.hour < self.MARKET_CLOSE_HOUR and now.weekday() < 5

    def _get_interval(self) -> int:
        """Return compute interval based on market hours."""
        if self._is_market_hours():
            return int(os.environ.get("CUBE_INTERVAL_MARKET", str(self.DEFAULT_INTERVAL_MARKET)))
        return int(os.environ.get("CUBE_INTERVAL_OFFHOURS", str(self.DEFAULT_INTERVAL_OFFHOURS)))

    def _initialize(self):
        """Initialize MetadronCube and MacroEngine."""
        if not _cube_available:
            raise RuntimeError("MetadronCube not available — check imports")

        self.cube = _MetadronCube()
        logger.info("MetadronCube initialized")

        if _MacroEngine:
            self.macro_engine = _MacroEngine()
            logger.info("MacroEngine initialized")
        else:
            logger.warning("MacroEngine not available — using default macro snapshot")

    def _get_macro_snapshot(self):
        """Get the current macro snapshot from MacroEngine or defaults."""
        if self.macro_engine:
            try:
                return self.macro_engine.get_snapshot()
            except Exception as e:
                logger.warning(f"MacroEngine.get_snapshot() failed: {e}")

        # Fallback: construct a default MacroSnapshot
        from engine.signals.macro_engine import MacroSnapshot
        return MacroSnapshot()

    def _compute_cycle(self):
        """Run one cube computation cycle."""
        try:
            macro = self._get_macro_snapshot()
            output = self.cube.compute(macro)

            self.compute_count += 1
            self.last_compute_time = datetime.now(timezone.utc).isoformat()
            self.last_regime = output.regime.value if hasattr(output.regime, "value") else str(output.regime)

            # Cache state to disk for other processes
            self._cache_state(output)

            logger.info(
                f"Cube cycle #{self.compute_count}: "
                f"regime={self.last_regime}, "
                f"beta_cap={output.beta_cap:.3f}, "
                f"leverage={output.max_leverage:.1f}x, "
                f"L={output.liquidity.value:.3f}, "
                f"R={output.risk.value:.3f}, "
                f"F={output.flow.value:.3f}"
            )

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Cube compute cycle failed: {e}")

    def _cache_state(self, output):
        """Write cube state to a JSON cache file for other processes to read."""
        try:
            _cached_state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "regime": output.regime.value if hasattr(output.regime, "value") else str(output.regime),
                "max_leverage": output.max_leverage,
                "beta_cap": output.beta_cap,
                "target_beta": output.target_beta,
                "liquidity": output.liquidity.value,
                "risk": output.risk.value,
                "flow": output.flow.value,
                "regime_confidence": output.regime_confidence,
                "sleeves": output.sleeves.as_dict() if hasattr(output.sleeves, "as_dict") else {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compute_count": self.compute_count,
            }

            with open(_cached_state_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to cache cube state: {e}")

    def _get_health(self) -> dict:
        """Build health status dict."""
        return {
            "status": "healthy" if self.running and self.cube else "degraded",
            "service": "metadron-cube-service",
            "mode": self.mode,
            "cube_available": _cube_available,
            "macro_available": _MacroEngine is not None,
            "started_at": self.started_at,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "compute_count": self.compute_count,
            "error_count": self.error_count,
            "last_regime": self.last_regime,
            "last_compute_time": self.last_compute_time,
            "last_error": self.last_error,
            "interval_seconds": self._get_interval(),
            "market_hours": self._is_market_hours(),
        }

    def start(self):
        """Start the cube service."""
        logger.info(f"MetadronCube Service starting (mode={self.mode})...")
        self.started_at = time.time()
        self.running = True

        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            if self.mode == "once":
                return
            logger.info("Entering idle loop — will retry initialization on next cycle")

        if self.mode == "once":
            if self.cube:
                self._compute_cycle()
            logger.info("Mode=once — exiting after single compute")
            return

        # Continuous mode
        logger.info("Entering continuous compute loop")
        while self.running:
            if not self.cube:
                try:
                    self._initialize()
                except Exception as e:
                    logger.warning(f"Re-initialization failed: {e}")
                    self._sleep(60)
                    continue

            self._compute_cycle()
            self._sleep(self._get_interval())

        # Final state dump
        if self.cube and self.cube.get_last():
            self._cache_state(self.cube.get_last())
        logger.info("MetadronCube Service exited")

    def _sleep(self, seconds: int):
        """Interruptible sleep."""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)

    def stop(self):
        """Stop the service."""
        self.running = False


# ─── Entry Point ───────────────────────────────────────────────────

def main():
    """Run the MetadronCube service as a PM2-managed process."""
    service = MetadronCubeService()
    service.start()


if __name__ == "__main__":
    main()
