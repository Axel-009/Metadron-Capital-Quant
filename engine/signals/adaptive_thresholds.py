"""Metadron Capital — Adaptive Threshold Calibration.

Replaces fixed regime thresholds (VIX > 35 = CRASH) with percentile-based
thresholds computed over a rolling window. Self-calibrates to evolving
market dynamics — what was extreme in 2017 may be normal in 2024.

Usage in MacroEngine:
    from .adaptive_thresholds import get_calibrator
    cal = get_calibrator()
    cal.update_vix_history(vix_value)
    if vix > cal.vix_crash_threshold():
        regime = MarketRegime.CRASH
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger("metadron.signals.adaptive_thresholds")


class AdaptiveThresholdCalibrator:
    """Rolling-window percentile-based threshold calibration.

    Default: 252-day window (1 trading year).
    VIX > 90th percentile = STRESS
    VIX > 97th percentile = CRASH

    Falls back to fixed thresholds (35 = CRASH, 28 = STRESS) until
    enough history accumulates (min 60 observations).
    """

    DEFAULT_WINDOW = 252
    MIN_HISTORY_FOR_ADAPTIVE = 60

    # Fallback fixed thresholds (used until history accumulates)
    FALLBACK_VIX_CRASH = 35.0
    FALLBACK_VIX_STRESS = 28.0
    FALLBACK_HY_OAS_CRASH_BPS = 800
    FALLBACK_HY_OAS_STRESS_BPS = 600

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._vix_history: deque = deque(maxlen=window)
        self._hy_oas_history: deque = deque(maxlen=window)

    # ── Update methods (called by MacroEngine on each cycle) ──

    def update_vix(self, vix: float):
        """Add today's VIX close to history."""
        if vix is not None and vix > 0:
            self._vix_history.append(float(vix))

    def update_hy_oas(self, oas_bps: float):
        """Add today's HY OAS spread to history."""
        if oas_bps is not None and oas_bps > 0:
            self._hy_oas_history.append(float(oas_bps))

    # ── Adaptive thresholds ──

    def _percentile(self, history: deque, pct: float, fallback: float) -> float:
        """Compute percentile of history, fallback if insufficient data."""
        if len(history) < self.MIN_HISTORY_FOR_ADAPTIVE:
            return fallback
        try:
            import numpy as np
            return float(np.percentile(list(history), pct))
        except Exception:
            return fallback

    def vix_crash_threshold(self) -> float:
        """VIX level above which regime = CRASH (97th percentile of rolling window)."""
        return self._percentile(self._vix_history, 97, self.FALLBACK_VIX_CRASH)

    def vix_stress_threshold(self) -> float:
        """VIX level above which regime = STRESS (90th percentile)."""
        return self._percentile(self._vix_history, 90, self.FALLBACK_VIX_STRESS)

    def hy_oas_crash_threshold(self) -> float:
        """HY OAS bps above which kill-switch component = CRASH (97th percentile)."""
        return self._percentile(self._hy_oas_history, 97, self.FALLBACK_HY_OAS_CRASH_BPS)

    def hy_oas_stress_threshold(self) -> float:
        """HY OAS bps above which kill-switch component = STRESS (90th percentile)."""
        return self._percentile(self._hy_oas_history, 90, self.FALLBACK_HY_OAS_STRESS_BPS)

    # ── Status for monitoring ──

    def get_status(self) -> dict:
        return {
            "window": self.window,
            "vix_observations": len(self._vix_history),
            "hy_oas_observations": len(self._hy_oas_history),
            "adaptive_active": len(self._vix_history) >= self.MIN_HISTORY_FOR_ADAPTIVE,
            "vix_crash_threshold": round(self.vix_crash_threshold(), 2),
            "vix_stress_threshold": round(self.vix_stress_threshold(), 2),
            "hy_oas_crash_threshold_bps": round(self.hy_oas_crash_threshold(), 1),
            "hy_oas_stress_threshold_bps": round(self.hy_oas_stress_threshold(), 1),
        }


# Singleton
_calibrator: Optional[AdaptiveThresholdCalibrator] = None


def get_calibrator() -> AdaptiveThresholdCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = AdaptiveThresholdCalibrator()
    return _calibrator
