"""
Temporal Fusion Transformer adapter for multi-horizon forecasting.
Produces TFT_BUY / TFT_SELL signals for MLVoteEnsemble.
Falls back to pure-numpy exponential smoothing when NVIDIA TFT is unavailable.
"""

import logging

import numpy as np

try:
    from tft_model import TemporalFusionTransformer  # type: ignore
    TFT_AVAILABLE = True
except Exception:
    TFT_AVAILABLE = False

try:
    from engine.execution.paper_broker import SignalType
except Exception:
    class SignalType:
        TFT_BUY = "TFT_BUY"
        TFT_SELL = "TFT_SELL"

logger = logging.getLogger(__name__)


class NvidiaTFTAdapter:
    """Adapter for Temporal Fusion Transformer multi-horizon forecasts."""

    DEFAULT_HORIZONS = [5, 10, 21]

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None

        if TFT_AVAILABLE and model_path is not None:
            try:
                self.model = TemporalFusionTransformer.load(model_path)
                logger.info("TFT model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load TFT model: %s. Using fallback.", exc)
                self.model = None
        else:
            logger.info("TFT not available. Using exponential smoothing fallback.")

    # ------------------------------------------------------------------
    # Exponential smoothing fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _exponential_smoothing(series: np.ndarray, alpha: float = 0.3) -> float:
        """Simple exponential smoothing; returns final smoothed value."""
        if len(series) == 0:
            return 0.0
        result = series[0]
        for val in series[1:]:
            result = alpha * val + (1.0 - alpha) * result
        return float(result)

    @staticmethod
    def _ses_forecast(returns: np.ndarray, horizon: int, alpha: float = 0.3) -> float:
        """
        Multi-step ahead forecast using Simple Exponential Smoothing.
        SES produces a flat forecast, so we scale by horizon to get
        cumulative expected return over the period.
        """
        if len(returns) < 5:
            return 0.0
        level = NvidiaTFTAdapter._exponential_smoothing(returns, alpha)
        # Compound the per-period forecast
        cumulative = (1.0 + level) ** horizon - 1.0
        return float(cumulative)

    @staticmethod
    def _double_exponential_smoothing(
        series: np.ndarray, alpha: float = 0.3, beta: float = 0.1
    ) -> tuple[float, float]:
        """Holt's linear trend method. Returns (level, trend)."""
        if len(series) < 2:
            return (float(series[-1]) if len(series) else 0.0, 0.0)
        level = series[0]
        trend = series[1] - series[0]
        for val in series[1:]:
            prev_level = level
            level = alpha * val + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
        return float(level), float(trend)

    def _fallback_forecast(
        self, returns: np.ndarray, horizons: list[int]
    ) -> dict[int, float]:
        """Multi-horizon forecast via double exponential smoothing."""
        if len(returns) < 10:
            return {h: 0.0 for h in horizons}

        level, trend = self._double_exponential_smoothing(returns[-60:])
        forecasts = {}
        for h in horizons:
            # Compound the per-period level+trend projection
            per_period = level + trend
            cumulative = (1.0 + per_period) ** h - 1.0
            forecasts[h] = float(np.clip(cumulative, -0.5, 0.5))
        return forecasts

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(forecasts: dict[int, float]) -> float:
        """Confidence based on agreement across horizons."""
        vals = list(forecasts.values())
        if not vals:
            return 0.0
        signs = [1 if v > 0 else -1 for v in vals]
        agreement = abs(sum(signs)) / len(signs)
        magnitude = np.mean([abs(v) for v in vals])
        confidence = agreement * min(magnitude * 20.0, 1.0)
        return float(np.clip(confidence, 0.0, 1.0))

    def forecast(
        self,
        ticker: str,
        returns: np.ndarray,
        horizons: list[int] | None = None,
    ) -> dict:
        """
        Multi-horizon forecast for the given ticker.

        Returns
        -------
        dict with keys: signal, forecasts, confidence
        """
        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        if self.model is not None:
            try:
                raw = self.model.predict(returns, horizons=horizons)
                forecasts = {h: float(raw[i]) for i, h in enumerate(horizons)}
            except Exception as exc:
                logger.warning("TFT predict failed: %s. Using fallback.", exc)
                forecasts = self._fallback_forecast(returns, horizons)
        else:
            forecasts = self._fallback_forecast(returns, horizons)

        # Aggregate direction from all horizons
        weighted_sum = sum(
            forecasts.get(h, 0.0) * w
            for h, w in zip(horizons, [0.5, 0.3, 0.2])
        )

        if weighted_sum > 0:
            signal = SignalType.TFT_BUY
        else:
            signal = SignalType.TFT_SELL

        confidence = self._compute_confidence(forecasts)

        return {
            "signal": signal,
            "forecasts": forecasts,
            "confidence": confidence,
        }
