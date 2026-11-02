"""
12-feature state vector builder for DRL models.
All features computed with pure numpy. Feeds into other bridge adapters.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DeepTradingFeatureBuilder:
    """Builds the 12-feature state vector consumed by DRL model bridges."""

    FEATURE_NAMES = [
        "close_norm",
        "volume_norm",
        "rsi_14",
        "macd",
        "cci_20",
        "adx_14",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "obv_norm",
        "momentum_10",
        "vol_20",
    ]

    # ------------------------------------------------------------------
    # Individual feature computations
    # ------------------------------------------------------------------

    @staticmethod
    def _close_norm(prices: np.ndarray) -> float:
        if len(prices) < 20:
            return 1.0
        return float(prices[-1] / (np.mean(prices[-20:]) + 1e-10))

    @staticmethod
    def _volume_norm(volumes: np.ndarray) -> float:
        if len(volumes) < 20 or np.mean(volumes[-20:]) < 1e-10:
            return 1.0
        return float(volumes[-1] / (np.mean(volumes[-20:]) + 1e-10))

    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains) + 1e-10
        avg_loss = np.mean(losses) + 1e-10
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1)
        n = min(len(data), span * 2)
        weights = (1 - alpha) ** np.arange(n)[::-1]
        weights /= weights.sum()
        return float(np.dot(data[-n:], weights))

    @staticmethod
    def _macd(prices: np.ndarray) -> float:
        if len(prices) < 26:
            return 0.0
        ema12 = DeepTradingFeatureBuilder._ema(prices, 12)
        ema26 = DeepTradingFeatureBuilder._ema(prices, 26)
        return float(ema12 - ema26)

    @staticmethod
    def _cci(prices: np.ndarray, period: int = 20) -> float:
        if len(prices) < period:
            return 0.0
        tp = prices[-period:]
        sma = np.mean(tp)
        mad = np.mean(np.abs(tp - sma)) + 1e-10
        return float((tp[-1] - sma) / (0.015 * mad))

    @staticmethod
    def _adx(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 25.0
        changes = np.diff(prices[-(period + 1):])
        pos = np.where(changes > 0, changes, 0.0)
        neg = np.where(changes < 0, -changes, 0.0)
        atr = np.mean(np.abs(changes)) + 1e-10
        plus_di = 100.0 * np.mean(pos) / atr
        minus_di = 100.0 * np.mean(neg) / atr
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return float(dx)

    @staticmethod
    def _bollinger_bands(prices: np.ndarray, period: int = 20) -> tuple[float, float]:
        if len(prices) < period:
            return 0.0, 0.0
        window = prices[-period:]
        sma = np.mean(window)
        std = np.std(window) + 1e-10
        price = prices[-1]
        bb_upper = (price - (sma + 2.0 * std)) / (4.0 * std)
        bb_lower = (price - (sma - 2.0 * std)) / (4.0 * std)
        return float(bb_upper), float(bb_lower)

    @staticmethod
    def _atr(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 0.0
        changes = np.abs(np.diff(prices[-(period + 1):]))
        return float(np.mean(changes) / (prices[-1] + 1e-10))

    @staticmethod
    def _obv_norm(returns: np.ndarray, volumes: np.ndarray) -> float:
        if len(returns) == 0 or len(volumes) == 0:
            return 0.0
        n = min(len(returns), len(volumes))
        signs = np.sign(returns[-n:])
        obv = np.cumsum(signs * volumes[-n:])
        if len(obv) < 2:
            return 0.0
        obv_range = np.max(obv) - np.min(obv)
        if obv_range < 1e-10:
            return 0.0
        return float((obv[-1] - np.mean(obv)) / (obv_range + 1e-10))

    @staticmethod
    def _momentum(returns: np.ndarray, period: int = 10) -> float:
        if len(returns) < period:
            return 0.0
        return float(np.sum(returns[-period:]))

    @staticmethod
    def _volatility(returns: np.ndarray, period: int = 20) -> float:
        if len(returns) < period:
            return 0.01
        return float(np.std(returns[-period:]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_state_vector(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build the 12-feature state vector.

        Parameters
        ----------
        returns : array of log or simple returns
        prices : array of close prices
        volumes : array of volume data (optional; defaults to ones)

        Returns
        -------
        np.ndarray of shape (12,)
        """
        if volumes is None:
            volumes = np.ones_like(prices)

        features = np.array([
            self._close_norm(prices),
            self._volume_norm(volumes),
            self._rsi(prices, 14) / 100.0,
            self._macd(prices),
            self._cci(prices, 20) / 200.0,
            self._adx(prices, 14) / 100.0,
            *self._bollinger_bands(prices, 20),
            self._atr(prices, 14),
            self._obv_norm(returns, volumes),
            self._momentum(returns, 10),
            self._volatility(returns, 20),
        ], dtype=np.float64)

        return features
