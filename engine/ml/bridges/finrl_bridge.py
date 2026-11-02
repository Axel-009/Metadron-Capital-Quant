"""
FinRL DRL Agent Bridge
Produces DRL_AGENT_BUY / DRL_AGENT_SELL signals for MLVoteEnsemble.
Falls back to pure-numpy momentum-based DRL proxy when FinRL is unavailable.
"""

import logging
import hashlib

import numpy as np

try:
    from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore
    FINRL_AVAILABLE = True
except Exception:
    FINRL_AVAILABLE = False

try:
    from engine.execution.paper_broker import SignalType
except Exception:
    class SignalType:
        DRL_AGENT_BUY = "DRL_AGENT_BUY"
        DRL_AGENT_SELL = "DRL_AGENT_SELL"

logger = logging.getLogger(__name__)


class FinRLBridge:
    """Bridge adapter for FinRL deep-reinforcement-learning agents."""

    def __init__(self, model_path: str | None = None, seed: int = 42):
        self.model_path = model_path
        self.seed = seed
        self.model = None
        self._rng = np.random.RandomState(seed)

        if FINRL_AVAILABLE and model_path is not None:
            try:
                self.model = DRLAgent.load(model_path)
                logger.info("FinRL model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load FinRL model: %s. Using fallback.", exc)
                self.model = None
        else:
            logger.info("FinRL not available. Using numpy momentum fallback.")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

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
    def _macd(prices: np.ndarray) -> float:
        if len(prices) < 26:
            return 0.0
        ema12 = FinRLBridge._ema(prices, 12)
        ema26 = FinRLBridge._ema(prices, 26)
        return float(ema12 - ema26)

    @staticmethod
    def _ema(prices: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1)
        weights = (1 - alpha) ** np.arange(min(len(prices), span))[::-1]
        weights /= weights.sum()
        return float(np.dot(prices[-len(weights):], weights))

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
    def _turbulence(returns: np.ndarray, window: int = 20) -> float:
        if len(returns) < window:
            return 0.0
        recent = returns[-window:]
        mu = np.mean(recent)
        sigma = np.std(recent) + 1e-10
        return float(((recent[-1] - mu) / sigma) ** 2)

    def build_observation(self, returns: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Build the 7-feature observation vector used by the DRL agent."""
        close_norm = prices[-1] / (np.mean(prices[-20:]) + 1e-10) if len(prices) >= 20 else 1.0
        vol_norm = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        rsi = self._rsi(prices) / 100.0
        macd = self._macd(prices)
        cci = self._cci(prices) / 200.0
        adx = self._adx(prices) / 100.0
        turb = self._turbulence(returns)
        return np.array([close_norm, vol_norm, rsi, macd, cci, adx, turb], dtype=np.float64)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _fallback_predict(self, obs: np.ndarray) -> tuple[float, float]:
        """Momentum-based DRL proxy using the observation vector."""
        close_norm, vol_norm, rsi, macd, cci, adx, turb = obs
        momentum_score = (
            0.25 * (rsi - 0.5)
            + 0.25 * np.tanh(macd)
            + 0.20 * np.tanh(cci)
            + 0.15 * (close_norm - 1.0)
            - 0.15 * np.tanh(turb)
        )
        confidence = min(abs(momentum_score) * 2.0, 1.0)
        return float(momentum_score), float(confidence)

    def predict(self, ticker: str, returns: np.ndarray, features: dict | None = None) -> dict:
        """
        Produce a DRL signal for the given ticker.

        Returns
        -------
        dict with keys: signal, confidence, features
        """
        prices = np.cumprod(1.0 + returns) * 100.0 if len(returns) > 0 else np.array([100.0])
        obs = self.build_observation(returns, prices)

        if self.model is not None:
            try:
                action = self.model.predict(obs.reshape(1, -1))
                score = float(np.squeeze(action))
                confidence = min(abs(score), 1.0)
            except Exception as exc:
                logger.warning("FinRL predict failed: %s. Using fallback.", exc)
                score, confidence = self._fallback_predict(obs)
        else:
            score, confidence = self._fallback_predict(obs)

        if score > 0.0:
            signal = SignalType.DRL_AGENT_BUY
        else:
            signal = SignalType.DRL_AGENT_SELL

        return {
            "signal": signal,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "features": {
                "close_norm": obs[0],
                "vol_norm": obs[1],
                "rsi": obs[2],
                "macd": obs[3],
                "cci": obs[4],
                "adx": obs[5],
                "turbulence": obs[6],
            },
        }
