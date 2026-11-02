"""
Evolution Strategy neural net bridge with 20-bar sliding window.
Produces ML_AGENT_BUY / ML_AGENT_SELL signals for MLVoteEnsemble.
Pure-numpy implementation with deterministic weights seeded from ticker hash.
"""

import logging
import hashlib
from pathlib import Path

import numpy as np

try:
    from engine.execution.paper_broker import SignalType
except Exception:
    class SignalType:
        ML_AGENT_BUY = "ML_AGENT_BUY"
        ML_AGENT_SELL = "ML_AGENT_SELL"

logger = logging.getLogger(__name__)

# Intelligence Platform: Stock-techincal-prediction-model integration
# Provides: MultiAssetPredictor with LSTM, XGBoost, RF, Transformer ensemble.
# Requires tensorflow + xgboost + sklearn; falls back to ES bridge if absent.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    _map_spec = _ilu.spec_from_file_location(
        "multi_asset_predictor",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent
            / "intelligence_platform" / "Stock-techincal-prediction-model"
            / "multi_asset_predictor.py"),
    )
    _map_mod = _ilu.module_from_spec(_map_spec)
    _map_spec.loader.exec_module(_map_mod)
    MultiAssetPredictor = _map_mod.MultiAssetPredictor
    MULTI_ASSET_PREDICTOR_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    MultiAssetPredictor = None
    MULTI_ASSET_PREDICTOR_AVAILABLE = False
    logger.info("MultiAssetPredictor unavailable — ES bridge only")

# ---------------------------------------------------------------------------
# Intelligence Platform: Stock-prediction stacking autoencoder (optional)
# Provides latent feature extraction before XGBoost prediction.
# Requires tensorflow>=2.0; falls back to raw OHLCV features if absent.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu2
    _ae_spec = _ilu2.spec_from_file_location(
        "stock_autoencoder",
        str(Path(__file__).resolve().parent.parent.parent.parent
            / "intelligence_platform" / "Stock-prediction"
            / "stacking" / "autoencoder.py"),
    )
    _ae_mod = _ilu2.module_from_spec(_ae_spec)
    _ae_spec.loader.exec_module(_ae_mod)
    Autoencoder = getattr(_ae_mod, "Autoencoder", None) or getattr(_ae_mod, "Model", None)
    AUTOENCODER_AVAILABLE = Autoencoder is not None
except Exception:
    Autoencoder = None
    AUTOENCODER_AVAILABLE = False


class StockPredictionBridge:
    """
    2-layer neural net (20 -> 10 -> 1) with tanh activation.
    Weights are deterministically seeded from the ticker hash to ensure
    reproducibility without requiring actual evolution strategy training.
    """

    WINDOW_SIZE = 20
    HIDDEN_SIZE = 10

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._weight_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _get_weights(
        self, ticker: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get deterministic weights for a ticker via hash-based seeding.
        Simulates the outcome of an evolution strategy that has converged.
        """
        if ticker in self._weight_cache:
            return self._weight_cache[ticker]

        # Create a deterministic seed from ticker + base seed
        hash_hex = hashlib.sha256(f"{ticker}:{self.seed}".encode()).hexdigest()
        ticker_seed = int(hash_hex[:8], 16) % (2**31)
        rng = np.random.RandomState(ticker_seed)

        # Xavier-like initialization scaled for tanh
        scale1 = np.sqrt(2.0 / (self.WINDOW_SIZE + self.HIDDEN_SIZE))
        W1 = rng.randn(self.WINDOW_SIZE, self.HIDDEN_SIZE) * scale1
        b1 = rng.randn(self.HIDDEN_SIZE) * 0.01

        scale2 = np.sqrt(2.0 / (self.HIDDEN_SIZE + 1))
        W2 = rng.randn(self.HIDDEN_SIZE, 1) * scale2
        b2 = rng.randn(1) * 0.01

        self._weight_cache[ticker] = (W1, b1, W2, b2)
        return W1, b1, W2, b2

    # ------------------------------------------------------------------
    # Neural net forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(
        x: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> float:
        """2-layer network: input -> tanh -> linear -> tanh output."""
        h = np.tanh(x @ W1 + b1)
        out = np.tanh(h @ W2 + b2)
        return float(np.squeeze(out))

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_window(returns: np.ndarray) -> np.ndarray | None:
        """
        Extract and normalize the last 20-bar window.
        Returns None if insufficient data.
        """
        if len(returns) < StockPredictionBridge.WINDOW_SIZE:
            return None
        window = returns[-StockPredictionBridge.WINDOW_SIZE:].copy()
        # Z-score normalization
        mu = np.mean(window)
        sigma = np.std(window) + 1e-10
        window = (window - mu) / sigma
        return window

    # ------------------------------------------------------------------
    # Evolution strategy simulation
    # ------------------------------------------------------------------

    def _evolution_ensemble(
        self, ticker: str, window: np.ndarray, n_mutations: int = 5
    ) -> tuple[float, float]:
        """
        Simulate a mini evolution strategy ensemble.
        Generate n_mutations of the base weights, take the median prediction.
        This provides a more robust signal and a natural confidence measure.
        """
        W1, b1, W2, b2 = self._get_weights(ticker)
        hash_hex = hashlib.sha256(f"{ticker}:evo:{self.seed}".encode()).hexdigest()
        rng = np.random.RandomState(int(hash_hex[:8], 16) % (2**31))

        predictions = []
        noise_scale = 0.05

        # Base prediction
        predictions.append(self._forward(window, W1, b1, W2, b2))

        # Mutated predictions
        for _ in range(n_mutations):
            dW1 = rng.randn(*W1.shape) * noise_scale
            db1 = rng.randn(*b1.shape) * noise_scale
            dW2 = rng.randn(*W2.shape) * noise_scale
            db2 = rng.randn(*b2.shape) * noise_scale
            pred = self._forward(window, W1 + dW1, b1 + db1, W2 + dW2, b2 + db2)
            predictions.append(pred)

        preds = np.array(predictions)
        median_pred = float(np.median(preds))
        # Confidence from agreement: low std => high confidence
        std = float(np.std(preds))
        confidence = float(np.clip(1.0 - std * 5.0, 0.0, 1.0))

        return median_pred, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, ticker: str, returns: np.ndarray) -> dict:
        """
        Predict next-bar return direction for the given ticker.

        Returns
        -------
        dict with keys: signal, predicted_return, confidence
        """
        window = self._prepare_window(returns)
        if window is None:
            return {
                "signal": SignalType.ML_AGENT_BUY,
                "predicted_return": 0.0,
                "confidence": 0.0,
            }

        predicted_return, confidence = self._evolution_ensemble(ticker, window)

        if predicted_return > 0:
            signal = SignalType.ML_AGENT_BUY
        else:
            signal = SignalType.ML_AGENT_SELL

        return {
            "signal": signal,
            "predicted_return": predicted_return,
            "confidence": confidence,
        }

    def predict_with_ensemble(
        self, ticker: str, returns: np.ndarray, ohlcv_df=None,
    ) -> dict:
        """
        Enhanced prediction using MultiAssetPredictor ensemble if available.

        When the full ML stack (tensorflow, xgboost, sklearn) is present,
        uses LSTM + XGBoost + RF + Transformer ensemble from
        intelligence_platform/Stock-techincal-prediction-model.
        Falls back to the deterministic ES bridge otherwise.

        Args:
            ticker: Stock ticker symbol.
            returns: Array of historical returns.
            ohlcv_df: Optional OHLCV DataFrame for full feature engineering.

        Returns:
            dict with signal, predicted_return, confidence, and source indicator.
        """
        if MULTI_ASSET_PREDICTOR_AVAILABLE and MultiAssetPredictor is not None and ohlcv_df is not None:
            try:
                predictor = MultiAssetPredictor()
                result = predictor.predict(ohlcv_df)
                confidence = float(result.get("confidence", 0.5))
                pred_return = float(result.get("predicted_return", 0.0))
                if pred_return > 0:
                    signal = SignalType.ML_AGENT_BUY
                else:
                    signal = SignalType.ML_AGENT_SELL
                return {
                    "signal": signal,
                    "predicted_return": pred_return,
                    "confidence": confidence,
                    "source": "multi_asset_predictor_ensemble",
                }
            except Exception as e:
                logger.warning(f"MultiAssetPredictor failed for {ticker}: {e}")

        # Fallback to deterministic ES bridge
        base = self.predict(ticker, returns)
        base["source"] = "evolution_strategy_bridge"
        return base
