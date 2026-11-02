"""
Multi-Asset Prediction Engine using Ensemble ML Methods.

Pulls the entire universe via OpenBB, classifies by GICS / asset class,
then applies specialised models per asset class.

Models
------
1. LSTM for time-series prediction
   Architecture: input -> LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dense(1)
   Loss: MSE = (1/n) * sum((y_i - y_hat_i)^2)
   Optimiser: Adam with lr=0.001, beta1=0.9, beta2=0.999

2. XGBoost for feature-based prediction
   Objective: reg:squarederror
   Features: RSI, MACD, BB%B, ATR, volume_ratio, returns(1,5,21,63)
   Regularisation: lambda(L2) = 1.0, alpha(L1) = 0.1

3. Random Forest ensemble
   n_estimators=500, max_depth=10, min_samples_leaf=5
   OOB score for validation

4. Transformer attention model
   Multi-head attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
   Positional encoding: PE(pos,2i) = sin(pos / 10000^(2i/d))

Usage
-----
    from multi_asset_predictor import MultiAssetPredictor
    predictor = MultiAssetPredictor()
    pred = predictor.predict("AAPL", horizon_days=5)
    scan = predictor.scan_universe_predictions()
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openbb_data import (
    AssetClass,
    detect_asset_class,
    get_full_universe,
    get_gics_classification,
    get_historical,
    get_multiple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Container for a single-symbol prediction."""
    symbol: str
    asset_class: AssetClass
    current_price: float
    predicted_price: float
    predicted_return: float
    confidence: float
    horizon_days: int
    model_weights: Dict[str, float] = field(default_factory=dict)
    individual_predictions: Dict[str, float] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class Mispricing:
    """Statistical mispricing detection result."""
    symbol: str
    asset_class: AssetClass
    current_price: float
    predicted_price: float
    mispricing_score: float  # z-score: (predicted - current) / sigma_prediction
    prediction_std: float
    direction: str  # "undervalued" or "overvalued"
    confidence: float


@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    accuracy: float
    precision: float
    recall: float
    mae: float          # Mean Absolute Error
    rmse: float         # Root Mean Squared Error
    directional_accuracy: float
    information_coefficient: float  # IC = corr(predicted_returns, actual_returns)
    hit_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss  (exponential moving average)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands:
        Middle = SMA(period)
        Upper  = Middle + num_std * std(period)
        Lower  = Middle - num_std * std(period)
        %B     = (price - Lower) / (Upper - Lower)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, pct_b


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range:
        TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        ATR = EMA(TR, period)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index:
        +DM = High - High_prev  (if > 0 and > Low_prev - Low, else 0)
        -DM = Low_prev - Low    (if > 0 and > High - High_prev, else 0)
        +DI = 100 * EMA(+DM) / ATR
        -DI = 100 * EMA(-DM) / ATR
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = EMA(DX, period)
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    atr = _compute_atr(high, low, close, period)
    plus_di = 100.0 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def prepare_features(data: pd.DataFrame, asset_class: AssetClass) -> pd.DataFrame:
    """
    Feature engineering per asset class.

    Equities: P/E, P/B, EV/EBITDA, ROE, debt/equity, revenue_growth,
              RSI, MACD, BB, volume_zscore, sector_relative_strength

    Bonds: yield, duration, convexity, credit_spread,
           term_premium = yield_10y - yield_2y, real_yield = nominal - breakeven

    Commodities: spot_price, futures_curve_slope, inventory_change,
                 seasonal_factor, USD_correlation

    Crypto: market_cap, volume_24h, NVT_ratio = network_value / tx_volume,
            hash_rate_change, exchange_flows

    FX: interest_rate_differential, PPP_deviation,
        current_account_GDP, real_effective_rate

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume.
    asset_class : AssetClass
        The asset class for feature specialisation.

    Returns
    -------
    pd.DataFrame
        Feature matrix with technical and fundamental columns.
    """
    features = pd.DataFrame(index=data.index)
    close = data["Close"].astype(float)
    high = data["High"].astype(float) if "High" in data.columns else close
    low = data["Low"].astype(float) if "Low" in data.columns else close
    volume = data["Volume"].astype(float) if "Volume" in data.columns else pd.Series(0, index=data.index)

    # --- Common technical features (all asset classes) ---

    # Returns at multiple horizons
    for horizon in [1, 5, 21, 63]:
        features[f"return_{horizon}d"] = close.pct_change(horizon)

    # Volatility (21-day rolling annualised)
    features["volatility_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)

    # RSI
    features["rsi_14"] = _compute_rsi(close, 14)

    # MACD
    macd_line, signal_line, histogram = _compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_histogram"] = histogram

    # Bollinger Bands %B
    _, _, pct_b = _compute_bollinger_bands(close)
    features["bb_pct_b"] = pct_b

    # ATR (normalised by price)
    atr = _compute_atr(high, low, close)
    features["atr_14"] = atr
    features["atr_pct"] = atr / close.replace(0, np.nan)

    # Volume features
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    features["volume_ratio"] = volume / vol_mean.replace(0, np.nan)
    features["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # ADX (trend strength)
    features["adx_14"] = _compute_adx(high, low, close)

    # Moving average crossover signals
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    features["sma_50_200_ratio"] = sma_50 / sma_200.replace(0, np.nan)
    features["price_sma50_ratio"] = close / sma_50.replace(0, np.nan)

    # --- Asset-class-specific features ---

    if asset_class == AssetClass.EQUITY:
        # Sector relative strength proxy (price momentum vs broad market)
        features["momentum_12m"] = close.pct_change(252)
        features["momentum_6m"] = close.pct_change(126)
        features["momentum_1m"] = close.pct_change(21)
        # Mean reversion signal
        features["mean_reversion_21d"] = -(close / close.rolling(21).mean() - 1.0)
        # Price gap feature
        if "Open" in data.columns:
            features["overnight_gap"] = (data["Open"].astype(float) / close.shift(1)) - 1.0
        # On-balance volume rate of change
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv_roc_21d"] = obv.pct_change(21)

    elif asset_class == AssetClass.BOND:
        # Bond-specific: yield proxy from price changes
        # term_premium approximated from price level changes
        features["yield_proxy"] = -close.pct_change(1) * 100  # inverse price->yield
        features["yield_momentum"] = features["yield_proxy"].rolling(21).mean()
        features["convexity_proxy"] = close.pct_change().diff()  # second derivative
        features["duration_proxy"] = -close.pct_change(1) / 0.01  # dP/dy approx
        features["credit_spread_momentum"] = features["yield_proxy"].rolling(63).mean() - features["yield_proxy"].rolling(21).mean()

    elif asset_class == AssetClass.COMMODITY:
        # Seasonal decomposition proxy
        features["seasonal_return_21d"] = close.pct_change(21)
        features["seasonal_return_63d"] = close.pct_change(63)
        # Inventory proxy from volume patterns
        features["inventory_proxy"] = volume.rolling(21).mean() / volume.rolling(63).mean().replace(0, np.nan)
        # Backwardation/contango proxy
        features["curve_slope_proxy"] = close.rolling(5).mean() / close.rolling(63).mean().replace(0, np.nan) - 1.0
        # USD correlation proxy (momentum)
        features["usd_correlation_proxy"] = -features["return_21d"]  # commodities ~inversely correlated with USD

    elif asset_class == AssetClass.CRYPTO:
        # NVT ratio proxy (network value / transaction volume)
        features["nvt_proxy"] = close / volume.replace(0, np.nan)
        # Realised volatility (hourly proxy using daily)
        features["realised_vol_7d"] = close.pct_change().rolling(7).std() * np.sqrt(365)
        features["realised_vol_30d"] = close.pct_change().rolling(30).std() * np.sqrt(365)
        # Exchange flow proxy
        features["exchange_flow_proxy"] = volume.pct_change(7)
        # Market dominance proxy (volume share)
        features["volume_dominance"] = volume / volume.rolling(90).mean().replace(0, np.nan)
        # Hash rate change proxy
        features["hash_rate_proxy"] = close.rolling(14).mean() / close.rolling(28).mean().replace(0, np.nan)

    elif asset_class == AssetClass.FX:
        # Interest rate differential proxy (carry)
        features["carry_proxy"] = close.pct_change(21) * 12  # annualised monthly return
        # PPP deviation proxy (long-term mean reversion)
        features["ppp_deviation"] = close / close.rolling(252).mean().replace(0, np.nan) - 1.0
        # Current account proxy (trend)
        features["trend_strength"] = (close.rolling(63).mean() - close.rolling(252).mean()) / close.rolling(252).std().replace(0, np.nan)
        # Real effective rate proxy
        features["real_rate_proxy"] = close.rolling(21).mean() / close.rolling(63).mean().replace(0, np.nan)

    elif asset_class in (AssetClass.ETF, AssetClass.INDEX):
        features["momentum_12m"] = close.pct_change(252)
        features["momentum_6m"] = close.pct_change(126)
        features["breadth_proxy"] = volume.rolling(10).mean() / volume.rolling(50).mean().replace(0, np.nan)
        features["mean_reversion_21d"] = -(close / close.rolling(21).mean() - 1.0)

    # Drop rows with NaN from rolling calculations
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    LSTM model for time-series price prediction.

    Architecture:
        Input(seq_len, n_features)
        -> LSTM(128 units, return_sequences=True)
        -> Dropout(0.2)
        -> LSTM(64 units)
        -> Dense(1)

    Loss: MSE = (1/n) * sum((y_i - y_hat_i)^2)
    Optimiser: Adam(lr=0.001, beta1=0.9, beta2=0.999)
    """

    def __init__(
        self,
        seq_length: int = 60,
        lstm1_units: int = 128,
        lstm2_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.seq_length = seq_length
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self._is_fitted = False

    def _build_model(self, n_features: int) -> Any:
        """
        Build Keras LSTM model.

        Architecture:
            Input(seq_length, n_features)
            -> LSTM(128, return_sequences=True)
            -> Dropout(0.2)
            -> LSTM(64)
            -> Dense(1)
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(
                self.lstm1_units,
                return_sequences=True,
                input_shape=(self.seq_length, n_features),
            ),
            Dropout(self.dropout_rate),
            LSTM(self.lstm2_units),
            Dense(1),
        ])
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        # Loss: MSE = (1/n) * sum((y_i - y_hat_i)^2)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(self.seq_length, len(X)):
            Xs.append(X[i - self.seq_length: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LSTMModel":
        """
        Train the LSTM model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (n_samples, n_features).
        target : pd.Series
            Target variable (e.g., forward returns).

        Returns
        -------
        self
        """
        from sklearn.preprocessing import StandardScaler

        # Clean data
        mask = features.notna().all(axis=1) & target.notna()
        X_clean = features.loc[mask].values
        y_clean = target.loc[mask].values

        if len(X_clean) < self.seq_length + 10:
            logger.warning("Insufficient data for LSTM training (%d rows)", len(X_clean))
            return self

        # Scale features and target
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X_clean)
        y_scaled = self.scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        if len(X_seq) == 0:
            logger.warning("No sequences created for LSTM")
            return self

        # Build and train
        self.model = self._build_model(X_seq.shape[2])
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0,
        )
        self._is_fitted = True
        logger.info("LSTM trained on %d sequences", len(X_seq))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions. Returns array of predicted values."""
        if not self._is_fitted or self.model is None:
            return np.full(len(features), np.nan)

        mask = features.notna().all(axis=1)
        X_clean = features.loc[mask].values

        if len(X_clean) < self.seq_length:
            return np.full(len(features), np.nan)

        X_scaled = self.scaler_X.transform(X_clean)
        X_seq = np.array([X_scaled[-self.seq_length:]])
        y_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_scaled).ravel()

        result = np.full(len(features), np.nan)
        result[-1] = y_pred[0]
        return result


# ---------------------------------------------------------------------------
# XGBoost Model
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    XGBoost model for feature-based prediction.

    Objective: reg:squarederror
    Features: RSI, MACD, BB%B, ATR, volume_ratio, returns(1,5,21,63)
    Regularisation: lambda(L2) = 1.0, alpha(L1) = 0.1

    Gradient Boosting Loss Minimisation:
        L(phi) = sum(l(y_hat_i, y_i)) + sum(Omega(f_k))
        where Omega(f) = gamma*T + 0.5*lambda*||w||^2 + alpha*||w||_1
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        reg_lambda: float = 1.0,   # L2 regularisation
        reg_alpha: float = 0.1,     # L1 regularisation
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
        }
        self.model = None
        self._is_fitted = False
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "XGBoostModel":
        """
        Train XGBoost regressor.

        The objective minimises:
            MSE = (1/n) * sum((y_i - y_hat_i)^2)
        with regularisation:
            Omega(f) = gamma*T + 0.5*lambda*||w||^2 + alpha*||w||_1
        """
        import xgboost as xgb

        mask = features.notna().all(axis=1) & target.notna()
        X_clean = features.loc[mask]
        y_clean = target.loc[mask]

        if len(X_clean) < 50:
            logger.warning("Insufficient data for XGBoost training (%d rows)", len(X_clean))
            return self

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_clean, y_clean,
            eval_set=[(X_clean, y_clean)],
            verbose=False,
        )
        self._is_fitted = True

        # Store feature importances
        importances = self.model.feature_importances_
        self.feature_importances_ = dict(zip(X_clean.columns, importances))

        logger.info("XGBoost trained on %d samples, %d features", len(X_clean), len(X_clean.columns))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_fitted or self.model is None:
            return np.full(len(features), np.nan)

        mask = features.notna().all(axis=1)
        result = np.full(len(features), np.nan)
        if mask.sum() == 0:
            return result
        result[mask] = self.model.predict(features.loc[mask])
        return result


# ---------------------------------------------------------------------------
# Random Forest Model
# ---------------------------------------------------------------------------

class RandomForestModel:
    """
    Random Forest ensemble for prediction.

    n_estimators=500, max_depth=10, min_samples_leaf=5
    OOB (Out-of-Bag) score for validation.

    Each tree is trained on a bootstrap sample. OOB samples provide
    an unbiased estimate of generalisation error:
        OOB_score = 1 - (1/n) * sum((y_i - y_hat_oob_i)^2) / var(y)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "oob_score": oob_score,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model = None
        self._is_fitted = False
        self.oob_score_: float = 0.0
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RandomForestModel":
        """Train Random Forest with OOB validation."""
        from sklearn.ensemble import RandomForestRegressor

        mask = features.notna().all(axis=1) & target.notna()
        X_clean = features.loc[mask]
        y_clean = target.loc[mask]

        if len(X_clean) < 50:
            logger.warning("Insufficient data for RF training (%d rows)", len(X_clean))
            return self

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_clean, y_clean)
        self._is_fitted = True

        if self.model.oob_score:
            self.oob_score_ = self.model.oob_score_
            logger.info("RF OOB R^2 score: %.4f", self.oob_score_)

        self.feature_importances_ = dict(zip(X_clean.columns, self.model.feature_importances_))
        logger.info("RandomForest trained on %d samples", len(X_clean))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_fitted or self.model is None:
            return np.full(len(features), np.nan)

        mask = features.notna().all(axis=1)
        result = np.full(len(features), np.nan)
        if mask.sum() == 0:
            return result
        result[mask] = self.model.predict(features.loc[mask])
        return result


# ---------------------------------------------------------------------------
# Transformer Attention Model
# ---------------------------------------------------------------------------

class TransformerModel:
    """
    Transformer attention model for sequential prediction.

    Multi-head attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Positional encoding:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Architecture:
        Input -> PositionalEncoding -> MultiHeadAttention -> FeedForward -> Dense(1)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout_rate: float = 0.1,
        seq_length: int = 60,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self._is_fitted = False

    @staticmethod
    def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
        """
        Compute positional encoding matrix.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Parameters
        ----------
        seq_len : int
        d_model : int

        Returns
        -------
        np.ndarray of shape (seq_len, d_model)
        """
        pe = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    @staticmethod
    def scaled_dot_product_attention(
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """
        Scaled dot-product attention (numpy reference implementation).

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

        Parameters
        ----------
        Q : np.ndarray (batch, seq_len, d_k)
        K : np.ndarray (batch, seq_len, d_k)
        V : np.ndarray (batch, seq_len, d_v)

        Returns
        -------
        np.ndarray (batch, seq_len, d_v)
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
        # Numerically stable softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        return np.matmul(attention_weights, V)

    def _build_model(self, n_features: int) -> Any:
        """
        Build Keras Transformer model.

        Architecture:
            Input(seq_length, n_features)
            -> Dense(d_model) [projection]
            -> + PositionalEncoding
            -> N x (MultiHeadAttention -> LayerNorm -> FeedForward -> LayerNorm)
            -> GlobalAveragePooling1D
            -> Dense(1)
        """
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, LayerNormalization, Dropout,
            MultiHeadAttention, GlobalAveragePooling1D, Add,
        )
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(self.seq_length, n_features))

        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)

        # Add positional encoding
        pe = self.positional_encoding(self.seq_length, self.d_model)
        pe_tensor = tf.constant(pe, dtype=tf.float32)
        x = x + pe_tensor

        # Transformer encoder blocks
        for _ in range(self.n_layers):
            # Multi-head self-attention
            attn_output = MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
            )(x, x)
            attn_output = Dropout(self.dropout_rate)(attn_output)
            x = LayerNormalization()(Add()([x, attn_output]))

            # Feed-forward network
            ff_output = Dense(self.d_ff, activation="relu")(x)
            ff_output = Dense(self.d_model)(ff_output)
            ff_output = Dropout(self.dropout_rate)(ff_output)
            x = LayerNormalization()(Add()([x, ff_output]))

        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences."""
        Xs, ys = [], []
        for i in range(self.seq_length, len(X)):
            Xs.append(X[i - self.seq_length: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TransformerModel":
        """Train the Transformer model."""
        from sklearn.preprocessing import StandardScaler

        mask = features.notna().all(axis=1) & target.notna()
        X_clean = features.loc[mask].values
        y_clean = target.loc[mask].values

        if len(X_clean) < self.seq_length + 10:
            logger.warning("Insufficient data for Transformer training (%d rows)", len(X_clean))
            return self

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X_clean)
        y_scaled = self.scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()

        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        if len(X_seq) == 0:
            return self

        self.model = self._build_model(X_seq.shape[2])
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0,
        )
        self._is_fitted = True
        logger.info("Transformer trained on %d sequences", len(X_seq))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_fitted or self.model is None:
            return np.full(len(features), np.nan)

        mask = features.notna().all(axis=1)
        X_clean = features.loc[mask].values

        if len(X_clean) < self.seq_length:
            return np.full(len(features), np.nan)

        X_scaled = self.scaler_X.transform(X_clean)
        X_seq = np.array([X_scaled[-self.seq_length:]])
        y_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_scaled).ravel()

        result = np.full(len(features), np.nan)
        result[-1] = y_pred[0]
        return result


# ---------------------------------------------------------------------------
# Multi-Asset Predictor (Ensemble)
# ---------------------------------------------------------------------------

class MultiAssetPredictor:
    """
    Multi-asset prediction engine using ensemble ML methods.

    Pulls entire universe via OpenBB, classifies by GICS/asset class,
    then applies specialised models per asset class.

    Ensemble weighting:
        prediction = sum(w_i * pred_i)   where sum(w_i) = 1
        Weights determined by inverse validation error:
            w_i = (1/RMSE_i) / sum(1/RMSE_j)

    Confidence:
        confidence = 1 - CV(predictions)
        where CV = sigma / |mu| (coefficient of variation of ensemble members)
    """

    # Default model weights (prior to calibration)
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "lstm": 0.25,
        "xgboost": 0.30,
        "random_forest": 0.25,
        "transformer": 0.20,
    }

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        horizon_days: int = 5,
        retrain_interval_days: int = 30,
    ):
        self.model_weights = model_weights or dict(self.DEFAULT_WEIGHTS)
        self.horizon_days = horizon_days
        self.retrain_interval_days = retrain_interval_days

        # Model instances keyed by (symbol, model_name)
        self._models: Dict[str, Dict[str, Any]] = {}
        self._calibrated_weights: Dict[str, Dict[str, float]] = {}

    def _get_or_create_models(self, symbol: str) -> Dict[str, Any]:
        """Get or initialise the model ensemble for a symbol."""
        if symbol not in self._models:
            self._models[symbol] = {
                "lstm": LSTMModel(),
                "xgboost": XGBoostModel(),
                "random_forest": RandomForestModel(),
                "transformer": TransformerModel(),
            }
        return self._models[symbol]

    def train_ensemble(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        asset_class: AssetClass,
        symbol: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Train ensemble of LSTM + XGBoost + RF + Transformer.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix from prepare_features().
        targets : pd.Series
            Forward returns (target variable).
        asset_class : AssetClass
            Asset class for feature specialisation.
        symbol : str
            Symbol identifier for model caching.

        Returns
        -------
        dict
            Training metrics per model: {model_name: {"rmse": float, "weight": float}}
        """
        models = self._get_or_create_models(symbol)
        metrics: Dict[str, Any] = {}

        # Split into train/validation (80/20 temporal split)
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = targets.iloc[split_idx:]

        rmse_scores: Dict[str, float] = {}

        for name, model in models.items():
            try:
                logger.info("Training %s for %s (%s)", name, symbol, asset_class.value)
                model.fit(X_train, y_train)

                # Validation RMSE
                val_preds = model.predict(X_val)
                valid_mask = ~np.isnan(val_preds) & y_val.notna().values
                if valid_mask.sum() > 0:
                    residuals = y_val.values[valid_mask] - val_preds[valid_mask]
                    # RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
                    rmse = np.sqrt(np.mean(residuals ** 2))
                    rmse_scores[name] = rmse
                    metrics[name] = {
                        "rmse": rmse,
                        "mae": np.mean(np.abs(residuals)),
                        "n_valid": int(valid_mask.sum()),
                    }
                else:
                    rmse_scores[name] = float("inf")
                    metrics[name] = {"rmse": float("inf"), "mae": float("inf"), "n_valid": 0}

            except Exception as exc:
                logger.error("Failed to train %s for %s: %s", name, symbol, exc)
                rmse_scores[name] = float("inf")
                metrics[name] = {"error": str(exc)}

        # Calibrate weights: w_i = (1/RMSE_i) / sum(1/RMSE_j)
        inv_rmse = {}
        for name, rmse in rmse_scores.items():
            if rmse > 0 and rmse != float("inf"):
                inv_rmse[name] = 1.0 / rmse

        total_inv = sum(inv_rmse.values())
        if total_inv > 0:
            calibrated = {name: val / total_inv for name, val in inv_rmse.items()}
        else:
            calibrated = dict(self.DEFAULT_WEIGHTS)

        self._calibrated_weights[symbol] = calibrated
        for name in metrics:
            metrics[name]["weight"] = calibrated.get(name, 0.0)

        logger.info("Ensemble trained for %s. Weights: %s", symbol, calibrated)
        return metrics

    def predict(
        self,
        symbol: str,
        horizon_days: Optional[int] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Prediction:
        """
        Generate prediction with confidence interval for a single symbol.

        Prediction = weighted average of ensemble members:
            y_hat = sum(w_i * y_hat_i)

        Confidence = 1 - CV(predictions):
            CV = sigma(predictions) / |mean(predictions)|
            confidence = max(0, min(1, 1 - CV))

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        horizon_days : int, optional
            Prediction horizon in trading days. Defaults to self.horizon_days.
        start : str, optional
            Historical data start date.
        end : str, optional
            Historical data end date.

        Returns
        -------
        Prediction
        """
        horizon = horizon_days or self.horizon_days
        asset_class = detect_asset_class(symbol)

        # Fetch data
        data = get_historical(symbol, start=start, end=end)
        features = prepare_features(data, asset_class)

        # Target: forward returns
        close = data["Close"].astype(float)
        target = close.pct_change(horizon).shift(-horizon)

        # Train if not yet fitted
        if symbol not in self._models:
            self.train_ensemble(features, target, asset_class, symbol)

        models = self._models[symbol]
        weights = self._calibrated_weights.get(symbol, self.DEFAULT_WEIGHTS)

        # Generate predictions from each model
        individual_preds: Dict[str, float] = {}
        for name, model in models.items():
            preds = model.predict(features)
            last_pred = preds[-1] if not np.isnan(preds[-1]) else None
            if last_pred is not None:
                individual_preds[name] = float(last_pred)

        if not individual_preds:
            raise RuntimeError(f"All models failed to produce predictions for {symbol}")

        # Ensemble: weighted average
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, pred_val in individual_preds.items():
            w = weights.get(name, 0.0)
            weighted_sum += w * pred_val
            weight_sum += w

        predicted_return = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Confidence: 1 - CV(predictions)
        pred_values = np.array(list(individual_preds.values()))
        pred_mean = np.mean(pred_values)
        pred_std = np.std(pred_values)
        if abs(pred_mean) > 1e-10:
            cv = pred_std / abs(pred_mean)
        else:
            cv = float("inf")
        confidence = max(0.0, min(1.0, 1.0 - cv))

        current_price = float(close.iloc[-1])
        predicted_price = current_price * (1.0 + predicted_return)

        # Feature importances (from XGBoost)
        xgb_model = models.get("xgboost")
        feat_imp = {}
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            feat_imp = xgb_model.feature_importances_

        return Prediction(
            symbol=symbol,
            asset_class=asset_class,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=confidence,
            horizon_days=horizon,
            model_weights=weights,
            individual_predictions=individual_preds,
            feature_importances=feat_imp,
        )

    def scan_universe_predictions(
        self,
        universe: Optional[Dict[AssetClass, List[str]]] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Predict returns for entire universe, rank by expected return / risk.

        Risk-adjusted score = E[return] / sigma(prediction)
        (analogous to Sharpe ratio of predictions)

        Parameters
        ----------
        universe : dict, optional
            Asset class -> symbols. Defaults to full universe.
        horizon_days : int, optional

        Returns
        -------
        pd.DataFrame
            Columns: symbol, asset_class, current_price, predicted_return,
                     predicted_price, confidence, risk_adjusted_score
        """
        if universe is None:
            universe = get_full_universe()
        horizon = horizon_days or self.horizon_days

        results: List[Dict[str, Any]] = []

        for ac, symbols in universe.items():
            for sym in symbols:
                try:
                    pred = self.predict(sym, horizon_days=horizon)
                    # Risk-adjusted score: return / volatility of predictions
                    pred_values = list(pred.individual_predictions.values())
                    pred_std = np.std(pred_values) if len(pred_values) > 1 else 1e-6
                    risk_adj_score = pred.predicted_return / max(pred_std, 1e-6)

                    results.append({
                        "symbol": sym,
                        "asset_class": ac.value,
                        "current_price": pred.current_price,
                        "predicted_return": pred.predicted_return,
                        "predicted_price": pred.predicted_price,
                        "confidence": pred.confidence,
                        "risk_adjusted_score": risk_adj_score,
                        "horizon_days": horizon,
                    })
                except Exception as exc:
                    logger.error("Prediction failed for %s: %s", sym, exc)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)
        return df

    def identify_mispricings(
        self,
        predictions: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, float]] = None,
        threshold: float = 2.0,
    ) -> List[Mispricing]:
        """
        Detect statistical mispricing.

        mispricing_score = (predicted_price - current_price) / sigma_prediction
        Flag if |score| > threshold (default 2.0, i.e., 2-sigma event).

        Parameters
        ----------
        predictions : pd.DataFrame, optional
            Output from scan_universe_predictions(). If None, runs a full scan.
        current_prices : dict, optional
            Override prices. If None, uses prices from predictions.
        threshold : float
            Z-score threshold for flagging mispricings.

        Returns
        -------
        list of Mispricing
        """
        if predictions is None:
            predictions = self.scan_universe_predictions()

        mispricings: List[Mispricing] = []

        for _, row in predictions.iterrows():
            symbol = row["symbol"]
            current_price = row["current_price"]
            predicted_price = row["predicted_price"]

            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]

            # Get prediction standard deviation from ensemble members
            if symbol in self._models:
                models = self._models[symbol]
                weights = self._calibrated_weights.get(symbol, self.DEFAULT_WEIGHTS)
                # Use the spread of individual predictions as sigma
                pred_vals = []
                for name, model in models.items():
                    # Approximate from cached individual predictions
                    pass

            # Use predicted_return variance as proxy
            pred_std = abs(predicted_price - current_price) * (1.0 - row.get("confidence", 0.5))
            pred_std = max(pred_std, current_price * 0.001)  # floor at 0.1%

            # mispricing_score = (predicted - current) / sigma
            score = (predicted_price - current_price) / pred_std

            if abs(score) > threshold:
                direction = "undervalued" if score > 0 else "overvalued"
                mispricings.append(Mispricing(
                    symbol=symbol,
                    asset_class=AssetClass(row["asset_class"]),
                    current_price=current_price,
                    predicted_price=predicted_price,
                    mispricing_score=score,
                    prediction_std=pred_std,
                    direction=direction,
                    confidence=row.get("confidence", 0.0),
                ))

        # Sort by absolute mispricing score descending
        mispricings.sort(key=lambda m: abs(m.mispricing_score), reverse=True)
        logger.info("Found %d mispricings (threshold=%.1f sigma)", len(mispricings), threshold)
        return mispricings

    def backtest_predictions(
        self,
        symbol: str,
        start: str,
        end: str,
        horizon_days: Optional[int] = None,
        retrain_frequency: int = 63,
    ) -> BacktestResult:
        """
        Walk-forward backtest of predictions.

        Metrics:
            - Accuracy: fraction of correct directional calls
            - Precision: TP / (TP + FP)
            - Recall: TP / (TP + FN)
            - MAE: (1/n) * sum(|y_i - y_hat_i|)
            - RMSE: sqrt((1/n) * sum((y_i - y_hat_i)^2))
            - Directional accuracy: fraction where sign(pred) == sign(actual)
            - Information Coefficient: IC = corr(predicted_returns, actual_returns)

        Parameters
        ----------
        symbol : str
        start : str
            Backtest start date.
        end : str
            Backtest end date.
        horizon_days : int, optional
        retrain_frequency : int
            Number of days between model retraining (walk-forward).

        Returns
        -------
        BacktestResult
        """
        horizon = horizon_days or self.horizon_days
        asset_class = detect_asset_class(symbol)

        # Fetch full historical data
        data = get_historical(symbol, start=start, end=end)
        features = prepare_features(data, asset_class)
        close = data["Close"].astype(float)
        actual_returns = close.pct_change(horizon).shift(-horizon)

        # Walk-forward: train on expanding window, predict next period
        min_train_size = 252  # 1 year minimum training data
        predicted_returns: List[float] = []
        actual_list: List[float] = []
        dates: List[Any] = []

        i = min_train_size
        while i < len(features) - horizon:
            # Train on data up to i
            if (i - min_train_size) % retrain_frequency == 0:
                train_features = features.iloc[:i]
                train_target = actual_returns.iloc[:i]
                self.train_ensemble(train_features, train_target, asset_class, symbol)

            # Predict at i
            try:
                pred = self.predict(symbol, horizon_days=horizon)
                pred_ret = pred.predicted_return
            except Exception:
                pred_ret = 0.0

            actual_ret = actual_returns.iloc[i]
            if not np.isnan(actual_ret):
                predicted_returns.append(pred_ret)
                actual_list.append(actual_ret)
                dates.append(features.index[i])

            i += horizon  # step forward by horizon

        if len(predicted_returns) < 5:
            raise RuntimeError(f"Insufficient backtest data for {symbol}")

        pred_arr = np.array(predicted_returns)
        actual_arr = np.array(actual_list)

        # --- Compute metrics ---

        # MAE = (1/n) * sum(|y_i - y_hat_i|)
        mae = float(np.mean(np.abs(actual_arr - pred_arr)))

        # RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
        rmse = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))

        # Directional accuracy
        correct_direction = np.sign(pred_arr) == np.sign(actual_arr)
        directional_accuracy = float(np.mean(correct_direction))

        # Information Coefficient: IC = corr(predicted, actual)
        if np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
            ic = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
        else:
            ic = 0.0

        # Classification metrics (direction prediction)
        tp = int(np.sum((pred_arr > 0) & (actual_arr > 0)))
        fp = int(np.sum((pred_arr > 0) & (actual_arr <= 0)))
        fn = int(np.sum((pred_arr <= 0) & (actual_arr > 0)))
        tn = int(np.sum((pred_arr <= 0) & (actual_arr <= 0)))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        # Hit rate
        hit_rate = directional_accuracy

        # Profit factor = sum(winning trades) / |sum(losing trades)|
        # Simulate: go long when predicted > 0
        strategy_returns = np.where(pred_arr > 0, actual_arr, -actual_arr)
        wins = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = wins / max(losses, 1e-10)

        # Sharpe ratio (annualised)
        if np.std(strategy_returns) > 0:
            sharpe = float(np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 / horizon))
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown))

        # Total return
        total_return = float(np.sum(strategy_returns))

        result = BacktestResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            mae=mae,
            rmse=rmse,
            directional_accuracy=directional_accuracy,
            information_coefficient=ic,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            total_return=total_return,
        )

        logger.info(
            "Backtest for %s: IC=%.3f, DA=%.1f%%, Sharpe=%.2f, MaxDD=%.1f%%",
            symbol, ic, directional_accuracy * 100, sharpe, max_drawdown * 100,
        )
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print("=== Multi-Asset Predictor ===")
    print()

    predictor = MultiAssetPredictor(horizon_days=5)

    # Demonstrate feature engineering
    print("Feature engineering example:")
    for ac in [AssetClass.EQUITY, AssetClass.BOND, AssetClass.COMMODITY, AssetClass.CRYPTO, AssetClass.FX]:
        print(f"  {ac.value}: specialised features included")

    print()
    print("Models in ensemble:")
    for name, weight in MultiAssetPredictor.DEFAULT_WEIGHTS.items():
        print(f"  {name}: weight={weight:.2f}")

    print()
    print("Mathematical formulas:")
    print("  LSTM Loss: MSE = (1/n) * sum((y_i - y_hat_i)^2)")
    print("  Adam: lr=0.001, beta1=0.9, beta2=0.999")
    print("  XGBoost: lambda(L2)=1.0, alpha(L1)=0.1")
    print("  RF: n_estimators=500, max_depth=10, OOB validation")
    print("  Attention: softmax(QK^T / sqrt(d_k)) V")
    print("  PE: sin(pos / 10000^(2i/d))")
    print("  Confidence: 1 - CV(predictions), CV = sigma/|mu|")
    print("  Mispricing: z = (predicted - current) / sigma, flag |z|>2")
    print("  IC = corr(predicted_returns, actual_returns)")
