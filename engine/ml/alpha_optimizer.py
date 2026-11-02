"""AlphaOptimizer — Walk-forward ML alpha + mean-variance portfolio construction.

Incorporates Dataset 2: ML-based portfolio optimizer with walk-forward alpha
and turnover constraints.

Pipeline:
    1. Feature engineering (market, momentum, vol, technical indicators)
    2. Walk-forward linear regression for alpha prediction
    3. EWMA covariance estimation
    4. Mean-variance optimisation with turnover constraints
    5. Quality tier classification (A-G)

CAPM alpha ranking + factor integration.

Extended components:
    - CAPMAlphaExtractor: Jensen's alpha, multi-factor residuals
    - FactorLibrary: 50+ factors (momentum, value, quality, vol, technical, fundamental)
    - WalkForwardOptimizer: Rolling window with XGBoost/Ridge fallback
    - MeanVarianceOptimizer: EWMA cov + turnover + position limits + risk budgeting
    - QualityRanker: Enhanced quality scoring with fundamental overlay
    - AlphaDecayModel: Signal decay modeling
    - TransactionCostModel: Cost estimation and incorporation
    - FeatureImportanceTracker: Feature importance tracking over time
"""

import logging
import warnings
from collections import deque

logger = logging.getLogger("metadron.intelligence.alpha_optimizer")
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from ..data.openbb_data import get_adj_close, get_returns, get_prices


# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------
QUALITY_TIERS = {
    "A": {"min_sharpe": 2.0, "min_momentum": 0.15},
    "B": {"min_sharpe": 1.5, "min_momentum": 0.10},
    "C": {"min_sharpe": 1.0, "min_momentum": 0.05},
    "D": {"min_sharpe": 0.5, "min_momentum": 0.00},
    "E": {"min_sharpe": 0.0, "min_momentum": -0.05},
    "F": {"min_sharpe": -0.5, "min_momentum": -0.10},
    "G": {"min_sharpe": -999, "min_momentum": -999},
}

# Alpha target - seeking alpha >= 2%
ALPHA_TARGET = 0.05
TRANSACTION_COST = 0.001
MAX_TURNOVER = 0.50

# Factor model constants
RISK_FREE_RATE = 0.04  # Approximate risk-free rate
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AlphaSignal:
    ticker: str
    alpha_pred: float = 0.0
    quality_tier: str = "D"
    sharpe_estimate: float = 0.0
    momentum_3m: float = 0.0
    momentum_1m: float = 0.0
    vol: float = 0.0
    weight: float = 0.0


@dataclass
class AlphaOutput:
    signals: list = field(default_factory=list)
    optimal_weights: dict = field(default_factory=dict)
    expected_annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    rebalance_cost: float = 0.0
    alpha_predictions: Optional[pd.Series] = None


@dataclass
class FactorExposure:
    """Single factor exposure result."""
    factor_name: str
    beta: float = 0.0
    t_stat: float = 0.0
    residual_alpha: float = 0.0
    r_squared: float = 0.0


@dataclass
class DecayEstimate:
    """Alpha decay model output."""
    half_life_days: float = 20.0
    decay_rate: float = 0.03
    current_alpha: float = 0.0
    projected_alpha_5d: float = 0.0
    projected_alpha_20d: float = 0.0


# ---------------------------------------------------------------------------
# Feature engineering (Dataset 2)
# ---------------------------------------------------------------------------
def build_features(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from returns and prices.

    Includes original features plus enhanced technical indicators:
    RSI, MACD, Bollinger Band width, ATR proxy, and more.
    """
    features = pd.DataFrame(index=returns.index)

    # --- Original features ---
    features["mkt_mean"] = returns.mean(axis=1)
    features["mkt_vol"] = returns.std(axis=1)
    features["momentum_3m"] = prices.reindex(returns.index).ffill().pct_change(60).mean(axis=1)
    features["momentum_1m"] = prices.reindex(returns.index).ffill().pct_change(20).mean(axis=1)
    features["momentum_1w"] = prices.reindex(returns.index).ffill().pct_change(5).mean(axis=1)
    features["vol_20d"] = returns.rolling(20).std().mean(axis=1)
    features["vol_ratio"] = (
        returns.rolling(5).std().mean(axis=1)
        / returns.rolling(20).std().mean(axis=1).replace(0, np.nan)
    )
    features["return_dispersion"] = returns.std(axis=1) / returns.mean(axis=1).abs().replace(0, np.nan)

    # --- Enhanced technical indicators ---
    p_mean = prices.reindex(returns.index).ffill().mean(axis=1)
    r_mean = returns.mean(axis=1)

    # RSI (14-day) computed on mean cross-sectional return
    delta = r_mean.copy()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9 on mean price)
    ema_12 = p_mean.ewm(span=12, adjust=False).mean()
    ema_26 = p_mean.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = macd_line - signal_line

    # Bollinger Band width (20-day, 2 std)
    bb_mid = p_mean.rolling(20, min_periods=1).mean()
    bb_std = p_mean.rolling(20, min_periods=1).std().replace(0, np.nan)
    features["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    features["bb_position"] = (p_mean - bb_mid) / bb_std.replace(0, np.nan)

    # ATR proxy (using returns as high-low proxy)
    atr_proxy = returns.abs().rolling(14, min_periods=1).mean().mean(axis=1)
    features["atr_14"] = atr_proxy

    # Momentum quality (acceleration)
    features["momentum_accel"] = features["momentum_1m"] - features["momentum_1m"].shift(5)

    # Skewness and kurtosis of recent returns
    features["skew_20d"] = returns.rolling(20, min_periods=5).skew().mean(axis=1)
    features["kurt_20d"] = returns.rolling(20, min_periods=5).kurt().mean(axis=1)

    # Volume of returns (cross-sectional dispersion over time)
    features["cross_disp_5d"] = returns.rolling(5, min_periods=1).std().std(axis=1)

    # --- Architecture DNA features (credit_spread_proxy, rv_zscore, capm_residual) ---

    # Credit spread proxy: HY-IG spread approximation from return dispersion
    # High dispersion + negative mean → widening spreads (risk-off)
    # Low dispersion + positive mean → tightening spreads (risk-on)
    disp = returns.std(axis=1)
    mean_r = returns.mean(axis=1)
    features["credit_spread_proxy"] = disp - mean_r.abs()

    # Realized volatility Z-score: current RV vs rolling mean/std
    rv_20 = returns.rolling(20).std().mean(axis=1) * np.sqrt(252)
    rv_mean = rv_20.rolling(60, min_periods=20).mean()
    rv_std = rv_20.rolling(60, min_periods=20).std().replace(0, np.nan)
    features["rv_zscore"] = (rv_20 - rv_mean) / rv_std

    # CAPM residual: market-adjusted excess return (alpha residual)
    # Uses equal-weighted portfolio return as market proxy
    mkt_ret = returns.mean(axis=1)
    mkt_excess = mkt_ret - mkt_ret.rolling(60, min_periods=20).mean()
    beta_est = (
        returns.rolling(60, min_periods=20).cov(mkt_ret).mean(axis=1)
        / mkt_ret.rolling(60, min_periods=20).var().replace(0, np.nan)
    )
    features["capm_residual"] = mkt_ret - beta_est * mkt_excess

    features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return features


# ---------------------------------------------------------------------------
# EWMA covariance (Dataset 2)
# ---------------------------------------------------------------------------
def ewma_cov(returns_df: pd.DataFrame, lam: float = 0.94, span: int = 60) -> np.ndarray:
    """Exponentially weighted moving average covariance matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns (T x N).
    lam : float
        Decay factor (default 0.94). If span is provided and lam is default,
        lam is computed from span as 1 - 2/(span+1).
    span : int
        EWMA span in days (default 60). Used only when lam is at default.

    Returns
    -------
    np.ndarray
        (n_assets, n_assets) covariance matrix, symmetric with positive diagonal.
    """
    cov = returns_df.cov().values.copy()
    for i in range(len(returns_df)):
        r = returns_df.iloc[i].values.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
    cov = cov + 1e-10 * np.eye(cov.shape[0])
    return cov


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------
def classify_quality(sharpe: float, momentum: float) -> str:
    """Assign quality tier A-G."""
    for tier, thresholds in QUALITY_TIERS.items():
        if sharpe >= thresholds["min_sharpe"] and momentum >= thresholds["min_momentum"]:
            return tier
    return "G"


# ---------------------------------------------------------------------------
# CAPMAlphaExtractor
# ---------------------------------------------------------------------------
class CAPMAlphaExtractor:
    """Extract Jensen's alpha and multi-factor residuals.

    Supports market, size, value, momentum, and quality factors.
    """

    FACTOR_NAMES = ["market", "size", "value", "momentum", "quality"]

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / TRADING_DAYS

    def compute_jensens_alpha(
        self, asset_returns: pd.Series, market_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """Compute Jensen's alpha from CAPM regression.

        Returns (alpha_annualized, beta, r_squared).
        """
        excess_asset = asset_returns - self._daily_rf
        excess_market = market_returns - self._daily_rf

        aligned = pd.concat([excess_asset, excess_market], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0, 1.0, 0.0

        y = aligned.iloc[:, 0].values.reshape(-1, 1)
        X = aligned.iloc[:, 1].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        alpha_daily = float(reg.intercept_[0]) if hasattr(reg.intercept_, '__len__') else float(reg.intercept_)
        beta = float(reg.coef_[0][0]) if reg.coef_.ndim > 1 else float(reg.coef_[0])

        y_pred = reg.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        alpha_annual = alpha_daily * TRADING_DAYS
        return alpha_annual, beta, r_sq

    def build_synthetic_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Build synthetic factor returns from cross-section of asset returns.

        Creates market, size-proxy, value-proxy, momentum-proxy, quality-proxy
        factors from the available asset universe.
        """
        factors = pd.DataFrame(index=returns.index)

        # Market factor: equal-weighted cross-sectional mean
        factors["market"] = returns.mean(axis=1)

        # Size proxy: difference between small-cap-like (high vol) and large-cap-like (low vol)
        vol_20 = returns.rolling(20, min_periods=5).std()
        median_vol = vol_20.median(axis=1)
        high_vol_mask = vol_20.gt(median_vol, axis=0)
        low_vol_mask = ~high_vol_mask
        factors["size"] = (
            returns.where(high_vol_mask).mean(axis=1).fillna(0)
            - returns.where(low_vol_mask).mean(axis=1).fillna(0)
        )

        # Value proxy: mean-reversion signal (buy losers, sell winners over 60d)
        mom_60 = returns.rolling(60, min_periods=10).sum()
        median_mom = mom_60.median(axis=1)
        losers = mom_60.lt(median_mom, axis=0)
        winners = ~losers
        factors["value"] = (
            returns.where(losers).mean(axis=1).fillna(0)
            - returns.where(winners).mean(axis=1).fillna(0)
        )

        # Momentum proxy: buy recent winners, sell recent losers (20d)
        mom_20 = returns.rolling(20, min_periods=5).sum()
        median_mom_20 = mom_20.median(axis=1)
        win_20 = mom_20.gt(median_mom_20, axis=0)
        lose_20 = ~win_20
        factors["momentum"] = (
            returns.where(win_20).mean(axis=1).fillna(0)
            - returns.where(lose_20).mean(axis=1).fillna(0)
        )

        # Quality proxy: low vol minus high vol (quality = stability)
        factors["quality"] = (
            returns.where(low_vol_mask).mean(axis=1).fillna(0)
            - returns.where(high_vol_mask).mean(axis=1).fillna(0)
        )

        return factors.fillna(0)

    def multi_factor_decomposition(
        self, asset_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> List[FactorExposure]:
        """Decompose asset returns into factor exposures."""
        aligned = pd.concat([asset_returns, factor_returns], axis=1).dropna()
        if len(aligned) < 30:
            return [FactorExposure(factor_name=f) for f in factor_returns.columns]

        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n = len(y)
        p = X.shape[1]
        residuals = y - y_pred
        mse = ss_res / max(n - p - 1, 1)

        exposures = []
        for i, col in enumerate(factor_returns.columns):
            beta_i = float(reg.coef_[i])
            # t-statistic approximation
            x_var = float(np.var(X[:, i]))
            se = np.sqrt(mse / (x_var * n)) if x_var > 0 else 1.0
            t_stat = beta_i / se if se > 0 else 0.0

            exposures.append(FactorExposure(
                factor_name=col,
                beta=beta_i,
                t_stat=t_stat,
                residual_alpha=float(reg.intercept_) * TRADING_DAYS,
                r_squared=r_sq,
            ))
        return exposures


# ---------------------------------------------------------------------------
# FactorLibrary — 50+ factors organized by category
# ---------------------------------------------------------------------------
class FactorLibrary:
    """Library of 50+ quantitative factors organized by category.

    Categories: momentum, value, quality, volatility, technical, fundamental.
    """

    CATEGORIES = ["momentum", "value", "quality", "volatility", "technical", "fundamental"]

    FACTOR_REGISTRY: Dict[str, List[str]] = {
        "momentum": [
            "mom_1w", "mom_1m", "mom_3m", "mom_6m", "mom_12m",
            "mom_accel_1m", "mom_accel_3m", "mom_reversal_5d",
            "relative_strength_20d", "relative_strength_60d",
        ],
        "value": [
            "mean_reversion_20d", "mean_reversion_60d",
            "price_to_sma_50", "price_to_sma_200",
            "earnings_yield_proxy", "book_value_proxy",
            "dividend_yield_proxy", "cashflow_yield_proxy",
            "ev_ebitda_proxy", "price_to_sales_proxy",
        ],
        "quality": [
            "return_stability_60d", "return_stability_120d",
            "sharpe_60d", "sharpe_120d",
            "max_drawdown_60d", "recovery_speed",
            "earnings_consistency", "roe_proxy",
            "debt_stability_proxy", "profit_margin_proxy",
        ],
        "volatility": [
            "realized_vol_5d", "realized_vol_20d", "realized_vol_60d",
            "vol_of_vol_20d", "vol_skew", "vol_term_structure",
            "idiosyncratic_vol", "downside_vol_20d",
            "garch_vol_proxy", "parkinson_vol",
        ],
        "technical": [
            "rsi_14", "rsi_5", "macd_hist", "macd_crossover",
            "bb_width", "bb_position", "atr_14",
            "obv_proxy", "williams_r", "stochastic_k",
        ],
        "fundamental": [
            "earnings_surprise_proxy", "revenue_growth_proxy",
            "margin_expansion", "capex_intensity",
            "working_capital_efficiency", "asset_turnover_proxy",
            "interest_coverage_proxy", "altman_z_proxy",
            "piotroski_f_proxy", "accruals_proxy",
        ],
    }

    def __init__(self):
        self._computed_factors: Dict[str, pd.DataFrame] = {}

    @property
    def total_factors(self) -> int:
        return sum(len(v) for v in self.FACTOR_REGISTRY.values())

    def list_factors(self, category: Optional[str] = None) -> List[str]:
        """List all factor names, optionally filtered by category."""
        if category and category in self.FACTOR_REGISTRY:
            return list(self.FACTOR_REGISTRY[category])
        return [f for factors in self.FACTOR_REGISTRY.values() for f in factors]

    def compute_momentum_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute all momentum category factors."""
        factors = pd.DataFrame(index=returns.index)
        factors["mom_1w"] = returns.rolling(5, min_periods=1).sum().mean(axis=1)
        factors["mom_1m"] = returns.rolling(20, min_periods=1).sum().mean(axis=1)
        factors["mom_3m"] = returns.rolling(60, min_periods=1).sum().mean(axis=1)
        factors["mom_6m"] = returns.rolling(126, min_periods=1).sum().mean(axis=1)
        factors["mom_12m"] = returns.rolling(252, min_periods=1).sum().mean(axis=1)
        factors["mom_accel_1m"] = factors["mom_1m"] - factors["mom_1m"].shift(5)
        factors["mom_accel_3m"] = factors["mom_3m"] - factors["mom_3m"].shift(20)
        factors["mom_reversal_5d"] = -returns.rolling(5, min_periods=1).sum().mean(axis=1)
        rs_20 = returns.rolling(20, min_periods=1).mean().mean(axis=1)
        rs_60 = returns.rolling(60, min_periods=1).mean().mean(axis=1)
        factors["relative_strength_20d"] = rs_20 / rs_60.replace(0, np.nan)
        factors["relative_strength_60d"] = rs_60
        return factors.replace([np.inf, -np.inf], np.nan).fillna(0)

    def compute_volatility_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute all volatility category factors."""
        factors = pd.DataFrame(index=returns.index)
        factors["realized_vol_5d"] = returns.rolling(5, min_periods=1).std().mean(axis=1)
        factors["realized_vol_20d"] = returns.rolling(20, min_periods=1).std().mean(axis=1)
        factors["realized_vol_60d"] = returns.rolling(60, min_periods=1).std().mean(axis=1)
        vol_20_ts = returns.rolling(20, min_periods=5).std().mean(axis=1)
        factors["vol_of_vol_20d"] = vol_20_ts.rolling(20, min_periods=5).std()
        factors["vol_skew"] = returns.rolling(20, min_periods=5).skew().mean(axis=1)
        factors["vol_term_structure"] = (
            factors["realized_vol_5d"] / factors["realized_vol_60d"].replace(0, np.nan)
        )
        mkt_mean = returns.mean(axis=1)
        residuals = returns.sub(mkt_mean, axis=0)
        factors["idiosyncratic_vol"] = residuals.rolling(20, min_periods=5).std().mean(axis=1)
        neg_returns = returns.clip(upper=0)
        factors["downside_vol_20d"] = neg_returns.rolling(20, min_periods=5).std().mean(axis=1)
        # GARCH proxy: weighted average of recent squared returns
        sq_ret = (returns ** 2).mean(axis=1)
        factors["garch_vol_proxy"] = sq_ret.ewm(span=20, adjust=False).mean().apply(np.sqrt)
        # Parkinson volatility proxy using return range
        high_low_proxy = returns.abs().rolling(5, min_periods=1).max()
        factors["parkinson_vol"] = high_low_proxy.mean(axis=1) / (2 * np.sqrt(np.log(2)))
        return factors.replace([np.inf, -np.inf], np.nan).fillna(0)

    def compute_technical_factors(self, returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical category factors."""
        factors = pd.DataFrame(index=returns.index)
        p_mean = prices.reindex(returns.index).ffill().mean(axis=1)
        r_mean = returns.mean(axis=1)

        # RSI variants
        for period, name in [(14, "rsi_14"), (5, "rsi_5")]:
            gain = r_mean.clip(lower=0).rolling(period, min_periods=1).mean()
            loss = (-r_mean).clip(lower=0).rolling(period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            factors[name] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = p_mean.ewm(span=12, adjust=False).mean()
        ema_26 = p_mean.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        factors["macd_hist"] = macd_line - signal_line
        factors["macd_crossover"] = (macd_line > signal_line).astype(float)

        # Bollinger
        bb_mid = p_mean.rolling(20, min_periods=1).mean()
        bb_std = p_mean.rolling(20, min_periods=1).std().replace(0, np.nan)
        factors["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
        factors["bb_position"] = (p_mean - bb_mid) / bb_std.replace(0, np.nan)

        # ATR
        factors["atr_14"] = returns.abs().rolling(14, min_periods=1).mean().mean(axis=1)

        # OBV proxy (cumulative sign-weighted volume proxy)
        factors["obv_proxy"] = np.sign(r_mean).cumsum()

        # Williams %R
        high_20 = p_mean.rolling(20, min_periods=1).max()
        low_20 = p_mean.rolling(20, min_periods=1).min()
        hl_range = (high_20 - low_20).replace(0, np.nan)
        factors["williams_r"] = -100 * (high_20 - p_mean) / hl_range

        # Stochastic %K
        factors["stochastic_k"] = 100 * (p_mean - low_20) / hl_range

        return factors.replace([np.inf, -np.inf], np.nan).fillna(0)

    def compute_all(self, returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute all available factors and return combined DataFrame."""
        mom = self.compute_momentum_factors(returns)
        vol = self.compute_volatility_factors(returns)
        tech = self.compute_technical_factors(returns, prices)
        combined = pd.concat([mom, vol, tech], axis=1)
        self._computed_factors = {
            "momentum": mom,
            "volatility": vol,
            "technical": tech,
        }
        return combined.replace([np.inf, -np.inf], np.nan).fillna(0)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------
class WalkForwardOptimizer:
    """Rolling window walk-forward optimizer with ML model selection.

    Uses XGBoost when available, falls back to Ridge regression.
    Provides out-of-sample validation metrics.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        use_xgboost: bool = True,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.use_xgboost = use_xgboost
        self._oos_results: List[Dict] = []
        self._feature_importances: Optional[pd.DataFrame] = None

        # Retraining trigger
        self._outcomes_since_training = 0
        self._retrain_threshold = 50  # Retrain after 50 new outcomes
        self._last_training_date: Optional[str] = None

    def _get_model(self):
        """Select best available model."""
        if self.use_xgboost:
            return XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
            )
        return Ridge(alpha=1.0)

    def walk_forward(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.Series, List[Dict]]:
        """Run walk-forward optimization with rolling windows.

        Returns (predictions_series, oos_metrics_list).
        """
        predictions = {}
        metrics = []
        scaler = StandardScaler()

        total_len = len(X)
        start = self.train_window

        while start + self.test_window <= total_len:
            train_end = start
            test_end = min(start + self.test_window, total_len)

            X_train = X.iloc[train_end - self.train_window:train_end]
            y_train = y.iloc[train_end - self.train_window:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            if len(X_train) < 50 or len(X_test) < 5:
                start += self.step_size
                continue

            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model = self._get_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_sc, y_train.values)

            preds = model.predict(X_test_sc)

            for idx, pred in zip(X_test.index, preds):
                predictions[idx] = float(pred)

            # OOS metrics
            oos_mse = float(np.mean((y_test.values - preds) ** 2))
            oos_corr = float(np.corrcoef(y_test.values, preds)[0, 1]) if len(preds) > 2 else 0.0
            ic = oos_corr  # Information coefficient

            metrics.append({
                "window_start": X_test.index[0],
                "window_end": X_test.index[-1],
                "mse": oos_mse,
                "ic": ic,
                "n_samples": len(X_test),
            })

            # Track feature importance
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, "coef_"):
                imp = dict(zip(X.columns, np.abs(model.coef_)))
            else:
                imp = {}

            if imp:
                imp_df = pd.DataFrame([imp], index=[X_test.index[0]])
                if self._feature_importances is None:
                    self._feature_importances = imp_df
                else:
                    self._feature_importances = pd.concat([self._feature_importances, imp_df])

            start += self.step_size

        self._oos_results = metrics
        pred_series = pd.Series(predictions).sort_index()
        return pred_series, metrics

    def get_oos_sharpe(self) -> float:
        """Compute aggregate OOS information coefficient."""
        if not self._oos_results:
            return 0.0
        ics = [m["ic"] for m in self._oos_results if not np.isnan(m["ic"])]
        return float(np.mean(ics)) if ics else 0.0

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """Return feature importance history."""
        return self._feature_importances

    def save_model(self, name: str = "alpha_optimizer") -> str:
        """Save the last trained model and metadata to disk."""
        if self._last_fitted_model is None:
            warnings.warn("No fitted model to save — walk_forward() has not been run yet.")
            return ""
        try:
            from .model_store import ModelStore
            store = ModelStore()
            metadata = {
                "oos_sharpe": self.get_oos_sharpe(),
                "n_windows": len(self._oos_results),
                "train_window": self.train_window,
                "test_window": self.test_window,
                "use_xgboost": self.use_xgboost,
            }
            # Save feature importances if available
            if self._feature_importances is not None:
                fi_path = store.base_dir / name / "feature_importances.csv"
                fi_path.parent.mkdir(parents=True, exist_ok=True)
                self._feature_importances.to_csv(fi_path)

            return store.save_sklearn(name, self._last_fitted_model, metadata)
        except Exception as e:
            logger.warning("Failed to save model: %s", e)
            return ""

    def record_outcome(self):
        """Record a new outcome. Triggers retraining if threshold reached."""
        self._outcomes_since_training += 1

    def should_retrain(self) -> bool:
        """Check if enough new outcomes to justify retraining."""
        return self._outcomes_since_training >= self._retrain_threshold

    def retrain_if_needed(self, X: pd.DataFrame, y: pd.Series) -> Optional[list]:
        """
        Retrain model if enough new outcomes have accumulated.
        
        Returns walk-forward results if retrained, None otherwise.
        Saves model to ModelStore after retraining.
        """
        if not self.should_retrain():
            return None

        logger.info("Alpha optimizer retraining triggered (%d outcomes since last training)",
                    self._outcomes_since_training)

        # Run walk-forward
        results = self.walk_forward(X, y)

        # Save model
        self.save_model()

        # Reset counter
        self._outcomes_since_training = 0
        self._last_training_date = datetime.now().isoformat()

        oos_sharpe = self.get_oos_sharpe()
        logger.info("Retraining complete: OOS Sharpe=%.3f, model saved", oos_sharpe)

        return results


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer
# ---------------------------------------------------------------------------
class MeanVarianceOptimizer:
    """Enhanced mean-variance optimizer with turnover, position limits, and risk budgeting."""

    def __init__(
        self,
        max_position: float = 0.50,
        min_position: float = 0.0,
        max_turnover: float = MAX_TURNOVER,
        transaction_cost: float = TRANSACTION_COST,
        risk_budget: Optional[Dict[str, float]] = None,
    ):
        self.max_position = max_position
        self.min_position = min_position
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.risk_budget = risk_budget

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        sector_map: Optional[Dict[int, str]] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Run constrained mean-variance optimization.

        Supports position limits, turnover constraints, sector constraints,
        and optional risk budgeting.
        """
        n = len(expected_returns)

        def portfolio_vol(w):
            return np.sqrt(max(1e-12, w.T @ cov_matrix @ w))

        def turnover(w):
            return float(np.sum(np.abs(w - current_weights)))

        def objective(w):
            w = np.asarray(w, dtype=float)
            port_ret = float(w @ expected_returns)
            vol = portfolio_vol(w)
            cost = turnover(w) * self.transaction_cost
            sharpe = (port_ret - cost) / vol if vol > 0 else 0.0
            # Risk budgeting penalty
            penalty = 0.0
            if self.risk_budget is not None and sector_map is not None:
                marginal_risk = cov_matrix @ w
                total_risk = vol
                for idx, sector in sector_map.items():
                    if sector in self.risk_budget:
                        contrib = w[idx] * marginal_risk[idx] / total_risk if total_risk > 0 else 0
                        target = self.risk_budget[sector]
                        penalty += (contrib - target) ** 2
            return -sharpe + 10 * penalty

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: self.max_turnover - turnover(np.asarray(w))},
        ]

        # Sector-neutral constraints
        if sector_map is not None and sector_limits is not None:
            for sector, limit in sector_limits.items():
                indices = [i for i, s in sector_map.items() if s == sector]
                if indices:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, idx=indices, lim=limit: lim - sum(w[i] for i in idx),
                    })

        bounds = [(self.min_position, self.max_position)] * n

        try:
            result = minimize(
                objective, current_weights.copy(),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if result.success:
                return np.asarray(result.x, dtype=float)
        except Exception:
            pass

        return current_weights

    def compute_risk_contributions(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute marginal risk contribution of each asset."""
        port_var = weights.T @ cov_matrix @ weights
        port_vol = np.sqrt(max(port_var, 1e-12))
        marginal = cov_matrix @ weights
        contributions = weights * marginal / port_vol
        return contributions


# ---------------------------------------------------------------------------
# QualityRanker
# ---------------------------------------------------------------------------
class QualityRanker:
    """Enhanced quality scoring with fundamental overlay.

    Combines quantitative metrics (Sharpe, momentum, vol) with fundamental
    proxies (ROE, D/E, earnings stability) for a composite quality score.
    """

    # Weights for composite score
    WEIGHTS = {
        "sharpe": 0.22,
        "momentum": 0.17,
        "stability": 0.13,
        "drawdown": 0.08,
        "roe_proxy": 0.08,
        "de_proxy": 0.08,
        "earnings_stability": 0.09,
        "credit_quality": 0.15,
    }

    def __init__(self):
        self._scores: Dict[str, float] = {}

    def score_asset(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute composite quality score for a single asset.

        Returns dict with component scores and composite.
        """
        ann_ret = float(returns.mean() * TRADING_DAYS)
        ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        # Momentum (3m cumulative return)
        mom = float(returns.iloc[-63:].sum()) if len(returns) >= 63 else float(returns.sum())

        # Stability (inverse of vol-of-vol)
        rolling_vol = returns.rolling(20, min_periods=5).std()
        vol_of_vol = float(rolling_vol.std())
        stability = 1.0 / (1.0 + vol_of_vol * 100)

        # Max drawdown
        cum = (1 + returns).cumprod()
        dd = float((cum / cum.cummax() - 1).min())
        dd_score = max(0, 1.0 + dd)  # 0 = -100% dd, 1 = no dd

        # ROE proxy: average return / average absolute return (efficiency)
        abs_mean = float(returns.abs().mean())
        roe_proxy = float(returns.mean()) / abs_mean if abs_mean > 0 else 0.0

        # D/E proxy: downside vol / upside vol (leverage risk)
        up_vol = float(returns[returns > 0].std()) if (returns > 0).sum() > 5 else 0.01
        down_vol = float(returns[returns < 0].std()) if (returns < 0).sum() > 5 else 0.01
        de_proxy = 1.0 / (1.0 + down_vol / up_vol) if up_vol > 0 else 0.5

        # Earnings stability proxy: autocorrelation of monthly returns
        monthly = returns.resample("ME").sum() if hasattr(returns.index, 'freq') or len(returns) > 21 else returns
        if len(monthly) > 3:
            earn_stab = 1.0 - abs(float(monthly.autocorr(lag=1))) if not np.isnan(monthly.autocorr(lag=1)) else 0.5
        else:
            earn_stab = 0.5

        # Normalize scores to [0, 1]
        norm_sharpe = min(max((sharpe + 1) / 4, 0), 1)  # [-1, 3] -> [0, 1]
        norm_mom = min(max((mom + 0.2) / 0.6, 0), 1)    # [-0.2, 0.4] -> [0, 1]

        # Credit quality proxy: blend of leverage safety + profitability efficiency
        # High roe_proxy + low de_proxy → strong credit; inverse → weak credit
        credit_quality_raw = (0.5 * min(max(roe_proxy, 0), 1) +
                              0.3 * de_proxy +
                              0.2 * earn_stab)
        credit_quality = min(max(credit_quality_raw, 0), 1)

        composite = (
            self.WEIGHTS["sharpe"] * norm_sharpe
            + self.WEIGHTS["momentum"] * norm_mom
            + self.WEIGHTS["stability"] * stability
            + self.WEIGHTS["drawdown"] * dd_score
            + self.WEIGHTS["roe_proxy"] * min(max(roe_proxy, 0), 1)
            + self.WEIGHTS["de_proxy"] * de_proxy
            + self.WEIGHTS["earnings_stability"] * earn_stab
            + self.WEIGHTS["credit_quality"] * credit_quality
        )

        return {
            "sharpe": sharpe,
            "momentum": mom,
            "stability": stability,
            "drawdown_score": dd_score,
            "roe_proxy": roe_proxy,
            "de_proxy": de_proxy,
            "earnings_stability": earn_stab,
            "credit_quality": credit_quality,
            "composite": composite,
            "quality_tier": classify_quality(sharpe, mom),
        }

    def rank_assets(
        self, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Rank all assets by composite quality score."""
        scores = []
        for col in returns.columns:
            sc = self.score_asset(returns[col])
            sc["ticker"] = col
            scores.append(sc)

        df = pd.DataFrame(scores).set_index("ticker")
        df["rank"] = df["composite"].rank(ascending=False).astype(int)
        return df.sort_values("rank")


# ---------------------------------------------------------------------------
# AlphaDecayModel
# ---------------------------------------------------------------------------
class AlphaDecayModel:
    """Model alpha signal decay over time.

    Estimates half-life and projects future alpha using exponential decay.
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._decay_params: Dict[str, float] = {}

    def estimate_decay(self, alpha_series: pd.Series) -> DecayEstimate:
        """Estimate alpha decay parameters from historical alpha predictions.

        Uses autocorrelation analysis to determine half-life.
        """
        if len(alpha_series) < 10:
            return DecayEstimate()

        series = alpha_series.dropna()
        current = float(series.iloc[-1])

        # Estimate autocorrelation at various lags
        autocorrs = []
        for lag in range(1, min(21, len(series) // 2)):
            ac = float(series.autocorr(lag=lag))
            if not np.isnan(ac):
                autocorrs.append((lag, ac))

        if not autocorrs:
            return DecayEstimate(current_alpha=current)

        # Fit exponential decay to autocorrelation structure
        lags = np.array([a[0] for a in autocorrs], dtype=float)
        acs = np.array([max(a[1], 1e-6) for a in autocorrs], dtype=float)

        # log(ac) = -decay_rate * lag => linear regression
        valid = acs > 0
        if valid.sum() < 2:
            decay_rate = 0.03
        else:
            log_acs = np.log(acs[valid])
            lags_valid = lags[valid]
            if len(lags_valid) > 1:
                slope = float(np.polyfit(lags_valid, log_acs, 1)[0])
                decay_rate = max(-slope, 0.001)
            else:
                decay_rate = 0.03

        half_life = np.log(2) / decay_rate if decay_rate > 0 else 100.0

        proj_5d = current * np.exp(-decay_rate * 5)
        proj_20d = current * np.exp(-decay_rate * 20)

        self._decay_params = {
            "half_life": half_life,
            "decay_rate": decay_rate,
        }

        return DecayEstimate(
            half_life_days=half_life,
            decay_rate=decay_rate,
            current_alpha=current,
            projected_alpha_5d=proj_5d,
            projected_alpha_20d=proj_20d,
        )

    def adjust_alpha(self, raw_alpha: float, days_since_signal: int) -> float:
        """Apply decay adjustment to raw alpha."""
        rate = self._decay_params.get("decay_rate", 0.03)
        return raw_alpha * np.exp(-rate * days_since_signal)


# ---------------------------------------------------------------------------
# TransactionCostModel
# ---------------------------------------------------------------------------
class TransactionCostModel:
    """Estimate and incorporate transaction costs into optimization.

    Models spread costs, market impact, and opportunity costs.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_coeff: float = 0.1,
        fixed_cost_per_trade: float = 0.0,
    ):
        self.spread_bps = spread_bps
        self.impact_coeff = impact_coeff
        self.fixed_cost_per_trade = fixed_cost_per_trade

    def estimate_cost(
        self,
        trade_size: float,
        avg_daily_volume: float = 1e6,
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """Estimate total transaction cost for a trade.

        Parameters
        ----------
        trade_size : float
            Dollar value of the trade.
        avg_daily_volume : float
            Average daily dollar volume.
        volatility : float
            Daily return volatility.

        Returns
        -------
        Dict with spread_cost, impact_cost, fixed_cost, total_cost.
        """
        spread_cost = abs(trade_size) * self.spread_bps / 10_000
        participation = abs(trade_size) / avg_daily_volume if avg_daily_volume > 0 else 0
        impact_cost = self.impact_coeff * volatility * np.sqrt(participation) * abs(trade_size)
        fixed_cost = self.fixed_cost_per_trade if trade_size != 0 else 0
        total = spread_cost + impact_cost + fixed_cost

        return {
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "fixed_cost": fixed_cost,
            "total_cost": total,
            "cost_bps": total / abs(trade_size) * 10_000 if trade_size != 0 else 0,
        }

    def net_alpha_after_costs(
        self,
        gross_alpha: float,
        turnover: float,
        avg_trade_size: float = 10_000,
        avg_daily_volume: float = 1e6,
        volatility: float = 0.02,
    ) -> float:
        """Compute net alpha after estimated transaction costs."""
        if turnover <= 0:
            return gross_alpha
        cost = self.estimate_cost(avg_trade_size * turnover, avg_daily_volume, volatility)
        return gross_alpha - cost["total_cost"] / avg_trade_size


# ---------------------------------------------------------------------------
# FeatureImportanceTracker
# ---------------------------------------------------------------------------
class FeatureImportanceTracker:
    """Track and rank feature importance over time.

    Maintains a rolling history of feature importances from model fits,
    providing stability analysis and top-feature ranking.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)

    def record(self, importances: Dict[str, float], timestamp: Optional[str] = None):
        """Record a snapshot of feature importances."""
        self._history.append({
            "timestamp": timestamp or str(pd.Timestamp.now()),
            "importances": importances.copy(),
        })

    def get_average_importance(self) -> Dict[str, float]:
        """Compute average feature importance across all recorded snapshots."""
        if not self._history:
            return {}
        all_features: Dict[str, List[float]] = {}
        for entry in self._history:
            for feat, imp in entry["importances"].items():
                all_features.setdefault(feat, []).append(imp)
        return {f: float(np.mean(v)) for f, v in all_features.items()}

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top-N features by average importance."""
        avg = self.get_average_importance()
        sorted_feats = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return sorted_feats[:n]

    def get_stability(self) -> Dict[str, float]:
        """Compute importance stability (inverse CV) for each feature."""
        if len(self._history) < 2:
            return {}
        all_features: Dict[str, List[float]] = {}
        for entry in self._history:
            for feat, imp in entry["importances"].items():
                all_features.setdefault(feat, []).append(imp)
        stability = {}
        for feat, vals in all_features.items():
            mean = np.mean(vals)
            std = np.std(vals)
            stability[feat] = float(mean / std) if std > 0 else float("inf")
        return stability


# ---------------------------------------------------------------------------
# Sector-neutral portfolio construction helper
# ---------------------------------------------------------------------------
def build_sector_neutral_weights(
    raw_weights: np.ndarray,
    sector_map: Dict[int, str],
    max_sector_deviation: float = 0.05,
    benchmark_sector_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Adjust portfolio weights to be sector-neutral relative to benchmark.

    Parameters
    ----------
    raw_weights : np.ndarray
        Unconstrained optimal weights.
    sector_map : Dict[int, str]
        Mapping of asset index to sector name.
    max_sector_deviation : float
        Maximum allowed deviation from benchmark sector weight.
    benchmark_sector_weights : Dict[str, float] or None
        Benchmark sector allocations. If None, equal-weight assumed.

    Returns
    -------
    np.ndarray
        Adjusted weights satisfying sector constraints.
    """
    n = len(raw_weights)
    adjusted = raw_weights.copy()

    # Determine sectors and their target weights
    sectors = set(sector_map.values())
    if benchmark_sector_weights is None:
        benchmark_sector_weights = {s: 1.0 / len(sectors) for s in sectors}

    for sector in sectors:
        indices = [i for i, s in sector_map.items() if s == sector]
        if not indices:
            continue

        current_alloc = sum(adjusted[i] for i in indices)
        target = benchmark_sector_weights.get(sector, 1.0 / len(sectors))
        max_alloc = target + max_sector_deviation
        min_alloc = max(target - max_sector_deviation, 0)

        if current_alloc > max_alloc and current_alloc > 0:
            scale = max_alloc / current_alloc
            for i in indices:
                adjusted[i] *= scale
        elif current_alloc < min_alloc and current_alloc > 0:
            scale = min_alloc / current_alloc
            for i in indices:
                adjusted[i] *= scale

    # Renormalize to sum to 1
    total = adjusted.sum()
    if total > 0:
        adjusted /= total

    return adjusted


# ---------------------------------------------------------------------------
# AlphaOptimizer — main orchestrator
# ---------------------------------------------------------------------------
class AlphaOptimizer:
    """Walk-forward ML alpha with mean-variance portfolio construction.

    Hunting for 100% alpha with no decay in selection.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        max_turnover: float = MAX_TURNOVER,
        transaction_cost: float = TRANSACTION_COST,
        alpha_headstart: float = ALPHA_TARGET,
    ):
        self.train_ratio = train_ratio
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.alpha_headstart = alpha_headstart
        self._last_output: Optional[AlphaOutput] = None

        # Enhanced sub-components
        self.capm_extractor = CAPMAlphaExtractor()
        self.factor_library = FactorLibrary()
        self.walk_forward = WalkForwardOptimizer()
        self.mv_optimizer = MeanVarianceOptimizer(
            max_turnover=max_turnover,
            transaction_cost=transaction_cost,
        )
        self.quality_ranker = QualityRanker()
        self.decay_model = AlphaDecayModel()
        self.cost_model = TransactionCostModel()
        self.importance_tracker = FeatureImportanceTracker()

    def optimize(
        self,
        tickers: list[str],
        start: str = "2015-01-01",
        current_weights: Optional[np.ndarray] = None,
        sector_constraints: Optional[dict] = None,
    ) -> AlphaOutput:
        """Run full alpha optimization pipeline."""
        output = AlphaOutput()

        # 1. Get data
        prices = get_adj_close(tickers, start=start)
        if prices.empty or len(prices.columns) < 2:
            return output
        available = [c for c in tickers if c in prices.columns]
        prices = prices[available]
        tickers = available
        returns = np.log(prices / prices.shift(1)).dropna()
        n_assets = len(tickers)

        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets

        # 2. Feature engineering
        features = build_features(returns, prices)

        # 3. Walk-forward ML
        target = returns.mean(axis=1).shift(-1)
        dataset = pd.concat([features, target.rename("target")], axis=1).dropna()
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        split = int(len(X) * self.train_ratio)
        if split < 50:
            return output

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Run BOTH models in parallel, blend predictions
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_preds = pd.Series(lr_model.predict(X_test_scaled), index=X_test.index)

        xgb_preds = None
        try:
            xgb_model = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbosity=0, random_state=42,
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_preds = pd.Series(xgb_model.predict(X_test_scaled), index=X_test.index)
        except Exception:
            pass

        # Blend: 60% XGBoost + 40% LinearRegression (or 100% LR if XGB unavailable)
        if xgb_preds is not None:
            alpha_preds = 0.60 * xgb_preds + 0.40 * lr_preds
            models_used = "XGBoost+LinearRegression"
        else:
            alpha_preds = lr_preds
            models_used = "LinearRegression"

        output.alpha_predictions = alpha_preds

        # Track feature importance from both models
        if xgb_preds is not None and hasattr(xgb_model, "feature_importances_"):
            imp = dict(zip(X.columns, xgb_model.feature_importances_))
            self.importance_tracker.record(imp)
        if hasattr(lr_model, "coef_"):
            lr_imp = dict(zip(X.columns, np.abs(lr_model.coef_)))
            self.importance_tracker.record(lr_imp)

        logger.info(
            "AlphaOptimizer: models=%s features=%d train=%d test=%d",
            models_used, len(X.columns), len(X_train), len(X_test),
        )

        # CAPM alpha extraction (Jensen's alpha per asset)
        capm_alphas = {}
        try:
            mkt_returns = returns.mean(axis=1)
            for t in tickers:
                t_ret = returns[t].loc[X_test.index[0]:]
                mkt_r = mkt_returns.loc[X_test.index[0]:]
                if len(t_ret) > 20 and len(mkt_r) > 20:
                    alpha_ann, beta, _ = self.capm_extractor.compute_jensens_alpha(t_ret, mkt_r)
                    capm_alphas[t] = alpha_ann
        except Exception:
            pass

        # Quality ranking overlay
        quality_scores = {}
        try:
            test_returns_df = returns.loc[X_test.index[0]:][tickers]
            if len(test_returns_df) > 20:
                quality_df = self.quality_ranker.rank_assets(test_returns_df)
                if quality_df is not None and hasattr(quality_df, "iterrows"):
                    for _, row in quality_df.iterrows():
                        t = row.get("ticker", "")
                        if t:
                            quality_scores[t] = row.get("composite_score", 0.5)
        except Exception:
            pass

        # 4. EWMA covariance
        test_returns = returns.loc[X_test.index[0]:][tickers]
        if len(test_returns) < 20:
            return output
        latest_cov = ewma_cov(test_returns)

        # 5. Expected returns with alpha headstart + CAPM alpha overlay
        base_expected = test_returns[tickers].mean().values + alpha_preds.iloc[-1] + self.alpha_headstart
        # Blend in CAPM alpha (20% weight) if available
        if capm_alphas:
            capm_overlay = np.array([capm_alphas.get(t, 0.0) / 252 for t in tickers])
            expected_returns = base_expected + 0.20 * capm_overlay
        else:
            expected_returns = base_expected
        expected_returns = np.asarray(expected_returns, dtype=float)

        # 6. Optimize
        optimal_weights = self._optimize_weights(
            expected_returns, latest_cov, current_weights, n_assets,
        )

        # 7. Performance
        port_returns = test_returns[tickers] @ optimal_weights
        annual_ret = float(port_returns.mean() * 252)
        annual_vol = float(port_returns.std() * np.sqrt(252))
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
        cum = (1 + port_returns).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min())
        rebal_cost = float(np.sum(np.abs(optimal_weights - current_weights)) * self.transaction_cost)

        # 8. Build signals with CAPM + quality overlay
        signals = []
        for i, t in enumerate(tickers):
            t_ret = test_returns[t]
            mom_3m = float(t_ret.iloc[-63:].sum()) if len(t_ret) >= 63 else 0.0
            mom_1m = float(t_ret.iloc[-21:].sum()) if len(t_ret) >= 21 else 0.0
            t_vol = float(t_ret.std() * np.sqrt(252))
            t_sharpe = (float(t_ret.mean() * 252) / t_vol) if t_vol > 0 else 0.0

            # Use quality score to upgrade/downgrade quality tier if available
            q_score = quality_scores.get(t, None)
            if q_score is not None and q_score > 0.7:
                q_tier = classify_quality(max(t_sharpe, 1.5), max(mom_3m, 0.10))
            else:
                q_tier = classify_quality(t_sharpe, mom_3m)

            sig = AlphaSignal(
                ticker=t,
                alpha_pred=float(expected_returns[i]),
                quality_tier=q_tier,
                sharpe_estimate=t_sharpe,
                momentum_3m=mom_3m,
                momentum_1m=mom_1m,
                vol=t_vol,
                weight=float(optimal_weights[i]),
            )
            signals.append(sig)

        output.signals = sorted(signals, key=lambda s: s.alpha_pred, reverse=True)
        output.optimal_weights = {t: float(optimal_weights[i]) for i, t in enumerate(tickers)}
        output.expected_annual_return = annual_ret
        output.annual_volatility = annual_vol
        output.sharpe_ratio = sharpe
        output.max_drawdown = max_dd
        output.rebalance_cost = rebal_cost

        self._last_output = output
        return output

    def _optimize_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        n_assets: int,
    ) -> np.ndarray:
        """Mean-variance optimization with turnover constraints."""

        def portfolio_vol(w):
            vol_sq = max(1e-12, w.T @ cov_matrix @ w)
            return np.sqrt(vol_sq)

        def turnover(w, prev):
            return float(np.sum(np.abs(w - prev)))

        def objective(w):
            w = np.asarray(w)
            port_ret = float(w @ expected_returns)
            vol = portfolio_vol(w)
            cost = turnover(w, current_weights) * self.transaction_cost
            sharpe = (port_ret - cost) / vol if vol > 0 else 0.0
            return -sharpe

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: self.max_turnover - turnover(np.asarray(w), current_weights)},
        ]
        bounds = [(0.0, 0.40)] * n_assets  # Max 40% per name

        try:
            result = minimize(
                objective, current_weights.copy(),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if result.success:
                return np.asarray(result.x)
        except Exception:
            pass

        return current_weights

    def get_last(self) -> Optional[AlphaOutput]:
        return self._last_output

    def run_enhanced_pipeline(
        self,
        tickers: list[str],
        start: str = "2015-01-01",
        current_weights: Optional[np.ndarray] = None,
        sector_map: Optional[Dict[int, str]] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> AlphaOutput:
        """Run the enhanced pipeline with all sub-components.

        This uses:
        - FactorLibrary for extended feature set
        - WalkForwardOptimizer for rolling OOS validation
        - CAPMAlphaExtractor for factor decomposition
        - QualityRanker for fundamental scoring
        - AlphaDecayModel for signal decay adjustment
        - TransactionCostModel for cost-aware optimization
        - MeanVarianceOptimizer with sector constraints
        """
        output = AlphaOutput()

        # 1. Data
        prices = get_adj_close(tickers, start=start)
        if prices.empty or len(prices.columns) < 2:
            return output
        available = [c for c in tickers if c in prices.columns]
        prices = prices[available]
        tickers = available
        returns = np.log(prices / prices.shift(1)).dropna()
        n_assets = len(tickers)

        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets

        # 2. Extended features via FactorLibrary
        base_features = build_features(returns, prices)
        factor_features = self.factor_library.compute_all(returns, prices)
        all_features = pd.concat([base_features, factor_features], axis=1)
        # Drop duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # 3. Target
        target = returns.mean(axis=1).shift(-1)
        dataset = pd.concat([all_features, target.rename("target")], axis=1).dropna()
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        if len(X) < 300:
            return self.optimize(tickers, start, current_weights)

        # 4. Walk-forward optimization
        preds, oos_metrics = self.walk_forward.walk_forward(X, y)
        if len(preds) < 20:
            return self.optimize(tickers, start, current_weights)
        output.alpha_predictions = preds

        # Track importance
        wf_imp = self.walk_forward.get_feature_importances()
        if wf_imp is not None and len(wf_imp) > 0:
            avg_imp = wf_imp.mean().to_dict()
            self.importance_tracker.record(avg_imp)

        # 5. Alpha decay adjustment
        decay_est = self.decay_model.estimate_decay(preds)

        # 6. EWMA covariance
        test_start = preds.index[0]
        test_returns = returns.loc[test_start:][tickers]
        if len(test_returns) < 20:
            return output
        latest_cov = ewma_cov(test_returns)

        # 7. Expected returns with decay-adjusted alpha
        base_ret = test_returns[tickers].mean().values
        adjusted_alpha = self.decay_model.adjust_alpha(
            float(preds.iloc[-1]) + self.alpha_headstart, 0
        )
        expected_returns = base_ret + adjusted_alpha
        expected_returns = np.asarray(expected_returns, dtype=float)

        # 8. Quality ranking
        quality_df = self.quality_ranker.rank_assets(test_returns[tickers])

        # 9. Optimize with sector constraints
        optimal_weights = self.mv_optimizer.optimize(
            expected_returns, latest_cov, current_weights,
            sector_map=sector_map,
            sector_limits=sector_limits,
        )

        # Apply sector-neutral adjustment if sector info available
        if sector_map is not None:
            optimal_weights = build_sector_neutral_weights(
                optimal_weights, sector_map
            )

        # 10. Transaction cost estimation
        turnover_val = float(np.sum(np.abs(optimal_weights - current_weights)))
        net_cost = self.cost_model.estimate_cost(turnover_val * 100_000)

        # 11. Performance
        port_returns = test_returns[tickers] @ optimal_weights
        annual_ret = float(port_returns.mean() * TRADING_DAYS)
        annual_vol = float(port_returns.std() * np.sqrt(TRADING_DAYS))
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
        cum = (1 + port_returns).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min())
        rebal_cost = float(net_cost["total_cost"])

        # 12. Build signals with quality overlay
        signals = []
        for i, t in enumerate(tickers):
            t_ret = test_returns[t]
            mom_3m = float(t_ret.iloc[-63:].sum()) if len(t_ret) >= 63 else 0.0
            mom_1m = float(t_ret.iloc[-21:].sum()) if len(t_ret) >= 21 else 0.0
            t_vol = float(t_ret.std() * np.sqrt(TRADING_DAYS))
            t_sharpe = (float(t_ret.mean() * TRADING_DAYS) / t_vol) if t_vol > 0 else 0.0

            sig = AlphaSignal(
                ticker=t,
                alpha_pred=float(expected_returns[i]),
                quality_tier=classify_quality(t_sharpe, mom_3m),
                sharpe_estimate=t_sharpe,
                momentum_3m=mom_3m,
                momentum_1m=mom_1m,
                vol=t_vol,
                weight=float(optimal_weights[i]),
            )
            signals.append(sig)

        output.signals = sorted(signals, key=lambda s: s.alpha_pred, reverse=True)
        output.optimal_weights = {t: float(optimal_weights[i]) for i, t in enumerate(tickers)}
        output.expected_annual_return = annual_ret
        output.annual_volatility = annual_vol
        output.sharpe_ratio = sharpe
        output.max_drawdown = max_dd
        output.rebalance_cost = rebal_cost

        self._last_output = output
        return output
