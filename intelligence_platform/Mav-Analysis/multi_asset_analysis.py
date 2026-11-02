"""
Multi-Asset Analysis Engine for Mav-Analysis.

Extends Maverick portfolio analytics to the full universe across all
asset classes with technical scanning, cross-asset momentum,
relative value screening, volatility surface analysis, and flow detection.

Usage:
    from multi_asset_analysis import MultiAssetAnalyzer
    analyzer = MultiAssetAnalyzer()
    signals = analyzer.full_universe_technical_scan(universe_data)
    momentum = analyzer.cross_asset_momentum(multi_asset_data)
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
    EQUITY_GICS_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TechnicalSignals:
    """Technical indicator signal output for a single symbol."""
    symbol: str
    asset_class: AssetClass
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_lower: float
    bb_pct_b: float
    adx_14: float
    atr_14: float
    atr_pct: float
    sma_50: float
    sma_200: float
    current_price: float
    signal_strength: float    # composite signal [-1, 1]
    trend_direction: str      # "bullish", "bearish", "neutral"


@dataclass
class RelativeValuePair:
    """A relative value pair within a GICS sector."""
    long_symbol: str
    short_symbol: str
    sector: str
    spread_zscore: float      # z-score of current spread vs historical
    expected_convergence: float
    correlation: float
    half_life_days: float     # mean reversion half-life
    confidence: float


@dataclass
class VolSurface:
    """Implied volatility surface representation."""
    symbol: str
    atm_vol: float            # at-the-money implied volatility
    skew_25d: float           # 25-delta put vol - 25-delta call vol
    term_structure_slope: float  # slope of vol term structure
    vol_of_vol: float         # volatility of volatility (vol clustering)
    put_call_skew: float      # put IV / call IV ratio
    surface_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowSignal:
    """Unusual volume/flow detection result."""
    symbol: str
    asset_class: AssetClass
    volume_zscore: float      # z_vol = (V_t - mu_V) / sigma_V
    volume_ratio: float       # V_t / avg(V, 20d)
    price_volume_divergence: float  # divergence between price and volume trends
    signal_type: str          # "accumulation", "distribution", "neutral"
    confidence: float


@dataclass
class MomentumSignal:
    """Time-series and cross-sectional momentum signal."""
    symbol: str
    asset_class: AssetClass
    tsmom_12m: float          # time-series momentum: sign(r_{t-12m}) * r_t / sigma_t
    tsmom_6m: float
    tsmom_1m: float
    xsmom_rank: float         # cross-sectional rank [0, 1]
    risk_adjusted_momentum: float  # momentum / volatility
    signal_direction: str     # "strong_long", "long", "neutral", "short", "strong_short"


# ---------------------------------------------------------------------------
# Technical indicator computations
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    RSI = 100 - 100 / (1 + RS)
    RS = EMA(gains, period) / EMA(losses, period)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


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
        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = EMA(TR, period)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index:
        +DM, -DM -> +DI, -DI -> DX -> ADX = EMA(DX)
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
    return dx.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Multi-Asset Analyzer
# ---------------------------------------------------------------------------

class MultiAssetAnalyzer:
    """
    Extends Mav-Analysis to the full universe across all asset classes.

    Provides:
    - Full universe technical scanning (RSI, MACD, BB, ADX, ATR)
    - Cross-asset momentum analysis
    - Relative value screening within GICS sectors
    - Volatility surface analysis
    - Unusual flow/volume detection
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def full_universe_technical_scan(
        self,
        universe_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, TechnicalSignals]:
        """
        Run RSI, MACD, BB, ADX, ATR on every symbol in the universe.

        Composite signal strength formula:
            signal = w_rsi * rsi_signal + w_macd * macd_signal
                     + w_bb * bb_signal + w_adx * adx_signal

        where:
            rsi_signal = (RSI - 50) / 50  (normalised to [-1, 1])
            macd_signal = sign(histogram) * min(|histogram| / ATR, 1)
            bb_signal = 2 * (%B - 0.5)    (normalised to [-1, 1])
            adx_signal = sign(price_trend) * min(ADX / 50, 1)

        Parameters
        ----------
        universe_data : dict
            symbol -> OHLCV DataFrame.

        Returns
        -------
        dict
            symbol -> TechnicalSignals
        """
        results: Dict[str, TechnicalSignals] = {}

        # Weights for composite signal
        w_rsi = 0.25
        w_macd = 0.30
        w_bb = 0.20
        w_adx = 0.25

        for symbol, data in universe_data.items():
            try:
                close = data["Close"].astype(float)
                high = data["High"].astype(float) if "High" in data.columns else close
                low = data["Low"].astype(float) if "Low" in data.columns else close

                if len(close) < 200:
                    logger.warning("Insufficient data for %s (%d rows)", symbol, len(close))
                    continue

                # Compute indicators
                rsi = _compute_rsi(close)
                macd_line, macd_signal_line, macd_hist = _compute_macd(close)
                bb_upper, bb_lower, bb_pct_b = _compute_bollinger_bands(close)
                adx = _compute_adx(high, low, close)
                atr = _compute_atr(high, low, close)
                sma_50 = close.rolling(50).mean()
                sma_200 = close.rolling(200).mean()

                # Latest values
                last = -1
                rsi_val = float(rsi.iloc[last]) if not np.isnan(rsi.iloc[last]) else 50.0
                macd_val = float(macd_line.iloc[last]) if not np.isnan(macd_line.iloc[last]) else 0.0
                macd_sig_val = float(macd_signal_line.iloc[last]) if not np.isnan(macd_signal_line.iloc[last]) else 0.0
                macd_hist_val = float(macd_hist.iloc[last]) if not np.isnan(macd_hist.iloc[last]) else 0.0
                bb_upper_val = float(bb_upper.iloc[last]) if not np.isnan(bb_upper.iloc[last]) else 0.0
                bb_lower_val = float(bb_lower.iloc[last]) if not np.isnan(bb_lower.iloc[last]) else 0.0
                bb_pct_b_val = float(bb_pct_b.iloc[last]) if not np.isnan(bb_pct_b.iloc[last]) else 0.5
                adx_val = float(adx.iloc[last]) if not np.isnan(adx.iloc[last]) else 0.0
                atr_val = float(atr.iloc[last]) if not np.isnan(atr.iloc[last]) else 0.0
                sma50_val = float(sma_50.iloc[last]) if not np.isnan(sma_50.iloc[last]) else float(close.iloc[last])
                sma200_val = float(sma_200.iloc[last]) if not np.isnan(sma_200.iloc[last]) else float(close.iloc[last])
                current_price = float(close.iloc[last])
                atr_pct_val = atr_val / max(current_price, 0.01)

                # Composite signal calculation
                rsi_signal = (rsi_val - 50.0) / 50.0  # [-1, 1]

                if atr_val > 0:
                    macd_norm = min(abs(macd_hist_val) / atr_val, 1.0) * np.sign(macd_hist_val)
                else:
                    macd_norm = 0.0

                bb_signal = 2.0 * (bb_pct_b_val - 0.5)  # [-1, 1]

                price_trend = 1.0 if current_price > sma200_val else -1.0
                adx_signal_val = price_trend * min(adx_val / 50.0, 1.0)

                signal_strength = (
                    w_rsi * rsi_signal
                    + w_macd * macd_norm
                    + w_bb * bb_signal
                    + w_adx * adx_signal_val
                )
                signal_strength = max(-1.0, min(1.0, signal_strength))

                if signal_strength > 0.3:
                    trend_direction = "bullish"
                elif signal_strength < -0.3:
                    trend_direction = "bearish"
                else:
                    trend_direction = "neutral"

                asset_class = detect_asset_class(symbol)

                results[symbol] = TechnicalSignals(
                    symbol=symbol,
                    asset_class=asset_class,
                    rsi_14=round(rsi_val, 2),
                    macd_line=round(macd_val, 4),
                    macd_signal=round(macd_sig_val, 4),
                    macd_histogram=round(macd_hist_val, 4),
                    bb_upper=round(bb_upper_val, 2),
                    bb_lower=round(bb_lower_val, 2),
                    bb_pct_b=round(bb_pct_b_val, 4),
                    adx_14=round(adx_val, 2),
                    atr_14=round(atr_val, 4),
                    atr_pct=round(atr_pct_val, 4),
                    sma_50=round(sma50_val, 2),
                    sma_200=round(sma200_val, 2),
                    current_price=round(current_price, 2),
                    signal_strength=round(signal_strength, 4),
                    trend_direction=trend_direction,
                )

            except Exception as exc:
                logger.error("Technical scan failed for %s: %s", symbol, exc)

        logger.info("Technical scan completed for %d / %d symbols",
                     len(results), len(universe_data))
        return results

    def cross_asset_momentum(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute time-series momentum across all asset classes.

        Time-series momentum (Moskowitz, Ooi, Pedersen 2012):
            r_tsmom = sign(r_{t-12m}) * r_t / sigma_t

        where:
            r_{t-12m} = 12-month return
            r_t = 1-month return
            sigma_t = annualised volatility (21-day rolling)

        Cross-sectional momentum:
            rank assets by 12m-1m return, go long top quintile, short bottom

        Risk-adjusted momentum:
            mom_risk_adj = momentum_return / sigma_t

        Parameters
        ----------
        multi_asset_data : dict
            symbol -> OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, asset_class, tsmom_12m, tsmom_6m, tsmom_1m,
                     xsmom_rank, risk_adjusted_momentum, signal_direction
        """
        momentum_data: List[Dict[str, Any]] = []

        for symbol, data in multi_asset_data.items():
            try:
                close = data["Close"].astype(float)
                if len(close) < 252:
                    continue

                # Returns
                r_12m = float(close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else 0.0
                r_6m = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0.0
                r_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0

                # Annualised volatility (21-day rolling)
                daily_returns = close.pct_change()
                sigma_t = float(daily_returns.iloc[-21:].std() * np.sqrt(252))
                sigma_t = max(sigma_t, 1e-6)

                # Time-series momentum: sign(r_{t-12m}) * r_t / sigma_t
                tsmom_12m = np.sign(r_12m) * r_1m / sigma_t
                tsmom_6m = np.sign(r_6m) * r_1m / sigma_t
                tsmom_1m = np.sign(r_1m) * r_1m / sigma_t

                # Risk-adjusted momentum
                # Use 12m-1m (skip last month to avoid reversal)
                r_12m_1m = float(close.iloc[-21] / close.iloc[-252] - 1) if len(close) >= 252 else 0.0
                risk_adj_mom = r_12m_1m / sigma_t

                momentum_data.append({
                    "symbol": symbol,
                    "asset_class": detect_asset_class(symbol).value,
                    "tsmom_12m": round(tsmom_12m, 4),
                    "tsmom_6m": round(tsmom_6m, 4),
                    "tsmom_1m": round(tsmom_1m, 4),
                    "return_12m": round(r_12m, 4),
                    "return_6m": round(r_6m, 4),
                    "return_1m": round(r_1m, 4),
                    "volatility": round(sigma_t, 4),
                    "risk_adjusted_momentum": round(risk_adj_mom, 4),
                })

            except Exception as exc:
                logger.error("Momentum calc failed for %s: %s", symbol, exc)

        df = pd.DataFrame(momentum_data)
        if df.empty:
            return df

        # Cross-sectional momentum rank [0, 1]
        df["xsmom_rank"] = df["risk_adjusted_momentum"].rank(pct=True)

        # Signal direction
        def _signal_dir(row):
            ram = row["risk_adjusted_momentum"]
            if ram > 1.5:
                return "strong_long"
            elif ram > 0.5:
                return "long"
            elif ram < -1.5:
                return "strong_short"
            elif ram < -0.5:
                return "short"
            return "neutral"

        df["signal_direction"] = df.apply(_signal_dir, axis=1)
        df = df.sort_values("risk_adjusted_momentum", ascending=False).reset_index(drop=True)

        logger.info("Momentum analysis: %d symbols, %d strong_long, %d strong_short",
                     len(df),
                     len(df[df["signal_direction"] == "strong_long"]),
                     len(df[df["signal_direction"] == "strong_short"]))
        return df

    def relative_value_screen(
        self,
        sector_data: Dict[str, pd.DataFrame],
        lookback_days: int = 252,
        zscore_threshold: float = 2.0,
    ) -> List[RelativeValuePair]:
        """
        Within each GICS sector, find overvalued vs undervalued pairs.

        Method:
        1. For each pair (A, B) in same sector, compute log price ratio:
            ratio_t = log(P_A_t / P_B_t)
        2. Compute z-score of current ratio vs rolling mean:
            z = (ratio_t - mean(ratio, lookback)) / std(ratio, lookback)
        3. If |z| > threshold, flag as relative value opportunity
        4. Estimate mean reversion half-life via OLS:
            delta_ratio_t = alpha + beta * ratio_{t-1}
            half_life = -ln(2) / beta

        Parameters
        ----------
        sector_data : dict
            symbol -> OHLCV DataFrame (should be equities in same universe).
        lookback_days : int
            Rolling window for z-score calculation.
        zscore_threshold : float
            Minimum |z-score| to flag a pair.

        Returns
        -------
        list of RelativeValuePair
        """
        pairs: List[RelativeValuePair] = []

        # Group by GICS sector
        sector_groups: Dict[str, List[str]] = {}
        for sym in sector_data:
            gics = get_gics_classification(sym)
            if gics:
                sec_name = gics["sector_name"]
                sector_groups.setdefault(sec_name, []).append(sym)

        for sector_name, symbols in sector_groups.items():
            if len(symbols) < 2:
                continue

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym_a = symbols[i]
                    sym_b = symbols[j]

                    if sym_a not in sector_data or sym_b not in sector_data:
                        continue

                    close_a = sector_data[sym_a]["Close"].astype(float)
                    close_b = sector_data[sym_b]["Close"].astype(float)

                    # Align dates
                    common = close_a.index.intersection(close_b.index)
                    if len(common) < lookback_days:
                        continue

                    ca = close_a.loc[common]
                    cb = close_b.loc[common]

                    # Log price ratio: ratio = log(P_A / P_B)
                    ratio = np.log(ca / cb.replace(0, np.nan)).dropna()
                    if len(ratio) < lookback_days:
                        continue

                    # Rolling z-score
                    roll_mean = ratio.rolling(lookback_days).mean()
                    roll_std = ratio.rolling(lookback_days).std()
                    z_score = (ratio - roll_mean) / roll_std.replace(0, np.nan)

                    current_z = float(z_score.iloc[-1]) if not np.isnan(z_score.iloc[-1]) else 0.0

                    if abs(current_z) < zscore_threshold:
                        continue

                    # Correlation
                    returns_a = ca.pct_change().dropna()
                    returns_b = cb.pct_change().dropna()
                    common_ret = returns_a.index.intersection(returns_b.index)
                    if len(common_ret) > 20:
                        corr = float(returns_a.loc[common_ret].corr(returns_b.loc[common_ret]))
                    else:
                        corr = 0.0

                    # Half-life of mean reversion via OLS
                    # delta_ratio_t = alpha + beta * ratio_{t-1}
                    # half_life = -ln(2) / beta
                    ratio_lag = ratio.shift(1).dropna()
                    delta_ratio = ratio.diff().dropna()
                    common_hl = ratio_lag.index.intersection(delta_ratio.index)
                    if len(common_hl) > 20:
                        x = ratio_lag.loc[common_hl].values
                        y = delta_ratio.loc[common_hl].values
                        beta_hl = np.polyfit(x, y, 1)[0]
                        if beta_hl < 0:
                            half_life = -math.log(2) / beta_hl
                        else:
                            half_life = float("inf")  # no mean reversion
                    else:
                        half_life = float("inf")

                    # Determine long/short
                    if current_z > 0:
                        long_sym, short_sym = sym_b, sym_a  # B is cheap, A is expensive
                    else:
                        long_sym, short_sym = sym_a, sym_b

                    # Expected convergence: mean reversion target
                    expected_conv = float(-current_z * float(roll_std.iloc[-1]))

                    confidence = min(abs(current_z) / 4.0, 0.95)  # cap at 95%
                    if half_life < 30:
                        confidence *= 1.1  # boost for fast mean reversion
                    confidence = min(confidence, 0.95)

                    pairs.append(RelativeValuePair(
                        long_symbol=long_sym,
                        short_symbol=short_sym,
                        sector=sector_name,
                        spread_zscore=round(current_z, 3),
                        expected_convergence=round(expected_conv, 4),
                        correlation=round(corr, 4),
                        half_life_days=round(half_life, 1),
                        confidence=round(confidence, 3),
                    ))

        # Sort by absolute z-score (most extreme first)
        pairs.sort(key=lambda p: abs(p.spread_zscore), reverse=True)
        logger.info("Relative value screen: %d pairs identified", len(pairs))
        return pairs

    def volatility_surface_analysis(
        self,
        options_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, VolSurface]:
        """
        Analyse implied volatility surface across strikes and maturities.

        IV surface model: sigma(K, T) parameterised by:
            - ATM vol: sigma_ATM(T)
            - Skew: d(sigma)/d(K) evaluated at K=S (25-delta proxy)
            - Term structure: d(sigma_ATM)/d(T)
            - Vol of vol: std(sigma_ATM changes)

        Skew measures:
            skew_25d = IV_25d_put - IV_25d_call
            Risk reversal = IV_25d_call - IV_25d_put
            Butterfly = 0.5 * (IV_25d_call + IV_25d_put) - IV_ATM

        Parameters
        ----------
        options_data : dict
            symbol -> DataFrame with columns: strike, expiry, iv, option_type, delta.

        Returns
        -------
        dict
            symbol -> VolSurface
        """
        results: Dict[str, VolSurface] = {}

        for symbol, df in options_data.items():
            try:
                if df.empty:
                    continue

                # ATM vol: closest to delta=0.5 (or strike closest to current price)
                if "delta" in df.columns:
                    atm_mask = (df["delta"].abs() - 0.5).abs() < 0.1
                    atm_data = df.loc[atm_mask]
                elif "strike" in df.columns and "underlying_price" in df.columns:
                    spot = df["underlying_price"].iloc[0]
                    moneyness = (df["strike"] / spot - 1.0).abs()
                    atm_data = df.loc[moneyness < 0.05]
                else:
                    atm_data = df

                if atm_data.empty:
                    atm_data = df

                atm_vol = float(atm_data["iv"].mean()) if "iv" in atm_data.columns else 0.0

                # Skew (25-delta)
                if "delta" in df.columns:
                    puts_25d = df[(df["option_type"] == "put") & ((df["delta"].abs() - 0.25).abs() < 0.05)]
                    calls_25d = df[(df["option_type"] == "call") & ((df["delta"] - 0.25).abs() < 0.05)]

                    put_vol_25d = float(puts_25d["iv"].mean()) if not puts_25d.empty else atm_vol
                    call_vol_25d = float(calls_25d["iv"].mean()) if not calls_25d.empty else atm_vol
                    skew_25d = put_vol_25d - call_vol_25d
                else:
                    skew_25d = 0.0
                    put_vol_25d = atm_vol
                    call_vol_25d = atm_vol

                # Term structure slope
                if "expiry" in df.columns and "iv" in df.columns:
                    term_data = atm_data.groupby("expiry")["iv"].mean().sort_index()
                    if len(term_data) >= 2:
                        # Simple slope: (far_vol - near_vol) / (far_T - near_T)
                        near_vol = float(term_data.iloc[0])
                        far_vol = float(term_data.iloc[-1])
                        term_slope = far_vol - near_vol
                    else:
                        term_slope = 0.0
                else:
                    term_slope = 0.0

                # Vol of vol (proxy from recent IV changes)
                if "iv" in df.columns:
                    vol_of_vol = float(df["iv"].std())
                else:
                    vol_of_vol = 0.0

                # Put/call skew
                puts = df[df["option_type"] == "put"] if "option_type" in df.columns else pd.DataFrame()
                calls = df[df["option_type"] == "call"] if "option_type" in df.columns else pd.DataFrame()
                if not puts.empty and not calls.empty:
                    put_call_skew = float(puts["iv"].mean() / max(calls["iv"].mean(), 1e-6))
                else:
                    put_call_skew = 1.0

                results[symbol] = VolSurface(
                    symbol=symbol,
                    atm_vol=round(atm_vol, 4),
                    skew_25d=round(skew_25d, 4),
                    term_structure_slope=round(term_slope, 4),
                    vol_of_vol=round(vol_of_vol, 4),
                    put_call_skew=round(put_call_skew, 4),
                    surface_data={
                        "atm_vol": atm_vol,
                        "put_25d_vol": put_vol_25d,
                        "call_25d_vol": call_vol_25d,
                        "butterfly": 0.5 * (call_vol_25d + put_vol_25d) - atm_vol,
                        "risk_reversal": call_vol_25d - put_vol_25d,
                    },
                )

            except Exception as exc:
                logger.error("Vol surface failed for %s: %s", symbol, exc)

        logger.info("Vol surface analysis: %d symbols", len(results))
        return results

    def flow_analysis(
        self,
        volume_data: Dict[str, pd.DataFrame],
        zscore_threshold: float = 2.0,
        lookback_days: int = 20,
    ) -> Dict[str, FlowSignal]:
        """
        Detect unusual volume/flow patterns.

        Volume z-score:
            z_vol = (V_t - mu_V) / sigma_V

        where:
            mu_V = mean(Volume, lookback_days)
            sigma_V = std(Volume, lookback_days)

        Price-volume divergence:
            divergence = corr(price_returns, volume_changes, lookback)
            Negative divergence (price up, volume down) = distribution warning

        Flow classification:
            - Accumulation: price up + volume up (z > threshold)
            - Distribution: price up + volume down, or price down + volume up
            - Neutral: volume within normal range

        Parameters
        ----------
        volume_data : dict
            symbol -> OHLCV DataFrame.
        zscore_threshold : float
            Flag if |z_vol| > threshold.
        lookback_days : int
            Rolling window for volume statistics.

        Returns
        -------
        dict
            symbol -> FlowSignal
        """
        results: Dict[str, FlowSignal] = {}

        for symbol, data in volume_data.items():
            try:
                if "Volume" not in data.columns:
                    continue

                close = data["Close"].astype(float)
                volume = data["Volume"].astype(float)

                if len(close) < lookback_days + 5:
                    continue

                # Volume statistics
                vol_mean = volume.rolling(lookback_days).mean()
                vol_std = volume.rolling(lookback_days).std()

                current_vol = float(volume.iloc[-1])
                mean_val = float(vol_mean.iloc[-1])
                std_val = float(vol_std.iloc[-1])

                if std_val > 0:
                    z_vol = (current_vol - mean_val) / std_val
                else:
                    z_vol = 0.0

                vol_ratio = current_vol / max(mean_val, 1.0)

                # Price-volume divergence
                price_returns = close.pct_change()
                volume_changes = volume.pct_change()
                common = price_returns.iloc[-lookback_days:].index
                if len(common) >= 10:
                    pv_corr = float(price_returns.loc[common].corr(volume_changes.loc[common]))
                else:
                    pv_corr = 0.0

                # Signal classification
                price_trend = float(close.iloc[-1] / close.iloc[-5] - 1)

                if z_vol > zscore_threshold and price_trend > 0:
                    signal_type = "accumulation"
                elif z_vol > zscore_threshold and price_trend < 0:
                    signal_type = "distribution"
                elif z_vol < -zscore_threshold:
                    signal_type = "low_volume_warning"
                elif abs(pv_corr) < 0.1 and abs(z_vol) > 1.0:
                    signal_type = "divergence"
                else:
                    signal_type = "neutral"

                confidence = min(abs(z_vol) / 4.0, 0.95) if abs(z_vol) > zscore_threshold else 0.3

                asset_class = detect_asset_class(symbol)

                results[symbol] = FlowSignal(
                    symbol=symbol,
                    asset_class=asset_class,
                    volume_zscore=round(z_vol, 3),
                    volume_ratio=round(vol_ratio, 3),
                    price_volume_divergence=round(pv_corr, 4),
                    signal_type=signal_type,
                    confidence=round(confidence, 3),
                )

            except Exception as exc:
                logger.error("Flow analysis failed for %s: %s", symbol, exc)

        # Filter to notable signals
        notable = {k: v for k, v in results.items() if v.signal_type != "neutral"}
        logger.info("Flow analysis: %d symbols scanned, %d notable signals",
                     len(results), len(notable))
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print("=== Multi-Asset Analyzer ===")
    print()

    analyzer = MultiAssetAnalyzer()

    print("Analysis capabilities:")
    print("  1. Full Universe Technical Scan (RSI, MACD, BB, ADX, ATR)")
    print("  2. Cross-Asset Momentum (time-series + cross-sectional)")
    print("  3. Relative Value Screen (GICS sector pairs)")
    print("  4. Volatility Surface Analysis")
    print("  5. Unusual Flow/Volume Detection")
    print()

    print("Mathematical formulas:")
    print("  RSI = 100 - 100 / (1 + RS)")
    print("  MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9)")
    print("  BB: Upper = SMA(20) + 2*std, Lower = SMA(20) - 2*std")
    print("  %B = (price - Lower) / (Upper - Lower)")
    print("  ATR = EMA(max(H-L, |H-C_prev|, |L-C_prev|), 14)")
    print("  ADX = EMA(|+DI - -DI| / (+DI + -DI) * 100)")
    print()
    print("  Time-series momentum: r_tsmom = sign(r_{t-12m}) * r_t / sigma_t")
    print("  Relative value z-score: z = (ratio_t - mean(ratio)) / std(ratio)")
    print("  Mean reversion half-life: t_half = -ln(2) / beta")
    print("  Volume z-score: z_vol = (V_t - mu_V) / sigma_V, flag if > 2")
    print()
    print("  Composite signal:")
    print("    S = 0.25*RSI_norm + 0.30*MACD_norm + 0.20*BB_norm + 0.25*ADX_norm")
    print("  where each component is normalised to [-1, 1]")
    print()
    print("  Vol surface skew_25d = IV_25d_put - IV_25d_call")
    print("  Butterfly = 0.5*(IV_25d_call + IV_25d_put) - IV_ATM")
