"""
Metadron Capital — Signal Decomposition Engine
Separates cyclical vs secular signals and analyzes money velocity.
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalDecomposition:
    """Result of decomposing a time series into cyclical and secular components."""
    original: pd.Series
    secular_trend: pd.Series       # Long-term structural trend
    cyclical_component: pd.Series  # Business cycle oscillations (2-10yr)
    seasonal_component: pd.Series  # Intra-year patterns
    noise: pd.Series               # Residual noise
    dominant_frequency: float      # Dominant cycle frequency in Hz
    dominant_period_days: int      # Dominant cycle period in trading days


@dataclass
class MoneyVelocityMetrics:
    """Money velocity and liquidity flow analysis."""
    velocity: pd.Series            # M2 velocity proxy
    velocity_trend: pd.Series      # Secular velocity trend
    velocity_acceleration: pd.Series  # Rate of change of velocity
    liquidity_impulse: pd.Series   # Liquidity injection/withdrawal signal
    regime: pd.Series              # Velocity regime labels


class SignalEngine:
    """
    Decomposes market signals into cyclical vs secular components
    and analyzes money velocity dynamics.
    """

    @staticmethod
    def decompose(series: pd.Series, secular_window: int = 252 * 5,
                  cyclical_band: tuple[int, int] = (126, 252 * 10)) -> SignalDecomposition:
        """
        Decompose a time series into secular trend, cyclical, seasonal, and noise.

        Args:
            series: Input price/return series
            secular_window: Rolling window for secular trend (default 5yr)
            cyclical_band: (min_period, max_period) in trading days for cyclical band
        """
        values = series.dropna().values.astype(float)
        n = len(values)

        if n < secular_window:
            secular_window = max(n // 4, 21)

        # 1. Extract secular trend via low-pass filter (Hodrick-Prescott-like)
        secular = pd.Series(
            SignalEngine._hodrick_prescott(values, lamb=secular_window ** 2),
            index=series.dropna().index
        )

        # 2. Remove secular to get cyclical + seasonal + noise
        detrended = values - secular.values

        # 3. Bandpass filter for cyclical component
        if n > cyclical_band[0] * 2:
            fs = 1.0  # 1 sample per trading day
            low_freq = 1.0 / cyclical_band[1]
            high_freq = 1.0 / cyclical_band[0]
            nyq = 0.5 * fs
            low = low_freq / nyq
            high = min(high_freq / nyq, 0.99)

            if low < high:
                b, a = sp_signal.butter(3, [low, high], btype='band')
                cyclical_vals = sp_signal.filtfilt(b, a, detrended)
            else:
                cyclical_vals = np.zeros(n)
        else:
            cyclical_vals = np.zeros(n)

        cyclical = pd.Series(cyclical_vals, index=series.dropna().index)

        # 4. Seasonal: 21-day rolling mean of residual after removing cyclical
        residual_after_cyclical = detrended - cyclical_vals
        if n > 42:
            seasonal_vals = pd.Series(residual_after_cyclical).rolling(21, center=True).mean().fillna(0).values
        else:
            seasonal_vals = np.zeros(n)
        seasonal = pd.Series(seasonal_vals, index=series.dropna().index)

        # 5. Noise is what remains
        noise = pd.Series(
            detrended - cyclical_vals - seasonal_vals,
            index=series.dropna().index
        )

        # 6. Find dominant frequency via FFT
        freqs = fftfreq(n, d=1.0)
        fft_vals = np.abs(fft(detrended))
        positive_mask = freqs > 0
        if positive_mask.any():
            dominant_idx = np.argmax(fft_vals[positive_mask])
            dominant_freq = freqs[positive_mask][dominant_idx]
            dominant_period = int(1.0 / dominant_freq) if dominant_freq > 0 else n
        else:
            dominant_freq = 0.0
            dominant_period = n

        return SignalDecomposition(
            original=series,
            secular_trend=secular,
            cyclical_component=cyclical,
            seasonal_component=seasonal,
            noise=noise,
            dominant_frequency=dominant_freq,
            dominant_period_days=dominant_period,
        )

    @staticmethod
    def _hodrick_prescott(y: np.ndarray, lamb: float = 1600.0) -> np.ndarray:
        """HP filter for trend extraction."""
        n = len(y)
        if n < 4:
            return y.copy()

        # Construct the penalty matrix
        diag_main = np.ones(n) + 2 * lamb
        diag_main[0] = 1 + lamb
        diag_main[1] = 1 + 5 * lamb
        diag_main[-2] = 1 + 5 * lamb
        diag_main[-1] = 1 + lamb

        diag_off1 = -4 * lamb * np.ones(n - 1)
        diag_off1[0] = -2 * lamb
        diag_off1[-1] = -2 * lamb

        diag_off2 = lamb * np.ones(n - 2)

        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        A = diags([diag_off2, diag_off1, diag_main, diag_off1, diag_off2],
                   [-2, -1, 0, 1, 2], shape=(n, n), format='csc')

        return spsolve(A, y)

    @staticmethod
    def money_velocity(m2_supply: pd.Series, nominal_gdp: pd.Series,
                       market_prices: Optional[pd.Series] = None) -> MoneyVelocityMetrics:
        """
        Analyze money velocity dynamics.

        V = GDP / M2 (classic Fisher equation)
        Also computes velocity acceleration and liquidity impulse signals.
        """
        # Align indices
        common_idx = m2_supply.index.intersection(nominal_gdp.index)
        m2 = m2_supply.loc[common_idx]
        gdp = nominal_gdp.loc[common_idx]

        # Velocity = GDP / M2
        velocity = gdp / m2.replace(0, np.nan)
        velocity = velocity.dropna()

        # Secular velocity trend
        if len(velocity) > 12:
            velocity_trend = velocity.rolling(12, min_periods=4).mean()
        else:
            velocity_trend = velocity.copy()

        # Velocity acceleration (2nd derivative)
        velocity_change = velocity.pct_change()
        velocity_accel = velocity_change.diff()

        # Liquidity impulse: M2 growth rate vs GDP growth rate
        m2_growth = m2.pct_change()
        gdp_growth = gdp.pct_change()
        liquidity_impulse = m2_growth - gdp_growth  # Excess liquidity

        # Regime classification
        regime = pd.Series("neutral", index=velocity.index)
        if len(velocity) > 4:
            vel_z = (velocity - velocity.rolling(8, min_periods=2).mean()) / velocity.rolling(8, min_periods=2).std().replace(0, 1)
            regime[vel_z > 1] = "accelerating"
            regime[vel_z < -1] = "decelerating"
            regime[(vel_z >= -0.5) & (vel_z <= 0.5)] = "stable"

        return MoneyVelocityMetrics(
            velocity=velocity,
            velocity_trend=velocity_trend,
            velocity_acceleration=velocity_accel,
            liquidity_impulse=liquidity_impulse,
            regime=regime,
        )
