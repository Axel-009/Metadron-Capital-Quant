"""
Multi-Asset Analysis Engine for Stock-Chain.

Provides cross-asset analysis, risk parity allocation, regime detection,
and sector rotation signals across the full investment universe.

Mathematical Foundation:
    Pearson Correlation:
        rho_{X,Y} = Cov(X, Y) / (sigma_X * sigma_Y)
        where Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)]

    Risk Parity Allocation:
        w_i = (1 / sigma_i) / sum_j(1 / sigma_j)
        This equalizes each asset's risk contribution to the portfolio.
        Marginal risk contribution: MRC_i = sigma_i * w_i
        Total risk contribution: TRC_i = w_i * (Sigma @ w)_i

    Regime Detection (Hidden Markov Model):
        States: S = {bull, bear, neutral}
        Transition matrix: A[i,j] = P(S_{t+1}=j | S_t=i)
        Emission: P(r_t | S_t) ~ N(mu_{S_t}, sigma_{S_t})
        Viterbi decoding finds most likely state sequence.

    Sharpe Ratio:
        SR = (E[R_p] - R_f) / sigma_p

    Information Ratio:
        IR = (R_p - R_b) / TE, where TE = sigma(R_p - R_b)

Usage:
    from asset_class_analyzer import AssetClassAnalyzer, MarketRegime
    from openbb_data import AssetClass

    analyzer = AssetClassAnalyzer()
    results = analyzer.analyze_by_asset_class(universe_data)
    correlation = analyzer.cross_asset_correlation(price_data)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from openbb_data import AssetClass, GICS_CLASSIFICATION, EQUITY_GICS_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Market regime states for HMM-based detection."""
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class AnalysisResult:
    """Result of analyzing a single asset class."""
    asset_class: AssetClass
    mean_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    best_performers: list[str] = field(default_factory=list)
    worst_performers: list[str] = field(default_factory=list)
    num_assets: int = 0


@dataclass
class SectorSignal:
    """Sector rotation signal for a GICS sector."""
    sector: str
    signal: str  # "overweight", "underweight", "neutral"
    momentum_score: float
    relative_strength: float
    mean_reversion_score: float
    composite_score: float


# ---------------------------------------------------------------------------
# Asset Class Analyzer
# ---------------------------------------------------------------------------

class AssetClassAnalyzer:
    """
    Multi-asset analysis engine providing cross-asset analytics.

    Core Capabilities:
        1. Per-asset-class statistical analysis
        2. Cross-asset correlation analysis (Pearson)
        3. Risk parity portfolio allocation
        4. HMM-based market regime detection
        5. GICS sector rotation signals
    """

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        """
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio calculation. Default 5%.
        trading_days : int
            Number of trading days per year. Default 252.
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    # -------------------------------------------------------------------
    # Per-Asset-Class Analysis
    # -------------------------------------------------------------------

    def analyze_by_asset_class(
        self, universe_data: dict[AssetClass, dict[str, pd.DataFrame]]
    ) -> dict[AssetClass, AnalysisResult]:
        """
        Analyze each asset class's return distribution and risk metrics.

        For each asset class:
            - Annualized return: R_annual = (1 + mean(r_daily))^252 - 1
            - Annualized volatility: sigma_annual = sigma_daily * sqrt(252)
            - Sharpe: SR = (R_annual - R_f) / sigma_annual
            - Max drawdown: MDD = max_{t} (max_{s<=t} P_s - P_t) / max_{s<=t} P_s
            - VaR(95%): -percentile(r, 5)
            - CVaR(95%): -mean(r[r <= percentile(r, 5)])  (expected shortfall)
            - Skewness: E[((r - mu)/sigma)^3]
            - Kurtosis: E[((r - mu)/sigma)^4] - 3  (excess kurtosis)

        Parameters
        ----------
        universe_data : dict[AssetClass, dict[str, pd.DataFrame]]
            Mapping of AssetClass -> {symbol: OHLCV DataFrame}.

        Returns
        -------
        dict[AssetClass, AnalysisResult]
            Analysis results per asset class.
        """
        results = {}

        for asset_class, symbol_data in universe_data.items():
            all_returns = []
            symbol_total_returns = {}

            for symbol, df in symbol_data.items():
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                prices = df["Close"].dropna()
                if len(prices) < 2:
                    continue

                # Daily log returns: R_t = ln(P_t / P_{t-1})
                log_returns = np.log(prices / prices.shift(1)).dropna()
                all_returns.append(log_returns)

                # Total return for ranking
                total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1.0
                symbol_total_returns[symbol] = total_ret

            if not all_returns:
                results[asset_class] = AnalysisResult(
                    asset_class=asset_class,
                    mean_return=0.0,
                    annualized_return=0.0,
                    annualized_volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    skewness=0.0,
                    kurtosis=0.0,
                    var_95=0.0,
                    cvar_95=0.0,
                    num_assets=0,
                )
                continue

            # Aggregate returns across all assets in the class
            combined = pd.concat(all_returns, axis=1)
            avg_returns = combined.mean(axis=1)

            mean_daily = float(avg_returns.mean())
            std_daily = float(avg_returns.std())

            ann_return = (1.0 + mean_daily) ** self.trading_days - 1.0
            ann_vol = std_daily * np.sqrt(self.trading_days)
            sharpe = (
                (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
            )

            # Max drawdown on cumulative average returns
            cum_returns = (1.0 + avg_returns).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_dd = float(drawdowns.min())

            # VaR and CVaR at 95% confidence
            sorted_returns = np.sort(avg_returns.values)
            var_idx = int(len(sorted_returns) * 0.05)
            var_95 = -float(sorted_returns[var_idx]) if var_idx < len(sorted_returns) else 0.0
            cvar_95 = (
                -float(sorted_returns[: max(var_idx, 1)].mean())
                if var_idx > 0
                else var_95
            )

            skew = float(avg_returns.skew())
            kurt = float(avg_returns.kurtosis())  # pandas uses excess kurtosis

            # Best and worst performers
            sorted_symbols = sorted(
                symbol_total_returns.items(), key=lambda x: x[1], reverse=True
            )
            best = [s for s, _ in sorted_symbols[:5]]
            worst = [s for s, _ in sorted_symbols[-5:]]

            results[asset_class] = AnalysisResult(
                asset_class=asset_class,
                mean_return=mean_daily,
                annualized_return=ann_return,
                annualized_volatility=ann_vol,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                skewness=skew,
                kurtosis=kurt,
                var_95=var_95,
                cvar_95=cvar_95,
                best_performers=best,
                worst_performers=worst,
                num_assets=len(symbol_data),
            )

        return results

    # -------------------------------------------------------------------
    # Cross-Asset Correlation
    # -------------------------------------------------------------------

    def cross_asset_correlation(
        self, data: dict[str, pd.DataFrame], method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute cross-asset correlation matrix using daily returns.

        Pearson correlation coefficient:
            rho_{X,Y} = sum((x_i - x_bar)(y_i - y_bar)) /
                         sqrt(sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2))

        Range: [-1, 1]
            rho = 1:  perfect positive linear relationship
            rho = 0:  no linear relationship
            rho = -1: perfect negative linear relationship

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Mapping of symbol/asset_name -> OHLCV DataFrame.
            Each DataFrame must have a "Close" column and a "Date" column or DatetimeIndex.
        method : str
            Correlation method: "pearson", "spearman", or "kendall".

        Returns
        -------
        pd.DataFrame
            N x N correlation matrix.
        """
        returns_dict = {}

        for name, df in data.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            prices = df["Close"].dropna()
            if len(prices) < 2:
                continue
            returns_dict[name] = np.log(prices / prices.shift(1)).dropna()

        if not returns_dict:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr(method=method)

    # -------------------------------------------------------------------
    # Risk Parity Allocation
    # -------------------------------------------------------------------

    def risk_parity_allocation(
        self, returns: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculate risk parity portfolio weights.

        Risk Parity Formula:
            w_i = (1 / sigma_i) / sum_j(1 / sigma_j)

        This ensures each asset contributes equally to total portfolio risk.

        Marginal Risk Contribution:
            MRC_i = (Sigma @ w)_i / sigma_p
            where sigma_p = sqrt(w^T @ Sigma @ w)

        Equal Risk Contribution condition:
            w_i * MRC_i = w_j * MRC_j  for all i, j

        The inverse-volatility approach is a first-order approximation
        that assumes zero correlations. For correlated assets, an
        iterative solver (e.g., Newton-Raphson) would be needed.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns, each column is an asset.

        Returns
        -------
        dict[str, float]
            Mapping of asset name -> portfolio weight (sums to 1.0).
        """
        if returns.empty:
            return {}

        # Annualized volatility for each asset
        vols = returns.std() * np.sqrt(self.trading_days)

        # Filter out zero-vol assets
        valid = vols[vols > 0]
        if valid.empty:
            return {col: 1.0 / len(returns.columns) for col in returns.columns}

        # Inverse volatility weights: w_i = (1/sigma_i) / sum(1/sigma_j)
        inv_vol = 1.0 / valid
        weights = inv_vol / inv_vol.sum()

        result = {col: 0.0 for col in returns.columns}
        for asset, w in weights.items():
            result[asset] = float(w)

        return result

    # -------------------------------------------------------------------
    # Regime Detection (HMM-based)
    # -------------------------------------------------------------------

    def regime_detection(
        self,
        macro_data: pd.DataFrame,
        returns_col: str = "Close",
        n_regimes: int = 3,
        lookback: int = 252,
    ) -> MarketRegime:
        """
        Detect current market regime using a simplified HMM approach.

        Hidden Markov Model:
            States: S = {BULL, BEAR, NEUTRAL}
            Observations: daily returns r_t

            Transition probabilities: A[i,j] = P(S_{t+1}=j | S_t=i)
            Emission distribution: P(r_t | S_t=k) ~ N(mu_k, sigma_k^2)

        Simplified implementation using rolling statistics:
            1. Compute rolling mean and std of returns
            2. Classify based on z-score of current return regime:
               z = (r_bar_recent - r_bar_long) / sigma_long
            3. Incorporate volatility regime:
               vol_ratio = sigma_recent / sigma_long

        Classification rules:
            z > 1.0 and vol_ratio < 1.5        -> BULL
            z < -1.0 and vol_ratio < 1.5       -> BEAR
            vol_ratio >= 1.5                    -> HIGH_VOLATILITY
            vol_ratio <= 0.5                    -> LOW_VOLATILITY
            otherwise                           -> NEUTRAL

        Parameters
        ----------
        macro_data : pd.DataFrame
            Price or returns data. Must contain `returns_col`.
        returns_col : str
            Column name for price data.
        n_regimes : int
            Number of regimes (used for documentation; simplified model uses 3).
        lookback : int
            Long-term lookback window in trading days.

        Returns
        -------
        MarketRegime
            Detected current market regime.
        """
        if macro_data.empty or returns_col not in macro_data.columns:
            return MarketRegime.NEUTRAL

        prices = macro_data[returns_col].dropna()
        if len(prices) < lookback:
            lookback = max(len(prices) // 2, 20)

        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) < lookback:
            return MarketRegime.NEUTRAL

        # Long-term statistics
        long_mean = float(log_returns.iloc[-lookback:].mean())
        long_std = float(log_returns.iloc[-lookback:].std())

        # Short-term statistics (last 21 trading days ~ 1 month)
        short_window = min(21, len(log_returns) // 4)
        short_mean = float(log_returns.iloc[-short_window:].mean())
        short_std = float(log_returns.iloc[-short_window:].std())

        if long_std == 0:
            return MarketRegime.NEUTRAL

        # Z-score of recent returns vs long-term
        z_score = (short_mean - long_mean) / long_std

        # Volatility ratio
        vol_ratio = short_std / long_std if long_std > 0 else 1.0

        # Classification
        if vol_ratio >= 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_ratio <= 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif z_score > 1.0:
            return MarketRegime.BULL
        elif z_score < -1.0:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    # -------------------------------------------------------------------
    # Sector Rotation Signal
    # -------------------------------------------------------------------

    def sector_rotation_signal(
        self,
        gics_data: dict[str, dict[str, pd.DataFrame]],
        momentum_window: int = 63,
        mean_reversion_window: int = 21,
    ) -> dict[str, str]:
        """
        Generate sector rotation signals for each GICS sector.

        Signals determine which sectors to overweight/underweight.

        Methodology:
            1. Momentum Score (63-day):
               R_momentum = (P_t / P_{t-63}) - 1
               Relative strength: RS = R_sector - R_market

            2. Mean Reversion Score (21-day):
               z_MR = (P_t - MA_21) / sigma_21
               If z_MR > 2: likely to revert down
               If z_MR < -2: likely to revert up

            3. Composite:
               composite = 0.6 * rank(RS) + 0.4 * rank(-z_MR)
               Top 3 sectors: OVERWEIGHT
               Bottom 3 sectors: UNDERWEIGHT
               Middle sectors: NEUTRAL

        Parameters
        ----------
        gics_data : dict[str, dict[str, pd.DataFrame]]
            Mapping of GICS sector name -> {symbol: OHLCV DataFrame}.
        momentum_window : int
            Lookback for momentum calculation (default 63 = ~3 months).
        mean_reversion_window : int
            Lookback for mean reversion (default 21 = ~1 month).

        Returns
        -------
        dict[str, str]
            Mapping of sector -> signal ("overweight", "underweight", "neutral").
        """
        sector_scores: list[SectorSignal] = []

        # Calculate market average return for relative strength
        all_returns = []
        for sector_name, symbols in gics_data.items():
            for sym, df in symbols.items():
                if df is not None and "Close" in df.columns and len(df) > momentum_window:
                    ret = (df["Close"].iloc[-1] / df["Close"].iloc[-momentum_window]) - 1
                    all_returns.append(ret)

        market_return = float(np.mean(all_returns)) if all_returns else 0.0

        for sector_name, symbols in gics_data.items():
            sector_returns = []
            sector_mr_scores = []

            for sym, df in symbols.items():
                if df is None or "Close" not in df.columns:
                    continue

                prices = df["Close"].dropna()
                if len(prices) <= momentum_window:
                    continue

                # Momentum: total return over window
                mom = (prices.iloc[-1] / prices.iloc[-momentum_window]) - 1
                sector_returns.append(mom)

                # Mean reversion: z-score vs moving average
                if len(prices) > mean_reversion_window:
                    ma = prices.iloc[-mean_reversion_window:].mean()
                    std = prices.iloc[-mean_reversion_window:].std()
                    if std > 0:
                        z = (prices.iloc[-1] - ma) / std
                        sector_mr_scores.append(z)

            if not sector_returns:
                continue

            avg_momentum = float(np.mean(sector_returns))
            relative_strength = avg_momentum - market_return
            avg_mr = float(np.mean(sector_mr_scores)) if sector_mr_scores else 0.0

            # Composite: higher is better for overweight
            composite = 0.6 * relative_strength + 0.4 * (-avg_mr * 0.01)

            sector_scores.append(
                SectorSignal(
                    sector=sector_name,
                    signal="neutral",  # will be updated below
                    momentum_score=avg_momentum,
                    relative_strength=relative_strength,
                    mean_reversion_score=avg_mr,
                    composite_score=composite,
                )
            )

        if not sector_scores:
            return {}

        # Rank by composite score
        sector_scores.sort(key=lambda x: x.composite_score, reverse=True)

        n = len(sector_scores)
        top_n = max(n // 3, 1)
        bottom_n = max(n // 3, 1)

        for i, sig in enumerate(sector_scores):
            if i < top_n:
                sig.signal = "overweight"
            elif i >= n - bottom_n:
                sig.signal = "underweight"
            else:
                sig.signal = "neutral"

        return {sig.sector: sig.signal for sig in sector_scores}


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = AssetClassAnalyzer()

    # Demo: risk parity with synthetic data
    np.random.seed(42)
    n_days = 252
    synthetic_returns = pd.DataFrame(
        {
            "US_Equity": np.random.normal(0.0004, 0.012, n_days),
            "US_Bonds": np.random.normal(0.0001, 0.004, n_days),
            "Gold": np.random.normal(0.0002, 0.008, n_days),
            "Crypto": np.random.normal(0.001, 0.04, n_days),
            "FX_EUR": np.random.normal(0.00005, 0.005, n_days),
        }
    )

    print("=== Risk Parity Allocation ===")
    weights = analyzer.risk_parity_allocation(synthetic_returns)
    for asset, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset}: {w:.4f}")

    # Demo: correlation matrix
    print("\n=== Cross-Asset Correlation ===")
    data_for_corr = {}
    for col in synthetic_returns.columns:
        cum = (1.0 + synthetic_returns[col]).cumprod() * 100
        data_for_corr[col] = pd.DataFrame({"Close": cum.values})
    corr = analyzer.cross_asset_correlation(data_for_corr)
    print(corr.round(3))

    # Demo: regime detection
    print("\n=== Regime Detection ===")
    spy_prices = pd.DataFrame(
        {"Close": (1.0 + synthetic_returns["US_Equity"]).cumprod() * 400}
    )
    regime = analyzer.regime_detection(spy_prices)
    print(f"  Current regime: {regime.value}")
