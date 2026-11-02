# -*- coding: utf-8 -*-
"""
Arbitrage Detector
==================

Detects multiple types of arbitrage opportunities across asset classes:

1. Statistical Arbitrage (Pairs Trading)
   - Cointegration via Engle-Granger: y_t = alpha + beta*x_t + epsilon_t
   - Test epsilon_t for stationarity using ADF test
   - Z-score: z = (spread - mu_spread) / sigma_spread
   - Half-life of mean reversion: theta = -ln(2) / ln(1 + beta)
     from OLS: delta_spread = alpha + beta * spread_{t-1}

2. Triangular FX Arbitrage
   - Cross rate: cross_rate = (A/B) * (B/C) vs quoted A/C
   - Profit if |implied - quoted| > transaction_cost
   - Round-trip: buy A/B -> buy B/C -> sell A/C (or reverse)

3. Index Arbitrage
   - Fair value: Index = sum(w_i * P_i) for i in constituents
   - If Index_market != Index_fair -> arbitrage
   - Buy basket + sell index (or reverse)

4. Cross-Asset Relative Value
   - Compare implied vs realized relationships:
     e.g., gold vs real rates, oil vs energy stocks, VIX vs realized vol
   - Z-score of ratio or spread deviation from historical

5. Calendar Spread (Term Structure)
   - Contango: front < back (normal for commodities)
   - Backwardation: front > back
   - Roll yield = (front - back) / back
   - Arbitrage when term structure deviates from cost-of-carry model:
     F(T) = S * e^{(r - y + c)*T}
     where r=risk-free, y=convenience yield, c=storage cost

Usage:
    from arbitrage_detector import ArbitrageDetector
    from openbb_data import get_full_universe

    universe = get_full_universe('2023-01-01', '2024-01-01')
    detector = ArbitrageDetector()

    stat_arb = detector.find_statistical_arb(universe.equities)
    fx_arb = detector.find_triangular_arb(universe.fx)
    index_arb = detector.find_index_arb(index_prices, constituent_prices)
    cross_arb = detector.find_cross_asset_arb(universe)
    cal_arb = detector.find_calendar_spread_arb(futures_data)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# Arbitrage Opportunity Data Classes
# ===================================================================
class ArbType(Enum):
    STATISTICAL = "statistical"
    TRIANGULAR_FX = "triangular_fx"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"
    CALENDAR_SPREAD = "calendar_spread"


@dataclass
class ArbOpportunity:
    """Represents a detected arbitrage opportunity."""

    arb_type: ArbType
    description: str
    symbols: List[str]
    expected_profit_pct: float
    confidence: float
    entry_signal: str
    exit_signal: str
    risk_factors: List[str]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ===================================================================
# Statistical Helpers
# ===================================================================
def _ols_regression(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    OLS regression: y = alpha + beta * x + epsilon.

    Returns (alpha, beta, residuals).
    """
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y.copy()

    alpha, beta = coeffs[0], coeffs[1]
    residuals = y - (alpha + beta * x)
    return float(alpha), float(beta), residuals


def _adf_test(series: np.ndarray, max_lag: int = 10) -> Tuple[float, float]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns (test_statistic, approximate_pvalue).

    Uses statsmodels if available, otherwise a simplified OLS-based
    approximation.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=max_lag, autolag="AIC")
        return float(result[0]), float(result[1])
    except ImportError:
        pass

    # Simplified ADF: regress delta_y on y_{t-1}
    if len(series) < 20:
        return 0.0, 1.0

    y = series[1:]
    y_lag = series[:-1]
    delta_y = np.diff(series)

    x = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        coeffs = np.linalg.lstsq(x, delta_y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    gamma = coeffs[1]  # coefficient on y_{t-1}

    residuals = delta_y - x @ coeffs
    se = np.sqrt(np.sum(residuals ** 2) / max(len(residuals) - 2, 1))
    se_gamma = se / max(np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2)), 1e-12)

    t_stat = gamma / max(se_gamma, 1e-12)

    # Approximate p-value using MacKinnon critical values (n=inf, no trend)
    # -3.43 (1%), -2.86 (5%), -2.57 (10%)
    if t_stat < -3.43:
        pvalue = 0.005
    elif t_stat < -2.86:
        pvalue = 0.03
    elif t_stat < -2.57:
        pvalue = 0.07
    elif t_stat < -1.94:
        pvalue = 0.15
    else:
        pvalue = 0.50

    return float(t_stat), pvalue


def _compute_half_life(spread: np.ndarray) -> float:
    """
    Half-life of mean reversion from OLS regression.

    Model: delta_spread = alpha + beta * spread_{t-1}
    Half-life: theta = -ln(2) / ln(1 + beta)

    Parameters
    ----------
    spread : np.ndarray

    Returns
    -------
    float
        Half-life in periods. Returns inf if no mean reversion.
    """
    if len(spread) < 10:
        return float("inf")

    lagged = spread[:-1]
    delta = np.diff(spread)

    x = np.column_stack([np.ones(len(lagged)), lagged])
    try:
        coeffs = np.linalg.lstsq(x, delta, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("inf")

    beta = coeffs[1]
    if beta >= 0:
        return float("inf")

    half_life = -math.log(2) / math.log(1 + beta)
    return max(float(half_life), 0.0)


def _compute_zscore_series(
    spread: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """
    Rolling z-score of a spread series.

    z_t = (spread_t - mean(spread_{t-N:t})) / std(spread_{t-N:t})
    """
    z = np.full_like(spread, np.nan, dtype=float)
    for i in range(lookback, len(spread)):
        window = spread[i - lookback : i]
        mu = window.mean()
        sigma = window.std()
        if sigma > 1e-10:
            z[i] = (spread[i] - mu) / sigma
    return z


# ===================================================================
# Arbitrage Detector
# ===================================================================
class ArbitrageDetector:
    """
    Detects arbitrage opportunities across multiple strategies.
    """

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
        adf_pvalue_threshold: float = 0.05,
        min_correlation: float = 0.5,
        fx_cost_bps: float = 3.0,
    ):
        """
        Parameters
        ----------
        z_entry : float
            Z-score threshold for entry (stat arb).
        z_exit : float
            Z-score threshold for exit (stat arb).
        min_half_life : float
            Minimum half-life of mean reversion (days).
        max_half_life : float
            Maximum half-life of mean reversion (days).
        adf_pvalue_threshold : float
            Maximum p-value for ADF test to consider pair cointegrated.
        min_correlation : float
            Minimum correlation for candidate pairs.
        fx_cost_bps : float
            Round-trip transaction cost for FX in basis points.
        """
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.adf_pvalue = adf_pvalue_threshold
        self.min_correlation = min_correlation
        self.fx_cost_bps = fx_cost_bps

    # ----- 1. Statistical Arbitrage -----
    def find_statistical_arb(
        self,
        prices: Dict[str, pd.DataFrame],
        sector_filter: Optional[str] = None,
        lookback: int = 252,
        zscore_lookback: int = 60,
    ) -> List[ArbOpportunity]:
        """
        Find statistical arbitrage opportunities via cointegration analysis.

        Methodology:
        1. Pre-filter pairs by correlation (> min_correlation)
        2. Run Engle-Granger cointegration test:
           y_t = alpha + beta * x_t + epsilon_t
           Test epsilon_t for stationarity (ADF test)
        3. For cointegrated pairs, compute:
           - Z-score: z = (spread - mu_spread) / sigma_spread
           - Half-life: theta = -ln(2) / ln(beta) from
             delta_spread = alpha + beta * spread_{t-1}
        4. Signal when |z| > z_entry, expect reversion to |z| < z_exit

        Parameters
        ----------
        prices : dict
            {symbol: DataFrame with 'Close' column}
        sector_filter : str, optional
            Only look at pairs within this sector.
        lookback : int
            How many bars of data to use.
        zscore_lookback : int
            Rolling window for z-score.

        Returns
        -------
        list of ArbOpportunity
        """
        # Build close price matrix
        close_dict: Dict[str, pd.Series] = {}
        for sym, df in prices.items():
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= lookback:
                close_dict[sym] = df["Close"].iloc[-lookback:]
            elif isinstance(df, pd.Series) and len(df) >= lookback:
                close_dict[sym] = df.iloc[-lookback:]

        if len(close_dict) < 2:
            return []

        close_df = pd.DataFrame(close_dict).dropna()
        if len(close_df) < lookback // 2:
            return []

        symbols = list(close_df.columns)
        opps: List[ArbOpportunity] = []

        # Pre-compute correlation matrix
        corr_matrix = close_df.corr()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a, sym_b = symbols[i], symbols[j]

                # Pre-filter by correlation
                corr = corr_matrix.loc[sym_a, sym_b]
                if abs(corr) < self.min_correlation:
                    continue

                y = close_df[sym_a].values
                x = close_df[sym_b].values

                # Engle-Granger: OLS then ADF on residuals
                alpha, beta, residuals = _ols_regression(y, x)

                # ADF test on residuals
                t_stat, pvalue = _adf_test(residuals)

                if pvalue > self.adf_pvalue:
                    continue

                # Half-life
                half_life = _compute_half_life(residuals)
                if half_life < self.min_half_life or half_life > self.max_half_life:
                    continue

                # Z-score
                z_series = _compute_zscore_series(residuals, lookback=zscore_lookback)
                current_z = z_series[-1]

                if np.isnan(current_z):
                    continue

                # Only report if z-score indicates actionable opportunity
                spread_mean = float(residuals.mean())
                spread_std = float(residuals.std())

                if abs(current_z) >= self.z_entry:
                    # Active opportunity
                    if current_z > self.z_entry:
                        direction = f"Short {sym_a}, Long {sym_b}"
                        expected_profit = (current_z - self.z_exit) * spread_std / y[-1]
                    else:
                        direction = f"Long {sym_a}, Short {sym_b}"
                        expected_profit = (abs(current_z) - self.z_exit) * spread_std / y[-1]

                    opps.append(ArbOpportunity(
                        arb_type=ArbType.STATISTICAL,
                        description=(
                            f"Stat-arb: {sym_a}/{sym_b}. "
                            f"Cointegrated (ADF p={pvalue:.4f}), "
                            f"z-score={current_z:.2f}, half-life={half_life:.1f}d. "
                            f"Action: {direction}"
                        ),
                        symbols=[sym_a, sym_b],
                        expected_profit_pct=float(expected_profit * 100),
                        confidence=min(0.85, 0.5 + (1 - pvalue) * 0.3 + abs(current_z) / 10),
                        entry_signal=f"|z-score| > {self.z_entry} (current: {current_z:.2f})",
                        exit_signal=f"|z-score| < {self.z_exit}",
                        risk_factors=[
                            "Regime change breaking cointegration",
                            "Corporate events (M&A, earnings, bankruptcy)",
                            "Sector rotation shifting fundamentals",
                            f"Half-life may extend beyond {half_life:.0f} days",
                        ],
                        metrics={
                            "z_score": float(current_z),
                            "half_life": float(half_life),
                            "adf_pvalue": float(pvalue),
                            "adf_statistic": float(t_stat),
                            "beta": float(beta),
                            "alpha": float(alpha),
                            "correlation": float(corr),
                            "spread_mean": spread_mean,
                            "spread_std": spread_std,
                        },
                    ))

        opps.sort(key=lambda o: abs(o.metrics.get("z_score", 0)), reverse=True)
        return opps

    # ----- 2. Triangular FX Arbitrage -----
    def find_triangular_arb(
        self,
        fx_rates: Dict[str, pd.DataFrame],
    ) -> List[ArbOpportunity]:
        """
        Detect triangular FX arbitrage opportunities.

        Theory:
            For currencies A, B, C:
            cross_rate_implied = (A/B) * (B/C)
            If |cross_rate_implied - quoted_A/C| > transaction_cost -> arbitrage

        Example (EUR, GBP, USD):
            If EURUSD = 1.10, GBPUSD = 1.30, EURGBP = 0.85
            Implied EURGBP = EURUSD / GBPUSD = 1.10 / 1.30 = 0.846
            If quoted EURGBP = 0.85, deviation = 0.85 - 0.846 = 0.004
            Profit if deviation > round-trip cost

        Parameters
        ----------
        fx_rates : dict
            {pair_symbol: DataFrame with 'Close' column}

        Returns
        -------
        list of ArbOpportunity
        """
        # Extract latest rates into a simple dict
        latest_rates: Dict[str, float] = {}
        for sym, df in fx_rates.items():
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) > 0:
                rate = float(df["Close"].iloc[-1])
            elif isinstance(df, pd.Series) and len(df) > 0:
                rate = float(df.iloc[-1])
            else:
                continue

            # Normalize symbol: remove =X suffix
            clean = sym.replace("=X", "").upper()
            latest_rates[clean] = rate

        if len(latest_rates) < 3:
            return []

        # Parse currency pairs into (base, quote) tuples
        pair_map: Dict[Tuple[str, str], float] = {}
        for pair_str, rate in latest_rates.items():
            if len(pair_str) == 6:
                base = pair_str[:3]
                quote = pair_str[3:]
                pair_map[(base, quote)] = rate

        # Extract all unique currencies
        currencies = set()
        for base, quote in pair_map:
            currencies.add(base)
            currencies.add(quote)
        currencies = sorted(currencies)

        if len(currencies) < 3:
            return []

        # Build rate lookup function (handles both direct and inverse)
        def get_rate(a: str, b: str) -> Optional[float]:
            if (a, b) in pair_map:
                return pair_map[(a, b)]
            if (b, a) in pair_map:
                return 1.0 / pair_map[(b, a)]
            return None

        opps: List[ArbOpportunity] = []
        cost_threshold = self.fx_cost_bps / 10000.0  # convert bps to decimal

        for i in range(len(currencies)):
            for j in range(len(currencies)):
                if j == i:
                    continue
                for k in range(len(currencies)):
                    if k == i or k == j:
                        continue

                    a, b, c = currencies[i], currencies[j], currencies[k]

                    # Get rates for the triangle
                    rate_ab = get_rate(a, b)
                    rate_bc = get_rate(b, c)
                    rate_ac = get_rate(a, c)

                    if rate_ab is None or rate_bc is None or rate_ac is None:
                        continue

                    # Implied A/C = (A/B) * (B/C)
                    implied_ac = rate_ab * rate_bc
                    deviation = (implied_ac - rate_ac) / rate_ac

                    if abs(deviation) > cost_threshold:
                        if deviation > 0:
                            # implied > quoted: buy A/C, sell implied
                            # Trade: Buy A/C at quoted rate, then sell A->B->C
                            direction = (
                                f"Buy {a}/{c} at {rate_ac:.6f}, "
                                f"Sell {a}/{b} at {rate_ab:.6f}, "
                                f"Sell {b}/{c} at {rate_bc:.6f}"
                            )
                        else:
                            # implied < quoted: sell A/C, buy implied
                            direction = (
                                f"Sell {a}/{c} at {rate_ac:.6f}, "
                                f"Buy {a}/{b} at {rate_ab:.6f}, "
                                f"Buy {b}/{c} at {rate_bc:.6f}"
                            )

                        profit_bps = abs(deviation) * 10000

                        opps.append(ArbOpportunity(
                            arb_type=ArbType.TRIANGULAR_FX,
                            description=(
                                f"Triangular FX arb: {a}/{b}/{c}. "
                                f"Implied {a}/{c} = {implied_ac:.6f}, "
                                f"Quoted {a}/{c} = {rate_ac:.6f}. "
                                f"Deviation: {deviation * 10000:.2f} bps "
                                f"(cost: {self.fx_cost_bps:.1f} bps). "
                                f"Net profit: {profit_bps - self.fx_cost_bps:.2f} bps."
                            ),
                            symbols=[f"{a}/{b}", f"{b}/{c}", f"{a}/{c}"],
                            expected_profit_pct=float(abs(deviation) * 100),
                            confidence=min(0.9, 0.6 + abs(deviation) * 50),
                            entry_signal=f"|deviation| > {self.fx_cost_bps:.0f} bps",
                            exit_signal="Immediate round-trip execution",
                            risk_factors=[
                                "Execution latency (rates may move)",
                                "Slippage on large notional",
                                "Spread widening during volatility",
                                "Settlement risk across different FX pairs",
                            ],
                            metrics={
                                "deviation_bps": float(deviation * 10000),
                                "implied_rate": float(implied_ac),
                                "quoted_rate": float(rate_ac),
                                "net_profit_bps": float(profit_bps - self.fx_cost_bps),
                                f"rate_{a}_{b}": float(rate_ab),
                                f"rate_{b}_{c}": float(rate_bc),
                                f"rate_{a}_{c}": float(rate_ac),
                            },
                        ))

        # Deduplicate (each triangle appears 2x with different orderings)
        seen = set()
        unique_opps: List[ArbOpportunity] = []
        for opp in opps:
            key = frozenset(opp.symbols)
            if key not in seen:
                seen.add(key)
                unique_opps.append(opp)

        unique_opps.sort(key=lambda o: abs(o.metrics.get("deviation_bps", 0)), reverse=True)
        return unique_opps

    # ----- 3. Index Arbitrage -----
    def find_index_arb(
        self,
        index_prices: pd.DataFrame,
        constituent_prices: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        threshold_bps: float = 10.0,
    ) -> List[ArbOpportunity]:
        """
        Detect index arbitrage: index ETF vs constituent basket.

        Fair value of index:
            Index_fair = sum(w_i * P_i) for all constituents
        Arbitrage signal:
            If |Index_market - Index_fair| / Index_fair > threshold

        Parameters
        ----------
        index_prices : pd.DataFrame
            Index/ETF price data with 'Close' column.
        constituent_prices : dict
            {symbol: DataFrame with 'Close' column}
        weights : dict, optional
            {symbol: weight}. If None, equal-weight assumed.
        threshold_bps : float
            Minimum deviation in basis points.

        Returns
        -------
        list of ArbOpportunity
        """
        if "Close" not in index_prices.columns or index_prices.empty:
            return []

        # Build constituent close matrix
        const_close: Dict[str, pd.Series] = {}
        for sym, df in constituent_prices.items():
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) > 0:
                const_close[sym] = df["Close"]

        if len(const_close) < 2:
            return []

        # Equal weights if not provided
        if weights is None:
            n = len(const_close)
            weights = {sym: 1.0 / n for sym in const_close}

        # Normalize weights to available constituents
        available = set(const_close.keys())
        active_weights = {s: w for s, w in weights.items() if s in available}
        total_w = sum(active_weights.values())
        if total_w == 0:
            return []
        active_weights = {s: w / total_w for s, w in active_weights.items()}

        # Compute fair value
        const_df = pd.DataFrame(const_close)
        dates = const_df.index.intersection(index_prices.index)
        if len(dates) < 10:
            return []

        const_aligned = const_df.loc[dates]
        index_aligned = index_prices.loc[dates, "Close"]

        # Weighted basket value (normalized to match index level)
        basket_value = pd.Series(0.0, index=dates)
        for sym, w in active_weights.items():
            if sym in const_aligned.columns:
                basket_value += w * const_aligned[sym]

        # Normalize basket to match index level at start
        scale = index_aligned.iloc[0] / basket_value.iloc[0] if basket_value.iloc[0] != 0 else 1
        basket_value *= scale

        # Compute deviation
        deviation = (index_aligned - basket_value) / basket_value
        current_dev = float(deviation.iloc[-1])
        current_dev_bps = current_dev * 10000

        opps: List[ArbOpportunity] = []

        if abs(current_dev_bps) > threshold_bps:
            if current_dev > 0:
                direction = "Short index ETF, Long constituent basket"
            else:
                direction = "Long index ETF, Short constituent basket"

            opps.append(ArbOpportunity(
                arb_type=ArbType.INDEX,
                description=(
                    f"Index arb: Index trades at {current_dev_bps:+.1f} bps vs fair value. "
                    f"Basket of {len(active_weights)} constituents. "
                    f"Action: {direction}."
                ),
                symbols=["INDEX"] + list(active_weights.keys())[:5],
                expected_profit_pct=float(abs(current_dev) * 100),
                confidence=min(0.8, 0.5 + abs(current_dev) * 20),
                entry_signal=f"|deviation| > {threshold_bps:.0f} bps (current: {abs(current_dev_bps):.1f} bps)",
                exit_signal="Deviation returns to < 5 bps",
                risk_factors=[
                    "Execution of full basket is complex and costly",
                    "Constituent weights may drift",
                    "Corporate actions (dividends, splits) cause tracking error",
                    "Borrowing costs for short leg",
                ],
                metrics={
                    "deviation_bps": float(current_dev_bps),
                    "mean_deviation_bps": float(deviation.mean() * 10000),
                    "std_deviation_bps": float(deviation.std() * 10000),
                    "max_deviation_bps": float(deviation.max() * 10000),
                    "min_deviation_bps": float(deviation.min() * 10000),
                    "num_constituents": len(active_weights),
                    "index_price": float(index_aligned.iloc[-1]),
                    "basket_price": float(basket_value.iloc[-1]),
                },
            ))

        return opps

    # ----- 4. Cross-Asset Relative Value -----
    def find_cross_asset_arb(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        lookback: int = 252,
        z_threshold: float = 2.0,
    ) -> List[ArbOpportunity]:
        """
        Find cross-asset relative value arbitrage opportunities.

        Relationships analyzed:
        1. Gold vs Real Rates: Gold ~ 1/real_rates.
           If gold is cheap relative to real rates -> buy gold
        2. Oil vs Energy Stocks: Energy stocks should track oil.
           Divergence -> relative value trade
        3. VIX vs Realized Vol: VIX is implied vol.
           If VIX >> realized -> sell vol (short VIX)
           If VIX << realized -> buy vol (long VIX)
        4. Credit spreads vs Equity: HY spreads widen -> equity should fall
           Divergence -> arbitrage

        Detection:
            Compute rolling ratio/spread between related assets.
            Z-score the ratio. Signal when |z| > threshold.

        Parameters
        ----------
        multi_asset_data : dict
            {symbol: DataFrame with 'Close' column}
        lookback : int
            Rolling window for z-score.
        z_threshold : float

        Returns
        -------
        list of ArbOpportunity
        """
        opps: List[ArbOpportunity] = []

        def _get_close(sym: str) -> Optional[pd.Series]:
            df = multi_asset_data.get(sym)
            if df is None:
                return None
            if isinstance(df, pd.DataFrame) and "Close" in df.columns:
                return df["Close"]
            if isinstance(df, pd.Series):
                return df
            return None

        # Define cross-asset relationships to check
        relationships = [
            {
                "name": "Gold vs Real Rates (TIP)",
                "long_asset": "GLD",
                "short_asset": "TIP",
                "mode": "ratio",  # analyze price ratio
                "theory": "Gold inversely correlated with real rates; TIP tracks real rates",
            },
            {
                "name": "Oil vs Energy Stocks (XOM)",
                "long_asset": "USO",
                "short_asset": "XOM",
                "mode": "ratio",
                "theory": "Energy stocks track oil prices; divergence is temporary",
            },
            {
                "name": "Gold vs Silver Ratio",
                "long_asset": "GLD",
                "short_asset": "SLV",
                "mode": "ratio",
                "theory": "Gold/silver ratio mean-reverts; historical avg ~65-80",
            },
            {
                "name": "Bond vs Equity (TLT vs SPY proxy)",
                "long_asset": "TLT",
                "short_asset": "AAPL",  # proxy for equity
                "mode": "ratio",
                "theory": "Bond-equity correlation regime; divergence signals opportunity",
            },
            {
                "name": "Copper vs Industrial (CAT)",
                "long_asset": "CPER",
                "short_asset": "CAT",
                "mode": "ratio",
                "theory": "Copper is Dr. Copper -- leading indicator for industrials",
            },
        ]

        for rel in relationships:
            prices_a = _get_close(rel["long_asset"])
            prices_b = _get_close(rel["short_asset"])

            if prices_a is None or prices_b is None:
                continue

            # Align dates
            common = prices_a.index.intersection(prices_b.index)
            if len(common) < lookback:
                continue

            a = prices_a.loc[common].values
            b = prices_b.loc[common].values

            if rel["mode"] == "ratio":
                # Compute log ratio
                with np.errstate(divide="ignore", invalid="ignore"):
                    series = np.log(a / b)
            else:
                # Spread
                series = a - b

            # Remove any NaN/inf
            valid = np.isfinite(series)
            if valid.sum() < lookback:
                continue
            series = series[valid]

            z = _compute_zscore_series(series, lookback=min(lookback, len(series) - 1))
            current_z = z[-1]

            if np.isnan(current_z) or abs(current_z) < z_threshold:
                continue

            half_life = _compute_half_life(series[-lookback:])

            if current_z > z_threshold:
                direction = f"Short {rel['long_asset']}, Long {rel['short_asset']}"
            else:
                direction = f"Long {rel['long_asset']}, Short {rel['short_asset']}"

            expected_profit = (abs(current_z) - self.z_exit) / abs(current_z) * 5  # rough estimate

            opps.append(ArbOpportunity(
                arb_type=ArbType.CROSS_ASSET,
                description=(
                    f"Cross-asset: {rel['name']}. "
                    f"Z-score={current_z:.2f}, half-life={half_life:.1f}d. "
                    f"Action: {direction}. "
                    f"Theory: {rel['theory']}"
                ),
                symbols=[rel["long_asset"], rel["short_asset"]],
                expected_profit_pct=float(expected_profit),
                confidence=min(0.75, 0.4 + abs(current_z) / 8 + (1 if half_life < 30 else 0) * 0.1),
                entry_signal=f"|z-score| > {z_threshold} (current: {current_z:.2f})",
                exit_signal=f"|z-score| < {self.z_exit}",
                risk_factors=[
                    "Structural break in cross-asset relationship",
                    "Correlation regime change",
                    "Different asset class dynamics (leverage, carry)",
                    "Liquidity mismatch between assets",
                ],
                metrics={
                    "z_score": float(current_z),
                    "half_life": float(half_life),
                    "ratio_mean": float(np.nanmean(series[-lookback:])),
                    "ratio_std": float(np.nanstd(series[-lookback:])),
                    "ratio_current": float(series[-1]),
                    "correlation": float(np.corrcoef(a[-lookback:], b[-lookback:])[0, 1]) if len(a) >= lookback else 0,
                },
            ))

        opps.sort(key=lambda o: abs(o.metrics.get("z_score", 0)), reverse=True)
        return opps

    # ----- 5. Calendar Spread Arbitrage -----
    def find_calendar_spread_arb(
        self,
        futures_data: Dict[str, pd.DataFrame],
        risk_free_rate: float = 0.05,
        storage_cost: float = 0.02,
        convenience_yield: float = 0.0,
    ) -> List[ArbOpportunity]:
        """
        Detect calendar spread arbitrage in term structure.

        Cost-of-carry model:
            F(T) = S * e^{(r - y + c) * T}
            where:
                r = risk-free rate
                y = convenience yield
                c = storage cost
                T = time to expiry (years)

        Contango: F(T2) > F(T1) for T2 > T1 (normal for commodities)
        Backwardation: F(T2) < F(T1) (supply shortage)

        Arbitrage when:
            Roll yield = (F_near - F_far) / F_far
            deviates significantly from cost-of-carry fair value.

        Term structure Z-score:
            z = (roll_yield - mean_roll) / std_roll

        Parameters
        ----------
        futures_data : dict
            {contract_label: DataFrame with 'Close' column}
            Labels should be sortable by expiry (e.g., 'CL_2024_03', 'CL_2024_06')
        risk_free_rate : float
        storage_cost : float
        convenience_yield : float

        Returns
        -------
        list of ArbOpportunity
        """
        opps: List[ArbOpportunity] = []

        # Sort contracts by label (assumes chronological ordering)
        sorted_contracts = sorted(futures_data.keys())
        if len(sorted_contracts) < 2:
            return []

        # Collect latest prices
        latest_prices: Dict[str, float] = {}
        for label in sorted_contracts:
            df = futures_data[label]
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) > 0:
                latest_prices[label] = float(df["Close"].iloc[-1])
            elif isinstance(df, pd.Series) and len(df) > 0:
                latest_prices[label] = float(df.iloc[-1])

        if len(latest_prices) < 2:
            return []

        sorted_labels = sorted(latest_prices.keys())

        for i in range(len(sorted_labels) - 1):
            near_label = sorted_labels[i]
            far_label = sorted_labels[i + 1]
            near_price = latest_prices[near_label]
            far_price = latest_prices[far_label]

            if near_price <= 0 or far_price <= 0:
                continue

            # Roll yield
            roll_yield = (near_price - far_price) / far_price

            # Cost-of-carry fair spread (assuming ~3 months between contracts)
            T = 0.25  # approximate time between contracts in years
            fair_ratio = math.exp((risk_free_rate - convenience_yield + storage_cost) * T)
            fair_spread = (fair_ratio - 1)

            # Deviation from fair value
            deviation = roll_yield - fair_spread

            # Compute historical roll yield z-score if we have time series
            near_df = futures_data.get(near_label)
            far_df = futures_data.get(far_label)

            z_score = 0.0
            half_life = float("inf")

            if isinstance(near_df, pd.DataFrame) and isinstance(far_df, pd.DataFrame):
                if "Close" in near_df.columns and "Close" in far_df.columns:
                    common_idx = near_df.index.intersection(far_df.index)
                    if len(common_idx) >= 60:
                        near_series = near_df.loc[common_idx, "Close"].values
                        far_series = far_df.loc[common_idx, "Close"].values

                        # Historical roll yield
                        hist_roll = (near_series - far_series) / far_series
                        z_arr = _compute_zscore_series(hist_roll, lookback=min(60, len(hist_roll) - 1))
                        if not np.isnan(z_arr[-1]):
                            z_score = float(z_arr[-1])
                        half_life = _compute_half_life(hist_roll)

            # Determine contango/backwardation
            if near_price < far_price:
                structure = "contango"
            else:
                structure = "backwardation"

            # Signal if deviation is significant
            if abs(deviation) > 0.02 or abs(z_score) > 2.0:
                if deviation > 0:
                    direction = f"Sell {near_label}, Buy {far_label} (sell roll)"
                else:
                    direction = f"Buy {near_label}, Sell {far_label} (buy roll)"

                opps.append(ArbOpportunity(
                    arb_type=ArbType.CALENDAR_SPREAD,
                    description=(
                        f"Calendar spread: {near_label} vs {far_label}. "
                        f"Term structure: {structure}. "
                        f"Roll yield: {roll_yield * 100:.2f}% vs fair value: {fair_spread * 100:.2f}%. "
                        f"Deviation: {deviation * 100:.2f}%. "
                        f"Z-score: {z_score:.2f}. "
                        f"Action: {direction}."
                    ),
                    symbols=[near_label, far_label],
                    expected_profit_pct=float(abs(deviation) * 100),
                    confidence=min(0.7, 0.4 + abs(z_score) / 8),
                    entry_signal=f"Roll yield deviation > 2% or |z| > 2.0 (current: {z_score:.2f})",
                    exit_signal="Deviation returns to fair value",
                    risk_factors=[
                        "Storage cost changes",
                        "Convenience yield shifts (supply disruption)",
                        "Interest rate changes affecting carry cost",
                        "Contract liquidity near expiry",
                        "Roll risk if position held through expiry",
                    ],
                    metrics={
                        "roll_yield_pct": float(roll_yield * 100),
                        "fair_spread_pct": float(fair_spread * 100),
                        "deviation_pct": float(deviation * 100),
                        "z_score": z_score,
                        "half_life": half_life,
                        "near_price": near_price,
                        "far_price": far_price,
                        "structure": 1.0 if structure == "contango" else -1.0,
                    },
                ))

        opps.sort(key=lambda o: abs(o.metrics.get("deviation_pct", 0)), reverse=True)
        return opps

    # ----- Convenience: scan everything -----
    def scan_all(
        self,
        equity_prices: Optional[Dict[str, pd.DataFrame]] = None,
        fx_rates: Optional[Dict[str, pd.DataFrame]] = None,
        multi_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        index_prices: Optional[pd.DataFrame] = None,
        constituent_prices: Optional[Dict[str, pd.DataFrame]] = None,
        futures_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, List[ArbOpportunity]]:
        """
        Run all arbitrage scans and return organized results.

        Parameters
        ----------
        equity_prices : dict, optional
        fx_rates : dict, optional
        multi_asset_data : dict, optional
        index_prices : pd.DataFrame, optional
        constituent_prices : dict, optional
        futures_data : dict, optional

        Returns
        -------
        dict
            {arb_type: [ArbOpportunity, ...]}
        """
        results: Dict[str, List[ArbOpportunity]] = {}

        if equity_prices:
            logger.info("Scanning for statistical arbitrage in %d equities...", len(equity_prices))
            stat_arb = self.find_statistical_arb(equity_prices)
            results["statistical"] = stat_arb
            logger.info("Found %d stat-arb opportunities", len(stat_arb))

        if fx_rates:
            logger.info("Scanning for triangular FX arbitrage in %d pairs...", len(fx_rates))
            tri_arb = self.find_triangular_arb(fx_rates)
            results["triangular_fx"] = tri_arb
            logger.info("Found %d triangular FX opportunities", len(tri_arb))

        if index_prices is not None and constituent_prices:
            logger.info("Scanning for index arbitrage...")
            idx_arb = self.find_index_arb(index_prices, constituent_prices)
            results["index"] = idx_arb
            logger.info("Found %d index arb opportunities", len(idx_arb))

        if multi_asset_data:
            logger.info("Scanning for cross-asset relative value...")
            cross_arb = self.find_cross_asset_arb(multi_asset_data)
            results["cross_asset"] = cross_arb
            logger.info("Found %d cross-asset opportunities", len(cross_arb))

        if futures_data:
            logger.info("Scanning for calendar spread arbitrage...")
            cal_arb = self.find_calendar_spread_arb(futures_data)
            results["calendar_spread"] = cal_arb
            logger.info("Found %d calendar spread opportunities", len(cal_arb))

        total = sum(len(v) for v in results.values())
        logger.info("Total arbitrage opportunities found: %d", total)

        return results

    def summary_report(
        self,
        results: Dict[str, List[ArbOpportunity]],
    ) -> str:
        """
        Generate a text summary of all detected opportunities.

        Parameters
        ----------
        results : dict from scan_all()

        Returns
        -------
        str
        """
        lines = [
            "=" * 60,
            "  ARBITRAGE DETECTOR REPORT",
            "=" * 60,
            "",
        ]

        for arb_type, opps in results.items():
            lines.append(f"--- {arb_type.upper()} ({len(opps)} opportunities) ---")
            for opp in opps[:5]:  # top 5 per type
                lines.append(f"  {opp.description}")
                lines.append(
                    f"    Expected profit: {opp.expected_profit_pct:.2f}% | "
                    f"Confidence: {opp.confidence:.2f}"
                )
                key_metrics = ", ".join(
                    f"{k}={v:.4f}" for k, v in list(opp.metrics.items())[:4]
                )
                lines.append(f"    Metrics: {key_metrics}")
                lines.append("")

        total = sum(len(v) for v in results.values())
        lines.append(f"Total opportunities: {total}")

        return "\n".join(lines)
