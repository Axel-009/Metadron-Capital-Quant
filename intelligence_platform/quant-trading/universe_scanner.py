# -*- coding: utf-8 -*-
"""
Full Universe Scanner for quant-trading backtests.

Pulls the entire investment universe via OpenBB, classifies by asset class
and GICS sector, computes screening metrics, and feeds filtered data to
individual strategy backtests.

Usage:
    from universe_scanner import UniverseScanner

    scanner = UniverseScanner('2022-01-01', '2024-01-01')
    scanner.fetch()
    equities = scanner.get_equities_by_sector('Information Technology')
    momentum_picks = scanner.screen_momentum(top_n=20)
    value_picks = scanner.screen_value(top_n=20)
    pairs = scanner.find_cointegrated_pairs(sector='Energy')

Mathematical Reference
======================
Momentum Score:
    M = w1 * ret_1m + w2 * ret_3m + w3 * ret_6m + w4 * ret_12m
    Default weights: w1=0.10, w2=0.20, w3=0.30, w4=0.40

Mean Reversion Score:
    Z = (price - SMA_N) / std_N
    Candidates: |Z| > 2.0

Volatility (annualized):
    sigma = std(daily_returns) * sqrt(252)

Relative Strength:
    RS = stock_return / benchmark_return
    RS > 1 => outperforming benchmark

Risk-Adjusted Momentum (Sharpe-Momentum):
    SM = ret_period / volatility_period
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openbb_data import (
    AssetClass,
    GICSClassification,
    UniverseData,
    classify_by_gics,
    classify_universe_by_gics,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_var,
    detect_asset_class,
    get_bond_universe,
    get_commodity_universe,
    get_crypto_universe,
    get_equity_universe,
    get_full_universe,
    get_fx_universe,
    DEFAULT_EQUITY_SYMBOLS,
    DEFAULT_BOND_SYMBOLS,
    DEFAULT_COMMODITY_SYMBOLS,
    DEFAULT_CRYPTO_SYMBOLS,
    DEFAULT_FX_PAIRS,
    GICS_SECTORS,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Screening Result
# ===================================================================
@dataclass
class ScreenResult:
    """Result from a universe screen."""

    symbol: str
    asset_class: AssetClass
    gics_sector: Optional[str]
    gics_industry: Optional[str]
    score: float
    metrics: Dict[str, float]
    last_price: float
    data: Optional[pd.DataFrame] = None


# ===================================================================
# Universe Scanner
# ===================================================================
class UniverseScanner:
    """
    Full universe scanning engine that all backtests can use.

    Workflow:
    1. Fetch full universe via OpenBB
    2. Classify by asset class and GICS
    3. Compute screening metrics (momentum, value, volatility, etc.)
    4. Feed filtered data to individual strategy backtests
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        provider: str = "yfinance",
        equity_symbols: Optional[List[str]] = None,
        bond_symbols: Optional[List[str]] = None,
        commodity_symbols: Optional[List[str]] = None,
        crypto_symbols: Optional[List[str]] = None,
        fx_symbols: Optional[List[str]] = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.provider = provider
        self.equity_symbols = equity_symbols
        self.bond_symbols = bond_symbols
        self.commodity_symbols = commodity_symbols
        self.crypto_symbols = crypto_symbols
        self.fx_symbols = fx_symbols

        self.universe: Optional[UniverseData] = None
        self._close_matrix: Optional[pd.DataFrame] = None
        self._returns_matrix: Optional[pd.DataFrame] = None

    def fetch(self) -> UniverseData:
        """
        Fetch the full universe from OpenBB.

        Returns
        -------
        UniverseData
        """
        logger.info("Fetching full universe: %s to %s", self.start_date, self.end_date)
        self.universe = get_full_universe(
            start=self.start_date,
            end=self.end_date,
            provider=self.provider,
            equity_symbols=self.equity_symbols,
            bond_symbols=self.bond_symbols,
            commodity_symbols=self.commodity_symbols,
            crypto_symbols=self.crypto_symbols,
            fx_symbols=self.fx_symbols,
        )
        self._build_matrices()
        logger.info(
            "Fetched %d total symbols across all asset classes",
            len(self.universe.all_symbols),
        )
        return self.universe

    def _build_matrices(self) -> None:
        """Build close price and return matrices from fetched data."""
        if self.universe is None:
            return

        close_dict: Dict[str, pd.Series] = {}
        for sym, df in self.universe.all_dataframes.items():
            if "Close" in df.columns and len(df) > 0:
                close_dict[sym] = df["Close"]

        if close_dict:
            self._close_matrix = pd.DataFrame(close_dict).sort_index()
            self._returns_matrix = self._close_matrix.pct_change()

    # ----- Access Methods -----
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get raw OHLCV data for a symbol."""
        if self.universe is None:
            return None
        return self.universe.all_dataframes.get(symbol)

    def get_close_prices(self) -> pd.DataFrame:
        """Return close price matrix (symbols as columns)."""
        if self._close_matrix is None:
            return pd.DataFrame()
        return self._close_matrix

    def get_returns(self) -> pd.DataFrame:
        """Return daily returns matrix."""
        if self._returns_matrix is None:
            return pd.DataFrame()
        return self._returns_matrix

    def get_equities_by_sector(self, sector_name: str) -> Dict[str, pd.DataFrame]:
        """
        Get all equity DataFrames for a given GICS sector.

        Parameters
        ----------
        sector_name : str
            e.g. 'Information Technology', 'Health Care'

        Returns
        -------
        dict
            {symbol: DataFrame}
        """
        if self.universe is None:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for sym, df in self.universe.equities.items():
            cls = classify_by_gics(sym)
            if cls and cls.sector_name == sector_name:
                result[sym] = df
        return result

    def get_by_asset_class(self, asset_class: AssetClass) -> Dict[str, pd.DataFrame]:
        """Get all DataFrames for a given asset class."""
        if self.universe is None:
            return {}

        mapping = {
            AssetClass.EQUITY: self.universe.equities,
            AssetClass.BOND: self.universe.bonds,
            AssetClass.COMMODITY: self.universe.commodities,
            AssetClass.CRYPTO: self.universe.crypto,
            AssetClass.FX: self.universe.fx,
        }
        return mapping.get(asset_class, {})

    def get_symbols_by_gics_level(
        self,
        level: str = "sector",
    ) -> Dict[str, List[str]]:
        """
        Group equity symbols by GICS level.

        Parameters
        ----------
        level : str
            One of 'sector', 'industry_group', 'industry', 'sub_industry'.

        Returns
        -------
        dict
            {level_name: [symbols]}
        """
        if self.universe is None:
            return {}

        groups: Dict[str, List[str]] = {}
        for sym in self.universe.equities:
            cls = classify_by_gics(sym)
            if cls is None:
                continue

            if level == "sector":
                key = cls.sector_name
            elif level == "industry_group":
                key = cls.industry_group_name
            elif level == "industry":
                key = cls.industry_name
            elif level == "sub_industry":
                key = cls.sub_industry_name
            else:
                key = cls.sector_name

            groups.setdefault(key, []).append(sym)

        return groups

    # ----- Screening Methods -----
    def screen_momentum(
        self,
        top_n: int = 20,
        weights: Optional[Dict[str, float]] = None,
        asset_class: Optional[AssetClass] = None,
    ) -> List[ScreenResult]:
        """
        Screen for momentum stocks/assets.

        Momentum Score:
            M = w1 * ret_1m + w2 * ret_3m + w3 * ret_6m + w4 * ret_12m

        Default weights: w1=0.10, w2=0.20, w3=0.30, w4=0.40

        Parameters
        ----------
        top_n : int
        weights : dict, optional
        asset_class : AssetClass, optional

        Returns
        -------
        list of ScreenResult
        """
        if self._close_matrix is None or self._close_matrix.empty:
            return []

        w = weights or {"1m": 0.10, "3m": 0.20, "6m": 0.30, "12m": 0.40}
        lookbacks = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}

        results: List[ScreenResult] = []
        for sym in self._close_matrix.columns:
            if asset_class is not None and detect_asset_class(sym) != asset_class:
                continue

            prices = self._close_matrix[sym].dropna()
            if len(prices) < 30:
                continue

            score = 0.0
            metrics: Dict[str, float] = {}
            last_price = float(prices.iloc[-1])

            for period, days in lookbacks.items():
                if len(prices) >= days + 1:
                    ret = (prices.iloc[-1] - prices.iloc[-days]) / prices.iloc[-days]
                    metrics[f"return_{period}"] = float(ret)
                    score += w.get(period, 0) * float(ret)
                else:
                    metrics[f"return_{period}"] = 0.0

            # Volatility
            daily_ret = prices.pct_change().dropna()
            if len(daily_ret) > 20:
                vol = float(daily_ret.std() * math.sqrt(252))
                metrics["volatility_ann"] = vol
                # Risk-adjusted momentum (Sharpe-Momentum)
                if vol > 0:
                    metrics["sharpe_momentum"] = score / vol
            else:
                metrics["volatility_ann"] = 0.0

            cls = classify_by_gics(sym)
            results.append(ScreenResult(
                symbol=sym,
                asset_class=detect_asset_class(sym),
                gics_sector=cls.sector_name if cls else None,
                gics_industry=cls.industry_name if cls else None,
                score=score,
                metrics=metrics,
                last_price=last_price,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def screen_mean_reversion(
        self,
        top_n: int = 20,
        lookback: int = 60,
        z_threshold: float = 2.0,
        asset_class: Optional[AssetClass] = None,
    ) -> List[ScreenResult]:
        """
        Screen for mean-reversion candidates.

        Z-score:
            Z = (price - SMA_N) / std_N
            Candidates: |Z| > z_threshold

        Parameters
        ----------
        top_n : int
        lookback : int - SMA period
        z_threshold : float
        asset_class : AssetClass, optional

        Returns
        -------
        list of ScreenResult
        """
        if self._close_matrix is None or self._close_matrix.empty:
            return []

        results: List[ScreenResult] = []
        for sym in self._close_matrix.columns:
            if asset_class is not None and detect_asset_class(sym) != asset_class:
                continue

            prices = self._close_matrix[sym].dropna()
            if len(prices) < lookback + 5:
                continue

            sma = prices.rolling(lookback).mean()
            std = prices.rolling(lookback).std()
            z_series = (prices - sma) / std.replace(0, np.nan)

            current_z = z_series.iloc[-1]
            if np.isnan(current_z):
                continue

            if abs(current_z) < z_threshold:
                continue

            last_price = float(prices.iloc[-1])
            target_price = float(sma.iloc[-1])

            cls = classify_by_gics(sym)
            results.append(ScreenResult(
                symbol=sym,
                asset_class=detect_asset_class(sym),
                gics_sector=cls.sector_name if cls else None,
                gics_industry=cls.industry_name if cls else None,
                score=abs(float(current_z)),
                metrics={
                    "z_score": float(current_z),
                    "sma": float(sma.iloc[-1]),
                    "distance_to_mean_pct": (last_price - target_price) / last_price * 100,
                    "std": float(std.iloc[-1]),
                },
                last_price=last_price,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def screen_volatility(
        self,
        top_n: int = 20,
        min_vol: float = 0.0,
        max_vol: float = float("inf"),
        asset_class: Optional[AssetClass] = None,
    ) -> List[ScreenResult]:
        """
        Screen by annualized volatility.

        Volatility:
            sigma = std(daily_returns) * sqrt(252)

        Parameters
        ----------
        top_n : int
        min_vol, max_vol : float
        asset_class : AssetClass, optional

        Returns
        -------
        list of ScreenResult
        """
        if self._returns_matrix is None or self._returns_matrix.empty:
            return []

        results: List[ScreenResult] = []
        for sym in self._returns_matrix.columns:
            if asset_class is not None and detect_asset_class(sym) != asset_class:
                continue

            rets = self._returns_matrix[sym].dropna()
            if len(rets) < 20:
                continue

            vol = float(rets.std() * math.sqrt(252))
            if vol < min_vol or vol > max_vol:
                continue

            last_price = float(self._close_matrix[sym].dropna().iloc[-1])

            # Additional metrics
            sharpe = compute_sharpe_ratio(rets)
            sortino = compute_sortino_ratio(rets)
            mdd = compute_max_drawdown(self._close_matrix[sym].dropna())

            cls = classify_by_gics(sym)
            results.append(ScreenResult(
                symbol=sym,
                asset_class=detect_asset_class(sym),
                gics_sector=cls.sector_name if cls else None,
                gics_industry=cls.industry_name if cls else None,
                score=vol,
                metrics={
                    "volatility_ann": vol,
                    "sharpe_ratio": sharpe,
                    "sortino_ratio": sortino,
                    "max_drawdown": mdd,
                },
                last_price=last_price,
            ))

        results.sort(key=lambda r: r.score)
        return results[:top_n]

    def screen_relative_strength(
        self,
        benchmark: str = "SPY",
        top_n: int = 20,
        period_days: int = 63,
    ) -> List[ScreenResult]:
        """
        Screen for relative strength vs a benchmark.

        RS = stock_return_period / benchmark_return_period
        RS > 1 => outperforming benchmark

        Parameters
        ----------
        benchmark : str
        top_n : int
        period_days : int

        Returns
        -------
        list of ScreenResult
        """
        if self._close_matrix is None or self._close_matrix.empty:
            return []

        # Compute benchmark return
        if benchmark in self._close_matrix.columns:
            bench_prices = self._close_matrix[benchmark].dropna()
        else:
            # If benchmark not in universe, just use equal-weight average
            bench_prices = self._close_matrix.mean(axis=1).dropna()

        if len(bench_prices) < period_days + 1:
            return []

        bench_ret = (bench_prices.iloc[-1] - bench_prices.iloc[-period_days]) / bench_prices.iloc[-period_days]

        results: List[ScreenResult] = []
        for sym in self._close_matrix.columns:
            if sym == benchmark:
                continue

            prices = self._close_matrix[sym].dropna()
            if len(prices) < period_days + 1:
                continue

            stock_ret = (prices.iloc[-1] - prices.iloc[-period_days]) / prices.iloc[-period_days]
            rs = float(stock_ret / bench_ret) if bench_ret != 0 else 0.0

            cls = classify_by_gics(sym)
            results.append(ScreenResult(
                symbol=sym,
                asset_class=detect_asset_class(sym),
                gics_sector=cls.sector_name if cls else None,
                gics_industry=cls.industry_name if cls else None,
                score=rs,
                metrics={
                    "relative_strength": rs,
                    "stock_return": float(stock_ret),
                    "benchmark_return": float(bench_ret),
                    "excess_return": float(stock_ret - bench_ret),
                },
                last_price=float(prices.iloc[-1]),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def find_cointegrated_pairs(
        self,
        sector: Optional[str] = None,
        pvalue_threshold: float = 0.05,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Find cointegrated pairs suitable for pair trading backtests.

        Cointegration (Engle-Granger two-step):
            Step 1: OLS regression y_t = alpha + beta * x_t + epsilon_t
            Step 2: Test epsilon_t for stationarity (ADF test)

        Half-life of mean reversion:
            OLS: delta_spread = alpha + beta * spread_{t-1}
            theta = -ln(2) / ln(1 + beta)

        Parameters
        ----------
        sector : str, optional
            Only search within this GICS sector.
        pvalue_threshold : float
            ADF test p-value threshold for cointegration.
        min_half_life, max_half_life : float
            Half-life bounds in days.

        Returns
        -------
        list of dict
            Each dict has: sym_a, sym_b, beta, pvalue, half_life, correlation.
        """
        if self._close_matrix is None or self._close_matrix.empty:
            return []

        # Select symbols
        if sector:
            symbols = []
            for sym in self._close_matrix.columns:
                cls = classify_by_gics(sym)
                if cls and cls.sector_name == sector:
                    symbols.append(sym)
        else:
            symbols = [
                sym for sym in self._close_matrix.columns
                if detect_asset_class(sym) == AssetClass.EQUITY
            ]

        if len(symbols) < 2:
            return []

        # Build price matrix for selected symbols
        prices = self._close_matrix[symbols].dropna()
        if len(prices) < 60:
            return []

        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("statsmodels not installed; using simplified ADF approximation")
            adfuller = None

        pairs: List[Dict[str, Any]] = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]

                y = prices[sym_a].values
                x = prices[sym_b].values

                # OLS: y = alpha + beta * x
                x_with_const = np.column_stack([np.ones(len(x)), x])
                try:
                    beta_vec = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                alpha, beta = beta_vec[0], beta_vec[1]
                spread = y - (alpha + beta * x)

                # Test for stationarity
                if adfuller is not None:
                    try:
                        adf_result = adfuller(spread, maxlag=10)
                        pvalue = adf_result[1]
                    except Exception:
                        continue
                else:
                    # Simplified: use OLS on delta_spread ~ spread_{t-1}
                    spread_lagged = spread[:-1]
                    delta_spread = np.diff(spread)
                    if len(spread_lagged) < 20:
                        continue
                    x_ols = np.column_stack([np.ones(len(spread_lagged)), spread_lagged])
                    try:
                        coeffs = np.linalg.lstsq(x_ols, delta_spread, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    ar_beta = coeffs[1]
                    # Approximate p-value from AR coefficient
                    pvalue = 0.01 if ar_beta < -0.1 else 0.10

                if pvalue > pvalue_threshold:
                    continue

                # Half-life calculation
                spread_lagged = spread[:-1]
                delta_spread = np.diff(spread)
                x_hl = np.column_stack([np.ones(len(spread_lagged)), spread_lagged])
                try:
                    hl_coeffs = np.linalg.lstsq(x_hl, delta_spread, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                ar_coeff = hl_coeffs[1]
                if ar_coeff >= 0:
                    continue  # no mean reversion

                half_life = -math.log(2) / math.log(1 + ar_coeff)

                if half_life < min_half_life or half_life > max_half_life:
                    continue

                correlation = float(np.corrcoef(y, x)[0, 1])

                pairs.append({
                    "sym_a": sym_a,
                    "sym_b": sym_b,
                    "beta": float(beta),
                    "alpha": float(alpha),
                    "pvalue": float(pvalue),
                    "half_life": float(half_life),
                    "correlation": correlation,
                    "spread_mean": float(spread.mean()),
                    "spread_std": float(spread.std()),
                    "current_z": float((spread[-1] - spread.mean()) / max(spread.std(), 1e-8)),
                })

        pairs.sort(key=lambda p: p["pvalue"])
        return pairs

    def compute_correlation_matrix(
        self,
        asset_class: Optional[AssetClass] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for the universe.

        Parameters
        ----------
        asset_class : AssetClass, optional
        method : str - 'pearson', 'spearman', or 'kendall'

        Returns
        -------
        pd.DataFrame
        """
        if self._returns_matrix is None:
            return pd.DataFrame()

        if asset_class is not None:
            cols = [
                c for c in self._returns_matrix.columns
                if detect_asset_class(c) == asset_class
            ]
            returns = self._returns_matrix[cols]
        else:
            returns = self._returns_matrix

        return returns.corr(method=method)

    def get_sector_performance(self, period_days: int = 63) -> pd.DataFrame:
        """
        Compute sector-level performance metrics.

        Parameters
        ----------
        period_days : int

        Returns
        -------
        pd.DataFrame with sector as index, columns: return, volatility, sharpe, num_stocks
        """
        if self.universe is None or self._close_matrix is None:
            return pd.DataFrame()

        sector_data: Dict[str, List[float]] = {}
        sector_vol: Dict[str, List[float]] = {}
        sector_count: Dict[str, int] = {}

        for sym in self.universe.equities:
            cls = classify_by_gics(sym)
            if cls is None or sym not in self._close_matrix.columns:
                continue

            prices = self._close_matrix[sym].dropna()
            if len(prices) < period_days + 1:
                continue

            ret = float((prices.iloc[-1] - prices.iloc[-period_days]) / prices.iloc[-period_days])
            daily_rets = prices.pct_change().dropna()
            vol = float(daily_rets.std() * math.sqrt(252)) if len(daily_rets) > 5 else 0

            sector_data.setdefault(cls.sector_name, []).append(ret)
            sector_vol.setdefault(cls.sector_name, []).append(vol)
            sector_count[cls.sector_name] = sector_count.get(cls.sector_name, 0) + 1

        rows = []
        for sector in sorted(sector_data.keys()):
            avg_ret = np.mean(sector_data[sector])
            avg_vol = np.mean(sector_vol[sector])
            sharpe = avg_ret / avg_vol if avg_vol > 0 else 0
            rows.append({
                "sector": sector,
                "return_pct": avg_ret * 100,
                "volatility_ann": avg_vol * 100,
                "sharpe": sharpe,
                "num_stocks": sector_count[sector],
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("sector").sort_values("return_pct", ascending=False)
        return df

    def export_for_backtest(
        self,
        symbols: List[str],
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        Export a clean price DataFrame suitable for backtesting.

        Parameters
        ----------
        symbols : list of str
        column : str
            Which column to extract.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, one column per symbol.
        """
        if self.universe is None:
            return pd.DataFrame()

        data: Dict[str, pd.Series] = {}
        for sym in symbols:
            df = self.universe.all_dataframes.get(sym)
            if df is not None and column in df.columns:
                data[sym] = df[column]

        return pd.DataFrame(data).sort_index().dropna(how="all")

    def summary(self) -> str:
        """Return a text summary of the fetched universe."""
        if self.universe is None:
            return "Universe not yet fetched. Call .fetch() first."

        lines = [
            f"Universe Summary ({self.start_date} to {self.end_date})",
            f"=" * 50,
            f"Equities:    {len(self.universe.equities):>5d}",
            f"Bonds:       {len(self.universe.bonds):>5d}",
            f"Commodities: {len(self.universe.commodities):>5d}",
            f"Crypto:      {len(self.universe.crypto):>5d}",
            f"FX:          {len(self.universe.fx):>5d}",
            f"Total:       {len(self.universe.all_symbols):>5d}",
            f"",
            f"GICS Sector Breakdown:",
        ]

        if self.universe.gics_classification:
            for sector, classifications in sorted(self.universe.gics_classification.items()):
                lines.append(f"  {sector:<35s} {len(classifications):>3d} stocks")

        return "\n".join(lines)
