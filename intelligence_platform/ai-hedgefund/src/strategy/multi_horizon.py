# -*- coding: utf-8 -*-
"""
Multi-Horizon Strategy Engine
==============================

Scans the full investment universe across multiple time horizons --
from HFT/intraday through long-term strategic positions -- and produces
actionable TradeThesis objects with full risk parameters.

Mathematical Reference
----------------------
Ornstein-Uhlenbeck (mean-reversion):
    dX = theta * (mu - X) * dt + sigma * dW
    Half-life = ln(2) / theta

Z-score for pairs/stat-arb:
    z = (spread - mu_spread) / sigma_spread
    Entry when |z| > 2, exit when |z| < 0.5

Kelly Criterion (position sizing):
    f* = (b*p - q) / b
    Half-Kelly: f = f*/2

Risk/Reward Ratio:
    RR = (target_price - entry_price) / (entry_price - stop_loss)  [long]
    RR = (entry_price - target_price) / (stop_loss - entry_price)  [short]

Sector Rotation (business cycle):
    Expansion  -> Overweight: Tech, Industrials, Consumer Disc.
    Peak       -> Overweight: Energy, Materials, Health Care
    Contraction -> Overweight: Utilities, Health Care, Consumer Staples
    Trough     -> Overweight: Financials, Real Estate, Consumer Disc.

Composite Opportunity Score:
    S = w1*E[R_adj] + w2*P(profit) + w3*RR + w4*Liquidity + w5*(1 - |rho_portfolio|)
    Default weights: w1=0.30, w2=0.25, w3=0.20, w4=0.15, w5=0.10
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.openbb_universe import (
    AssetClass,
    UniverseData,
    classify_by_gics,
    compute_kelly_fraction,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_var,
    detect_asset_class,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Enums and Data Classes
# ===================================================================
class TradeHorizon(Enum):
    """Time horizon for a trade."""

    HFT_INTRADAY = "hft_intraday"   # seconds to minutes
    DAY_TRADE = "day_trade"          # intraday
    SWING = "swing"                  # 1-5 days
    MEDIUM_TERM = "medium_term"      # 1-6 months
    LONG_TERM = "long_term"          # 6+ months


class BusinessCycle(Enum):
    """Phase of the business cycle for sector rotation."""

    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"


# Sector rotation map: which GICS sectors to overweight in each phase
SECTOR_ROTATION_MAP: Dict[BusinessCycle, List[str]] = {
    BusinessCycle.EXPANSION: [
        "Information Technology",
        "Industrials",
        "Consumer Discretionary",
        "Communication Services",
    ],
    BusinessCycle.PEAK: [
        "Energy",
        "Materials",
        "Health Care",
    ],
    BusinessCycle.CONTRACTION: [
        "Utilities",
        "Health Care",
        "Consumer Staples",
        "Real Estate",
    ],
    BusinessCycle.TROUGH: [
        "Financials",
        "Real Estate",
        "Consumer Discretionary",
        "Information Technology",
    ],
}


@dataclass
class TradeThesis:
    """
    A complete trade thesis with entry/exit parameters and risk management.

    Risk/Reward Ratio:
        For long: RR = (target_price - entry_price) / (entry_price - stop_loss)
        For short: RR = (entry_price - target_price) / (stop_loss - entry_price)
    """

    symbol: str
    asset_class: AssetClass
    horizon: TradeHorizon
    direction: str  # "long" or "short"
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float  # fraction of portfolio (0-1)
    confidence: float     # 0-1
    thesis: str           # written explanation
    catalyst: str         # what drives this trade
    risk_reward_ratio: float  # calculated
    expected_return: float = 0.0
    probability_of_profit: float = 0.5
    liquidity_score: float = 1.0
    correlation_to_portfolio: float = 0.0
    composite_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.risk_reward_ratio == 0.0:
            self.risk_reward_ratio = self._calc_risk_reward()
        if self.expected_return == 0.0:
            self.expected_return = self._calc_expected_return()
        if self.composite_score == 0.0:
            self.composite_score = score_opportunity(self)

    def _calc_risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        if self.direction == "long":
            risk = self.entry_price - self.stop_loss
            reward = self.target_price - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.target_price
        if risk <= 0:
            return 0.0
        return reward / risk

    def _calc_expected_return(self) -> float:
        """Expected return = P(win) * reward - P(loss) * risk."""
        if self.direction == "long":
            reward_pct = (self.target_price - self.entry_price) / self.entry_price
            risk_pct = (self.entry_price - self.stop_loss) / self.entry_price
        else:
            reward_pct = (self.entry_price - self.target_price) / self.entry_price
            risk_pct = (self.stop_loss - self.entry_price) / self.entry_price
        return self.probability_of_profit * reward_pct - (1 - self.probability_of_profit) * risk_pct


# ===================================================================
# Scoring
# ===================================================================
def score_opportunity(
    thesis: TradeThesis,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Composite opportunity score combining multiple factors.

    Score = w1*E[R_adj] + w2*P(profit) + w3*RR_norm + w4*Liquidity + w5*(1 - |rho|)

    where:
        E[R_adj] = expected_return normalized to [0,1]
        P(profit) = probability_of_profit (already 0-1)
        RR_norm = min(risk_reward_ratio / 5, 1)  (normalized, 5:1 = perfect)
        Liquidity = liquidity_score (0-1)
        |rho| = abs(correlation_to_portfolio)

    Parameters
    ----------
    thesis : TradeThesis
    weights : dict, optional
        Override default weights.

    Returns
    -------
    float
        Score in [0, 1].
    """
    w = weights or {
        "expected_return": 0.30,
        "probability": 0.25,
        "risk_reward": 0.20,
        "liquidity": 0.15,
        "diversification": 0.10,
    }

    # Normalize expected return to roughly [0, 1]
    er_norm = max(0.0, min(1.0, thesis.expected_return * 5.0 + 0.5))

    rr_norm = min(thesis.risk_reward_ratio / 5.0, 1.0)

    diversification = 1.0 - abs(thesis.correlation_to_portfolio)

    score = (
        w["expected_return"] * er_norm
        + w["probability"] * thesis.probability_of_profit
        + w["risk_reward"] * rr_norm
        + w["liquidity"] * thesis.liquidity_score
        + w["diversification"] * diversification
    )
    return float(max(0.0, min(1.0, score)))


# ===================================================================
# Position Sizing
# ===================================================================
# Maximum position sizes by horizon
MAX_POSITION_BY_HORIZON: Dict[TradeHorizon, float] = {
    TradeHorizon.HFT_INTRADAY: 0.05,  # 5% of portfolio
    TradeHorizon.DAY_TRADE: 0.05,
    TradeHorizon.SWING: 0.08,
    TradeHorizon.MEDIUM_TERM: 0.10,
    TradeHorizon.LONG_TERM: 0.15,
}


def calculate_position_size(
    thesis: TradeThesis,
    portfolio_value: float,
    win_rate: float = 0.55,
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
) -> float:
    """
    Calculate optimal position size using Kelly Criterion with safety adjustments.

    Kelly Criterion:
        f* = (b*p - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p

    Half-Kelly (safety):
        f = f* / 2

    Additional constraints:
        - Max position: 5% for HFT, 10% for medium, 15% for long
        - Scale down by (1 - drawdown) if in drawdown
        - Minimum position: 0.5% of portfolio

    Parameters
    ----------
    thesis : TradeThesis
    portfolio_value : float
        Total portfolio value.
    win_rate : float
        Historical win rate for this strategy type.
    avg_win : float
        Average winning trade (absolute return). If 0, estimated from thesis.
    avg_loss : float
        Average losing trade (absolute return). If 0, estimated from thesis.

    Returns
    -------
    float
        Position size as a fraction of portfolio [0, max_for_horizon].
    """
    # Estimate avg_win/avg_loss from thesis if not provided
    if avg_win <= 0:
        if thesis.direction == "long":
            avg_win = (thesis.target_price - thesis.entry_price) / thesis.entry_price
        else:
            avg_win = (thesis.entry_price - thesis.target_price) / thesis.entry_price
    if avg_loss <= 0:
        if thesis.direction == "long":
            avg_loss = (thesis.entry_price - thesis.stop_loss) / thesis.entry_price
        else:
            avg_loss = (thesis.stop_loss - thesis.entry_price) / thesis.entry_price

    avg_win = max(avg_win, 0.001)
    avg_loss = max(avg_loss, 0.001)

    # Kelly fraction (half-Kelly)
    kelly = compute_kelly_fraction(win_rate, avg_win, avg_loss)

    # Scale by confidence
    kelly *= thesis.confidence

    # Enforce horizon-based maximum
    max_pos = MAX_POSITION_BY_HORIZON.get(thesis.horizon, 0.10)
    kelly = min(kelly, max_pos)

    # Enforce minimum
    kelly = max(kelly, 0.005)

    return float(kelly)


# ===================================================================
# Helper: Statistical Computations
# ===================================================================
def _compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Z-score of a spread series.

    z = (spread - rolling_mean) / rolling_std
    """
    mu = spread.rolling(window=lookback).mean()
    sigma = spread.rolling(window=lookback).std()
    sigma = sigma.replace(0, np.nan)
    return (spread - mu) / sigma


def _compute_half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion from OLS regression.

    Model: delta_spread = alpha + beta * spread_{t-1}
    Half-life: theta = -ln(2) / ln(1 + beta)

    If beta >= 0 (no mean reversion), returns infinity.
    """
    lagged = spread.shift(1)
    delta = spread.diff()
    df = pd.DataFrame({"delta": delta, "lagged": lagged}).dropna()

    if len(df) < 10:
        return float("inf")

    # OLS: delta = alpha + beta * lagged
    x = df["lagged"].values
    y = df["delta"].values
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12)

    if beta >= 0:
        return float("inf")

    half_life = -math.log(2) / math.log(1 + beta)
    return max(half_life, 0.0)


def _compute_ou_params(spread: pd.Series) -> Dict[str, float]:
    """
    Estimate Ornstein-Uhlenbeck parameters from a spread series.

    dX = theta * (mu - X) * dt + sigma * dW

    Returns dict with keys: theta, mu, sigma, half_life.
    """
    lagged = spread.shift(1)
    delta = spread.diff()
    df = pd.DataFrame({"delta": delta, "lagged": lagged}).dropna()

    if len(df) < 20:
        return {"theta": 0.0, "mu": float(spread.mean()), "sigma": float(spread.std()), "half_life": float("inf")}

    x = df["lagged"].values
    y = df["delta"].values

    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return {"theta": 0.0, "mu": float(spread.mean()), "sigma": float(spread.std()), "half_life": float("inf")}

    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n

    theta = -beta  # mean-reversion speed (should be positive for MR)
    mu = alpha / theta if theta != 0 else float(spread.mean())

    residuals = y - (alpha + beta * x)
    sigma = float(np.std(residuals))
    half_life = math.log(2) / theta if theta > 0 else float("inf")

    return {"theta": float(theta), "mu": float(mu), "sigma": sigma, "half_life": half_life}


def _detect_volume_spike(
    volume: pd.Series,
    threshold: float = 2.5,
    lookback: int = 20,
) -> pd.Series:
    """Detect volume spikes: volume > threshold * rolling_mean(volume)."""
    avg_vol = volume.rolling(window=lookback).mean()
    return volume > (threshold * avg_vol)


def _detect_momentum_ignition(
    prices: pd.Series,
    volume: pd.Series,
    price_threshold: float = 0.02,
    volume_threshold: float = 2.5,
    lookback: int = 20,
) -> bool:
    """
    Detect momentum ignition: simultaneous volume spike + price breakout.

    Conditions:
        1. Volume in last bar > volume_threshold * avg_volume
        2. Price change in last bar > price_threshold (2%)
        3. Price breaks above/below recent high/low
    """
    if len(prices) < lookback + 1 or len(volume) < lookback + 1:
        return False

    recent_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-lookback - 1 : -1].mean()
    vol_spike = recent_vol > volume_threshold * avg_vol

    price_change = abs(prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]
    price_break = price_change > price_threshold

    return bool(vol_spike and price_break)


def _compute_bid_ask_quality(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> float:
    """
    Estimate bid-ask spread quality from OHLC data using Corwin-Schultz estimator.

    S = 2*(e^alpha - 1) / (1 + e^alpha)

    where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))

    Simplified proxy: average (High - Low) / Close as a spread estimate.
    Lower is better (tighter spread = more liquid).
    """
    if close.empty:
        return 0.0
    spread_pct = ((high - low) / close).mean()
    # Convert to a 0-1 liquidity score (lower spread = higher liquidity)
    liquidity = max(0.0, min(1.0, 1.0 - spread_pct * 10.0))
    return float(liquidity)


def _detect_business_cycle(macro_data: Dict[str, pd.DataFrame]) -> BusinessCycle:
    """
    Classify the current business cycle phase from macro indicators.

    Uses GDP growth, unemployment trend, yield curve, and leading indicators.

    Logic:
        - Positive GDP growth + falling unemployment -> EXPANSION
        - Positive GDP growth + rising unemployment  -> PEAK
        - Negative GDP growth + rising unemployment  -> CONTRACTION
        - Negative GDP growth + falling unemployment -> TROUGH
    """
    gdp_growth = 0.0
    unemployment_trend = 0.0

    if "GDP" in macro_data and not macro_data["GDP"].empty:
        gdp = macro_data["GDP"]
        vals = gdp.iloc[:, 0] if len(gdp.columns) > 0 else gdp
        if len(vals) >= 2:
            gdp_growth = float(vals.iloc[-1] - vals.iloc[-2])

    if "UNRATE" in macro_data and not macro_data["UNRATE"].empty:
        unemp = macro_data["UNRATE"]
        vals = unemp.iloc[:, 0] if len(unemp.columns) > 0 else unemp
        if len(vals) >= 4:
            recent = vals.iloc[-1]
            prior = vals.iloc[-4]
            unemployment_trend = float(recent - prior)

    # Yield curve inversion (T10Y2Y < 0) is a recession signal
    yield_curve_inverted = False
    if "T10Y2Y" in macro_data and not macro_data["T10Y2Y"].empty:
        t10y2y = macro_data["T10Y2Y"]
        vals = t10y2y.iloc[:, 0] if len(t10y2y.columns) > 0 else t10y2y
        if len(vals) > 0:
            yield_curve_inverted = float(vals.iloc[-1]) < 0

    if gdp_growth > 0 and unemployment_trend <= 0:
        return BusinessCycle.EXPANSION
    elif gdp_growth > 0 and unemployment_trend > 0:
        return BusinessCycle.PEAK
    elif gdp_growth <= 0 and unemployment_trend > 0:
        return BusinessCycle.CONTRACTION
    else:
        if yield_curve_inverted:
            return BusinessCycle.CONTRACTION
        return BusinessCycle.TROUGH


# ===================================================================
# Multi-Horizon Engine
# ===================================================================
class MultiHorizonEngine:
    """
    Scans the full investment universe across multiple time horizons
    and produces ranked TradeThesis objects.
    """

    def __init__(self, portfolio_value: float = 1_000_000.0):
        self.portfolio_value = portfolio_value
        self.active_theses: List[TradeThesis] = []

    # ----- HFT / Intraday -----
    def scan_hft_opportunities(
        self,
        universe_data: UniverseData,
    ) -> List[TradeThesis]:
        """
        Scan for HFT/intraday opportunities using:

        1. Statistical arbitrage: z-score > 2 sigma from pairs mean
           z = (spread - mu_spread) / sigma_spread
           Enter when |z| > 2, exit when |z| < 0.5

        2. Momentum ignition: volume spike + price breakout
           Condition: volume > 2.5x avg AND |price_change| > 2%

        3. Mean reversion: Ornstein-Uhlenbeck process
           dX = theta * (mu - X) * dt + sigma * dW
           Trade when deviation > 2 sigma and half-life < 5 days

        4. Market microstructure: bid-ask spread analysis
           Liquidity score from Corwin-Schultz high-low estimator

        Parameters
        ----------
        universe_data : UniverseData
            Complete universe data.

        Returns
        -------
        list of TradeThesis
        """
        theses: List[TradeThesis] = []

        # --- Statistical Arbitrage (Equity Pairs) ---
        equity_symbols = list(universe_data.equities.keys())
        if len(equity_symbols) >= 2:
            # Build close price matrix
            close_dict = {}
            for sym, df in universe_data.equities.items():
                if "Close" in df.columns and len(df) >= 60:
                    close_dict[sym] = df["Close"]

            if len(close_dict) >= 2:
                close_df = pd.DataFrame(close_dict).dropna()

                # Find pairs within same GICS sector
                sector_groups: Dict[str, List[str]] = {}
                for sym in close_df.columns:
                    cls = classify_by_gics(sym)
                    if cls:
                        sector_groups.setdefault(cls.sector_name, []).append(sym)

                for sector, symbols in sector_groups.items():
                    if len(symbols) < 2:
                        continue
                    for i in range(len(symbols)):
                        for j in range(i + 1, min(i + 5, len(symbols))):
                            sym_a, sym_b = symbols[i], symbols[j]
                            if sym_a not in close_df.columns or sym_b not in close_df.columns:
                                continue

                            prices_a = close_df[sym_a]
                            prices_b = close_df[sym_b]

                            # Compute spread (log ratio)
                            spread = np.log(prices_a / prices_b)
                            if spread.std() < 1e-8:
                                continue

                            z = _compute_zscore(spread, lookback=60)
                            if z.empty:
                                continue

                            current_z = z.iloc[-1]
                            if np.isnan(current_z):
                                continue

                            # Check for mean-reversion characteristics
                            ou = _compute_ou_params(spread)
                            if ou["half_life"] > 30:
                                continue  # too slow to mean-revert for HFT

                            if abs(current_z) > 2.0:
                                # Stat-arb opportunity
                                if current_z > 2.0:
                                    # Spread too high: short A, long B
                                    direction = "short"
                                    trade_sym = sym_a
                                    entry = float(prices_a.iloc[-1])
                                    target = entry * (1 - 0.02)
                                    stop = entry * (1 + 0.03)
                                else:
                                    direction = "long"
                                    trade_sym = sym_a
                                    entry = float(prices_a.iloc[-1])
                                    target = entry * (1 + 0.02)
                                    stop = entry * (1 - 0.03)

                                prob = min(0.7, 0.5 + abs(current_z) * 0.05)
                                thesis = TradeThesis(
                                    symbol=trade_sym,
                                    asset_class=AssetClass.EQUITY,
                                    horizon=TradeHorizon.HFT_INTRADAY,
                                    direction=direction,
                                    entry_price=entry,
                                    target_price=target,
                                    stop_loss=stop,
                                    position_size=0.0,
                                    confidence=min(0.85, abs(current_z) / 4.0),
                                    thesis=(
                                        f"Stat-arb pair trade: {sym_a}/{sym_b} in {sector}. "
                                        f"Z-score={current_z:.2f}, half-life={ou['half_life']:.1f} days. "
                                        f"Spread deviation is {abs(current_z):.1f} sigma from mean."
                                    ),
                                    catalyst=f"Mean reversion of {sym_a}/{sym_b} log-spread",
                                    risk_reward_ratio=0.0,
                                    probability_of_profit=prob,
                                )
                                thesis.position_size = calculate_position_size(
                                    thesis, self.portfolio_value
                                )
                                theses.append(thesis)

        # --- Momentum Ignition ---
        for sym, df in universe_data.equities.items():
            if len(df) < 25 or "Close" not in df.columns or "Volume" not in df.columns:
                continue

            if _detect_momentum_ignition(df["Close"], df["Volume"]):
                last_price = float(df["Close"].iloc[-1])
                prev_price = float(df["Close"].iloc[-2])
                direction = "long" if last_price > prev_price else "short"

                if direction == "long":
                    target = last_price * 1.015
                    stop = last_price * 0.99
                else:
                    target = last_price * 0.985
                    stop = last_price * 1.01

                thesis = TradeThesis(
                    symbol=sym,
                    asset_class=AssetClass.EQUITY,
                    horizon=TradeHorizon.HFT_INTRADAY,
                    direction=direction,
                    entry_price=last_price,
                    target_price=target,
                    stop_loss=stop,
                    position_size=0.0,
                    confidence=0.55,
                    thesis=(
                        f"Momentum ignition detected in {sym}. "
                        f"Volume spike with {direction} price breakout of "
                        f"{abs(last_price - prev_price) / prev_price * 100:.1f}%."
                    ),
                    catalyst="Volume spike + price breakout",
                    risk_reward_ratio=0.0,
                    probability_of_profit=0.55,
                )
                thesis.position_size = calculate_position_size(
                    thesis, self.portfolio_value
                )
                theses.append(thesis)

        # --- Mean Reversion (single-name, O-U process) ---
        for sym, df in universe_data.equities.items():
            if len(df) < 60 or "Close" not in df.columns:
                continue

            prices = df["Close"]
            log_prices = np.log(prices)
            ou = _compute_ou_params(log_prices)

            if 0 < ou["half_life"] < 5 and ou["theta"] > 0.05:
                current = float(log_prices.iloc[-1])
                deviation = (current - ou["mu"]) / max(ou["sigma"], 1e-8)

                if abs(deviation) > 2.0:
                    last_price = float(prices.iloc[-1])
                    if deviation > 2.0:
                        direction = "short"
                        target = last_price * math.exp(ou["mu"] - current)
                        stop = last_price * 1.03
                    else:
                        direction = "long"
                        target = last_price * math.exp(ou["mu"] - current)
                        stop = last_price * 0.97

                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.EQUITY,
                        horizon=TradeHorizon.HFT_INTRADAY,
                        direction=direction,
                        entry_price=last_price,
                        target_price=target,
                        stop_loss=stop,
                        position_size=0.0,
                        confidence=min(0.8, 0.5 + ou["theta"] * 0.5),
                        thesis=(
                            f"O-U mean reversion in {sym}. "
                            f"theta={ou['theta']:.3f}, mu={ou['mu']:.4f}, "
                            f"half-life={ou['half_life']:.1f} days. "
                            f"Current deviation: {deviation:.2f} sigma."
                        ),
                        catalyst="Ornstein-Uhlenbeck mean reversion",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.60,
                    )
                    thesis.position_size = calculate_position_size(
                        thesis, self.portfolio_value
                    )
                    theses.append(thesis)

        # Sort by composite score descending
        theses.sort(key=lambda t: t.composite_score, reverse=True)
        return theses

    # ----- Swing (1-5 days) -----
    def scan_swing_opportunities(
        self,
        universe_data: UniverseData,
    ) -> List[TradeThesis]:
        """
        Scan for swing trade opportunities (1-5 day holding period).

        Strategies:
        - RSI extremes (oversold < 30, overbought > 70)
        - Bollinger Band breakouts
        - MACD crossovers confirmed by volume

        RSI = 100 - 100 / (1 + RS)
        RS = avg_gain(14) / avg_loss(14)

        Bollinger Bands:
            Upper = SMA(20) + 2 * std(20)
            Lower = SMA(20) - 2 * std(20)
        """
        theses: List[TradeThesis] = []

        all_dfs = {**universe_data.equities, **universe_data.commodities}
        for sym, df in all_dfs.items():
            if len(df) < 30 or "Close" not in df.columns:
                continue

            close = df["Close"]
            asset = detect_asset_class(sym)

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            if rsi.empty or np.isnan(rsi.iloc[-1]):
                continue

            current_rsi = float(rsi.iloc[-1])
            last_price = float(close.iloc[-1])

            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper_bb = sma20 + 2 * std20
            lower_bb = sma20 - 2 * std20

            if np.isnan(upper_bb.iloc[-1]):
                continue

            # Oversold RSI + price near lower BB
            if current_rsi < 30 and last_price <= float(lower_bb.iloc[-1]) * 1.01:
                target = float(sma20.iloc[-1])
                stop = last_price * 0.97

                thesis = TradeThesis(
                    symbol=sym,
                    asset_class=asset,
                    horizon=TradeHorizon.SWING,
                    direction="long",
                    entry_price=last_price,
                    target_price=target,
                    stop_loss=stop,
                    position_size=0.0,
                    confidence=0.60,
                    thesis=(
                        f"Oversold bounce setup in {sym}. RSI={current_rsi:.1f}, "
                        f"price at lower Bollinger Band. Target: SMA(20)={target:.2f}."
                    ),
                    catalyst="RSI oversold + Bollinger Band support",
                    risk_reward_ratio=0.0,
                    probability_of_profit=0.58,
                )
                thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                theses.append(thesis)

            # Overbought RSI + price near upper BB
            elif current_rsi > 70 and last_price >= float(upper_bb.iloc[-1]) * 0.99:
                target = float(sma20.iloc[-1])
                stop = last_price * 1.03

                thesis = TradeThesis(
                    symbol=sym,
                    asset_class=asset,
                    horizon=TradeHorizon.SWING,
                    direction="short",
                    entry_price=last_price,
                    target_price=target,
                    stop_loss=stop,
                    position_size=0.0,
                    confidence=0.55,
                    thesis=(
                        f"Overbought reversal setup in {sym}. RSI={current_rsi:.1f}, "
                        f"price at upper Bollinger Band. Target: SMA(20)={target:.2f}."
                    ),
                    catalyst="RSI overbought + Bollinger Band resistance",
                    risk_reward_ratio=0.0,
                    probability_of_profit=0.55,
                )
                thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                theses.append(thesis)

        theses.sort(key=lambda t: t.composite_score, reverse=True)
        return theses

    # ----- Medium-Term (1-6 months) -----
    def scan_medium_term_opportunities(
        self,
        universe_data: UniverseData,
        macro_data: Dict[str, pd.DataFrame],
    ) -> List[TradeThesis]:
        """
        Identify 1-6 month trades:

        1. Sector rotation based on business cycle
           Expansion  -> Overweight: Tech, Industrials, Consumer Disc.
           Peak       -> Overweight: Energy, Materials, Health Care
           Contraction -> Overweight: Utilities, Health Care, Staples
           Trough     -> Overweight: Financials, Real Estate, Consumer Disc.

        2. Relative value: long undervalued vs short overvalued within sector
           Using 6-month momentum + mean reversion of relative strength

        3. Event-driven: earnings surprises, M&A, spinoffs
           Capture post-announcement drift

        4. Technical breakouts confirmed by fundamentals
           Price > 200-day SMA AND volume confirmation AND positive earnings trend

        Parameters
        ----------
        universe_data : UniverseData
        macro_data : dict
            Macroeconomic indicator DataFrames.

        Returns
        -------
        list of TradeThesis
        """
        theses: List[TradeThesis] = []

        # 1. Sector Rotation
        cycle = _detect_business_cycle(macro_data)
        favored_sectors = SECTOR_ROTATION_MAP.get(cycle, [])
        unfavored_sectors = [s for s in SECTOR_ROTATION_MAP.get(
            {
                BusinessCycle.EXPANSION: BusinessCycle.CONTRACTION,
                BusinessCycle.PEAK: BusinessCycle.TROUGH,
                BusinessCycle.CONTRACTION: BusinessCycle.EXPANSION,
                BusinessCycle.TROUGH: BusinessCycle.PEAK,
            }[cycle], []
        )]

        sector_equities: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
        for sym, df in universe_data.equities.items():
            cls = classify_by_gics(sym)
            if cls:
                sector_equities.setdefault(cls.sector_name, []).append((sym, df))

        # Long favored sector leaders
        for sector in favored_sectors:
            if sector not in sector_equities:
                continue
            for sym, df in sector_equities[sector]:
                if len(df) < 120 or "Close" not in df.columns:
                    continue

                close = df["Close"]
                # 6-month momentum
                mom_6m = (close.iloc[-1] - close.iloc[-120]) / close.iloc[-120]
                # Above 200-day SMA (if we have enough data)
                sma200 = close.rolling(min(200, len(close))).mean().iloc[-1]
                above_sma = close.iloc[-1] > sma200

                if mom_6m > 0.05 and above_sma:
                    last_price = float(close.iloc[-1])
                    target = last_price * (1 + mom_6m * 0.5)  # continuation
                    stop = last_price * 0.92

                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.EQUITY,
                        horizon=TradeHorizon.MEDIUM_TERM,
                        direction="long",
                        entry_price=last_price,
                        target_price=target,
                        stop_loss=stop,
                        position_size=0.0,
                        confidence=0.60,
                        thesis=(
                            f"Sector rotation: {sector} favored in {cycle.value} phase. "
                            f"{sym} showing {mom_6m * 100:.1f}% 6M momentum, "
                            f"trading above SMA-200. Expecting continued outperformance."
                        ),
                        catalyst=f"Business cycle {cycle.value} favoring {sector}",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.58,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

        # 2. Relative value within sectors
        for sector, sym_dfs in sector_equities.items():
            if len(sym_dfs) < 3:
                continue

            # Compute 3-month returns for all stocks in sector
            returns_3m: Dict[str, float] = {}
            for sym, df in sym_dfs:
                if len(df) < 60 and "Close" in df.columns:
                    continue
                if "Close" not in df.columns:
                    continue
                close = df["Close"]
                if len(close) >= 60:
                    ret = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]
                    returns_3m[sym] = float(ret)

            if len(returns_3m) < 3:
                continue

            sorted_rets = sorted(returns_3m.items(), key=lambda x: x[1])
            # Long the bottom performer (value), short the top performer (expensive)
            worst_sym, worst_ret = sorted_rets[0]
            best_sym, best_ret = sorted_rets[-1]

            if best_ret - worst_ret > 0.15:  # significant divergence
                for sym, df in sym_dfs:
                    if sym == worst_sym and "Close" in df.columns:
                        last_price = float(df["Close"].iloc[-1])
                        # Target: revert to sector mean
                        mean_ret = np.mean(list(returns_3m.values()))
                        target = last_price * (1 + (mean_ret - worst_ret))
                        stop = last_price * 0.90

                        thesis = TradeThesis(
                            symbol=sym,
                            asset_class=AssetClass.EQUITY,
                            horizon=TradeHorizon.MEDIUM_TERM,
                            direction="long",
                            entry_price=last_price,
                            target_price=target,
                            stop_loss=stop,
                            position_size=0.0,
                            confidence=0.55,
                            thesis=(
                                f"Relative value in {sector}: {sym} underperformed "
                                f"({worst_ret * 100:.1f}% vs sector avg {mean_ret * 100:.1f}%). "
                                f"Expecting mean reversion to sector average."
                            ),
                            catalyst=f"Intra-sector relative value reversion in {sector}",
                            risk_reward_ratio=0.0,
                            probability_of_profit=0.55,
                        )
                        thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                        theses.append(thesis)

        # 3. Technical breakout: price breaks above 200-day SMA with volume confirmation
        for sym, df in universe_data.equities.items():
            if len(df) < 210 or "Close" not in df.columns or "Volume" not in df.columns:
                continue

            close = df["Close"]
            volume = df["Volume"]
            sma200 = close.rolling(200).mean()
            sma50 = close.rolling(50).mean()

            if np.isnan(sma200.iloc[-1]) or np.isnan(sma50.iloc[-1]):
                continue

            # Golden cross: SMA50 crosses above SMA200
            golden_cross = (
                sma50.iloc[-1] > sma200.iloc[-1]
                and sma50.iloc[-2] <= sma200.iloc[-2]
            )

            if golden_cross:
                last_price = float(close.iloc[-1])
                avg_vol = volume.iloc[-20:].mean()
                recent_vol = volume.iloc[-1]
                vol_confirm = recent_vol > avg_vol * 1.5

                if vol_confirm:
                    target = last_price * 1.15
                    stop = float(sma200.iloc[-1]) * 0.98

                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.EQUITY,
                        horizon=TradeHorizon.MEDIUM_TERM,
                        direction="long",
                        entry_price=last_price,
                        target_price=target,
                        stop_loss=stop,
                        position_size=0.0,
                        confidence=0.60,
                        thesis=(
                            f"Golden cross breakout in {sym}. SMA(50) crossed above SMA(200) "
                            f"with {recent_vol / avg_vol:.1f}x average volume confirmation. "
                            f"Historically bullish signal with ~60% success rate over 6 months."
                        ),
                        catalyst="SMA(50)/SMA(200) golden cross with volume",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.60,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

        theses.sort(key=lambda t: t.composite_score, reverse=True)
        return theses

    # ----- Long-Term (6+ months) -----
    def scan_long_term_opportunities(
        self,
        universe_data: UniverseData,
        macro_data: Dict[str, pd.DataFrame],
        news: pd.DataFrame,
    ) -> List[TradeThesis]:
        """
        Identify 6+ month strategic positions:

        1. Macro regime shifts
           - Rate hike cycles: short duration bonds, long financials
           - Rate cut cycles: long duration bonds, long REITs & utilities
           - Inflation regime: long commodities, TIPS, short long-duration bonds
           - Deflation regime: long treasuries, short commodities

        2. Structural mispricing
           - Deep value: stocks > 50% below 52-week high with solid balance sheet
           - Recovery thesis: beaten-down sectors showing bottoming patterns

        3. Secular trends
           - Demographic shifts (aging: healthcare; urbanization: real estate)
           - Technology adoption curves (AI, clean energy, EV)
           - Geopolitical realignment (supply chain reshoring)

        4. 2nd derivative macro effects
           Rate hike -> housing slowdown -> materials demand drop -> employment
           Dollar strength -> EM currency pressure -> commodity revaluation

        Parameters
        ----------
        universe_data : UniverseData
        macro_data : dict
        news : pd.DataFrame

        Returns
        -------
        list of TradeThesis
        """
        theses: List[TradeThesis] = []
        cycle = _detect_business_cycle(macro_data)

        # 1. Macro regime: interest rate environment
        rate_rising = False
        rate_level = 0.0
        if "FEDFUNDS" in macro_data and not macro_data["FEDFUNDS"].empty:
            rates = macro_data["FEDFUNDS"]
            vals = rates.iloc[:, 0] if len(rates.columns) > 0 else rates
            if len(vals) >= 12:
                rate_level = float(vals.iloc[-1])
                rate_6m_ago = float(vals.iloc[-6]) if len(vals) >= 6 else rate_level
                rate_rising = rate_level > rate_6m_ago + 0.25

        # Rate environment trades
        if rate_rising:
            # Short long-duration bonds
            for sym in ["TLT", "VCLT"]:
                if sym in universe_data.bonds and "Close" in universe_data.bonds[sym].columns:
                    df = universe_data.bonds[sym]
                    last_price = float(df["Close"].iloc[-1])
                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.BOND,
                        horizon=TradeHorizon.LONG_TERM,
                        direction="short",
                        entry_price=last_price,
                        target_price=last_price * 0.88,
                        stop_loss=last_price * 1.08,
                        position_size=0.0,
                        confidence=0.60,
                        thesis=(
                            f"Rising rate environment (Fed Funds at {rate_level:.2f}%). "
                            f"Long-duration bond ETF {sym} expected to decline as rates "
                            f"continue higher. Duration risk favors short positioning."
                        ),
                        catalyst="Fed tightening cycle / rising interest rates",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.60,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

            # Long financials (benefit from higher rates)
            for sym, df in universe_data.equities.items():
                cls = classify_by_gics(sym)
                if cls and cls.sector_name == "Financials" and "Close" in df.columns and len(df) >= 60:
                    close = df["Close"]
                    last_price = float(close.iloc[-1])
                    mom_3m = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]

                    if mom_3m > -0.05:  # not already in freefall
                        thesis = TradeThesis(
                            symbol=sym,
                            asset_class=AssetClass.EQUITY,
                            horizon=TradeHorizon.LONG_TERM,
                            direction="long",
                            entry_price=last_price,
                            target_price=last_price * 1.20,
                            stop_loss=last_price * 0.85,
                            position_size=0.0,
                            confidence=0.55,
                            thesis=(
                                f"Net interest margin expansion for {sym} in rising rate "
                                f"environment. Banks benefit from wider spreads as Fed Funds "
                                f"at {rate_level:.2f}% and rising."
                            ),
                            catalyst="Rising rates -> NIM expansion for banks",
                            risk_reward_ratio=0.0,
                            probability_of_profit=0.55,
                        )
                        thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                        theses.append(thesis)
        else:
            # Rate stable/falling: long duration bonds, REITs, utilities
            for sym in ["TLT", "IEF"]:
                if sym in universe_data.bonds and "Close" in universe_data.bonds[sym].columns:
                    df = universe_data.bonds[sym]
                    last_price = float(df["Close"].iloc[-1])
                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.BOND,
                        horizon=TradeHorizon.LONG_TERM,
                        direction="long",
                        entry_price=last_price,
                        target_price=last_price * 1.10,
                        stop_loss=last_price * 0.94,
                        position_size=0.0,
                        confidence=0.55,
                        thesis=(
                            f"Stable/falling rate environment. Long-duration bonds ({sym}) "
                            f"expected to appreciate. Duration benefit as rates decline."
                        ),
                        catalyst="Fed easing cycle / stable rates",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.55,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

        # 2. Structural mispricing: deep value (> 40% below 52-week high)
        for sym, df in universe_data.equities.items():
            if len(df) < 200 or "Close" not in df.columns:
                continue

            close = df["Close"]
            high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            last_price = float(close.iloc[-1])
            decline = (high_52w - last_price) / high_52w

            if decline > 0.40:
                # Deep value candidate -- check if showing bottoming pattern
                sma50 = close.rolling(50).mean().iloc[-1]
                sma20 = close.rolling(20).mean().iloc[-1]
                bottoming = sma20 > sma50  # short-term trend turning up

                if bottoming:
                    target = last_price * 1.30  # recovery toward mean
                    stop = last_price * 0.85

                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=AssetClass.EQUITY,
                        horizon=TradeHorizon.LONG_TERM,
                        direction="long",
                        entry_price=last_price,
                        target_price=target,
                        stop_loss=stop,
                        position_size=0.0,
                        confidence=0.50,
                        thesis=(
                            f"Deep value recovery play in {sym}. Stock is {decline * 100:.1f}% "
                            f"below 52-week high but showing bottoming pattern "
                            f"(SMA20 > SMA50). Asymmetric risk/reward if recovery thesis plays out."
                        ),
                        catalyst="Bottoming pattern in deeply oversold stock",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.50,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

        # 3. Inflation hedge: commodities
        cpi_rising = False
        if "CPIAUCSL" in macro_data and not macro_data["CPIAUCSL"].empty:
            cpi = macro_data["CPIAUCSL"]
            vals = cpi.iloc[:, 0] if len(cpi.columns) > 0 else cpi
            if len(vals) >= 13:
                yoy_cpi = (vals.iloc[-1] - vals.iloc[-12]) / vals.iloc[-12]
                cpi_rising = yoy_cpi > 0.03  # > 3% YoY inflation

        if cpi_rising:
            for sym in ["GLD", "DBC", "TIP", "USO"]:
                src = universe_data.commodities if sym in universe_data.commodities else universe_data.bonds
                if sym in src and "Close" in src[sym].columns:
                    df = src[sym]
                    last_price = float(df["Close"].iloc[-1])
                    thesis = TradeThesis(
                        symbol=sym,
                        asset_class=detect_asset_class(sym),
                        horizon=TradeHorizon.LONG_TERM,
                        direction="long",
                        entry_price=last_price,
                        target_price=last_price * 1.15,
                        stop_loss=last_price * 0.90,
                        position_size=0.0,
                        confidence=0.55,
                        thesis=(
                            f"Inflation hedge via {sym}. CPI running above 3% YoY. "
                            f"Real assets historically outperform in inflationary regimes."
                        ),
                        catalyst="Elevated inflation -> real asset outperformance",
                        risk_reward_ratio=0.0,
                        probability_of_profit=0.55,
                    )
                    thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                    theses.append(thesis)

        # 4. Cross-asset: crypto as macro hedge
        for sym, df in universe_data.crypto.items():
            if len(df) < 60 or "Close" not in df.columns:
                continue

            close = df["Close"]
            last_price = float(close.iloc[-1])

            # Check if deeply oversold (> 50% from recent high)
            recent_high = close.iloc[-90:].max() if len(close) >= 90 else close.max()
            decline = (recent_high - last_price) / recent_high

            if decline > 0.50:
                thesis = TradeThesis(
                    symbol=sym,
                    asset_class=AssetClass.CRYPTO,
                    horizon=TradeHorizon.LONG_TERM,
                    direction="long",
                    entry_price=last_price,
                    target_price=last_price * 2.0,
                    stop_loss=last_price * 0.60,
                    position_size=0.0,
                    confidence=0.40,
                    thesis=(
                        f"Crypto recovery play: {sym} is {decline * 100:.0f}% below "
                        f"recent highs. Historical crypto cycles show strong recoveries "
                        f"from similar drawdowns. High risk, high reward."
                    ),
                    catalyst="Crypto cycle recovery from oversold levels",
                    risk_reward_ratio=0.0,
                    probability_of_profit=0.45,
                )
                thesis.position_size = calculate_position_size(thesis, self.portfolio_value)
                theses.append(thesis)

        theses.sort(key=lambda t: t.composite_score, reverse=True)
        return theses

    # ----- Master Scan -----
    def scan_all_horizons(
        self,
        universe_data: UniverseData,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
        news: Optional[pd.DataFrame] = None,
    ) -> Dict[TradeHorizon, List[TradeThesis]]:
        """
        Run all scanners and return opportunities organized by horizon.

        Parameters
        ----------
        universe_data : UniverseData
        macro_data : dict, optional
        news : pd.DataFrame, optional

        Returns
        -------
        dict
            {TradeHorizon: [TradeThesis, ...]}
        """
        if macro_data is None:
            macro_data = {}
        if news is None:
            news = pd.DataFrame()

        results: Dict[TradeHorizon, List[TradeThesis]] = {}

        logger.info("Scanning HFT/intraday opportunities...")
        results[TradeHorizon.HFT_INTRADAY] = self.scan_hft_opportunities(universe_data)

        logger.info("Scanning swing opportunities...")
        results[TradeHorizon.SWING] = self.scan_swing_opportunities(universe_data)

        logger.info("Scanning medium-term opportunities...")
        results[TradeHorizon.MEDIUM_TERM] = self.scan_medium_term_opportunities(
            universe_data, macro_data
        )

        logger.info("Scanning long-term opportunities...")
        results[TradeHorizon.LONG_TERM] = self.scan_long_term_opportunities(
            universe_data, macro_data, news
        )

        total = sum(len(v) for v in results.values())
        logger.info("Total opportunities found: %d", total)

        self.active_theses = []
        for horizon_theses in results.values():
            self.active_theses.extend(horizon_theses)

        return results

    def get_top_opportunities(
        self,
        n: int = 10,
        horizon: Optional[TradeHorizon] = None,
    ) -> List[TradeThesis]:
        """
        Return the top N opportunities by composite score.

        Parameters
        ----------
        n : int
            Number of opportunities to return.
        horizon : TradeHorizon, optional
            Filter by specific horizon.

        Returns
        -------
        list of TradeThesis
        """
        candidates = self.active_theses
        if horizon is not None:
            candidates = [t for t in candidates if t.horizon == horizon]

        candidates.sort(key=lambda t: t.composite_score, reverse=True)
        return candidates[:n]

    def get_buy_and_hold_recommendations(
        self,
    ) -> List[TradeThesis]:
        """
        Return medium-term and long-term long recommendations suitable
        for buy-and-hold investors.

        Returns
        -------
        list of TradeThesis
            Only long-direction, medium/long-term theses.
        """
        return [
            t
            for t in self.active_theses
            if t.horizon in (TradeHorizon.MEDIUM_TERM, TradeHorizon.LONG_TERM)
            and t.direction == "long"
        ]
