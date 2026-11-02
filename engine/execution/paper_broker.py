"""PaperBroker — Simulation broker for backtesting and futures paper trading.

PURPOSE:
    1. Backtesting — historical simulation with synthetic fills
    2. Futures — paper trading until Rithmic connection is live

NOT for live equity/options trading — use AlpacaBroker.
NOT for live futures — will use RithmicBroker (future build).

Provides:
    - Order placement (market, limit)
    - Position tracking with GICS sector classification
    - P&L calculation
    - NAV computation
    - Trade history / audit trail
    - Fill simulation with micro-price model
    - Pre-trade risk checks (RiskLimiter)
    - Performance tracking (Sharpe, drawdown, win rate)
    - 5% daily compound target with risk dial-down after hit
    - Live dashboard state emission for real-time observation
    - Alpha maximization through positioning + selection
    - Beta management through leverage execution multipliers

Risk Dial-Down Logic (post-target):
    AGGRESSIVE  — Pre-target: full leverage, max opportunity scanning
    MODERATE    — Target hit (5%): reduce leverage 50%, widen stops
    DEFENSIVE   — Target + buffer (6%): minimal new positions, protect gains

Designed to be swapped for a live broker API later.
"""

import uuid
import json
import time
import csv
import io
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

from ..data.openbb_data import get_adj_close
from ..utils.money import D, money, to_float, safe_div


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalType(str, Enum):
    """Signal types for trade classification."""
    MICRO_PRICE_BUY = "MICRO_PRICE_BUY"
    MICRO_PRICE_SELL = "MICRO_PRICE_SELL"
    RV_LONG = "RV_LONG"
    RV_SHORT = "RV_SHORT"
    FALLEN_ANGEL_BUY = "FALLEN_ANGEL_BUY"
    ML_AGENT_BUY = "ML_AGENT_BUY"
    ML_AGENT_SELL = "ML_AGENT_SELL"
    DRL_AGENT_BUY = "DRL_AGENT_BUY"
    DRL_AGENT_SELL = "DRL_AGENT_SELL"
    TFT_BUY = "TFT_BUY"
    TFT_SELL = "TFT_SELL"
    MC_BUY = "MC_BUY"
    MC_SELL = "MC_SELL"
    QUALITY_BUY = "QUALITY_BUY"
    QUALITY_SELL = "QUALITY_SELL"
    SOCIAL_BULLISH = "SOCIAL_BULLISH"
    SOCIAL_BEARISH = "SOCIAL_BEARISH"
    SOCIAL_MOMENTUM = "SOCIAL_MOMENTUM"
    SOCIAL_REVERSAL = "SOCIAL_REVERSAL"
    # Distressed asset signals
    DISTRESS_FALLEN_ANGEL = "DISTRESS_FALLEN_ANGEL"
    DISTRESS_RECOVERY = "DISTRESS_RECOVERY"
    DISTRESS_AVOID = "DISTRESS_AVOID"
    # CVR signals
    CVR_BUY = "CVR_BUY"
    CVR_SELL = "CVR_SELL"
    # Event-driven signals
    EVENT_MERGER_ARB = "EVENT_MERGER_ARB"
    EVENT_PEAD_LONG = "EVENT_PEAD_LONG"
    EVENT_PEAD_SHORT = "EVENT_PEAD_SHORT"
    EVENT_CATALYST = "EVENT_CATALYST"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Order:
    id: str = ""
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: Optional[float] = None
    fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    signal_type: SignalType = SignalType.HOLD
    timestamp: str = ""
    fill_timestamp: str = ""
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Enum) else v for k, v in asdict(self).items()}


@dataclass
class Position:
    ticker: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    sector: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # ── Decimal-precise financial properties ──────────────────
    @property
    def market_value(self) -> float:
        return to_float(money(D(self.quantity) * D(self.current_price)))

    @property
    def cost_basis(self) -> float:
        return to_float(money(D(abs(self.quantity)) * D(self.avg_cost)))

    def settle_buy(self, qty: int, price: float) -> None:
        """Settle a buy fill with Decimal-precise cost averaging."""
        d_cost = D(self.avg_cost) * D(self.quantity) + D(price) * D(qty)
        new_qty = self.quantity + qty
        self.avg_cost = to_float(safe_div(d_cost, new_qty)) if new_qty != 0 else 0.0
        self.quantity = new_qty

    def settle_sell(self, qty: int, price: float) -> float:
        """Settle a sell fill. Returns realized P&L with Decimal precision."""
        d_pnl = D(price - self.avg_cost) * D(qty)
        self.quantity -= qty
        realized = to_float(money(d_pnl))
        self.realized_pnl = to_float(money(D(self.realized_pnl) + d_pnl))
        return realized


@dataclass
class PortfolioState:
    cash: float = 1_000_000.0
    positions: dict = field(default_factory=dict)  # ticker → Position
    nav: float = 1_000_000.0
    total_pnl: float = 0.0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    beta: float = 0.0


# ---------------------------------------------------------------------------
# MicroPriceModel — Realistic fill simulation
# ---------------------------------------------------------------------------
class MicroPriceModel:
    """Simulates realistic order fill prices using bid/ask spread modelling,
    market impact estimation, and time-of-day slippage adjustments.

    All parameters are calibrated to US equity markets and driven entirely
    by OpenBB data — no live order-book feed required.
    """

    # Approximate average daily volumes for liquidity tiers (shares)
    _LIQUIDITY_TIERS = {
        "mega":  50_000_000,   # AAPL, MSFT, TSLA
        "large": 10_000_000,   # Mid-large S&P 500
        "mid":    2_000_000,   # S&P 400
        "small":    500_000,   # S&P 600
    }

    # Typical half-spread in bps by liquidity tier
    _HALF_SPREAD_BPS = {
        "mega":  0.5,
        "large": 1.5,
        "mid":   3.0,
        "small": 6.0,
    }

    # Time-of-day slippage multiplier (hour of day ET → multiplier)
    # Higher at open (9) and close (15-16), lower midday
    _TOD_MULTIPLIER = {
        9:  1.80,   # Opening auction volatility
        10: 1.30,
        11: 1.00,
        12: 0.90,
        13: 0.90,
        14: 1.00,
        15: 1.40,   # Approaching close
        16: 1.60,   # Closing auction
    }

    def __init__(self, spread_multiplier: float = 1.0):
        self.spread_multiplier = spread_multiplier
        self._adv_cache: dict[str, float] = {}

    def classify_liquidity(self, adv: float) -> str:
        """Return liquidity tier string based on average daily volume."""
        if adv >= self._LIQUIDITY_TIERS["mega"]:
            return "mega"
        elif adv >= self._LIQUIDITY_TIERS["large"]:
            return "large"
        elif adv >= self._LIQUIDITY_TIERS["mid"]:
            return "mid"
        return "small"

    def estimate_adv(self, ticker: str) -> float:
        """Estimate average daily volume from recent OpenBB data."""
        if ticker in self._adv_cache:
            return self._adv_cache[ticker]
        try:
            from ..data.openbb_data import get_prices
            from datetime import timedelta
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            hist = get_prices(ticker, start=start, end=end)
            if hist is not None and not hist.empty:
                # Extract volume from DataFrame (may be MultiIndex or flat)
                if isinstance(hist.columns, pd.MultiIndex):
                    vol = hist["Volume"] if "Volume" in hist.columns.get_level_values(0) else None
                else:
                    vol = hist["Volume"] if "Volume" in hist.columns else None
                if vol is not None and len(vol) > 0:
                    adv = float(vol.mean())
                else:
                    adv = float(self._LIQUIDITY_TIERS["mid"])
            else:
                adv = float(self._LIQUIDITY_TIERS["mid"])
        except Exception:
            adv = float(self._LIQUIDITY_TIERS["mid"])
        self._adv_cache[ticker] = adv
        return adv

    def half_spread(self, ticker: str) -> float:
        """Return estimated half-spread in bps for *ticker*."""
        adv = self.estimate_adv(ticker)
        tier = self.classify_liquidity(adv)
        return self._HALF_SPREAD_BPS[tier] * self.spread_multiplier

    def impact_cost_bps(self, ticker: str, order_size: int) -> float:
        """Market-impact cost using square-root model.

        impact = sqrt(order_size / ADV) * spread_multiplier * base_spread
        Returns cost in basis points.
        """
        adv = self.estimate_adv(ticker)
        if adv <= 0:
            return 0.0
        participation = order_size / adv
        tier = self.classify_liquidity(adv)
        base_spread = self._HALF_SPREAD_BPS[tier]
        impact = np.sqrt(participation) * base_spread * self.spread_multiplier
        return float(impact)

    def time_of_day_multiplier(self) -> float:
        """Return slippage multiplier based on current hour (ET approximation)."""
        try:
            from datetime import timezone, timedelta
            utc_now = datetime.now(timezone.utc)
            et_hour = (utc_now - timedelta(hours=5)).hour  # rough EST
        except Exception:
            et_hour = 12
        return self._TOD_MULTIPLIER.get(et_hour, 1.0)

    def fill_probability(self, ticker: str, order_size: int,
                         order_type: OrderType = OrderType.MARKET) -> float:
        """Estimate probability that the order fills.

        Market orders always fill (1.0).  Limit orders get a heuristic
        probability based on size relative to ADV.
        """
        if order_type == OrderType.MARKET:
            return 1.0
        adv = self.estimate_adv(ticker)
        if adv <= 0:
            return 0.5
        participation = order_size / adv
        # Larger participation → lower limit fill probability
        prob = max(0.1, 1.0 - np.sqrt(participation) * 2.0)
        return float(min(prob, 0.99))

    def compute_fill_price(self, mid_price: float, ticker: str,
                           order_size: int, side: OrderSide) -> float:
        """Return a realistic simulated fill price.

        Components:
            1. Half-spread cost (always paid)
            2. Market-impact cost (sqrt model)
            3. Time-of-day multiplier
        """
        spread_bps = self.half_spread(ticker)
        impact_bps = self.impact_cost_bps(ticker, order_size)
        tod = self.time_of_day_multiplier()

        total_bps = (spread_bps + impact_bps) * tod
        cost_frac = total_bps / 10_000.0

        if side in (OrderSide.BUY, OrderSide.COVER):
            return mid_price * (1.0 + cost_frac)
        else:
            return mid_price * (1.0 - cost_frac)


# ---------------------------------------------------------------------------
# RiskLimiter — Pre-trade risk checks
# ---------------------------------------------------------------------------
class RiskLimiter:
    """Enforces portfolio-level risk limits before order execution.

    All thresholds are expressed as fractions of NAV unless noted.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        max_single_name_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.03,
        max_gross_exposure: float = 2.5,
        max_net_exposure: float = 1.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_single_name_pct = max_single_name_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure

    def check_position_size(self, order_value: float, nav: float) -> tuple[bool, str]:
        """Check if order value exceeds max position size as % of NAV."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        pct = abs(order_value) / nav
        if pct > self.max_position_pct:
            return False, (f"Position size {pct:.1%} exceeds max "
                           f"{self.max_position_pct:.1%} of NAV")
        return True, ""

    def check_single_name_exposure(self, ticker: str, new_value: float,
                                   positions: dict, nav: float) -> tuple[bool, str]:
        """Check single-name concentration after proposed trade."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        existing = 0.0
        if ticker in positions:
            existing = abs(positions[ticker].market_value)
        total = existing + abs(new_value)
        pct = total / nav
        if pct > self.max_single_name_pct:
            return False, (f"Single-name exposure for {ticker} would be "
                           f"{pct:.1%}, exceeds {self.max_single_name_pct:.1%}")
        return True, ""

    def check_sector_concentration(self, sector: str, new_value: float,
                                   positions: dict, nav: float) -> tuple[bool, str]:
        """Check sector concentration after proposed trade."""
        if nav <= 0 or not sector:
            return True, ""  # Skip if sector unknown
        sector_total = sum(
            abs(p.market_value) for p in positions.values()
            if p.sector == sector
        )
        sector_total += abs(new_value)
        pct = sector_total / nav
        if pct > self.max_sector_pct:
            return False, (f"Sector {sector} concentration {pct:.1%} exceeds "
                           f"max {self.max_sector_pct:.1%}")
        return True, ""

    def check_daily_loss(self, daily_pnl: float, nav: float) -> tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        loss_pct = -daily_pnl / nav if daily_pnl < 0 else 0.0
        if loss_pct > self.daily_loss_limit_pct:
            return False, (f"Daily loss {loss_pct:.1%} exceeds limit "
                           f"{self.daily_loss_limit_pct:.1%}")
        return True, ""

    def check_gross_exposure(self, gross_exposure: float) -> tuple[bool, str]:
        """Check gross exposure limit."""
        if gross_exposure > self.max_gross_exposure:
            return False, (f"Gross exposure {gross_exposure:.2f}x exceeds "
                           f"max {self.max_gross_exposure:.2f}x")
        return True, ""

    def check_net_exposure(self, net_exposure: float) -> tuple[bool, str]:
        """Check net exposure limit."""
        if abs(net_exposure) > self.max_net_exposure:
            return False, (f"Net exposure {net_exposure:.2f}x exceeds "
                           f"max {self.max_net_exposure:.2f}x")
        return True, ""

    def run_all_checks(self, order_value: float, ticker: str, sector: str,
                       positions: dict, nav: float, daily_pnl: float,
                       gross_exposure: float, net_exposure: float
                       ) -> tuple[bool, list[str]]:
        """Run every pre-trade risk check. Return (passed, list_of_reasons)."""
        failures: list[str] = []

        ok, msg = self.check_position_size(order_value, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_single_name_exposure(ticker, order_value,
                                                  positions, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_sector_concentration(sector, order_value,
                                                  positions, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_daily_loss(daily_pnl, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_gross_exposure(gross_exposure)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_net_exposure(net_exposure)
        if not ok:
            failures.append(msg)

        return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# PerformanceTracker — Broker-level performance analytics
# ---------------------------------------------------------------------------
class PerformanceTracker:
    """Tracks and computes performance metrics for a paper broker session.

    Maintains a daily P&L series and derives drawdown, Sharpe ratio, win-rate
    by signal type, average win/loss ratio, and trade-frequency statistics.
    """

    def __init__(self, initial_nav: float = 1_000_000.0, risk_free_rate: float = 0.05):
        self.initial_nav = initial_nav
        self.risk_free_rate = risk_free_rate  # annualised
        self._daily_navs: list[tuple[str, float]] = []  # (date_str, nav)
        self._daily_pnls: list[tuple[str, float]] = []  # (date_str, pnl)
        self._trade_records: list[dict] = []
        self._high_water_mark: float = initial_nav

    # --- Recording -----------------------------------------------------------

    def record_nav(self, nav: float, date_str: Optional[str] = None):
        """Record an end-of-day NAV snapshot."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        self._daily_navs.append((date_str, nav))
        # Compute daily P&L from previous NAV
        if len(self._daily_navs) >= 2:
            prev_nav = self._daily_navs[-2][1]
            pnl = nav - prev_nav
        else:
            pnl = nav - self.initial_nav
        self._daily_pnls.append((date_str, pnl))
        # Update high water mark
        if nav > self._high_water_mark:
            self._high_water_mark = nav

    def record_trade(self, order_dict: dict):
        """Record a completed trade for signal-type analysis."""
        self._trade_records.append(order_dict)

    # --- Daily P&L -----------------------------------------------------------

    def get_daily_pnl_series(self) -> pd.DataFrame:
        """Return a DataFrame with columns ['date', 'pnl']."""
        if not self._daily_pnls:
            return pd.DataFrame(columns=["date", "pnl"])
        df = pd.DataFrame(self._daily_pnls, columns=["date", "pnl"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_daily_returns(self) -> np.ndarray:
        """Return array of daily simple returns from NAV series."""
        if len(self._daily_navs) < 2:
            return np.array([])
        navs = np.array([n for _, n in self._daily_navs], dtype=np.float64)
        returns = np.diff(navs) / navs[:-1]
        return returns

    # --- Drawdown ------------------------------------------------------------

    def get_drawdown(self) -> dict:
        """Return current drawdown and max drawdown as fractions."""
        if not self._daily_navs:
            return {"current_drawdown": 0.0, "max_drawdown": 0.0,
                    "high_water_mark": self.initial_nav}

        navs = np.array([n for _, n in self._daily_navs], dtype=np.float64)
        running_max = np.maximum.accumulate(navs)
        drawdowns = (running_max - navs) / np.where(running_max > 0, running_max, 1.0)

        current_dd = float(drawdowns[-1])
        max_dd = float(np.max(drawdowns))

        return {
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "high_water_mark": float(running_max[-1]),
        }

    # --- Sharpe ratio --------------------------------------------------------

    def sharpe_ratio(self, rolling_window: Optional[int] = None) -> float:
        """Compute annualised Sharpe ratio.

        If *rolling_window* is given, compute over last N days only.
        Uses 252 trading days for annualisation.
        """
        returns = self.get_daily_returns()
        if len(returns) < 2:
            return 0.0
        if rolling_window and rolling_window < len(returns):
            returns = returns[-rolling_window:]

        daily_rf = self.risk_free_rate / 252.0
        excess = returns - daily_rf
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess == 0:
            return 0.0
        return float(mean_excess / std_excess * np.sqrt(252))

    def rolling_sharpe(self, window: int = 60) -> list[tuple[str, float]]:
        """Return list of (date, sharpe) for rolling window."""
        returns = self.get_daily_returns()
        dates = [d for d, _ in self._daily_navs]
        result: list[tuple[str, float]] = []
        daily_rf = self.risk_free_rate / 252.0
        for i in range(window, len(returns) + 1):
            chunk = returns[i - window:i]
            excess = chunk - daily_rf
            m = np.mean(excess)
            s = np.std(excess, ddof=1)
            sr = float(m / s * np.sqrt(252)) if s > 0 else 0.0
            # dates are 1-indexed relative to returns (returns[0] goes with dates[1])
            result.append((dates[i], sr))
        return result

    # --- Win rate by signal type ---------------------------------------------

    def win_rate_by_signal(self) -> dict[str, dict]:
        """Return win rate and trade count grouped by signal_type."""
        buckets: dict[str, dict] = {}
        for trade in self._trade_records:
            sig = trade.get("signal_type", "HOLD")
            if sig not in buckets:
                buckets[sig] = {"wins": 0, "losses": 0, "total": 0,
                                "total_pnl": 0.0}
            pnl = trade.get("realized_pnl", 0.0)
            buckets[sig]["total"] += 1
            buckets[sig]["total_pnl"] += pnl
            if pnl > 0:
                buckets[sig]["wins"] += 1
            elif pnl < 0:
                buckets[sig]["losses"] += 1

        for sig, b in buckets.items():
            b["win_rate"] = b["wins"] / b["total"] if b["total"] > 0 else 0.0
        return buckets

    # --- Average win / loss ratio --------------------------------------------

    def avg_win_loss_ratio(self) -> float:
        """Return average winning trade P&L / average losing trade P&L (abs).

        Returns 0.0 if no losing trades.
        """
        wins = [t["realized_pnl"] for t in self._trade_records
                if t.get("realized_pnl", 0) > 0]
        losses = [t["realized_pnl"] for t in self._trade_records
                  if t.get("realized_pnl", 0) < 0]
        if not losses:
            return float("inf") if wins else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses))
        if avg_loss == 0:
            return 0.0
        return float(avg_win / avg_loss)

    # --- Trade frequency -----------------------------------------------------

    def trade_frequency(self) -> dict:
        """Return trade-frequency statistics."""
        if not self._trade_records:
            return {"total_trades": 0, "trades_per_day": 0.0, "days_traded": 0}
        dates = set()
        for t in self._trade_records:
            ts = t.get("fill_timestamp", t.get("timestamp", ""))
            if ts:
                dates.add(ts[:10])
        n_days = max(len(dates), 1)
        total = len(self._trade_records)
        return {
            "total_trades": total,
            "trades_per_day": total / n_days,
            "days_traded": n_days,
        }

    # --- Aggregate metrics ---------------------------------------------------

    def get_all_metrics(self) -> dict:
        """Return a dictionary with all performance metrics."""
        dd = self.get_drawdown()
        wl = self.avg_win_loss_ratio()
        freq = self.trade_frequency()
        return {
            "sharpe_total": self.sharpe_ratio(),
            "sharpe_60d": self.sharpe_ratio(rolling_window=60),
            "current_drawdown": dd["current_drawdown"],
            "max_drawdown": dd["max_drawdown"],
            "high_water_mark": dd["high_water_mark"],
            "avg_win_loss_ratio": wl,
            "win_rate_by_signal": self.win_rate_by_signal(),
            "trade_frequency": freq,
        }


# ---------------------------------------------------------------------------
# Risk Profile — Daily Target Dial-Down
# ---------------------------------------------------------------------------
class RiskProfile(str, Enum):
    """Risk profile tiers for daily target management."""
    AGGRESSIVE = "AGGRESSIVE"    # Pre-target: full leverage, max scanning
    MODERATE = "MODERATE"        # Target hit (5%): reduce leverage 50%
    DEFENSIVE = "DEFENSIVE"      # Target + buffer (6%+): protect gains


class DailyTargetManager:
    """Manages 5% daily compound return target with risk dial-down.

    Once the daily target is hit, the system shifts from alpha-seeking
    to capital-preservation mode. Execution multipliers, position sizing,
    and stop widths all adjust based on progress toward target.

    Target: 5% daily compound (minimum)
    Buffer: 6%+ triggers full defensive mode
    """

    DAILY_TARGET_PCT = 0.05      # 5% daily compound return
    BUFFER_TARGET_PCT = 0.06     # 6% triggers full defensive
    FLOOR_TARGET_PCT = 0.03      # Below 3% = max aggressive

    # Leverage multipliers by risk profile
    _LEVERAGE_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,   # Full leverage
        RiskProfile.MODERATE: 0.50,    # Half leverage
        RiskProfile.DEFENSIVE: 0.20,   # Minimal leverage
    }

    # Position size multipliers
    _POSITION_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,
        RiskProfile.MODERATE: 0.60,
        RiskProfile.DEFENSIVE: 0.25,
    }

    # Stop-loss width multipliers (wider = more protective)
    _STOP_WIDTH_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,
        RiskProfile.MODERATE: 1.5,
        RiskProfile.DEFENSIVE: 2.5,
    }

    def __init__(self, initial_nav: float = 1_000.0):
        self._initial_nav_today = initial_nav
        self._current_profile = RiskProfile.AGGRESSIVE
        self._target_hit_time: Optional[str] = None
        self._profile_history: list[tuple[str, str]] = []

    def reset_day(self, current_nav: float):
        """Reset for new trading day."""
        self._initial_nav_today = current_nav
        self._current_profile = RiskProfile.AGGRESSIVE
        self._target_hit_time = None

    @property
    def profile(self) -> RiskProfile:
        return self._current_profile

    @property
    def daily_return_pct(self) -> float:
        """Current day return as decimal (0.05 = 5%)."""
        if self._initial_nav_today <= 0:
            return 0.0
        return 0.0  # Will be set by update()

    def update(self, current_nav: float) -> RiskProfile:
        """Update risk profile based on current NAV vs start-of-day NAV.

        Returns the current risk profile after update.
        """
        if self._initial_nav_today <= 0:
            return self._current_profile

        daily_return = (current_nav - self._initial_nav_today) / self._initial_nav_today
        now_str = datetime.now().isoformat()

        old_profile = self._current_profile

        if daily_return >= self.BUFFER_TARGET_PCT:
            self._current_profile = RiskProfile.DEFENSIVE
        elif daily_return >= self.DAILY_TARGET_PCT:
            self._current_profile = RiskProfile.MODERATE
            if self._target_hit_time is None:
                self._target_hit_time = now_str
        else:
            self._current_profile = RiskProfile.AGGRESSIVE

        if old_profile != self._current_profile:
            self._profile_history.append((now_str, self._current_profile.value))

        return self._current_profile

    def get_leverage_multiplier(self) -> float:
        """Return leverage multiplier for current risk profile."""
        return self._LEVERAGE_MULT[self._current_profile]

    def get_position_multiplier(self) -> float:
        """Return position size multiplier for current risk profile."""
        return self._POSITION_MULT[self._current_profile]

    def get_stop_width_multiplier(self) -> float:
        """Return stop-loss width multiplier for current risk profile."""
        return self._STOP_WIDTH_MULT[self._current_profile]

    def allow_new_positions(self) -> bool:
        """Whether new positions are allowed under current profile."""
        return self._current_profile != RiskProfile.DEFENSIVE

    def get_state(self) -> dict:
        """Return full state for dashboard emission."""
        return {
            "risk_profile": self._current_profile.value,
            "initial_nav_today": self._initial_nav_today,
            "target_pct": self.DAILY_TARGET_PCT,
            "target_hit_time": self._target_hit_time,
            "leverage_mult": self.get_leverage_multiplier(),
            "position_mult": self.get_position_multiplier(),
            "stop_width_mult": self.get_stop_width_multiplier(),
            "allow_new_positions": self.allow_new_positions(),
            "profile_history": self._profile_history,
        }


# ---------------------------------------------------------------------------
# Live Dashboard State Emitter
# ---------------------------------------------------------------------------
class LiveDashboardState:
    """Collects and emits state for the live observation dashboard.

    When connected to internet and OpenBB API is running, this state
    can be consumed by engine/monitoring/live_dashboard.py for real-time
    terminal visualization or by the FastAPI backend for web dashboard.
    """

    def __init__(self):
        self._last_snapshot: dict = {}
        self._snapshot_history: list[dict] = []
        self._callbacks: list = []

    def register_callback(self, callback):
        """Register a callback for live state updates."""
        self._callbacks.append(callback)

    def emit(self, broker_state: dict, target_state: dict,
             pipeline_state: Optional[dict] = None):
        """Emit a dashboard snapshot.

        Args:
            broker_state: Portfolio summary from PaperBroker
            target_state: Daily target state from DailyTargetManager
            pipeline_state: Optional signal pipeline state
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": broker_state,
            "daily_target": target_state,
            "pipeline": pipeline_state or {},
        }
        self._last_snapshot = snapshot
        self._snapshot_history.append(snapshot)

        # Keep only last 1000 snapshots in memory
        if len(self._snapshot_history) > 1000:
            self._snapshot_history = self._snapshot_history[-500:]

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(snapshot)
            except Exception:
                pass

    def get_latest(self) -> dict:
        return self._last_snapshot

    def get_history(self, n: int = 100) -> list[dict]:
        return self._snapshot_history[-n:]


# ---------------------------------------------------------------------------
# Paper Broker
# ---------------------------------------------------------------------------
class PaperBroker:
    """Live HFT opportunity-based paper portfolio execution engine.

    Continuously active throughout the trading day. Manages a 5% daily
    compound return target with automatic risk dial-down after target hit.
    Observable via live dashboard when connected to internet.

    Drop-in replacement for live broker — same interface.
    """

    def __init__(
        self,
        initial_cash: float = 1_000.0,
        log_dir: Optional[Path] = None,
        slippage_bps: float = 2.0,
        enable_risk_limits: bool = True,
        enable_micro_price: bool = True,
        daily_target_pct: float = 0.05,
    ):
        self.state = PortfolioState(cash=initial_cash, nav=initial_cash)
        self.slippage_bps = slippage_bps
        self.log_dir = log_dir or Path("logs/paper_broker")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._orders: list[Order] = []
        self._trade_log: list[dict] = []

        # Enhanced components
        self._initial_cash = initial_cash
        self._micro_price = MicroPriceModel() if enable_micro_price else None
        self._risk_limiter = RiskLimiter() if enable_risk_limits else None
        self._perf_tracker = PerformanceTracker(initial_nav=initial_cash)
        self._daily_pnl_today: float = 0.0
        self._last_eod_date: Optional[str] = None
        self._eod_nav_history: list[tuple[str, float]] = []

        # Daily target manager — 5% compound daily minimum
        self._target_manager = DailyTargetManager(initial_nav=initial_cash)
        self._target_manager.DAILY_TARGET_PCT = daily_target_pct

        # Live dashboard state emitter
        self._dashboard = LiveDashboardState()

        # State persistence
        self._state_file = self.log_dir / "broker_state.json"
        self._load_state()

    # --- State persistence ---------------------------------------------------

    def _save_state(self):
        """Persist portfolio state to disk with atomic write (crash-safe)."""
        try:
            import math
            state = {
                "cash": self.state.cash,
                "nav": self.state.nav,
                "total_pnl": self.state.total_pnl,
                "win_count": self.state.win_count,
                "loss_count": self.state.loss_count,
                "positions": {
                    ticker: {
                        "ticker": p.ticker,
                        "quantity": p.quantity,
                        "avg_cost": p.avg_cost,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "realized_pnl": p.realized_pnl,
                        "sector": p.sector,
                    }
                    for ticker, p in self.state.positions.items()
                    if p.quantity != 0  # Don't persist closed positions
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "eod_nav_history": self._eod_nav_history[-252:],  # Keep 1 year
            }
            # Atomic write: write to .tmp, then rename (POSIX atomic)
            tmp_file = self._state_file.with_suffix('.tmp')
            tmp_file.write_text(json.dumps(state, indent=2, default=str))
            tmp_file.rename(self._state_file)
        except Exception as e:
            logger.warning("Failed to save broker state: %s", e)

    def _load_state(self):
        """Resume from last known state on startup with validation."""
        import math
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text())

            # Validate scalars — reject corrupt state
            cash = float(state.get("cash", self._initial_cash))
            if cash < 0 or math.isnan(cash) or math.isinf(cash):
                logger.error("Corrupt state: cash=%s — resetting to initial", cash)
                return

            nav = float(state.get("nav", cash))
            if nav < 0 or math.isnan(nav) or math.isinf(nav):
                logger.error("Corrupt state: nav=%s — resetting to initial", nav)
                return

            self.state.cash = cash
            self.state.nav = nav
            self.state.total_pnl = float(state.get("total_pnl", 0.0))
            self.state.win_count = int(state.get("win_count", 0))
            self.state.loss_count = int(state.get("loss_count", 0))
            self._eod_nav_history = state.get("eod_nav_history", [])

            # Validate and reconstruct positions
            for ticker, pdata in state.get("positions", {}).items():
                try:
                    qty = int(pdata.get("quantity", 0))
                    if qty <= 0:
                        continue  # Skip invalid positions
                    avg_cost = float(pdata.get("avg_cost", 0))
                    if avg_cost <= 0 or math.isnan(avg_cost):
                        continue
                    current_price = float(pdata.get("current_price", avg_cost))
                    if current_price <= 0 or math.isnan(current_price):
                        current_price = avg_cost

                    self.state.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=qty,
                        avg_cost=avg_cost,
                        current_price=current_price,
                        unrealized_pnl=float(pdata.get("unrealized_pnl", 0.0)),
                        realized_pnl=float(pdata.get("realized_pnl", 0.0)),
                        sector=pdata.get("sector", ""),
                    )
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning("Skipping corrupt position %s: %s", ticker, e)
                    continue

            saved_at = state.get("timestamp", "unknown")
            logger.info("Broker state restored from %s (NAV=$%.2f, %d positions)",
                        saved_at, self.state.nav, len(self.state.positions))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("Corrupt broker state file — starting fresh: %s", e)

    # --- Pre-trade risk checks -----------------------------------------------

    def _check_risk_limits(self, ticker: str, side: OrderSide,
                           quantity: int, price: float) -> tuple[bool, str]:
        """Run all pre-trade risk checks. Returns (passed, reason)."""
        if self._risk_limiter is None:
            return True, ""

        order_value = quantity * price
        nav = self.state.nav if self.state.nav > 0 else self._initial_cash

        # Determine sector for the position
        sector = ""
        if ticker in self.state.positions:
            sector = self.state.positions[ticker].sector

        # Get current exposures
        exposures = self.compute_exposures()
        gross = exposures.get("gross", 0.0)
        net = exposures.get("net", 0.0)

        passed, failures = self._risk_limiter.run_all_checks(
            order_value=order_value,
            ticker=ticker,
            sector=sector,
            positions=self.state.positions,
            nav=nav,
            daily_pnl=self._daily_pnl_today,
            gross_exposure=gross,
            net_exposure=net,
        )

        if not passed:
            return False, "; ".join(failures)
        return True, ""

    # --- Order execution -----------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """Place and immediately fill a market order using latest price."""
        order = Order(
            id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=quantity,
            limit_price=limit_price,
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Get current price
        price = self._get_current_price(ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"No price data for {ticker}"
            self._orders.append(order)
            return order

        # Update daily target manager with current NAV
        nav = self.compute_nav()
        self._target_manager.update(nav)

        # Check if new positions allowed under current risk profile
        if side in (OrderSide.BUY, OrderSide.SHORT):
            if not self._target_manager.allow_new_positions():
                order.status = OrderStatus.REJECTED
                order.reason = (f"DEFENSIVE mode — daily target exceeded "
                                f"({self._target_manager.profile.value}), "
                                f"no new positions")
                self._orders.append(order)
                return order

            # Apply position size multiplier from risk profile
            pos_mult = self._target_manager.get_position_multiplier()
            if pos_mult < 1.0:
                quantity = max(1, int(quantity * pos_mult))
                order.quantity = quantity

        # Pre-trade risk checks
        risk_ok, risk_reason = self._check_risk_limits(ticker, side, quantity, price)
        if not risk_ok:
            order.status = OrderStatus.REJECTED
            order.reason = f"Risk limit: {risk_reason}"
            self._orders.append(order)
            return order

        # Apply slippage — use MicroPriceModel if available, else simple bps
        if self._micro_price is not None:
            fill_price = self._micro_price.compute_fill_price(
                mid_price=price, ticker=ticker,
                order_size=quantity, side=side,
            )
        else:
            slip = price * self.slippage_bps / 10_000
            if side in (OrderSide.BUY, OrderSide.COVER):
                fill_price = price + slip
            else:
                fill_price = price - slip

        # Limit check
        if limit_price:
            if side in (OrderSide.BUY, OrderSide.COVER) and fill_price > limit_price:
                order.status = OrderStatus.CANCELLED
                order.reason = f"Limit {limit_price} < fill {fill_price}"
                self._orders.append(order)
                return order
            if side in (OrderSide.SELL, OrderSide.SHORT) and fill_price < limit_price:
                order.status = OrderStatus.CANCELLED
                order.reason = f"Limit {limit_price} > fill {fill_price}"
                self._orders.append(order)
                return order

        # Execute
        order.fill_price = fill_price
        order.fill_timestamp = datetime.now().isoformat()
        order.status = OrderStatus.FILLED

        self._update_position(order)
        self._orders.append(order)
        self._log_trade(order)

        return order

    def _update_position(self, order: Order):
        """Update positions and cash after fill."""
        ticker = order.ticker
        qty = order.quantity
        price = order.fill_price

        # Import GICS sector map for position classification
        try:
            from ..data.cross_asset_universe import SECTOR_MAP
            sector = SECTOR_MAP.get(ticker, "")
        except ImportError:
            sector = ""

        pos = self.state.positions.get(ticker, Position(ticker=ticker, sector=sector))

        realized_pnl_this_trade = 0.0

        if order.side == OrderSide.BUY:
            cost = qty * price
            if cost > self.state.cash:
                qty = int(self.state.cash / price)
                cost = qty * price
            if qty <= 0:
                return
            # Update avg cost
            total_qty = pos.quantity + qty
            if total_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            self.state.cash -= cost

        elif order.side == OrderSide.SELL:
            sell_qty = min(qty, pos.quantity)
            if sell_qty <= 0:
                return
            proceeds = sell_qty * price
            pnl = (price - pos.avg_cost) * sell_qty
            realized_pnl_this_trade = pnl
            pos.realized_pnl += pnl
            pos.quantity -= sell_qty
            self.state.cash += proceeds
            self.state.total_pnl += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        elif order.side == OrderSide.SHORT:
            pos.quantity -= qty
            pos.avg_cost = price
            self.state.cash += qty * price

        elif order.side == OrderSide.COVER:
            cover_qty = min(qty, abs(pos.quantity))
            cost = cover_qty * price
            pnl = (pos.avg_cost - price) * cover_qty
            realized_pnl_this_trade = pnl
            pos.quantity += cover_qty
            pos.realized_pnl += pnl
            self.state.cash -= cost
            self.state.total_pnl += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        pos.current_price = price
        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        # Attach stop/take_profit from quant strategy analysis
        if order.stop_loss is not None and order.stop_loss > 0:
            pos.stop_loss = order.stop_loss
        if order.take_profit is not None and order.take_profit > 0:
            pos.take_profit = order.take_profit

        if pos.quantity != 0:
            self.state.positions[ticker] = pos
        elif ticker in self.state.positions:
            del self.state.positions[ticker]

        self.state.total_trades += 1

        # Track daily P&L
        self._daily_pnl_today += realized_pnl_this_trade

        # Persist state after every trade
        self._save_state()

    # --- Portfolio state -----------------------------------------------------

    def refresh_prices(self):
        """Update all position prices from OpenBB."""
        tickers = list(self.state.positions.keys())
        if not tickers:
            return
        for ticker in tickers:
            price = self._get_current_price(ticker)
            if price > 0 and ticker in self.state.positions:
                pos = self.state.positions[ticker]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def compute_nav(self) -> float:
        """Compute current NAV."""
        self.refresh_prices()
        positions_value = sum(
            p.quantity * p.current_price
            for p in self.state.positions.values()
        )
        self.state.nav = self.state.cash + positions_value
        return self.state.nav

    def compute_exposures(self) -> dict:
        """Compute gross and net exposure."""
        nav = self.compute_nav()
        if nav <= 0:
            return {"gross": 0, "net": 0, "long": 0, "short": 0}
        long_val = sum(p.market_value for p in self.state.positions.values() if p.quantity > 0)
        short_val = sum(abs(p.market_value) for p in self.state.positions.values() if p.quantity < 0)
        self.state.gross_exposure = (long_val + short_val) / nav
        self.state.net_exposure = (long_val - short_val) / nav
        return {
            "gross": self.state.gross_exposure,
            "net": self.state.net_exposure,
            "long": long_val / nav,
            "short": short_val / nav,
        }

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self.state.positions)

    def get_portfolio_summary(self) -> dict:
        """Full portfolio summary."""
        nav = self.compute_nav()
        exp = self.compute_exposures()
        win_rate = (
            self.state.win_count / (self.state.win_count + self.state.loss_count)
            if (self.state.win_count + self.state.loss_count) > 0 else 0
        )
        return {
            "nav": nav,
            "cash": self.state.cash,
            "total_pnl": self.state.total_pnl,
            "total_trades": self.state.total_trades,
            "win_rate": win_rate,
            "positions": len(self.state.positions),
            "gross_exposure": exp["gross"],
            "net_exposure": exp["net"],
        }

    # --- Daily P&L -----------------------------------------------------------

    def get_daily_pnl(self) -> pd.DataFrame:
        """Return daily P&L as a DataFrame with columns ['date', 'pnl']."""
        return self._perf_tracker.get_daily_pnl_series()

    # --- Drawdown ------------------------------------------------------------

    def get_drawdown(self) -> dict:
        """Return current and max drawdown as fractions of peak NAV.

        Returns dict with keys: current_drawdown, max_drawdown, high_water_mark.
        """
        return self._perf_tracker.get_drawdown()

    # --- Performance metrics -------------------------------------------------

    def get_performance_metrics(self) -> dict:
        """Return dict with Sharpe, win rate, drawdown, etc."""
        metrics = self._perf_tracker.get_all_metrics()
        # Overlay broker-level win rate
        total = self.state.win_count + self.state.loss_count
        metrics["overall_win_rate"] = (
            self.state.win_count / total if total > 0 else 0.0
        )
        metrics["total_pnl"] = self.state.total_pnl
        metrics["nav"] = self.state.nav
        metrics["cash"] = self.state.cash
        return metrics

    # --- EOD reconciliation --------------------------------------------------

    def reconcile(self) -> dict:
        """End-of-day reconciliation.

        - Refreshes all prices
        - Recomputes NAV and exposures
        - Records daily NAV snapshot for performance tracking
        - Resets daily P&L counter
        - Logs reconciliation to disk

        Returns a summary dict.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Refresh and compute
        nav = self.compute_nav()
        exposures = self.compute_exposures()

        # Record to performance tracker
        self._perf_tracker.record_nav(nav, date_str=today)

        # Build reconciliation summary
        summary = {
            "date": today,
            "nav": nav,
            "cash": self.state.cash,
            "positions_count": len(self.state.positions),
            "total_pnl": self.state.total_pnl,
            "daily_pnl": self._daily_pnl_today,
            "gross_exposure": exposures["gross"],
            "net_exposure": exposures["net"],
            "total_trades_today": sum(
                1 for o in self._orders
                if o.status == OrderStatus.FILLED
                and o.fill_timestamp.startswith(today)
            ),
        }

        # Store EOD nav history
        self._eod_nav_history.append((today, nav))

        # Persist reconciliation
        recon_file = self.log_dir / f"reconcile_{today}.json"
        try:
            with open(recon_file, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        # Reset daily P&L and daily target for next day
        self._daily_pnl_today = 0.0
        self._last_eod_date = today
        self._target_manager.reset_day(nav)

        # Emit final dashboard state for the day
        self.emit_dashboard_state(pipeline_state={"event": "EOD_RECONCILE"})

        return summary

    # --- Position export -----------------------------------------------------

    def export_positions_csv(self, filepath: Optional[str] = None) -> str:
        """Export current positions to CSV.

        If *filepath* is None, returns the CSV as a string.
        Otherwise writes to *filepath* and returns the path.
        """
        self.refresh_prices()
        headers = [
            "ticker", "quantity", "avg_cost", "current_price",
            "market_value", "unrealized_pnl", "realized_pnl", "sector",
        ]
        rows = []
        for ticker, pos in sorted(self.state.positions.items()):
            rows.append([
                ticker,
                pos.quantity,
                round(pos.avg_cost, 4),
                round(pos.current_price, 4),
                round(pos.market_value, 2),
                round(pos.unrealized_pnl, 2),
                round(pos.realized_pnl, 2),
                pos.sector,
            ])

        if filepath:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            return str(path)
        else:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(headers)
            writer.writerows(rows)
            return buf.getvalue()

    # --- Price fetching ------------------------------------------------------

    def _get_current_price(self, ticker: str) -> float:
        """Get latest price from OpenBB."""
        try:
            prices = get_adj_close(ticker, start=(
                pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            )
            if prices.empty:
                return 0.0
            return float(prices.iloc[-1].iloc[0] if isinstance(prices, pd.DataFrame) else prices.iloc[-1])
        except Exception:
            return 0.0

    # --- Logging -------------------------------------------------------------

    def _log_trade(self, order: Order):
        """Log a filled trade with extended metadata."""
        entry = order.to_dict()

        # Enhanced metadata
        entry["nav_at_fill"] = self.state.nav
        entry["cash_after"] = self.state.cash
        entry["positions_count"] = len(self.state.positions)
        entry["daily_pnl"] = self._daily_pnl_today
        entry["total_pnl"] = self.state.total_pnl
        entry["total_trades"] = self.state.total_trades

        # Position-level info if still held
        if order.ticker in self.state.positions:
            pos = self.state.positions[order.ticker]
            entry["pos_quantity"] = pos.quantity
            entry["pos_avg_cost"] = pos.avg_cost
            entry["pos_unrealized_pnl"] = pos.unrealized_pnl
            entry["pos_realized_pnl"] = pos.realized_pnl
            entry["pos_sector"] = pos.sector
        else:
            entry["pos_quantity"] = 0
            entry["pos_avg_cost"] = 0.0
            entry["pos_unrealized_pnl"] = 0.0
            entry["pos_realized_pnl"] = 0.0
            entry["pos_sector"] = ""

        # Micro-price model info
        if self._micro_price is not None:
            entry["spread_bps"] = self._micro_price.half_spread(order.ticker)
            entry["impact_bps"] = self._micro_price.impact_cost_bps(
                order.ticker, order.quantity)
            entry["tod_multiplier"] = self._micro_price.time_of_day_multiplier()

        # Record realized PnL into performance tracker for signal analysis
        realized = 0.0
        if order.side in (OrderSide.SELL, OrderSide.COVER):
            realized = entry.get("pos_realized_pnl", 0.0)
        perf_entry = dict(entry)
        perf_entry["realized_pnl"] = realized
        self._perf_tracker.record_trade(perf_entry)

        self._trade_log.append(entry)
        log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_log)

    # --- Daily target & risk profile -----------------------------------------

    def get_risk_profile(self) -> str:
        """Return current risk profile (AGGRESSIVE/MODERATE/DEFENSIVE)."""
        return self._target_manager.profile.value

    def get_daily_target_state(self) -> dict:
        """Return full daily target state for dashboard."""
        nav = self.compute_nav()
        self._target_manager.update(nav)
        state = self._target_manager.get_state()
        state["current_nav"] = nav
        if self._target_manager._initial_nav_today > 0:
            state["daily_return_pct"] = (
                (nav - self._target_manager._initial_nav_today)
                / self._target_manager._initial_nav_today
            )
        else:
            state["daily_return_pct"] = 0.0
        return state

    def reset_daily_target(self):
        """Reset daily target for new trading day (called at market open)."""
        nav = self.compute_nav()
        self._target_manager.reset_day(nav)

    def get_leverage_multiplier(self) -> float:
        """Get current leverage multiplier from risk profile."""
        return self._target_manager.get_leverage_multiplier()

    # --- Live dashboard ------------------------------------------------------

    def emit_dashboard_state(self, pipeline_state: Optional[dict] = None):
        """Emit current state to live dashboard."""
        self._dashboard.emit(
            broker_state=self.get_portfolio_summary(),
            target_state=self.get_daily_target_state(),
            pipeline_state=pipeline_state,
        )

    def get_dashboard_snapshot(self) -> dict:
        """Get latest dashboard snapshot."""
        return self._dashboard.get_latest()

    def get_dashboard_history(self, n: int = 100) -> list[dict]:
        """Get dashboard snapshot history."""
        return self._dashboard.get_history(n)

    def register_dashboard_callback(self, callback):
        """Register callback for live dashboard updates."""
        self._dashboard.register_callback(callback)
