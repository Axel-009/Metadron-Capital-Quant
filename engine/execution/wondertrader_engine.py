"""WonderTraderEngine -- CTA/HFT execution wrapper (pure Python).

Native Python implementation of WonderTrader core concepts:
    - CTA trend-following signal generation (dual MA, channel breakout, momentum)
    - HFT micro-price estimation from OHLCV data
    - Low-latency order routing simulation (TWAP, VWAP, smart split)
    - Execution quality scoring (slippage, timing, fill metrics)

Pipeline position:
    DecisionMatrix (Stage 5.5) -> HFT Technical (Stage 6.5) -> **WonderTrader (Stage 7)** -> PaperBroker
    This module sits between QuantStrategyExecutor and the final order submission,
    providing CTA overlay signals and micro-price-aware execution.

WonderTrader reference: C++ high-performance trading platform (wondertrader.github.io)
    - wtpy Python API wrapper concepts adapted here as pure-numpy implementations
    - CTA engine: CtaStrategy base with on_bar / on_tick callbacks
    - HFT engine: HftStrategy base with on_order_queue / on_transaction callbacks
    - Execution engine: order routing, slippage model, latency simulation

Dependencies: numpy only (pure-numpy, no external ML frameworks).
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("metadron.execution.wondertrader")

# ---------------------------------------------------------------------------
# wtpy — WonderTrader native Python API (optional, falls back to pure-numpy)
# ---------------------------------------------------------------------------
try:
    from wtpy import WtBtEngine, EngineType, WtDtServo  # noqa: F401
    from wtpy.apps import WtCtaOptimizer
    WTPY_AVAILABLE = True
except ImportError:
    WTPY_AVAILABLE = False

try:
    from wtpy.wrapper import WtDataHelper
    WTPY_DATA_HELPER_AVAILABLE = True
except ImportError:
    WTPY_DATA_HELPER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CTASignal:
    """Signal produced by the CTA trend-following engine."""
    ticker: str
    direction: int              # +1 long, -1 short, 0 flat
    strategy: str               # which CTA sub-strategy fired
    strength: float             # signal strength in [0, 1]
    timeframe: str              # bar period that produced the signal
    stop_loss: float            # suggested stop-loss price
    take_profit: float          # suggested take-profit price
    regime: str                 # detected market regime
    timestamp: float = 0.0     # epoch seconds

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MicroPriceResult:
    """Output of the HFT micro-price estimator."""
    mid_price: float            # simple mid from OHLCV
    micro_price: float          # order-flow-adjusted micro-price
    imbalance: float            # order flow imbalance in [-1, +1]
    urgency: float              # urgency score in [0, 1]
    tick_signal: int            # +1 buy pressure, -1 sell pressure, 0 neutral
    opportunity_score: float    # sub-second opportunity magnitude in [0, 1]
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ExecutionSlice:
    """Single slice of a routed order."""
    slice_id: int
    quantity: float
    target_price: float
    scheduled_time: float       # relative seconds from order start
    latency_ms: float           # simulated latency with jitter


@dataclass
class ExecutionPlan:
    """Full execution plan returned by the order router."""
    order_id: str
    strategy: str               # TWAP, VWAP, or SMART
    total_quantity: float
    slices: List[ExecutionSlice] = field(default_factory=list)
    benchmark_price: float = 0.0
    estimated_slippage_bps: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class ExecutionQuality:
    """Post-trade execution quality metrics."""
    slippage_bps: float         # realized slippage vs arrival price
    timing_score: float         # 0-1, how well timed vs VWAP
    fill_rate: float            # fraction of order filled
    avg_latency_ms: float       # average per-slice latency
    implementation_shortfall_bps: float  # IS vs decision price


# ---------------------------------------------------------------------------
# CTA Strategy Components
# ---------------------------------------------------------------------------

class _DualMovingAverage:
    """Dual moving average crossover strategy.

    Generates long signal when fast MA crosses above slow MA,
    short signal on the reverse crossover.  Signal strength is
    proportional to the normalised spread between the two averages.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def generate(self, closes: np.ndarray) -> Tuple[int, float]:
        """Return (direction, strength) from the latest bar."""
        if len(closes) < self.slow:
            return 0, 0.0
        fast_ma = np.mean(closes[-self.fast:])
        slow_ma = np.mean(closes[-self.slow:])
        spread = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
        if fast_ma > slow_ma:
            return 1, min(abs(spread) * 20.0, 1.0)
        elif fast_ma < slow_ma:
            return -1, min(abs(spread) * 20.0, 1.0)
        return 0, 0.0


class _ChannelBreakout:
    """Donchian channel breakout strategy.

    Long when price breaks above the N-bar high channel,
    short when price breaks below the N-bar low channel.
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate(self, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray) -> Tuple[int, float]:
        """Return (direction, strength) from the latest bar."""
        if len(closes) < self.lookback + 1:
            return 0, 0.0
        channel_high = np.max(highs[-(self.lookback + 1):-1])
        channel_low = np.min(lows[-(self.lookback + 1):-1])
        channel_range = channel_high - channel_low
        if channel_range <= 0:
            return 0, 0.0
        price = closes[-1]
        if price > channel_high:
            penetration = (price - channel_high) / channel_range
            return 1, min(penetration * 5.0, 1.0)
        elif price < channel_low:
            penetration = (channel_low - price) / channel_range
            return -1, min(penetration * 5.0, 1.0)
        return 0, 0.0


class _MomentumStrategy:
    """Rate-of-change momentum strategy.

    Measures N-bar rate of change and z-scores it against
    a rolling window of historical momentum readings.
    """

    def __init__(self, roc_period: int = 12, zscore_window: int = 60):
        self.roc_period = roc_period
        self.zscore_window = zscore_window

    def generate(self, closes: np.ndarray) -> Tuple[int, float]:
        """Return (direction, strength) from the latest bar."""
        needed = self.roc_period + self.zscore_window
        if len(closes) < needed:
            return 0, 0.0
        # compute rolling ROC series
        roc_series = (closes[self.roc_period:] - closes[:-self.roc_period]) / (
            closes[:-self.roc_period] + 1e-12
        )
        recent = roc_series[-self.zscore_window:]
        mu = np.mean(recent)
        sigma = np.std(recent)
        if sigma < 1e-12:
            return 0, 0.0
        z = (roc_series[-1] - mu) / sigma
        if z > 1.0:
            return 1, min(abs(z) / 3.0, 1.0)
        elif z < -1.0:
            return -1, min(abs(z) / 3.0, 1.0)
        return 0, 0.0


# ---------------------------------------------------------------------------
# Multi-Timeframe Resampler
# ---------------------------------------------------------------------------

def _resample_ohlcv(opens: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, closes: np.ndarray,
                    volumes: np.ndarray, factor: int) -> Tuple[
                        np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray]:
    """Resample 1-minute OHLCV bars into higher-period bars.

    Parameters
    ----------
    opens, highs, lows, closes, volumes : np.ndarray
        Base-period (1-minute) OHLCV arrays of equal length.
    factor : int
        Number of base bars per resampled bar (e.g. 5 for 5-minute bars).

    Returns
    -------
    Tuple of resampled (opens, highs, lows, closes, volumes).
    """
    n = len(closes)
    n_bars = n // factor
    if n_bars == 0:
        return opens, highs, lows, closes, volumes
    trimmed = n_bars * factor
    o = opens[:trimmed].reshape(n_bars, factor)
    h = highs[:trimmed].reshape(n_bars, factor)
    lo = lows[:trimmed].reshape(n_bars, factor)
    c = closes[:trimmed].reshape(n_bars, factor)
    v = volumes[:trimmed].reshape(n_bars, factor)
    return (
        o[:, 0],            # first open per bar
        np.max(h, axis=1),  # highest high
        np.min(lo, axis=1), # lowest low
        c[:, -1],           # last close
        np.sum(v, axis=1),  # summed volume
    )


# Timeframe label -> resample factor (assumes 1-min base bars)
_TIMEFRAME_FACTORS: Dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


# ---------------------------------------------------------------------------
# Regime Detector (lightweight)
# ---------------------------------------------------------------------------

def _detect_regime(closes: np.ndarray, window: int = 60) -> str:
    """Classify the recent price regime.

    Uses a combination of trend slope and realised volatility to
    categorise the market into one of four regimes that align with
    the MetadronCube regime taxonomy.

    Returns
    -------
    str : one of TRENDING, RANGE, STRESS, CRASH
    """
    if len(closes) < window:
        return "RANGE"
    recent = closes[-window:]
    returns = np.diff(recent) / (recent[:-1] + 1e-12)
    vol = np.std(returns) * np.sqrt(252 * 390)  # annualised from 1-min
    cum_return = (recent[-1] / recent[0]) - 1.0
    trend_strength = abs(cum_return) / (vol + 1e-12)

    if vol > 0.60:
        return "CRASH" if cum_return < -0.05 else "STRESS"
    if trend_strength > 0.4:
        return "TRENDING"
    return "RANGE"


# ---------------------------------------------------------------------------
# Dynamic Stop-Loss / Take-Profit
# ---------------------------------------------------------------------------

def _dynamic_stops(closes: np.ndarray, direction: int,
                   atr_period: int = 14,
                   sl_multiplier: float = 2.0,
                   tp_multiplier: float = 3.0) -> Tuple[float, float]:
    """Compute ATR-based stop-loss and take-profit levels.

    Parameters
    ----------
    closes : np.ndarray
        Close prices (minimum length = atr_period + 1).
    direction : int
        +1 for long, -1 for short.
    atr_period : int
        Lookback for average true range approximation.
    sl_multiplier, tp_multiplier : float
        Multiples of ATR for stop and target distances.

    Returns
    -------
    (stop_loss, take_profit) : Tuple[float, float]
    """
    if len(closes) < atr_period + 1:
        price = closes[-1]
        return price * (1 - 0.02 * direction), price * (1 + 0.03 * direction)
    # Approximate ATR from close-to-close ranges (no H/L needed)
    abs_changes = np.abs(np.diff(closes[-atr_period - 1:]))
    atr = np.mean(abs_changes)
    price = closes[-1]
    stop_loss = price - direction * sl_multiplier * atr
    take_profit = price + direction * tp_multiplier * atr
    return stop_loss, take_profit


# ---------------------------------------------------------------------------
# WonderTraderEngine
# ---------------------------------------------------------------------------

class WonderTraderEngine:
    """CTA/HFT execution engine inspired by WonderTrader.

    Provides three execution capabilities:
        1. CTA trend-following signal generation across multiple timeframes
        2. HFT micro-price estimation and urgency scoring from OHLCV
        3. Low-latency order routing simulation (TWAP / VWAP / SMART)

    All computations are pure-numpy with no external framework dependencies.

    Usage
    -----
    >>> engine = WonderTraderEngine()
    >>> signals = engine.generate_cta_signals(prices_df)
    >>> mp = engine.compute_micro_price(ohlcv_dict)
    >>> plan = engine.route_order({"ticker": "AAPL", "qty": 1000, "side": "BUY"})
    >>> quality = engine.get_execution_quality()
    """

    # CTA sub-strategy configuration per regime
    _REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
        "TRENDING": {"dual_ma": 0.35, "breakout": 0.40, "momentum": 0.25},
        "RANGE":    {"dual_ma": 0.25, "breakout": 0.20, "momentum": 0.55},
        "STRESS":   {"dual_ma": 0.40, "breakout": 0.30, "momentum": 0.30},
        "CRASH":    {"dual_ma": 0.50, "breakout": 0.35, "momentum": 0.15},
    }

    def __init__(
        self,
        fast_ma: int = 10,
        slow_ma: int = 30,
        channel_lookback: int = 20,
        roc_period: int = 12,
        base_latency_ms: float = 0.5,
        latency_jitter_ms: float = 0.3,
        default_slices: int = 10,
    ):
        """Initialise the WonderTrader engine.

        Parameters
        ----------
        fast_ma : int
            Fast moving average period for dual-MA strategy.
        slow_ma : int
            Slow moving average period for dual-MA strategy.
        channel_lookback : int
            Lookback period for Donchian channel breakout.
        roc_period : int
            Rate-of-change period for momentum strategy.
        base_latency_ms : float
            Simulated base network latency in milliseconds.
        latency_jitter_ms : float
            Random jitter added to base latency (uniform distribution).
        default_slices : int
            Default number of child-order slices for order routing.
        """
        # CTA strategies
        self._dual_ma = _DualMovingAverage(fast=fast_ma, slow=slow_ma)
        self._breakout = _ChannelBreakout(lookback=channel_lookback)
        self._momentum = _MomentumStrategy(roc_period=roc_period)

        # Latency simulation
        self._base_latency_ms = base_latency_ms
        self._latency_jitter_ms = latency_jitter_ms
        self._default_slices = default_slices

        # Execution history for quality tracking
        self._execution_log: List[Dict] = []

        if WTPY_AVAILABLE:
            logger.info("WonderTraderEngine: wtpy native bindings loaded (C++ backend)")
        else:
            logger.info("WonderTraderEngine: wtpy unavailable — running pure-numpy simulation")

        logger.info(
            "WonderTraderEngine initialised (MA=%d/%d, channel=%d, ROC=%d, "
            "latency=%.1fms +/- %.1fms)",
            fast_ma, slow_ma, channel_lookback, roc_period,
            base_latency_ms, latency_jitter_ms,
        )

    @property
    def has_native_wtpy(self) -> bool:
        """True if wtpy C++ bindings are available, False if running pure-numpy simulation."""
        return WTPY_AVAILABLE

    # ------------------------------------------------------------------
    # 1. CTA Signal Generation
    # ------------------------------------------------------------------

    def generate_cta_signals(
        self,
        prices_df,
        timeframes: Optional[List[str]] = None,
    ) -> List[CTASignal]:
        """Generate CTA trend-following signals from OHLCV data.

        Runs dual-MA, channel breakout, and momentum strategies across
        multiple timeframes, then applies regime-aware weighting to
        produce a consolidated signal per ticker.

        Parameters
        ----------
        prices_df : pandas.DataFrame or dict
            Must contain columns: ticker, open, high, low, close, volume.
            If a dict, keys are column names and values are arrays.
            All rows are assumed to be 1-minute bars sorted by time.
        timeframes : list of str, optional
            Timeframes to analyse.  Defaults to all five
            (1m, 5m, 15m, 1h, 4h).

        Returns
        -------
        list of CTASignal
            One signal per (ticker, timeframe) combination that fires.
        """
        if timeframes is None:
            timeframes = list(_TIMEFRAME_FACTORS.keys())

        # Normalise input to dict-of-arrays
        cols = self._normalise_prices(prices_df)
        tickers = np.unique(cols["ticker"])
        signals: List[CTASignal] = []

        for ticker in tickers:
            mask = cols["ticker"] == ticker
            o = cols["open"][mask].astype(float)
            h = cols["high"][mask].astype(float)
            lo = cols["low"][mask].astype(float)
            c = cols["close"][mask].astype(float)
            v = cols["volume"][mask].astype(float)

            regime = _detect_regime(c)
            weights = self._REGIME_WEIGHTS.get(regime, self._REGIME_WEIGHTS["RANGE"])

            for tf in timeframes:
                factor = _TIMEFRAME_FACTORS.get(tf, 1)
                ro, rh, rl, rc, rv = _resample_ohlcv(o, h, lo, c, v, factor)
                if len(rc) < 5:
                    continue

                # Run each sub-strategy
                d_ma, s_ma = self._dual_ma.generate(rc)
                d_bo, s_bo = self._breakout.generate(rh, rl, rc)
                d_mo, s_mo = self._momentum.generate(rc)

                # Weighted consensus
                w_score = (
                    weights["dual_ma"] * d_ma * s_ma
                    + weights["breakout"] * d_bo * s_bo
                    + weights["momentum"] * d_mo * s_mo
                )
                if abs(w_score) < 0.05:
                    continue  # below noise threshold

                direction = 1 if w_score > 0 else -1
                strength = min(abs(w_score), 1.0)

                # Pick dominant strategy name
                contributions = {
                    "dual_ma": abs(d_ma * s_ma * weights["dual_ma"]),
                    "breakout": abs(d_bo * s_bo * weights["breakout"]),
                    "momentum": abs(d_mo * s_mo * weights["momentum"]),
                }
                dominant = max(contributions, key=contributions.get)

                sl, tp = _dynamic_stops(rc, direction)

                signals.append(CTASignal(
                    ticker=str(ticker),
                    direction=direction,
                    strategy=dominant,
                    strength=round(strength, 4),
                    timeframe=tf,
                    stop_loss=round(sl, 4),
                    take_profit=round(tp, 4),
                    regime=regime,
                ))

        logger.info("CTA engine produced %d signals across %d tickers",
                     len(signals), len(tickers))
        return signals

    # ------------------------------------------------------------------
    # 2. HFT Micro-Price Engine
    # ------------------------------------------------------------------

    def compute_micro_price(self, ohlcv: Dict[str, np.ndarray]) -> MicroPriceResult:
        """Estimate micro-price and urgency from OHLCV data.

        Since we do not have Level-2 order book data, the micro-price is
        approximated from OHLCV bars:
            mid_price   = (high + low) / 2  of the last bar
            imbalance   = (close - open) / (high - low)  (proxy for order flow)
            micro_price = mid + imbalance * (high - low) / 2

        Urgency is derived from volume acceleration and price velocity.
        Tick-level signal is the sign of the imbalance weighted by
        recent momentum.

        Parameters
        ----------
        ohlcv : dict
            Keys: open, high, low, close, volume -- each a numpy array
            of equal length (1-minute bars).

        Returns
        -------
        MicroPriceResult
        """
        # Accept any case for column names (Open/open/OPEN)
        def _get(key):
            for k in (key, key.capitalize(), key.upper()):
                if hasattr(ohlcv, 'columns'):
                    if k in ohlcv.columns:
                        return np.asarray(ohlcv[k], dtype=float)
                elif isinstance(ohlcv, dict) and k in ohlcv:
                    return np.asarray(ohlcv[k], dtype=float)
            raise KeyError(f"Missing column: {key}")
        o = _get("open")
        h = _get("high")
        lo = _get("low")
        c = _get("close")
        v = _get("volume")

        if len(c) == 0:
            return MicroPriceResult(
                mid_price=0.0, micro_price=0.0, imbalance=0.0,
                urgency=0.0, tick_signal=0, opportunity_score=0.0,
            )

        # Mid-price from latest bar
        mid = (h[-1] + lo[-1]) / 2.0
        bar_range = h[-1] - lo[-1]

        # Order flow imbalance proxy: how close did close land to high vs low
        if bar_range > 1e-12:
            imbalance = (c[-1] - o[-1]) / bar_range  # in [-1, +1] approx
        else:
            imbalance = 0.0
        imbalance = float(np.clip(imbalance, -1.0, 1.0))

        # Micro-price: shift mid toward close based on imbalance
        micro = mid + imbalance * bar_range * 0.5

        # Urgency: volume acceleration + price velocity
        urgency = 0.0
        if len(v) >= 10:
            vol_recent = np.mean(v[-5:])
            vol_baseline = np.mean(v[-10:-5]) + 1e-12
            vol_accel = vol_recent / vol_baseline - 1.0
            price_vel = abs(c[-1] - c[-5]) / (c[-5] + 1e-12) if len(c) >= 5 else 0.0
            urgency = float(np.clip(
                0.5 * min(vol_accel, 3.0) / 3.0 + 0.5 * min(price_vel * 100, 1.0),
                0.0, 1.0,
            ))

        # Tick signal: sign of imbalance weighted by short-term momentum
        if len(c) >= 3:
            short_mom = (c[-1] - c[-3]) / (c[-3] + 1e-12)
            raw_tick = imbalance * 0.6 + np.sign(short_mom) * 0.4
            tick_signal = int(np.sign(raw_tick)) if abs(raw_tick) > 0.1 else 0
        else:
            tick_signal = int(np.sign(imbalance)) if abs(imbalance) > 0.3 else 0

        # Sub-second opportunity score: high urgency + strong imbalance + tight range
        opportunity = float(np.clip(
            urgency * 0.4 + abs(imbalance) * 0.4 + (1.0 - min(bar_range / (mid + 1e-12) * 100, 1.0)) * 0.2,
            0.0, 1.0,
        ))

        return MicroPriceResult(
            mid_price=round(mid, 6),
            micro_price=round(micro, 6),
            imbalance=round(imbalance, 6),
            urgency=round(urgency, 4),
            tick_signal=tick_signal,
            opportunity_score=round(opportunity, 4),
        )

    # ------------------------------------------------------------------
    # 3. Low-Latency Order Routing
    # ------------------------------------------------------------------

    def route_order(
        self,
        order: Dict,
        strategy: str = "TWAP",
        duration_seconds: float = 60.0,
        num_slices: Optional[int] = None,
    ) -> ExecutionPlan:
        """Create an execution plan for a parent order.

        Splits the order into child slices according to the chosen
        algorithm and simulates network latency with jitter.

        Parameters
        ----------
        order : dict
            Must contain: ticker (str), qty (float), side (str BUY/SELL).
            Optional: price (float) -- used as benchmark / limit.
        strategy : str
            Execution algorithm: TWAP, VWAP, or SMART.
        duration_seconds : float
            Total execution window for the parent order.
        num_slices : int, optional
            Number of child orders.  Defaults to ``self._default_slices``.

        Returns
        -------
        ExecutionPlan
        """
        strategy = strategy.upper()
        if strategy not in ("TWAP", "VWAP", "SMART"):
            logger.warning("Unknown routing strategy '%s', falling back to TWAP", strategy)
            strategy = "TWAP"

        n = num_slices or self._default_slices
        qty = float(order.get("qty", 0))
        benchmark = float(order.get("price", 0.0))
        ticker = order.get("ticker", "UNKNOWN")
        side = order.get("side", "BUY").upper()
        order_id = f"WT-{ticker}-{int(time.time() * 1000) % 1_000_000:06d}"

        if strategy == "TWAP":
            slices = self._route_twap(qty, benchmark, n, duration_seconds)
        elif strategy == "VWAP":
            slices = self._route_vwap(qty, benchmark, n, duration_seconds)
        else:  # SMART
            slices = self._route_smart(qty, benchmark, n, duration_seconds)

        # Estimate slippage (impact model: sqrt of participation rate)
        avg_slice = qty / n if n > 0 else qty
        participation = min(avg_slice / max(qty, 1.0), 1.0)
        est_slippage_bps = 0.5 * np.sqrt(participation) * 10  # simplified model

        plan = ExecutionPlan(
            order_id=order_id,
            strategy=strategy,
            total_quantity=qty,
            slices=slices,
            benchmark_price=benchmark,
            estimated_slippage_bps=round(est_slippage_bps, 2),
            duration_seconds=duration_seconds,
        )

        # Log for quality tracking
        self._execution_log.append({
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "strategy": strategy,
            "benchmark": benchmark,
            "slices": n,
            "est_slippage_bps": est_slippage_bps,
            "timestamp": time.time(),
        })

        logger.info(
            "Routed order %s: %s %s %.0f via %s (%d slices, est slip %.1f bps)",
            order_id, side, ticker, qty, strategy, n, est_slippage_bps,
        )
        return plan

    # ------------------------------------------------------------------
    # 4. Execution Quality
    # ------------------------------------------------------------------

    def get_execution_quality(self) -> ExecutionQuality:
        """Compute aggregate execution quality from the execution log.

        Returns
        -------
        ExecutionQuality
            Slippage, timing, fill rate, latency, and implementation
            shortfall metrics aggregated across all routed orders.
        """
        if not self._execution_log:
            return ExecutionQuality(
                slippage_bps=0.0,
                timing_score=1.0,
                fill_rate=1.0,
                avg_latency_ms=self._base_latency_ms,
                implementation_shortfall_bps=0.0,
            )

        slippages = [e["est_slippage_bps"] for e in self._execution_log]
        avg_slip = float(np.mean(slippages))

        # Timing score: how evenly spaced were our executions
        timestamps = [e["timestamp"] for e in self._execution_log]
        if len(timestamps) > 1:
            diffs = np.diff(sorted(timestamps))
            cv = np.std(diffs) / (np.mean(diffs) + 1e-12)
            timing = float(np.clip(1.0 - cv, 0.0, 1.0))
        else:
            timing = 1.0

        avg_latency = self._base_latency_ms + self._latency_jitter_ms * 0.5

        # Implementation shortfall: slippage + latency cost estimate
        latency_cost_bps = avg_latency * 0.01  # 0.01 bps per ms of latency
        is_bps = avg_slip + latency_cost_bps

        return ExecutionQuality(
            slippage_bps=round(avg_slip, 2),
            timing_score=round(timing, 4),
            fill_rate=1.0,  # simulated: always fill
            avg_latency_ms=round(avg_latency, 2),
            implementation_shortfall_bps=round(is_bps, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_twap(self, qty: float, benchmark: float,
                    n: int, duration: float) -> List[ExecutionSlice]:
        """Time-weighted average price: equal slices at equal intervals."""
        slice_qty = qty / n
        interval = duration / n
        slices = []
        for i in range(n):
            latency = self._sim_latency()
            slices.append(ExecutionSlice(
                slice_id=i,
                quantity=round(slice_qty, 4),
                target_price=benchmark,
                scheduled_time=round(i * interval, 4),
                latency_ms=round(latency, 3),
            ))
        return slices

    def _route_vwap(self, qty: float, benchmark: float,
                    n: int, duration: float) -> List[ExecutionSlice]:
        """Volume-weighted average price: U-shaped volume profile.

        Simulates higher participation near market open and close
        using a quadratic volume curve.
        """
        # Generate U-shaped weights (higher at start and end)
        t = np.linspace(-1, 1, n)
        raw_weights = 1.0 + 0.8 * t ** 2  # U-shape
        weights = raw_weights / raw_weights.sum()
        interval = duration / n

        slices = []
        for i in range(n):
            latency = self._sim_latency()
            slices.append(ExecutionSlice(
                slice_id=i,
                quantity=round(float(qty * weights[i]), 4),
                target_price=benchmark,
                scheduled_time=round(i * interval, 4),
                latency_ms=round(latency, 3),
            ))
        return slices

    def _route_smart(self, qty: float, benchmark: float,
                     n: int, duration: float) -> List[ExecutionSlice]:
        """Smart order routing: adaptive sizing with randomised timing.

        Adds timing jitter and skews slice sizes toward the beginning
        to capture early liquidity, then uses smaller slices to
        minimise information leakage.
        """
        # Exponentially decaying slice sizes
        decay = np.exp(-np.linspace(0, 2, n))
        weights = decay / decay.sum()
        # Randomise timing within each interval bucket
        interval = duration / n
        rng = np.random.default_rng(int(time.time()) % 2**31)
        jitter = rng.uniform(0, interval * 0.4, size=n)

        slices = []
        for i in range(n):
            latency = self._sim_latency()
            slices.append(ExecutionSlice(
                slice_id=i,
                quantity=round(float(qty * weights[i]), 4),
                target_price=benchmark,
                scheduled_time=round(i * interval + float(jitter[i]), 4),
                latency_ms=round(latency, 3),
            ))
        return slices

    def _sim_latency(self) -> float:
        """Simulate network latency with uniform jitter."""
        jitter = np.random.uniform(0, self._latency_jitter_ms)
        return self._base_latency_ms + jitter

    @staticmethod
    def _normalise_prices(prices) -> Dict[str, np.ndarray]:
        """Convert a DataFrame or dict into a dict of numpy arrays.

        Accepts columns in any case (Open/OPEN/open). If 'ticker' column
        is missing, fills with 'UNKNOWN'. This allows standard OHLCV
        DataFrames (from OpenBB, yfinance, etc.) to work directly.
        """
        # Column name mapping: accept any case
        _ALIASES = {
            "open": ["open", "Open", "OPEN"],
            "high": ["high", "High", "HIGH"],
            "low": ["low", "Low", "LOW"],
            "close": ["close", "Close", "CLOSE", "Adj Close", "adj_close"],
            "volume": ["volume", "Volume", "VOLUME"],
            "ticker": ["ticker", "Ticker", "TICKER", "symbol", "Symbol"],
        }

        def _find_col(source_cols, target):
            for alias in _ALIASES.get(target, [target]):
                if alias in source_cols:
                    return alias
            return None

        if hasattr(prices, "columns"):
            # pandas DataFrame path
            cols = list(prices.columns)
            out = {}
            for key in ("open", "high", "low", "close", "volume"):
                found = _find_col(cols, key)
                if found is not None:
                    out[key] = np.asarray(prices[found])
                else:
                    raise ValueError(f"prices_df missing required column: {key}")
            # ticker is optional — fill with 'UNKNOWN' if absent
            ticker_col = _find_col(cols, "ticker")
            if ticker_col is not None:
                out["ticker"] = np.asarray(prices[ticker_col])
            else:
                out["ticker"] = np.array(["UNKNOWN"] * len(prices))
            return out
        elif isinstance(prices, dict):
            out = {}
            keys = list(prices.keys())
            for key in ("open", "high", "low", "close", "volume"):
                found = _find_col(keys, key)
                if found is not None:
                    out[key] = np.asarray(prices[found])
                else:
                    raise ValueError(f"prices dict missing required key: {key}")
            ticker_key = _find_col(keys, "ticker")
            if ticker_key is not None:
                out["ticker"] = np.asarray(prices[ticker_key])
            else:
                out["ticker"] = np.array(["UNKNOWN"])
            return out
        else:
            raise TypeError(
                f"prices_df must be a pandas DataFrame or dict, got {type(prices)}"
            )

    # ------------------------------------------------------------------
    # Convenience: reset execution log
    # ------------------------------------------------------------------

    def reset_execution_log(self) -> None:
        """Clear all recorded executions (e.g. between trading sessions)."""
        count = len(self._execution_log)
        self._execution_log.clear()
        logger.info("Execution log cleared (%d entries removed)", count)
