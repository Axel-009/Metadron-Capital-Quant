"""QuantStrategyExecutor — L7 HFT Execution Stage.

Independent technical strategy execution that runs inside the ExecutionEngine
pipeline immediately after DecisionMatrix approval, before orders hit
exchange-core / wondertrader.

Each strategy runs independently and produces its own trade signal.
The executor collects all signals and computes a weighted technical
consensus that adjusts position sizing and provides stop/target levels.

Pipeline position:
    DecisionMatrix (Stage 5.5) → approved tickers
        ↓
    **HFT Technical Execution (Stage 6.5)** ← THIS MODULE
        Each approved ticker runs through 12 independent strategies:
        ┌─ Bollinger W-bottom         ┌─ Dual Thrust breakout
        ├─ MACD oscillator            ├─ RSI divergence
        ├─ Parabolic SAR trend        ├─ Heikin-Ashi candles
        ├─ Shooting Star reversal     ├─ Awesome Oscillator
        ├─ Pair Trading (Engle-Granger)├─ VIX regime gate
        ├─ Arbitrage detection         └─ London Breakout
        ↓
    Weighted consensus → size adjustment → risk gates → exchange-core

Source: intelligence_platform/quant-trading (je-suis-tm strategies)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intelligence Platform: quant-trading sub-repo integration
# Provides: ArbitrageDetector (stat arb, triangular FX, index arb, cross-asset
# RV, calendar spread) and UniverseScanner (momentum/value screens, pair finder).
# The 12 strategies above are simplified re-implementations; these provide the
# full-featured originals for optional use.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    _arb_spec = _ilu.spec_from_file_location(
        "arbitrage_detector",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
            / "intelligence_platform" / "quant-trading" / "arbitrage_detector.py"),
    )
    _arb_mod = _ilu.module_from_spec(_arb_spec)
    _arb_spec.loader.exec_module(_arb_mod)
    ArbitrageDetector = _arb_mod.ArbitrageDetector
    ARBITRAGE_DETECTOR_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    ArbitrageDetector = None
    ARBITRAGE_DETECTOR_AVAILABLE = False
    logger.info("quant-trading ArbitrageDetector unavailable — inline strategies only")

try:
    _us_spec = _ilu.spec_from_file_location(
        "universe_scanner",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
            / "intelligence_platform" / "quant-trading" / "universe_scanner.py"),
    )
    _us_mod = _ilu.module_from_spec(_us_spec)
    _us_spec.loader.exec_module(_us_mod)
    UniverseScanner = _us_mod.UniverseScanner
    UNIVERSE_SCANNER_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    UniverseScanner = None
    UNIVERSE_SCANNER_AVAILABLE = False
    logger.info("quant-trading UniverseScanner unavailable")


# ---------------------------------------------------------------------------
# Bollinger Bands W-Bottom (Bollinger Bands Pattern Recognition backtest.py)
# ---------------------------------------------------------------------------

class BollingerWStrategy:
    """Bollinger Bands with W-bottom/M-top pattern recognition.

    Arithmetic pattern detection — no ML needed.
    Detects W-bottom (bullish) and M-top (bearish) formations
    relative to the upper/lower bands.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def compute_bands(self, prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"price": prices})
        df["std"] = prices.rolling(window=self.window, min_periods=self.window).std()
        df["mid"] = prices.rolling(window=self.window, min_periods=self.window).mean()
        df["upper"] = df["mid"] + self.num_std * df["std"]
        df["lower"] = df["mid"] - self.num_std * df["std"]
        return df

    def detect_w_bottom(self, df: pd.DataFrame) -> bool:
        """W-bottom: two troughs near lower band, second higher, then break above mid."""
        if len(df) < 40:
            return False
        recent = df.iloc[-40:].dropna()
        if len(recent) < 20:
            return False
        lower, price, mid = recent["lower"].values, recent["price"].values, recent["mid"].values
        touches = np.where(price <= lower * 1.01)[0]
        if len(touches) < 2:
            return False
        first = touches[0]
        later = touches[touches > first + 5]
        if len(later) == 0:
            return False
        second = later[0]
        if price[second] > price[first]:
            post = price[second:]
            if len(post) > 2 and post[-1] > mid[second]:
                return True
        return False

    def detect_m_top(self, df: pd.DataFrame) -> bool:
        """M-top: two peaks near upper band, second lower, then break below mid."""
        if len(df) < 40:
            return False
        recent = df.iloc[-40:].dropna()
        if len(recent) < 20:
            return False
        upper, price, mid = recent["upper"].values, recent["price"].values, recent["mid"].values
        touches = np.where(price >= upper * 0.99)[0]
        if len(touches) < 2:
            return False
        first = touches[0]
        later = touches[touches > first + 5]
        if len(later) == 0:
            return False
        second = later[0]
        if price[second] < price[first]:
            post = price[second:]
            if len(post) > 2 and post[-1] < mid[second]:
                return True
        return False

    def run(self, prices: pd.Series) -> dict:
        df = self.compute_bands(prices)
        w = self.detect_w_bottom(df)
        m = self.detect_m_top(df)
        direction = 1 if w else (-1 if m else 0)
        return {
            "strategy": "bollinger_w",
            "direction": direction,
            "signal": direction * 0.7,
            "confidence": 0.65 if direction != 0 else 0.0,
            "pattern": "w_bottom" if w else ("m_top" if m else "none"),
            "stop_loss": float(df["lower"].iloc[-1]) if not pd.isna(df["lower"].iloc[-1]) else 0.0,
            "take_profit": float(df["upper"].iloc[-1]) if not pd.isna(df["upper"].iloc[-1]) else 0.0,
        }


# ---------------------------------------------------------------------------
# MACD Oscillator (MACD Oscillator backtest.py)
# ---------------------------------------------------------------------------

class MACDStrategy:
    """MACD convergence/divergence oscillator with crossover detection."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def run(self, prices: pd.Series) -> dict:
        if len(prices) < self.slow + 1:
            return {"strategy": "macd", "direction": 0, "signal": 0.0, "confidence": 0.0}
        ma_fast = prices.rolling(window=self.fast, min_periods=1).mean()
        ma_slow = prices.rolling(window=self.slow, min_periods=1).mean()
        osc = float((ma_fast - ma_slow).iloc[-1])
        prev_osc = float((ma_fast - ma_slow).iloc[-2])
        crossover = 0
        if osc > 0 and prev_osc <= 0:
            crossover = 1
        elif osc < 0 and prev_osc >= 0:
            crossover = -1
        direction = int(np.sign(crossover)) if crossover != 0 else int(np.sign(osc))
        std = max(float(prices.std()), 1e-10)
        return {
            "strategy": "macd",
            "direction": direction,
            "signal": float(np.clip(osc / std, -1, 1)),
            "confidence": min(abs(osc) / std, 1.0),
            "oscillator": osc,
            "crossover": crossover,
        }


# ---------------------------------------------------------------------------
# RSI Pattern Recognition (RSI Pattern Recognition backtest.py)
# ---------------------------------------------------------------------------

class RSIStrategy:
    """RSI with overbought/oversold pattern recognition."""

    def __init__(self, window: int = 14):
        self.window = window

    def compute_rsi(self, prices: pd.Series) -> float:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.window).mean()
        rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
        return float(100.0 - (100.0 / (1.0 + rs)))

    def run(self, prices: pd.Series) -> dict:
        if len(prices) < self.window + 2:
            return {"strategy": "rsi", "direction": 0, "signal": 0.0, "confidence": 0.0}
        rsi = self.compute_rsi(prices)
        if rsi < 30:
            direction, signal = 1, (30 - rsi) / 30
        elif rsi > 70:
            direction, signal = -1, (rsi - 70) / 30
        else:
            direction, signal = 0, 0.0
        return {
            "strategy": "rsi",
            "direction": direction,
            "signal": signal * direction,
            "confidence": abs(signal),
            "rsi": rsi,
        }


# ---------------------------------------------------------------------------
# Parabolic SAR (Parabolic SAR backtest.py)
# ---------------------------------------------------------------------------

class ParabolicSARStrategy:
    """Parabolic Stop-and-Reverse trend-following indicator."""

    def __init__(self, af_init: float = 0.02, af_step: float = 0.02, af_max: float = 0.20):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

    def run(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        if len(close) < 3:
            return {"strategy": "parabolic_sar", "direction": 0, "signal": 0.0, "confidence": 0.0}
        h, l, c = high.values, low.values, close.values
        trend = 1 if c[1] > c[0] else -1
        sar = h[0] if trend > 0 else l[0]
        ep = h[1] if trend > 0 else l[1]
        af = self.af_init

        for i in range(2, len(c)):
            temp = sar + af * (ep - sar)
            if trend < 0:
                sar_new = max(temp, h[i - 1], h[i - 2])
                if sar_new < h[i]:
                    trend, sar_new, ep, af = 1, ep, h[i], self.af_init
                else:
                    if l[i] < ep:
                        ep = l[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                sar_new = min(temp, l[i - 1], l[i - 2])
                if sar_new > l[i]:
                    trend, sar_new, ep, af = -1, ep, l[i], self.af_init
                else:
                    if h[i] > ep:
                        ep = h[i]
                        af = min(af + self.af_step, self.af_max)
            sar = sar_new

        return {
            "strategy": "parabolic_sar",
            "direction": trend,
            "signal": float(trend) * 0.6,
            "confidence": 0.55,
            "sar_value": float(sar),
            "trend": "up" if trend > 0 else "down",
        }


# ---------------------------------------------------------------------------
# Heikin-Ashi (Heikin-Ashi backtest.py)
# ---------------------------------------------------------------------------

class HeikinAshiStrategy:
    """Heikin-Ashi smoothed candle trend detection."""

    def run(self, open_: pd.Series, high: pd.Series,
            low: pd.Series, close: pd.Series) -> dict:
        if len(close) < 5:
            return {"strategy": "heikin_ashi", "direction": 0, "signal": 0.0, "confidence": 0.0}
        ha_close = (open_ + high + low + close) / 4
        ha_open = pd.Series(np.zeros(len(close)), index=close.index)
        ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
        for i in range(1, len(close)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

        last3_bull = all(ha_close.iloc[-j] > ha_open.iloc[-j] for j in range(1, 4))
        last3_bear = all(ha_close.iloc[-j] < ha_open.iloc[-j] for j in range(1, 4))
        direction = 1 if last3_bull else (-1 if last3_bear else 0)
        return {
            "strategy": "heikin_ashi",
            "direction": direction,
            "signal": float(direction) * 0.5,
            "confidence": 0.6 if direction != 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Shooting Star (Shooting Star backtest.py)
# ---------------------------------------------------------------------------

class ShootingStarStrategy:
    """Shooting star candlestick reversal pattern detection."""

    def run(self, open_: pd.Series, high: pd.Series,
            low: pd.Series, close: pd.Series) -> dict:
        if len(close) < 2:
            return {"strategy": "shooting_star", "direction": 0, "signal": 0.0, "confidence": 0.0}
        o, h, l, c = open_.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]
        body = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l
        detected = (body > 1e-10 and upper_shadow >= 2 * body and lower_shadow <= body * 0.3)
        return {
            "strategy": "shooting_star",
            "direction": -1 if detected else 0,
            "signal": -0.7 if detected else 0.0,
            "confidence": 0.6 if detected else 0.0,
            "detected": detected,
        }


# ---------------------------------------------------------------------------
# Dual Thrust (Dual Thrust backtest.py)
# ---------------------------------------------------------------------------

class DualThrustStrategy:
    """Opening range breakout strategy.

    Sets upper/lower thresholds from previous day's OHLC range.
    Long when price exceeds upper threshold, short below lower.
    """

    def __init__(self, k1: float = 0.5, k2: float = 0.5):
        self.k1 = k1
        self.k2 = k2

    def run(self, open_: pd.Series, high: pd.Series,
            low: pd.Series, close: pd.Series) -> dict:
        if len(close) < 3:
            return {"strategy": "dual_thrust", "direction": 0, "signal": 0.0, "confidence": 0.0}
        hh, ll, hc, lc = high.iloc[-2], low.iloc[-2], close.iloc[-2], close.iloc[-2]
        range_val = max(hh - lc, hc - ll)
        today_open = open_.iloc[-1]
        upper = today_open + self.k1 * range_val
        lower = today_open - self.k2 * range_val
        current = close.iloc[-1]
        if current > upper:
            direction = 1
        elif current < lower:
            direction = -1
        else:
            direction = 0
        return {
            "strategy": "dual_thrust",
            "direction": direction,
            "signal": float(direction) * 0.65,
            "confidence": 0.6 if direction != 0 else 0.0,
            "upper_threshold": float(upper),
            "lower_threshold": float(lower),
            "breakout": "upper" if direction > 0 else ("lower" if direction < 0 else "none"),
        }


# ---------------------------------------------------------------------------
# London Breakout (London Breakout backtest.py)
# ---------------------------------------------------------------------------

class LondonBreakoutStrategy:
    """FX session breakout: Tokyo→London handoff.

    For equities, adapts to pre-market/opening range breakout.
    Uses high/low of the last N bars as the breakout range.
    """

    def __init__(self, lookback: int = 5, multiplier: float = 1.0):
        self.lookback = lookback
        self.multiplier = multiplier

    def run(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        if len(close) < self.lookback + 1:
            return {"strategy": "london_breakout", "direction": 0, "signal": 0.0, "confidence": 0.0}
        range_high = high.iloc[-self.lookback - 1:-1].max()
        range_low = low.iloc[-self.lookback - 1:-1].min()
        range_size = range_high - range_low
        upper = range_high + self.multiplier * range_size * 0.1
        lower = range_low - self.multiplier * range_size * 0.1
        current = close.iloc[-1]
        if current > upper:
            direction = 1
        elif current < lower:
            direction = -1
        else:
            direction = 0
        return {
            "strategy": "london_breakout",
            "direction": direction,
            "signal": float(direction) * 0.6,
            "confidence": 0.55 if direction != 0 else 0.0,
            "range_high": float(range_high),
            "range_low": float(range_low),
        }


# ---------------------------------------------------------------------------
# Awesome Oscillator (Awesome Oscillator backtest.py)
# ---------------------------------------------------------------------------

class AwesomeOscillatorStrategy:
    """Awesome Oscillator = SMA5(midpoint) - SMA34(midpoint)."""

    def run(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        if len(close) < 35:
            return {"strategy": "awesome_osc", "direction": 0, "signal": 0.0, "confidence": 0.0}
        midpoint = (high + low) / 2
        ao = midpoint.rolling(5).mean() - midpoint.rolling(34).mean()
        ao_val = float(ao.iloc[-1]) if not pd.isna(ao.iloc[-1]) else 0.0
        std = max(float(close.std()), 1e-10)
        ao_norm = float(np.clip(ao_val / std, -1, 1))
        return {
            "strategy": "awesome_osc",
            "direction": int(np.sign(ao_norm)),
            "signal": ao_norm,
            "confidence": min(abs(ao_norm), 1.0),
            "ao_raw": ao_val,
        }


# ---------------------------------------------------------------------------
# Pair Trading (Pair trading backtest.py)  — Engle-Granger cointegration
# ---------------------------------------------------------------------------

class PairTradingStrategy:
    """Cointegration-based mean reversion (Engle-Granger VECM).

    Computes spread z-score between two cointegrated assets.
    Mean reverts when z-score exceeds ±2σ.
    """

    def __init__(self, window: int = 60, entry_z: float = 2.0):
        self.window = window
        self.entry_z = entry_z

    def run(self, prices_a: pd.Series, prices_b: pd.Series) -> dict:
        if len(prices_a) < self.window or len(prices_b) < self.window:
            return {"strategy": "pair_trading", "direction": 0, "signal": 0.0, "confidence": 0.0}
        ratio = prices_a / prices_b.replace(0, np.nan)
        ratio = ratio.dropna()
        if len(ratio) < self.window:
            return {"strategy": "pair_trading", "direction": 0, "signal": 0.0, "confidence": 0.0}
        mu = ratio.rolling(window=self.window).mean().iloc[-1]
        sigma = ratio.rolling(window=self.window).std().iloc[-1]
        if sigma < 1e-10:
            return {"strategy": "pair_trading", "direction": 0, "signal": 0.0, "confidence": 0.0}
        zscore = float((ratio.iloc[-1] - mu) / sigma)
        if abs(zscore) > self.entry_z:
            direction = -1 if zscore > 0 else 1  # mean reversion
            signal = float(np.clip(abs(zscore) / 3.0, 0, 1))
        else:
            direction, signal = 0, 0.0
        return {
            "strategy": "pair_trading",
            "direction": direction,
            "signal": signal * direction,
            "confidence": min(abs(zscore) / 3.0, 1.0),
            "zscore": zscore,
            "half_life_est": max(1, int(self.window * 0.3)),
        }


# ---------------------------------------------------------------------------
# Arbitrage Detector (arbitrage_detector.py — Metadron-specific)
# ---------------------------------------------------------------------------

class ArbitrageStrategy:
    """Multi-class arbitrage detection.

    Types: statistical arb, index arb, cross-asset relative value.
    Simplified for execution-time signals (full detector in
    intelligence_platform/quant-trading/arbitrage_detector.py).
    """

    def run(self, prices_a: pd.Series, prices_b: pd.Series,
            window: int = 60) -> dict:
        """Cross-asset relative value signal."""
        if len(prices_a) < window or len(prices_b) < window:
            return {"strategy": "arbitrage", "direction": 0, "signal": 0.0, "confidence": 0.0}
        ratio = (prices_a / prices_b.replace(0, np.nan)).dropna()
        if len(ratio) < window:
            return {"strategy": "arbitrage", "direction": 0, "signal": 0.0, "confidence": 0.0}
        mu = ratio.rolling(window).mean().iloc[-1]
        sigma = ratio.rolling(window).std().iloc[-1]
        if sigma < 1e-10:
            return {"strategy": "arbitrage", "direction": 0, "signal": 0.0, "confidence": 0.0}
        zscore = float((ratio.iloc[-1] - mu) / sigma)
        if abs(zscore) > 2.5:
            direction = -1 if zscore > 0 else 1
            signal = float(np.clip(abs(zscore) / 4.0, 0, 1))
        else:
            direction, signal = 0, 0.0
        return {
            "strategy": "arbitrage",
            "direction": direction,
            "signal": signal * direction,
            "confidence": min(abs(zscore) / 4.0, 1.0),
            "zscore": zscore,
            "type": "cross_asset_rv",
        }


# ---------------------------------------------------------------------------
# VIX Regime Gate (VIX Calculator.py)
# ---------------------------------------------------------------------------

class VIXRegimeGate:
    """VIX-based regime classification and position scaling.

    Acts as a kill-switch in crisis conditions (VIX > 40).
    Scales all HFT strategy signals based on vol regime.
    """

    REGIMES = {
        (0, 15): ("low_vol", 1.0),
        (15, 20): ("normal", 0.85),
        (20, 30): ("elevated", 0.5),
        (30, 40): ("high_stress", 0.25),
        (40, 999): ("crisis", 0.0),
    }

    def classify(self, vix: float) -> tuple[str, float]:
        for (lo, hi), (name, scale) in self.REGIMES.items():
            if lo <= vix < hi:
                return name, scale
        return "crisis", 0.0


# ---------------------------------------------------------------------------
# Options Straddle (Options Straddle backtest.py)
# ---------------------------------------------------------------------------

class OptionsStraddleStrategy:
    """Volatility straddle signal based on implied vs realized vol.

    When implied vol is significantly above realized, sell straddle (direction=-1).
    When implied vol is below realized, buy straddle (direction=+1).
    For equity execution: translates to vol expectation signal.
    """

    def run(self, close: pd.Series, vix: float = 20.0, window: int = 20) -> dict:
        if len(close) < window + 1:
            return {"strategy": "options_straddle", "direction": 0, "signal": 0.0, "confidence": 0.0}
        realized_vol = float(close.pct_change().rolling(window).std().iloc[-1] * np.sqrt(252) * 100)
        implied_vol = vix
        vol_spread = implied_vol - realized_vol
        if abs(vol_spread) > 5:
            # Implied > realized → sell vol (bearish signal for directional)
            direction = -1 if vol_spread > 0 else 1
            signal = float(np.clip(abs(vol_spread) / 20, 0, 1))
        else:
            direction, signal = 0, 0.0
        return {
            "strategy": "options_straddle",
            "direction": direction,
            "signal": signal * direction,
            "confidence": min(abs(vol_spread) / 20, 1.0),
            "implied_vol": implied_vol,
            "realized_vol": realized_vol,
            "vol_spread": vol_spread,
        }


# ===========================================================================
# QuantStrategyExecutor — runs all 12 strategies independently
# ===========================================================================

class QuantStrategyExecutor:
    """L7 HFT Technical Execution Stage.

    Runs 12 independent technical strategies from quant-trading on every
    approved ticker. Each strategy fires independently. The executor
    computes a weighted consensus that adjusts position sizing before
    orders are submitted to exchange-core.

    Integrated directly into ExecutionEngine.run_pipeline() as Stage 6.5.
    """

    def __init__(self):
        # Initialize all strategies independently
        self.bollinger = BollingerWStrategy()
        self.macd = MACDStrategy()
        self.rsi = RSIStrategy()
        self.sar = ParabolicSARStrategy()
        self.heikin_ashi = HeikinAshiStrategy()
        self.shooting_star = ShootingStarStrategy()
        self.dual_thrust = DualThrustStrategy()
        self.london_breakout = LondonBreakoutStrategy()
        self.awesome_osc = AwesomeOscillatorStrategy()
        self.pair_trading = PairTradingStrategy()
        self.arbitrage = ArbitrageStrategy()
        self.vix_gate = VIXRegimeGate()
        self.options_straddle = OptionsStraddleStrategy()
        self._execution_log: list[dict] = []
        logger.info("QuantStrategyExecutor (L7 HFT) initialized — 12 independent strategies")

    def execute(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        vix: float = 20.0,
        pair_ticker: Optional[str] = None,
        pair_prices: Optional[pd.Series] = None,
    ) -> dict:
        """Run all 12 strategies independently on a ticker.

        Args:
            ticker: Symbol being traded.
            ohlcv: DataFrame with Open, High, Low, Close, Volume.
            vix: Current VIX level.
            pair_ticker: Optional pair for stat-arb strategies.
            pair_prices: Optional pair close prices.

        Returns:
            dict with per-strategy results + consensus.
        """
        close = ohlcv["Close"]
        high = ohlcv["High"]
        low = ohlcv["Low"]
        open_ = ohlcv["Open"]

        # VIX regime gate — scales all signals
        regime, scale = self.vix_gate.classify(vix)
        if scale == 0.0:
            result = {
                "ticker": ticker, "regime": regime, "vix": vix, "scale": 0.0,
                "kill_switch": True, "strategies": {}, "consensus_direction": 0,
                "consensus_signal": 0.0, "size_multiplier": 0.0,
            }
            self._execution_log.append(result)
            logger.warning(f"[{ticker}] VIX={vix:.1f} CRISIS — kill switch active")
            return result

        # Run each strategy independently
        strategies = {}

        strategies["bollinger_w"] = self.bollinger.run(close)
        strategies["macd"] = self.macd.run(close)
        strategies["rsi"] = self.rsi.run(close)
        strategies["parabolic_sar"] = self.sar.run(high, low, close)
        strategies["heikin_ashi"] = self.heikin_ashi.run(open_, high, low, close)
        strategies["shooting_star"] = self.shooting_star.run(open_, high, low, close)
        strategies["dual_thrust"] = self.dual_thrust.run(open_, high, low, close)
        strategies["london_breakout"] = self.london_breakout.run(high, low, close)
        strategies["awesome_osc"] = self.awesome_osc.run(high, low, close)
        strategies["options_straddle"] = self.options_straddle.run(close, vix)

        # Pair-dependent strategies
        if pair_prices is not None and len(pair_prices) > 0:
            strategies["pair_trading"] = self.pair_trading.run(close, pair_prices)
            strategies["arbitrage"] = self.arbitrage.run(close, pair_prices)

        # Apply VIX scale to all signals
        for name, s in strategies.items():
            s["signal"] = s.get("signal", 0.0) * scale

        # Compute weighted consensus across all active strategies
        active = [(name, s) for name, s in strategies.items()
                  if s.get("direction", 0) != 0 and s.get("confidence", 0) > 0]

        if active:
            weighted_sum = sum(s["signal"] * s["confidence"] for _, s in active)
            total_weight = sum(s["confidence"] for _, s in active)
            consensus_signal = weighted_sum / total_weight if total_weight > 0 else 0.0

            dirs = [s["direction"] for _, s in active]
            agreement = abs(sum(dirs)) / len(dirs) if dirs else 0.0
            consensus_direction = int(np.sign(sum(dirs)))
        else:
            consensus_signal, consensus_direction, agreement = 0.0, 0, 0.0

        # Size multiplier: based on consensus strength
        # 0.5x base + up to 0.5x from agreement ratio
        size_multiplier = (0.5 + 0.5 * agreement) * scale

        # --- Composite stop/take-profit from all strategies ---
        # Collect stop/target levels from every strategy that provides them
        current_price = float(close.iloc[-1])
        stop_candidates = []
        target_candidates = []

        # Bollinger: lower band = stop, upper band = target
        bb = strategies.get("bollinger_w", {})
        if bb.get("stop_loss", 0) > 0:
            stop_candidates.append(("bollinger", bb["stop_loss"]))
        if bb.get("take_profit", 0) > 0:
            target_candidates.append(("bollinger", bb["take_profit"]))

        # Parabolic SAR: sar_value = trailing stop (direction-aware)
        sar = strategies.get("parabolic_sar", {})
        sar_val = sar.get("sar_value", 0)
        if sar_val > 0 and sar.get("direction", 0) != 0:
            if sar.get("direction", 0) > 0:
                # Long trend: SAR is below price → stop
                if sar_val < current_price:
                    stop_candidates.append(("sar", sar_val))
            else:
                # Short trend: SAR is above price → stop for shorts
                if sar_val > current_price:
                    stop_candidates.append(("sar", sar_val))

        # Dual Thrust: breakout thresholds as support/resistance
        dt = strategies.get("dual_thrust", {})
        dt_lower = dt.get("lower_threshold", 0)
        dt_upper = dt.get("upper_threshold", 0)
        if dt_lower > 0 and dt_lower < current_price:
            stop_candidates.append(("dual_thrust", dt_lower))
        if dt_upper > 0 and dt_upper > current_price:
            target_candidates.append(("dual_thrust", dt_upper))

        # RSI: compute price-implied stop from oversold/overbought zones
        rsi_data = strategies.get("rsi", {})
        rsi_val = rsi_data.get("rsi", 50)
        if rsi_val < 30 and current_price > 0:
            # Oversold bounce — stop at 2% below entry
            stop_candidates.append(("rsi_oversold", current_price * 0.98))
        elif rsi_val > 70 and current_price > 0:
            # Overbought — target at current (take profit here)
            target_candidates.append(("rsi_overbought", current_price * 1.01))

        # Compute best stop: tightest (highest for longs) to protect capital
        # Compute best target: most conservative (lowest for longs) to lock gains
        if consensus_direction >= 0:
            # Long bias: stop = max of candidates (tightest), target = min (nearest)
            best_stop = max((v for _, v in stop_candidates), default=0.0)
            best_target = min((v for _, v in target_candidates), default=0.0)
        else:
            # Short bias: stop = min of candidates (tightest above), target = max (furthest down)
            best_stop = min((v for _, v in stop_candidates), default=0.0) if stop_candidates else 0.0
            best_target = max((v for _, v in target_candidates), default=0.0) if target_candidates else 0.0

        # Sanity: stop must be between price and target
        if best_stop > 0 and best_target > 0:
            if consensus_direction >= 0 and best_stop >= current_price:
                best_stop = current_price * 0.97  # fallback: 3% below
            elif consensus_direction < 0 and best_stop <= current_price:
                best_stop = current_price * 1.03  # fallback: 3% above

        result = {
            "ticker": ticker,
            "regime": regime,
            "vix": vix,
            "scale": scale,
            "kill_switch": False,
            "strategies": strategies,
            "active_count": len(active),
            "active_names": [name for name, _ in active],
            "consensus_direction": consensus_direction,
            "consensus_signal": float(np.clip(consensus_signal, -1, 1)),
            "agreement": agreement,
            "size_multiplier": size_multiplier,
            "stop_loss": float(best_stop),
            "take_profit": float(best_target),
            "stop_sources": [(name, float(v)) for name, v in stop_candidates],
            "target_sources": [(name, float(v)) for name, v in target_candidates],
        }

        self._execution_log.append(result)
        logger.info(
            f"[{ticker}] HFT Technical: {len(active)}/{len(strategies)} active, "
            f"consensus={consensus_direction:+d} signal={consensus_signal:+.3f} "
            f"agreement={agreement:.2f} size={size_multiplier:.2f} regime={regime}"
        )
        return result

    def get_execution_log(self) -> list[dict]:
        return list(self._execution_log)
