"""
Pattern Recognition & Conviction Signal Engine
================================================
Metadron Capital — Technical pattern detection, statistical anomaly
identification, and multi-factor conviction scoring for the investment
decision matrix.

All calculations use pure numpy. No external ML frameworks.

Formulas and methodology are documented inline for auditability.
"""

from __future__ import annotations

import logging
import datetime as dt

logger = logging.getLogger("metadron.intelligence.pattern_recognition")
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..data.yahoo_data import get_adj_close, get_returns, get_prices
    from ..data.universe_engine import SECTOR_ETFS, GICS_SECTORS, RV_PAIRS
except ImportError:
    def get_adj_close(*a, **kw) -> pd.DataFrame: return pd.DataFrame()
    def get_returns(*a, **kw) -> pd.DataFrame: return pd.DataFrame()
    def get_prices(*a, **kw) -> pd.DataFrame: return pd.DataFrame()
    SECTOR_ETFS: Dict[str, str] = {}
    GICS_SECTORS: Dict[str, str] = {}
    RV_PAIRS: List[Tuple[str, str]] = []


# ---------------------------------------------------------------------------
# Enums & Data Structures
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ConvictionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TrendRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class CorrelationRegime(str, Enum):
    DISPERSED = "DISPERSED"
    CORRELATED = "CORRELATED"


class LiquidityRegime(str, Enum):
    ABUNDANT = "ABUNDANT"
    NORMAL = "NORMAL"
    SCARCE = "SCARCE"


@dataclass
class ConvictionSignal:
    """Immutable record of a detected conviction-grade signal."""
    ticker: str
    pattern_type: str
    direction: Direction
    conviction_level: ConvictionLevel
    confirming_indicators: List[str]
    confidence: float  # [0, 1]
    entry_price: float
    stop_loss: float
    target: float
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)
    expiry: Optional[dt.datetime] = None
    audit_trail: List[str] = field(default_factory=list)

    @property
    def reward_risk(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        return abs(self.target - self.entry_price) / risk

    @property
    def is_expired(self) -> bool:
        if self.expiry is None:
            return False
        return dt.datetime.utcnow() > self.expiry


# ---------------------------------------------------------------------------
# 1. Candlestick Pattern Detector
# ---------------------------------------------------------------------------

class CandlestickPatternDetector:
    """
    Detects classic single- and multi-bar candlestick patterns on an OHLCV
    DataFrame.  Each pattern returns a confidence score in [0, 1].

    Expected columns: Open, High, Low, Close, Volume
    """

    def __init__(self, df: pd.DataFrame):
        self.o = df["Open"].values.astype(float)
        self.h = df["High"].values.astype(float)
        self.l = df["Low"].values.astype(float)
        self.c = df["Close"].values.astype(float)
        self.v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.ones(len(df))
        self.n = len(df)
        self.index = df.index

    # -- helpers ----------------------------------------------------------
    def _body(self, i: int) -> float:
        """Absolute body size = |Close - Open|."""
        return abs(self.c[i] - self.o[i])

    def _range(self, i: int) -> float:
        """Full candle range = High - Low."""
        return self.h[i] - self.l[i]

    def _upper_shadow(self, i: int) -> float:
        return self.h[i] - max(self.o[i], self.c[i])

    def _lower_shadow(self, i: int) -> float:
        return min(self.o[i], self.c[i]) - self.l[i]

    def _is_bullish(self, i: int) -> bool:
        return self.c[i] > self.o[i]

    def _is_bearish(self, i: int) -> bool:
        return self.c[i] < self.o[i]

    def _avg_body(self, i: int, lookback: int = 14) -> float:
        start = max(0, i - lookback)
        bodies = np.array([self._body(j) for j in range(start, i)])
        return float(np.mean(bodies)) if len(bodies) > 0 else 1e-9

    # -- patterns ---------------------------------------------------------
    def detect_hammer(self, i: int) -> float:
        """
        Hammer: small body at top, long lower shadow >= 2x body,
        minimal upper shadow.  Confidence based on shadow/body ratio.
        """
        if i < 1:
            return 0.0
        rng = self._range(i)
        if rng == 0:
            return 0.0
        body = self._body(i)
        lower = self._lower_shadow(i)
        upper = self._upper_shadow(i)
        if body == 0:
            return 0.0
        ratio = lower / body
        # Hammer criteria: lower shadow >= 2x body, upper shadow < 30% range
        if ratio >= 2.0 and upper / rng < 0.30:
            # Prior downtrend confirmation
            prior_trend = (self.c[i - 1] - self.c[max(0, i - 5)]) / (self.c[max(0, i - 5)] + 1e-9)
            if prior_trend < 0:
                return float(np.clip(ratio / 4.0, 0.3, 1.0))
        return 0.0

    def detect_engulfing(self, i: int) -> Tuple[float, Direction]:
        """
        Bullish engulfing: bearish bar followed by bullish bar whose body
        fully engulfs the prior body.  Returns (confidence, direction).
        """
        if i < 1:
            return 0.0, Direction.LONG
        prev_body = self._body(i - 1)
        curr_body = self._body(i)
        avg = self._avg_body(i)
        if avg == 0:
            return 0.0, Direction.LONG

        # Bullish engulfing
        if (self._is_bearish(i - 1) and self._is_bullish(i)
                and self.o[i] <= self.c[i - 1] and self.c[i] >= self.o[i - 1]):
            conf = float(np.clip(curr_body / (avg * 1.5), 0.3, 1.0))
            return conf, Direction.LONG

        # Bearish engulfing
        if (self._is_bullish(i - 1) and self._is_bearish(i)
                and self.o[i] >= self.c[i - 1] and self.c[i] <= self.o[i - 1]):
            conf = float(np.clip(curr_body / (avg * 1.5), 0.3, 1.0))
            return conf, Direction.SHORT

        return 0.0, Direction.LONG

    def detect_doji(self, i: int) -> float:
        """
        Doji: body < 10% of range.  Indicates indecision.
        Confidence = 1 - (body/range ratio * 10).
        """
        rng = self._range(i)
        if rng == 0:
            return 0.0
        ratio = self._body(i) / rng
        if ratio < 0.10:
            return float(np.clip(1.0 - ratio * 10.0, 0.5, 1.0))
        return 0.0

    def detect_morning_star(self, i: int) -> float:
        """
        Morning star (3 bars): large bearish, small body (gap down),
        large bullish closing above midpoint of first bar.
        """
        if i < 2:
            return 0.0
        avg = self._avg_body(i)
        if avg == 0:
            return 0.0
        b0, b1, b2 = self._body(i - 2), self._body(i - 1), self._body(i)
        if (self._is_bearish(i - 2) and b0 > avg
                and b1 < avg * 0.5
                and self._is_bullish(i) and b2 > avg
                and self.c[i] > (self.o[i - 2] + self.c[i - 2]) / 2.0):
            return float(np.clip(b2 / (avg * 1.5), 0.4, 1.0))
        return 0.0

    def detect_evening_star(self, i: int) -> float:
        """Mirror of morning star — bearish reversal."""
        if i < 2:
            return 0.0
        avg = self._avg_body(i)
        if avg == 0:
            return 0.0
        b0, b1, b2 = self._body(i - 2), self._body(i - 1), self._body(i)
        if (self._is_bullish(i - 2) and b0 > avg
                and b1 < avg * 0.5
                and self._is_bearish(i) and b2 > avg
                and self.c[i] < (self.o[i - 2] + self.c[i - 2]) / 2.0):
            return float(np.clip(b2 / (avg * 1.5), 0.4, 1.0))
        return 0.0

    def detect_three_white_soldiers(self, i: int) -> float:
        """Three consecutive bullish bars, each closing higher with bodies > avg."""
        if i < 2:
            return 0.0
        avg = self._avg_body(i)
        if avg == 0:
            return 0.0
        for k in range(3):
            idx = i - 2 + k
            if not self._is_bullish(idx) or self._body(idx) < avg * 0.6:
                return 0.0
        if self.c[i] > self.c[i - 1] > self.c[i - 2]:
            return float(np.clip(self._body(i) / (avg * 1.2), 0.5, 1.0))
        return 0.0

    def detect_three_black_crows(self, i: int) -> float:
        """Mirror of three white soldiers — bearish."""
        if i < 2:
            return 0.0
        avg = self._avg_body(i)
        if avg == 0:
            return 0.0
        for k in range(3):
            idx = i - 2 + k
            if not self._is_bearish(idx) or self._body(idx) < avg * 0.6:
                return 0.0
        if self.c[i] < self.c[i - 1] < self.c[i - 2]:
            return float(np.clip(self._body(i) / (avg * 1.2), 0.5, 1.0))
        return 0.0

    def scan(self) -> List[Dict]:
        """Scan the full series and return all detected patterns."""
        results: List[Dict] = []
        for i in range(2, self.n):
            ts = self.index[i] if hasattr(self.index[i], "isoformat") else i
            hammer = self.detect_hammer(i)
            if hammer > 0:
                results.append({"bar": i, "ts": ts, "pattern": "hammer",
                                "direction": Direction.LONG, "confidence": hammer})
            eng_conf, eng_dir = self.detect_engulfing(i)
            if eng_conf > 0:
                results.append({"bar": i, "ts": ts, "pattern": "engulfing",
                                "direction": eng_dir, "confidence": eng_conf})
            doji = self.detect_doji(i)
            if doji > 0:
                results.append({"bar": i, "ts": ts, "pattern": "doji",
                                "direction": Direction.LONG, "confidence": doji})
            ms = self.detect_morning_star(i)
            if ms > 0:
                results.append({"bar": i, "ts": ts, "pattern": "morning_star",
                                "direction": Direction.LONG, "confidence": ms})
            es = self.detect_evening_star(i)
            if es > 0:
                results.append({"bar": i, "ts": ts, "pattern": "evening_star",
                                "direction": Direction.SHORT, "confidence": es})
            tws = self.detect_three_white_soldiers(i)
            if tws > 0:
                results.append({"bar": i, "ts": ts, "pattern": "three_white_soldiers",
                                "direction": Direction.LONG, "confidence": tws})
            tbc = self.detect_three_black_crows(i)
            if tbc > 0:
                results.append({"bar": i, "ts": ts, "pattern": "three_black_crows",
                                "direction": Direction.SHORT, "confidence": tbc})
        return results


# ---------------------------------------------------------------------------
# 2. Chart Pattern Detector
# ---------------------------------------------------------------------------

class ChartPatternDetector:
    """
    Identifies geometric chart patterns via price pivots and trendline
    fitting.  All fitting uses numpy least-squares (np.polyfit).
    """

    def __init__(self, prices: np.ndarray, window: int = 5):
        """
        Parameters
        ----------
        prices : 1-D array of closing prices.
        window : half-window for local pivot detection.
        """
        self.prices = np.asarray(prices, dtype=float)
        self.n = len(self.prices)
        self.window = window
        self._pivot_highs: Optional[np.ndarray] = None
        self._pivot_lows: Optional[np.ndarray] = None

    # -- pivots -----------------------------------------------------------
    def _find_pivots(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of local highs and local lows."""
        if self._pivot_highs is not None:
            return self._pivot_highs, self._pivot_lows
        w = self.window
        highs, lows = [], []
        for i in range(w, self.n - w):
            seg = self.prices[i - w: i + w + 1]
            if self.prices[i] == np.max(seg):
                highs.append(i)
            if self.prices[i] == np.min(seg):
                lows.append(i)
        self._pivot_highs = np.array(highs, dtype=int)
        self._pivot_lows = np.array(lows, dtype=int)
        return self._pivot_highs, self._pivot_lows

    # -- support / resistance ---------------------------------------------
    def support_resistance(self, num_levels: int = 5) -> Dict[str, List[float]]:
        """
        Cluster pivot prices using a simple histogram-bin approach.
        Returns dict with 'support' and 'resistance' price levels.
        """
        highs_idx, lows_idx = self._find_pivots()
        if len(highs_idx) == 0 or len(lows_idx) == 0:
            return {"support": [], "resistance": []}

        high_prices = self.prices[highs_idx]
        low_prices = self.prices[lows_idx]

        def _cluster(vals: np.ndarray, n: int) -> List[float]:
            if len(vals) == 0:
                return []
            counts, edges = np.histogram(vals, bins=max(n * 3, 10))
            top_bins = np.argsort(counts)[-n:]
            return sorted([(edges[b] + edges[b + 1]) / 2.0 for b in top_bins])

        return {
            "resistance": _cluster(high_prices, num_levels),
            "support": _cluster(low_prices, num_levels),
        }

    # -- trendline fitting ------------------------------------------------
    @staticmethod
    def _fit_line(indices: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit y = slope * x + intercept via numpy polyfit.
        Returns (slope, intercept, r_squared).
        """
        if len(indices) < 2:
            return 0.0, 0.0, 0.0
        coeffs = np.polyfit(indices.astype(float), values, 1)
        slope, intercept = coeffs
        fitted = np.polyval(coeffs, indices.astype(float))
        ss_res = np.sum((values - fitted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return float(slope), float(intercept), float(r2)

    # -- head and shoulders -----------------------------------------------
    def detect_head_and_shoulders(self, inverse: bool = False) -> Dict:
        """
        Head & shoulders: three pivot highs where the middle is the highest,
        flanked by two lower "shoulders" at roughly the same level.

        For inverse H&S, we look at pivot lows (middle is the lowest).

        Returns dict with confidence and neckline info, or empty dict.
        """
        highs_idx, lows_idx = self._find_pivots()
        pivots = lows_idx if inverse else highs_idx
        pivot_prices = self.prices[pivots]

        if len(pivots) < 3:
            return {}

        best: Dict = {}
        best_conf = 0.0

        for i in range(len(pivots) - 2):
            ls, head, rs = pivot_prices[i], pivot_prices[i + 1], pivot_prices[i + 2]
            if inverse:
                # Head must be lowest, shoulders higher and roughly equal
                if head < ls and head < rs:
                    symmetry = 1.0 - abs(ls - rs) / (abs(head) + 1e-9)
                    prominence = (min(ls, rs) - head) / (abs(head) + 1e-9)
                else:
                    continue
            else:
                if head > ls and head > rs:
                    symmetry = 1.0 - abs(ls - rs) / (abs(head) + 1e-9)
                    prominence = (head - max(ls, rs)) / (abs(head) + 1e-9)
                else:
                    continue
            conf = float(np.clip(symmetry * 0.5 + prominence * 0.5, 0.0, 1.0))
            if conf > best_conf:
                best_conf = conf
                neckline = (ls + rs) / 2.0 if inverse else (ls + rs) / 2.0
                best = {
                    "pattern": "inverse_head_and_shoulders" if inverse else "head_and_shoulders",
                    "confidence": round(conf, 4),
                    "direction": Direction.LONG if inverse else Direction.SHORT,
                    "head_idx": int(pivots[i + 1]),
                    "neckline": round(neckline, 4),
                }
        return best

    # -- double top / bottom ----------------------------------------------
    def detect_double_top_bottom(self) -> List[Dict]:
        """
        Double top: two pivot highs at similar price.
        Double bottom: two pivot lows at similar price.
        Similarity threshold: within 1.5% of each other.
        """
        results: List[Dict] = []
        highs_idx, lows_idx = self._find_pivots()

        def _scan(idx_arr: np.ndarray, is_top: bool) -> None:
            prices_at = self.prices[idx_arr]
            for i in range(len(idx_arr) - 1):
                for j in range(i + 1, min(i + 5, len(idx_arr))):
                    pct_diff = abs(prices_at[i] - prices_at[j]) / (prices_at[i] + 1e-9)
                    if pct_diff < 0.015:
                        conf = float(np.clip(1.0 - pct_diff / 0.015, 0.3, 1.0))
                        results.append({
                            "pattern": "double_top" if is_top else "double_bottom",
                            "confidence": round(conf, 4),
                            "direction": Direction.SHORT if is_top else Direction.LONG,
                            "indices": (int(idx_arr[i]), int(idx_arr[j])),
                            "price_level": round((prices_at[i] + prices_at[j]) / 2, 4),
                        })

        _scan(highs_idx, is_top=True)
        _scan(lows_idx, is_top=False)
        return results

    # -- triangle detection -----------------------------------------------
    def detect_triangle(self) -> Dict:
        """
        Ascending triangle  : flat resistance + rising support.
        Descending triangle : flat support + falling resistance.
        Symmetric triangle  : converging resistance and support.

        Uses trendline slopes on pivot highs and lows over last 30 bars.
        """
        highs_idx, lows_idx = self._find_pivots()
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return {}

        # Use most recent pivots
        recent_h = highs_idx[highs_idx > self.n - 60] if len(highs_idx) else highs_idx
        recent_l = lows_idx[lows_idx > self.n - 60] if len(lows_idx) else lows_idx
        if len(recent_h) < 2 or len(recent_l) < 2:
            return {}

        slope_h, _, r2_h = self._fit_line(recent_h, self.prices[recent_h])
        slope_l, _, r2_l = self._fit_line(recent_l, self.prices[recent_l])

        avg_price = np.mean(self.prices[-60:])
        norm_h = slope_h / (avg_price + 1e-9) * 100
        norm_l = slope_l / (avg_price + 1e-9) * 100

        # Flat threshold: normalised slope < 0.02% per bar
        flat = 0.02
        if abs(norm_h) < flat and norm_l > flat:
            pat, direction = "ascending_triangle", Direction.LONG
        elif abs(norm_l) < flat and norm_h < -flat:
            pat, direction = "descending_triangle", Direction.SHORT
        elif norm_h < -flat and norm_l > flat:
            pat, direction = "symmetric_triangle", Direction.LONG  # neutral; bias long
        else:
            return {}

        conf = float(np.clip((r2_h + r2_l) / 2.0, 0.0, 1.0))
        return {"pattern": pat, "confidence": round(conf, 4), "direction": direction}

    # -- cup and handle ---------------------------------------------------
    def detect_cup_and_handle(self) -> Dict:
        """
        Cup: U-shaped price pattern over ~30-60 bars.
        Handle: shallow pullback after the right rim of the cup.

        Approximated by checking that the mid-section low is >=10% below the
        rims and that the rims are at similar levels.
        """
        if self.n < 40:
            return {}
        seg = self.prices[-60:] if self.n >= 60 else self.prices
        L = len(seg)
        left_rim = np.max(seg[: L // 4])
        right_rim = np.max(seg[3 * L // 4:])
        cup_low = np.min(seg[L // 4: 3 * L // 4])
        rim_avg = (left_rim + right_rim) / 2.0
        depth = (rim_avg - cup_low) / (rim_avg + 1e-9)
        symmetry = 1.0 - abs(left_rim - right_rim) / (rim_avg + 1e-9)

        if depth > 0.08 and symmetry > 0.85:
            conf = float(np.clip(depth * symmetry, 0.2, 1.0))
            return {"pattern": "cup_and_handle", "confidence": round(conf, 4),
                    "direction": Direction.LONG, "depth_pct": round(depth * 100, 2)}
        return {}

    # -- breakout / breakdown ---------------------------------------------
    def detect_breakout(self, lookback: int = 20) -> Dict:
        """
        Breakout : close > max of prior `lookback` bars.
        Breakdown: close < min of prior `lookback` bars.
        """
        if self.n < lookback + 1:
            return {}
        recent = self.prices[-lookback - 1: -1]
        last = self.prices[-1]
        hi, lo = np.max(recent), np.min(recent)
        if last > hi:
            pct = (last - hi) / (hi + 1e-9)
            return {"pattern": "breakout", "confidence": round(float(np.clip(pct / 0.03, 0.3, 1.0)), 4),
                    "direction": Direction.LONG, "level": round(float(hi), 4)}
        if last < lo:
            pct = (lo - last) / (lo + 1e-9)
            return {"pattern": "breakdown", "confidence": round(float(np.clip(pct / 0.03, 0.3, 1.0)), 4),
                    "direction": Direction.SHORT, "level": round(float(lo), 4)}
        return {}

    def scan_all(self) -> List[Dict]:
        """Run all chart pattern detectors and return findings."""
        results: List[Dict] = []
        hs = self.detect_head_and_shoulders(inverse=False)
        if hs:
            results.append(hs)
        ihs = self.detect_head_and_shoulders(inverse=True)
        if ihs:
            results.append(ihs)
        results.extend(self.detect_double_top_bottom())
        tri = self.detect_triangle()
        if tri:
            results.append(tri)
        cup = self.detect_cup_and_handle()
        if cup:
            results.append(cup)
        bo = self.detect_breakout()
        if bo:
            results.append(bo)
        return results


# ---------------------------------------------------------------------------
# 3. Statistical Anomaly Detector
# ---------------------------------------------------------------------------

class StatisticalAnomalyDetector:
    """
    Identifies statistical anomalies in price and volume series using
    z-scores, rolling correlations, and regime detection.
    """

    def __init__(self, close: np.ndarray, volume: np.ndarray | None = None):
        self.close = np.asarray(close, dtype=float)
        self.volume = np.asarray(volume, dtype=float) if volume is not None else None
        self.returns = np.diff(np.log(self.close + 1e-12))

    def zscore_outliers(self, lookback: int = 60, threshold: float = 2.5) -> List[Dict]:
        """
        Flag returns whose absolute z-score > threshold over a rolling
        window.  z_i = (r_i - mu) / sigma,  where mu, sigma are computed
        on [i-lookback, i).
        """
        results: List[Dict] = []
        for i in range(lookback, len(self.returns)):
            window = self.returns[i - lookback: i]
            mu, sigma = np.mean(window), np.std(window)
            if sigma < 1e-12:
                continue
            z = (self.returns[i] - mu) / sigma
            if abs(z) > threshold:
                results.append({"bar": i + 1, "z_score": round(float(z), 4),
                                "return": round(float(self.returns[i]) * 100, 4),
                                "direction": Direction.SHORT if z > 0 else Direction.LONG})
        return results

    def volume_anomaly(self, lookback: int = 20, multiplier: float = 2.0) -> List[Dict]:
        """
        Volume anomaly: volume_i > multiplier * mean(volume[i-lookback:i]).
        """
        if self.volume is None:
            return []
        results: List[Dict] = []
        for i in range(lookback, len(self.volume)):
            avg_vol = np.mean(self.volume[i - lookback: i])
            if avg_vol < 1:
                continue
            ratio = self.volume[i] / avg_vol
            if ratio > multiplier:
                results.append({"bar": i, "volume_ratio": round(float(ratio), 2),
                                "avg_volume": round(float(avg_vol), 0)})
        return results

    def correlation_breakdown(self, other: np.ndarray, short: int = 21,
                              long_: int = 252, threshold: float = 0.4) -> List[Dict]:
        """
        Rolling Pearson correlation between self.returns and another
        return series.  Flags when |short_corr - long_corr| > threshold.
        """
        other_ret = np.diff(np.log(np.asarray(other, dtype=float) + 1e-12))
        min_len = min(len(self.returns), len(other_ret))
        a, b = self.returns[:min_len], other_ret[:min_len]
        results: List[Dict] = []
        for i in range(long_, min_len):
            short_a, short_b = a[i - short: i], b[i - short: i]
            long_a, long_b = a[i - long_: i], b[i - long_: i]
            corr_s = float(np.corrcoef(short_a, short_b)[0, 1]) if len(short_a) > 1 else 0.0
            corr_l = float(np.corrcoef(long_a, long_b)[0, 1]) if len(long_a) > 1 else 0.0
            if abs(corr_s - corr_l) > threshold:
                results.append({"bar": i + 1, "short_corr": round(corr_s, 4),
                                "long_corr": round(corr_l, 4),
                                "deviation": round(abs(corr_s - corr_l), 4)})
        return results

    def regime_change(self, lookback: int = 60) -> List[Dict]:
        """
        Simple regime detection using rolling mean and variance of returns.
        A regime change is flagged when both the rolling mean and variance
        differ from the prior window by >1.5 sigma (computed on their own
        history).  This is a lightweight alternative to HMM.
        """
        results: List[Dict] = []
        half = lookback // 2
        for i in range(lookback, len(self.returns)):
            w1 = self.returns[i - lookback: i - half]
            w2 = self.returns[i - half: i]
            mu1, mu2 = np.mean(w1), np.mean(w2)
            s1, s2 = np.std(w1), np.std(w2)
            if s1 < 1e-12:
                continue
            mean_shift = abs(mu2 - mu1) / s1
            var_shift = abs(s2 - s1) / s1
            if mean_shift > 1.5 or var_shift > 1.5:
                results.append({"bar": i + 1, "mean_shift": round(float(mean_shift), 4),
                                "var_shift": round(float(var_shift), 4),
                                "new_regime_mu": round(float(mu2) * 252, 4),
                                "new_regime_vol": round(float(s2) * np.sqrt(252), 4)})
        return results

    def gap_detection(self, threshold_pct: float = 2.0) -> List[Dict]:
        """
        Gap = |Open_i - Close_{i-1}| / Close_{i-1} > threshold_pct / 100.
        Since we only have close data here, approximate as
        |close_i - close_{i-1}| with daily return > threshold.
        """
        results: List[Dict] = []
        for i in range(1, len(self.close)):
            pct = abs(self.close[i] - self.close[i - 1]) / (self.close[i - 1] + 1e-9) * 100
            if pct > threshold_pct:
                direction = Direction.LONG if self.close[i] > self.close[i - 1] else Direction.SHORT
                results.append({"bar": i, "gap_pct": round(float(pct), 2), "direction": direction})
        return results


# ---------------------------------------------------------------------------
# 4. Momentum Signal Engine
# ---------------------------------------------------------------------------

class MomentumSignalEngine:
    """
    Multi-timeframe momentum, RSI divergence, MACD divergence, and
    Bollinger Band squeeze detection.
    """

    PERIODS = (5, 10, 21, 63, 126, 252)

    def __init__(self, close: np.ndarray):
        self.close = np.asarray(close, dtype=float)
        self.n = len(self.close)

    def multi_timeframe_momentum(self) -> Dict[int, float]:
        """
        Momentum_t(p) = close_t / close_{t-p} - 1.
        Returns dict {period: momentum_pct} for the latest bar.
        """
        result: Dict[int, float] = {}
        for p in self.PERIODS:
            if self.n > p:
                result[p] = round(float((self.close[-1] / self.close[-1 - p] - 1.0) * 100), 4)
        return result

    def momentum_acceleration(self, period: int = 21) -> float:
        """
        Acceleration = momentum_t - momentum_{t-period}.
        Positive => accelerating, negative => decelerating.
        """
        if self.n < 2 * period + 1:
            return 0.0
        mom_now = self.close[-1] / self.close[-1 - period] - 1.0
        mom_prev = self.close[-1 - period] / self.close[-1 - 2 * period] - 1.0
        return round(float((mom_now - mom_prev) * 100), 4)

    def rsi(self, period: int = 14) -> np.ndarray:
        """
        RSI = 100 - 100 / (1 + RS)
        RS = EMA(gains, period) / EMA(losses, period)
        """
        deltas = np.diff(self.close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        alpha = 1.0 / period
        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))
        avg_gain[period - 1] = np.mean(gains[:period])
        avg_loss[period - 1] = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gains[i] * alpha
            avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + losses[i] * alpha
        rs = avg_gain / (avg_loss + 1e-12)
        rsi_vals = 100.0 - 100.0 / (1.0 + rs)
        rsi_vals[:period - 1] = np.nan
        return rsi_vals

    def rsi_divergence(self, period: int = 14, lookback: int = 30) -> Dict:
        """
        Bullish divergence: price makes lower low but RSI makes higher low.
        Bearish divergence: price makes higher high but RSI makes lower high.
        """
        rsi_vals = self.rsi(period)
        if self.n < lookback + period:
            return {}
        price_seg = self.close[-lookback:]
        rsi_seg = rsi_vals[-lookback:]
        mid = lookback // 2
        # Bullish: compare first-half low vs second-half low
        p_low1 = np.min(price_seg[:mid])
        p_low2 = np.min(price_seg[mid:])
        r_low1 = np.nanmin(rsi_seg[:mid])
        r_low2 = np.nanmin(rsi_seg[mid:])
        if p_low2 < p_low1 and r_low2 > r_low1:
            return {"divergence": "bullish", "direction": Direction.LONG,
                    "confidence": round(float(np.clip((r_low2 - r_low1) / 20.0, 0.2, 1.0)), 4)}
        # Bearish
        p_hi1 = np.max(price_seg[:mid])
        p_hi2 = np.max(price_seg[mid:])
        r_hi1 = np.nanmax(rsi_seg[:mid])
        r_hi2 = np.nanmax(rsi_seg[mid:])
        if p_hi2 > p_hi1 and r_hi2 < r_hi1:
            return {"divergence": "bearish", "direction": Direction.SHORT,
                    "confidence": round(float(np.clip((r_hi1 - r_hi2) / 20.0, 0.2, 1.0)), 4)}
        return {}

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD line = EMA(fast) - EMA(slow)
        Signal line = EMA(MACD, signal)
        Histogram = MACD - Signal
        """
        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            alpha = 2.0 / (span + 1)
            out = np.empty_like(arr)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
            return out

        ema_f = _ema(self.close, fast)
        ema_s = _ema(self.close, slow)
        macd_line = ema_f - ema_s
        sig_line = _ema(macd_line, signal)
        hist = macd_line - sig_line
        return macd_line, sig_line, hist

    def macd_histogram_divergence(self, lookback: int = 30) -> Dict:
        """
        Histogram divergence: price makes new extreme but histogram
        magnitude is lower (momentum fading).
        """
        _, _, hist = self.macd()
        if self.n < lookback + 26:
            return {}
        price_seg = self.close[-lookback:]
        hist_seg = hist[-lookback:]
        mid = lookback // 2
        # Bearish hist divergence
        p_hi1, p_hi2 = np.max(price_seg[:mid]), np.max(price_seg[mid:])
        h_hi1, h_hi2 = np.max(hist_seg[:mid]), np.max(hist_seg[mid:])
        if p_hi2 > p_hi1 and h_hi2 < h_hi1 and h_hi2 > 0:
            return {"divergence": "bearish_histogram", "direction": Direction.SHORT,
                    "confidence": round(float(np.clip((h_hi1 - h_hi2) / (abs(h_hi1) + 1e-9), 0.2, 1.0)), 4)}
        # Bullish
        p_lo1, p_lo2 = np.min(price_seg[:mid]), np.min(price_seg[mid:])
        h_lo1, h_lo2 = np.min(hist_seg[:mid]), np.min(hist_seg[mid:])
        if p_lo2 < p_lo1 and h_lo2 > h_lo1 and h_lo2 < 0:
            return {"divergence": "bullish_histogram", "direction": Direction.LONG,
                    "confidence": round(float(np.clip((h_lo2 - h_lo1) / (abs(h_lo1) + 1e-9), 0.2, 1.0)), 4)}
        return {}

    def bollinger_squeeze(self, period: int = 20, squeeze_pctile: float = 10.0) -> Dict:
        """
        Bollinger Band squeeze: bandwidth at historically low levels.
        Bandwidth = (upper - lower) / middle * 100.
        Squeeze flagged when current bandwidth < squeeze_pctile percentile
        of trailing 126-day bandwidth history.
        """
        if self.n < max(period, 126):
            return {}
        sma = np.convolve(self.close, np.ones(period) / period, mode="valid")
        # Align: sma[i] corresponds to close[period-1+i]
        offset = period - 1
        bw = np.zeros(len(sma))
        for i in range(len(sma)):
            seg = self.close[i: i + period]
            std = np.std(seg)
            bw[i] = (4.0 * std) / (sma[i] + 1e-9) * 100  # 2-sigma bands, width = 4*std

        if len(bw) < 126:
            return {}
        current_bw = bw[-1]
        hist_bw = bw[-126:]
        pctile = float(np.sum(hist_bw < current_bw)) / len(hist_bw) * 100
        if pctile < squeeze_pctile:
            return {"pattern": "bollinger_squeeze", "bandwidth_pctile": round(pctile, 2),
                    "bandwidth": round(float(current_bw), 4),
                    "direction": Direction.LONG}  # direction determined by breakout
        return {}


# ---------------------------------------------------------------------------
# 5. Relative Value Scanner
# ---------------------------------------------------------------------------

class RelativeValueScanner:
    """
    Pair-trading and relative value analysis for RV_PAIRS.
    Uses log-price spread, z-scores, and an Engle-Granger style
    cointegration approximation (pure numpy).
    """

    def __init__(self, price_a: np.ndarray, price_b: np.ndarray,
                 ticker_a: str = "A", ticker_b: str = "B"):
        self.pa = np.asarray(price_a, dtype=float)
        self.pb = np.asarray(price_b, dtype=float)
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        min_len = min(len(self.pa), len(self.pb))
        self.pa = self.pa[-min_len:]
        self.pb = self.pb[-min_len:]
        # Log spread: log(A) - beta * log(B)
        self.beta, self.spread = self._compute_spread()

    def _compute_spread(self) -> Tuple[float, np.ndarray]:
        """
        OLS regression: log(A) = alpha + beta * log(B) + residual.
        Spread = residual.
        """
        la, lb = np.log(self.pa + 1e-12), np.log(self.pb + 1e-12)
        coeffs = np.polyfit(lb, la, 1)
        beta = float(coeffs[0])
        spread = la - beta * lb - coeffs[1]
        return beta, spread

    def spread_zscore(self, lookback: int = 60) -> float:
        """
        z = (spread_t - mean(spread[t-lookback:t])) / std(spread[t-lookback:t])
        """
        if len(self.spread) < lookback:
            return 0.0
        window = self.spread[-lookback:]
        mu, sigma = np.mean(window), np.std(window)
        if sigma < 1e-12:
            return 0.0
        return float((self.spread[-1] - mu) / sigma)

    def cointegration_test(self) -> Dict:
        """
        Engle-Granger cointegration approximation (pure numpy):
        1. Run OLS: log(A) = a + b * log(B) + e
        2. Test stationarity of residuals via ADF-like check:
           Run AR(1) on residuals: e_t = rho * e_{t-1} + u_t
           If rho < 1 (significantly), spread is mean-reverting.
        """
        n = len(self.spread)
        if n < 30:
            return {"cointegrated": False, "reason": "insufficient_data"}
        # AR(1) on spread
        y = self.spread[1:]
        x = self.spread[:-1]
        rho = float(np.sum(x * y) / (np.sum(x ** 2) + 1e-12))
        residuals = y - rho * x
        se_rho = float(np.std(residuals) / (np.sqrt(np.sum(x ** 2)) + 1e-12))
        t_stat = (rho - 1.0) / (se_rho + 1e-12)
        # ADF critical values (approximate): -3.45 (1%), -2.87 (5%), -2.57 (10%)
        cointegrated = t_stat < -2.87
        return {
            "cointegrated": cointegrated,
            "rho": round(rho, 6),
            "t_stat": round(float(t_stat), 4),
            "beta": round(self.beta, 6),
            "critical_5pct": -2.87,
        }

    def half_life(self) -> float:
        """
        Half-life of mean reversion = -ln(2) / ln(rho),
        where rho is the AR(1) coefficient of the spread.
        """
        y, x = self.spread[1:], self.spread[:-1]
        rho = float(np.sum(x * y) / (np.sum(x ** 2) + 1e-12))
        if rho <= 0 or rho >= 1:
            return float("inf")
        return float(-np.log(2) / np.log(rho))

    def entry_exit_signals(self, entry_z: float = 2.0, exit_z: float = 0.5,
                           stop_z: float = 3.5) -> Dict:
        """
        Generate entry/exit signals based on current z-score:
        - |z| > entry_z : entry signal (short spread if z>0, long if z<0)
        - |z| < exit_z  : exit signal
        - |z| > stop_z  : stop loss
        """
        z = self.spread_zscore()
        signal: Dict = {"z_score": round(z, 4), "action": "hold"}
        if abs(z) > stop_z:
            signal["action"] = "stop_loss"
            signal["alert"] = "EXTREME"
        elif abs(z) > entry_z:
            signal["action"] = "enter"
            signal["direction"] = "short_spread" if z > 0 else "long_spread"
            signal["alert"] = "HIGH"
        elif abs(z) < exit_z:
            signal["action"] = "exit"
            signal["alert"] = "LOW"
        return signal

    def full_analysis(self) -> Dict:
        """Convenience: return a complete relative value summary."""
        return {
            "pair": f"{self.ticker_a}/{self.ticker_b}",
            "beta": round(self.beta, 6),
            "z_score": round(self.spread_zscore(), 4),
            "cointegration": self.cointegration_test(),
            "half_life_days": round(self.half_life(), 2),
            "signal": self.entry_exit_signals(),
        }


# ---------------------------------------------------------------------------
# 6. Conviction Engine
# ---------------------------------------------------------------------------

class ConvictionEngine:
    """
    Aggregates signals from all detectors into a single conviction score.
    Enforces the anti-hallucination rule: only flag signals with >= 3
    confirming indicators.

    Conviction levels:
        LOW      : score in [0.20, 0.40)
        MEDIUM   : score in [0.40, 0.60)
        HIGH     : score in [0.60, 0.80)
        EXTREME  : score in [0.80, 1.00]
    """

    LEVEL_THRESHOLDS = [
        (0.80, ConvictionLevel.EXTREME),
        (0.60, ConvictionLevel.HIGH),
        (0.40, ConvictionLevel.MEDIUM),
        (0.20, ConvictionLevel.LOW),
    ]

    # Factor weights for multi-factor confirmation
    WEIGHTS = {
        "candlestick": 0.15,
        "chart_pattern": 0.20,
        "momentum": 0.20,
        "volume": 0.15,
        "statistical": 0.15,
        "relative_value": 0.15,
    }

    MIN_CONFIRMING = 3  # anti-hallucination threshold

    def __init__(self):
        self._signals: List[ConvictionSignal] = []
        self._audit: List[str] = []

    def _classify_level(self, score: float) -> ConvictionLevel:
        for threshold, level in self.LEVEL_THRESHOLDS:
            if score >= threshold:
                return level
        return ConvictionLevel.LOW

    def _compute_decay(self, signal: ConvictionSignal, current_time: dt.datetime) -> float:
        """
        Signal decay: conviction decays exponentially with a half-life of
        5 trading days (~7 calendar days) without re-confirmation.

        decay_factor = 2^(-elapsed_days / 7)
        """
        elapsed = (current_time - signal.timestamp).total_seconds() / 86400.0
        return float(2.0 ** (-elapsed / 7.0))

    def score_signal(self, ticker: str, factor_scores: Dict[str, float],
                     direction: Direction, price: float,
                     atr: float = 0.0) -> Optional[ConvictionSignal]:
        """
        Combine factor scores into a single conviction score.

        Parameters
        ----------
        ticker : str
        factor_scores : dict mapping factor name -> confidence [0, 1]
        direction : Direction.LONG or Direction.SHORT
        price : current price
        atr : average true range (for stop/target calculation)

        Returns
        -------
        ConvictionSignal or None if anti-hallucination check fails.
        """
        # Count confirming indicators (score > 0.2)
        confirming = [k for k, v in factor_scores.items() if v > 0.2]
        if len(confirming) < self.MIN_CONFIRMING:
            self._audit.append(
                f"[REJECTED] {ticker}: only {len(confirming)} confirming "
                f"indicators ({confirming}), need >= {self.MIN_CONFIRMING}")
            return None

        # Weighted average score
        total_weight = 0.0
        weighted_sum = 0.0
        for factor, score in factor_scores.items():
            w = self.WEIGHTS.get(factor, 0.10)
            weighted_sum += score * w
            total_weight += w
        conviction_score = weighted_sum / (total_weight + 1e-12)
        conviction_score = float(np.clip(conviction_score, 0.0, 1.0))

        level = self._classify_level(conviction_score)

        # Stop loss and target from ATR (default 2% if ATR unavailable)
        if atr <= 0:
            atr = price * 0.02
        if direction == Direction.LONG:
            stop_loss = price - 2.0 * atr
            target = price + 3.0 * atr
        else:
            stop_loss = price + 2.0 * atr
            target = price - 3.0 * atr

        # Build pattern type description from top factors
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        pattern_type = "+".join(f[0] for f in sorted_factors[:3])

        audit_trail = [
            f"Factors: {factor_scores}",
            f"Confirming ({len(confirming)}): {confirming}",
            f"Weighted score: {conviction_score:.4f}",
            f"Level: {level.value}",
            f"ATR: {atr:.4f}, Stop: {stop_loss:.4f}, Target: {target:.4f}",
        ]

        signal = ConvictionSignal(
            ticker=ticker,
            pattern_type=pattern_type,
            direction=direction,
            conviction_level=level,
            confirming_indicators=confirming,
            confidence=round(conviction_score, 4),
            entry_price=round(price, 4),
            stop_loss=round(stop_loss, 4),
            target=round(target, 4),
            expiry=dt.datetime.utcnow() + dt.timedelta(days=10),
            audit_trail=audit_trail,
        )
        self._signals.append(signal)
        self._audit.append(
            f"[ACCEPTED] {ticker} {direction.value} conviction={conviction_score:.4f} "
            f"level={level.value} confirming={len(confirming)}")
        return signal

    def get_active_signals(self) -> List[ConvictionSignal]:
        """Return signals that have not expired, with decay applied."""
        now = dt.datetime.utcnow()
        active: List[ConvictionSignal] = []
        for s in self._signals:
            if s.is_expired:
                continue
            decay = self._compute_decay(s, now)
            decayed_conf = s.confidence * decay
            if decayed_conf > 0.15:
                active.append(s)
        return active

    def get_high_conviction(self) -> List[ConvictionSignal]:
        """Return only HIGH and EXTREME conviction signals."""
        return [s for s in self.get_active_signals()
                if s.conviction_level in (ConvictionLevel.HIGH, ConvictionLevel.EXTREME)]

    def get_audit_log(self) -> List[str]:
        return list(self._audit)

    def clear(self) -> None:
        self._signals.clear()
        self._audit.clear()


# ---------------------------------------------------------------------------
# 7. Market Regime Identifier
# ---------------------------------------------------------------------------

class MarketRegimeIdentifier:
    """
    Classifies the current market regime across four dimensions:
    trend, volatility, correlation, and liquidity.
    """

    def __init__(self, prices: np.ndarray, volume: np.ndarray | None = None,
                 benchmark: np.ndarray | None = None):
        self.prices = np.asarray(prices, dtype=float)
        self.returns = np.diff(np.log(self.prices + 1e-12))
        self.volume = np.asarray(volume, dtype=float) if volume is not None else None
        self.benchmark = np.asarray(benchmark, dtype=float) if benchmark is not None else None

    def trend_regime(self, short: int = 21, long_: int = 63) -> TrendRegime:
        """
        Trending  : |SMA_short - SMA_long| / SMA_long > 3% AND ADX-like
                     metric > 25 (approximated by directional efficiency).
        Breakout  : price just crossed above/below Donchian channel (20-day).
        Ranging   : otherwise.

        Directional efficiency = |price_t - price_{t-n}| / sum(|daily moves|)
        """
        if len(self.prices) < long_ + 1:
            return TrendRegime.RANGING
        sma_s = np.mean(self.prices[-short:])
        sma_l = np.mean(self.prices[-long_:])
        spread = abs(sma_s - sma_l) / (sma_l + 1e-9)

        # Directional efficiency over short period
        net_move = abs(self.prices[-1] - self.prices[-short])
        total_move = np.sum(np.abs(np.diff(self.prices[-short:])))
        efficiency = net_move / (total_move + 1e-9)

        # Breakout check
        dc_high = np.max(self.prices[-21:-1]) if len(self.prices) > 21 else self.prices[-1]
        dc_low = np.min(self.prices[-21:-1]) if len(self.prices) > 21 else self.prices[-1]
        if self.prices[-1] > dc_high or self.prices[-1] < dc_low:
            return TrendRegime.BREAKOUT

        if spread > 0.03 and efficiency > 0.35:
            return TrendRegime.TRENDING
        return TrendRegime.RANGING

    def volatility_regime(self, lookback: int = 21, long_lookback: int = 252) -> VolatilityRegime:
        """
        Annualised volatility percentile vs long-term history.
        LOW     : < 25th percentile
        NORMAL  : 25th - 75th
        HIGH    : 75th - 95th
        EXTREME : > 95th percentile
        """
        if len(self.returns) < long_lookback:
            return VolatilityRegime.NORMAL
        current_vol = np.std(self.returns[-lookback:]) * np.sqrt(252)
        # Rolling vol history
        vols = np.array([np.std(self.returns[i - lookback: i]) * np.sqrt(252)
                         for i in range(lookback, len(self.returns) + 1)])
        pctile = float(np.sum(vols < current_vol)) / len(vols) * 100
        if pctile > 95:
            return VolatilityRegime.EXTREME
        if pctile > 75:
            return VolatilityRegime.HIGH
        if pctile < 25:
            return VolatilityRegime.LOW
        return VolatilityRegime.NORMAL

    def correlation_regime(self, assets: np.ndarray | None = None,
                           threshold: float = 0.65) -> CorrelationRegime:
        """
        Average pairwise correlation among assets (columns).
        CORRELATED if avg corr > threshold, else DISPERSED.
        If no multi-asset data, compare to benchmark.
        """
        if assets is not None and assets.ndim == 2 and assets.shape[1] > 1:
            rets = np.diff(np.log(assets + 1e-12), axis=0)
            if len(rets) < 21:
                return CorrelationRegime.DISPERSED
            corr_matrix = np.corrcoef(rets[-63:].T)
            n = corr_matrix.shape[0]
            upper = corr_matrix[np.triu_indices(n, k=1)]
            avg_corr = float(np.mean(upper))
            return CorrelationRegime.CORRELATED if avg_corr > threshold else CorrelationRegime.DISPERSED
        if self.benchmark is not None:
            bench_ret = np.diff(np.log(self.benchmark + 1e-12))
            min_len = min(len(self.returns), len(bench_ret))
            if min_len < 21:
                return CorrelationRegime.DISPERSED
            corr = float(np.corrcoef(self.returns[-63:min_len],
                                     bench_ret[-63:min_len])[0, 1])
            return CorrelationRegime.CORRELATED if corr > threshold else CorrelationRegime.DISPERSED
        return CorrelationRegime.DISPERSED

    def liquidity_regime(self, lookback: int = 21, long_lookback: int = 252) -> LiquidityRegime:
        """
        Based on volume relative to long-term average.
        ABUNDANT : current avg volume > 120% of long-term
        SCARCE   : current avg volume < 70% of long-term
        NORMAL   : otherwise
        """
        if self.volume is None or len(self.volume) < long_lookback:
            return LiquidityRegime.NORMAL
        current_avg = np.mean(self.volume[-lookback:])
        long_avg = np.mean(self.volume[-long_lookback:])
        ratio = current_avg / (long_avg + 1e-9)
        if ratio > 1.20:
            return LiquidityRegime.ABUNDANT
        if ratio < 0.70:
            return LiquidityRegime.SCARCE
        return LiquidityRegime.NORMAL

    def full_regime(self, assets: np.ndarray | None = None) -> Dict[str, str]:
        return {
            "trend": self.trend_regime().value,
            "volatility": self.volatility_regime().value,
            "correlation": self.correlation_regime(assets).value,
            "liquidity": self.liquidity_regime().value,
        }


# ---------------------------------------------------------------------------
# 8. Pattern Recognition Engine (Master Orchestrator)
# ---------------------------------------------------------------------------

def _dir_label(d) -> str:
    """Normalise Direction enum / string to 'Bullish' | 'Bearish' | 'Neutral'."""
    if d is None:
        return "Neutral"
    s = d.value if hasattr(d, "value") else str(d)
    s = s.upper()
    if s in ("LONG", "BULLISH", "BUY"):
        return "Bullish"
    if s in ("SHORT", "BEARISH", "SELL"):
        return "Bearish"
    return "Neutral"


# Alias so the router can import as ``PatternRecognition``
# (the router historically used that name)
PatternRecognition = None  # set after class definition below


class PatternRecognitionEngine:
    """
    Master class that orchestrates all sub-engines and provides a unified
    API for scanning, signal generation, and reporting.
    """

    def __init__(self):
        self.conviction_engine = ConvictionEngine()
        self._scan_results: Dict[str, Dict] = {}
        self._rv_results: List[Dict] = []

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> float:
        """
        Average True Range:
        TR_i = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)
        ATR  = SMA(TR, period)
        """
        if len(close) < period + 1:
            return float(np.mean(high - low)) if len(high) > 0 else 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-period:]))

    def scan_ticker(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        Run all detectors on a single ticker's OHLCV DataFrame.

        Parameters
        ----------
        df : DataFrame with Open, High, Low, Close, Volume columns and
             DatetimeIndex.
        ticker : str

        Returns
        -------
        Dict with all detected patterns, anomalies, and signals.
        """
        close = df["Close"].values.astype(float)
        volume = df["Volume"].values.astype(float) if "Volume" in df.columns else None
        high = df["High"].values.astype(float) if "High" in df.columns else close
        low = df["Low"].values.astype(float) if "Low" in df.columns else close

        result: Dict = {"ticker": ticker, "bars": len(close)}

        # Candlestick patterns
        candle_det = CandlestickPatternDetector(df)
        candle_patterns = candle_det.scan()
        result["candlestick"] = candle_patterns
        candle_score = max((p["confidence"] for p in candle_patterns), default=0.0)

        # Chart patterns
        chart_det = ChartPatternDetector(close)
        chart_patterns = chart_det.scan_all()
        result["chart_patterns"] = chart_patterns
        chart_score = max((p.get("confidence", 0) for p in chart_patterns), default=0.0)

        # Statistical anomalies
        stat_det = StatisticalAnomalyDetector(close, volume)
        result["z_score_outliers"] = stat_det.zscore_outliers()
        result["volume_anomalies"] = stat_det.volume_anomaly()
        result["regime_changes"] = stat_det.regime_change()
        result["gaps"] = stat_det.gap_detection()
        stat_score = 0.0
        if result["z_score_outliers"]:
            stat_score = max(stat_score, min(abs(result["z_score_outliers"][-1]["z_score"]) / 4.0, 1.0))
        if result["volume_anomalies"]:
            stat_score = max(stat_score, min(result["volume_anomalies"][-1]["volume_ratio"] / 5.0, 1.0))

        # Momentum signals
        mom_eng = MomentumSignalEngine(close)
        result["momentum"] = mom_eng.multi_timeframe_momentum()
        result["momentum_accel"] = mom_eng.momentum_acceleration()
        result["rsi_divergence"] = mom_eng.rsi_divergence()
        result["macd_divergence"] = mom_eng.macd_histogram_divergence()
        result["bb_squeeze"] = mom_eng.bollinger_squeeze()
        mom_score = 0.0
        if result["rsi_divergence"]:
            mom_score = max(mom_score, result["rsi_divergence"].get("confidence", 0))
        if result["macd_divergence"]:
            mom_score = max(mom_score, result["macd_divergence"].get("confidence", 0))
        if result["bb_squeeze"]:
            mom_score = max(mom_score, 0.4)

        # Volume factor
        vol_score = 0.0
        if result["volume_anomalies"]:
            vol_score = min(result["volume_anomalies"][-1]["volume_ratio"] / 4.0, 1.0)

        # Market regime
        regime_id = MarketRegimeIdentifier(close, volume)
        result["regime"] = regime_id.full_regime()

        # Determine direction from strongest signal
        direction = Direction.LONG
        for cp in chart_patterns:
            if cp.get("confidence", 0) == chart_score and "direction" in cp:
                direction = cp["direction"]
                break
        for rsi_d in [result["rsi_divergence"]]:
            if rsi_d and rsi_d.get("confidence", 0) > chart_score:
                direction = rsi_d.get("direction", Direction.LONG)

        # Build conviction signal
        atr = self._atr(high, low, close)
        factor_scores = {
            "candlestick": candle_score,
            "chart_pattern": chart_score,
            "momentum": mom_score,
            "volume": vol_score,
            "statistical": stat_score,
        }
        result["factor_scores"] = factor_scores

        signal = self.conviction_engine.score_signal(
            ticker=ticker,
            factor_scores=factor_scores,
            direction=direction,
            price=float(close[-1]),
            atr=atr,
        )
        result["conviction_signal"] = signal
        self._scan_results[ticker] = result
        return result

    def scan_universe(self, tickers: List[str],
                      period: str = "1y") -> Dict[str, Dict]:
        """
        Scan a list of tickers by fetching OHLCV data and running all
        detectors.  Uses get_prices from the data layer.
        """
        results: Dict[str, Dict] = {}
        for ticker in tickers:
            try:
                df = get_prices(ticker, period=period)
                if df is None or df.empty or len(df) < 30:
                    continue
                results[ticker] = self.scan_ticker(df, ticker)
            except Exception as e:
                results[ticker] = {"ticker": ticker, "error": str(e)}
        return results

    def get_high_conviction_signals(self) -> List[ConvictionSignal]:
        """Return only HIGH and EXTREME conviction signals."""
        return self.conviction_engine.get_high_conviction()

    def get_rv_opportunities(self, pairs: List[Tuple[str, str]] | None = None,
                             period: str = "2y") -> List[Dict]:
        """
        Scan relative value pairs for entry opportunities.
        Uses RV_PAIRS from universe_engine if no pairs provided.
        """
        if pairs is None:
            pairs = RV_PAIRS if RV_PAIRS else []
        results: List[Dict] = []
        for ticker_a, ticker_b in pairs:
            try:
                pa = get_adj_close(ticker_a, period=period)
                pb = get_adj_close(ticker_b, period=period)
                if pa is None or pb is None:
                    continue
                pa_vals = pa.values.flatten() if hasattr(pa, "values") else np.array(pa)
                pb_vals = pb.values.flatten() if hasattr(pb, "values") else np.array(pb)
                scanner = RelativeValueScanner(pa_vals, pb_vals, ticker_a, ticker_b)
                analysis = scanner.full_analysis()
                results.append(analysis)
            except Exception as e:
                results.append({"pair": f"{ticker_a}/{ticker_b}", "error": str(e)})
        self._rv_results = results
        return results

    def get_daily_report(self) -> str:
        """
        Generate an ASCII summary of all detected patterns and signals.
        """
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f"  PATTERN RECOGNITION DAILY REPORT — {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 72)

        # High conviction signals
        high_conv = self.get_high_conviction_signals()
        lines.append(f"\n  HIGH CONVICTION SIGNALS ({len(high_conv)})")
        lines.append("-" * 72)
        if not high_conv:
            lines.append("  No high-conviction signals detected.")
        for s in high_conv:
            lines.append(f"  {s.ticker:<8} {s.direction.value:<6} "
                         f"conviction={s.confidence:.2f} [{s.conviction_level.value}]")
            lines.append(f"           entry={s.entry_price:.2f}  stop={s.stop_loss:.2f}  "
                         f"target={s.target:.2f}  R:R={s.reward_risk:.1f}")
            lines.append(f"           pattern: {s.pattern_type}")
            lines.append(f"           confirming: {', '.join(s.confirming_indicators)}")
            lines.append("")

        # Scan summaries
        lines.append(f"\n  SCANNED TICKERS ({len(self._scan_results)})")
        lines.append("-" * 72)
        for ticker, res in self._scan_results.items():
            n_candle = len(res.get("candlestick", []))
            n_chart = len(res.get("chart_patterns", []))
            regime = res.get("regime", {})
            sig = res.get("conviction_signal")
            conv_str = (f"{sig.conviction_level.value} ({sig.confidence:.2f})"
                        if sig else "NONE")
            lines.append(f"  {ticker:<8} candle={n_candle:<3} chart={n_chart:<3} "
                         f"conviction={conv_str:<20} "
                         f"trend={regime.get('trend', '?'):<10} "
                         f"vol={regime.get('volatility', '?')}")

        # RV opportunities
        if self._rv_results:
            lines.append(f"\n  RELATIVE VALUE OPPORTUNITIES ({len(self._rv_results)})")
            lines.append("-" * 72)
            for rv in self._rv_results:
                if "error" in rv:
                    lines.append(f"  {rv['pair']}: ERROR — {rv['error']}")
                    continue
                sig = rv.get("signal", {})
                coint = rv.get("cointegration", {})
                lines.append(
                    f"  {rv['pair']:<15} z={rv.get('z_score', 0):+.2f}  "
                    f"half_life={rv.get('half_life_days', 0):.0f}d  "
                    f"coint={'YES' if coint.get('cointegrated') else 'NO'}  "
                    f"action={sig.get('action', '?')}")

        # Audit summary
        audit = self.conviction_engine.get_audit_log()
        n_accepted = sum(1 for a in audit if "[ACCEPTED]" in a)
        n_rejected = sum(1 for a in audit if "[REJECTED]" in a)
        lines.append(f"\n  AUDIT: {n_accepted} accepted, {n_rejected} rejected "
                     f"(anti-hallucination filter: min {ConvictionEngine.MIN_CONFIRMING} confirmations)")
        lines.append("=" * 72)
        return "\n".join(lines)

    def backtest_signal(self, signal: ConvictionSignal, df: pd.DataFrame,
                        days: int = 20) -> Dict:
        """
        Quick forward-looking backtest of a signal using post-signal price
        data.

        Parameters
        ----------
        signal : ConvictionSignal to evaluate
        df : OHLCV DataFrame (must extend beyond signal timestamp)
        days : forward-looking window

        Returns
        -------
        Dict with PnL, max drawdown, hit rate, and outcome.
        """
        if df is None or df.empty:
            return {"error": "no_data"}

        # Find the signal bar
        close = df["Close"].values.astype(float)
        idx = df.index
        signal_bar = None
        for i, ts in enumerate(idx):
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if hasattr(ts_dt, "date") and hasattr(signal.timestamp, "date"):
                if ts_dt.date() >= signal.timestamp.date():
                    signal_bar = i
                    break
            elif i == len(idx) - 1:
                signal_bar = max(0, i - days)

        if signal_bar is None or signal_bar + days > len(close):
            return {"error": "insufficient_forward_data", "bars_available": len(close) - (signal_bar or 0)}

        forward = close[signal_bar: signal_bar + days]
        entry = signal.entry_price

        if signal.direction == Direction.LONG:
            pnl_series = (forward - entry) / entry * 100
        else:
            pnl_series = (entry - forward) / entry * 100

        max_pnl = float(np.max(pnl_series))
        min_pnl = float(np.min(pnl_series))
        final_pnl = float(pnl_series[-1])
        max_dd = float(np.min(pnl_series - np.maximum.accumulate(pnl_series)))

        # Check if target or stop hit
        hit_target = False
        hit_stop = False
        for p in forward:
            if signal.direction == Direction.LONG:
                if p >= signal.target:
                    hit_target = True
                    break
                if p <= signal.stop_loss:
                    hit_stop = True
                    break
            else:
                if p <= signal.target:
                    hit_target = True
                    break
                if p >= signal.stop_loss:
                    hit_stop = True
                    break

        outcome = "target_hit" if hit_target else ("stopped_out" if hit_stop else "expired")

        return {
            "ticker": signal.ticker,
            "direction": signal.direction.value,
            "entry_price": round(entry, 4),
            "final_pnl_pct": round(final_pnl, 4),
            "max_pnl_pct": round(max_pnl, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "outcome": outcome,
            "days": days,
        }

    def analyze(self, ticker: str, period: str = "1y") -> Dict:
        """Convenience method: fetch OHLCV data for *ticker* and run the full
        pattern-recognition scan.  Returns a JSON-serialisable dict.

        This is the primary entry point used by the API router
        ``/ml/patterns?ticker=SPY``.
        """
        df = get_prices(ticker, period=period)
        if df is None or (hasattr(df, "empty") and df.empty) or len(df) < 30:
            return {"ticker": ticker, "patterns": [], "error": "insufficient_data"}

        raw = self.scan_ticker(df, ticker)

        # ── Flatten into a UI-friendly pattern list ──
        patterns: List[Dict] = []

        for cp in raw.get("chart_patterns", []):
            patterns.append({
                "pattern": str(cp.get("pattern", "")),
                "asset": ticker,
                "confidence": float(cp.get("confidence", 0)),
                "timeframe": "1D",
                "direction": _dir_label(cp.get("direction")),
            })

        for candle in raw.get("candlestick", []):
            patterns.append({
                "pattern": str(candle.get("pattern", "")),
                "asset": ticker,
                "confidence": float(candle.get("confidence", 0)),
                "timeframe": "1D",
                "direction": _dir_label(candle.get("direction")),
            })

        # Add momentum-derived signals as pseudo-patterns
        rsi_div = raw.get("rsi_divergence")
        if rsi_div and rsi_div.get("confidence", 0) > 0:
            patterns.append({
                "pattern": "RSI Divergence",
                "asset": ticker,
                "confidence": float(rsi_div.get("confidence", 0)),
                "timeframe": "1D",
                "direction": _dir_label(rsi_div.get("direction")),
            })

        macd_div = raw.get("macd_divergence")
        if macd_div and macd_div.get("confidence", 0) > 0:
            patterns.append({
                "pattern": "MACD Divergence",
                "asset": ticker,
                "confidence": float(macd_div.get("confidence", 0)),
                "timeframe": "1D",
                "direction": _dir_label(macd_div.get("direction")),
            })

        bb = raw.get("bb_squeeze")
        if bb:
            patterns.append({
                "pattern": "Bollinger Squeeze",
                "asset": ticker,
                "confidence": 0.4,
                "timeframe": "1D",
                "direction": "Neutral",
            })

        # Sort by confidence desc, keep top 8
        patterns.sort(key=lambda p: p["confidence"], reverse=True)
        patterns = patterns[:8]

        # Conviction signal summary
        sig = raw.get("conviction_signal")
        conviction = None
        if sig and hasattr(sig, "confidence"):
            conviction = {
                "level": sig.conviction_level.value if hasattr(sig, "conviction_level") else "LOW",
                "confidence": round(sig.confidence, 4),
                "direction": sig.direction.value if hasattr(sig, "direction") else "LONG",
                "entry": round(sig.entry_price, 4) if hasattr(sig, "entry_price") else None,
                "stop": round(sig.stop_loss, 4) if hasattr(sig, "stop_loss") else None,
                "target": round(sig.target, 4) if hasattr(sig, "target") else None,
                "rr": round(sig.reward_risk, 2) if hasattr(sig, "reward_risk") else None,
            }

        return {
            "ticker": ticker,
            "patterns": patterns,
            "regime": raw.get("regime", {}),
            "factor_scores": raw.get("factor_scores", {}),
            "conviction": conviction,
        }

    def reset(self) -> None:
        """Clear all cached results and signals."""
        self.conviction_engine.clear()
        self._scan_results.clear()
        self._rv_results.clear()


# Alias for backward-compatible import from the API router
PatternRecognition = PatternRecognitionEngine
