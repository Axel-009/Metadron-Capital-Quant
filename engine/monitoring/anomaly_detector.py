"""Anomaly Detector — Statistical anomaly detection for Metadron Capital.

Detects:
    - Volume anomalies (unusual spikes)
    - Price anomalies (gaps, extreme moves)
    - Correlation breakdown detection
    - Sector rotation anomalies
    - Macro regime transition alerts
    - VIX term structure inversion
    - Credit spread spike detection
    - Real-time alert queue
    - Historical anomaly database
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
from collections import deque

import numpy as np
import pandas as pd

try:
    from ..data.openbb_data import get_adj_close, get_returns, get_prices
    from ..data.universe_engine import SECTOR_ETFS
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_prices(*a, **kw): return pd.DataFrame()
    SECTOR_ETFS = {}

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    PRICE_GAP = "PRICE_GAP"
    EXTREME_MOVE = "EXTREME_MOVE"
    CORRELATION_BREAK = "CORRELATION_BREAK"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    VIX_INVERSION = "VIX_INVERSION"
    CREDIT_SPIKE = "CREDIT_SPIKE"
    LIQUIDITY_DRY = "LIQUIDITY_DRY"
    BREADTH_DIVERGENCE = "BREADTH_DIVERGENCE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Anomaly:
    """Single detected anomaly."""
    anomaly_type: AnomalyType = AnomalyType.EXTREME_MOVE
    severity: Severity = Severity.LOW
    ticker: str = ""
    description: str = ""
    zscore: float = 0.0
    value: float = 0.0
    threshold: float = 0.0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "ticker": self.ticker,
            "description": self.description,
            "zscore": self.zscore,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class AnomalyAlert:
    """Alert generated from anomaly."""
    anomaly: Anomaly = field(default_factory=Anomaly)
    action_required: bool = False
    suggested_action: str = ""
    acknowledged: bool = False
    created_at: str = ""


# ---------------------------------------------------------------------------
# Z-Score Detector
# ---------------------------------------------------------------------------
class ZScoreDetector:
    """Detects anomalies using z-score method."""

    def __init__(self, threshold: float = 3.0, lookback: int = 60):
        self.threshold = threshold
        self.lookback = lookback

    def detect(self, series: pd.Series, ticker: str = "") -> list[Anomaly]:
        anomalies = []
        if len(series) < self.lookback:
            return anomalies

        rolling_mean = series.rolling(self.lookback).mean()
        rolling_std = series.rolling(self.lookback).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscores = ((series - rolling_mean) / rolling_std).dropna()
        if zscores.empty:
            return anomalies

        extreme = zscores[abs(zscores) > self.threshold]
        for idx, z in extreme.items():
            severity = Severity.LOW
            if abs(z) > 5:
                severity = Severity.CRITICAL
            elif abs(z) > 4:
                severity = Severity.HIGH
            elif abs(z) > 3:
                severity = Severity.MEDIUM

            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.EXTREME_MOVE,
                severity=severity,
                ticker=ticker,
                description=f"Z-score {z:.2f} at {idx}",
                zscore=float(z),
                value=float(series.loc[idx]) if idx in series.index else 0,
                threshold=self.threshold,
                timestamp=str(idx),
            ))
        return anomalies


# ---------------------------------------------------------------------------
# IQR Detector
# ---------------------------------------------------------------------------
class IQRDetector:
    """Interquartile range outlier detection."""

    def __init__(self, factor: float = 1.5, lookback: int = 60):
        self.factor = factor
        self.lookback = lookback

    def detect(self, series: pd.Series, ticker: str = "") -> list[Anomaly]:
        anomalies = []
        if len(series) < self.lookback:
            return anomalies

        recent = series.iloc[-self.lookback:]
        q1 = recent.quantile(0.25)
        q3 = recent.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.factor * iqr
        upper = q3 + self.factor * iqr

        current = float(series.iloc[-1])
        if current < lower or current > upper:
            severity = Severity.HIGH if abs(current - (q1 + q3) / 2) > 2 * iqr else Severity.MEDIUM
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.EXTREME_MOVE,
                severity=severity,
                ticker=ticker,
                description=f"IQR outlier: {current:.4f} outside [{lower:.4f}, {upper:.4f}]",
                value=current,
                threshold=float(upper if current > upper else lower),
                timestamp=datetime.now().isoformat(),
            ))
        return anomalies


# ---------------------------------------------------------------------------
# Volume Anomaly Detector
# ---------------------------------------------------------------------------
class VolumeAnomalyDetector:
    """Detect unusual volume spikes."""

    def __init__(self, zscore_threshold: float = 3.0, lookback: int = 20):
        self.zscore_threshold = zscore_threshold
        self.lookback = lookback

    def scan(self, volume_data: pd.DataFrame) -> list[Anomaly]:
        anomalies = []
        for col in volume_data.columns:
            vol = volume_data[col].dropna()
            if len(vol) < self.lookback:
                continue

            avg = vol.iloc[-self.lookback:].mean()
            std = vol.iloc[-self.lookback:].std()
            if std == 0 or avg == 0:
                continue

            current = float(vol.iloc[-1])
            zscore = (current - avg) / std
            ratio = current / avg

            if abs(zscore) > self.zscore_threshold:
                severity = Severity.CRITICAL if ratio > 5 else (
                    Severity.HIGH if ratio > 3 else Severity.MEDIUM
                )
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    ticker=col,
                    description=f"Volume {ratio:.1f}x average ({current:,.0f} vs avg {avg:,.0f})",
                    zscore=zscore,
                    value=current,
                    threshold=avg * self.zscore_threshold,
                    timestamp=datetime.now().isoformat(),
                    metadata={"ratio": ratio, "avg_volume": avg},
                ))
        return anomalies


# ---------------------------------------------------------------------------
# Price Gap Detector
# ---------------------------------------------------------------------------
class PriceGapDetector:
    """Detect overnight gaps and intraday gaps."""

    def __init__(self, gap_threshold_pct: float = 3.0):
        self.gap_threshold_pct = gap_threshold_pct

    def scan(self, prices: pd.DataFrame) -> list[Anomaly]:
        anomalies = []
        returns = prices.pct_change().dropna()
        if returns.empty:
            return anomalies

        for col in returns.columns:
            r = returns[col].dropna()
            if r.empty:
                continue

            last_return = float(r.iloc[-1]) * 100
            if abs(last_return) > self.gap_threshold_pct:
                direction = "UP" if last_return > 0 else "DOWN"
                severity = Severity.CRITICAL if abs(last_return) > 10 else (
                    Severity.HIGH if abs(last_return) > 5 else Severity.MEDIUM
                )
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.PRICE_GAP,
                    severity=severity,
                    ticker=col,
                    description=f"Price gap {direction} {abs(last_return):.1f}%",
                    value=last_return,
                    threshold=self.gap_threshold_pct,
                    timestamp=datetime.now().isoformat(),
                ))
        return anomalies


# ---------------------------------------------------------------------------
# Correlation Breakdown Detector
# ---------------------------------------------------------------------------
class CorrelationBreakdownDetector:
    """Detect when correlations deviate from historical norms."""

    def __init__(self, window_short: int = 20, window_long: int = 120, threshold: float = 0.4):
        self.window_short = window_short
        self.window_long = window_long
        self.threshold = threshold

    def scan(self, returns: pd.DataFrame) -> list[Anomaly]:
        anomalies = []
        if len(returns) < self.window_long or returns.shape[1] < 2:
            return anomalies

        corr_short = returns.iloc[-self.window_short:].corr()
        corr_long = returns.iloc[-self.window_long:].corr()

        diff = (corr_short - corr_long).abs()
        for i in range(len(diff.columns)):
            for j in range(i + 1, len(diff.columns)):
                delta = float(diff.iloc[i, j])
                if delta > self.threshold:
                    a, b = diff.columns[i], diff.columns[j]
                    short_corr = float(corr_short.iloc[i, j])
                    long_corr = float(corr_long.iloc[i, j])
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=Severity.HIGH if delta > 0.6 else Severity.MEDIUM,
                        ticker=f"{a}/{b}",
                        description=f"Correlation shift: {long_corr:.2f} → {short_corr:.2f} (Δ{delta:.2f})",
                        value=delta,
                        threshold=self.threshold,
                        timestamp=datetime.now().isoformat(),
                        metadata={"short_corr": short_corr, "long_corr": long_corr},
                    ))
        return anomalies


# ---------------------------------------------------------------------------
# VIX Term Structure Monitor
# ---------------------------------------------------------------------------
class VIXTermStructureMonitor:
    """Monitor VIX term structure for inversions (backwardation)."""

    VIX_TICKERS = ["^VIX", "^VIX3M"]

    def check_inversion(self) -> list[Anomaly]:
        anomalies = []
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            prices = get_adj_close(self.VIX_TICKERS, start=start)
            if prices.empty or len(prices.columns) < 2:
                return anomalies

            vix_spot = float(prices.iloc[-1, 0])
            vix_3m = float(prices.iloc[-1, 1]) if prices.shape[1] > 1 else vix_spot

            if vix_spot > vix_3m:
                ratio = vix_spot / vix_3m if vix_3m > 0 else 1
                severity = Severity.CRITICAL if ratio > 1.2 else (
                    Severity.HIGH if ratio > 1.1 else Severity.MEDIUM
                )
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VIX_INVERSION,
                    severity=severity,
                    description=f"VIX backwardation: spot {vix_spot:.1f} > 3M {vix_3m:.1f} (ratio {ratio:.2f})",
                    value=ratio,
                    threshold=1.0,
                    timestamp=datetime.now().isoformat(),
                    metadata={"vix_spot": vix_spot, "vix_3m": vix_3m},
                ))
        except Exception as e:
            logger.warning(f"VIX term structure check failed: {e}")
        return anomalies


# ---------------------------------------------------------------------------
# Credit Spread Monitor
# ---------------------------------------------------------------------------
class CreditSpreadMonitor:
    """Monitor credit spreads for spikes."""

    def __init__(self, spike_threshold_pct: float = 10.0):
        self.spike_threshold_pct = spike_threshold_pct

    def scan(self) -> list[Anomaly]:
        anomalies = []
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            prices = get_adj_close(["HYG", "LQD"], start=start)
            if prices.empty or "HYG" not in prices.columns or "LQD" not in prices.columns:
                return anomalies

            spread = (prices["LQD"] / prices["HYG"]).dropna()
            if len(spread) < 20:
                return anomalies

            avg = float(spread.iloc[-20:].mean())
            current = float(spread.iloc[-1])
            change_pct = abs(current - avg) / avg * 100 if avg > 0 else 0

            if change_pct > self.spike_threshold_pct:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.CREDIT_SPIKE,
                    severity=Severity.HIGH if change_pct > 20 else Severity.MEDIUM,
                    description=f"Credit spread deviation: {change_pct:.1f}% from 20d avg",
                    value=change_pct,
                    threshold=self.spike_threshold_pct,
                    timestamp=datetime.now().isoformat(),
                    metadata={"current": current, "avg_20d": avg},
                ))
        except Exception as e:
            logger.warning(f"Credit spread scan failed: {e}")
        return anomalies


# ---------------------------------------------------------------------------
# Sector Rotation Anomaly
# ---------------------------------------------------------------------------
class SectorRotationAnomalyDetector:
    """Detect unusual sector rotation patterns."""

    def __init__(self, threshold_pct: float = 5.0):
        self.threshold_pct = threshold_pct

    def scan(self) -> list[Anomaly]:
        anomalies = []
        if not SECTOR_ETFS:
            return anomalies
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            prices = get_adj_close(list(SECTOR_ETFS.values()), start=start)
            if prices.empty:
                return anomalies

            returns_1d = prices.pct_change().iloc[-1].dropna()
            returns_5d = (prices.iloc[-1] / prices.iloc[-5] - 1).dropna() if len(prices) >= 5 else returns_1d

            inv_map = {v: k for k, v in SECTOR_ETFS.items()}
            spread = float(returns_1d.max() - returns_1d.min()) * 100

            if spread > self.threshold_pct:
                best_etf = returns_1d.idxmax()
                worst_etf = returns_1d.idxmin()
                best_sector = inv_map.get(best_etf, best_etf)
                worst_sector = inv_map.get(worst_etf, worst_etf)

                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SECTOR_ROTATION,
                    severity=Severity.HIGH if spread > 8 else Severity.MEDIUM,
                    description=f"Sector dispersion {spread:.1f}%: {best_sector} (+{float(returns_1d.max())*100:.1f}%) vs {worst_sector} ({float(returns_1d.min())*100:.1f}%)",
                    value=spread,
                    threshold=self.threshold_pct,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "best_sector": best_sector, "worst_sector": worst_sector,
                        "best_return": float(returns_1d.max()), "worst_return": float(returns_1d.min()),
                    },
                ))
        except Exception as e:
            logger.warning(f"Sector rotation scan failed: {e}")
        return anomalies


# ---------------------------------------------------------------------------
# Breadth Divergence
# ---------------------------------------------------------------------------
class BreadthDivergenceDetector:
    """Detect market breadth divergences."""

    def __init__(self):
        self._history: list[dict] = []

    def check(self, index_return: float, advance_ratio: float) -> list[Anomaly]:
        anomalies = []
        if index_return > 0.01 and advance_ratio < 0.4:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.BREADTH_DIVERGENCE,
                severity=Severity.HIGH,
                description=f"Bearish divergence: index +{index_return*100:.1f}% but only {advance_ratio*100:.0f}% advancing",
                value=advance_ratio,
                threshold=0.4,
                timestamp=datetime.now().isoformat(),
            ))
        elif index_return < -0.01 and advance_ratio > 0.6:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.BREADTH_DIVERGENCE,
                severity=Severity.MEDIUM,
                description=f"Bullish divergence: index {index_return*100:.1f}% but {advance_ratio*100:.0f}% advancing",
                value=advance_ratio,
                threshold=0.6,
                timestamp=datetime.now().isoformat(),
            ))
        return anomalies


# ---------------------------------------------------------------------------
# Alert Queue
# ---------------------------------------------------------------------------
class AlertQueue:
    """Real-time alert queue with priority handling."""

    def __init__(self, max_size: int = 1000):
        self._queue: deque[AnomalyAlert] = deque(maxlen=max_size)
        self._history: list[AnomalyAlert] = []

    def push(self, anomaly: Anomaly, action_required: bool = False, suggested_action: str = ""):
        alert = AnomalyAlert(
            anomaly=anomaly,
            action_required=action_required,
            suggested_action=suggested_action,
            created_at=datetime.now().isoformat(),
        )
        self._queue.append(alert)
        self._history.append(alert)

    def pop(self) -> Optional[AnomalyAlert]:
        if self._queue:
            return self._queue.popleft()
        return None

    def peek(self) -> Optional[AnomalyAlert]:
        if self._queue:
            return self._queue[0]
        return None

    def get_critical(self) -> list[AnomalyAlert]:
        return [a for a in self._queue if a.anomaly.severity == Severity.CRITICAL]

    def get_unacknowledged(self) -> list[AnomalyAlert]:
        return [a for a in self._queue if not a.acknowledged]

    @property
    def size(self) -> int:
        return len(self._queue)

    def clear(self):
        self._queue.clear()

    def get_summary(self) -> dict:
        counts = {}
        for alert in self._queue:
            sev = alert.anomaly.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return {"total": len(self._queue), "by_severity": counts}


# ---------------------------------------------------------------------------
# Anomaly Database
# ---------------------------------------------------------------------------
class AnomalyDatabase:
    """In-memory historical anomaly database."""

    def __init__(self):
        self._anomalies: list[Anomaly] = []

    def add(self, anomaly: Anomaly):
        self._anomalies.append(anomaly)

    def add_many(self, anomalies: list[Anomaly]):
        self._anomalies.extend(anomalies)

    def query(
        self,
        anomaly_type: Optional[AnomalyType] = None,
        severity: Optional[Severity] = None,
        ticker: Optional[str] = None,
        last_n: int = 100,
    ) -> list[Anomaly]:
        results = list(self._anomalies)
        if anomaly_type:
            results = [a for a in results if a.anomaly_type == anomaly_type]
        if severity:
            results = [a for a in results if a.severity == severity]
        if ticker:
            results = [a for a in results if a.ticker == ticker]
        return results[-last_n:]

    def count_by_type(self) -> dict[str, int]:
        counts = {}
        for a in self._anomalies:
            t = a.anomaly_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def count_by_severity(self) -> dict[str, int]:
        counts = {}
        for a in self._anomalies:
            s = a.severity.value
            counts[s] = counts.get(s, 0) + 1
        return counts

    @property
    def total(self) -> int:
        return len(self._anomalies)


# ---------------------------------------------------------------------------
# Master Anomaly Detector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Master anomaly detection engine.

    Coordinates all sub-detectors and manages the alert queue.
    """

    def __init__(self):
        self._zscore = ZScoreDetector(threshold=3.0)
        self._iqr = IQRDetector(factor=1.5)
        self._volume = VolumeAnomalyDetector()
        self._price_gap = PriceGapDetector()
        self._correlation = CorrelationBreakdownDetector()
        self._vix = VIXTermStructureMonitor()
        self._credit = CreditSpreadMonitor()
        self._sector = SectorRotationAnomalyDetector()
        self._breadth = BreadthDivergenceDetector()
        self._alert_queue = AlertQueue()
        self._database = AnomalyDatabase()

    def run_full_scan(self) -> dict:
        """Run all anomaly detectors."""
        all_anomalies = []

        # VIX term structure
        vix_anomalies = self._vix.check_inversion()
        all_anomalies.extend(vix_anomalies)

        # Credit spreads
        credit_anomalies = self._credit.scan()
        all_anomalies.extend(credit_anomalies)

        # Sector rotation
        sector_anomalies = self._sector.scan()
        all_anomalies.extend(sector_anomalies)

        # Store in database
        self._database.add_many(all_anomalies)

        # Push critical alerts
        for a in all_anomalies:
            action_required = a.severity in (Severity.HIGH, Severity.CRITICAL)
            self._alert_queue.push(a, action_required=action_required)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_anomalies": len(all_anomalies),
            "by_type": self._database.count_by_type(),
            "by_severity": self._database.count_by_severity(),
            "critical_alerts": len(self._alert_queue.get_critical()),
            "anomalies": [a.to_dict() for a in all_anomalies[:20]],
        }

    def scan_returns(self, returns: pd.DataFrame) -> list[Anomaly]:
        """Scan return series for anomalies."""
        all_anomalies = []
        for col in returns.columns:
            r = returns[col].dropna()
            all_anomalies.extend(self._zscore.detect(r, ticker=col))
        all_anomalies.extend(self._correlation.scan(returns))
        self._database.add_many(all_anomalies)
        return all_anomalies

    def scan_prices(self, prices: pd.DataFrame) -> list[Anomaly]:
        """Scan prices for gaps and extreme moves."""
        anomalies = self._price_gap.scan(prices)
        self._database.add_many(anomalies)
        return anomalies

    def get_alerts(self) -> AlertQueue:
        return self._alert_queue

    def get_database(self) -> AnomalyDatabase:
        return self._database

    def print_summary(self) -> str:
        """ASCII anomaly summary."""
        db = self._database
        lines = [
            "=" * 60,
            "ANOMALY DETECTOR SUMMARY",
            "=" * 60,
            f"  Total anomalies detected: {db.total}",
            "",
            "  By Type:",
        ]
        for t, c in sorted(db.count_by_type().items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    {t:<25} {c:>5}")
        lines.append("")
        lines.append("  By Severity:")
        for s, c in sorted(db.count_by_severity().items()):
            lines.append(f"    {s:<12} {c:>5}")
        lines.append("")
        lines.append(f"  Pending alerts: {self._alert_queue.size}")
        lines.append(f"  Critical alerts: {len(self._alert_queue.get_critical())}")
        lines.append("=" * 60)
        return "\n".join(lines)
