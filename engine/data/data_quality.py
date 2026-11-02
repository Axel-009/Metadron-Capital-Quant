"""Metadron Capital — Data Quality Gates.

Validates market data freshness, completeness, and outliers before
downstream signal engines consume it. Sits between DataIngestionOrchestrator
and UniversalDataPool.

Three checks per data frame:
  1. Stale-data detection: timestamp age vs cadence
  2. Completeness: non-null ratio per column
  3. Outlier rejection: z-score filter (>5σ flagged)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("metadron.scan.data_quality")


class DataQualityGate:
    """Pre-pool validator for market data frames."""

    # Default thresholds (configurable via __init__)
    DEFAULT_STALE_SECONDS = 600         # Data older than 10 min = stale
    DEFAULT_COMPLETENESS_PCT = 0.80     # Require 80% non-null cells
    DEFAULT_OUTLIER_ZSCORE = 5.0        # Flag values beyond 5 sigma

    def __init__(
        self,
        max_stale_seconds: int = DEFAULT_STALE_SECONDS,
        min_completeness_pct: float = DEFAULT_COMPLETENESS_PCT,
        outlier_zscore: float = DEFAULT_OUTLIER_ZSCORE,
    ):
        self.max_stale_seconds = max_stale_seconds
        self.min_completeness_pct = min_completeness_pct
        self.outlier_zscore = outlier_zscore
        self._stale_count = 0
        self._incomplete_count = 0
        self._outlier_count = 0
        self._total_validations = 0

    def validate(self, df, source: str = "unknown", timestamp: Optional[datetime] = None) -> dict:
        """Validate a DataFrame. Returns quality report.

        Args:
            df: pandas DataFrame to validate
            source: data source name for logging
            timestamp: data timestamp (uses now if not provided)

        Returns:
            {
                "valid": bool,
                "stale": bool,
                "completeness_pct": float,
                "outliers": int,
                "issues": list[str]
            }
        """
        self._total_validations += 1
        issues = []
        report = {
            "valid": True,
            "stale": False,
            "completeness_pct": 1.0,
            "outliers": 0,
            "issues": issues,
            "source": source,
        }

        # 1. Stale-data check
        if timestamp:
            now = datetime.now(timezone.utc) if timestamp.tzinfo else datetime.now()
            age_seconds = (now - timestamp).total_seconds()
            if age_seconds > self.max_stale_seconds:
                report["stale"] = True
                issues.append(f"stale ({int(age_seconds)}s old, max {self.max_stale_seconds}s)")
                self._stale_count += 1

        # 2. Completeness check
        try:
            if df is None or len(df) == 0:
                report["completeness_pct"] = 0.0
                issues.append("empty frame")
                self._incomplete_count += 1
            else:
                total_cells = df.size
                non_null = df.count().sum() if hasattr(df, "count") else total_cells
                completeness = non_null / total_cells if total_cells > 0 else 0
                report["completeness_pct"] = round(completeness, 3)
                if completeness < self.min_completeness_pct:
                    issues.append(f"incomplete ({completeness:.1%} < {self.min_completeness_pct:.0%})")
                    self._incomplete_count += 1
        except Exception as e:
            issues.append(f"completeness check failed: {e}")

        # 3. Outlier check (only on numeric columns)
        try:
            if df is not None and len(df) > 5:
                import numpy as np
                numeric_cols = df.select_dtypes(include=[np.number]).columns if hasattr(df, "select_dtypes") else []
                outlier_total = 0
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) > 5:
                        mean = series.mean()
                        std = series.std()
                        if std > 0:
                            z = abs((series - mean) / std)
                            outlier_total += int((z > self.outlier_zscore).sum())
                report["outliers"] = outlier_total
                if outlier_total > 0:
                    issues.append(f"{outlier_total} outliers (>{self.outlier_zscore}σ)")
                    self._outlier_count += outlier_total
        except Exception:
            pass

        report["valid"] = len(issues) == 0

        if not report["valid"]:
            logger.warning("[DataQuality] %s: %s", source, "; ".join(issues))

        return report

    def get_stats(self) -> dict:
        """Return validator statistics for monitoring."""
        return {
            "total_validations": self._total_validations,
            "stale_count": self._stale_count,
            "incomplete_count": self._incomplete_count,
            "outlier_count": self._outlier_count,
            "stale_rate": round(self._stale_count / max(self._total_validations, 1), 3),
            "incomplete_rate": round(self._incomplete_count / max(self._total_validations, 1), 3),
        }


# Singleton
_gate: Optional[DataQualityGate] = None


def get_quality_gate() -> DataQualityGate:
    global _gate
    if _gate is None:
        _gate = DataQualityGate()
    return _gate
