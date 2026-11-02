"""Platinum Report module.

Re-exports PlatinumReportGenerator and related functions from daily_report.py
where the implementation lives. This module exists so that run_close.py and
the API router can import from engine.monitoring.platinum_report directly.
"""
from engine.monitoring.daily_report import (
    PlatinumReport,
    PlatinumReportGenerator,
    format_platinum_report,
)

# Also import V2 generator if available
try:
    from engine.monitoring.platinum_report_v2 import PlatinumReportV2Generator
except ImportError:
    PlatinumReportV2Generator = None  # type: ignore

__all__ = [
    "PlatinumReport",
    "PlatinumReportGenerator",
    "PlatinumReportV2Generator",
    "format_platinum_report",
]
