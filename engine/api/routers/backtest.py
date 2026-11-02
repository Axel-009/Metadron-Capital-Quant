"""Backtest router — BACKTEST tab.

Wraps: EveningBacktester
Endpoints for backtest results, mispricings, RV, correlations, patterns, opportunities.
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.backtest")
router = APIRouter()

# ─── Lazy engine singleton ─────────────────────────────────
_backtester = None


def _get_backtester():
    global _backtester
    if _backtester is None:
        try:
            from engine.ml.evening_backtester import EveningBacktester
            _backtester = EveningBacktester()
        except Exception as e:
            logger.error(f"EveningBacktester init failed: {e}")
            from engine.ml.evening_backtester import EveningBacktester
            _backtester = EveningBacktester()
    return _backtester


# ─── Endpoints ─────────────────────────────────────────────


@router.get("/latest")
async def backtest_latest():
    """Most recent evening backtest results."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            **results,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/latest error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/history")
async def backtest_history(days: int = Query(default=7, ge=1, le=90)):
    """List of recent backtest dates with summary stats."""
    try:
        bt = _get_backtester()
        history = bt.get_backtest_history(days=days)
        return {
            "history": history,
            "total": len(history),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/history error: {e}")
        return {"history": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/date/{date_str}")
async def backtest_date(date_str: str):
    """Full results for a specific date."""
    try:
        bt = _get_backtester()
        results = bt.get_backtest_by_date(date_str)
        return {
            **results,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/date/{date_str} error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.post("/trigger")
async def backtest_trigger():
    """Manually trigger a backtest run."""
    try:
        bt = _get_backtester()
        results = bt.run_evening_backtest()
        return {
            "status": "completed",
            "summary": results.get("summary", {}),
            "regime": results.get("regime", "UNKNOWN"),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/trigger error: {e}")
        return {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/mispricings")
async def backtest_mispricings():
    """Current mispricing signals."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            "mispricings": results.get("mispricings", []),
            "total": len(results.get("mispricings", [])),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/mispricings error: {e}")
        return {"mispricings": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/relative-value")
async def backtest_relative_value():
    """Current RV rankings."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            "relative_value": results.get("relative_value", []),
            "total": len(results.get("relative_value", [])),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/relative-value error: {e}")
        return {"relative_value": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/correlations")
async def backtest_correlations():
    """Current correlation matrix and breakdowns."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            "correlations": results.get("correlations", {}),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/correlations error: {e}")
        return {"correlations": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/patterns")
async def backtest_patterns():
    """Recently detected patterns."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            "patterns": results.get("patterns", []),
            "total": len(results.get("patterns", [])),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/patterns error: {e}")
        return {"patterns": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/opportunities")
async def backtest_opportunities():
    """Aggregated high-conviction trade ideas from all analysis."""
    try:
        bt = _get_backtester()
        results = bt.get_latest_results()
        return {
            "opportunities": results.get("opportunities", []),
            "total": len(results.get("opportunities", [])),
            "summary": results.get("summary", {}),
            "regime": results.get("regime", "UNKNOWN"),
            "date": results.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"backtest/opportunities error: {e}")
        return {"opportunities": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}
