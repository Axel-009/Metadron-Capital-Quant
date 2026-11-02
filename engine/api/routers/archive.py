"""Archive router — ARCHIVE tab.

Wraps: ArchiveEngine, DailySummaryGenerator
Endpoints for browsing archived files, triggering archival, and daily summaries.
"""
from fastapi import APIRouter, Query
from datetime import datetime, date
import logging

logger = logging.getLogger("metadron-api.archive")
router = APIRouter()

# ─── Lazy engine singletons ────────────────────────────────
_archive_engine = None
_summary_generator = None


def _get_archive_engine():
    global _archive_engine
    if _archive_engine is None:
        try:
            from engine.ops.archive_engine import ArchiveEngine
            _archive_engine = ArchiveEngine()
        except Exception as e:
            logger.error(f"ArchiveEngine init failed: {e}")
            from engine.ops.archive_engine import ArchiveEngine
            _archive_engine = ArchiveEngine()
    return _archive_engine


def _get_summary_generator():
    global _summary_generator
    if _summary_generator is None:
        try:
            from engine.monitoring.daily_summary_generator import DailySummaryGenerator
            _summary_generator = DailySummaryGenerator()
        except Exception as e:
            logger.error(f"DailySummaryGenerator init failed: {e}")
            from engine.monitoring.daily_summary_generator import DailySummaryGenerator
            _summary_generator = DailySummaryGenerator()
    return _summary_generator


# ─── Endpoints ────────────────────────────────────────────


@router.get("/dates")
async def archive_dates(
    year: int = Query(default=None, description="Filter by year"),
    month: int = Query(default=None, description="Filter by month"),
):
    """List available archive dates with file counts."""
    try:
        engine = _get_archive_engine()
        dates = engine.get_archive_dates(year=year, month=month)
        return {
            "dates": dates,
            "total": len(dates),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/dates error: {e}")
        return {"dates": [], "total": 0, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/daily/{date_str}")
async def archive_daily(date_str: str):
    """All files for a specific date."""
    try:
        engine = _get_archive_engine()
        data = engine.get_archive_by_date(date_str)
        return {
            **data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/daily/{date_str} error: {e}")
        return {"date": date_str, "files": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/daily/{date_str}/{filename}")
async def archive_file(date_str: str, filename: str):
    """Specific file content for a date."""
    try:
        engine = _get_archive_engine()
        data = engine.get_archive_file(date_str, filename)
        return {
            "date": date_str,
            "filename": filename,
            "content": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/daily/{date_str}/{filename} error: {e}")
        return {"date": date_str, "filename": filename, "content": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/monthly-summary")
async def archive_monthly_summary(
    year: int = Query(default=None, description="Year"),
    month: int = Query(default=None, description="Month"),
):
    """Monthly aggregated stats."""
    try:
        y = year or date.today().year
        m = month or date.today().month
        engine = _get_archive_engine()
        summary = engine.get_archive_summary(y, m)
        return {
            **summary,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/monthly-summary error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.post("/trigger")
async def archive_trigger():
    """Manually trigger daily archival."""
    try:
        engine = _get_archive_engine()
        result = engine.archive_daily()
        return {
            **result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/trigger error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/daily-summary/{date_str}")
async def archive_daily_summary(date_str: str):
    """Daily summary report for a specific date."""
    try:
        # First try to read from archive
        engine = _get_archive_engine()
        archive_data = engine.get_archive_file(date_str, "daily_summary.json")
        if "error" not in archive_data:
            return {
                "date": date_str,
                "summary": archive_data,
                "source": "archive",
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Generate fresh if not archived
        gen = _get_summary_generator()
        summary = gen.generate_daily_summary(date_str)
        return {
            "date": date_str,
            "summary": summary,
            "source": "generated",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/daily-summary/{date_str} error: {e}")
        return {"date": date_str, "summary": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/daily-summary/latest")
async def archive_daily_summary_latest():
    """Most recent daily summary."""
    try:
        engine = _get_archive_engine()
        dates = engine.get_archive_dates()
        if dates:
            latest_date = dates[-1]["date"]
            archive_data = engine.get_archive_file(latest_date, "daily_summary.json")
            if "error" not in archive_data:
                return {
                    "date": latest_date,
                    "summary": archive_data,
                    "source": "archive",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        # Generate for today if no archive exists
        gen = _get_summary_generator()
        today = date.today().isoformat()
        summary = gen.generate_daily_summary(today)
        return {
            "date": today,
            "summary": summary,
            "source": "generated",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"archive/daily-summary/latest error: {e}")
        return {"summary": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}
