"""
Fixed Income router — Tab 18 Fixed Income Dashboard
Sources: FixedIncomeEngine → OpenBB FRED (treasury yields, credit spreads),
         MacroEngine (yield curve analysis, credit pulse), broker positions.

Mounts at: /api/engine/fixed-income
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.fixed-income")
router = APIRouter()

_fi_engine = None


def _get_fi():
    global _fi_engine
    if _fi_engine is None:
        from engine.signals.fixed_income_engine import FixedIncomeEngine
        _fi_engine = FixedIncomeEngine()
    return _fi_engine


# ─── Endpoints ─────────────────────────────────────────────

@router.get("/summary")
async def fi_summary():
    """Portfolio-level FI summary: exposure, duration, yield, DV01."""
    try:
        fi = _get_fi()
        data = fi.get_summary()
        return {**data, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/summary error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/holdings")
async def fi_holdings():
    """Bond/FI ETF holdings from broker positions."""
    try:
        fi = _get_fi()
        holdings = fi.get_holdings()
        return {"holdings": holdings, "count": len(holdings), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/holdings error: {e}")
        return {"holdings": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/yield-curve")
async def fi_yield_curve():
    """Real treasury yield curve across 11 tenors from FRED."""
    try:
        fi = _get_fi()
        curve = fi.get_yield_curve()
        return {"curve": curve, "count": len(curve), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/yield-curve error: {e}")
        return {"curve": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/credit-quality")
async def fi_credit_quality():
    """Credit quality distribution derived from market spreads."""
    try:
        fi = _get_fi()
        quality = fi.get_credit_quality()
        return {"quality": quality, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/credit-quality error: {e}")
        return {"quality": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/duration-ladder")
async def fi_duration_ladder():
    """Duration bucket distribution (DV01 by maturity bucket)."""
    try:
        fi = _get_fi()
        ladder = fi.get_duration_ladder()
        return {"ladder": ladder, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/duration-ladder error: {e}")
        return {"ladder": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/spread-history")
async def fi_spread_history(days: int = Query(default=90, ge=7, le=365)):
    """Historical IG and HY OAS spread data from FRED."""
    try:
        fi = _get_fi()
        data = fi.get_spread_history(days=days)
        return {**data, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/spread-history error: {e}")
        return {"data": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}
