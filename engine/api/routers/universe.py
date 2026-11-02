"""
Universe router — OPENBB, ETF tabs
Wraps: UniverseEngine, openbb_data functions (direct, no class wrapper)
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.universe")
router = APIRouter()

_universe = None


def _get_universe():
    global _universe
    if _universe is None:
        from engine.data.universe_engine import UniverseEngine
        _universe = UniverseEngine()
        _universe.load_universe()
    return _universe



# OpenBB functions imported directly in each endpoint — no class wrapper


# ─── Universe ──────────────────────────────────────────────

@router.get("/securities")
async def universe_securities(
    sector: str = Query(None),
    limit: int = Query(100, ge=1, le=2000),
):
    """Universe securities with optional sector filter."""
    try:
        uni = _get_universe()
        if sector:
            secs = uni.get_by_sector(sector)
        else:
            secs = uni.get_all()

        result = []
        for s in secs[:limit]:
            result.append({
                "ticker": s.ticker if hasattr(s, "ticker") else "",
                "name": s.name if hasattr(s, "name") else "",
                "sector": s.sector if hasattr(s, "sector") else "",
                "gics_sector": s.gics_sector if hasattr(s, "gics_sector") else "",
                "industry_group": s.industry_group if hasattr(s, "industry_group") else "",
                "market_cap": s.market_cap if hasattr(s, "market_cap") else 0,
                "pe_ratio": s.pe_ratio if hasattr(s, "pe_ratio") else 0,
                "beta": s.beta if hasattr(s, "beta") else 0,
                "roe": s.roe if hasattr(s, "roe") else 0,
                "sharpe_12m": s.sharpe_12m if hasattr(s, "sharpe_12m") else 0,
                "momentum_3m": s.momentum_3m if hasattr(s, "momentum_3m") else 0,
                "quality_tier": s.quality_tier if hasattr(s, "quality_tier") else "?",
                "is_tradeable": s.is_fallen_angel if hasattr(s, "is_fallen_angel") else False,
                "options_eligible": s.options_eligible if hasattr(s, "options_eligible") else False,
            })

        return {
            "securities": result,
            "total": uni.size(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"universe/securities error: {e}")
        return {"securities": [], "error": str(e)}


@router.get("/sectors")
async def universe_sectors():
    """Sector list with counts and momentum."""
    try:
        uni = _get_universe()
        counts = uni.get_sector_counts()
        momentum = uni.get_sector_momentum()

        sectors = []
        for sector, count in counts.items():
            sectors.append({
                "sector": sector,
                "count": count,
                "momentum": momentum.get(sector, 0),
            })

        return {"sectors": sectors, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/sectors error: {e}")
        return {"sectors": [], "error": str(e)}


@router.get("/rv-pairs")
async def universe_rv_pairs():
    """Relative value pair candidates."""
    try:
        uni = _get_universe()
        pairs = uni.scan_rv_pairs()
        return {"pairs": pairs, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/rv-pairs error: {e}")
        return {"pairs": [], "error": str(e)}


@router.get("/morning-scan")
async def universe_morning_scan():
    """Daily universe scan: momentum, volume anomalies, sector flows."""
    try:
        uni = _get_universe()
        scan = uni.run_morning_scan()
        return {**(scan if isinstance(scan, dict) else {"data": str(scan)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/morning-scan error: {e}")
        return {"error": str(e)}


# ─── OpenBB data (direct function calls — no class indirection) ────

@router.get("/openbb/search")
async def openbb_search(query: str = Query(..., min_length=1)):
    """Search ALL securities via OpenBB (not just our universe)."""
    try:
        from engine.data.openbb_data import search_equities
        df = search_equities(query=query)
        if df.empty:
            return {"results": [], "timestamp": datetime.utcnow().isoformat()}
        records = df.head(50).to_dict(orient="records")
        return {"results": records, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/search error: {e}")
        return {"results": [], "error": str(e)}


@router.get("/openbb/quote")
async def openbb_quote(ticker: str = Query(...)):
    """Get live quote via OpenBB."""
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
        df = get_prices(ticker, start=start, end=end)
        if df.empty:
            return {"ticker": ticker, "error": "No price data", "timestamp": datetime.utcnow().isoformat()}

        # Extract latest price
        if hasattr(df.columns, "levels"):  # MultiIndex
            close_col = "Close" if "Close" in df.columns.get_level_values(0) else "Adj Close"
            last_row = df[close_col].iloc[-1]
            price = float(last_row.iloc[0]) if hasattr(last_row, "iloc") else float(last_row)
        else:
            price = float(df.iloc[-1].get("Close", df.iloc[-1].get("close", 0)))

        prev = float(df.iloc[-2].values[0]) if len(df) >= 2 else price
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"universe/openbb/quote error: {e}")
        return {"ticker": ticker, "error": str(e)}


@router.get("/openbb/historical")
async def openbb_historical(
    ticker: str = Query(...),
    days: int = Query(252, ge=5, le=1260),
):
    """Get historical OHLCV via OpenBB."""
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = get_prices(ticker, start=start, end=end)
        if df.empty:
            return {"data": [], "ticker": ticker, "error": "No data", "timestamp": datetime.utcnow().isoformat()}

        # Flatten to records
        if hasattr(df.columns, "levels"):  # MultiIndex
            flat = {}
            for field in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                if field in df.columns.get_level_values(0):
                    flat[field.lower()] = df[field].iloc[:, 0] if df[field].ndim > 1 else df[field]
            import pandas as pd
            flat_df = pd.DataFrame(flat, index=df.index)
        else:
            flat_df = df

        records = flat_df.reset_index().to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
                elif hasattr(v, "item"):
                    r[k] = v.item()  # numpy scalar → python
        return {"data": records, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/historical error: {e}")
        return {"data": [], "error": str(e)}


@router.get("/openbb/fred")
async def openbb_fred(series_id: str = Query(...)):
    """Get FRED economic data series via OpenBB."""
    try:
        from engine.data.openbb_data import get_fred_series
        df = get_fred_series(series_id)
        if hasattr(df, "empty") and df.empty:
            return {"data": [], "series_id": series_id, "error": "No data", "timestamp": datetime.utcnow().isoformat()}
        if hasattr(df, "to_dict"):
            records = df.reset_index().to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
                    elif hasattr(v, "item"):
                        r[k] = v.item()
            return {"data": records, "series_id": series_id, "timestamp": datetime.utcnow().isoformat()}
        return {"data": df if isinstance(df, (list, dict)) else str(df), "series_id": series_id, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/fred error: {e}")
        return {"data": [], "error": str(e)}


@router.get("/openbb/fundamentals")
async def openbb_fundamentals(ticker: str = Query(...)):
    """Get fundamental data for a ticker via OpenBB."""
    try:
        from engine.data.openbb_data import get_fundamentals
        data = get_fundamentals(ticker)
        if not data:
            return {"ticker": ticker, "error": "No fundamental data", "timestamp": datetime.utcnow().isoformat()}
        return {"ticker": ticker, "fundamentals": data, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/fundamentals error: {e}")
        return {"ticker": ticker, "error": str(e)}


@router.get("/openbb/quotes")
async def openbb_quotes(tickers: str = Query(..., description="Comma-separated tickers")):
    """Bulk quotes via OpenBB/FMP, falling back to IBKRBroker prices."""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            return {"quotes": [], "timestamp": datetime.utcnow().isoformat()}

        results = []

        # Try OpenBB/FMP bulk quote first
        from engine.data.openbb_data import get_quote
        raw = get_quote(ticker_list)
        found = set()
        if raw:
            for r in raw:
                sym = r.get("symbol", r.get("ticker", ""))
                if sym:
                    found.add(sym)
                    results.append({
                        "ticker": sym,
                        "price": r.get("price", r.get("last_price", 0)),
                        "change": r.get("change", 0),
                        "change_pct": r.get("change_percent", r.get("change_pct", 0)),
                        "volume": r.get("volume", 0),
                        "market_cap": r.get("market_cap", 0),
                        "source": "openbb",
                    })

        # Fallback to IBKRBroker for any missing tickers
        missing = [t for t in ticker_list if t not in found]
        if missing:
            try:
                from engine.data.openbb_data import get_prices
                from datetime import timedelta
                end = datetime.utcnow().strftime("%Y-%m-%d")
                start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
                df = get_prices(missing, start=start, end=end)
                if hasattr(df, "empty") and not df.empty:
                    import pandas as pd
                    if hasattr(df.columns, "levels"):
                        for t in missing:
                            try:
                                close_col = "Close" if "Close" in df.columns.get_level_values(0) else "Adj Close"
                                series = df[close_col][t] if t in df[close_col].columns else None
                                if series is not None and len(series) > 0:
                                    price = float(series.iloc[-1])
                                    prev = float(series.iloc[-2]) if len(series) > 1 else price
                                    results.append({
                                        "ticker": t,
                                        "price": round(price, 2),
                                        "change": round(price - prev, 2),
                                        "change_pct": round((price - prev) / prev * 100, 2) if prev else 0,
                                        "volume": 0,
                                        "market_cap": 0,
                                        "source": "ibkr" if "ibkr" in str(type(df)).lower() else "fmp",
                                    })
                                    found.add(t)
                            except Exception:
                                continue
                    else:
                        for t in missing:
                            try:
                                price = float(df.iloc[-1].get("Close", df.iloc[-1].get("close", 0)))
                                prev = float(df.iloc[-2].get("Close", df.iloc[-2].get("close", price))) if len(df) > 1 else price
                                results.append({
                                    "ticker": t,
                                    "price": round(price, 2),
                                    "change": round(price - prev, 2),
                                    "change_pct": round((price - prev) / prev * 100, 2) if prev else 0,
                                    "volume": 0,
                                    "market_cap": 0,
                                    "source": "price_history",
                                })
                                found.add(t)
                            except Exception:
                                continue
            except Exception as e:
                logger.debug(f"Bulk quote fallback failed: {e}")

        return {"quotes": results, "source": "openbb+ibkr", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/quotes error: {e}")
        return {"quotes": [], "error": str(e)}
