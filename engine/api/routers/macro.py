"""
Macro router — MACRO, VELOCITY, WRAP tabs
Wraps: MacroEngine, FedLiquidityPlumbing, MarketWrapGenerator
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.macro")
router = APIRouter()

_macro = None
_fed = None
_wrap = None


def _get_macro():
    global _macro
    if _macro is None:
        from engine.signals.macro_engine import MacroEngine
        _macro = MacroEngine()
    return _macro


def _get_fed():
    global _fed
    if _fed is None:
        from engine.signals.fed_liquidity_plumbing import FedLiquidityPlumbing
        _fed = FedLiquidityPlumbing()
    return _fed


def _get_wrap():
    global _wrap
    if _wrap is None:
        from engine.monitoring.market_wrap import MarketWrapGenerator
        _wrap = MarketWrapGenerator()
    return _wrap


# ─── MACRO tab endpoints ───────────────────────────────────

@router.get("/snapshot")
async def macro_snapshot():
    """Full macro snapshot: regime, VIX, yields, credit, sectors."""
    try:
        macro = _get_macro()
        snap = macro.get_snapshot()

        return {
            "regime": snap.regime if hasattr(snap, "regime") else "UNKNOWN",
            "vix": snap.vix if hasattr(snap, "vix") else 0,
            "spy_return_1m": snap.spy_return_1m if hasattr(snap, "spy_return_1m") else 0,
            "spy_return_3m": snap.spy_return_3m if hasattr(snap, "spy_return_3m") else 0,
            "yield_10y": snap.yield_10y if hasattr(snap, "yield_10y") else 0,
            "yield_2y": snap.yield_2y if hasattr(snap, "yield_2y") else 0,
            "yield_spread": snap.yield_spread if hasattr(snap, "yield_spread") else 0,
            "credit_spread": snap.credit_spread if hasattr(snap, "credit_spread") else 0,
            "gold_momentum": snap.gold_momentum if hasattr(snap, "gold_momentum") else 0,
            "sector_rankings": snap.sector_rankings if hasattr(snap, "sector_rankings") else {},
            "gmtf_score": snap.gmtf_score if hasattr(snap, "gmtf_score") else 0,
            "money_velocity_signal": snap.money_velocity_signal if hasattr(snap, "money_velocity_signal") else 0,
            "cube_regime": snap.cube_regime if hasattr(snap, "cube_regime") else "UNKNOWN",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/snapshot error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/features")
async def macro_features():
    """50+ macro features for ML models."""
    try:
        macro = _get_macro()
        features = macro.get_macro_features()
        return {"features": features, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/features error: {e}")
        return {"features": {}, "error": str(e)}


@router.get("/yield-curve")
async def macro_yield_curve():
    """Yield curve analysis: 2s10s, 3m10y, term premium."""
    try:
        macro = _get_macro()
        yc = macro.get_yield_curve_analysis()
        return {**yc, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/yield-curve error: {e}")
        return {"error": str(e)}


@router.get("/credit-pulse")
async def macro_credit_pulse():
    """Credit spread monitoring: HY/IG differential."""
    try:
        macro = _get_macro()
        cp = macro.get_credit_pulse()
        return {**cp, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/credit-pulse error: {e}")
        return {"error": str(e)}


@router.get("/rotation")
async def macro_rotation():
    """GICS sector rotation signals."""
    try:
        macro = _get_macro()
        rot = macro.get_rotation_signals()
        return {**rot, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/rotation error: {e}")
        return {"error": str(e)}


@router.get("/regime-transition")
async def macro_regime_transition():
    """Latest regime transition detection."""
    try:
        macro = _get_macro()
        rt = macro.get_regime_transition()
        return {**rt, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/regime-transition error: {e}")
        return {"error": str(e)}


@router.get("/monetary-tension")
async def macro_monetary_tension():
    """SDR-weighted monetary tension index."""
    try:
        macro = _get_macro()
        mt = macro.get_monetary_tension()
        return {**mt, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/monetary-tension error: {e}")
        return {"error": str(e)}


# ─── VELOCITY tab endpoints ────────────────────────────────

@router.get("/velocity")
async def macro_velocity():
    """Money velocity state: V = GDP/M2, credit impulse, SOFR."""
    try:
        macro = _get_macro()
        mv = macro.get_money_velocity_state()
        return {**mv, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/velocity error: {e}")
        return {"error": str(e)}


@router.get("/velocity-regime")
async def macro_velocity_regime():
    """Velocity engine regime classification."""
    try:
        macro = _get_macro()
        vr = macro.get_velocity_regime()
        return {**vr, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/velocity-regime error: {e}")
        return {"error": str(e)}


@router.get("/fed-balance-sheet")
async def fed_balance_sheet():
    """Fed balance sheet: WALCL, SOMA, reserves, ON-RRP, TGA."""
    try:
        fed = _get_fed()
        fed.update()
        bs = fed._balance_sheet if hasattr(fed, "_balance_sheet") else None
        if bs is None:
            return {"status": "no_data", "timestamp": datetime.utcnow().isoformat()}

        return {
            "walcl": bs.walcl if hasattr(bs, "walcl") else 0,
            "soma_treasuries": bs.soma_treasuries if hasattr(bs, "soma_treasuries") else 0,
            "soma_mbs": bs.soma_mbs if hasattr(bs, "soma_mbs") else 0,
            "reserves": bs.reserves if hasattr(bs, "reserves") else 0,
            "excess_reserves": bs.excess_reserves if hasattr(bs, "excess_reserves") else 0,
            "on_rrp": bs.on_rrp if hasattr(bs, "on_rrp") else 0,
            "tga": bs.tga if hasattr(bs, "tga") else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/fed-balance-sheet error: {e}")
        return {"error": str(e)}


@router.get("/reserves-flow")
async def reserves_flow():
    """Reserve distribution cascade: Fed → PD → GSIB → Shadow → Market."""
    try:
        fed = _get_fed()
        dist = fed.get_reserve_distribution()
        return {
            "fed_to_pd": dist.fed_to_pd if hasattr(dist, "fed_to_pd") else 0,
            "pd_to_gsib": dist.pd_to_gsib if hasattr(dist, "pd_to_gsib") else 0,
            "gsib_to_shadow": dist.gsib_to_shadow if hasattr(dist, "gsib_to_shadow") else 0,
            "shadow_to_market": dist.shadow_to_market if hasattr(dist, "shadow_to_market") else 0,
            "net_market_liquidity": dist.net_market_liquidity if hasattr(dist, "net_market_liquidity") else 0,
            "bottleneck": dist.bottleneck if hasattr(dist, "bottleneck") else "none",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/reserves-flow error: {e}")
        return {"error": str(e)}


@router.get("/liquidity-score")
async def liquidity_score():
    """Aggregate L(t) score in [-1, +1]."""
    try:
        fed = _get_fed()
        score = fed.get_liquidity_score()
        regime = fed.get_liquidity_regime()
        return {
            "score": score,
            "regime": regime.value if hasattr(regime, "value") else str(regime),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/liquidity-score error: {e}")
        return {"error": str(e)}


@router.get("/sector-flows")
async def sector_flows():
    """Sector allocation from money flow analysis."""
    try:
        fed = _get_fed()
        alloc = fed.get_sector_flow_allocation()
        return {
            "sector_scores": alloc.sector_scores if hasattr(alloc, "sector_scores") else {},
            "sector_weights": alloc.sector_weights if hasattr(alloc, "sector_weights") else {},
            "overweight": alloc.overweight if hasattr(alloc, "overweight") else [],
            "underweight": alloc.underweight if hasattr(alloc, "underweight") else [],
            "flow_regime": alloc.flow_regime if hasattr(alloc, "flow_regime") else "UNKNOWN",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/sector-flows error: {e}")
        return {"error": str(e)}


@router.get("/credit-impulse")
async def credit_impulse():
    """Credit impulse: rate of change of new credit."""
    try:
        fed = _get_fed()
        ci = fed.get_credit_impulse()
        return {
            "impulse": ci.impulse if hasattr(ci, "impulse") else 0,
            "regime": ci.regime if hasattr(ci, "regime") else "UNKNOWN",
            "z_score": ci.z_score if hasattr(ci, "z_score") else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/credit-impulse error: {e}")
        return {"error": str(e)}


@router.get("/drain-warning")
async def drain_warning():
    """Early warning of liquidity drain."""
    try:
        fed = _get_fed()
        dw = fed.get_drain_warning()
        return {
            "warning_level": dw.warning_level if hasattr(dw, "warning_level") else 0,
            "triggers": dw.triggers if hasattr(dw, "triggers") else [],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/drain-warning error: {e}")
        return {"error": str(e)}


# ─── G10 Macro (per-country FRED data) ─────────────────────

@router.get("/g10")
async def macro_g10():
    """G10 country economic indicators via MacroEngine → OpenBB FRED.

    Fits in system: L2 MacroEngine uses these for GMTF monetary tension framework,
    SDR-weighted currency tensions, and regime transition detection.
    """
    try:
        from engine.data.openbb_data import get_fred_series, get_adj_close
        from datetime import timedelta

        # FRED series for key G10 indicators
        G10_FRED = {
            "US": {"gdp": "A191RL1Q225SBEA", "cpi": "CPIAUCSL", "unemp": "UNRATE", "rate": "FEDFUNDS", "y10": "GS10", "currency": "USD", "flag": "🇺🇸", "country": "United States"},
            "UK": {"gdp": "CLVMNACSCAB1GQUK", "cpi": "GBRCPIALLMINMEI", "unemp": "LRHUTTTTGBM156S", "rate": "INTDSRGBM193N", "y10": "IRLTLT01GBM156N", "currency": "GBP", "flag": "🇬🇧", "country": "United Kingdom"},
            "EUR": {"gdp": "CLVMNACSCAB1GQEA19", "cpi": "EA19CPALTT01GYM", "unemp": "LRHUTTTTEZM156S", "rate": "ECBDFR", "y10": "IRLTLT01DEM156N", "currency": "EUR", "flag": "🇪🇺", "country": "Eurozone"},
            "JP": {"gdp": "JPNRGDPEXP", "cpi": "JPNCPIALLMINMEI", "unemp": "LRHUTTTTJPM156S", "rate": "INTDSRJPM193N", "y10": "IRLTLT01JPM156N", "currency": "JPY", "flag": "🇯🇵", "country": "Japan"},
            "CA": {"gdp": "NGDPRSAXDCCAQ", "cpi": "CANCPIALLMINMEI", "unemp": "LRHUTTTTCAM156S", "rate": "INTDSRCAM193N", "y10": "IRLTLT01CAM156N", "currency": "CAD", "flag": "🇨🇦", "country": "Canada"},
            "AU": {"gdp": "AUSGDPDEFQISMEI", "cpi": "AUSCPIALLQINMEI", "unemp": "LRHUTTTTAUM156S", "rate": "INTDSRAUM193N", "y10": "IRLTLT01AUM156N", "currency": "AUD", "flag": "🇦🇺", "country": "Australia"},
            "NZ": {"gdp": "NZLGDPNQDSMEI", "cpi": "NZLCPIALLQINMEI", "unemp": "LRHUTTTTNZM156S", "rate": "INTDSRNZM193N", "y10": "IRLTLT01NZM156N", "currency": "NZD", "flag": "🇳🇿", "country": "New Zealand"},
            "CH": {"gdp": "CHEGDPDEFQISMEI", "cpi": "CHECPIALLMINMEI", "unemp": "LRHUTTTTCHM156S", "rate": "INTDSRCHM193N", "y10": "IRLTLT01CHM156N", "currency": "CHF", "flag": "🇨🇭", "country": "Switzerland"},
            "SE": {"gdp": "SWERGDPDEFQISMEI", "cpi": "SWECPIALLMINMEI", "unemp": "LRHUTTTTSEM156S", "rate": "INTDSRSEM193N", "y10": "IRLTLT01SEM156N", "currency": "SEK", "flag": "🇸🇪", "country": "Sweden"},
            "NO": {"gdp": "NORGDPNQDSMEI", "cpi": "NORCPIALLQINMEI", "unemp": "LRHUTTTTNOM156S", "rate": "INTDSRNOM193N", "y10": "IRLTLT01NOM156N", "currency": "NOK", "flag": "🇳🇴", "country": "Norway"},
        }

        countries = []
        for code, cfg in G10_FRED.items():
            row = {"code": code, "flag": cfg["flag"], "country": cfg["country"], "currency": cfg["currency"]}

            # Fetch each indicator — get latest value
            for key, series_id in cfg.items():
                if key in ("flag", "country", "currency"):
                    continue
                try:
                    df = get_fred_series(series_id)
                    if hasattr(df, "empty") and not df.empty:
                        val = float(df.iloc[-1].iloc[0]) if df.ndim > 1 else float(df.iloc[-1])
                        row[key] = round(val, 2)
                    else:
                        row[key] = None
                except Exception:
                    row[key] = None

            countries.append(row)

        return {"countries": countries, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/g10 error: {e}")
        return {"countries": [], "error": str(e)}


# ─── Economic calendar ─────────────────────────────────────

@router.get("/calendar")
async def economic_calendar():
    """Upcoming economic events via OpenBB."""
    try:
        from engine.data.openbb_data import get_economic_calendar
        df = get_economic_calendar()
        if hasattr(df, "empty") and df.empty:
            return {"events": [], "timestamp": datetime.utcnow().isoformat()}
        if hasattr(df, "to_dict"):
            records = df.head(20).to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return {"events": records, "timestamp": datetime.utcnow().isoformat()}
        return {"events": [], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/calendar error: {e}")
        return {"events": [], "error": str(e)}


@router.get("/news")
async def macro_news():
    """Market news routed through the NewsEngine.

    The NewsEngine is the single news source for the entire platform.
    Primary: newsfilter.io WebSocket (10,000+ sources, real-time)
    Fallback: OpenBB news (Tiingo free tier — no key configured but
    included in OpenBB's free access; Benzinga NOT available, no key)

    This endpoint feeds the WRAP tab news sidebar. The same NewsEngine
    singleton is shared with EventDrivenEngine and CVREngine so all
    news consumers see the same warmed feed.
    """
    try:
        from engine.signals.news_engine import NewsEngine

        # Use module-level singleton to share with signals router
        global _news_engine
        if "_news_engine" not in globals() or _news_engine is None:
            _news_engine = NewsEngine()

        feed = _news_engine.get_live_feed(limit=15)
        news = []
        for item in feed:
            news.append({
                "time": getattr(item, "timestamp", "") if hasattr(item, "timestamp") else str(getattr(item, "published", "")),
                "headline": getattr(item, "headline", "") or getattr(item, "title", ""),
                "source": getattr(item, "source", "NewsEngine"),
                "tickers": getattr(item, "tickers", []),
                "sentiment": getattr(item, "sentiment", "neutral"),
                "url": getattr(item, "url", ""),
            })

        return {"news": news, "source": "news_engine", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/news error: {e}")
        return {"news": [], "error": str(e)}


# ─── Historical time series for Macro charts ──────────────

@router.get("/spread-history")
async def macro_spread_history():
    """2s10s yield spread time series from FRED (DGS10 - DGS2)."""
    try:
        from engine.data.openbb_data import get_fred_series
        from datetime import timedelta

        start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
        df_10y = get_fred_series("DGS10", start_date=start)
        df_2y = get_fred_series("DGS2", start_date=start)

        data = []
        if hasattr(df_10y, "empty") and not df_10y.empty and hasattr(df_2y, "empty") and not df_2y.empty:
            # Align on common dates
            common = df_10y.index.intersection(df_2y.index)
            for i, dt in enumerate(common[-30:]):
                val_10 = float(df_10y.loc[dt].iloc[0]) if df_10y.ndim > 1 else float(df_10y.loc[dt])
                val_2 = float(df_2y.loc[dt].iloc[0]) if df_2y.ndim > 1 else float(df_2y.loc[dt])
                spread = round(val_10 - val_2, 4)
                data.append({"day": i + 1, "val": spread, "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)})

        return {"data": data, "series": "2s10s", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/spread-history error: {e}")
        return {"data": [], "error": str(e)}


@router.get("/vix-history")
async def macro_vix_history():
    """VIX index time series from FRED (VIXCLS)."""
    try:
        from engine.data.openbb_data import get_fred_series
        from datetime import timedelta

        start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
        df = get_fred_series("VIXCLS", start_date=start)

        data = []
        if hasattr(df, "empty") and not df.empty:
            rows = df.tail(30)
            for i, (dt, row) in enumerate(rows.iterrows()):
                val = float(row.iloc[0]) if df.ndim > 1 else float(row)
                data.append({"day": i + 1, "val": round(val, 2), "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)})

        return {"data": data, "series": "VIX", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/vix-history error: {e}")
        return {"data": [], "error": str(e)}


@router.get("/dxy-history")
async def macro_dxy_history():
    """Trade-weighted US Dollar Index from FRED (DTWEXBGS)."""
    try:
        from engine.data.openbb_data import get_fred_series
        from datetime import timedelta

        start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
        df = get_fred_series("DTWEXBGS", start_date=start)

        data = []
        if hasattr(df, "empty") and not df.empty:
            rows = df.tail(30)
            for i, (dt, row) in enumerate(rows.iterrows()):
                val = float(row.iloc[0]) if df.ndim > 1 else float(row)
                data.append({"day": i + 1, "val": round(val, 2), "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)})

        return {"data": data, "series": "DXY", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"macro/dxy-history error: {e}")
        return {"data": [], "error": str(e)}


# ─── WRAP tab endpoints ────────────────────────────────────

@router.get("/wrap")
async def market_wrap():
    """Full market wrap report."""
    try:
        wrap_gen = _get_wrap()
        report = wrap_gen.generate()

        indices = []
        for idx in (report.indices or []):
            indices.append({
                "ticker": idx.ticker if hasattr(idx, "ticker") else "",
                "name": idx.name if hasattr(idx, "name") else "",
                "last_price": idx.last_price if hasattr(idx, "last_price") else 0,
                "change_1d": idx.change_1d if hasattr(idx, "change_1d") else 0,
                "change_1w": idx.change_1w if hasattr(idx, "change_1w") else 0,
                "change_1m": idx.change_1m if hasattr(idx, "change_1m") else 0,
                "change_ytd": idx.change_ytd if hasattr(idx, "change_ytd") else 0,
            })

        sectors = []
        for sec in (report.sectors or []):
            sectors.append({
                "sector": sec.sector if hasattr(sec, "sector") else "",
                "etf": sec.etf if hasattr(sec, "etf") else "",
                "return_1d": sec.return_1d if hasattr(sec, "return_1d") else 0,
                "return_1w": sec.return_1w if hasattr(sec, "return_1w") else 0,
                "return_1m": sec.return_1m if hasattr(sec, "return_1m") else 0,
                "relative_strength": sec.relative_strength if hasattr(sec, "relative_strength") else 0,
            })

        breadth = {}
        if report.breadth:
            b = report.breadth
            breadth = {
                "advancing": b.advancing if hasattr(b, "advancing") else 0,
                "declining": b.declining if hasattr(b, "declining") else 0,
                "advance_decline_ratio": b.advance_decline_ratio if hasattr(b, "advance_decline_ratio") else 0,
                "breadth_thrust": b.breadth_thrust if hasattr(b, "breadth_thrust") else 0,
            }

        macro_summary = {}
        if report.macro:
            m = report.macro
            macro_summary = {
                "yield_10y": m.yield_10y if hasattr(m, "yield_10y") else 0,
                "yield_2y": m.yield_2y if hasattr(m, "yield_2y") else 0,
                "yield_spread": m.yield_spread if hasattr(m, "yield_spread") else 0,
                "vix": m.vix if hasattr(m, "vix") else 0,
                "dxy": m.dxy if hasattr(m, "dxy") else 0,
                "gold": m.gold if hasattr(m, "gold") else 0,
                "oil": m.oil if hasattr(m, "oil") else 0,
            }

        return {
            "indices": indices,
            "sectors": sectors,
            "top_gainers": [{"ticker": g.ticker, "change": g.change_1d} for g in (report.top_gainers or [])] if hasattr(report, "top_gainers") else [],
            "top_losers": [{"ticker": l.ticker, "change": l.change_1d} for l in (report.top_losers or [])] if hasattr(report, "top_losers") else [],
            "breadth": breadth,
            "macro": macro_summary,
            "market_tone": report.market_tone if hasattr(report, "market_tone") else "neutral",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"macro/wrap error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
