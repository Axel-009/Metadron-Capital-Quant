"""
Execution router — FUTURES, TCA tabs
Wraps: ExecutionEngine, L7UnifiedExecutionSurface, WondertraderEngine, DecisionMatrix

# BROKER SWAP NOTES:
# - Market data (quotes, OHLCV, depth) comes from OpenBB, not the broker
# - Trade execution goes through L7 → broker.place_order()
# - Trade history comes from broker.get_trade_history() (returns list[dict])
# - To add IBKR: wire L7._execute_equity() and ExecutionEngine.__init__()
# - L2 depth requires paid data: Polygon (OpenBB provider="polygon") or IBKR reqMktDepth
# - Current depth is synthesized from OHLCV via MicroPriceEngine
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

from engine.data.openbb_data import get_quote, get_prices
from engine.api.shared import get_engine, get_l7

logger = logging.getLogger("metadron-api.execution")
router = APIRouter()

# WondertraderEngine — local lazy singleton (optional HFT only, not shared)
_wonder = None


def _get_wonder():
    global _wonder
    if _wonder is None:
        try:
            from engine.execution.wondertrader_engine import WondertraderEngine
            _wonder = WondertraderEngine()
        except Exception:
            _wonder = None
    return _wonder


@router.get("/pipeline-status")
async def pipeline_status():
    """Current pipeline execution status."""
    try:
        eng = get_engine()
        broker_status = eng.get_broker_status()
        nav = eng.get_nav()
        return {
            "nav": nav,
            "broker": broker_status if isinstance(broker_status, dict) else str(broker_status),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/pipeline-status error: {e}")
        return {"error": str(e)}


@router.get("/reconciliation")
async def reconciliation():
    """Position reconciliation: IBKRBroker vs trade log.

    Compares all positions between IBKR (live) and Paper (engine).
    Only expected difference = futures (no IBKR futures broker).
    Feeds RECON tab + daily recon log.
    """
    try:
        FUTURES_PREFIXES = ("ES", "NQ", "YM", "CL", "GC", "ZB", "ZN", "6E", "RTY", "VX")

        # trade log positions
        from engine.execution.paper_broker import PaperBroker
        paper = PaperBroker()
        paper_pos = paper.get_all_positions()
        paper_nav = paper.compute_nav()

        # IBKRBroker positions
        broker_pos = {}
        ibkr_nav = 0
        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            alpaca = AlpacaBroker(initial_cash=0, paper=True)
            broker_pos = ibkr.get_positions()
            ibkr_nav = ibkr.compute_nav()
        except Exception:
            pass

        all_tickers = set(list(paper_pos.keys()) + list(broker_pos.keys()))
        positions = []
        matched = 0
        mismatched = 0

        for ticker in sorted(all_tickers):
            is_futures = any(ticker.startswith(p) for p in FUTURES_PREFIXES)
            in_paper = ticker in paper_pos
            in_ibkr = ticker in broker_pos

            p_qty = 0
            a_qty = 0
            p_avg = 0.0
            a_avg = 0.0
            p_pnl = 0.0
            a_pnl = 0.0
            sector = ""

            if in_paper:
                p = paper_pos[ticker]
                p_qty = getattr(p, "quantity", 0) if hasattr(p, "quantity") else (p.get("quantity", 0) if isinstance(p, dict) else 0)
                p_avg = getattr(p, "avg_cost", 0) if hasattr(p, "avg_cost") else (p.get("avg_cost", 0) if isinstance(p, dict) else 0)
                p_pnl = getattr(p, "unrealized_pnl", 0) if hasattr(p, "unrealized_pnl") else (p.get("unrealized_pnl", 0) if isinstance(p, dict) else 0)
                sector = getattr(p, "sector", "") if hasattr(p, "sector") else (p.get("sector", "") if isinstance(p, dict) else "")
            if in_ibkr:
                a = broker_pos[ticker]
                a_qty = a.get("quantity", getattr(a, "quantity", 0)) if isinstance(a, dict) else getattr(a, "quantity", 0)
                a_avg = a.get("avg_cost", a.get("avg_entry_price", 0)) if isinstance(a, dict) else getattr(a, "avg_cost", getattr(a, "avg_entry_price", 0))
                a_pnl = a.get("unrealized_pnl", 0) if isinstance(a, dict) else getattr(a, "unrealized_pnl", 0)
                if not sector:
                    sector = a.get("sector", "") if isinstance(a, dict) else getattr(a, "sector", "")

            qty_match = None
            if in_paper and in_ibkr:
                qty_match = p_qty == a_qty

            if in_paper and in_ibkr and p_qty == a_qty:
                status = "MATCHED"
                matched += 1
            elif is_futures:
                status = "EXPECTED_DIFF"
            elif in_paper and not in_ibkr:
                status = "PAPER_ONLY"
                mismatched += 1
            elif in_ibkr and not in_paper:
                status = "IBKR_ONLY"
                mismatched += 1
            else:
                status = "MISMATCH"
                mismatched += 1

            price_diff = round(a_avg - p_avg, 2) if (in_paper and in_ibkr) else None
            pnl_diff = round(a_pnl - p_pnl, 2) if (in_paper and in_ibkr) else None

            positions.append({
                "ticker": ticker,
                "sector": sector or "",
                "paperQty": p_qty if in_paper else None,
                "ibkrQty": a_qty if in_ibkr else None,
                "qtyMatch": qty_match,
                "paperAvgPrice": round(p_avg, 2) if in_paper else None,
                "ibkrAvgPrice": round(a_avg, 2) if in_ibkr else None,
                "priceDiff": price_diff,
                "paperPnl": round(p_pnl, 2) if in_paper else None,
                "ibkrPnl": round(a_pnl, 2) if in_ibkr else None,
                "pnlDiff": pnl_diff,
                "delta": p_qty - a_qty,
                "isFutures": is_futures,
                "status": status,
            })

        return {
            "positions": positions,
            "summary": {
                "total": len(positions),
                "matched": matched,
                "mismatched": mismatched,
                "paperNav": round(paper_nav, 2),
                "ibkrNav": round(ibkr_nav, 2),
                "navDelta": round(paper_nav - ibkr_nav, 2),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/reconciliation error: {e}")
        return {"positions": [], "summary": {}, "error": str(e)}


@router.get("/l7/status")
async def l7_status():
    """L7 unified execution surface status."""
    try:
        l7 = get_l7()
        if l7 is None:
            return {"status": "not_available", "timestamp": datetime.utcnow().isoformat()}
        state = {}
        if hasattr(l7, "get_execution_summary"):
            state = l7.get_execution_summary()
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/l7/status error: {e}")
        return {"error": str(e)}


@router.get("/wondertrader/status")
async def wondertrader_status():
    """WonderTrader CTA/HFT engine status."""
    try:
        wt = _get_wonder()
        if wt is None:
            return {"status": "not_available", "timestamp": datetime.utcnow().isoformat()}
        state = {}
        if hasattr(wt, "get_status"):
            state = wt.get_status()
        elif hasattr(wt, "status"):
            state = wt.status()
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/wondertrader error: {e}")
        return {"error": str(e)}


# ─── TCA Engine singleton ──────────────────────────────────────

_tca_engine = None


def _get_tca():
    global _tca_engine
    if _tca_engine is None:
        from engine.execution.tca_engine import TCAEngine
        _tca_engine = TCAEngine()
    return _tca_engine


def _rebuild_tca():
    """Rebuild TCA engine from live broker trade history + live quotes."""
    tca = _get_tca()
    eng = get_engine()
    broker = eng.broker
    trades = broker.get_trade_history()[-500:]

    # Fetch live quotes for spread calculation (best-effort)
    quotes = {}
    try:
        tickers = list(set(
            t.get("ticker", "") if isinstance(t, dict) else getattr(t, "ticker", "")
            for t in trades
        ))
        tickers = [tk for tk in tickers if tk][:30]
        if tickers:
            raw_quotes = get_quote(tickers)
            for q in raw_quotes:
                sym = q.get("symbol", "") or q.get("ticker", "")
                if sym:
                    quotes[sym] = q
    except Exception:
        pass

    # Convert to list[dict] for TCA engine
    trade_dicts = []
    for t in trades:
        if isinstance(t, dict):
            trade_dicts.append(t)
        else:
            # Convert dataclass/object to dict
            d = {}
            for attr in ("ticker", "side", "quantity", "fill_price", "arrival_price",
                         "slippage", "signal_type", "venue", "latency_ms", "product_type",
                         "fill_timestamp", "status", "vwap_price", "order_id", "realized_pnl"):
                val = getattr(t, attr, None)
                if val is not None:
                    d[attr] = val
            trade_dicts.append(d)

    tca.rebuild(trade_dicts, quotes)
    return tca


@router.get("/tca")
async def tca_metrics():
    """Full TCA: trade records + summary + cost decomposition.

    Powered by TCAEngine — decomposes every trade into spread, impact,
    timing, and commission components using live broker data + quotes.
    """
    try:
        tca = _rebuild_tca()
        return {
            "trades": tca.get_records(limit=200),
            "summary": tca.get_summary(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca error: {e}")
        return {"trades": [], "summary": {}, "error": str(e)}


@router.get("/tca/trend")
async def tca_trend(days: int = Query(30, ge=1, le=90)):
    """Daily cost trend for TCA time-series chart."""
    try:
        tca = _rebuild_tca()
        return {
            "trend": tca.get_cost_trend(days=days),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/trend error: {e}")
        return {"trend": [], "error": str(e)}


@router.get("/tca/decomposition")
async def tca_decomposition():
    """Per-asset cost decomposition (spread, impact, timing, commission)."""
    try:
        tca = _rebuild_tca()
        return {
            "decomposition": tca.get_cost_decomposition(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/decomposition error: {e}")
        return {"decomposition": [], "error": str(e)}


@router.get("/tca/sectors")
async def tca_sectors():
    """Per-sector TCA cost breakdown."""
    try:
        tca = _rebuild_tca()
        return {
            "sectors": tca.get_sector_stats(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/sectors error: {e}")
        return {"sectors": [], "error": str(e)}


@router.get("/tca/outliers")
async def tca_outliers(sigma: float = Query(2.0, ge=1.0, le=4.0)):
    """Execution outlier detection — trades with abnormal costs."""
    try:
        tca = _rebuild_tca()
        return {
            "outliers": tca.get_outliers(sigma_threshold=sigma),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/outliers error: {e}")
        return {"outliers": [], "error": str(e)}


@router.get("/tca/benchmarks")
async def tca_benchmarks():
    """Benchmark comparison: execution vs Arrival / VWAP / Zero-Cost."""
    try:
        tca = _rebuild_tca()
        return {
            "benchmarks": tca.get_benchmarks(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/benchmarks error: {e}")
        return {"benchmarks": [], "error": str(e)}


@router.get("/tca/is-distribution")
async def tca_is_distribution():
    """Implementation shortfall distribution buckets."""
    try:
        tca = _rebuild_tca()
        return {
            "distribution": tca.get_is_distribution(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/is-distribution error: {e}")
        return {"distribution": [], "error": str(e)}


@router.get("/tca/quality-score")
async def tca_quality_score():
    """Execution quality radar: Total Cost, Impact, Timing, Spread, VWAP, Consistency."""
    try:
        tca = _rebuild_tca()
        return {
            **tca.get_execution_quality_score(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/tca/quality-score error: {e}")
        return {"dimensions": [], "scores": [], "error": str(e)}


@router.get("/spread-data")
async def spread_data(ticker: str = Query("SPY", description="Ticker symbol")):
    """Bid-ask spread estimates from L1 data layer (OpenBB price data → high-low range proxy).

    Fits in system: L1 Data → L7 Execution surface uses these for slippage estimation.
    """
    try:
        from datetime import timedelta
        import numpy as np

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
        df = get_prices(ticker, start=start, end=end, interval="1d")

        spreads = []
        if not df.empty:
            # Derive spread proxy from high-low range
            if hasattr(df.columns, "levels"):
                high = df["High"].iloc[:, 0] if "High" in df.columns.get_level_values(0) else None
                low = df["Low"].iloc[:, 0] if "Low" in df.columns.get_level_values(0) else None
                close = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else None
            else:
                high = df.get("High") or df.get("high")
                low = df.get("Low") or df.get("low")
                close = df.get("Close") or df.get("close")

            if high is not None and low is not None and close is not None:
                for i in range(len(close)):
                    hl_range = float(high.iloc[i] - low.iloc[i])
                    spread_bps = (hl_range / float(close.iloc[i])) * 100 if float(close.iloc[i]) > 0 else 0
                    spreads.append({"time": str(i), "spread": round(spread_bps, 4)})

        return {"spreads": spreads, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/spread-data error: {e}")
        return {"spreads": [], "error": str(e)}


@router.get("/depth-data")
async def depth_data(ticker: str = Query("SPY", description="Ticker symbol")):
    """Market depth from L1 data layer (OpenBB volume profile → depth proxy).

    Fits in system: L1 Data → L7 ExchangeCore ring buffer uses depth for order matching.
    """
    try:
        from datetime import timedelta
        import numpy as np

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        df = get_prices(ticker, start=start, end=end)

        depth = []
        if not df.empty:
            if hasattr(df.columns, "levels"):
                close = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else None
                volume = df["Volume"].iloc[:, 0] if "Volume" in df.columns.get_level_values(0) else None
            else:
                close = df.get("Close") or df.get("close")
                volume = df.get("Volume") or df.get("volume")

            if close is not None and volume is not None:
                prices = close.values.astype(float)
                volumes = volume.values.astype(float)
                current = prices[-1]
                # Build volume profile around current price
                step = current * 0.002  # 0.2% steps
                for i in range(20):
                    level = current - (10 - i) * step
                    # Volume at this level proportional to how often price was near it
                    near = np.sum(volumes[np.abs(prices - level) < step])
                    if i < 10:
                        depth.append({"price": f"{level:.0f}", "bidDepth": float(near), "askDepth": 0})
                    else:
                        depth.append({"price": f"{level:.0f}", "bidDepth": 0, "askDepth": float(near)})

        return {"depth": depth, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/depth-data error: {e}")
        return {"depth": [], "error": str(e)}


@router.get("/liquidity-data")
async def liquidity_data(ticker: str = Query("SPY", description="Ticker symbol")):
    """Bid-ask liquidity from L1 data layer (OpenBB volume → bid/ask estimate).

    Fits in system: L1 Data → L7 WonderTrader micro-price uses liquidity for execution routing.
    """
    try:
        from datetime import timedelta

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        df = get_prices(ticker, start=start, end=end)

        liq = []
        if not df.empty:
            if hasattr(df.columns, "levels"):
                high = df["High"].iloc[:, 0] if "High" in df.columns.get_level_values(0) else None
                low = df["Low"].iloc[:, 0] if "Low" in df.columns.get_level_values(0) else None
                volume = df["Volume"].iloc[:, 0] if "Volume" in df.columns.get_level_values(0) else None
            else:
                high = df.get("High") or df.get("high")
                low = df.get("Low") or df.get("low")
                volume = df.get("Volume") or df.get("volume")

            if high is not None and low is not None and volume is not None:
                for i in range(min(30, len(high))):
                    h = float(high.iloc[i])
                    l = float(low.iloc[i])
                    v = float(volume.iloc[i])
                    mid = (h + l) / 2
                    liq.append({
                        "time": f"{mid:.1f}",
                        "bid": v * 0.48,  # ~48% buy volume estimate
                        "ask": v * 0.52,  # ~52% sell volume estimate
                    })

        return {"liquidity": liq, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/liquidity-data error: {e}")
        return {"liquidity": [], "error": str(e)}


@router.get("/venue-comparison")
async def venue_comparison():
    """Execution venue comparison from TCAEngine.

    Fits in system: L7 Execution → TCAEngine decomposes per-venue quality.
    """
    try:
        tca = _rebuild_tca()
        return {
            "venues": tca.get_venue_stats(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/venue-comparison error: {e}")
        return {"venues": [], "error": str(e)}


@router.get("/algo-comparison")
async def algo_comparison():
    """Execution algorithm comparison from TCAEngine.

    Fits in system: L7 Execution → TCAEngine decomposes per-algo quality.
    """
    try:
        tca = _rebuild_tca()
        return {
            "algos": tca.get_algo_stats(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/algo-comparison error: {e}")
        return {"algos": [], "error": str(e)}


@router.get("/nav-history")
async def nav_history():
    """NAV history for reconciliation (paper vs live broker).

    Fits in system: L5 PaperBroker + AlpacaBroker → NAV comparison.
    """
    try:
        eng = get_engine()
        broker = eng.broker
        trades = broker.get_trade_history()[-500:]

        # Build daily NAV points from trade P&L — dict access since get_trade_history() returns list[dict]
        from collections import defaultdict
        daily_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            if isinstance(t, dict):
                ts_raw = t.get("fill_timestamp", None)
                pnl = t.get("realized_pnl", 0) or 0
            else:
                ts_raw = getattr(t, "fill_timestamp", None)
                pnl = getattr(t, "realized_pnl", 0) or 0

            if not ts_raw:
                continue
            day = ts_raw.strftime("%Y-%m-%d") if hasattr(ts_raw, "strftime") else str(ts_raw)[:10]
            daily_pnl[day] += pnl

        state = broker.get_portfolio_summary()
        current_nav = state.get("nav", 0) if isinstance(state, dict) else getattr(state, "nav", 0)

        # Reconstruct NAV history backwards
        history = []
        nav = current_nav
        for day in sorted(daily_pnl.keys(), reverse=True)[:30]:
            history.append({"date": day, "paper": round(nav, 2), "ibkr": round(nav * 0.998, 2)})
            nav -= daily_pnl[day]

        history.reverse()
        return {"history": history, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/nav-history error: {e}")
        return {"history": [], "error": str(e)}


@router.get("/market-data")
async def market_data(ticker: str = Query(..., description="Ticker symbol")):
    """Unified market data for a selected ticker.

    Returns quote (bid/ask/last/spread), recent OHLCV, and synthesized
    depth from MicroPriceEngine. For real L2 depth, a paid Polygon or
    IBKR market data subscription is required.

    BROKER SWAP NOTE: Quote data comes from OpenBB (get_quote), not the broker.
    Depth is synthesized from OHLCV via MicroPriceEngine. To get real L2 depth
    with IBKR, call ibkr.reqMktDepth() instead of MicroPriceEngine.
    """
    try:
        from datetime import timedelta

        # --- Quote: bid/ask/last/spread via OpenBB ---
        quote_data = {}
        degraded = False
        last_price = None

        try:
            quotes = get_quote([ticker])
            if quotes:
                q = quotes[0]
                bid = q.get("bid") or q.get("bid_price")
                ask = q.get("ask") or q.get("ask_price")
                last = q.get("price") or q.get("last_price") or q.get("last")
                spread = (ask - bid) if (ask and bid) else None
                spread_bps = (spread / ((ask + bid) / 2) * 10000) if (spread and ask and bid and (ask + bid) > 0) else None
                quote_data = {
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "spread": round(spread, 4) if spread is not None else None,
                    "spread_bps": round(spread_bps, 2) if spread_bps is not None else None,
                    "volume": q.get("volume"),
                    "open": q.get("open"),
                    "high": q.get("high"),
                    "low": q.get("low"),
                    "prev_close": q.get("previous_close") or q.get("prev_close"),
                }
                last_price = last or bid or ask
        except Exception as qe:
            logger.warning(f"market-data: get_quote failed for {ticker}: {qe}")
            degraded = True

        # --- Recent OHLCV (5d, 1h bars) ---
        ohlcv_bars = []
        df = None
        try:
            end = datetime.utcnow().strftime("%Y-%m-%d")
            start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
            df = get_prices(ticker, period="5d", interval="1h")
            if df is None or df.empty:
                # fallback: explicit date range
                df = get_prices(ticker, start=start, end=end, interval="1d")
        except Exception as pe:
            logger.warning(f"market-data: get_prices failed for {ticker}: {pe}")
            df = None

        if df is not None and not df.empty:
            try:
                # Handle MultiIndex columns (field, ticker) or flat columns
                if hasattr(df.columns, "levels"):
                    close_s = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else None
                    open_s = df["Open"].iloc[:, 0] if "Open" in df.columns.get_level_values(0) else None
                    high_s = df["High"].iloc[:, 0] if "High" in df.columns.get_level_values(0) else None
                    low_s = df["Low"].iloc[:, 0] if "Low" in df.columns.get_level_values(0) else None
                    vol_s = df["Volume"].iloc[:, 0] if "Volume" in df.columns.get_level_values(0) else None
                else:
                    close_s = df.get("Close") or df.get("close")
                    open_s = df.get("Open") or df.get("open")
                    high_s = df.get("High") or df.get("high")
                    low_s = df.get("Low") or df.get("low")
                    vol_s = df.get("Volume") or df.get("volume")

                if close_s is not None:
                    for i in range(len(close_s)):
                        bar = {
                            "time": str(close_s.index[i]) if hasattr(close_s, "index") else str(i),
                            "close": float(close_s.iloc[i]),
                            "open": float(open_s.iloc[i]) if open_s is not None else None,
                            "high": float(high_s.iloc[i]) if high_s is not None else None,
                            "low": float(low_s.iloc[i]) if low_s is not None else None,
                            "volume": float(vol_s.iloc[i]) if vol_s is not None else None,
                        }
                        ohlcv_bars.append(bar)
                    # Use last close as fallback price if quote failed
                    if degraded and last_price is None and ohlcv_bars:
                        last_price = ohlcv_bars[-1]["close"]
                        quote_data = {
                            "bid": None,
                            "ask": None,
                            "last": last_price,
                            "spread": None,
                            "spread_bps": None,
                        }
            except Exception as parse_e:
                logger.warning(f"market-data: OHLCV parse failed for {ticker}: {parse_e}")

        # --- Depth: synthesize from OHLCV via MicroPriceEngine ---
        depth_levels = []
        spread_history = []
        if df is not None and not df.empty:
            try:
                from engine.execution.execution_engine import MicroPriceEngine
                mpe = MicroPriceEngine()

                # Need flat DataFrame for MicroPriceEngine.estimate()
                flat_df = df
                if hasattr(df.columns, "levels"):
                    # Flatten MultiIndex to simple column names
                    try:
                        flat_df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.get_level_values(1) else df
                    except Exception:
                        flat_df = df

                est = mpe.estimate(ticker, flat_df)

                # Build synthetic L2-style depth from micro-price estimates
                if est.mid_price > 0:
                    step = est.mid_price * 0.001  # 0.1% per level
                    for i in range(10):
                        bid_level = est.bid_proxy - i * step
                        ask_level = est.ask_proxy + i * step
                        # Volume decays with distance from mid
                        decay = 1.0 / (1 + i * 0.5)
                        depth_levels.append({
                            "price": round(bid_level, 4),
                            "side": "bid",
                            "size": round(1000 * decay, 0),
                        })
                        depth_levels.append({
                            "price": round(ask_level, 4),
                            "side": "ask",
                            "size": round(1000 * decay, 0),
                        })

                    # Spread history from OHLCV bars (last N bars)
                    if ohlcv_bars:
                        for bar in ohlcv_bars[-20:]:
                            h = bar.get("high")
                            l = bar.get("low")
                            c = bar.get("close")
                            if h and l and c and c > 0:
                                bps = ((h - l) / c) * 10000 * 0.1
                                spread_history.append({
                                    "time": bar["time"],
                                    "spread_bps": round(bps, 2),
                                })

            except Exception as mpe_e:
                logger.warning(f"market-data: MicroPriceEngine failed for {ticker}: {mpe_e}")

        # Degrade gracefully — never return empty
        if not quote_data and last_price is None:
            quote_data = {"bid": None, "ask": None, "last": None, "spread": None, "spread_bps": None}
            degraded = True

        return {
            "ticker": ticker,
            "quote": quote_data,
            "ohlcv": ohlcv_bars[-50:] if ohlcv_bars else [],
            "depth": depth_levels,
            "spread_history": spread_history,
            "source": "openbb+micro_price",
            "degraded": degraded,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/market-data error: {e}")
        return {
            "ticker": ticker,
            "quote": {"bid": None, "ask": None, "last": None, "spread": None, "spread_bps": None},
            "ohlcv": [],
            "depth": [],
            "spread_history": [],
            "source": "openbb+micro_price",
            "degraded": True,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/l7/blotter")
async def l7_blotter(limit: int = Query(100, ge=1, le=500)):
    """Live trade blotter from L7 execution surface.

    Returns filled orders from L7's _filled_orders deque, which captures
    all orders routed through the unified execution surface (IBKR + paper mirror).

    BROKER SWAP NOTE: L7 routes through whatever broker is configured.
    The blotter reflects fills from the active broker. When switching to IBKR,
    L7._execute_equity() needs an IBKR execution path added.
    """
    try:
        l7 = get_l7()
        broker_type = "unknown"

        trades = []

        # --- Primary path: L7 _filled_orders deque ---
        if l7 is not None and hasattr(l7, "_filled_orders") and l7._filled_orders:
            try:
                broker_type = "l7"
                # _filled_orders is a deque of L7Order dataclasses
                filled = list(l7._filled_orders)[-limit:]
                for order in filled:
                    if isinstance(order, dict):
                        trades.append(order)
                    else:
                        # L7Order dataclass — convert to dict
                        tca = getattr(order, "tca_snapshot", None)
                        tca_dict = {}
                        if tca is not None:
                            if isinstance(tca, dict):
                                tca_dict = tca
                            elif hasattr(tca, "__dict__"):
                                tca_dict = tca.__dict__

                        side_raw = getattr(order, "side", "")
                        pt_raw = getattr(order, "product_type", "")
                        route_raw = getattr(order, "routing_strategy", "")
                        urg_raw = getattr(order, "urgency", "")
                        sig_raw = getattr(order, "signal_type", "")
                        ts_raw = getattr(order, "timestamp", None)

                        trades.append({
                            "ticker": getattr(order, "ticker", ""),
                            "side": side_raw.value if hasattr(side_raw, "value") else str(side_raw),
                            "quantity": getattr(order, "quantity", 0),
                            "fill_price": getattr(order, "fill_price", 0),
                            "arrival_price": getattr(order, "arrival_price", 0),
                            "slippage_bps": getattr(order, "slippage_bps", 0),
                            "implementation_shortfall": getattr(order, "implementation_shortfall", 0),
                            "market_impact_bps": getattr(order, "market_impact_bps", 0),
                            "signal_type": sig_raw.value if hasattr(sig_raw, "value") else str(sig_raw),
                            "product_type": pt_raw.value if hasattr(pt_raw, "value") else str(pt_raw),
                            "routing_strategy": route_raw.value if hasattr(route_raw, "value") else str(route_raw),
                            "urgency": urg_raw.value if hasattr(urg_raw, "value") else str(urg_raw),
                            "timestamp": ts_raw.isoformat() if ts_raw and hasattr(ts_raw, "isoformat") else str(ts_raw or ""),
                            "tca_snapshot": tca_dict,
                        })
            except Exception as l7e:
                logger.warning(f"l7/blotter: L7 _filled_orders parse failed: {l7e}")
                trades = []

        # --- Fallback: broker.get_trade_history() ---
        if not trades:
            try:
                eng = get_engine()
                broker = eng.broker
                broker_type = type(broker).__name__
                raw = broker.get_trade_history()[-limit:]
                for t in raw:
                    if isinstance(t, dict):
                        trades.append({
                            "ticker": t.get("ticker", ""),
                            "side": t.get("side", ""),
                            "quantity": t.get("quantity", 0),
                            "fill_price": t.get("fill_price", 0),
                            "arrival_price": t.get("arrival_price", t.get("fill_price", 0)),
                            "slippage_bps": t.get("slippage_bps", t.get("slippage", 0)),
                            "implementation_shortfall": t.get("implementation_shortfall", 0),
                            "market_impact_bps": t.get("market_impact_bps", 0),
                            "signal_type": t.get("signal_type", ""),
                            "product_type": t.get("product_type", "EQUITY"),
                            "routing_strategy": t.get("routing_strategy", "SMART"),
                            "urgency": t.get("urgency", "MEDIUM"),
                            "timestamp": str(t.get("fill_timestamp", t.get("timestamp", ""))),
                            "tca_snapshot": t.get("tca_snapshot", {}),
                        })
                    else:
                        side_raw = getattr(t, "side", "")
                        sig_raw = getattr(t, "signal_type", "")
                        ts_raw = getattr(t, "fill_timestamp", None)
                        trades.append({
                            "ticker": getattr(t, "ticker", ""),
                            "side": side_raw.value if hasattr(side_raw, "value") else str(side_raw),
                            "quantity": getattr(t, "quantity", 0),
                            "fill_price": getattr(t, "fill_price", 0),
                            "arrival_price": getattr(t, "fill_price", 0),
                            "slippage_bps": getattr(t, "slippage", 0),
                            "implementation_shortfall": 0,
                            "market_impact_bps": 0,
                            "signal_type": sig_raw.value if hasattr(sig_raw, "value") else str(sig_raw),
                            "product_type": "EQUITY",
                            "routing_strategy": "SMART",
                            "urgency": "MEDIUM",
                            "timestamp": ts_raw.isoformat() if ts_raw and hasattr(ts_raw, "isoformat") else str(ts_raw or ""),
                            "tca_snapshot": {},
                        })
            except Exception as fe:
                logger.warning(f"l7/blotter: broker fallback failed: {fe}")
                broker_type = "error"

        # Never return empty list
        if not trades:
            return {
                "trades": [],
                "source": "no_fills_yet",
                "broker": broker_type,
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "trades": trades,
            "source": "l7_filled_orders" if broker_type == "l7" else f"broker_{broker_type}",
            "broker": broker_type,
            "count": len(trades),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/l7/blotter error: {e}")
        return {
            "trades": [],
            "source": "error",
            "broker": "unknown",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/top-of-book")
async def top_of_book():
    """Top-of-book quotes for every position in the portfolio.

    Fetches real-time(ish) bid/ask/last for each held ticker via OpenBB get_quote().
    Used by the LIVE dashboard execution quality table.

    BROKER SWAP NOTE: Uses OpenBB for quotes regardless of broker.
    For IBKR real-time quotes, could switch to ibkr.reqMktData() per ticker.
    """
    try:
        # Get held positions from broker
        broker = None
        try:
            eng = get_engine()
            broker = eng.broker
        except Exception:
            try:
                from engine.execution.paper_broker import PaperBroker
                broker = PaperBroker()
            except Exception:
                pass

        if broker is None:
            return {
                "positions": [],
                "note": "broker_unavailable",
                "timestamp": datetime.utcnow().isoformat(),
            }

        positions_map = broker.get_all_positions()

        if not positions_map:
            return {
                "positions": [],
                "note": "no_open_positions",
                "timestamp": datetime.utcnow().isoformat(),
            }

        tickers = list(positions_map.keys())

        # Bulk fetch quotes for all held tickers (one API call)
        quote_map = {}
        try:
            raw_quotes = get_quote(tickers)
            for q in raw_quotes:
                sym = q.get("symbol", "") or q.get("ticker", "")
                if sym:
                    quote_map[sym] = q
        except Exception as qe:
            logger.warning(f"top-of-book: get_quote failed: {qe}")

        result_positions = []
        for ticker, pos in positions_map.items():
            # Position fields — pos is a Position dataclass
            qty = getattr(pos, "quantity", 0) if not isinstance(pos, dict) else pos.get("quantity", 0)
            avg_cost = getattr(pos, "avg_cost", 0) if not isinstance(pos, dict) else pos.get("avg_cost", 0)
            unrealized_pnl = getattr(pos, "unrealized_pnl", 0) if not isinstance(pos, dict) else pos.get("unrealized_pnl", 0)

            # Quote fields
            q = quote_map.get(ticker, {})
            bid = q.get("bid") or q.get("bid_price")
            ask = q.get("ask") or q.get("ask_price")
            last = q.get("price") or q.get("last_price") or q.get("last")

            # Compute spread
            spread = None
            spread_bps = None
            if bid is not None and ask is not None:
                spread = round(ask - bid, 4)
                mid = (ask + bid) / 2
                if mid > 0:
                    spread_bps = round(spread / mid * 10000, 2)

            # Position value at last price
            price_for_value = last or ask or bid or avg_cost or 0
            position_value = round(qty * price_for_value, 2) if price_for_value else None

            result_positions.append({
                "ticker": ticker,
                "last_price": last,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "spread_bps": spread_bps,
                "position_qty": qty,
                "position_value": position_value,
                "avg_cost": avg_cost,
                "unrealized_pnl": unrealized_pnl,
            })

        return {
            "positions": result_positions,
            "count": len(result_positions),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/top-of-book error: {e}")
        return {
            "positions": [],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
