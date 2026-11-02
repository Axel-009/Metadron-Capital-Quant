# BROKER SWAP NOTE: This router accesses broker via get_broker() which
# tries ExecutionEngine (default: IBKRBroker) and falls back to PaperBroker.
# Trade records are dict-shaped (from broker.get_trade_history()).
# When adding IBKR/IBKRBroker (legacy): ensure get_trade_history() returns list[dict]
# with keys: ticker, side, quantity, fill_price, fill_timestamp, signal_type,
# order_type, status, reason, stop_loss, take_profit, spread_bps, impact_bps

"""
Portfolio router — LIVE, ALLOC, TXLOG tabs
Wraps: IBKRBroker, AlphaOptimizer, BetaCorridor
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging
import traceback

from engine.api.shared import get_broker, get_engine, get_beta

logger = logging.getLogger("metadron-api.portfolio")
router = APIRouter()

# ─── Alpha optimizer — local lazy singleton (not on ExecutionEngine) ──
_alpha = None


def _get_alpha():
    global _alpha
    if _alpha is None:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        _alpha = AlphaOptimizer()
    return _alpha


# ─── LIVE tab endpoints ────────────────────────────────────

@router.get("/live")
async def portfolio_live():
    """Current portfolio state: NAV, P&L, positions, exposure."""
    try:
        broker = get_broker()
        state = broker.get_portfolio_summary()
        # get_portfolio_summary() returns a dict
        s = state if isinstance(state, dict) else {}
        return {
            "nav": s.get("nav", 0),
            "cash": s.get("cash", 0),
            "total_pnl": s.get("total_pnl", 0),
            "gross_exposure": s.get("gross_exposure", 0),
            "net_exposure": s.get("net_exposure", 0),
            "positions_count": s.get("positions", 0),
            "win_count": s.get("win_count", 0),
            "loss_count": s.get("loss_count", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/live error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/positions")
async def portfolio_positions():
    """All current positions with P&L."""
    try:
        broker = get_broker()
        positions = broker.get_all_positions()
        result = []
        for ticker, pos in (positions.items() if isinstance(positions, dict) else []):
            result.append({
                "ticker": ticker,
                "quantity": pos.quantity if hasattr(pos, "quantity") else 0,
                "avg_cost": pos.avg_cost if hasattr(pos, "avg_cost") else 0,
                "current_price": pos.current_price if hasattr(pos, "current_price") else 0,
                "unrealized_pnl": pos.unrealized_pnl if hasattr(pos, "unrealized_pnl") else 0,
                "realized_pnl": pos.realized_pnl if hasattr(pos, "realized_pnl") else 0,
                "sector": pos.sector if hasattr(pos, "sector") else "Unknown",
            })
        return {"positions": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/positions error: {e}")
        return {"positions": [], "error": str(e)}


@router.get("/trades")
async def portfolio_trades(limit: int = Query(50, ge=1, le=500)):
    """Recent trade history."""
    try:
        broker = get_broker()
        trades = broker.get_trade_history()[-limit:]
        result = []
        for t in trades:
            # Trade records are dicts from both PaperBroker and IBKRBroker
            ts = t.get("fill_timestamp", "")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            result.append({
                "id": str(t.get("id", "")),
                "ticker": t.get("ticker", ""),
                "side": t.get("side", ""),
                "quantity": t.get("quantity", 0),
                "fill_price": t.get("fill_price", 0),
                "signal_type": t.get("signal_type", ""),
                "timestamp": ts,
                "reason": t.get("reason", ""),
            })
        return {"trades": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/trades error: {e}")
        return {"trades": [], "error": str(e)}


# ─── ALLOC tab endpoints ───────────────────────────────────

@router.get("/allocation")
async def portfolio_allocation():
    """Current sleeve allocation and sector weights."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"allocation": {}, "weights": {}, "status": "no_optimization_run"}

        signals = []
        for s in (last.signals or []):
            signals.append({
                "ticker": s.ticker,
                "weight": s.weight if hasattr(s, "weight") else 0,
                "quality_tier": s.quality_tier if hasattr(s, "quality_tier") else "?",
                "alpha_pred": s.alpha_pred if hasattr(s, "alpha_pred") else 0,
                "sharpe_estimate": s.sharpe_estimate if hasattr(s, "sharpe_estimate") else 0,
                "momentum_3m": s.momentum_3m if hasattr(s, "momentum_3m") else 0,
            })

        return {
            "signals": signals,
            "optimal_weights": last.optimal_weights if hasattr(last, "optimal_weights") else {},
            "expected_return": last.expected_annual_return if hasattr(last, "expected_annual_return") else 0,
            "annual_volatility": last.annual_volatility if hasattr(last, "annual_volatility") else 0,
            "sharpe_ratio": last.sharpe_ratio if hasattr(last, "sharpe_ratio") else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/allocation error: {e}")
        return {"error": str(e)}


@router.get("/beta")
async def portfolio_beta():
    """Beta corridor state and analytics."""
    try:
        beta = get_beta()
        analytics = beta.get_corridor_analytics()
        history = beta.get_history()
        latest = history[-1] if history else None

        state = {}
        if latest:
            state = {
                "current_beta": latest.current_beta if hasattr(latest, "current_beta") else 0,
                "target_beta": latest.target_beta if hasattr(latest, "target_beta") else 0,
                "corridor_position": latest.corridor_position if hasattr(latest, "corridor_position") else "UNKNOWN",
                "vol_adjustment": latest.vol_adjustment if hasattr(latest, "vol_adjustment") else 0,
            }

        return {
            "state": state,
            "analytics": analytics,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/beta error: {e}")
        return {"error": str(e)}


@router.get("/exposures")
async def portfolio_exposures():
    """Gross/net exposure breakdown."""
    try:
        broker = get_broker()
        exposures = broker.compute_exposures()
        return {**exposures, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/exposures error: {e}")
        return {"error": str(e)}


# ─── Market indices (LIVE + ALLOC tabs) ────────────────────

@router.get("/indices")
async def portfolio_indices():
    """Live market index quotes via OpenBB."""
    try:
        from engine.data.openbb_data import get_adj_close
        from datetime import timedelta
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")

        tickers = ["SPY", "QQQ", "IWM", "DIA"]
        indices = []
        for t in tickers:
            try:
                df = get_adj_close(t, start=start, end=end)
                if df.empty or len(df) < 2:
                    continue
                col = df.iloc[:, 0] if df.ndim > 1 else df
                price = float(col.iloc[-1])
                prev = float(col.iloc[-2])
                change = price - prev
                change_pct = (change / prev * 100) if prev else 0
                spark = [float(v) for v in col.tail(8).values]
                indices.append({
                    "ticker": t, "price": round(price, 2),
                    "change": round(change_pct, 2),
                    "data": spark,
                })
            except Exception:
                continue

        return {"indices": indices, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/indices error: {e}")
        return {"indices": [], "error": str(e)}


@router.get("/movers")
async def portfolio_movers():
    """Top movers from universe based on momentum."""
    try:
        from engine.data.universe_engine import UniverseEngine
        uni = UniverseEngine()
        uni.load_universe()
        all_secs = uni.get_all()

        # Sort by momentum_3m for top/bottom movers
        with_momentum = [s for s in all_secs if hasattr(s, "momentum_3m") and s.momentum_3m != 0]
        with_momentum.sort(key=lambda s: s.momentum_3m, reverse=True)

        movers = []
        for s in with_momentum[:5]:
            movers.append({
                "ticker": s.ticker,
                "change": round(s.momentum_3m * 100, 1),
                "momentum": "strong" if s.momentum_3m > 0.05 else "moderate" if s.momentum_3m > 0 else "weak",
            })
        for s in with_momentum[-3:]:
            movers.append({
                "ticker": s.ticker,
                "change": round(s.momentum_3m * 100, 1),
                "momentum": "weak",
            })

        return {"movers": movers, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/movers error: {e}")
        return {"movers": [], "error": str(e)}


@router.get("/sector-allocation")
async def portfolio_sector_allocation():
    """Sector allocation from current positions."""
    try:
        broker = get_broker()
        positions = broker.get_all_positions()
        sector_weights: dict[str, float] = {}
        total_value = 0

        for ticker, pos in positions.items():
            val = abs(pos.quantity * pos.current_price) if hasattr(pos, "quantity") else 0
            sector = pos.sector if hasattr(pos, "sector") else "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0) + val
            total_value += val

        colors = ["#00d4aa", "#58a6ff", "#3fb950", "#bc8cff", "#f0883e", "#d29922", "#484f58", "#f85149", "#4ecdc4", "#da3633", "#a855f7"]
        allocation = []
        for i, (sector, val) in enumerate(sorted(sector_weights.items(), key=lambda x: -x[1])):
            pct = (val / total_value * 100) if total_value > 0 else 0
            allocation.append({
                "name": sector,
                "value": round(pct, 1),
                "color": colors[i % len(colors)],
            })

        return {"allocation": allocation, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/sector-allocation error: {e}")
        return {"allocation": [], "error": str(e)}


@router.get("/pnl-series")
async def portfolio_pnl_series():
    """Intraday P&L time-series from trade history."""
    try:
        broker = get_broker()

        # Try IBKRBroker portfolio history first (real brokerage PnL)
        if hasattr(broker, "get_portfolio_history"):
            try:
                hist = broker.get_portfolio_history(period="1D", timeframe="15Min")
                if hist and hasattr(hist, "timestamp"):
                    timestamps = hist.timestamp or []
                    equity = hist.equity or []
                    if timestamps and equity and len(timestamps) == len(equity):
                        base = equity[0] if equity[0] else 0
                        series = []
                        for ts_val, eq_val in zip(timestamps, equity):
                            if ts_val is None or eq_val is None:
                                continue
                            # Format timestamp
                            if hasattr(ts_val, "isoformat"):
                                t_str = ts_val.isoformat()
                            else:
                                t_str = str(ts_val)
                            # Express as P&L relative to start of period
                            pnl_val = round(float(eq_val) - float(base), 2)
                            series.append({"time": t_str, "value": pnl_val})
                        if series:
                            return {"series": series, "timestamp": datetime.utcnow().isoformat()}
            except Exception:
                pass  # Fall through to trade-based PnL

        # Trade-based PnL computation (dict access fix)
        trades = broker.get_trade_history()[-500:]
        if not trades:
            return {"series": [], "timestamp": datetime.utcnow().isoformat()}

        # Aggregate P&L by time bucket (15-min intervals)
        from collections import defaultdict
        buckets: dict[str, float] = defaultdict(float)
        cumulative = 0
        for t in reversed(trades):
            # Trade records are dicts from both PaperBroker and IBKRBroker
            ts = t.get("fill_timestamp", None)
            if not ts:
                continue
            if hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            bucket = ts_str[11:16] if len(ts_str) >= 16 else ts_str[:5]  # "HH:MM"
            fill_price = t.get("fill_price", 0) or 0
            quantity = t.get("quantity", 0) or 0
            side = t.get("side", "") or ""
            realized_pnl = t.get("realized_pnl", None)
            if realized_pnl is not None:
                pnl = realized_pnl
            else:
                pnl = fill_price * quantity * (1 if str(side).upper() in ("SELL", "SHORT") else -1) * 0.001
            cumulative += pnl
            buckets[bucket] = cumulative

        series = [{"time": k, "value": round(v, 2)} for k, v in sorted(buckets.items())]
        return {"series": series, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/pnl-series error: {e}")
        return {"series": [], "error": str(e)}


# ─── TXLOG: IBKRBroker orders fallback ────────────────────────────

@router.get("/orders")
async def portfolio_orders(limit: int = Query(100, ge=1, le=500)):
    """Fetch broker orders (IBKRBroker API primary, trade_history fallback).

    Returns unified order/fill records for the TXLOG tab.
    IBKR's get_orders() gives richer data (status, filled_qty, type).
    PaperBroker falls back to get_trade_history().
    """
    try:
        broker = get_broker()

        # --- Primary: IBKRBroker get_orders() ---
        if hasattr(broker, "get_orders") and callable(broker.get_orders):
            try:
                raw = broker.get_orders()
                if raw:
                    orders = []
                    for o in raw[-limit:]:
                        filled_price = float(o.get("filled_avg_price") or 0)
                        filled_qty = float(o.get("filled_qty") or 0)
                        qty = float(o.get("qty") or 0)
                        status = str(o.get("status", "")).upper()
                        fill_type = "FULL" if status == "FILLED" else (
                            "PARTIAL" if "PARTIAL" in status else (
                            "REJECTED" if status in ("REJECTED", "CANCELED", "EXPIRED") else "PENDING"))
                        orders.append({
                            "id": o.get("id", ""),
                            "ticker": o.get("symbol", ""),
                            "side": str(o.get("side", "")).upper().replace("ORDERSIDETYPE.", "").replace("ORDERSIDE.", ""),
                            "qty": filled_qty if filled_qty > 0 else qty,
                            "price": filled_price,
                            "notional": round(filled_price * (filled_qty or qty), 2),
                            "fill_type": fill_type,
                            "order_type": str(o.get("type", "MARKET")).upper(),
                            "status": status,
                            "submitted_at": str(o.get("submitted_at", "")),
                            "filled_at": str(o.get("filled_at", "")),
                            "signal_type": o.get("signal_type", "BROKER"),
                        })
                    return {"orders": orders, "source": "ibkr", "timestamp": datetime.utcnow().isoformat()}
            except Exception as oe:
                logger.warning(f"portfolio/orders: IBKR get_orders failed, falling back: {oe}")

        # --- Fallback: trade history ---
        trades = broker.get_trade_history()[-limit:]
        orders = []
        for t in trades:
            ts = t.get("fill_timestamp", "")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            orders.append({
                "id": str(t.get("id", "")),
                "ticker": t.get("ticker", ""),
                "side": str(t.get("side", "")).upper(),
                "qty": t.get("quantity", 0),
                "price": t.get("fill_price", 0),
                "notional": round(t.get("fill_price", 0) * t.get("quantity", 0), 2),
                "fill_type": "FULL",
                "order_type": t.get("order_type", "MARKET"),
                "status": "FILLED",
                "submitted_at": str(ts),
                "filled_at": str(ts),
                "signal_type": t.get("signal_type", ""),
            })
        return {"orders": orders, "source": "trade_history", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/orders error: {e}")
        return {"orders": [], "error": str(e)}
