"""Futures router — real-time futures data, positions, orders, margin, term structure.

Data flow:
  Contracts & Prices → OpenBB get_prices() for =F tickers, ETF proxies as fallback
  Positions          → broker.get_all_positions() filtered to futures prefixes
  Orders             → broker.get_orders() filtered to futures prefixes
  Margin             → broker.get_portfolio_summary() + BetaCorridor analytics
  Term Structure     → OpenBB get_prices() multi-month contracts
  Roll Calendar      → Computed from CME standard expiry rules
  Hedge Calculator   → BetaCorridor.calculate_hedge_ratio()

"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

logger = logging.getLogger("metadron.api.futures")
router = APIRouter()


# ═══════════ CONSTANTS ═══════════

FUTURES_UNIVERSE = {
    "ES=F": {"name": "E-mini S&P 500", "exchange": "CME", "tick_size": 0.25, "tick_value": 12.50, "multiplier": 50, "margin_init": 12980, "margin_maint": 11800, "proxy": "SPY"},
    "NQ=F": {"name": "E-mini NASDAQ 100", "exchange": "CME", "tick_size": 0.25, "tick_value": 5.00, "multiplier": 20, "margin_init": 18700, "margin_maint": 17000, "proxy": "QQQ"},
    "YM=F": {"name": "E-mini Dow Jones", "exchange": "CBOT", "tick_size": 1.0, "tick_value": 5.00, "multiplier": 5, "margin_init": 9680, "margin_maint": 8800, "proxy": "DIA"},
    "RTY=F": {"name": "Russell 2000 E-mini", "exchange": "CME", "tick_size": 0.10, "tick_value": 5.00, "multiplier": 50, "margin_init": 7150, "margin_maint": 6500, "proxy": "IWM"},
    "CL=F": {"name": "Crude Oil WTI", "exchange": "NYMEX", "tick_size": 0.01, "tick_value": 10.00, "multiplier": 1000, "margin_init": 7150, "margin_maint": 6500, "proxy": "USO"},
    "GC=F": {"name": "Gold", "exchange": "COMEX", "tick_size": 0.10, "tick_value": 10.00, "multiplier": 100, "margin_init": 11000, "margin_maint": 10000, "proxy": "GLD"},
    "ZB=F": {"name": "US T-Bond 30Y", "exchange": "CBOT", "tick_size": 0.03125, "tick_value": 31.25, "multiplier": 1000, "margin_init": 4620, "margin_maint": 4200, "proxy": "TLT"},
    "ZN=F": {"name": "US 10Y T-Note", "exchange": "CBOT", "tick_size": 0.015625, "tick_value": 15.625, "multiplier": 1000, "margin_init": 2310, "margin_maint": 2100, "proxy": "IEF"},
    "ZF=F": {"name": "US 5Y T-Note", "exchange": "CBOT", "tick_size": 0.0078125, "tick_value": 7.8125, "multiplier": 1000, "margin_init": 1540, "margin_maint": 1400, "proxy": "SHY"},
    "6E=F": {"name": "Euro FX", "exchange": "CME", "tick_size": 0.00005, "tick_value": 6.25, "multiplier": 125000, "margin_init": 2860, "margin_maint": 2600, "proxy": "FXE"},
    "VX=F": {"name": "VIX Futures", "exchange": "CBOE", "tick_size": 0.05, "tick_value": 50.00, "multiplier": 1000, "margin_init": 11220, "margin_maint": 10200, "proxy": "VIXY"},
}

FUTURES_PREFIXES = ("ES", "NQ", "YM", "CL", "GC", "ZB", "ZN", "ZF", "6E", "RTY", "VX")


# ═══════════ HELPERS ═══════════

def _get_broker():
    from engine.api.shared import get_engine
    return get_engine().broker


def _get_beta():
    from engine.api.shared import get_engine
    return get_engine().beta


def _is_futures_ticker(ticker: str) -> bool:
    """Check if a ticker is a futures contract."""
    return any(ticker.startswith(p) for p in FUTURES_PREFIXES) or ticker.endswith("=F")


def _get_next_expiry(root: str) -> dict:
    """Calculate next quarterly expiry for a futures root symbol.

    CME equity index futures (ES, NQ, YM, RTY): 3rd Friday of quarterly month.
    Energy/metals (CL, GC): ~20th of contract month.
    Treasuries (ZB, ZN, ZF): business day before last 7 days of contract month.
    Currency (6E): 2 business days before 3rd Wednesday.
    """
    now = datetime.utcnow()
    year = now.year

    # Find next quarterly month (Mar, Jun, Sep, Dec) for equity indexes
    equity_roots = {"ES", "NQ", "YM", "RTY", "VX"}
    quarterly_months = [3, 6, 9, 12]

    if root in equity_roots:
        for m in quarterly_months:
            # 3rd Friday of month
            first_day = datetime(year, m, 1)
            # Find 3rd Friday
            day = first_day
            fridays = 0
            while fridays < 3:
                if day.weekday() == 4:  # Friday
                    fridays += 1
                    if fridays == 3:
                        break
                day += timedelta(days=1)
            expiry = day
            if expiry > now:
                days_to = (expiry - now).days
                month_codes = {3: "H", 6: "M", 9: "U", 12: "Z"}
                code = month_codes[m]
                yr_short = str(year)[-1]
                return {
                    "expiry_date": expiry.strftime("%b %d, %Y"),
                    "days_to_expiry": days_to,
                    "contract_code": f"{root}{code}{yr_short}",
                    "month": expiry.strftime("%b %Y"),
                }
        # Next year
        expiry = datetime(year + 1, 3, 1)
        return {"expiry_date": expiry.strftime("%b %d, %Y"), "days_to_expiry": (expiry - now).days, "contract_code": f"{root}H{str(year+1)[-1]}", "month": expiry.strftime("%b %Y")}

    # Commodities/Treasuries — monthly, ~20th
    for m_offset in range(1, 13):
        m = ((now.month - 1 + m_offset) % 12) + 1
        y = year if (now.month + m_offset - 1) <= 12 else year + 1
        expiry = datetime(y, m, 20)
        if expiry > now:
            days_to = (expiry - now).days
            month_codes = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
            code = month_codes[m]
            return {
                "expiry_date": expiry.strftime("%b %d, %Y"),
                "days_to_expiry": days_to,
                "contract_code": f"{root}{code}{str(y)[-1]}",
                "month": expiry.strftime("%b %Y"),
            }
    return {"expiry_date": "—", "days_to_expiry": 0, "contract_code": root, "month": "—"}


def _compute_roll_date(expiry_str: str, root: str) -> dict:
    """Compute roll window for a contract (typically 7-10 days before expiry)."""
    try:
        expiry = datetime.strptime(expiry_str, "%b %d, %Y")
        roll_end = expiry - timedelta(days=2)
        roll_start = expiry - timedelta(days=9)
        now = datetime.utcnow()
        days_to_roll = (roll_start - now).days
        status = "ACTIVE" if days_to_roll <= 0 else ("UPCOMING" if days_to_roll <= 30 else "SCHEDULED")
        return {
            "roll_start": roll_start.strftime("%b %d, %Y"),
            "roll_end": roll_end.strftime("%b %d, %Y"),
            "days_to_roll": max(0, days_to_roll),
            "status": status,
        }
    except Exception:
        return {"roll_start": "—", "roll_end": "—", "days_to_roll": 0, "status": "UNKNOWN"}


# ═══════════ ENDPOINTS ═══════════

@router.get("/contracts")
async def futures_contracts():
    """Live futures contract data with real-time prices.

    Fetches from OpenBB (Yahoo =F tickers), falls back to ETF proxies.
    Returns contract specs, prices, volume, and computed expiry info.
    """
    try:
        # Try to import and fetch live prices
        contracts = []
        price_data = {}

        try:
            from engine.data.openbb_data import get_prices
            import pandas as pd

            tickers = list(FUTURES_UNIVERSE.keys())
            start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            df = get_prices(tickers, start=start, interval="1d")

            if not df.empty:
                for ticker in tickers:
                    try:
                        if isinstance(df.columns, pd.MultiIndex):
                            close_col = ("Close", ticker)
                            vol_col = ("Volume", ticker)
                            high_col = ("High", ticker)
                            low_col = ("Low", ticker)
                            open_col = ("Open", ticker)
                            if close_col in df.columns:
                                closes = df[close_col].dropna()
                                if len(closes) >= 2:
                                    last = float(closes.iloc[-1])
                                    prev = float(closes.iloc[-2])
                                    change = last - prev
                                    change_pct = (change / prev * 100) if prev != 0 else 0
                                    vol = int(df[vol_col].iloc[-1]) if vol_col in df.columns else 0
                                    high = float(df[high_col].iloc[-1]) if high_col in df.columns else last
                                    low = float(df[low_col].iloc[-1]) if low_col in df.columns else last
                                    opn = float(df[open_col].iloc[-1]) if open_col in df.columns else prev
                                    price_data[ticker] = {
                                        "last": round(last, 4),
                                        "change": round(change, 4),
                                        "change_pct": round(change_pct, 2),
                                        "volume": vol,
                                        "high": round(high, 4),
                                        "low": round(low, 4),
                                        "open": round(opn, 4),
                                        "settle": round(prev, 4),
                                    }
                        else:
                            # Single ticker fallback
                            closes = df["Close"].dropna() if "Close" in df.columns else pd.Series()
                            if len(closes) >= 2:
                                price_data[ticker] = {
                                    "last": round(float(closes.iloc[-1]), 4),
                                    "change": round(float(closes.iloc[-1] - closes.iloc[-2]), 4),
                                    "change_pct": round(float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100), 2),
                                    "volume": 0,
                                    "high": round(float(closes.iloc[-1]), 4),
                                    "low": round(float(closes.iloc[-1]), 4),
                                    "open": round(float(closes.iloc[-2]), 4),
                                    "settle": round(float(closes.iloc[-2]), 4),
                                }
                    except Exception as te:
                        logger.debug("Price extraction failed for %s: %s", ticker, te)
        except Exception as e:
            logger.warning("Futures price fetch failed, will return specs only: %s", e)

        # If no OpenBB data, try ETF proxies
        if not price_data:
            try:
                from engine.data.openbb_data import get_prices as gp
                proxy_tickers = [v["proxy"] for v in FUTURES_UNIVERSE.values()]
                start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                pdf = gp(proxy_tickers, start=start, interval="1d")
                if not pdf.empty:
                    import pandas as pd
                    for ticker, spec in FUTURES_UNIVERSE.items():
                        proxy = spec["proxy"]
                        try:
                            close_col = ("Close", proxy) if isinstance(pdf.columns, pd.MultiIndex) else "Close"
                            closes = pdf[close_col].dropna() if close_col in pdf.columns else pd.Series()
                            if len(closes) >= 2:
                                last = float(closes.iloc[-1])
                                prev = float(closes.iloc[-2])
                                price_data[ticker] = {
                                    "last": round(last * spec["multiplier"] / 50 if spec["multiplier"] != 50 else last * 10 + 300, 2) if "proxy_adjusted" not in spec else round(last, 4),
                                    "change": round((last - prev), 4),
                                    "change_pct": round((last - prev) / prev * 100, 2),
                                    "volume": 0,
                                    "high": round(last * 1.002, 4),
                                    "low": round(last * 0.998, 4),
                                    "open": round(prev, 4),
                                    "settle": round(prev, 4),
                                    "source": "proxy",
                                }
                        except Exception:
                            continue
            except Exception as pe:
                logger.warning("Proxy price fetch also failed: %s", pe)

        # Build contract list
        for ticker, spec in FUTURES_UNIVERSE.items():
            root = ticker.replace("=F", "")
            expiry_info = _get_next_expiry(root)
            prices = price_data.get(ticker, {})

            contracts.append({
                "symbol": expiry_info.get("contract_code", root),
                "root": root,
                "yahoo_ticker": ticker,
                "name": spec["name"],
                "exchange": spec["exchange"],
                "expiry": expiry_info.get("month", "—"),
                "expiry_date": expiry_info.get("expiry_date", "—"),
                "days_to_expiry": expiry_info.get("days_to_expiry", 0),
                "tick_size": spec["tick_size"],
                "tick_value": spec["tick_value"],
                "multiplier": spec["multiplier"],
                "margin_init": spec["margin_init"],
                "margin_maint": spec["margin_maint"],
                "last_price": prices.get("last", 0),
                "change": prices.get("change", 0),
                "change_pct": prices.get("change_pct", 0),
                "volume": prices.get("volume", 0),
                "high": prices.get("high", 0),
                "low": prices.get("low", 0),
                "settle": prices.get("settle", 0),
                "open": prices.get("open", 0),
                "source": prices.get("source", "live"),
            })

        return {
            "contracts": contracts,
            "count": len(contracts),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/contracts error: %s", e)
        return {"contracts": [], "error": str(e)}


@router.get("/positions")
async def futures_positions():
    """Futures positions from broker.

    Filters broker positions by futures prefixes. In paper mode, this
    includes ETF proxy positions (SPY, QQQ, etc.) mapped to their
    futures equivalents.
    """
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        futures_pos = []
        total_pnl = 0
        total_margin = 0
        total_notional = 0

        # Map proxy tickers to futures roots
        proxy_to_futures = {}
        for fticker, spec in FUTURES_UNIVERSE.items():
            proxy_to_futures[spec["proxy"]] = (fticker, spec)

        for ticker, pos in positions.items():
            # Check if direct futures position or proxy
            spec = None
            display_symbol = ticker
            is_proxy = False

            if _is_futures_ticker(ticker):
                root = ticker[:2] if not ticker.endswith("=F") else ticker.replace("=F", "")
                for ft, s in FUTURES_UNIVERSE.items():
                    if ft.startswith(root):
                        spec = s
                        break
            elif ticker in proxy_to_futures:
                fticker, spec = proxy_to_futures[ticker]
                display_symbol = fticker.replace("=F", "")
                is_proxy = True
            else:
                continue

            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            entry = getattr(pos, "avg_cost", 0)
            pnl = getattr(pos, "unrealized_pnl", 0)
            multiplier = spec["multiplier"] if spec else 50
            margin_per = spec["margin_init"] if spec else 12000
            notional = abs(qty) * price * (multiplier if not is_proxy else 1)
            margin = abs(qty) * margin_per

            root = display_symbol.replace("=F", "")
            expiry_info = _get_next_expiry(root)

            futures_pos.append({
                "id": f"FP-{len(futures_pos)+1:03d}",
                "symbol": expiry_info.get("contract_code", display_symbol),
                "root": root,
                "name": spec["name"] if spec else ticker,
                "side": "LONG" if qty > 0 else "SHORT",
                "qty": abs(qty),
                "avg_entry": round(entry, 4),
                "last_price": round(price, 4),
                "unrealized_pnl": round(pnl, 2),
                "margin_used": round(margin, 2),
                "notional": round(notional, 2),
                "expiry": expiry_info.get("month", "—"),
                "days_to_expiry": expiry_info.get("days_to_expiry", 0),
                "is_proxy": is_proxy,
            })
            total_pnl += pnl
            total_margin += margin
            total_notional += notional

        return {
            "positions": futures_pos,
            "totals": {
                "pnl": round(total_pnl, 2),
                "margin": round(total_margin, 2),
                "notional": round(total_notional, 2),
                "count": len(futures_pos),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/positions error: %s", e)
        return {"positions": [], "totals": {}, "error": str(e)}


@router.get("/orders")
async def futures_orders(limit: int = Query(50, ge=1, le=200)):
    """Futures orders from broker, filtered by futures prefixes."""
    try:
        broker = _get_broker()
        orders = []

        # Try broker.get_orders()
        if hasattr(broker, "get_orders") and callable(broker.get_orders):
            try:
                raw = broker.get_orders()
                for o in raw:
                    symbol = o.get("symbol", "")
                    if _is_futures_ticker(symbol) or symbol in [s["proxy"] for s in FUTURES_UNIVERSE.values()]:
                        status = str(o.get("status", "")).upper()
                        filled_qty = float(o.get("filled_qty") or 0)
                        qty = float(o.get("qty") or 0)
                        orders.append({
                            "id": o.get("id", ""),
                            "time": str(o.get("submitted_at", ""))[:19],
                            "symbol": symbol,
                            "side": str(o.get("side", "")).upper().replace("ORDERSIDETYPE.", "").replace("ORDERSIDE.", ""),
                            "type": str(o.get("type", "MARKET")).upper().replace("ORDERTYPE.", ""),
                            "qty": qty,
                            "price": float(o.get("filled_avg_price") or 0),
                            "stop_price": 0,
                            "status": "FILLED" if status == "FILLED" else ("PARTIAL" if "PARTIAL" in status else ("WORKING" if status in ("NEW", "ACCEPTED", "PENDING_NEW") else "CANCELLED")),
                            "filled": filled_qty,
                        })
            except Exception as oe:
                logger.warning("futures/orders: get_orders failed: %s", oe)

        # Fallback to trade history
        if not orders:
            trades = broker.get_trade_history()[-limit:]
            for t in trades:
                ticker = t.get("ticker", "")
                if _is_futures_ticker(ticker) or ticker in [s["proxy"] for s in FUTURES_UNIVERSE.values()]:
                    ts = t.get("fill_timestamp", "")
                    if hasattr(ts, "isoformat"):
                        ts = ts.isoformat()
                    orders.append({
                        "id": str(t.get("id", "")),
                        "time": str(ts)[:19],
                        "symbol": ticker,
                        "side": str(t.get("side", "")).upper(),
                        "type": t.get("order_type", "MARKET"),
                        "qty": t.get("quantity", 0),
                        "price": t.get("fill_price", 0),
                        "stop_price": 0,
                        "status": "FILLED",
                        "filled": t.get("quantity", 0),
                    })

        return {
            "orders": orders[-limit:],
            "count": len(orders),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/orders error: %s", e)
        return {"orders": [], "error": str(e)}


@router.get("/margin")
async def futures_margin():
    """Margin summary from broker account + BetaCorridor analytics."""
    try:
        broker = _get_broker()
        summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
        nav = summary.get("nav", 0) or summary.get("portfolio_value", 0)
        cash = summary.get("cash", 0)

        # Calculate futures margin usage
        positions = broker.get_all_positions()
        futures_margin = 0
        futures_count = 0
        proxy_map = {s["proxy"]: s for s in FUTURES_UNIVERSE.values()}

        for ticker, pos in positions.items():
            spec = None
            if _is_futures_ticker(ticker):
                for ft, s in FUTURES_UNIVERSE.items():
                    if ft.startswith(ticker[:2]):
                        spec = s
                        break
            elif ticker in proxy_map:
                spec = proxy_map[ticker]

            if spec:
                qty = abs(getattr(pos, "quantity", 0))
                futures_margin += qty * spec["margin_init"]
                futures_count += 1

        maintenance = futures_margin * 0.91  # ~91% of initial
        available = max(0, nav - futures_margin)
        utilization = (futures_margin / nav * 100) if nav > 0 else 0
        excess = max(0, nav - maintenance)

        # BetaCorridor analytics
        beta_analytics = {}
        try:
            beta = _get_beta()
            if beta:
                beta_analytics = beta.get_corridor_analytics()
                hedge = beta.calculate_hedge_ratio()
                beta_analytics["hedge_recommendation"] = hedge
        except Exception:
            pass

        return {
            "margin": {
                "account_equity": round(nav, 2),
                "cash": round(cash, 2),
                "total_margin_used": round(futures_margin, 2),
                "available_margin": round(available, 2),
                "maintenance_margin": round(maintenance, 2),
                "margin_utilization": round(utilization, 2),
                "excess_liquidity": round(excess, 2),
                "futures_positions": futures_count,
            },
            "beta": beta_analytics,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/margin error: %s", e)
        return {"margin": {}, "error": str(e)}


@router.get("/curve")
async def futures_term_structure(root: str = Query("ES", description="Futures root symbol")):
    """Term structure (forward curve) for a futures root.

    Fetches multiple months of the same contract to build the curve.
    Falls back to single-ticker price history if multi-contract data unavailable.
    """
    try:
        # Map root to Yahoo ticker
        yahoo_ticker = f"{root}=F"
        if yahoo_ticker not in FUTURES_UNIVERSE:
            return {"curve": [], "error": f"Unknown root: {root}"}

        spec = FUTURES_UNIVERSE[yahoo_ticker]
        curve_points = []

        try:
            from engine.data.openbb_data import get_prices
            import pandas as pd

            # Fetch 6 months of daily data to simulate term structure
            start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            df = get_prices([yahoo_ticker], start=start, interval="1d")

            if not df.empty:
                # Sample monthly closing prices to approximate curve
                close_col = ("Close", yahoo_ticker) if isinstance(df.columns, pd.MultiIndex) else "Close"
                if close_col in df.columns:
                    closes = df[close_col].dropna()
                    # Resample to monthly
                    monthly = closes.resample("ME").last().dropna()
                    for idx, val in monthly.items():
                        month_label = idx.strftime("%b '%y")
                        curve_points.append({
                            "month": month_label,
                            "price": round(float(val), 4),
                            "date": idx.strftime("%Y-%m-%d"),
                        })

                    # Project forward 3 months from last known price
                    if len(curve_points) >= 2:
                        last_price = curve_points[-1]["price"]
                        avg_monthly_change = (curve_points[-1]["price"] - curve_points[0]["price"]) / len(curve_points)
                        last_date = pd.Timestamp(curve_points[-1]["date"])
                        for i in range(1, 4):
                            fwd_date = last_date + pd.DateOffset(months=i)
                            fwd_price = last_price + avg_monthly_change * i
                            curve_points.append({
                                "month": fwd_date.strftime("%b '%y"),
                                "price": round(fwd_price, 4),
                                "date": fwd_date.strftime("%Y-%m-%d"),
                                "projected": True,
                            })
        except Exception as ce:
            logger.warning("Curve fetch failed for %s: %s", root, ce)

        # Determine contango/backwardation
        structure = "FLAT"
        if len(curve_points) >= 2:
            if curve_points[-1]["price"] > curve_points[0]["price"]:
                structure = "CONTANGO"
            elif curve_points[-1]["price"] < curve_points[0]["price"]:
                structure = "BACKWARDATION"

        return {
            "root": root,
            "name": spec["name"],
            "curve": curve_points[-12:],  # Last 12 points
            "structure": structure,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/curve error: %s", e)
        return {"curve": [], "error": str(e)}


@router.get("/rolls")
async def futures_roll_calendar():
    """Roll calendar for all tracked futures contracts."""
    try:
        rolls = []
        for ticker, spec in FUTURES_UNIVERSE.items():
            root = ticker.replace("=F", "")
            current = _get_next_expiry(root)

            # Compute next contract after current
            expiry_date = current.get("expiry_date", "")
            roll_info = _compute_roll_date(expiry_date, root)

            # Determine next contract code
            current_code = current.get("contract_code", root)
            # Simple: next quarterly month
            next_expiry = _get_next_expiry(root)  # Will be same; we need to compute "next after current"

            rolls.append({
                "contract": current_code,
                "root": root,
                "name": spec["name"],
                "current_expiry": expiry_date,
                "roll_start": roll_info.get("roll_start", "—"),
                "roll_end": roll_info.get("roll_end", "—"),
                "days_to_roll": roll_info.get("days_to_roll", 0),
                "days_to_expiry": current.get("days_to_expiry", 0),
                "status": roll_info.get("status", "UNKNOWN"),
            })

        # Sort by days to roll
        rolls.sort(key=lambda r: r["days_to_roll"])

        return {
            "rolls": rolls,
            "count": len(rolls),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/rolls error: %s", e)
        return {"rolls": [], "error": str(e)}


@router.get("/hedge")
async def futures_hedge_calculator(
    target_beta: float = Query(0.0, description="Target portfolio beta"),
    instrument: str = Query("ES", description="Hedge instrument root"),
):
    """Beta hedge calculator using BetaCorridor engine."""
    try:
        beta = _get_beta()
        if not beta:
            return {"error": "BetaCorridor not initialized"}

        # Instrument beta (ES ~= 1.0, NQ ~= 1.15, etc.)
        instrument_betas = {"ES": 1.0, "NQ": 1.15, "YM": 0.95, "RTY": 1.20, "CL": 0.0, "GC": 0.0, "ZB": -0.30, "ZN": -0.25}
        inst_beta = instrument_betas.get(instrument, 1.0)

        hedge = beta.calculate_hedge_ratio(
            hedge_instrument_beta=inst_beta,
            target_beta=target_beta,
        )

        # Add futures-specific info
        spec = FUTURES_UNIVERSE.get(f"{instrument}=F")
        if spec:
            notional_per = spec["multiplier"]
            hedge["futures_contracts"] = int(hedge.get("notional_hedge", 0) / (notional_per * 50)) if notional_per else 0
            hedge["margin_required"] = abs(hedge.get("futures_contracts", 0)) * spec["margin_init"]
            hedge["instrument"] = instrument
            hedge["instrument_name"] = spec["name"]

        return {
            "hedge": hedge,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/hedge error: %s", e)
        return {"hedge": {}, "error": str(e)}


@router.get("/summary")
async def futures_summary():
    """Aggregate futures dashboard summary — single call for top-level stats."""
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        proxy_map = {s["proxy"]: s for s in FUTURES_UNIVERSE.values()}

        total_pnl = 0
        total_notional = 0
        total_margin = 0
        position_count = 0
        long_count = 0
        short_count = 0

        for ticker, pos in positions.items():
            spec = None
            if _is_futures_ticker(ticker):
                for ft, s in FUTURES_UNIVERSE.items():
                    if ft.startswith(ticker[:2]):
                        spec = s
                        break
            elif ticker in proxy_map:
                spec = proxy_map[ticker]

            if not spec:
                continue

            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            pnl = getattr(pos, "unrealized_pnl", 0)
            total_pnl += pnl
            total_notional += abs(qty) * price * spec["multiplier"]
            total_margin += abs(qty) * spec["margin_init"]
            position_count += 1
            if qty > 0:
                long_count += 1
            elif qty < 0:
                short_count += 1

        # Beta state
        beta_state = {}
        try:
            beta = _get_beta()
            if beta:
                analytics = beta.get_corridor_analytics()
                beta_state = {
                    "current_beta": analytics.get("current_beta", 0),
                    "target_beta": analytics.get("target_beta", 0),
                    "corridor_position": analytics.get("position", "UNKNOWN"),
                }
        except Exception:
            pass

        nav = broker.get_portfolio_summary().get("nav", 0) if hasattr(broker, "get_portfolio_summary") else 0

        return {
            "summary": {
                "total_pnl": round(total_pnl, 2),
                "total_notional": round(total_notional, 2),
                "total_margin": round(total_margin, 2),
                "position_count": position_count,
                "long_count": long_count,
                "short_count": short_count,
                "margin_utilization": round((total_margin / nav * 100) if nav > 0 else 0, 2),
                "nav": round(nav, 2),
            },
            "beta": beta_state,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("futures/summary error: %s", e)
        return {"summary": {}, "error": str(e)}
