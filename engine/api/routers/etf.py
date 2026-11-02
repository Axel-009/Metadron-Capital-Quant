"""
ETF router — Tab 17 ETF Dashboard
Live ETF holdings, sector heatmap, flows, movers, and category breakdown.
Sources: IBKR broker positions + OpenBB price data + UniverseEngine ETF definitions.
"""
from fastapi import APIRouter, Query
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("metadron-api.etf")
router = APIRouter()

# ─── Lazy singletons ──────────────────────────────────────────────

_broker = None
_universe = None


def _get_broker():
    global _broker
    if _broker is None:
        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            _broker = AlpacaBroker()
        except Exception:
            from engine.execution.paper_broker import PaperBroker
            _broker = PaperBroker()
    return _broker


def _get_universe():
    global _universe
    if _universe is None:
        from engine.data.universe_engine import UniverseEngine
        _universe = UniverseEngine()
        _universe.load_universe()
    return _universe


# ─── ETF category maps from universe engine ──────────────────────

def _get_etf_categories():
    """Build ticker → category mapping from universe engine definitions."""
    try:
        from engine.data.universe_engine import (
            SECTOR_ETFS, FACTOR_ETFS, COMMODITY_ETFS, FIXED_INCOME_ETFS,
            VOLATILITY_ETFS, INTERNATIONAL_ETFS, INDEX_ETFS, THEMATIC_ETFS,
        )
    except ImportError:
        return {}, {}

    category_map = {}  # ticker → category
    name_map = {}      # ticker → human name

    for name, ticker in SECTOR_ETFS.items():
        category_map[ticker] = "Sector"
        name_map[ticker] = name
    for name, ticker in FACTOR_ETFS.items():
        category_map[ticker] = "Factor"
        name_map[ticker] = name
    for name, ticker in COMMODITY_ETFS.items():
        category_map[ticker] = "Commodity"
        name_map[ticker] = name
    for name, ticker in FIXED_INCOME_ETFS.items():
        category_map[ticker] = "Bond"
        name_map[ticker] = name
    for name, ticker in VOLATILITY_ETFS.items():
        category_map[ticker] = "Volatility"
        name_map[ticker] = name
    for name, ticker in INTERNATIONAL_ETFS.items():
        category_map[ticker] = "International"
        name_map[ticker] = name
    for name, ticker in INDEX_ETFS.items():
        category_map[ticker] = "Equity"
        name_map[ticker] = name
    for name, ticker in THEMATIC_ETFS.items():
        category_map[ticker] = "Thematic"
        name_map[ticker] = name

    return category_map, name_map


def _get_all_etf_tickers():
    """Return full list of ETF tickers from universe engine."""
    try:
        from engine.data.universe_engine import ALL_ETFS
        return ALL_ETFS
    except ImportError:
        return []


# ─── Endpoints ────────────────────────────────────────────────────

@router.get("/holdings")
async def etf_holdings():
    """
    ETF holdings from broker positions, enriched with category info.
    Falls back to IBKR price data for tracked ETFs with no open position.
    """
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        category_map, name_map = _get_etf_categories()
        all_tickers = _get_all_etf_tickers()

        # Identify which positions are ETFs
        etf_tickers = set(all_tickers)
        holdings = []

        for ticker, pos in (positions.items() if isinstance(positions, dict) else []):
            if ticker not in etf_tickers:
                continue
            qty = getattr(pos, "quantity", 0)
            avg_cost = getattr(pos, "avg_cost", 0)
            current = getattr(pos, "current_price", 0)
            pnl = getattr(pos, "unrealized_pnl", 0)
            market_val = qty * current if current else 0

            holdings.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "category": category_map.get(ticker, "Other"),
                "quantity": qty,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(current, 2),
                "change_pct": round(((current - avg_cost) / avg_cost * 100) if avg_cost else 0, 2),
                "unrealized_pnl": round(pnl, 2),
                "market_value": round(market_val, 2),
            })

        # If no ETF positions from broker, fetch prices for tracked ETFs
        if not holdings:
            try:
                from engine.data.openbb_data import get_adj_close
                end = datetime.utcnow().strftime("%Y-%m-%d")
                start = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
                sample = all_tickers[:25]  # limit to avoid timeout
                for ticker in sample:
                    try:
                        df = get_adj_close(ticker, start=start, end=end)
                        if df.empty or len(df) < 2:
                            continue
                        col = df.iloc[:, 0] if df.ndim > 1 else df
                        price = float(col.iloc[-1])
                        prev = float(col.iloc[-2])
                        chg_pct = ((price - prev) / prev * 100) if prev else 0
                        holdings.append({
                            "ticker": ticker,
                            "name": name_map.get(ticker, ticker),
                            "category": category_map.get(ticker, "Other"),
                            "quantity": 0,
                            "avg_cost": round(prev, 2),
                            "current_price": round(price, 2),
                            "change_pct": round(chg_pct, 2),
                            "unrealized_pnl": 0,
                            "market_value": 0,
                        })
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"ETF fallback price fetch failed: {e}")

        # Compute total market value for weight calculation
        total_mv = sum(h["market_value"] for h in holdings) or 1
        for h in holdings:
            h["weight"] = round(h["market_value"] / total_mv * 100, 2) if h["market_value"] else 0

        return {
            "holdings": holdings,
            "total_positions": len(holdings),
            "total_market_value": round(total_mv, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"etf/holdings error: {e}")
        return {"holdings": [], "error": str(e)}


@router.get("/sector-heatmap")
async def etf_sector_heatmap():
    """Sector ETF heatmap — live price changes for SPDR sector ETFs."""
    try:
        from engine.data.universe_engine import SECTOR_ETFS
        from engine.data.openbb_data import get_adj_close

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
        sectors = []

        for name, ticker in SECTOR_ETFS.items():
            try:
                df = get_adj_close(ticker, start=start, end=end)
                if df.empty or len(df) < 2:
                    sectors.append({"ticker": ticker, "name": name, "change_pct": 0, "price": 0})
                    continue
                col = df.iloc[:, 0] if df.ndim > 1 else df
                price = float(col.iloc[-1])
                prev = float(col.iloc[-2])
                chg = ((price - prev) / prev * 100) if prev else 0
                sectors.append({
                    "ticker": ticker,
                    "name": name,
                    "price": round(price, 2),
                    "change_pct": round(chg, 2),
                })
            except Exception:
                sectors.append({"ticker": ticker, "name": name, "change_pct": 0, "price": 0})

        return {"sectors": sectors, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"etf/sector-heatmap error: {e}")
        return {"sectors": [], "error": str(e)}


@router.get("/flows")
async def etf_flows():
    """
    ETF flow approximation — uses recent price momentum as a proxy.
    Positive momentum ≈ inflows, negative ≈ outflows.
    """
    try:
        from engine.data.openbb_data import get_adj_close
        from engine.data.universe_engine import INDEX_ETFS, SECTOR_ETFS, THEMATIC_ETFS

        # Monitor flows for key ETFs
        flow_tickers = list(INDEX_ETFS.values()) + list(SECTOR_ETFS.values())[:6] + list(THEMATIC_ETFS.values())[:4]
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        _, name_map = _get_etf_categories()

        flows = []
        for ticker in flow_tickers:
            try:
                df = get_adj_close(ticker, start=start, end=end)
                if df.empty or len(df) < 5:
                    continue
                col = df.iloc[:, 0] if df.ndim > 1 else df
                recent = float(col.iloc[-1])
                week_ago = float(col.iloc[-5]) if len(col) >= 5 else float(col.iloc[0])
                month_ago = float(col.iloc[0])
                # Approximate flow from momentum * volume proxy
                weekly_return = ((recent - week_ago) / week_ago * 100) if week_ago else 0
                monthly_return = ((recent - month_ago) / month_ago * 100) if month_ago else 0
                direction = "in" if weekly_return >= 0 else "out"
                # Scale flow magnitude (proxy — not real AUM flow)
                flow_magnitude = abs(weekly_return) * 100  # scaled for display

                flows.append({
                    "ticker": ticker,
                    "name": name_map.get(ticker, ticker),
                    "flow": round(flow_magnitude, 1),
                    "direction": direction,
                    "weekly_return": round(weekly_return, 2),
                    "monthly_return": round(monthly_return, 2),
                })
            except Exception:
                continue

        # Sort: inflows first (largest), then outflows (largest)
        inflows = sorted([f for f in flows if f["direction"] == "in"], key=lambda x: -x["flow"])
        outflows = sorted([f for f in flows if f["direction"] == "out"], key=lambda x: -x["flow"])
        flows = inflows + outflows

        return {"flows": flows, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"etf/flows error: {e}")
        return {"flows": [], "error": str(e)}


@router.get("/movers")
async def etf_movers():
    """Top and bottom ETF movers by daily change."""
    try:
        from engine.data.openbb_data import get_adj_close
        all_tickers = _get_all_etf_tickers()
        _, name_map = _get_etf_categories()
        category_map, _ = _get_etf_categories()

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")

        movers = []
        for ticker in all_tickers[:40]:  # top 40 most traded
            try:
                df = get_adj_close(ticker, start=start, end=end)
                if df.empty or len(df) < 2:
                    continue
                col = df.iloc[:, 0] if df.ndim > 1 else df
                price = float(col.iloc[-1])
                prev = float(col.iloc[-2])
                chg = ((price - prev) / prev * 100) if prev else 0
                movers.append({
                    "ticker": ticker,
                    "name": name_map.get(ticker, ticker),
                    "category": category_map.get(ticker, "Other"),
                    "price": round(price, 2),
                    "change_pct": round(chg, 2),
                })
            except Exception:
                continue

        movers.sort(key=lambda x: x["change_pct"], reverse=True)
        top = movers[:5]
        bottom = movers[-5:] if len(movers) >= 5 else []

        return {
            "top_movers": top,
            "bottom_movers": bottom,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"etf/movers error: {e}")
        return {"top_movers": [], "bottom_movers": [], "error": str(e)}


@router.get("/categories")
async def etf_categories():
    """Category breakdown with total value and position count."""
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        category_map, _ = _get_etf_categories()
        all_tickers = set(_get_all_etf_tickers())

        cat_summary = {}
        for ticker, pos in (positions.items() if isinstance(positions, dict) else []):
            if ticker not in all_tickers:
                continue
            cat = category_map.get(ticker, "Other")
            if cat not in cat_summary:
                cat_summary[cat] = {"count": 0, "market_value": 0, "pnl": 0}
            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            cat_summary[cat]["count"] += 1
            cat_summary[cat]["market_value"] += qty * price
            cat_summary[cat]["pnl"] += getattr(pos, "unrealized_pnl", 0)

        total_mv = sum(c["market_value"] for c in cat_summary.values()) or 1
        categories = []
        for cat, data in cat_summary.items():
            categories.append({
                "category": cat,
                "count": data["count"],
                "market_value": round(data["market_value"], 2),
                "pnl": round(data["pnl"], 2),
                "weight": round(data["market_value"] / total_mv * 100, 1),
            })

        # If empty broker positions, return category definitions with counts
        if not categories:
            try:
                from engine.data.universe_engine import (
                    SECTOR_ETFS, FACTOR_ETFS, COMMODITY_ETFS, FIXED_INCOME_ETFS,
                    VOLATILITY_ETFS, INTERNATIONAL_ETFS, INDEX_ETFS, THEMATIC_ETFS,
                )
                defs = [
                    ("Sector", SECTOR_ETFS), ("Factor", FACTOR_ETFS),
                    ("Commodity", COMMODITY_ETFS), ("Bond", FIXED_INCOME_ETFS),
                    ("Volatility", VOLATILITY_ETFS), ("International", INTERNATIONAL_ETFS),
                    ("Equity", INDEX_ETFS), ("Thematic", THEMATIC_ETFS),
                ]
                for cat, etf_dict in defs:
                    categories.append({
                        "category": cat,
                        "count": len(etf_dict),
                        "market_value": 0,
                        "pnl": 0,
                        "weight": round(len(etf_dict) / len(all_tickers) * 100, 1) if all_tickers else 0,
                    })
            except Exception:
                pass

        return {"categories": categories, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"etf/categories error: {e}")
        return {"categories": [], "error": str(e)}


@router.get("/summary")
async def etf_summary():
    """High-level ETF portfolio summary stats."""
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        all_tickers = set(_get_all_etf_tickers())
        category_map, _ = _get_etf_categories()

        etf_count = 0
        total_value = 0
        total_pnl = 0
        categories_seen = set()

        for ticker, pos in (positions.items() if isinstance(positions, dict) else []):
            if ticker not in all_tickers:
                continue
            etf_count += 1
            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            total_value += qty * price
            total_pnl += getattr(pos, "unrealized_pnl", 0)
            categories_seen.add(category_map.get(ticker, "Other"))

        nav = 0
        try:
            summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
            nav = summary.get("nav", 0)
        except Exception:
            pass

        return {
            "etf_positions": etf_count,
            "total_etfs_tracked": len(all_tickers),
            "etf_market_value": round(total_value, 2),
            "etf_unrealized_pnl": round(total_pnl, 2),
            "etf_weight_of_portfolio": round((total_value / nav * 100) if nav else 0, 1),
            "categories_active": len(categories_seen),
            "portfolio_nav": round(nav, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"etf/summary error: {e}")
        return {"error": str(e)}
