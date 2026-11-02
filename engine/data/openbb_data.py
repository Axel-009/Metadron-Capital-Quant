"""OpenBB unified data provider for Metadron Capital.

OpenBB is the SOLE data source, providing access to 34+ data providers
(FRED, SEC, Polygon, FMP, Intrinio, CBOE, ECB, OECD, etc.) through a
single unified API.

try/except on ALL external imports — system runs degraded, never broken.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenBB SDK import with graceful fallback
# ---------------------------------------------------------------------------
_obb = None
_openbb_available = False

try:
    from openbb import obb
    _obb = obb
    _openbb_available = True
    logger.info("OpenBB SDK loaded — primary data source active")
except ImportError:
    logger.warning("OpenBB SDK not available — data fetching will return empty frames")

# ---------------------------------------------------------------------------
# FMP API Key integration — load from .env or environment
# ---------------------------------------------------------------------------
import os as _os
_fmp_key = _os.environ.get("FMP_API_KEY", "")
if not _fmp_key:
    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        _env_path = _os.path.join(_os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(_env_path)
        _fmp_key = _os.environ.get("FMP_API_KEY", "")
    except Exception:
        pass

if _fmp_key and _fmp_key != "PASTE_KEY_HERE" and _obb is not None:
    try:
        _obb.user.credentials.fmp_api_key = _fmp_key
        logger.info("FMP API key configured — fundamentals available")
    except Exception as e:
        logger.warning(f"FMP credential setup failed: {e}")
elif not _fmp_key:
    logger.info("FMP_API_KEY not set — using IBKR prices only")

# ---------------------------------------------------------------------------
# Data source mode — IBKRBroker real-time during market hours, OpenBB EOD after
# ---------------------------------------------------------------------------
_DATA_SOURCE_MODE = "openbb"  # "auto" | "ibkr" | "openbb" — OpenBB/FMP primary, IBKR for real-time

def _is_market_hours() -> bool:
    """Check if US market is currently open (09:30-16:00 ET, Mon-Fri)."""
    try:
        from datetime import timezone
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        return False


def get_active_data_source() -> str:
    """Return the currently active data source label for dashboard display."""
    if _use_ibkr_for_prices():
        return "LIVE:IBKR"
    return "EOD:OpenBB/FMP"


def set_data_source_mode(mode: str):
    """Switch data source mode. Called by dashboard or config.

    Args:
        mode: "auto" (market-hours aware), "ibkr" (force real-time), "openbb" (force historical)
    """
    global _DATA_SOURCE_MODE
    if mode in ("auto", "ibkr", "openbb"):
        _DATA_SOURCE_MODE = mode
        logger.info("Data source mode set to: %s (active: %s)", mode, get_active_data_source())
    else:
        logger.warning("Invalid data source mode: %s (must be auto/ibkr/openbb)", mode)


# ---------------------------------------------------------------------------
# IBKRBroker Data API — direct client for real-time equity prices during market hours
# ---------------------------------------------------------------------------
_ibkr_data_client = None
_IBKR_DATA_AVAILABLE = False

try:
    import os
    from pathlib import Path as _Path
    try:
        from dotenv import load_dotenv as _load_dotenv
        _env_path = _Path(__file__).parent.parent.parent / ".env"
        if _env_path.exists():
            _load_dotenv(_env_path)
    except ImportError:
        pass

    _ibkr_key = os.environ.get("IBKR_API_KEY", "")
    _ibkr_secret = os.environ.get("IBKR_SECRET_KEY", "")
    if _ibkr_key and _ibkr_secret:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
        from alpaca.data.timeframe import TimeFrame
        _ibkr_data_client = StockHistoricalDataClient(_ibkr_key, _ibkr_secret)
        _IBKR_DATA_AVAILABLE = True
        logger.info("IBKR Data API available — real-time equity prices during market hours")
except ImportError:
    logger.debug("ib_insync not installed — IBKR data path disabled")
except Exception as e:
    logger.debug("IBKR data client init failed: %s", e)


def _fetch_ibkr_bars(tickers: list[str], start: str, end: Optional[str],
                       interval: str) -> pd.DataFrame:
    """Fetch OHLCV bars directly from IBKRBroker Data API.

    Returns DataFrame in the same format as OpenBB for seamless substitution.
    """
    if not _IBKR_DATA_AVAILABLE or not _ibkr_data_client:
        return pd.DataFrame()

    try:
        # Map interval string to IBKRBroker TimeFrame
        tf_map = {
            "1d": TimeFrame.Day, "1D": TimeFrame.Day,
            "1h": TimeFrame.Hour, "1H": TimeFrame.Hour,
            "1min": TimeFrame.Minute, "1m": TimeFrame.Minute,
        }
        timeframe = tf_map.get(interval, TimeFrame.Day)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime() if end else None,
            timeframe=timeframe,
        )
        bars = _ibkr_data_client.get_stock_bars(request)
        df = bars.df if hasattr(bars, "df") else pd.DataFrame()
        if df.empty:
            return pd.DataFrame()

        # Normalize IBKRBroker output to match OpenBB schema
        # IBKRBroker returns MultiIndex (symbol, timestamp) with columns: open, high, low, close, volume, ...
        if isinstance(df.index, pd.MultiIndex):
            # Multi-ticker: pivot to flat format per ticker
            frames = {}
            for ticker in tickers:
                try:
                    tdf = df.xs(ticker, level="symbol") if "symbol" in df.index.names else df
                    tdf = tdf[["open", "high", "low", "close", "volume"]].copy()
                    tdf.columns = ["Open", "High", "Low", "Close", "Volume"]
                    tdf["Adj Close"] = tdf["Close"]
                    frames[ticker] = tdf
                except (KeyError, Exception):
                    continue

            if not frames:
                return pd.DataFrame()

            if len(frames) == 1:
                result = list(frames.values())[0]
                result.columns = pd.MultiIndex.from_tuples(
                    [(c, tickers[0]) for c in result.columns]
                )
                return result

            parts = []
            for ticker, tdf in frames.items():
                tdf.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in tdf.columns])
                parts.append(tdf)
            return pd.concat(parts, axis=1)
        else:
            # Single ticker flat
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            df["Adj Close"] = df["Close"]
            ticker = tickers[0]
            df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns])
            return df

    except Exception as e:
        logger.warning("IBKR bars fetch failed: %s", e)
        return pd.DataFrame()


def _use_ibkr_for_prices() -> bool:
    """Check if we should route equity prices through IBKRBroker.

    IBKR replaces polygon during market hours for real-time data.
    After hours / backtesting falls back to OpenBB/FMP (default).

    Mode behavior:
      "auto":    IBKR during market hours, FMP after hours
      "ibkr":  Force IBKR always (even after hours)
      "openbb":  Force FMP always (backtesting mode)
    """
    if not _IBKR_DATA_AVAILABLE:
        return False
    if _DATA_SOURCE_MODE == "openbb":
        return False
    if _DATA_SOURCE_MODE == "ibkr":
        return True
    # auto: IBKRBroker during market hours only
    return _is_market_hours()


# Default provider for equity data (FMP has a free tier; alternatives: intrinio, polygon, tiingo)
DEFAULT_EQUITY_PROVIDER = "fmp"
# Provider for macro/economic data
DEFAULT_MACRO_PROVIDER = "fred"
# Provider for fundamental data
DEFAULT_FUNDAMENTAL_PROVIDER = "fmp"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _obbject_to_dataframe(result) -> pd.DataFrame:
    """Convert an OpenBB OBBject result to a pandas DataFrame."""
    if result is None:
        return pd.DataFrame()
    try:
        df = result.to_dataframe()
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        try:
            # Some results expose .results as list of dicts
            if hasattr(result, "results") and result.results:
                return pd.DataFrame([dict(r) for r in result.results])
        except Exception:
            pass
    return pd.DataFrame()


def _retry(func, retries: int = 2, delay: float = 1.0):
    """Simple retry wrapper for API calls."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
                logger.debug(f"Retry {attempt + 1}/{retries} after: {e}")
    raise last_exc


# ═══════════════════════════════════════════════════════════════════════════
# PRICE DATA — OpenBB (sole source)
# ═══════════════════════════════════════════════════════════════════════════

def get_prices(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV price data.

    Routing (when provider=None):
      Market hours:  IBKR real-time → fallback OpenBB/FMP
      After hours:   OpenBB/FMP (default)
      mode="openbb": OpenBB/FMP only (for backtesting)

    When provider is explicitly passed, OpenBB uses that provider directly.

    Returns DataFrame with MultiIndex columns (field, ticker) for compatibility
    with existing engine code.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # --- IBKRBroker path (market hours — replaces polygon for real-time) ---
    if provider is None and _use_ibkr_for_prices():
        # Split tickers: IBKRBroker handles equities/ETFs but NOT index symbols (^VIX, ^TNX, etc.)
        equity_tickers = [t for t in tickers if not t.startswith("^")]
        index_tickers = [t for t in tickers if t.startswith("^")]

        # Fetch equities via IBKRBroker
        equity_df = pd.DataFrame()
        if equity_tickers:
            equity_df = _fetch_ibkr_bars(equity_tickers, start, end, interval)

        # Index symbols fall through to OpenBB/FMP
        if equity_df.empty and not index_tickers:
            logger.debug("IBKR bars empty for %s — falling back to OpenBB/FMP", tickers[:3])
        elif not equity_df.empty:
            return equity_df

    # --- OpenBB path (FMP default — always available fallback) ---
    prov = provider or DEFAULT_EQUITY_PROVIDER  # "fmp"

    if _openbb_available:
        try:
            symbol_str = ",".join(tickers)
            result = _retry(lambda: _obb.equity.price.historical(
                symbol=symbol_str,
                start_date=start,
                end_date=end,
                interval=interval,
                provider=prov,
            ))
            df = _obbject_to_dataframe(result)
            if not df.empty:
                return _normalize_ohlcv(df, tickers)
        except Exception as e:
            logger.warning(f"OpenBB price fetch failed ({prov}): {e}")

    return pd.DataFrame()


def _normalize_ohlcv(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize OpenBB price data to standard MultiIndex format.

    Output: MultiIndex columns (field, ticker) where field is
    Open/High/Low/Close/Adj Close/Volume.
    """
    if df.empty:
        return df

    # Set date index if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # If single ticker, check if data is already flat OHLCV
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    has_ohlcv = ohlcv_cols.issubset({c.lower() for c in df.columns})

    if len(tickers) == 1 and has_ohlcv:
        # Map to standard column names
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "open": col_map[c] = "Open"
            elif cl == "high": col_map[c] = "High"
            elif cl == "low": col_map[c] = "Low"
            elif cl == "close": col_map[c] = "Close"
            elif cl == "volume": col_map[c] = "Volume"
            elif cl in ("adj_close", "adj close", "adjusted_close"):
                col_map[c] = "Adj Close"
        df = df.rename(columns=col_map)
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        return df

    # Multi-ticker: need to pivot into MultiIndex
    if "symbol" in df.columns:
        frames = {}
        for field in ["open", "high", "low", "close", "volume"]:
            if field in [c.lower() for c in df.columns]:
                actual_col = [c for c in df.columns if c.lower() == field][0]
                pivot = df.pivot_table(index=df.index, columns="symbol", values=actual_col)
                yf_name = field.capitalize()
                if yf_name == "Close":
                    frames["Close"] = pivot
                    frames["Adj Close"] = pivot.copy()
                else:
                    frames[yf_name] = pivot
        if frames:
            result = pd.concat(frames, axis=1)
            return result

    return df


def get_adj_close(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Get adjusted close prices, always returns DataFrame."""
    raw = get_prices(tickers, start, end, provider=provider)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        key = "Adj Close" if "Adj Close" in top else "Close"
        data = raw[key].copy()
    else:
        data = raw[["Adj Close" if "Adj Close" in raw.columns else "Close"]].copy()
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data


def get_returns(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    log_returns: bool = True,
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Compute returns from adjusted close prices."""
    prices = get_adj_close(tickers, start, end, provider=provider)
    if prices.empty:
        return pd.DataFrame()
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ═══════════════════════════════════════════════════════════════════════════
# MARKET STATS (for beta corridor engine)
# ═══════════════════════════════════════════════════════════════════════════

def get_market_stats(
    benchmark: str = "^GSPC",
    lookback_years: int = 1,
) -> dict:
    """Compute annualised drift (Rm), realised vol (sigma), and last price."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    prices = get_adj_close(benchmark, start, end)
    if prices.empty:
        logger.error("get_market_stats: no price data for %s — returning error state", benchmark)
        return {"Rm": None, "sigma_m": None, "last_price": None, "error": f"No price data for {benchmark}"}
    close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices
    log_ret = np.log(close / close.shift(1)).dropna()
    sigma_m = float(log_ret.std() * np.sqrt(252))
    Rm = float(log_ret.sum())
    return {"Rm": Rm, "sigma_m": sigma_m, "last_price": float(close.iloc[-1])}


# ═══════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DATA — OpenBB multi-provider
# ═══════════════════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str, provider: Optional[str] = None) -> dict:
    """Fetch key fundamental data for a single ticker.

    OpenBB providers: fmp (free tier), intrinio, polygon.
    """
    prov = provider or DEFAULT_FUNDAMENTAL_PROVIDER

    # --- OpenBB path ---
    if _openbb_available:
        try:
            result = _obb.equity.fundamental.metrics(
                symbol=ticker, provider=prov,
            )
            df = _obbject_to_dataframe(result)
            if not df.empty:
                row = df.iloc[0] if len(df) > 0 else {}
                return _parse_openbb_fundamentals(ticker, row)
        except Exception as e:
            logger.debug(f"OpenBB fundamentals failed for {ticker}: {e}")

    logger.warning("get_fundamentals: no data for %s — OpenBB unavailable or provider failed", ticker)
    return {"ticker": ticker, "error": "No fundamental data available"}


def _parse_openbb_fundamentals(ticker: str, row) -> dict:
    """Parse OpenBB fundamental metrics into standard dict format."""
    def _get(key, default=None):
        try:
            v = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return v
        except Exception:
            return default

    return {
        "ticker": ticker,
        "market_cap": _get("market_cap", 0),
        "pe_ratio": _get("pe_ratio") or _get("trailing_pe"),
        "forward_pe": _get("forward_pe"),
        "pb_ratio": _get("pb_ratio") or _get("price_to_book"),
        "dividend_yield": _get("dividend_yield"),
        "roe": _get("return_on_equity") or _get("roe"),
        "roa": _get("return_on_assets") or _get("roa"),
        "debt_to_equity": _get("debt_to_equity"),
        "free_cash_flow": _get("free_cash_flow"),
        "revenue_growth": _get("revenue_growth"),
        "earnings_growth": _get("earnings_growth"),
        "profit_margin": _get("profit_margin") or _get("net_profit_margin"),
        "sector": _get("sector", ""),
        "industry": _get("industry", ""),
        "beta": _get("beta"),
        "52w_high": _get("fifty_two_week_high") or _get("year_high"),
        "52w_low": _get("fifty_two_week_low") or _get("year_low"),
        "avg_volume": _get("average_volume") or _get("avg_volume"),
    }


def get_bulk_fundamentals(tickers: list[str], provider: Optional[str] = None) -> pd.DataFrame:
    """Fetch fundamentals for multiple tickers. Returns DataFrame."""
    records = []
    for t in tickers:
        records.append(get_fundamentals(t, provider=provider))
    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# MACRO / ECONOMIC DATA — OpenBB FRED + Federal Reserve
# ═══════════════════════════════════════════════════════════════════════════

# Key macro ETF proxies (when FRED not available)
MACRO_PROXIES = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "TLT": "20Y Treasury",
    "IEF": "7-10Y Treasury",
    "SHY": "1-3Y Treasury",
    "LQD": "IG Corporate",
    "HYG": "HY Corporate",
    "GLD": "Gold",
    "USO": "Crude Oil",
    "DXY": "US Dollar Index",
    "^VIX": "VIX",
    "^TNX": "10Y Yield",
    "^FVX": "5Y Yield",
    "^IRX": "3M T-Bill",
}

# FRED series IDs for direct macro data (replaces ETF proxies where possible)
FRED_SERIES = {
    "M2SL": "M2 Money Supply",
    "GDP": "GDP (Nominal)",
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "DFF": "Fed Funds Effective Daily",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "DGS5": "5Y Treasury Yield",
    "DGS30": "30Y Treasury Yield",
    "DTB3": "3M T-Bill Rate",
    "BAMLH0A0HYM2": "HY OAS Spread",
    "BAMLC0A4CBBB": "BBB OAS Spread",
    "WALCL": "Fed Balance Sheet (Total Assets)",
    "RRPONTSYD": "ON-RRP Balance",
    "WTREGEN": "Treasury General Account",
    "SOFR": "SOFR Rate",
    "TEDRATE": "TED Spread",
    "CPIAUCSL": "CPI (All Urban)",
    "PCEPI": "PCE Price Index",
    "VIXCLS": "VIX (FRED)",
    "DTWEXBGS": "Trade-Weighted Dollar Index",
    "DCOILWTICO": "WTI Crude Oil",
    "GOLDAMGBD228NLBM": "Gold (London PM Fix)",
}


def get_fred_series(
    series_ids: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch FRED economic data series via OpenBB.

    Returns DataFrame with columns named by series ID.
    Falls back to ETF proxies if FRED unavailable.
    """
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if not _openbb_available:
        logger.warning("OpenBB unavailable — cannot fetch FRED series")
        return pd.DataFrame()

    frames = {}
    for sid in series_ids:
        try:
            result = _obb.economy.fred_series(
                symbol=sid,
                start_date=start,
                end_date=end,
                provider="fred",
            )
            df = _obbject_to_dataframe(result)
            if not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                # Use the value column
                val_col = "value" if "value" in df.columns else df.columns[0]
                frames[sid] = df[val_col]
        except Exception as e:
            logger.debug(f"FRED series {sid} fetch failed: {e}")

    if frames:
        return pd.DataFrame(frames)
    return pd.DataFrame()


def get_treasury_rates(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch Treasury yield curve data via OpenBB FRED.

    Returns DataFrame with columns: DGS2, DGS5, DGS10, DGS30, DTB3.
    """
    series = ["DGS2", "DGS5", "DGS10", "DGS30", "DTB3"]
    return get_fred_series(series, start, end)


def get_fed_balance_sheet(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch Fed balance sheet components via FRED.

    Returns DataFrame with WALCL (total assets), RRPONTSYD (ON-RRP),
    WTREGEN (TGA).
    """
    series = ["WALCL", "RRPONTSYD", "WTREGEN"]
    return get_fred_series(series, start, end)


def get_credit_spreads(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch credit spread data (HY OAS, BBB OAS) via FRED."""
    series = ["BAMLH0A0HYM2", "BAMLC0A4CBBB"]
    return get_fred_series(series, start, end)


def get_monetary_data(
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch M2, GDP, unemployment, CPI — core GMTF inputs via FRED."""
    series = ["M2SL", "GDP", "UNRATE", "FEDFUNDS", "CPIAUCSL"]
    return get_fred_series(series, start, end)


def get_sofr_rate(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch SOFR rate via OpenBB."""
    if _openbb_available:
        try:
            result = _obb.fixedincome.rate.sofr(
                start_date=start,
                end_date=end,
                provider="fred",
            )
            df = _obbject_to_dataframe(result)
            if not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                return df
        except Exception as e:
            logger.debug(f"SOFR fetch failed: {e}")
    # Fallback to FRED series
    return get_fred_series("SOFR", start, end)


def get_effr_rate(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch Effective Federal Funds Rate via OpenBB."""
    if _openbb_available:
        try:
            result = _obb.fixedincome.rate.effr(
                start_date=start,
                end_date=end,
                provider="fred",
            )
            return _obbject_to_dataframe(result)
        except Exception as e:
            logger.debug(f"EFFR fetch failed: {e}")
    return get_fred_series("DFF", start, end)


def get_cpi(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    provider: str = "fred",
) -> pd.DataFrame:
    """Fetch Consumer Price Index via OpenBB."""
    if _openbb_available:
        try:
            result = _obb.economy.cpi(
                start_date=start,
                end_date=end,
                provider=provider,
            )
            return _obbject_to_dataframe(result)
        except Exception as e:
            logger.debug(f"CPI fetch failed: {e}")
    return get_fred_series("CPIAUCSL", start, end)


def get_unemployment(
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch unemployment rate via OpenBB."""
    if _openbb_available:
        try:
            result = _obb.economy.unemployment(
                start_date=start,
                end_date=end,
                provider="oecd",
            )
            return _obbject_to_dataframe(result)
        except Exception as e:
            logger.debug(f"Unemployment fetch failed: {e}")
    return get_fred_series("UNRATE", start, end)


# ═══════════════════════════════════════════════════════════════════════════
# MACRO DATA (ETF proxy path — preserves existing interface)
# ═══════════════════════════════════════════════════════════════════════════

def get_macro_data(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch macro proxy data via ETFs/indices (existing interface preserved).

    Now also enriches with FRED data when available.
    Batches requests to stay within FMP free tier limits.
    """
    tickers = list(MACRO_PROXIES.keys())
    # Batch: FMP free tier limits symbols per request
    batch_size = 10
    all_prices = pd.DataFrame()
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            prices = get_adj_close(batch, start, end)
            if not prices.empty:
                all_prices = pd.concat([all_prices, prices], axis=1)
        except Exception as e:
            logger.debug(f"Macro batch {i//batch_size} failed: {e}")

    if not all_prices.empty:
        all_prices.columns = [MACRO_PROXIES.get(c, c) for c in all_prices.columns]
    return all_prices


def get_macro_data_enriched(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch enriched macro data: ETF proxies + FRED direct series.

    Returns dict with keys:
        'etf_proxies': DataFrame of ETF-based macro proxies
        'treasury_yields': DataFrame of actual yield curve data
        'credit_spreads': DataFrame of HY/IG spreads
        'fed_balance_sheet': DataFrame of WALCL, ON-RRP, TGA
        'monetary': DataFrame of M2, GDP, unemployment, Fed funds, CPI
    """
    result = {}
    result["etf_proxies"] = get_macro_data(start, end)
    result["treasury_yields"] = get_treasury_rates(start, end)
    result["credit_spreads"] = get_credit_spreads(start, end)
    result["fed_balance_sheet"] = get_fed_balance_sheet(start, end)
    result["monetary"] = get_monetary_data(start, end)
    return result


def get_sector_performance(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Sector ETF performance matrix."""
    from .universe_engine import SECTOR_ETFS
    etfs = list(SECTOR_ETFS.values())
    prices = get_adj_close(etfs, start, end)
    if not prices.empty:
        inv = {v: k for k, v in SECTOR_ETFS.items()}
        prices.columns = [inv.get(c, c) for c in prices.columns]
    return prices


# ═══════════════════════════════════════════════════════════════════════════
# SEC FILINGS — via OpenBB SEC provider
# ═══════════════════════════════════════════════════════════════════════════

def get_company_filings(
    ticker: str,
    filing_type: str = "10-K",
    limit: int = 5,
) -> pd.DataFrame:
    """Fetch SEC filings for a company via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.equity.fundamental.filings(
            symbol=ticker,
            type=filing_type,
            limit=limit,
            provider="sec",
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"SEC filings fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def get_insider_trading(
    ticker: str,
    limit: int = 50,
) -> pd.DataFrame:
    """Fetch insider trading data via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.equity.ownership.insider_trading(
            symbol=ticker,
            limit=limit,
            provider="sec",
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"Insider trading fetch failed for {ticker}: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# NEWS — via OpenBB multi-provider
# ═══════════════════════════════════════════════════════════════════════════

def get_company_news(
    ticker: str,
    limit: int = 20,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch company news via OpenBB (default: FMP — configured key)."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.news.company(
            symbol=ticker,
            limit=limit,
            provider=provider,
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"News fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def get_world_news(
    limit: int = 30,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch world/market news via OpenBB (default: FMP — configured key)."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.news.world(
            limit=limit,
            provider=provider,
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"World news fetch failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONS — via OpenBB (CBOE, Intrinio)
# ═══════════════════════════════════════════════════════════════════════════

def get_options_chains(
    ticker: str,
    provider: str = "cboe",
) -> pd.DataFrame:
    """Fetch options chain data via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.derivatives.options.chains(
            symbol=ticker,
            provider=provider,
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"Options chain fetch failed for {ticker}: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# ETF DATA — via OpenBB
# ═══════════════════════════════════════════════════════════════════════════

def get_etf_holdings(
    ticker: str,
    provider: str = "sec",
) -> pd.DataFrame:
    """Fetch ETF holdings via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.etf.holdings(
            symbol=ticker,
            provider=provider,
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"ETF holdings fetch failed for {ticker}: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# ECONOMIC CALENDAR — via OpenBB
# ═══════════════════════════════════════════════════════════════════════════

def get_economic_calendar(
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "nasdaq",
) -> pd.DataFrame:
    """Fetch economic calendar via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        kwargs = {"provider": provider}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        result = _obb.economy.calendar(**kwargs)
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"Economic calendar fetch failed: {e}")
    return pd.DataFrame()


def get_earnings_calendar(
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Upcoming earnings dates via OpenBB (columns include symbol + report_date).

    Used by the event tail hedge to find positions with earnings before the
    hedge horizon. Returns an empty frame when OpenBB is unavailable.
    """
    if not _openbb_available:
        return pd.DataFrame()
    try:
        kwargs = {"provider": provider}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        result = _obb.equity.calendar.earnings(**kwargs)
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"Earnings calendar fetch failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# EQUITY QUOTE — bid/ask/volume via OpenBB (FMP provider)
# ═══════════════════════════════════════════════════════════════════════════

def get_quote(tickers: list[str] | str, provider: Optional[str] = None) -> list[dict]:
    """Get real-time quotes with bid/ask for order book analysis.

    FMP provides: price, bid, ask, volume, open, high, low, previous close.
    Used by TCA tab for execution quality benchmarking.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    prov = provider or DEFAULT_EQUITY_PROVIDER

    if _openbb_available:
        try:
            result = _obb.equity.price.quote(
                symbol=",".join(tickers),
                provider=prov,
            )
            df = _obbject_to_dataframe(result)
            if not df.empty:
                records = df.to_dict(orient="records")
                return records
        except Exception as e:
            logger.debug(f"Quote fetch failed: {e}")

    return []


# ═══════════════════════════════════════════════════════════════════════════
# EQUITY SCREENER — via OpenBB
# ═══════════════════════════════════════════════════════════════════════════

def search_equities(
    query: str = "",
    provider: str = "nasdaq",
) -> pd.DataFrame:
    """Search for equities via OpenBB."""
    if not _openbb_available:
        return pd.DataFrame()
    try:
        result = _obb.equity.search(
            query=query,
            provider=provider,
        )
        return _obbject_to_dataframe(result)
    except Exception as e:
        logger.debug(f"Equity search failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STATUS / DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def get_data_source_status() -> dict:
    """Return status of all data source backends."""
    status = {
        "openbb_available": _openbb_available,
        "ibkr_data_available": _IBKR_DATA_AVAILABLE,
        "primary_equity_source": "ibkr" if _use_ibkr_for_prices() else DEFAULT_EQUITY_PROVIDER,
        "active_data_source": get_active_data_source(),
        "data_source_mode": _DATA_SOURCE_MODE,
        "equity_fallback_provider": DEFAULT_EQUITY_PROVIDER,
        "macro_provider": DEFAULT_MACRO_PROVIDER if _openbb_available else "etf_proxy",
        "fundamental_provider": DEFAULT_FUNDAMENTAL_PROVIDER,
    }

    # Check FRED availability
    if _openbb_available:
        try:
            _obb.economy.fred_series(symbol="DFF", limit=1, provider="fred")
            status["fred_available"] = True
        except Exception:
            status["fred_available"] = False
    else:
        status["fred_available"] = False

    return status
