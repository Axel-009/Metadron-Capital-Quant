"""
Stock history data module for ML-Macro-Market.

Uses the unified OpenBB universe as the SOLE data source.
No yfinance, no Yahoo CSV, no urllib scraping.

Usage:
    from common.stockhistory import Get_OpenBB, Get_Historical, Get_Multiple

    df = Get_Historical("AAPL", start="2020-01-01", end="2024-01-01")
    dfs = Get_Multiple(["AAPL", "MSFT", "GOOG"])
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to title-case and ensure Date index."""
    rename_map = {}
    for col in df.columns:
        canonical = col.strip().lower()
        if canonical == "open":
            rename_map[col] = "Open"
        elif canonical == "high":
            rename_map[col] = "High"
        elif canonical == "low":
            rename_map[col] = "Low"
        elif canonical in ("close", "adj close", "adj_close", "adjclose"):
            rename_map[col] = "Close"
        elif canonical in ("adj close", "adj_close", "adjclose"):
            rename_map[col] = "Adj Close"
        elif canonical == "volume":
            rename_map[col] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure Date index
    if df.index.name != "Date":
        if "date" in df.columns:
            df = df.set_index("date")
        elif "Date" in df.columns:
            df = df.set_index("Date")
        df.index.name = "Date"

    return df


def Get_OpenBB(
    symbol: str = "AAPL",
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    provider: str = "openbb",
) -> pd.DataFrame:
    """
    Fetch historical price data via OpenBB (sole data source).

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL", "^GSPC", "BTC-USD").
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    provider : str
        OpenBB provider backend (default "openbb").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with OHLCV columns.

    Raises
    ------
    RuntimeError
        If OpenBB cannot retrieve data for the symbol.
    """
    from openbb import obb

    logger.info("Fetching %s via OpenBB (provider=%s)", symbol, provider)

    # Detect asset class for routing
    sym = symbol.upper().strip()
    try:
        if sym.startswith("^"):
            result = obb.index.price.historical(
                symbol=sym.replace("^", ""),
                start_date=start,
                end_date=end,
                provider=provider,
            )
        elif sym.endswith("-USD") or sym.endswith("-USDT"):
            result = obb.crypto.price.historical(
                symbol=sym.replace("-USD", "").replace("-USDT", ""),
                start_date=start,
                end_date=end,
                provider=provider,
            )
        elif sym.endswith("=X"):
            result = obb.currency.price.historical(
                symbol=sym.replace("=X", ""),
                start_date=start,
                end_date=end,
                provider=provider,
            )
        elif sym.endswith("=F"):
            result = obb.equity.price.historical(
                symbol=sym,
                start_date=start,
                end_date=end,
                provider=provider,
            )
        else:
            result = obb.equity.price.historical(
                symbol=sym,
                start_date=start,
                end_date=end,
                provider=provider,
            )

        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"OpenBB returned empty DataFrame for {symbol}")

        logger.info("OpenBB returned %d rows for %s", len(df), symbol)
        return _normalise(df)

    except Exception as exc:
        raise RuntimeError(f"OpenBB failed for {symbol}: {exc}") from exc


def Get_Historical(
    symbol: str = "AAPL",
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> pd.DataFrame:
    """
    Fetch historical price data. OpenBB is the sole source.

    This is the primary public API -- equivalent to the legacy Get_Yahoo()
    function but using OpenBB exclusively.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str, optional
        Start date YYYY-MM-DD. Defaults to 1 year ago.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    provider : str
        OpenBB provider backend.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with OHLCV columns.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    return Get_OpenBB(symbol, start, end, provider=provider)


def Get_Multiple(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols.

    Parameters
    ----------
    symbols : list of str
    start, end, provider
        Passed to Get_Historical().

    Returns
    -------
    dict
        symbol -> DataFrame. Failed symbols are logged and omitted.
    """
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            results[sym] = Get_Historical(sym, start=start, end=end, provider=provider)
        except Exception as exc:
            logger.error("Skipping %s: %s", sym, exc)
    return results


def Get_Macro(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "openbb",
) -> pd.DataFrame:
    """
    Fetch macroeconomic indicator data via OpenBB FRED integration.

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g., "GDP", "UNRATE", "CPIAUCSL").
    start, end : str, optional
    provider : str

    Returns
    -------
    pd.DataFrame
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    from openbb import obb

    try:
        result = obb.economy.fred_series(
            symbol=series_id,
            start_date=start,
            end_date=end,
            provider=provider,
        )
        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"No macro data for {series_id}")
        return df
    except Exception as exc:
        raise RuntimeError(f"OpenBB macro data failed for {series_id}: {exc}") from exc


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

# Legacy function names pointing to OpenBB-only implementations
Get_Yahoo = Get_Historical
Get_Yahoo_CSV = Get_Historical  # CSV fallback removed; uses OpenBB


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Stock history module -- OpenBB sole source")
    print("Functions: Get_Historical, Get_OpenBB, Get_Multiple, Get_Macro")
    print("Legacy aliases: Get_Yahoo -> Get_Historical")
