"""OpenBB data provider adapter for ai-hedgefund.

Wraps OpenBB SDK calls to return data in the same Pydantic model format
that the existing agents expect (Price, FinancialMetrics, InsiderTrade, etc.).

Falls back to the original Financial Datasets API when OpenBB is unavailable.
"""

import logging
import os
from datetime import datetime

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
    logger.info("OpenBB SDK loaded — primary data source active for ai-hedgefund")
except ImportError:
    logger.warning("OpenBB SDK not available — using Financial Datasets API fallback")

from src.data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews,
    LineItem,
)


def _obbject_to_df(result) -> pd.DataFrame:
    """Convert OpenBB OBBject to DataFrame."""
    if result is None:
        return pd.DataFrame()
    try:
        df = result.to_dataframe()
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        if hasattr(result, "results") and result.results:
            return pd.DataFrame([dict(r) for r in result.results])
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# PRICE DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_prices_openbb(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data via OpenBB, returning Price model objects."""
    if not _openbb_available:
        return []

    try:
        result = _obb.equity.price.historical(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            provider="yfinance",
        )
        df = _obbject_to_df(result)
        if df.empty:
            return []

        prices = []
        for idx, row in df.iterrows():
            dt = idx if isinstance(idx, (datetime, pd.Timestamp)) else pd.to_datetime(row.get("date", idx))
            prices.append(Price(
                open=float(row.get("open", 0)),
                close=float(row.get("close", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                volume=int(row.get("volume", 0)),
                time=dt.strftime("%Y-%m-%dT%H:%M:%S"),
            ))
        return prices
    except Exception as e:
        logger.debug(f"OpenBB price fetch failed for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# FINANCIAL METRICS
# ═══════════════════════════════════════════════════════════════════════════

def get_financial_metrics_openbb(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics via OpenBB, returning FinancialMetrics model objects."""
    if not _openbb_available:
        return []

    try:
        result = _obb.equity.fundamental.metrics(
            symbol=ticker,
            provider="yfinance",
        )
        df = _obbject_to_df(result)
        if df.empty:
            return []

        metrics_list = []
        for _, row in df.head(limit).iterrows():
            def _g(key, default=None):
                v = row.get(key, default)
                if v is not None and isinstance(v, float) and pd.isna(v):
                    return default
                return v

            metrics_list.append(FinancialMetrics(
                ticker=ticker,
                report_period=_g("report_period", end_date),
                period=period,
                currency=_g("currency", "USD"),
                market_cap=_g("market_cap"),
                enterprise_value=_g("enterprise_value"),
                price_to_earnings_ratio=_g("pe_ratio") or _g("price_to_earnings_ratio"),
                price_to_book_ratio=_g("pb_ratio") or _g("price_to_book_ratio"),
                price_to_sales_ratio=_g("price_to_sales_ratio"),
                enterprise_value_to_ebitda_ratio=_g("enterprise_value_to_ebitda_ratio"),
                enterprise_value_to_revenue_ratio=_g("enterprise_value_to_revenue_ratio"),
                free_cash_flow_yield=_g("free_cash_flow_yield"),
                peg_ratio=_g("peg_ratio"),
                gross_margin=_g("gross_margin"),
                operating_margin=_g("operating_margin"),
                net_margin=_g("net_margin") or _g("net_profit_margin"),
                return_on_equity=_g("return_on_equity"),
                return_on_assets=_g("return_on_assets"),
                return_on_invested_capital=_g("return_on_invested_capital"),
                asset_turnover=_g("asset_turnover"),
                inventory_turnover=_g("inventory_turnover"),
                receivables_turnover=_g("receivables_turnover"),
                days_sales_outstanding=_g("days_sales_outstanding"),
                operating_cycle=_g("operating_cycle"),
                working_capital_turnover=_g("working_capital_turnover"),
                current_ratio=_g("current_ratio"),
                quick_ratio=_g("quick_ratio"),
                cash_ratio=_g("cash_ratio"),
                operating_cash_flow_ratio=_g("operating_cash_flow_ratio"),
                debt_to_equity=_g("debt_to_equity"),
                debt_to_assets=_g("debt_to_assets"),
                interest_coverage=_g("interest_coverage"),
                revenue_growth=_g("revenue_growth"),
                earnings_growth=_g("earnings_growth"),
                book_value_growth=_g("book_value_growth"),
                earnings_per_share_growth=_g("earnings_per_share_growth"),
                free_cash_flow_growth=_g("free_cash_flow_growth"),
                operating_income_growth=_g("operating_income_growth"),
                ebitda_growth=_g("ebitda_growth"),
                payout_ratio=_g("payout_ratio"),
                earnings_per_share=_g("earnings_per_share"),
                book_value_per_share=_g("book_value_per_share"),
                free_cash_flow_per_share=_g("free_cash_flow_per_share"),
            ))
        return metrics_list
    except Exception as e:
        logger.debug(f"OpenBB metrics fetch failed for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# INSIDER TRADES
# ═══════════════════════════════════════════════════════════════════════════

def get_insider_trades_openbb(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 100,
) -> list[InsiderTrade]:
    """Fetch insider trades via OpenBB (SEC provider)."""
    if not _openbb_available:
        return []

    try:
        result = _obb.equity.ownership.insider_trading(
            symbol=ticker,
            limit=limit,
            provider="sec",
        )
        df = _obbject_to_df(result)
        if df.empty:
            return []

        trades = []
        for _, row in df.iterrows():
            filing_date = str(row.get("filing_date", row.get("date", "")))
            if not filing_date:
                continue
            # Filter by date range
            if end_date and filing_date > end_date:
                continue
            if start_date and filing_date < start_date:
                continue

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=row.get("issuer", None),
                name=row.get("owner_name", row.get("name", None)),
                title=row.get("owner_title", row.get("title", None)),
                is_board_director=row.get("is_board_director", None),
                transaction_date=str(row.get("transaction_date", "")),
                transaction_shares=row.get("securities_transacted", row.get("transaction_shares", None)),
                transaction_price_per_share=row.get("price", row.get("transaction_price_per_share", None)),
                transaction_value=row.get("transaction_value", None),
                shares_owned_before_transaction=row.get("shares_owned_before_transaction", None),
                shares_owned_after_transaction=row.get("securities_owned", row.get("shares_owned_after_transaction", None)),
                security_title=row.get("security_title", None),
                filing_date=filing_date,
            ))
        return trades[:limit]
    except Exception as e:
        logger.debug(f"OpenBB insider trades fetch failed for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# COMPANY NEWS
# ═══════════════════════════════════════════════════════════════════════════

def get_company_news_openbb(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 50,
) -> list[CompanyNews]:
    """Fetch company news via OpenBB (Tiingo, Benzinga, FMP, etc.)."""
    if not _openbb_available:
        return []

    for provider in ["tiingo", "benzinga", "fmp"]:
        try:
            kwargs = {"symbol": ticker, "limit": limit, "provider": provider}
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            result = _obb.news.company(**kwargs)
            df = _obbject_to_df(result)
            if df.empty:
                continue

            news_items = []
            for _, row in df.iterrows():
                news_items.append(CompanyNews(
                    ticker=ticker,
                    title=str(row.get("title", "")),
                    author=str(row.get("author", row.get("author_name", ""))),
                    source=str(row.get("source", provider)),
                    date=str(row.get("date", row.get("published", ""))),
                    url=str(row.get("url", row.get("article_url", ""))),
                    sentiment=row.get("sentiment", None),
                ))
            return news_items[:limit]
        except Exception as e:
            logger.debug(f"OpenBB news fetch ({provider}) failed for {ticker}: {e}")
            continue
    return []


# ═══════════════════════════════════════════════════════════════════════════
# MARKET CAP
# ═══════════════════════════════════════════════════════════════════════════

def get_market_cap_openbb(ticker: str) -> float | None:
    """Fetch current market cap via OpenBB."""
    if not _openbb_available:
        return None

    try:
        result = _obb.equity.fundamental.metrics(
            symbol=ticker,
            provider="yfinance",
        )
        df = _obbject_to_df(result)
        if not df.empty:
            mc = df.iloc[0].get("market_cap")
            if mc is not None and not (isinstance(mc, float) and pd.isna(mc)):
                return float(mc)
    except Exception as e:
        logger.debug(f"OpenBB market cap fetch failed for {ticker}: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════

def is_openbb_available() -> bool:
    """Check if OpenBB is available."""
    return _openbb_available
