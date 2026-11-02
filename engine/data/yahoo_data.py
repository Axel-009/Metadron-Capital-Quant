"""DEPRECATED — Redirected to FMP via OpenBB per platform policy.

Yahoo Finance is not an authorized API. All data calls are forwarded
to engine.data.openbb_data (FMP provider) which is the sole authorized
market data source.

New code MUST import from engine.data.openbb_data directly.
This shim exists only so existing imports (e.g. universal_pooling.py)
continue to work without modification.
"""
import logging as _logging

_logger = _logging.getLogger(__name__)
_logger.debug("yahoo_data imported — Redirected to FMP via OpenBB per platform policy")

# Direct re-exports from openbb_data (FMP provider) — sole authorized data source.
from .openbb_data import (  # noqa: F401
    get_prices,
    get_adj_close,
    get_returns,
    get_market_stats,
    get_fundamentals,
    get_bulk_fundamentals,
    get_macro_data,
    get_sector_performance,
    MACRO_PROXIES,
    get_fred_series,
    get_treasury_rates,
    get_fed_balance_sheet,
    get_credit_spreads,
    get_monetary_data,
    get_sofr_rate,
    get_effr_rate,
    get_cpi,
    get_unemployment,
    get_macro_data_enriched,
    get_company_filings,
    get_insider_trading,
    get_company_news,
    get_world_news,
    get_options_chains,
    get_etf_holdings,
    get_economic_calendar,
    search_equities,
    get_data_source_status,
    FRED_SERIES,
)
