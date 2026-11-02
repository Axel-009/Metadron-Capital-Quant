"""DEPRECATED — AlpacaBroker is no longer used for execution.

IBKR is the sole execution broker (see ibkr_broker.py).
This file is retained ONLY because paper_broker.py data types (Order,
OrderSide, Position, etc.) are imported from it by other modules.
Do NOT use AlpacaBroker for new code. All orders route through
L7UnifiedExecutionSurface → IBKRBroker.

Original: Alpaca Brokerage API (alpaca-py SDK). Implements the same
interface so ExecutionEngine can swap between paper and Alpaca
with a single config toggle.

Environment variables:
    ALPACA_API_KEY       — API key ID
    ALPACA_SECRET_KEY    — API secret key
    ALPACA_PAPER_TRADE   — "True" (default) for paper trading, "False" for live

Alpaca SDK docs: https://alpaca.markets/docs/python/

Commission: $0 stock trades, $0 option trades (Alpaca commission-free).
"""

import os
import json
import time
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

# Load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from .paper_broker import (
    Order, OrderSide, OrderType, OrderStatus, SignalType,
    Position, PortfolioState, RiskLimiter, PerformanceTracker,
    DailyTargetManager, LiveDashboardState, RiskProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca SDK imports (lazy-safe)
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        OrderType as AlpacaOrderType,
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        ClosePositionRequest,
        GetOrdersRequest,
        GetPortfolioHistoryRequest,
    )
    from alpaca.common.exceptions import APIError as AlpacaAPIError
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning("alpaca-py SDK not installed. pip install alpaca-py")

# ---------------------------------------------------------------------------
# Constants — mappings
# ---------------------------------------------------------------------------
_ALPACA_BASE_URLS = {
    True: "https://paper-api.alpaca.markets",
    False: "https://api.alpaca.markets",
}

# Metadron OrderSide → Alpaca OrderSide
_SIDE_TO_ALPACA = {}
# Alpaca order side string → Metadron OrderSide
_ALPACA_SIDE_MAP = {}

# Metadron OrderType → Alpaca OrderType
_TYPE_TO_ALPACA = {}

# Alpaca order status string → Metadron OrderStatus (no SDK dependency)
_ALPACA_STATUS_MAP = {
    "new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PENDING,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "accepted": OrderStatus.PENDING,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "stopped": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.CANCELLED,
    "calculated": OrderStatus.PENDING,
}

if _ALPACA_AVAILABLE:
    _SIDE_TO_ALPACA = {
        OrderSide.BUY: AlpacaOrderSide.BUY,
        OrderSide.SELL: AlpacaOrderSide.SELL,
        OrderSide.SHORT: AlpacaOrderSide.SELL,   # Alpaca handles short via sell without shares
        OrderSide.COVER: AlpacaOrderSide.BUY,     # Buy to cover
    }
    _ALPACA_SIDE_MAP = {
        "buy": OrderSide.BUY,
        "sell": OrderSide.SELL,
    }
    _TYPE_TO_ALPACA = {
        OrderType.MARKET: AlpacaOrderType.MARKET,
        OrderType.LIMIT: AlpacaOrderType.LIMIT,
    }


# ---------------------------------------------------------------------------
# AlpacaBroker — Drop-in replacement for PaperBroker
# ---------------------------------------------------------------------------
class AlpacaBroker:
    """Live execution broker via Alpaca API.

    Implements the same interface as PaperBroker so
    ExecutionEngine can swap between paper and live with zero code changes.

    Paper mode uses Alpaca's paper trading endpoint. Live mode uses the
    production endpoint. Both are commission-free ($0 stock trades).
    """

    MAX_RETRIES = 4
    BACKOFF_BASE = 2  # seconds

    def __init__(
        self,
        initial_cash: float = 1_000.0,
        log_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        enable_risk_limits: bool = True,
        daily_target_pct: float = 0.05,
    ):
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py SDK is required for AlpacaBroker. "
                "Install with: pip install alpaca-py"
            )

        # Resolve credentials
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        # FAIL FAST — never silently degrade
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY required. "
                "Set in .env or pass directly. "
                "For paper trading, use PaperBroker explicitly with broker_type='paper'."
            )

        if paper is None:
            paper_str = os.environ.get("ALPACA_PAPER_TRADE", "")
            if not paper_str:
                raise RuntimeError(
                    "ALPACA_PAPER_TRADE not set — must be explicitly 'True' (paper) or 'False' (live). "
                    "This is a safety requirement. Set it in .env or the API Vault."
                )
            self.paper = paper_str.lower() not in ("false", "0", "no", "off")
        else:
            self.paper = paper

        # Initialize Alpaca trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        # Initialize stock data client for real-time quotes
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        # Validate credentials with lightweight account call
        try:
            _acct = self.trading_client.get_account()
            self._account_number = _acct.account_number
            logger.info("Alpaca connected: account %s, equity $%s, paper=%s",
                        _acct.account_number, _acct.equity, self.paper)
        except Exception as e:
            # Sanitize error — never log full exception (may contain auth headers)
            status = getattr(e, 'status_code', 'unknown')
            raise ConnectionError(
                f"Alpaca credential validation failed (status={status}). "
                "Check your API key and secret key."
            ) from None

        # Portfolio state — synced from Alpaca on each call
        self.state = PortfolioState(cash=initial_cash, nav=initial_cash)

        # Logging
        self.log_dir = log_dir or Path("logs/alpaca_broker")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._orders: list[Order] = []
        self._trade_log: list[dict] = []

        # Risk & performance (reuse from PaperBroker)
        self._initial_cash = initial_cash
        self._risk_limiter = RiskLimiter() if enable_risk_limits else None
        self._perf_tracker = PerformanceTracker(initial_nav=initial_cash)
        self._daily_pnl_today: float = 0.0
        self._last_eod_date: Optional[str] = None
        self._eod_nav_history: list[tuple[str, float]] = []

        # Daily target manager — 5% compound daily minimum
        self._target_manager = DailyTargetManager(initial_nav=initial_cash)
        self._target_manager.DAILY_TARGET_PCT = daily_target_pct

        # Live dashboard state emitter
        self._dashboard = LiveDashboardState()

        # Order ID mapping: local_id → alpaca_order_id
        self._order_id_map: dict[str, str] = {}

        # Quote cache: ticker → (timestamp, price)
        self._quote_cache: dict[str, tuple[float, float]] = {}
        self._quote_cache_ttl = 5.0  # seconds

        # Sync initial state from Alpaca
        try:
            self._sync_account()
            self._sync_positions()
            logger.info(
                "AlpacaBroker initialized (paper=%s): cash=$%.2f, %d positions",
                self.paper, self.state.cash, len(self.state.positions),
            )
        except Exception as e:
            logger.warning("AlpacaBroker init sync failed (offline?): %s", e)

    # --- Alpaca state sync ---------------------------------------------------

    def _sync_account(self):
        """Sync cash and NAV from Alpaca account."""
        account = self.trading_client.get_account()
        self.state.cash = float(account.cash)
        self.state.nav = float(account.portfolio_value)

    def _sync_positions(self):
        """Sync positions from Alpaca."""
        alpaca_positions = self.trading_client.get_all_positions()
        synced: dict[str, Position] = {}
        for ap in alpaca_positions:
            ticker = ap.symbol
            qty = int(ap.qty)
            avg_cost = float(ap.avg_entry_price)
            current_price = float(ap.current_price or ap.avg_entry_price)
            unrealized = float(ap.unrealized_pl or 0)
            synced[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                unrealized_pnl=unrealized,
                realized_pnl=0.0,
            )
        self.state.positions = synced

    # --- Pre-trade risk checks (same as PaperBroker) -------------------------

    def _check_risk_limits(self, ticker: str, side: OrderSide,
                           quantity: int, price: float) -> tuple[bool, str]:
        if self._risk_limiter is None:
            return True, ""
        order_value = quantity * price
        nav = self.state.nav if self.state.nav > 0 else self._initial_cash
        sector = ""
        if ticker in self.state.positions:
            sector = self.state.positions[ticker].sector
        exposures = self.compute_exposures()
        gross = exposures.get("gross", 0.0)
        net = exposures.get("net", 0.0)
        passed, failures = self._risk_limiter.run_all_checks(
            order_value=order_value, ticker=ticker, sector=sector,
            positions=self.state.positions, nav=nav,
            daily_pnl=self._daily_pnl_today,
            gross_exposure=gross, net_exposure=net,
        )
        if not passed:
            return False, "; ".join(failures)
        return True, ""

    # --- Retry helper --------------------------------------------------------

    def _retry_request(self, func, *args, **kwargs):
        """Execute an Alpaca SDK call with retry + exponential backoff."""
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except AlpacaAPIError as e:
                status = getattr(e, "status_code", 0)
                if status and status < 500 and status != 429:
                    logger.error("Alpaca API error: status=%d", status)
                    raise
                last_exc = e
                wait = self.BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Alpaca request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, self.MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                wait = self.BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Alpaca request error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, self.MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
        raise ConnectionError(
            f"Alpaca request failed after {self.MAX_RETRIES} retries"
        ) from last_exc

    # --- Order execution (LIVE via Alpaca API) -------------------------------

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """Place a live order via Alpaca API.

        Maintains the same signature as PaperBroker.place_order().
        When stop_loss/take_profit are provided, submits a bracket order
        (entry + stop + take-profit as a single atomic OCO group).
        """
        local_id = str(uuid.uuid4())[:8]
        order = Order(
            id=local_id,
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=quantity,
            limit_price=limit_price,
            signal_type=signal_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Get current price for risk checks
        price = self._get_current_price(ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"No price data for {ticker}"
            self._orders.append(order)
            return order

        # Update daily target manager
        nav = self.compute_nav()
        self._target_manager.update(nav)

        # Check if new positions allowed under current risk profile
        if side in (OrderSide.BUY, OrderSide.SHORT):
            if not self._target_manager.allow_new_positions():
                order.status = OrderStatus.REJECTED
                order.reason = (
                    f"DEFENSIVE mode — daily target exceeded "
                    f"({self._target_manager.profile.value}), no new positions"
                )
                self._orders.append(order)
                return order

            # Apply position size multiplier from risk profile
            pos_mult = self._target_manager.get_position_multiplier()
            if pos_mult < 1.0:
                quantity = max(1, int(quantity * pos_mult))
                order.quantity = quantity

        # Pre-trade risk checks
        risk_ok, risk_reason = self._check_risk_limits(ticker, side, quantity, price)
        if not risk_ok:
            order.status = OrderStatus.REJECTED
            order.reason = f"Risk limit: {risk_reason}"
            self._orders.append(order)
            return order

        # --- Submit to Alpaca API ---
        try:
            alpaca_side = _SIDE_TO_ALPACA[side]

            # Build bracket order kwargs if stop/take-profit provided
            bracket_kwargs = {}
            if stop_loss and stop_loss > 0:
                bracket_kwargs["stop_loss"] = {"stop_price": round(stop_loss, 2)}
            if take_profit and take_profit > 0:
                bracket_kwargs["take_profit"] = {"limit_price": round(take_profit, 2)}
            # Bracket requires class=bracket with both legs, or OTO with one
            order_class_str = None
            if bracket_kwargs:
                if "stop_loss" in bracket_kwargs and "take_profit" in bracket_kwargs:
                    order_class_str = "bracket"
                else:
                    order_class_str = "oto"

            if limit_price:
                req = LimitOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    **({"order_class": order_class_str, **bracket_kwargs} if order_class_str else {}),
                )
            else:
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    **({"order_class": order_class_str, **bracket_kwargs} if order_class_str else {}),
                )

            result = self._retry_request(self.trading_client.submit_order, req)
            alpaca_order_id = str(result.id)
            if not alpaca_order_id:
                order.status = OrderStatus.REJECTED
                order.reason = f"Alpaca returned no order ID: {result}"
                self._orders.append(order)
                return order

            self._order_id_map[local_id] = alpaca_order_id
            order.status = OrderStatus.PENDING

            # Poll for fill
            filled_order = self._poll_order_fill(alpaca_order_id, timeout=15.0)
            if filled_order:
                fill_price = float(getattr(filled_order, "filled_avg_price", 0) or 0)
                order.fill_price = fill_price if fill_price > 0 else price
                order.fill_timestamp = (
                    str(getattr(filled_order, "filled_at", "") or "")
                    or datetime.now(timezone.utc).isoformat()
                )
                status_str = str(getattr(filled_order, "status", "pending")).lower()
                order.status = _ALPACA_STATUS_MAP.get(status_str, OrderStatus.PENDING)
            else:
                # Market orders: assume filled at current price if poll timed out
                if order.order_type == OrderType.MARKET:
                    order.fill_price = price
                    order.fill_timestamp = datetime.now(timezone.utc).isoformat()
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PENDING

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = f"Alpaca API error: {e}"
            logger.error("Alpaca place_order failed for %s: status=%s", ticker, getattr(e, "status_code", "unknown"))
            self._orders.append(order)
            return order

        # If filled, sync state
        if order.status == OrderStatus.FILLED:
            self._sync_after_fill(order)

        self._orders.append(order)
        self._log_trade(order)
        return order

    def _poll_order_fill(self, alpaca_order_id: str,
                         timeout: float = 15.0) -> Optional[object]:
        """Poll Alpaca for order fill status. Returns order object or None."""
        start = time.time()
        interval = 0.5
        while time.time() - start < timeout:
            try:
                order_data = self._retry_request(
                    self.trading_client.get_order_by_id, alpaca_order_id
                )
                status = str(getattr(order_data, "status", "")).lower()
                if status in ("filled", "rejected", "canceled", "expired", "stopped"):
                    return order_data
            except Exception as e:
                logger.debug("Poll error for order %s: %s", alpaca_order_id, e)
            time.sleep(interval)
            interval = min(interval * 1.5, 2.0)
        return None

    def _sync_after_fill(self, order: Order):
        """Update local state after a confirmed fill."""
        try:
            self._sync_account()
            self._sync_positions()
        except Exception as e:
            logger.warning("Post-fill sync failed: %s", e)
            # Fall back to local accounting
            self._local_update_position(order)

        self.state.total_trades += 1

    def _local_update_position(self, order: Order):
        """Local position update as fallback when Alpaca sync fails."""
        ticker = order.ticker
        qty = order.quantity
        price = order.fill_price
        pos = self.state.positions.get(ticker, Position(ticker=ticker))

        if order.side == OrderSide.BUY:
            cost = qty * price
            total_qty = pos.quantity + qty
            if total_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            self.state.cash -= cost

        elif order.side == OrderSide.SELL:
            sell_qty = min(qty, pos.quantity)
            proceeds = sell_qty * price
            pnl = (price - pos.avg_cost) * sell_qty
            pos.realized_pnl += pnl
            pos.quantity -= sell_qty
            self.state.cash += proceeds
            self.state.total_pnl += pnl
            self._daily_pnl_today += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        elif order.side == OrderSide.SHORT:
            pos.quantity -= qty
            pos.avg_cost = price
            self.state.cash += qty * price

        elif order.side == OrderSide.COVER:
            cover_qty = min(qty, abs(pos.quantity))
            cost = cover_qty * price
            pnl = (pos.avg_cost - price) * cover_qty
            pos.quantity += cover_qty
            pos.realized_pnl += pnl
            self.state.cash -= cost
            self.state.total_pnl += pnl
            self._daily_pnl_today += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        pos.current_price = price
        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        if pos.quantity != 0:
            self.state.positions[ticker] = pos
        elif ticker in self.state.positions:
            del self.state.positions[ticker]

    # --- Portfolio state (PaperBroker-compatible interface) -------------------

    def refresh_prices(self):
        """Update all position prices from Alpaca data."""
        tickers = list(self.state.positions.keys())
        if not tickers:
            return
        try:
            for ticker in tickers:
                if ticker in self.state.positions:
                    price = self._get_current_price(ticker)
                    if price > 0:
                        pos = self.state.positions[ticker]
                        pos.current_price = price
                        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
        except Exception as e:
            logger.warning("Alpaca price refresh failed: %s", e)

    def compute_nav(self) -> float:
        """Compute current NAV (syncs from Alpaca when possible)."""
        try:
            self._sync_account()
            return self.state.nav
        except Exception:
            # Fallback: compute locally
            self.refresh_prices()
            positions_value = sum(
                p.quantity * p.current_price
                for p in self.state.positions.values()
            )
            self.state.nav = self.state.cash + positions_value
            return self.state.nav

    def compute_exposures(self) -> dict:
        nav = self.compute_nav()
        if nav <= 0:
            return {"gross": 0, "net": 0, "long": 0, "short": 0}
        long_val = sum(
            p.market_value for p in self.state.positions.values()
            if p.quantity > 0
        )
        short_val = sum(
            abs(p.market_value) for p in self.state.positions.values()
            if p.quantity < 0
        )
        self.state.gross_exposure = (long_val + short_val) / nav
        self.state.net_exposure = (long_val - short_val) / nav
        return {
            "gross": self.state.gross_exposure,
            "net": self.state.net_exposure,
            "long": long_val / nav,
            "short": short_val / nav,
        }

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self.state.positions)

    def get_portfolio_summary(self) -> dict:
        nav = self.compute_nav()
        exp = self.compute_exposures()
        win_rate = (
            self.state.win_count / (self.state.win_count + self.state.loss_count)
            if (self.state.win_count + self.state.loss_count) > 0 else 0
        )
        return {
            "nav": nav,
            "cash": self.state.cash,
            "total_pnl": self.state.total_pnl,
            "total_trades": self.state.total_trades,
            "win_rate": win_rate,
            "positions": len(self.state.positions),
            "gross_exposure": exp["gross"],
            "net_exposure": exp["net"],
            "broker": "alpaca",
            "environment": "paper" if self.paper else "live",
        }

    # --- Price fetching (Alpaca real-time data) ------------------------------

    def _get_current_price(self, ticker: str) -> float:
        """Get latest price via Alpaca data API with short cache."""
        now = time.time()
        if ticker in self._quote_cache:
            cached_time, cached_price = self._quote_cache[ticker]
            if now - cached_time < self._quote_cache_ttl:
                return cached_price

        try:
            request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trades = self.data_client.get_stock_latest_trade(request)
            trade = trades.get(ticker) if hasattr(trades, "get") else None
            if trade:
                price = float(getattr(trade, "price", 0) or 0)
                if price > 0:
                    self._quote_cache[ticker] = (now, price)
                    return price
        except Exception as e:
            logger.debug("Alpaca trade for %s failed: %s", ticker, e)

        # Fallback to OpenBB if Alpaca quote unavailable
        try:
            from ..data.openbb_data import get_adj_close
            prices = get_adj_close(ticker, start=(
                pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            )
            if not prices.empty:
                price = float(
                    prices.iloc[-1].iloc[0]
                    if isinstance(prices, pd.DataFrame)
                    else prices.iloc[-1]
                )
                if price > 0:
                    self._quote_cache[ticker] = (now, price)
                    return price
        except Exception:
            pass

        return 0.0

    # --- Daily P&L -----------------------------------------------------------

    def get_daily_pnl(self) -> pd.DataFrame:
        return self._perf_tracker.get_daily_pnl_series()

    def get_drawdown(self) -> dict:
        return self._perf_tracker.get_drawdown()

    def get_performance_metrics(self) -> dict:
        metrics = self._perf_tracker.get_all_metrics()
        total = self.state.win_count + self.state.loss_count
        metrics["overall_win_rate"] = (
            self.state.win_count / total if total > 0 else 0.0
        )
        metrics["total_pnl"] = self.state.total_pnl
        metrics["nav"] = self.state.nav
        metrics["cash"] = self.state.cash
        metrics["broker"] = "alpaca"
        metrics["environment"] = "paper" if self.paper else "live"
        return metrics

    # --- EOD reconciliation --------------------------------------------------

    def reconcile(self) -> dict:
        """End-of-day reconciliation via Alpaca API."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Full sync from Alpaca
        try:
            self._sync_account()
            self._sync_positions()
        except Exception as e:
            logger.warning("Alpaca EOD sync failed: %s", e)

        nav = self.state.nav
        exposures = self.compute_exposures()
        self._perf_tracker.record_nav(nav, date_str=today)

        summary = {
            "date": today,
            "nav": nav,
            "cash": self.state.cash,
            "positions_count": len(self.state.positions),
            "total_pnl": self.state.total_pnl,
            "daily_pnl": self._daily_pnl_today,
            "gross_exposure": exposures["gross"],
            "net_exposure": exposures["net"],
            "broker": "alpaca",
            "environment": "paper" if self.paper else "live",
            "total_trades_today": sum(
                1 for o in self._orders
                if o.status == OrderStatus.FILLED
                and o.fill_timestamp.startswith(today)
            ),
            "commission": 0.0,  # Alpaca is commission-free
        }

        self._eod_nav_history.append((today, nav))

        recon_file = self.log_dir / f"reconcile_{today}.json"
        try:
            with open(recon_file, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        self._daily_pnl_today = 0.0
        self._last_eod_date = today
        self._target_manager.reset_day(nav)
        self.emit_dashboard_state(pipeline_state={"event": "EOD_RECONCILE"})

        return summary

    # --- Position export -----------------------------------------------------

    def export_positions_csv(self, filepath: Optional[str] = None) -> str:
        """Export current positions to CSV (same as PaperBroker)."""
        import csv
        import io
        self.refresh_prices()
        headers = [
            "ticker", "quantity", "avg_cost", "current_price",
            "market_value", "unrealized_pnl", "realized_pnl", "sector",
        ]
        rows = []
        for ticker, pos in sorted(self.state.positions.items()):
            rows.append([
                ticker, pos.quantity,
                round(pos.avg_cost, 4), round(pos.current_price, 4),
                round(pos.market_value, 2), round(pos.unrealized_pnl, 2),
                round(pos.realized_pnl, 2), pos.sector,
            ])
        if filepath:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            return str(path)
        else:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(headers)
            writer.writerows(rows)
            return buf.getvalue()

    # --- Logging -------------------------------------------------------------

    def _log_trade(self, order: Order):
        """Log a trade with extended metadata."""
        entry = order.to_dict()
        entry["nav_at_fill"] = self.state.nav
        entry["cash_after"] = self.state.cash
        entry["positions_count"] = len(self.state.positions)
        entry["daily_pnl"] = self._daily_pnl_today
        entry["total_pnl"] = self.state.total_pnl
        entry["total_trades"] = self.state.total_trades
        entry["broker"] = "alpaca"
        entry["environment"] = "paper" if self.paper else "live"
        entry["commission"] = 0.0  # Alpaca commission-free

        if order.ticker in self.state.positions:
            pos = self.state.positions[order.ticker]
            entry["pos_quantity"] = pos.quantity
            entry["pos_avg_cost"] = pos.avg_cost
            entry["pos_unrealized_pnl"] = pos.unrealized_pnl
            entry["pos_realized_pnl"] = pos.realized_pnl
            entry["pos_sector"] = pos.sector
        else:
            entry["pos_quantity"] = 0
            entry["pos_avg_cost"] = 0.0
            entry["pos_unrealized_pnl"] = 0.0
            entry["pos_realized_pnl"] = 0.0
            entry["pos_sector"] = ""

        # Alpaca order ID
        alpaca_id = self._order_id_map.get(order.id, "")
        entry["alpaca_order_id"] = alpaca_id

        realized = 0.0
        if order.side in (OrderSide.SELL, OrderSide.COVER):
            realized = entry.get("pos_realized_pnl", 0.0)
        perf_entry = dict(entry)
        perf_entry["realized_pnl"] = realized
        self._perf_tracker.record_trade(perf_entry)

        self._trade_log.append(entry)
        log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_log)

    # --- Daily target & risk profile (same as PaperBroker) -------------------

    def get_risk_profile(self) -> str:
        return self._target_manager.profile.value

    def get_daily_target_state(self) -> dict:
        nav = self.compute_nav()
        self._target_manager.update(nav)
        state = self._target_manager.get_state()
        state["current_nav"] = nav
        if self._target_manager._initial_nav_today > 0:
            state["daily_return_pct"] = (
                (nav - self._target_manager._initial_nav_today)
                / self._target_manager._initial_nav_today
            )
        else:
            state["daily_return_pct"] = 0.0
        return state

    def reset_daily_target(self):
        nav = self.compute_nav()
        self._target_manager.reset_day(nav)

    def get_leverage_multiplier(self) -> float:
        return self._target_manager.get_leverage_multiplier()

    # --- Live dashboard (same as PaperBroker) --------------------------------

    def emit_dashboard_state(self, pipeline_state: Optional[dict] = None):
        self._dashboard.emit(
            broker_state=self.get_portfolio_summary(),
            target_state=self.get_daily_target_state(),
            pipeline_state=pipeline_state,
        )

    def get_dashboard_snapshot(self) -> dict:
        return self._dashboard.get_latest()

    def get_dashboard_history(self, n: int = 100) -> list[dict]:
        return self._dashboard.get_history(n)

    def register_dashboard_callback(self, callback):
        self._dashboard.register_callback(callback)

    # --- Alpaca-specific methods ---------------------------------------------

    def get_orders(self) -> list[dict]:
        """Fetch all orders from Alpaca API."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)
            alpaca_orders = self._retry_request(self.trading_client.get_orders, filter=req)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side),
                    "qty": str(o.qty),
                    "type": str(o.type),
                    "status": str(o.status),
                    "filled_qty": str(getattr(o, "filled_qty", "0")),
                    "filled_avg_price": str(getattr(o, "filled_avg_price", "0")),
                    "submitted_at": str(getattr(o, "submitted_at", "")),
                    "filled_at": str(getattr(o, "filled_at", "")),
                }
                for o in alpaca_orders
            ]
        except Exception as e:
            logger.error("Failed to fetch Alpaca orders: status=%s", getattr(e, "status_code", "unknown"))
            return []


    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order on Alpaca by its order ID."""
        try:
            self._retry_request(self.trading_client.cancel_order_by_id, order_id)
            return {"id": order_id, "status": "canceled"}
        except Exception as e:
            logger.error("Failed to cancel Alpaca order %s: status=%s", order_id, getattr(e, "status_code", "unknown"))
            return {"id": order_id, "status": "error", "error": str(e)}

    def cancel_order_by_id(self, order_id: str) -> dict:
        """Alpaca method."""
        return self.cancel_order(order_id)

    def get_gainloss(self) -> list[dict]:
        """Fetch realized gain/loss from Alpaca (via activities API).

        Note: Alpaca gainloss.
        We use portfolio history for realized P&L tracking.
        """
        try:
            req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            history = self._retry_request(
                self.trading_client.get_portfolio_history, filter=req
            )
            # Convert to gainloss-compatible format
            entries = []
            if history and hasattr(history, "timestamp") and history.timestamp:
                for i, ts in enumerate(history.timestamp):
                    entry = {
                        "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "profit_loss": str(history.profit_loss[i]) if history.profit_loss and i < len(history.profit_loss) else "0",
                        "equity": str(history.equity[i]) if history.equity and i < len(history.equity) else "0",
                    }
                    entries.append(entry)
            return entries
        except Exception as e:
            logger.error("Failed to fetch Alpaca gain/loss: status=%s", getattr(e, "status_code", "unknown"))
            return []

    def get_gainloss_report(self) -> list[dict]:
        """Alpaca method."""
        return self.get_gainloss()

    # --- Standard interface aliases --------------------------------------------

    def get_nav(self) -> float:
        """Get current NAV (alias for compute_nav)."""
        return self.compute_nav()

    def get_account_summary(self) -> dict:
        """Get account summary (alias for get_portfolio_summary)."""
        return self.get_portfolio_summary()

    def get_positions(self) -> dict:
        """Get all positions as dict keyed by ticker, including GICS sector."""
        try:
            from ..data.cross_asset_universe import SECTOR_MAP
        except ImportError:
            SECTOR_MAP = {}

        try:
            alpaca_positions = self.trading_client.get_all_positions()
            return {
                p.symbol: {
                    "ticker": p.symbol,
                    "quantity": int(p.qty),
                    "side": p.side,
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price else 0,
                    "market_value": float(p.market_value) if p.market_value else 0,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0,
                    "sector": SECTOR_MAP.get(p.symbol, "Unknown"),
                }
                for p in alpaca_positions
            }
        except Exception as e:
            logger.warning("Failed to get positions: %s", e)
            return {}

    def update_prices(self) -> None:
        """Refresh position prices (alias for refresh_prices)."""
        self.refresh_prices()

    def get_order_history(self) -> list:
        """Get order history (alias for get_trade_history)."""
        return self.get_trade_history()

    def preview_order(
        self,
        ticker: str,
        side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> dict:
        """Preview an order without executing.

        Alpaca preview.
        We simulate by computing estimated costs.
        """
        # Accept both OrderSide enum and string
        _SIDE_STR_MAP = {"buy": "BUY", "sell": "SELL", "short": "SHORT", "cover": "COVER"}
        if isinstance(side, str):
            side_str = side.upper() if side.upper() in ("BUY", "SELL", "SHORT", "COVER") else _SIDE_STR_MAP.get(side.lower(), "BUY")
            side = OrderSide(side_str)
        else:
            side_str = side.value

        if isinstance(order_type, str):
            ot_upper = order_type.upper()
            order_type = OrderType(ot_upper) if ot_upper in ("MARKET", "LIMIT") else OrderType.MARKET
            order_type_str = order_type.value
        else:
            order_type_str = order_type.value

        price = self._get_current_price(ticker) or limit_price or 0
        estimated_cost = quantity * price
        commission = 0.0  # Alpaca is commission-free

        return {
            "ticker": ticker,
            "side": side_str,
            "quantity": quantity,
            "order_type": order_type_str,
            "limit_price": limit_price,
            "estimated_price": price,
            "estimated_cost": estimated_cost,
            "estimated_commission": commission,
            "estimated_total": estimated_cost + commission,
            "broker": "alpaca",
            "environment": "paper" if self.paper else "live",
        }

    # --- Alpaca-specific features --------------------------------------------

    def get_asset(self, ticker: str) -> dict:
        """Get asset information from Alpaca."""
        try:
            asset = self._retry_request(self.trading_client.get_asset, ticker)
            return {
                "id": str(asset.id),
                "symbol": asset.symbol,
                "name": asset.name,
                "exchange": str(asset.exchange),
                "status": str(asset.status),
                "tradable": asset.tradable,
                "marginable": getattr(asset, "marginable", False),
                "shortable": getattr(asset, "shortable", False),
                "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
                "fractionable": getattr(asset, "fractionable", False),
                "class": str(getattr(asset, "asset_class", "")),
            }
        except Exception as e:
            logger.error("Failed to fetch asset %s: %s", ticker, e)
            return {"symbol": ticker, "error": str(e)}

    def get_clock(self) -> dict:
        """Get market clock status from Alpaca."""
        try:
            clock = self._retry_request(self.trading_client.get_clock)
            return {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open) if clock.next_open else "",
                "next_close": str(clock.next_close) if clock.next_close else "",
                "timestamp": str(clock.timestamp) if clock.timestamp else "",
            }
        except Exception as e:
            logger.error("Failed to fetch market clock: %s", e)
            return {"error": str(e)}

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict:
        """Get portfolio history from Alpaca."""
        try:
            req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
            history = self._retry_request(
                self.trading_client.get_portfolio_history, filter=req
            )
            return {
                "timestamp": [
                    datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    for ts in (history.timestamp or [])
                ],
                "equity": list(history.equity or []),
                "profit_loss": list(history.profit_loss or []),
                "profit_loss_pct": list(history.profit_loss_pct or []),
                "base_value": getattr(history, "base_value", 0),
            }
        except Exception as e:
            logger.error("Failed to fetch portfolio history: %s", e)
            return {"error": str(e)}

    # --- Close positions (Alpaca native) -------------------------------------

    def close_position(self, ticker: str, quantity: Optional[int] = None) -> Order:
        """Close a position via Alpaca API."""
        price = self._get_current_price(ticker)
        local_id = str(uuid.uuid4())[:8]
        order = Order(
            id=local_id,
            ticker=ticker,
            side=OrderSide.SELL if (ticker in self.state.positions and self.state.positions[ticker].quantity > 0) else OrderSide.COVER,
            order_type=OrderType.MARKET,
            quantity=quantity or 0,
            fill_price=price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            if quantity:
                req = ClosePositionRequest(qty=str(quantity))
                self._retry_request(
                    self.trading_client.close_position, ticker, close_options=req
                )
            else:
                self._retry_request(self.trading_client.close_position, ticker)

            order.status = OrderStatus.FILLED
            order.fill_price = price
            order.fill_timestamp = datetime.now(timezone.utc).isoformat()

            # Sync after close
            try:
                self._sync_account()
                self._sync_positions()
            except Exception:
                pass
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = f"Alpaca close_position error: {e}"
            logger.error("Failed to close position %s: %s", ticker, e)

        self._orders.append(order)
        self._log_trade(order)
        return order

    def close_all_positions(self) -> list[Order]:
        """Close all positions via Alpaca API."""
        closed = []
        tickers = list(self.state.positions.keys())
        for ticker in tickers:
            order = self.close_position(ticker)
            closed.append(order)
        return closed
