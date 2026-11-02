"""IBKRBroker — Live execution via Interactive Brokers TWS/Gateway (ib_insync SDK).

Sole execution broker for Metadron Capital. All orders route through
L7UnifiedExecutionSurface → IBKRBroker. Native TWAP/VWAP algo orders
executed server-side by IBKR's algo engine.

Environment variables:
    IBKR_HOST           — TWS/Gateway host (default: 127.0.0.1)
    IBKR_PORT           — TWS/Gateway port (default: 7497 paper, 7496 live)
    IBKR_CLIENT_ID      — TWS client ID (default: 1)
    IBKR_PAPER_TRADE    — "True" (default) for paper, "False" for live

ib_insync docs: https://ib-insync.readthedocs.io/
IBKR API: https://interactivebrokers.github.io/tws-api/

Commission: ~$0.005/share equity (min $1), ~$0.65/contract options.
"""

import os
import json
import time
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

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

logger = logging.getLogger("metadron.execution.ibkr")

# ---------------------------------------------------------------------------
# ib_insync imports (lazy-safe)
# ---------------------------------------------------------------------------
try:
    from ib_insync import (
        IB, Stock, Option, Future, Contract,
        MarketOrder, LimitOrder, StopOrder, Order as IBOrder,
        Trade as IBTrade, Fill as IBFill,
        util,
    )
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False
    logger.warning("ib_insync not installed. pip install ib_insync")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PAPER_PORT = 7497
_DEFAULT_LIVE_PORT = 7496
_DEFAULT_CLIENT_ID = 1

_SIDE_MAP = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
    OrderSide.SHORT: "SELL",
    OrderSide.COVER: "BUY",
}

_STATUS_MAP = {
    "Submitted": OrderStatus.PENDING,
    "PreSubmitted": OrderStatus.PENDING,
    "PendingSubmit": OrderStatus.PENDING,
    "PendingCancel": OrderStatus.PENDING,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Inactive": OrderStatus.CANCELLED,
    "ApiCancelled": OrderStatus.CANCELLED,
}


class IBKRAlgoType(str, Enum):
    """IBKR native algo order types."""
    TWAP = "Twap"
    VWAP = "Vwap"
    ADAPTIVE = "Adaptive"
    MARKET = "Market"
    LIMIT = "Limit"


# ---------------------------------------------------------------------------
# IBKR Broker
# ---------------------------------------------------------------------------

class IBKRBroker:
    """Interactive Brokers execution broker with native TWAP/VWAP algo support.

    Connects to TWS or IB Gateway via ib_insync. Implements the standard
    broker interface (place_order, get_positions, sync_account).

    Key features:
        - Native TWAP/VWAP algo orders (server-side splitting by IBKR)
        - Adaptive algo for optimal fill quality
        - Real-time position sync
        - Quote caching with 5s TTL
        - Full audit trail to logs/ibkr_broker/
        - 4-retry exponential backoff on connection failures
    """

    RETRY_COUNT = 4
    RETRY_DELAYS = [2, 4, 8, 16]
    QUOTE_CACHE_TTL = 5.0
    COMMISSION_PER_SHARE = 0.005
    COMMISSION_MIN = 1.00
    COMMISSION_PER_OPTION = 0.65
    COMMISSION_PER_FUTURE = 1.25

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        log_dir=None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        paper: bool = True,
        daily_target_pct: float = 0.05,
    ):
        self._paper = paper
        self._host = host or os.environ.get("IBKR_HOST", _DEFAULT_HOST)
        self._port = port or int(os.environ.get(
            "IBKR_PORT",
            str(_DEFAULT_PAPER_PORT if paper else _DEFAULT_LIVE_PORT)
        ))
        self._client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", str(_DEFAULT_CLIENT_ID)))

        self._log_dir = Path(log_dir or "logs/ibkr_broker")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._ib: Optional[object] = None
        self._connected = False

        # Internal tracking (mirrors PaperBroker interface)
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: List[Order] = []
        self._trade_count = 0

        # Quote cache
        self._quote_cache: Dict[str, tuple] = {}

        # Risk + performance (reuse PaperBroker components)
        self._risk_limiter = RiskLimiter(
            max_position_pct=0.10,
            max_sector_pct=0.30,
            max_daily_loss_pct=0.03,
            max_gross_exposure=2.5,
            max_net_exposure=1.5,
        )
        self._performance = PerformanceTracker()
        self._daily_target = DailyTargetManager(daily_target_pct)

        # IBKR-specific state
        self._ibkr_orders: Dict[str, object] = {}
        self._algo_executions: List[dict] = []

        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        """Connect to TWS/Gateway with retry logic."""
        if not _IB_AVAILABLE:
            logger.warning("IBKRBroker: ib_insync not available — operating in offline mode")
            return

        self._ib = IB()
        for attempt in range(self.RETRY_COUNT):
            try:
                self._ib.connect(
                    self._host, self._port, clientId=self._client_id,
                    timeout=10, readonly=False,
                )
                self._connected = True
                logger.info(
                    "IBKRBroker connected: %s:%d (paper=%s, clientId=%d)",
                    self._host, self._port, self._paper, self._client_id,
                )
                return
            except Exception as e:
                delay = self.RETRY_DELAYS[attempt] if attempt < len(self.RETRY_DELAYS) else 16
                logger.warning(
                    "IBKRBroker connection attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, self.RETRY_COUNT, e, delay,
                )
                time.sleep(delay)

        logger.error("IBKRBroker: all connection attempts failed — offline mode")

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("IBKRBroker disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _make_stock_contract(self, ticker: str) -> object:
        """Create an IBKR Stock contract."""
        if not _IB_AVAILABLE:
            return None
        return Stock(ticker, "SMART", "USD")

    def _make_option_contract(
        self, ticker: str, expiry: str, strike: float, right: str
    ) -> object:
        """Create an IBKR Option contract."""
        if not _IB_AVAILABLE:
            return None
        return Option(ticker, expiry.replace("-", ""), strike, right, "SMART")

    def _make_future_contract(self, symbol: str, expiry: str = "") -> object:
        """Create an IBKR Future contract."""
        if not _IB_AVAILABLE:
            return None
        return Future(symbol, expiry, "CME")

    # ------------------------------------------------------------------
    # TWAP / VWAP algo orders (IBKR native)
    # ------------------------------------------------------------------

    def place_twap_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        duration_minutes: int = 30,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        contract: Optional[object] = None,
        reason: str = "",
    ) -> Order:
        """Place a TWAP algo order via IBKR's native algo engine.

        IBKR server-side TWAP splits the order into equal slices over the
        specified duration, minimizing market impact by distributing
        execution evenly across time.

        Parameters
        ----------
        duration_minutes : int
            Time horizon for TWAP execution (default 30 min).
        """
        if not self.is_connected:
            return self._offline_order(ticker, side, quantity, signal_type, "TWAP", reason)

        if contract is None:
            contract = self._make_stock_contract(ticker)
        self._ib.qualifyContracts(contract)

        action = _SIDE_MAP.get(side, "BUY")
        end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        end_str = end_time.strftime("%Y%m%d %H:%M:%S") + " UTC"

        algo_params = [
            ("strategyType", "Twap"),
            ("startTime", datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S") + " UTC"),
            ("endTime", end_str),
            ("allowPastEndTime", "1"),
            ("monetaryValue", ""),
        ]

        order = IBOrder(
            action=action,
            totalQuantity=quantity,
            orderType="LMT" if limit_price else "MKT",
            lmtPrice=limit_price or 0,
            algoStrategy="Twap",
            algoParams=algo_params,
            tif="DAY",
        )

        return self._submit_ibkr_order(contract, order, ticker, side, quantity, signal_type, "TWAP", reason)

    def place_vwap_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        duration_minutes: int = 60,
        max_pct_volume: float = 0.25,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        contract: Optional[object] = None,
        reason: str = "",
    ) -> Order:
        """Place a VWAP algo order via IBKR's native algo engine.

        IBKR server-side VWAP targets the volume-weighted average price
        by participating proportionally in market volume, tracking the
        intraday volume curve.

        Parameters
        ----------
        duration_minutes : int
            Time horizon for VWAP execution (default 60 min).
        max_pct_volume : float
            Maximum participation rate as fraction of volume (default 25%).
        """
        if not self.is_connected:
            return self._offline_order(ticker, side, quantity, signal_type, "VWAP", reason)

        if contract is None:
            contract = self._make_stock_contract(ticker)
        self._ib.qualifyContracts(contract)

        action = _SIDE_MAP.get(side, "BUY")
        end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        end_str = end_time.strftime("%Y%m%d %H:%M:%S") + " UTC"

        algo_params = [
            ("strategyType", "Vwap"),
            ("startTime", datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S") + " UTC"),
            ("endTime", end_str),
            ("maxPctVol", str(max_pct_volume)),
            ("noTakeLiq", "0"),
            ("monetaryValue", ""),
            ("allowPastEndTime", "1"),
        ]

        order = IBOrder(
            action=action,
            totalQuantity=quantity,
            orderType="LMT" if limit_price else "MKT",
            lmtPrice=limit_price or 0,
            algoStrategy="Vwap",
            algoParams=algo_params,
            tif="DAY",
        )

        return self._submit_ibkr_order(contract, order, ticker, side, quantity, signal_type, "VWAP", reason)

    def place_adaptive_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        priority: str = "Normal",
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
    ) -> Order:
        """Place an Adaptive algo order — IBKR chooses optimal execution strategy.

        Priority levels: Patient, Normal, Urgent.
        """
        if not self.is_connected:
            return self._offline_order(ticker, side, quantity, signal_type, "ADAPTIVE", reason)

        contract = self._make_stock_contract(ticker)
        self._ib.qualifyContracts(contract)

        action = _SIDE_MAP.get(side, "BUY")
        algo_params = [
            ("adaptivePriority", priority),
        ]

        order = IBOrder(
            action=action,
            totalQuantity=quantity,
            orderType="LMT" if limit_price else "MKT",
            lmtPrice=limit_price or 0,
            algoStrategy="Adaptive",
            algoParams=algo_params,
            tif="DAY",
        )

        return self._submit_ibkr_order(contract, order, ticker, side, quantity, signal_type, "ADAPTIVE", reason)

    # ------------------------------------------------------------------
    # Standard orders (PaperBroker-compatible interface)
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
    ) -> Order:
        """Place a standard market/limit order (PaperBroker-compatible interface)."""
        if not self.is_connected:
            return self._offline_order(ticker, side, quantity, signal_type, "MARKET", reason)

        contract = self._make_stock_contract(ticker)
        self._ib.qualifyContracts(contract)

        action = _SIDE_MAP.get(side, "BUY")
        if limit_price:
            ib_order = LimitOrder(action, quantity, limit_price)
        else:
            ib_order = MarketOrder(action, quantity)

        return self._submit_ibkr_order(contract, ib_order, ticker, side, quantity, signal_type, "MARKET", reason)

    # ------------------------------------------------------------------
    # Core IBKR order submission
    # ------------------------------------------------------------------

    def _submit_ibkr_order(
        self,
        contract: object,
        ib_order: object,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType,
        algo_type: str,
        reason: str,
    ) -> Order:
        """Submit order to IBKR and return Metadron Order object."""
        order_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc)

        metadron_order = Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT if hasattr(ib_order, 'lmtPrice') and ib_order.lmtPrice > 0 else OrderType.MARKET,
            signal_type=signal_type,
            status=OrderStatus.PENDING,
            reason=f"IBKR:{algo_type} {reason}".strip(),
            timestamp=now,
        )

        try:
            trade = self._ib.placeOrder(contract, ib_order)
            self._ibkr_orders[order_id] = trade

            self._ib.sleep(0.5)

            if trade.orderStatus.status == "Filled":
                avg_price = trade.orderStatus.avgFillPrice
                metadron_order.fill_price = avg_price
                metadron_order.status = OrderStatus.FILLED
                metadron_order.filled_at = datetime.now(timezone.utc)
                self._update_position(ticker, side, quantity, avg_price, signal_type)
            elif trade.orderStatus.status in ("Submitted", "PreSubmitted"):
                metadron_order.status = OrderStatus.PENDING
            else:
                metadron_order.status = _STATUS_MAP.get(
                    trade.orderStatus.status, OrderStatus.PENDING
                )

            self._algo_executions.append({
                "order_id": order_id,
                "ticker": ticker,
                "algo": algo_type,
                "side": side.value,
                "quantity": quantity,
                "status": metadron_order.status.value,
                "fill_price": metadron_order.fill_price,
                "ibkr_order_id": trade.order.orderId,
                "timestamp": now.isoformat(),
            })

            logger.info(
                "IBKR %s: %s %s %d — status=%s, fill=$%.2f",
                algo_type, side.value, ticker, quantity,
                metadron_order.status.value, metadron_order.fill_price or 0,
            )

        except Exception as e:
            metadron_order.status = OrderStatus.CANCELLED
            metadron_order.reason = f"IBKR error: {e}"
            logger.error("IBKR order failed: %s %s — %s", ticker, algo_type, e)

        self._orders.append(metadron_order)
        self._trade_count += 1
        self._log_order(metadron_order, algo_type)

        return metadron_order

    def _offline_order(
        self, ticker: str, side: OrderSide, quantity: int,
        signal_type: SignalType, algo_type: str, reason: str,
    ) -> Order:
        """Create order record when IBKR is offline."""
        order = Order(
            order_id=str(uuid.uuid4())[:12],
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            signal_type=signal_type,
            status=OrderStatus.CANCELLED,
            reason=f"IBKR offline — {algo_type} order not submitted. {reason}",
            timestamp=datetime.now(timezone.utc),
        )
        self._orders.append(order)
        logger.warning("IBKR offline: %s %s %d not submitted", algo_type, ticker, quantity)
        return order

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(
        self, ticker: str, side: OrderSide, quantity: int,
        fill_price: float, signal_type: SignalType,
    ):
        """Update internal position tracking after a fill."""
        if ticker in self._positions:
            pos = self._positions[ticker]
            if side in (OrderSide.BUY, OrderSide.COVER):
                new_qty = pos.quantity + quantity
                if new_qty != 0:
                    pos.avg_price = (pos.avg_price * pos.quantity + fill_price * quantity) / new_qty
                pos.quantity = new_qty
            else:
                pos.quantity -= quantity
            pos.current_price = fill_price
            if pos.quantity == 0:
                del self._positions[ticker]
        else:
            if side in (OrderSide.BUY, OrderSide.COVER):
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_price=fill_price,
                    current_price=fill_price,
                    signal_type=signal_type,
                )
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=-quantity,
                    avg_price=fill_price,
                    current_price=fill_price,
                    signal_type=signal_type,
                )

    # ------------------------------------------------------------------
    # Sync from IBKR
    # ------------------------------------------------------------------

    def sync_positions(self):
        """Sync positions from IBKR account."""
        if not self.is_connected:
            return
        try:
            ib_positions = self._ib.positions()
            for pos in ib_positions:
                ticker = pos.contract.symbol
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=int(pos.position),
                    avg_price=pos.avgCost,
                    current_price=pos.avgCost,
                    signal_type=SignalType.HOLD,
                )
            logger.info("IBKR: synced %d positions", len(ib_positions))
        except Exception as e:
            logger.error("IBKR position sync failed: %s", e)

    def sync_account(self) -> dict:
        """Sync account balances from IBKR."""
        if not self.is_connected:
            return {"cash": self._initial_cash, "nav": self._initial_cash}
        try:
            account_values = self._ib.accountSummary()
            result = {}
            for av in account_values:
                if av.tag == "TotalCashValue":
                    result["cash"] = float(av.value)
                elif av.tag == "NetLiquidation":
                    result["nav"] = float(av.value)
                elif av.tag == "GrossPositionValue":
                    result["gross_position"] = float(av.value)
                elif av.tag == "BuyingPower":
                    result["buying_power"] = float(av.value)
            return result
        except Exception as e:
            logger.error("IBKR account sync failed: %s", e)
            return {"cash": self._initial_cash, "nav": self._initial_cash}

    # ------------------------------------------------------------------
    # Quote fetching
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> Optional[float]:
        """Get latest price for ticker (cached 5s)."""
        now = time.time()
        if ticker in self._quote_cache:
            price, ts = self._quote_cache[ticker]
            if now - ts < self.QUOTE_CACHE_TTL:
                return price

        if not self.is_connected:
            return None

        try:
            contract = self._make_stock_contract(ticker)
            self._ib.qualifyContracts(contract)
            ticker_data = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(1)
            price = ticker_data.last or ticker_data.close or ticker_data.bid
            if price and price > 0:
                self._quote_cache[ticker] = (float(price), now)
                self._ib.cancelMktData(contract)
                return float(price)
            self._ib.cancelMktData(contract)
        except Exception as e:
            logger.debug("IBKR quote failed for %s: %s", ticker, e)

        return None

    # ------------------------------------------------------------------
    # Check pending algo orders
    # ------------------------------------------------------------------

    def check_algo_fills(self) -> List[dict]:
        """Check and update pending algo orders (TWAP/VWAP may take minutes to fill)."""
        fills = []
        if not self.is_connected:
            return fills

        for order_id, trade in list(self._ibkr_orders.items()):
            try:
                self._ib.sleep(0)
                status = trade.orderStatus.status
                if status == "Filled" and any(
                    o.order_id == order_id and o.status != OrderStatus.FILLED
                    for o in self._orders
                ):
                    for o in self._orders:
                        if o.order_id == order_id:
                            o.fill_price = trade.orderStatus.avgFillPrice
                            o.status = OrderStatus.FILLED
                            o.filled_at = datetime.now(timezone.utc)
                            self._update_position(
                                o.ticker, o.side, o.quantity,
                                o.fill_price, o.signal_type,
                            )
                            fills.append({
                                "order_id": order_id,
                                "ticker": o.ticker,
                                "fill_price": o.fill_price,
                                "quantity": o.quantity,
                            })
                            break
            except Exception as e:
                logger.debug("Check fill failed for %s: %s", order_id, e)

        return fills

    # ------------------------------------------------------------------
    # Portfolio state (PaperBroker-compatible)
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> dict:
        """Return portfolio summary matching PaperBroker interface."""
        acct = self.sync_account()
        nav = acct.get("nav", self._initial_cash)
        cash = acct.get("cash", self._initial_cash)
        total_pnl = nav - self._initial_cash

        return {
            "nav": nav,
            "cash": cash,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self._initial_cash if self._initial_cash > 0 else 0,
            "position_count": len(self._positions),
            "trade_count": self._trade_count,
            "broker": "IBKR",
            "connected": self.is_connected,
            "paper": self._paper,
        }

    def get_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def get_orders(self) -> List[Order]:
        return list(self._orders)

    def get_algo_executions(self) -> List[dict]:
        return list(self._algo_executions)

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _log_order(self, order: Order, algo_type: str):
        """Append order to audit log."""
        log_file = self._log_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            entry = {
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "signal_type": order.signal_type.value,
                "status": order.status.value,
                "fill_price": order.fill_price,
                "algo_type": algo_type,
                "reason": order.reason,
                "timestamp": order.timestamp.isoformat() if order.timestamp else "",
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("IBKR log write failed: %s", e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "broker": "IBKR",
            "connected": self.is_connected,
            "paper": self._paper,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "position_count": len(self._positions),
            "order_count": len(self._orders),
            "algo_executions": len(self._algo_executions),
            "trade_count": self._trade_count,
        }

    def __repr__(self) -> str:
        return (
            f"IBKRBroker(host={self._host}, port={self._port}, "
            f"paper={self._paper}, connected={self.is_connected})"
        )
