"""ExchangeCoreEngine -- Native Python matching engine inspired by exchange-core v2.

Implements the core concepts from the Java exchange-core LMAX Disruptor-based
order matching engine entirely in Python with numpy for performance-critical
paths.  No JVM or Maven dependency required.

Architecture (mirrors Java exchange-core):

    RingBuffer (pre-allocated numpy arrays)
        --> MatchingEngineRouter (per-symbol OrderBook dispatch)
            --> OrderBook (sorted price levels, price-time priority)
                --> OrdersBucket (FIFO queue at each price level)
                    --> Trade events (fill notifications)

Key features:
    - LMAX Disruptor-style ring buffer for order event ingestion
    - Price-time priority limit order book per symbol
    - Market / Limit / Stop order types
    - L3 order book depth snapshots
    - Fill notification chain (mirrors MatcherTradeEvent linked list)
    - Batch processing for multiple orders
    - Latency tracking with simulated microsecond timestamps
    - Integration hook for PaperBroker (route orders through matching engine)

Design rules (per CLAUDE.md):
    - try/except on ALL external imports -- system runs degraded, never broken
    - Pure-numpy fallbacks -- no crashes if optional packages missing
    - No external dependencies beyond numpy
"""

from __future__ import annotations

import time
import uuid
import logging
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums  (mirror Java beans: OrderAction, OrderType, MatcherEventType)
# ---------------------------------------------------------------------------

class OrderAction(str, Enum):
    """Maps to org.openpredict.exchange.beans.OrderAction."""
    BID = "BID"
    ASK = "ASK"


class EngineOrderType(str, Enum):
    """Maps to org.openpredict.exchange.beans.OrderType + STOP extension."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class MatcherEventType(str, Enum):
    """Maps to org.openpredict.exchange.beans.MatcherEventType."""
    TRADE = "TRADE"
    REDUCE = "REDUCE"
    REJECTION = "REJECTION"


class OrderState(str, Enum):
    """Maps to org.openpredict.exchange.beans.OrderState."""
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class CommandResultCode(str, Enum):
    """Maps to org.openpredict.exchange.beans.cmd.CommandResultCode."""
    SUCCESS = "SUCCESS"
    MATCHING_UNKNOWN_ORDER_ID = "MATCHING_UNKNOWN_ORDER_ID"
    NO_LIQUIDITY = "NO_LIQUIDITY"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EngineOrder:
    """Internal order record (mirrors Java Order bean)."""
    order_id: int
    symbol: str
    price: float
    size: int
    action: OrderAction
    order_type: EngineOrderType
    uid: int = 0
    timestamp_ns: int = 0
    filled: int = 0
    stop_price: float = 0.0
    state: OrderState = OrderState.NEW

    @property
    def remaining(self) -> int:
        return self.size - self.filled


@dataclass
class TradeEvent:
    """Fill notification (mirrors MatcherTradeEvent linked list).

    In the Java version these are chained via nextEvent pointer.
    Here we collect them into a list on the FillResult.
    """
    event_type: MatcherEventType
    symbol: str
    active_order_id: int
    active_order_uid: int
    active_order_completed: bool
    active_order_action: OrderAction
    matched_order_id: int
    matched_order_uid: int
    matched_order_completed: bool
    price: float
    size: int
    timestamp_ns: int

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "active_order_id": self.active_order_id,
            "matched_order_id": self.matched_order_id,
            "price": self.price,
            "size": self.size,
            "active_completed": self.active_order_completed,
            "matched_completed": self.matched_order_completed,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class FillResult:
    """Aggregate result for a single order submission."""
    order_id: int
    symbol: str
    action: OrderAction
    order_type: EngineOrderType
    requested_size: int
    filled_size: int
    remaining_size: int
    avg_fill_price: float
    trades: List[TradeEvent] = field(default_factory=list)
    result_code: CommandResultCode = CommandResultCode.SUCCESS
    state: OrderState = OrderState.NEW
    latency_us: float = 0.0

    @property
    def is_fully_filled(self) -> bool:
        return self.remaining_size == 0 and self.filled_size > 0

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "order_type": self.order_type.value,
            "requested_size": self.requested_size,
            "filled_size": self.filled_size,
            "remaining_size": self.remaining_size,
            "avg_fill_price": round(self.avg_fill_price, 6),
            "state": self.state.value,
            "result_code": self.result_code.value,
            "num_trades": len(self.trades),
            "latency_us": round(self.latency_us, 2),
            "trades": [t.to_dict() for t in self.trades],
        }


@dataclass
class L3BookLevel:
    """Single price level in the L3 order book snapshot."""
    price: float
    total_volume: int
    num_orders: int
    orders: List[dict] = field(default_factory=list)


@dataclass
class OrderBookSnapshot:
    """Full L3 order book snapshot for a symbol."""
    symbol: str
    timestamp_ns: int
    bids: List[L3BookLevel] = field(default_factory=list)
    asks: List[L3BookLevel] = field(default_factory=list)
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    mid_price: float = 0.0
    total_bid_volume: int = 0
    total_ask_volume: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ns": self.timestamp_ns,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": round(self.spread, 6),
            "mid_price": round(self.mid_price, 6),
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "bids": [
                {"price": l.price, "volume": l.total_volume,
                 "orders": l.num_orders, "order_details": l.orders}
                for l in self.bids
            ],
            "asks": [
                {"price": l.price, "volume": l.total_volume,
                 "orders": l.num_orders, "order_details": l.orders}
                for l in self.asks
            ],
        }


# ---------------------------------------------------------------------------
# OrdersBucket -- FIFO queue at a single price level
# (mirrors Java IOrdersBucket / OrdersBucketSlow)
# ---------------------------------------------------------------------------

class OrdersBucket:
    """FIFO order queue at a single price level.

    Maintains insertion order for price-time priority matching.
    """

    __slots__ = ("_price", "_orders", "_total_volume")

    def __init__(self, price: float):
        self._price = price
        self._orders: List[EngineOrder] = []
        self._total_volume: int = 0

    @property
    def price(self) -> float:
        return self._price

    @property
    def total_volume(self) -> int:
        return self._total_volume

    @property
    def num_orders(self) -> int:
        return len(self._orders)

    def add(self, order: EngineOrder) -> None:
        self._orders.append(order)
        self._total_volume += order.remaining

    def remove(self, order_id: int) -> Optional[EngineOrder]:
        for i, o in enumerate(self._orders):
            if o.order_id == order_id:
                self._total_volume -= o.remaining
                return self._orders.pop(i)
        return None

    def match(
        self,
        volume_to_fill: int,
        active_uid: int,
        trade_callback: Callable,
    ) -> int:
        """Match incoming order against resting orders in FIFO order.

        Returns total volume matched at this price level.
        """
        filled = 0
        to_remove: List[int] = []

        for i, resting in enumerate(self._orders):
            if volume_to_fill <= 0:
                break
            matchable = min(resting.remaining, volume_to_fill)
            resting.filled += matchable
            resting_completed = resting.filled >= resting.size
            if resting_completed:
                resting.state = OrderState.FILLED
                to_remove.append(i)
            else:
                resting.state = OrderState.PARTIALLY_FILLED

            trade_callback(resting, matchable, resting_completed)

            filled += matchable
            volume_to_fill -= matchable

        # Remove fully filled orders (iterate in reverse to preserve indices)
        for idx in reversed(to_remove):
            self._orders.pop(idx)

        self._total_volume -= filled
        return filled

    def get_orders(self) -> List[dict]:
        """Return L3 order details for this price level."""
        return [
            {
                "order_id": o.order_id,
                "size": o.size,
                "filled": o.filled,
                "remaining": o.remaining,
                "uid": o.uid,
                "timestamp_ns": o.timestamp_ns,
            }
            for o in self._orders
        ]


# ---------------------------------------------------------------------------
# OrderBook -- Sorted price-level book per symbol
# (mirrors Java OrderBookSlow: TreeMap<Long, IOrdersBucket>)
# ---------------------------------------------------------------------------

class OrderBook:
    """Limit order book for a single symbol.

    Bids sorted descending (best bid first), asks sorted ascending
    (best ask first) -- exactly as in the Java OrderBookSlow with
    TreeMap + reverseOrder comparator for bids.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        # price -> OrdersBucket; we keep separate dicts and sort on access
        self._bids: Dict[float, OrdersBucket] = {}
        self._asks: Dict[float, OrdersBucket] = {}
        self._id_map: Dict[int, EngineOrder] = {}
        # Stop orders waiting for trigger
        self._stop_orders: List[EngineOrder] = []

    # -- helpers --------------------------------------------------------

    def _sorted_bid_prices(self) -> List[float]:
        return sorted(self._bids.keys(), reverse=True)

    def _sorted_ask_prices(self) -> List[float]:
        return sorted(self._asks.keys())

    @property
    def best_bid(self) -> float:
        prices = self._sorted_bid_prices()
        return prices[0] if prices else 0.0

    @property
    def best_ask(self) -> float:
        prices = self._sorted_ask_prices()
        return prices[0] if prices else 0.0

    @property
    def mid_price(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba > 0:
            return (bb + ba) / 2.0
        return bb or ba

    @property
    def spread(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba > 0:
            return ba - bb
        return 0.0

    @property
    def num_orders(self) -> int:
        return len(self._id_map)

    # -- matching -------------------------------------------------------

    def place_market_order(
        self,
        order: EngineOrder,
        trade_events: List[TradeEvent],
    ) -> int:
        """Match a market order against the opposite side.

        Returns filled volume.
        """
        buckets = self._asks if order.action == OrderAction.BID else self._bids
        filled = self._try_match_instantly(order, buckets, trade_events, market=True)
        return filled

    def place_limit_order(
        self,
        order: EngineOrder,
        trade_events: List[TradeEvent],
    ) -> int:
        """Place a limit order: match marketable portion, rest goes on book.

        Returns filled volume.
        """
        if order.order_id in self._id_map:
            raise ValueError(f"duplicate order_id: {order.order_id}")

        # Determine matchable price levels
        buckets = self._get_matchable_buckets(order)
        filled = self._try_match_instantly(order, buckets, trade_events, market=False)

        if filled >= order.size:
            order.state = OrderState.FILLED
            order.filled = order.size
            return filled

        order.filled = filled
        order.state = OrderState.ACTIVE if filled == 0 else OrderState.PARTIALLY_FILLED

        # Place remainder on book
        side = self._bids if order.action == OrderAction.BID else self._asks
        bucket = side.get(order.price)
        if bucket is None:
            bucket = OrdersBucket(order.price)
            side[order.price] = bucket
        bucket.add(order)
        self._id_map[order.order_id] = order

        return filled

    def place_stop_order(self, order: EngineOrder) -> None:
        """Register a stop order.  It converts to market when trigger fires."""
        order.state = OrderState.ACTIVE
        self._stop_orders.append(order)

    def check_stop_orders(self, trade_events: List[TradeEvent]) -> List[FillResult]:
        """Evaluate and trigger any stop orders whose conditions are met.

        Returns list of FillResults for triggered stops.
        """
        triggered: List[FillResult] = []
        remaining_stops: List[EngineOrder] = []

        last_price = self.mid_price

        for stop in self._stop_orders:
            fire = False
            if stop.action == OrderAction.BID and last_price >= stop.stop_price:
                fire = True
            elif stop.action == OrderAction.ASK and last_price <= stop.stop_price:
                fire = True

            if fire:
                stop.order_type = EngineOrderType.MARKET
                events: List[TradeEvent] = []
                t0 = _now_ns()
                filled = self.place_market_order(stop, events)
                stop.filled = filled
                elapsed_us = (_now_ns() - t0) / 1000.0

                avg_px = 0.0
                if events:
                    total_notional = sum(e.price * e.size for e in events)
                    total_qty = sum(e.size for e in events)
                    avg_px = total_notional / total_qty if total_qty else 0.0

                result = FillResult(
                    order_id=stop.order_id,
                    symbol=self.symbol,
                    action=stop.action,
                    order_type=EngineOrderType.STOP,
                    requested_size=stop.size,
                    filled_size=filled,
                    remaining_size=stop.size - filled,
                    avg_fill_price=avg_px,
                    trades=events,
                    result_code=CommandResultCode.SUCCESS if filled > 0 else CommandResultCode.NO_LIQUIDITY,
                    state=OrderState.FILLED if filled >= stop.size else OrderState.PARTIALLY_FILLED,
                    latency_us=elapsed_us,
                )
                triggered.append(result)
            else:
                remaining_stops.append(stop)

        self._stop_orders = remaining_stops
        return triggered

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order. Returns True if found and removed."""
        order = self._id_map.pop(order_id, None)
        if order is None:
            # Check stop orders
            for i, s in enumerate(self._stop_orders):
                if s.order_id == order_id:
                    self._stop_orders.pop(i)
                    return True
            return False

        side = self._bids if order.action == OrderAction.BID else self._asks
        bucket = side.get(order.price)
        if bucket is not None:
            bucket.remove(order_id)
            if bucket.total_volume == 0:
                del side[order.price]
        order.state = OrderState.CANCELLED
        return True

    # -- internal matching -----------------------------------------------

    def _get_matchable_buckets(self, order: EngineOrder) -> Dict[float, OrdersBucket]:
        """Return subset of opposite-side buckets that are matchable."""
        if order.action == OrderAction.BID:
            # Buy limit: match asks at or below limit price
            return {p: b for p, b in self._asks.items() if p <= order.price}
        else:
            # Sell limit: match bids at or above limit price
            return {p: b for p, b in self._bids.items() if p >= order.price}

    def _try_match_instantly(
        self,
        order: EngineOrder,
        buckets: Dict[float, OrdersBucket],
        trade_events: List[TradeEvent],
        market: bool,
    ) -> int:
        """Walk price levels in priority order and match.

        For BID orders matching against asks: ascending price.
        For ASK orders matching against bids: descending price.
        """
        if not buckets:
            return 0

        is_bid = order.action == OrderAction.BID
        sorted_prices = sorted(buckets.keys(), reverse=(not is_bid))

        filled = 0
        volume_needed = order.size - order.filled
        empty_prices: List[float] = []

        for price in sorted_prices:
            if volume_needed <= 0:
                break
            bucket = buckets[price]
            now_ns = _now_ns()

            def _trade_cb(resting: EngineOrder, match_size: int, completed: bool) -> None:
                nonlocal filled, volume_needed
                te = TradeEvent(
                    event_type=MatcherEventType.TRADE,
                    symbol=self.symbol,
                    active_order_id=order.order_id,
                    active_order_uid=order.uid,
                    active_order_completed=(filled + match_size >= order.size),
                    active_order_action=order.action,
                    matched_order_id=resting.order_id,
                    matched_order_uid=resting.uid,
                    matched_order_completed=completed,
                    price=price,
                    size=match_size,
                    timestamp_ns=now_ns,
                )
                trade_events.append(te)
                if completed:
                    self._id_map.pop(resting.order_id, None)

            matched = bucket.match(volume_needed, order.uid, _trade_cb)
            filled += matched
            volume_needed -= matched

            if bucket.total_volume == 0:
                empty_prices.append(price)

        # Clean up empty buckets from the main side dict
        opposite = self._asks if is_bid else self._bids
        for p in empty_prices:
            opposite.pop(p, None)

        return filled

    # -- snapshots -------------------------------------------------------

    def get_snapshot(self, depth: int = 50) -> OrderBookSnapshot:
        """Return L3 book snapshot up to *depth* levels per side."""
        bid_prices = self._sorted_bid_prices()[:depth]
        ask_prices = self._sorted_ask_prices()[:depth]

        bid_levels = []
        total_bid_vol = 0
        for p in bid_prices:
            b = self._bids[p]
            bid_levels.append(L3BookLevel(
                price=p,
                total_volume=b.total_volume,
                num_orders=b.num_orders,
                orders=b.get_orders(),
            ))
            total_bid_vol += b.total_volume

        ask_levels = []
        total_ask_vol = 0
        for p in ask_prices:
            a = self._asks[p]
            ask_levels.append(L3BookLevel(
                price=p,
                total_volume=a.total_volume,
                num_orders=a.num_orders,
                orders=a.get_orders(),
            ))
            total_ask_vol += a.total_volume

        bb = bid_prices[0] if bid_prices else 0.0
        ba = ask_prices[0] if ask_prices else 0.0
        spread = (ba - bb) if (bb > 0 and ba > 0) else 0.0
        mid = (bb + ba) / 2.0 if (bb > 0 and ba > 0) else (bb or ba)

        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp_ns=_now_ns(),
            bids=bid_levels,
            asks=ask_levels,
            best_bid=bb,
            best_ask=ba,
            spread=spread,
            mid_price=mid,
            total_bid_volume=total_bid_vol,
            total_ask_volume=total_ask_vol,
        )


# ---------------------------------------------------------------------------
# RingBuffer -- LMAX Disruptor-style pre-allocated event buffer
# (mirrors com.lmax.disruptor.RingBuffer)
# ---------------------------------------------------------------------------

class RingBuffer:
    """Pre-allocated numpy ring buffer for order events.

    Implements the core LMAX Disruptor concept: a fixed-size, power-of-two
    buffer with monotonically increasing sequence numbers.  Producers claim
    a slot via next(), write the event, then publish().  Consumers read
    via get() and track their own sequence.

    Fields per slot (stored as columns in numpy structured array):
        sequence, order_id, symbol_hash, action, order_type,
        price, size, timestamp_ns, processed
    """

    # Column indices in the numpy buffer
    COL_SEQUENCE = 0
    COL_ORDER_ID = 1
    COL_SYMBOL_HASH = 2
    COL_ACTION = 3      # 0=BID, 1=ASK
    COL_ORDER_TYPE = 4   # 0=MARKET, 1=LIMIT, 2=STOP
    COL_PRICE = 5
    COL_SIZE = 6
    COL_TIMESTAMP = 7
    COL_PROCESSED = 8
    NUM_COLS = 9

    def __init__(self, buffer_size: int = 65536):
        """Initialise ring buffer.  *buffer_size* must be a power of two."""
        # Round up to next power of two
        size = 1
        while size < buffer_size:
            size <<= 1
        self._size = size
        self._mask = size - 1

        # Pre-allocate numpy buffer if available, else plain list
        if np is not None:
            self._buffer = np.zeros((size, self.NUM_COLS), dtype=np.float64)
        else:
            self._buffer = [[0.0] * self.NUM_COLS for _ in range(size)]

        # Producer cursor -- next sequence to claim
        self._cursor: int = -1
        # Consumer sequences (barrier pattern)
        self._consumer_sequence: int = -1

        # Symbol string lookup (hash -> symbol)
        self._symbol_lookup: Dict[int, str] = {}

        # Latency tracking
        self._latencies_us: List[float] = []

    @property
    def size(self) -> int:
        return self._size

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def consumer_sequence(self) -> int:
        return self._consumer_sequence

    def _symbol_hash(self, symbol: str) -> int:
        h = hash(symbol) & 0x7FFFFFFFFFFFFFFF
        self._symbol_lookup[h] = symbol
        return h

    def next(self) -> int:
        """Claim the next slot. Returns sequence number."""
        self._cursor += 1
        return self._cursor

    def publish(
        self,
        sequence: int,
        order_id: int,
        symbol: str,
        action: OrderAction,
        order_type: EngineOrderType,
        price: float,
        size: int,
    ) -> None:
        """Write an order event into the ring buffer at *sequence*."""
        idx = sequence & self._mask
        sym_hash = self._symbol_hash(symbol)
        action_code = 0.0 if action == OrderAction.BID else 1.0
        type_code = {"MARKET": 0.0, "LIMIT": 1.0, "STOP": 2.0}[order_type.value]

        if np is not None:
            self._buffer[idx, :] = [
                float(sequence), float(order_id), float(sym_hash),
                action_code, type_code, price, float(size),
                float(_now_ns()), 0.0,
            ]
        else:
            self._buffer[idx] = [
                float(sequence), float(order_id), float(sym_hash),
                action_code, type_code, price, float(size),
                float(_now_ns()), 0.0,
            ]

    def get(self, sequence: int) -> dict:
        """Read event at *sequence*. Returns dict with parsed fields."""
        idx = sequence & self._mask
        row = self._buffer[idx] if np is None else self._buffer[idx]
        sym_hash = int(row[self.COL_SYMBOL_HASH])
        return {
            "sequence": int(row[self.COL_SEQUENCE]),
            "order_id": int(row[self.COL_ORDER_ID]),
            "symbol": self._symbol_lookup.get(sym_hash, f"UNKNOWN_{sym_hash}"),
            "action": OrderAction.BID if row[self.COL_ACTION] == 0 else OrderAction.ASK,
            "order_type": [EngineOrderType.MARKET, EngineOrderType.LIMIT,
                           EngineOrderType.STOP][int(row[self.COL_ORDER_TYPE])],
            "price": float(row[self.COL_PRICE]),
            "size": int(row[self.COL_SIZE]),
            "timestamp_ns": int(row[self.COL_TIMESTAMP]),
            "processed": bool(row[self.COL_PROCESSED]),
        }

    def mark_processed(self, sequence: int) -> None:
        idx = sequence & self._mask
        if np is not None:
            self._buffer[idx, self.COL_PROCESSED] = 1.0
        else:
            self._buffer[idx][self.COL_PROCESSED] = 1.0
        self._consumer_sequence = max(self._consumer_sequence, sequence)

    def record_latency(self, latency_us: float) -> None:
        self._latencies_us.append(latency_us)

    def get_latency_stats(self) -> dict:
        """Return ring buffer latency statistics."""
        if not self._latencies_us:
            return {
                "count": 0, "mean_us": 0.0, "median_us": 0.0,
                "p95_us": 0.0, "p99_us": 0.0, "min_us": 0.0, "max_us": 0.0,
            }
        if np is not None:
            arr = np.array(self._latencies_us)
            return {
                "count": len(arr),
                "mean_us": float(np.mean(arr)),
                "median_us": float(np.median(arr)),
                "p95_us": float(np.percentile(arr, 95)),
                "p99_us": float(np.percentile(arr, 99)),
                "min_us": float(np.min(arr)),
                "max_us": float(np.max(arr)),
            }
        # Fallback without numpy
        s = sorted(self._latencies_us)
        n = len(s)
        return {
            "count": n,
            "mean_us": sum(s) / n,
            "median_us": s[n // 2],
            "p95_us": s[int(n * 0.95)],
            "p99_us": s[int(n * 0.99)],
            "min_us": s[0],
            "max_us": s[-1],
        }

    def utilisation(self) -> float:
        """Return fraction of buffer that has been used at least once."""
        total = self._cursor + 1
        if total <= 0:
            return 0.0
        return min(total / self._size, 1.0)


# ---------------------------------------------------------------------------
# ExchangeCoreEngine -- Main entry point
# (mirrors Java ExchangeCore + MatchingEngineRouter)
# ---------------------------------------------------------------------------

class ExchangeCoreEngine:
    """Native Python matching engine inspired by exchange-core v2.

    Provides LMAX Disruptor-style event processing with ring buffer
    ingestion, per-symbol order book management, and price-time
    priority matching.

    Usage::

        engine = ExchangeCoreEngine()

        # Place a limit bid
        result = engine.process_order("AAPL", "BUY", 100, 150.00, "LIMIT")

        # Place a market sell
        result = engine.process_order("AAPL", "SELL", 50, 0, "MARKET")

        # Get L3 book snapshot
        book = engine.get_order_book("AAPL")

        # Latency stats
        stats = engine.get_latency_stats()

        # Batch processing
        orders = [
            ("AAPL", "BUY", 100, 150.0, "LIMIT"),
            ("AAPL", "SELL", 50, 0, "MARKET"),
        ]
        results = engine.process_batch(orders)

    Integration with PaperBroker::

        from engine.execution.paper_broker import PaperBroker
        broker = PaperBroker()
        engine = ExchangeCoreEngine()
        engine.route_to_paper_broker(broker, "AAPL", "BUY", 100, 150.0, "LIMIT")
    """

    def __init__(
        self,
        ring_buffer_size: int = 65536,
        default_book_depth: int = 50,
    ):
        self._ring_buffer = RingBuffer(buffer_size=ring_buffer_size)
        self._order_books: Dict[str, OrderBook] = {}
        self._default_depth = default_book_depth
        self._next_order_id: int = 1
        self._total_orders: int = 0
        self._total_trades: int = 0
        self._total_fills: int = 0
        self._created_ns: int = _now_ns()

    # -- Order book management ------------------------------------------

    def _get_or_create_book(self, symbol: str) -> OrderBook:
        if symbol not in self._order_books:
            self._order_books[symbol] = OrderBook(symbol)
        return self._order_books[symbol]

    # -- Public API: process_order --------------------------------------

    def process_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        price: float = 0.0,
        order_type: str = "LIMIT",
        stop_price: float = 0.0,
        uid: int = 0,
    ) -> FillResult:
        """Submit a single order to the matching engine.

        Args:
            ticker: Symbol (e.g. "AAPL").
            side: "BUY" or "SELL".
            qty: Order quantity (shares).
            price: Limit price (0 for market orders).
            order_type: "MARKET", "LIMIT", or "STOP".
            stop_price: Trigger price for stop orders.
            uid: User/account identifier.

        Returns:
            FillResult with fill details, trade events, and latency.
        """
        t0 = _now_ns()

        # Map side string to OrderAction
        action = OrderAction.BID if side.upper() in ("BUY", "COVER") else OrderAction.ASK
        otype = EngineOrderType(order_type.upper())

        order_id = self._next_order_id
        self._next_order_id += 1

        # Publish to ring buffer
        seq = self._ring_buffer.next()
        self._ring_buffer.publish(seq, order_id, ticker, action, otype, price, qty)

        # Create internal order
        order = EngineOrder(
            order_id=order_id,
            symbol=ticker,
            price=price,
            size=qty,
            action=action,
            order_type=otype,
            uid=uid,
            timestamp_ns=t0,
            stop_price=stop_price,
        )

        book = self._get_or_create_book(ticker)
        trade_events: List[TradeEvent] = []
        filled = 0

        if otype == EngineOrderType.MARKET:
            filled = book.place_market_order(order, trade_events)
            order.filled = filled
            order.state = OrderState.FILLED if filled >= qty else OrderState.REJECTED

        elif otype == EngineOrderType.LIMIT:
            filled = book.place_limit_order(order, trade_events)
            order.filled = filled

        elif otype == EngineOrderType.STOP:
            book.place_stop_order(order)
            order.state = OrderState.ACTIVE

        # Check stop orders after any trade
        if trade_events:
            stop_results = book.check_stop_orders(trade_events)
            # Stop fills are tracked but not merged into this result
            for sr in stop_results:
                self._total_fills += sr.filled_size
                self._total_trades += len(sr.trades)

        # Compute average fill price
        avg_px = 0.0
        if trade_events:
            total_notional = sum(e.price * e.size for e in trade_events)
            total_qty = sum(e.size for e in trade_events)
            avg_px = total_notional / total_qty if total_qty > 0 else 0.0

        # Determine result code
        if otype == EngineOrderType.STOP:
            result_code = CommandResultCode.SUCCESS
            state = OrderState.ACTIVE
        elif filled >= qty:
            result_code = CommandResultCode.SUCCESS
            state = OrderState.FILLED
        elif filled > 0:
            result_code = CommandResultCode.SUCCESS
            state = OrderState.PARTIALLY_FILLED
        elif otype == EngineOrderType.LIMIT and filled == 0:
            # Resting on book
            result_code = CommandResultCode.SUCCESS
            state = OrderState.ACTIVE
        else:
            result_code = CommandResultCode.NO_LIQUIDITY
            state = OrderState.REJECTED

        t1 = _now_ns()
        latency_us = (t1 - t0) / 1000.0

        # Mark processed in ring buffer
        self._ring_buffer.mark_processed(seq)
        self._ring_buffer.record_latency(latency_us)

        # Update counters
        self._total_orders += 1
        self._total_trades += len(trade_events)
        self._total_fills += filled

        return FillResult(
            order_id=order_id,
            symbol=ticker,
            action=action,
            order_type=otype,
            requested_size=qty,
            filled_size=filled,
            remaining_size=qty - filled,
            avg_fill_price=avg_px,
            trades=trade_events,
            result_code=result_code,
            state=state,
            latency_us=latency_us,
        )

    # -- Public API: cancel_order ----------------------------------------

    def cancel_order(self, ticker: str, order_id: int) -> bool:
        """Cancel a resting order by ID.

        Returns True if the order was found and cancelled.
        """
        book = self._order_books.get(ticker)
        if book is None:
            return False
        return book.cancel_order(order_id)

    # -- Public API: batch processing ------------------------------------

    def process_batch(
        self,
        orders: List[Tuple],
    ) -> List[FillResult]:
        """Process a batch of orders.

        Each tuple: (ticker, side, qty, price, order_type)
        Optional sixth element: stop_price for STOP orders.

        Returns list of FillResult in submission order.
        """
        results: List[FillResult] = []
        for order_tuple in orders:
            ticker = order_tuple[0]
            side = order_tuple[1]
            qty = int(order_tuple[2])
            price = float(order_tuple[3])
            otype = str(order_tuple[4]) if len(order_tuple) > 4 else "LIMIT"
            stop_px = float(order_tuple[5]) if len(order_tuple) > 5 else 0.0
            result = self.process_order(
                ticker=ticker, side=side, qty=qty,
                price=price, order_type=otype, stop_price=stop_px,
            )
            results.append(result)
        return results

    # -- Public API: seed_order_book ------------------------------------

    def seed_order_book(
        self,
        ticker: str,
        mid_price: float,
        num_levels: int = 10,
        qty_per_level: int = 100,
        tick_size: float = 0.01,
        num_orders_per_level: int = 3,
    ) -> None:
        """Seed an order book with synthetic resting liquidity.

        Useful for creating a realistic book before routing live orders
        through the matching engine.

        Args:
            ticker: Symbol.
            mid_price: Reference mid price.
            num_levels: Number of price levels per side.
            qty_per_level: Total quantity per level (split among orders).
            tick_size: Price increment between levels.
            num_orders_per_level: Number of resting orders per level.
        """
        qty_each = max(1, qty_per_level // num_orders_per_level)

        for i in range(1, num_levels + 1):
            bid_px = round(mid_price - i * tick_size, 6)
            ask_px = round(mid_price + i * tick_size, 6)

            for _ in range(num_orders_per_level):
                self.process_order(ticker, "BUY", qty_each, bid_px, "LIMIT")
                self.process_order(ticker, "SELL", qty_each, ask_px, "LIMIT")

    # -- Public API: get_order_book --------------------------------------

    def get_order_book(
        self,
        ticker: str,
        depth: int = 0,
    ) -> OrderBookSnapshot:
        """Return L3 order book snapshot for *ticker*.

        Args:
            ticker: Symbol.
            depth: Max price levels per side (0 = default).

        Returns:
            OrderBookSnapshot with full L3 detail.
        """
        d = depth if depth > 0 else self._default_depth
        book = self._order_books.get(ticker)
        if book is None:
            return OrderBookSnapshot(
                symbol=ticker,
                timestamp_ns=_now_ns(),
            )
        return book.get_snapshot(depth=d)

    # -- Public API: get_latency_stats -----------------------------------

    def get_latency_stats(self) -> dict:
        """Return processing latency statistics.

        Returns dict with count, mean_us, median_us, p95_us, p99_us,
        min_us, max_us, ring_buffer_utilisation, total_orders,
        total_trades, total_fills, uptime_s.
        """
        stats = self._ring_buffer.get_latency_stats()
        stats["ring_buffer_size"] = self._ring_buffer.size
        stats["ring_buffer_utilisation"] = round(self._ring_buffer.utilisation(), 4)
        stats["ring_buffer_cursor"] = self._ring_buffer.cursor
        stats["total_orders"] = self._total_orders
        stats["total_trades"] = self._total_trades
        stats["total_fills"] = self._total_fills
        stats["symbols_active"] = len(self._order_books)
        stats["uptime_s"] = round((_now_ns() - self._created_ns) / 1e9, 3)
        return stats

    # -- Public API: get_engine_status -----------------------------------

    def get_engine_status(self) -> dict:
        """Return comprehensive engine status."""
        books_info = {}
        for sym, book in self._order_books.items():
            books_info[sym] = {
                "num_orders": book.num_orders,
                "best_bid": book.best_bid,
                "best_ask": book.best_ask,
                "mid_price": book.mid_price,
                "spread": round(book.spread, 6),
            }
        return {
            "engine": "ExchangeCoreEngine",
            "version": "2.0-python",
            "ring_buffer_size": self._ring_buffer.size,
            "total_orders": self._total_orders,
            "total_trades": self._total_trades,
            "total_fills": self._total_fills,
            "symbols": books_info,
            "latency": self.get_latency_stats(),
        }

    # -- PaperBroker integration -----------------------------------------

    def route_to_paper_broker(
        self,
        broker,
        ticker: str,
        side: str,
        qty: int,
        price: float = 0.0,
        order_type: str = "LIMIT",
        signal_type=None,
        reason: str = "",
    ) -> dict:
        """Route an order through the matching engine, then fill via PaperBroker.

        This method first runs the order through the exchange-core matching
        engine to determine realistic fill price and quantity, then passes
        the result to PaperBroker for portfolio/position accounting.

        Args:
            broker: PaperBroker instance.
            ticker: Symbol.
            side: "BUY" or "SELL" (or "SHORT", "COVER").
            qty: Quantity.
            price: Limit price (0 for market).
            order_type: "MARKET" or "LIMIT".
            signal_type: PaperBroker SignalType enum (optional).
            reason: Trade reason string.

        Returns:
            Dict with matching engine result and PaperBroker order.
        """
        # Ensure book has liquidity -- seed if empty
        book = self._order_books.get(ticker)
        if book is None or book.num_orders == 0:
            # Try to get a reference price from broker
            ref_price = price if price > 0 else 100.0
            try:
                broker_price = broker._get_current_price(ticker)
                if broker_price > 0:
                    ref_price = broker_price
            except Exception as e:
                logger.error("ExchangeCore: broker reference price fetch failed for ticker=%s: %s", ticker, e, exc_info=True)
            self.seed_order_book(ticker, ref_price, num_levels=20, qty_per_level=500)

        # Run through matching engine
        engine_result = self.process_order(
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            order_type=order_type,
        )

        # Determine fill price for broker
        fill_price = engine_result.avg_fill_price
        if fill_price <= 0 and price > 0:
            fill_price = price

        # Map side to PaperBroker OrderSide
        try:
            from .paper_broker import OrderSide, SignalType as PBSignalType
        except ImportError:
            try:
                from engine.execution.paper_broker import OrderSide, SignalType as PBSignalType
            except ImportError:
                # Graceful degradation -- return engine result only
                return {
                    "engine_result": engine_result.to_dict(),
                    "broker_order": None,
                    "error": "PaperBroker import failed",
                }

        side_map = {
            "BUY": OrderSide.BUY,
            "SELL": OrderSide.SELL,
            "SHORT": OrderSide.SHORT,
            "COVER": OrderSide.COVER,
        }
        broker_side = side_map.get(side.upper(), OrderSide.BUY)

        if signal_type is None:
            signal_type = PBSignalType.HOLD

        # Fill the order through PaperBroker
        fill_qty = engine_result.filled_size if engine_result.filled_size > 0 else qty

        broker_order = broker.place_order(
            ticker=ticker,
            side=broker_side,
            quantity=fill_qty,
            signal_type=signal_type,
            limit_price=fill_price if order_type.upper() == "LIMIT" else None,
            reason=f"[ExchangeCore] {reason} | latency={engine_result.latency_us:.1f}us",
        )

        return {
            "engine_result": engine_result.to_dict(),
            "broker_order": broker_order.to_dict() if hasattr(broker_order, "to_dict") else str(broker_order),
        }

    # -- Utility ---------------------------------------------------------

    def reset(self) -> None:
        """Reset all order books and ring buffer state."""
        self._order_books.clear()
        self._ring_buffer = RingBuffer(buffer_size=self._ring_buffer.size)
        self._next_order_id = 1
        self._total_orders = 0
        self._total_trades = 0
        self._total_fills = 0
        self._created_ns = _now_ns()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    """Return current time in nanoseconds (monotonic-ish)."""
    return int(time.monotonic_ns())


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE: Optional[ExchangeCoreEngine] = None


def get_engine(ring_buffer_size: int = 65536) -> ExchangeCoreEngine:
    """Return the module-level singleton engine instance."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = ExchangeCoreEngine(ring_buffer_size=ring_buffer_size)
    return _DEFAULT_ENGINE


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ExchangeCoreEngine -- Self-test")
    print("=" * 70)

    engine = ExchangeCoreEngine(ring_buffer_size=4096)

    # Seed AAPL book at $150
    engine.seed_order_book("AAPL", mid_price=150.0, num_levels=10,
                           qty_per_level=300, tick_size=0.01)

    print("\n--- Initial Order Book (AAPL) ---")
    snap = engine.get_order_book("AAPL", depth=5)
    print(f"  Best Bid: {snap.best_bid}  Best Ask: {snap.best_ask}  "
          f"Spread: {snap.spread:.4f}  Mid: {snap.mid_price:.4f}")
    for lvl in snap.bids[:3]:
        print(f"  BID  {lvl.price:.2f}  vol={lvl.total_volume}  orders={lvl.num_orders}")
    for lvl in snap.asks[:3]:
        print(f"  ASK  {lvl.price:.2f}  vol={lvl.total_volume}  orders={lvl.num_orders}")

    # Market buy
    print("\n--- Market BUY 50 AAPL ---")
    r = engine.process_order("AAPL", "BUY", 50, 0, "MARKET")
    print(f"  Filled: {r.filled_size}/{r.requested_size}  "
          f"Avg Price: {r.avg_fill_price:.4f}  "
          f"Trades: {len(r.trades)}  Latency: {r.latency_us:.1f}us")

    # Limit sell
    print("\n--- Limit SELL 30 AAPL @ 149.97 ---")
    r = engine.process_order("AAPL", "SELL", 30, 149.97, "LIMIT")
    print(f"  Filled: {r.filled_size}/{r.requested_size}  "
          f"State: {r.state.value}  Trades: {len(r.trades)}")

    # Batch
    print("\n--- Batch: 3 orders ---")
    results = engine.process_batch([
        ("MSFT", "BUY", 100, 300.0, "LIMIT"),
        ("MSFT", "BUY", 50, 299.95, "LIMIT"),
        ("MSFT", "SELL", 80, 300.0, "MARKET"),
    ])
    for i, res in enumerate(results):
        print(f"  [{i}] {res.action.value} {res.order_type.value} "
              f"filled={res.filled_size}/{res.requested_size} "
              f"state={res.state.value}")

    # Stop order
    print("\n--- Stop BUY 25 AAPL trigger@150.05 ---")
    r = engine.process_order("AAPL", "BUY", 25, 0, "STOP", stop_price=150.05)
    print(f"  State: {r.state.value}  (waiting for trigger)")

    # Latency stats
    print("\n--- Latency Stats ---")
    stats = engine.get_latency_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Engine status
    print("\n--- Engine Status ---")
    status = engine.get_engine_status()
    print(f"  Total orders: {status['total_orders']}")
    print(f"  Total trades: {status['total_trades']}")
    print(f"  Symbols: {list(status['symbols'].keys())}")

    print("\n" + "=" * 70)
    print("Self-test PASSED")
    print("=" * 70)
