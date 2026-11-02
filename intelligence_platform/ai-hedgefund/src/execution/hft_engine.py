# -*- coding: utf-8 -*-
"""
HFT Execution Engine
=====================

Manages trade execution, position tracking, stop-loss monitoring,
portfolio rebalancing, and P&L calculation for the AI hedge fund.

Execution Quality Metrics:
    Slippage = (execution_price - decision_price) / decision_price
    Implementation Shortfall = (paper_return - actual_return)
    Fill Rate = filled_quantity / ordered_quantity

P&L Calculations:
    Realized P&L (long):  (exit_price - entry_price) * quantity
    Realized P&L (short): (entry_price - exit_price) * quantity
    Unrealized P&L:       (current_price - entry_price) * quantity  [long]
                          (entry_price - current_price) * quantity  [short]
    Total P&L = Realized + Unrealized

Position Tracking:
    Net Exposure = sum(long_notional) - sum(short_notional)
    Gross Exposure = sum(long_notional) + sum(short_notional)
    Net Leverage = Net Exposure / Portfolio Value
    Gross Leverage = Gross Exposure / Portfolio Value
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.openbb_universe import (
    AssetClass,
    UniverseData,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_var,
    detect_asset_class,
    get_historical,
)
from src.strategy.multi_horizon import (
    MultiHorizonEngine,
    TradeHorizon,
    TradeThesis,
    calculate_position_size,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Enums and Data Classes
# ===================================================================
class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Represents a trade order."""

    order_id: str
    symbol: str
    direction: str  # "buy" or "sell"
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fill_timestamp: Optional[datetime] = None
    thesis_id: Optional[str] = None


@dataclass
class Position:
    """Represents an open position."""

    position_id: str
    symbol: str
    asset_class: AssetClass
    direction: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    target_price: float
    horizon: TradeHorizon
    entry_time: datetime
    thesis: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_favorable_excursion: float = 0.0  # best unrealized P&L during life
    max_adverse_excursion: float = 0.0    # worst unrealized P&L during life

    def update_price(self, price: float) -> None:
        """Update current price and recalculate unrealized P&L."""
        self.current_price = price
        if self.direction == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - price) / self.entry_price

        self.max_favorable_excursion = max(
            self.max_favorable_excursion, self.unrealized_pnl
        )
        self.max_adverse_excursion = min(
            self.max_adverse_excursion, self.unrealized_pnl
        )


@dataclass
class PositionUpdate:
    """Represents an update action on an existing position."""

    position_id: str
    symbol: str
    action: str  # "hold", "close", "reduce", "add", "trail_stop"
    reason: str
    new_stop: Optional[float] = None
    new_target: Optional[float] = None
    close_quantity: Optional[float] = None


@dataclass
class StopTriggered:
    """Information about a triggered stop-loss."""

    position_id: str
    symbol: str
    direction: str
    stop_price: float
    trigger_price: float
    realized_pnl: float
    pnl_pct: float
    holding_period_days: float


@dataclass
class TradeResult:
    """Result of a trade execution."""

    order: Order
    position: Optional[Position]
    success: bool
    message: str
    execution_cost: float = 0.0  # commissions + slippage


@dataclass
class DailyPnL:
    """Daily P&L breakdown."""

    date: datetime
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    pnl_by_asset_class: Dict[str, float]
    pnl_by_horizon: Dict[str, float]
    pnl_by_symbol: Dict[str, float]
    num_trades: int
    num_winners: int
    num_losers: int
    win_rate: float
    gross_exposure: float
    net_exposure: float
    portfolio_value: float
    sharpe_ratio_mtd: float
    max_drawdown_mtd: float
    var_95: float


# ===================================================================
# HFT Execution Engine
# ===================================================================
class HFTExecutionEngine:
    """
    Manages the full lifecycle of trades from execution through
    position management, stop-loss enforcement, and P&L tracking.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_gross_leverage: float = 2.0,
        max_net_leverage: float = 1.0,
        default_commission_bps: float = 1.0,  # 1 basis point
        default_slippage_bps: float = 2.0,    # 2 basis points
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_gross_leverage = max_gross_leverage
        self.max_net_leverage = max_net_leverage
        self.commission_bps = default_commission_bps
        self.slippage_bps = default_slippage_bps

        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.closed_trades: List[Dict[str, Any]] = []
        self.orders: List[Order] = []
        self.daily_pnl_history: List[DailyPnL] = []

        # Track daily returns for Sharpe/drawdown
        self._daily_returns: List[float] = []
        self._portfolio_values: List[float] = [initial_capital]

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value = cash + unrealized P&L."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute notional of all positions."""
        return sum(
            abs(p.current_price * p.quantity) for p in self.positions.values()
        )

    @property
    def net_exposure(self) -> float:
        """Long notional minus short notional."""
        long_not = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
            if p.direction == "long"
        )
        short_not = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
            if p.direction == "short"
        )
        return long_not - short_not

    @property
    def gross_leverage(self) -> float:
        pv = self.portfolio_value
        return self.gross_exposure / pv if pv > 0 else 0.0

    @property
    def net_leverage(self) -> float:
        pv = self.portfolio_value
        return self.net_exposure / pv if pv > 0 else 0.0

    # ----- Trade Execution -----
    def execute_trade(self, thesis: TradeThesis) -> TradeResult:
        """
        Execute a trade based on a TradeThesis.

        Process:
        1. Validate position limits (leverage, concentration)
        2. Calculate execution price (with slippage model)
        3. Create order and simulate fill
        4. Open position and debit/credit cash
        5. Return TradeResult

        Slippage Model:
            execution_price = decision_price * (1 + direction_sign * slippage_bps/10000)
            where direction_sign = +1 for buy, -1 for sell

        Commission Model:
            commission = notional * commission_bps / 10000

        Parameters
        ----------
        thesis : TradeThesis

        Returns
        -------
        TradeResult
        """
        # 1. Validate leverage limits
        notional = thesis.entry_price * thesis.position_size * self.portfolio_value / thesis.entry_price
        quantity = thesis.position_size * self.portfolio_value / thesis.entry_price

        if quantity <= 0:
            return TradeResult(
                order=Order(
                    order_id=str(uuid.uuid4()),
                    symbol=thesis.symbol,
                    direction="buy" if thesis.direction == "long" else "sell",
                    quantity=0,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                ),
                position=None,
                success=False,
                message="Position size too small",
            )

        projected_gross = self.gross_exposure + abs(quantity * thesis.entry_price)
        if projected_gross / max(self.portfolio_value, 1) > self.max_gross_leverage:
            return TradeResult(
                order=Order(
                    order_id=str(uuid.uuid4()),
                    symbol=thesis.symbol,
                    direction="buy" if thesis.direction == "long" else "sell",
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                ),
                position=None,
                success=False,
                message=f"Would exceed max gross leverage ({self.max_gross_leverage}x)",
            )

        # 2. Calculate execution price with slippage
        direction_sign = 1.0 if thesis.direction == "long" else -1.0
        slippage_mult = 1.0 + direction_sign * self.slippage_bps / 10000.0
        exec_price = thesis.entry_price * slippage_mult
        slippage_cost = abs(exec_price - thesis.entry_price) * quantity

        # 3. Commission
        notional_value = exec_price * quantity
        commission = notional_value * self.commission_bps / 10000.0

        # 4. Check cash sufficiency for long trades
        total_cost = notional_value + commission if thesis.direction == "long" else commission
        if thesis.direction == "long" and total_cost > self.cash:
            # Reduce quantity to fit available cash
            max_quantity = (self.cash - commission) / exec_price
            if max_quantity <= 0:
                return TradeResult(
                    order=Order(
                        order_id=str(uuid.uuid4()),
                        symbol=thesis.symbol,
                        direction="buy",
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.REJECTED,
                    ),
                    position=None,
                    success=False,
                    message="Insufficient cash",
                )
            quantity = max_quantity
            notional_value = exec_price * quantity
            commission = notional_value * self.commission_bps / 10000.0

        # 5. Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=thesis.symbol,
            direction="buy" if thesis.direction == "long" else "sell",
            quantity=quantity,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            fill_price=exec_price,
            fill_quantity=quantity,
            slippage=abs(exec_price - thesis.entry_price),
            fill_timestamp=datetime.utcnow(),
            thesis_id=thesis.symbol + "_" + thesis.horizon.value,
        )
        self.orders.append(order)

        # 6. Open position
        position = Position(
            position_id=str(uuid.uuid4()),
            symbol=thesis.symbol,
            asset_class=thesis.asset_class,
            direction=thesis.direction,
            quantity=quantity,
            entry_price=exec_price,
            current_price=exec_price,
            stop_loss=thesis.stop_loss,
            target_price=thesis.target_price,
            horizon=thesis.horizon,
            entry_time=datetime.utcnow(),
            thesis=thesis.thesis,
        )
        self.positions[position.position_id] = position

        # 7. Update cash
        if thesis.direction == "long":
            self.cash -= notional_value + commission
        else:
            # Short: receive proceeds but set aside margin
            self.cash -= commission  # margin handled implicitly via leverage limits

        logger.info(
            "Executed %s %s %.2f shares of %s at %.4f (slippage: %.4f, commission: %.2f)",
            thesis.direction,
            thesis.horizon.value,
            quantity,
            thesis.symbol,
            exec_price,
            slippage_cost,
            commission,
        )

        return TradeResult(
            order=order,
            position=position,
            success=True,
            message=f"Filled {quantity:.2f} shares at {exec_price:.4f}",
            execution_cost=slippage_cost + commission,
        )

    # ----- Position Management -----
    def manage_open_positions(
        self,
        current_prices: Dict[str, float],
    ) -> List[PositionUpdate]:
        """
        Review and manage all open positions.

        Management rules:
        1. Update current prices and unrealized P&L
        2. Trail stops for profitable positions:
           - If P&L > 2x risk, move stop to breakeven
           - If P&L > 3x risk, trail stop to lock in 50% of profit
        3. Time-based management:
           - HFT: close if open > 1 hour
           - Day trade: close at EOD
           - Swing: review after 5 days
        4. Reduce positions approaching target

        Parameters
        ----------
        current_prices : dict
            {symbol: current_price}

        Returns
        -------
        list of PositionUpdate
        """
        updates: List[PositionUpdate] = []

        for pos_id, pos in list(self.positions.items()):
            if pos.symbol not in current_prices:
                updates.append(PositionUpdate(
                    position_id=pos_id,
                    symbol=pos.symbol,
                    action="hold",
                    reason="No current price available",
                ))
                continue

            price = current_prices[pos.symbol]
            pos.update_price(price)

            # Calculate risk (distance from entry to stop)
            if pos.direction == "long":
                initial_risk = pos.entry_price - pos.stop_loss
                profit = price - pos.entry_price
                distance_to_target = pos.target_price - price
            else:
                initial_risk = pos.stop_loss - pos.entry_price
                profit = pos.entry_price - price
                distance_to_target = price - pos.target_price

            if initial_risk <= 0:
                initial_risk = pos.entry_price * 0.02  # fallback 2%

            risk_multiple = profit / initial_risk if initial_risk > 0 else 0

            # Time-based management
            holding_hours = (datetime.utcnow() - pos.entry_time).total_seconds() / 3600
            holding_days = holding_hours / 24

            # HFT: close after 1 hour
            if pos.horizon == TradeHorizon.HFT_INTRADAY and holding_hours > 1.0:
                self._close_position(pos_id, price, "Time limit reached (HFT > 1 hour)")
                updates.append(PositionUpdate(
                    position_id=pos_id,
                    symbol=pos.symbol,
                    action="close",
                    reason="HFT time limit (> 1 hour)",
                ))
                continue

            # Day trade: close after market hours
            if pos.horizon == TradeHorizon.DAY_TRADE and holding_hours > 8.0:
                self._close_position(pos_id, price, "Day trade EOD close")
                updates.append(PositionUpdate(
                    position_id=pos_id,
                    symbol=pos.symbol,
                    action="close",
                    reason="Day trade EOD close",
                ))
                continue

            # Target reached (within 1%)
            if distance_to_target <= 0 or (distance_to_target / pos.entry_price < 0.01):
                self._close_position(pos_id, price, "Target price reached")
                updates.append(PositionUpdate(
                    position_id=pos_id,
                    symbol=pos.symbol,
                    action="close",
                    reason=f"Target reached at {price:.2f}",
                ))
                continue

            # Trail stops
            if risk_multiple >= 3.0:
                # Lock in 50% of profit
                if pos.direction == "long":
                    new_stop = pos.entry_price + profit * 0.5
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop
                        updates.append(PositionUpdate(
                            position_id=pos_id,
                            symbol=pos.symbol,
                            action="trail_stop",
                            reason=f"Profit > 3R, trailing stop to lock 50% ({new_stop:.2f})",
                            new_stop=new_stop,
                        ))
                        continue
                else:
                    new_stop = pos.entry_price - profit * 0.5
                    if new_stop < pos.stop_loss:
                        pos.stop_loss = new_stop
                        updates.append(PositionUpdate(
                            position_id=pos_id,
                            symbol=pos.symbol,
                            action="trail_stop",
                            reason=f"Profit > 3R, trailing stop to lock 50% ({new_stop:.2f})",
                            new_stop=new_stop,
                        ))
                        continue

            elif risk_multiple >= 2.0:
                # Move stop to breakeven
                if pos.direction == "long" and pos.stop_loss < pos.entry_price:
                    pos.stop_loss = pos.entry_price
                    updates.append(PositionUpdate(
                        position_id=pos_id,
                        symbol=pos.symbol,
                        action="trail_stop",
                        reason="Profit > 2R, stop moved to breakeven",
                        new_stop=pos.entry_price,
                    ))
                    continue
                elif pos.direction == "short" and pos.stop_loss > pos.entry_price:
                    pos.stop_loss = pos.entry_price
                    updates.append(PositionUpdate(
                        position_id=pos_id,
                        symbol=pos.symbol,
                        action="trail_stop",
                        reason="Profit > 2R, stop moved to breakeven",
                        new_stop=pos.entry_price,
                    ))
                    continue

            updates.append(PositionUpdate(
                position_id=pos_id,
                symbol=pos.symbol,
                action="hold",
                reason=f"P&L: {pos.unrealized_pnl_pct * 100:.2f}%, R-multiple: {risk_multiple:.1f}",
            ))

        return updates

    def check_stop_losses(
        self,
        current_prices: Dict[str, float],
    ) -> List[StopTriggered]:
        """
        Check all positions for stop-loss triggers.

        Stop-loss logic:
            Long:  triggered if current_price <= stop_loss
            Short: triggered if current_price >= stop_loss

        Parameters
        ----------
        current_prices : dict
            {symbol: current_price}

        Returns
        -------
        list of StopTriggered
        """
        triggered: List[StopTriggered] = []

        for pos_id, pos in list(self.positions.items()):
            if pos.symbol not in current_prices:
                continue

            price = current_prices[pos.symbol]
            stop_hit = False

            if pos.direction == "long" and price <= pos.stop_loss:
                stop_hit = True
            elif pos.direction == "short" and price >= pos.stop_loss:
                stop_hit = True

            if stop_hit:
                # Close position at stop price (or current price, whichever is worse)
                if pos.direction == "long":
                    exit_price = min(price, pos.stop_loss)
                    realized_pnl = (exit_price - pos.entry_price) * pos.quantity
                else:
                    exit_price = max(price, pos.stop_loss)
                    realized_pnl = (pos.entry_price - exit_price) * pos.quantity

                pnl_pct = realized_pnl / (pos.entry_price * pos.quantity) if pos.quantity > 0 else 0
                holding_days = (datetime.utcnow() - pos.entry_time).total_seconds() / 86400

                triggered.append(StopTriggered(
                    position_id=pos_id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    stop_price=pos.stop_loss,
                    trigger_price=price,
                    realized_pnl=realized_pnl,
                    pnl_pct=pnl_pct,
                    holding_period_days=holding_days,
                ))

                self._close_position(pos_id, exit_price, "Stop-loss triggered")

                logger.warning(
                    "STOP TRIGGERED: %s %s %s at %.2f (stop: %.2f), P&L: %.2f (%.2f%%)",
                    pos.direction, pos.symbol, pos.horizon.value,
                    price, pos.stop_loss, realized_pnl, pnl_pct * 100,
                )

        return triggered

    def _close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str,
    ) -> float:
        """
        Close a position and return realized P&L.

        Parameters
        ----------
        position_id : str
        exit_price : float
        reason : str

        Returns
        -------
        float
            Realized P&L.
        """
        if position_id not in self.positions:
            return 0.0

        pos = self.positions.pop(position_id)

        if pos.direction == "long":
            realized_pnl = (exit_price - pos.entry_price) * pos.quantity
            notional_return = exit_price * pos.quantity
        else:
            realized_pnl = (pos.entry_price - exit_price) * pos.quantity
            notional_return = 0  # short cover

        # Commission on exit
        commission = abs(exit_price * pos.quantity) * self.commission_bps / 10000.0

        # Update cash
        if pos.direction == "long":
            self.cash += notional_return - commission
        else:
            self.cash += realized_pnl - commission

        self.closed_trades.append({
            "position_id": position_id,
            "symbol": pos.symbol,
            "asset_class": pos.asset_class.value,
            "direction": pos.direction,
            "horizon": pos.horizon.value,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "realized_pnl": realized_pnl,
            "pnl_pct": realized_pnl / (pos.entry_price * pos.quantity) if pos.quantity > 0 else 0,
            "commission": commission,
            "entry_time": pos.entry_time,
            "exit_time": datetime.utcnow(),
            "holding_period_days": (datetime.utcnow() - pos.entry_time).total_seconds() / 86400,
            "reason": reason,
            "max_favorable_excursion": pos.max_favorable_excursion,
            "max_adverse_excursion": pos.max_adverse_excursion,
        })

        logger.info(
            "Closed %s %s: P&L=%.2f (%.2f%%), reason=%s",
            pos.direction, pos.symbol, realized_pnl,
            (realized_pnl / (pos.entry_price * pos.quantity) * 100) if pos.quantity > 0 else 0,
            reason,
        )

        return realized_pnl

    # ----- Rebalancing -----
    def rebalance_portfolio(
        self,
        engine: MultiHorizonEngine,
        universe_data: UniverseData,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[TradeThesis]:
        """
        Rebalance the portfolio by:
        1. Closing positions that no longer have valid theses
        2. Reducing overweight positions
        3. Adding new top-scoring opportunities

        Rebalancing rules:
            - Max single-name concentration: 15% of portfolio
            - Max sector concentration: 30% of portfolio
            - Maintain target gross leverage <= max_gross_leverage
            - Prioritize highest composite-score opportunities

        Parameters
        ----------
        engine : MultiHorizonEngine
        universe_data : UniverseData
        macro_data : dict, optional

        Returns
        -------
        list of TradeThesis
            New trades to execute.
        """
        new_trades: List[TradeThesis] = []

        # Check concentration limits
        symbol_exposure: Dict[str, float] = {}
        sector_exposure: Dict[str, float] = {}
        pv = self.portfolio_value

        for pos in self.positions.values():
            notional = abs(pos.current_price * pos.quantity)
            symbol_exposure[pos.symbol] = symbol_exposure.get(pos.symbol, 0) + notional

            from src.data.openbb_universe import classify_by_gics
            cls = classify_by_gics(pos.symbol)
            if cls:
                sector_exposure[cls.sector_name] = (
                    sector_exposure.get(cls.sector_name, 0) + notional
                )

        # Get top opportunities from engine
        top_opps = engine.get_top_opportunities(n=20)

        for thesis in top_opps:
            # Skip if already have a position in this symbol
            if thesis.symbol in symbol_exposure:
                current_pct = symbol_exposure[thesis.symbol] / pv
                if current_pct > 0.12:  # already near max concentration
                    continue

            # Check sector concentration
            cls = classify_by_gics(thesis.symbol)
            if cls and cls.sector_name in sector_exposure:
                sector_pct = sector_exposure[cls.sector_name] / pv
                if sector_pct > 0.25:  # near max sector concentration
                    continue

            # Check leverage capacity
            new_notional = thesis.position_size * pv
            if (self.gross_exposure + new_notional) / pv > self.max_gross_leverage:
                continue

            new_trades.append(thesis)

            # Update tracking
            symbol_exposure[thesis.symbol] = (
                symbol_exposure.get(thesis.symbol, 0) + new_notional
            )
            if cls:
                sector_exposure[cls.sector_name] = (
                    sector_exposure.get(cls.sector_name, 0) + new_notional
                )

        return new_trades

    # ----- P&L -----
    def calculate_pnl(self) -> DailyPnL:
        """
        Calculate daily P&L breakdown across all dimensions.

        P&L Components:
            Realized:   sum of all closed trades today
            Unrealized: sum of mark-to-market on open positions
            Total:      Realized + Unrealized

        Risk Metrics:
            Sharpe (MTD):     annualized(mean(daily_returns) / std(daily_returns))
            Max Drawdown MTD: max peak-to-trough decline in portfolio value
            VaR (95%):        5th percentile of daily returns * portfolio_value

        Returns
        -------
        DailyPnL
        """
        now = datetime.utcnow()

        # Realized P&L from today's closed trades
        today_closed = [
            t for t in self.closed_trades
            if t.get("exit_time") and t["exit_time"].date() == now.date()
        ]
        realized_pnl = sum(t["realized_pnl"] for t in today_closed)

        # Unrealized P&L
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        total_pnl = realized_pnl + unrealized_pnl

        # Breakdown by asset class
        pnl_by_asset: Dict[str, float] = {}
        for pos in self.positions.values():
            ac = pos.asset_class.value
            pnl_by_asset[ac] = pnl_by_asset.get(ac, 0) + pos.unrealized_pnl
        for t in today_closed:
            ac = t.get("asset_class", "unknown")
            pnl_by_asset[ac] = pnl_by_asset.get(ac, 0) + t["realized_pnl"]

        # Breakdown by horizon
        pnl_by_horizon: Dict[str, float] = {}
        for pos in self.positions.values():
            h = pos.horizon.value
            pnl_by_horizon[h] = pnl_by_horizon.get(h, 0) + pos.unrealized_pnl
        for t in today_closed:
            h = t.get("horizon", "unknown")
            pnl_by_horizon[h] = pnl_by_horizon.get(h, 0) + t["realized_pnl"]

        # Breakdown by symbol
        pnl_by_symbol: Dict[str, float] = {}
        for pos in self.positions.values():
            pnl_by_symbol[pos.symbol] = (
                pnl_by_symbol.get(pos.symbol, 0) + pos.unrealized_pnl
            )
        for t in today_closed:
            pnl_by_symbol[t["symbol"]] = (
                pnl_by_symbol.get(t["symbol"], 0) + t["realized_pnl"]
            )

        # Trade statistics
        num_trades = len(today_closed)
        winners = [t for t in today_closed if t["realized_pnl"] > 0]
        losers = [t for t in today_closed if t["realized_pnl"] <= 0]
        win_rate = len(winners) / num_trades if num_trades > 0 else 0.0

        # Update portfolio value tracking
        pv = self.portfolio_value
        self._portfolio_values.append(pv)
        if len(self._portfolio_values) >= 2:
            daily_ret = (
                self._portfolio_values[-1] / self._portfolio_values[-2] - 1
            )
            self._daily_returns.append(daily_ret)

        # Risk metrics (MTD)
        returns_series = pd.Series(self._daily_returns) if self._daily_returns else pd.Series(dtype=float)
        sharpe_mtd = compute_sharpe_ratio(returns_series, periods_per_year=252) if len(returns_series) > 1 else 0.0

        pv_series = pd.Series(self._portfolio_values) if self._portfolio_values else pd.Series(dtype=float)
        mdd_mtd = compute_max_drawdown(pv_series) if len(pv_series) > 1 else 0.0

        var_95 = compute_var(returns_series, confidence=0.95) * pv if len(returns_series) > 5 else 0.0

        daily_pnl = DailyPnL(
            date=now,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            pnl_by_asset_class=pnl_by_asset,
            pnl_by_horizon=pnl_by_horizon,
            pnl_by_symbol=pnl_by_symbol,
            num_trades=num_trades,
            num_winners=len(winners),
            num_losers=len(losers),
            win_rate=win_rate,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            portfolio_value=pv,
            sharpe_ratio_mtd=sharpe_mtd,
            max_drawdown_mtd=mdd_mtd,
            var_95=var_95,
        )

        self.daily_pnl_history.append(daily_pnl)
        return daily_pnl

    # ----- Reporting Helpers -----
    def get_position_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all open positions."""
        if not self.positions:
            return pd.DataFrame()

        rows = []
        for pos in self.positions.values():
            rows.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "target": pos.target_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct * 100,
                "horizon": pos.horizon.value,
                "asset_class": pos.asset_class.value,
                "holding_days": (datetime.utcnow() - pos.entry_time).total_seconds() / 86400,
            })
        return pd.DataFrame(rows).sort_values("unrealized_pnl", ascending=False)

    def get_trade_history(self) -> pd.DataFrame:
        """Return a DataFrame of all closed trades."""
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_trades).sort_values("exit_time", ascending=False)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Compute overall performance statistics.

        Metrics:
            Total Return = (Portfolio Value - Initial Capital) / Initial Capital
            CAGR = (PV / IC)^(1/years) - 1
            Sharpe Ratio (annualized)
            Sortino Ratio (annualized)
            Max Drawdown
            Win Rate
            Profit Factor = gross_profits / gross_losses
            Average Win / Average Loss
            Expectancy = win_rate * avg_win - loss_rate * avg_loss
        """
        pv = self.portfolio_value
        total_return = (pv - self.initial_capital) / self.initial_capital

        returns_series = pd.Series(self._daily_returns) if self._daily_returns else pd.Series(dtype=float)
        pv_series = pd.Series(self._portfolio_values) if self._portfolio_values else pd.Series(dtype=float)

        # Trade-level stats
        wins = [t for t in self.closed_trades if t["realized_pnl"] > 0]
        losses = [t for t in self.closed_trades if t["realized_pnl"] <= 0]
        num_trades = len(self.closed_trades)

        gross_profit = sum(t["realized_pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["realized_pnl"] for t in losses)) if losses else 0

        avg_win = np.mean([t["realized_pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["realized_pnl"] for t in losses])) if losses else 0

        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        from src.data.openbb_universe import compute_sortino_ratio as _sortino, compute_sharpe_ratio as _sharpe

        return {
            "portfolio_value": pv,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": _sharpe(returns_series) if len(returns_series) > 1 else 0,
            "sortino_ratio": _sortino(returns_series) if len(returns_series) > 1 else 0,
            "max_drawdown_pct": compute_max_drawdown(pv_series) * 100 if len(pv_series) > 1 else 0,
            "num_trades": num_trades,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "gross_leverage": self.gross_leverage,
            "net_leverage": self.net_leverage,
            "num_open_positions": len(self.positions),
        }
