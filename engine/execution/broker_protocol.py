"""
BrokerProtocol — Formal typing.Protocol for all Metadron broker implementations.

Defines the shared interface that IBKRBroker implements
all implement by convention. This Protocol makes that contract explicit and
runtime-checkable via isinstance().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BROKER SWAP NOTES — READ BEFORE ADDING A NEW BROKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IBKRBroker implements the shared broker interface
by convention.  BrokerProtocol formalises that contract so:
  (a) type checkers (mypy, pyright) catch mismatches at dev-time, and
  (b) runtime guards like `assert isinstance(broker, BrokerProtocol)` work.

HOW TO ADD IBKR (or any other new broker):
────────────────────────────────────────────
STEP 1 — Import shared dataclasses from paper_broker.py
    The paper_broker module is the canonical source of all shared data types.
    IBKRBroker implements this pattern:

        from engine.execution.paper_broker import (
            Order, OrderSide, OrderType, OrderStatus, SignalType,
            Position, PortfolioState, RiskLimiter, PerformanceTracker,
            DailyTargetManager, LiveDashboardState, RiskProfile,
        )

    DO NOT redefine these types — share the originals so isinstance() checks
    remain consistent across all brokers and routers.

STEP 2 — Implement all Protocol methods
    Your new broker class must implement every method listed in BrokerProtocol
    below.  The minimal required surface is:

        place_order, get_trade_history, get_portfolio_summary,
        get_all_positions, compute_nav, compute_exposures, get_drawdown,
        refresh_prices, get_risk_profile, get_daily_target_state,
        reset_daily_target, get_leverage_multiplier, emit_dashboard_state,
        get_dashboard_snapshot, get_dashboard_history,
        register_dashboard_callback, get_performance_metrics,
        get_daily_pnl, reconcile, export_positions_csv

    And required attributes:
        state            : PortfolioState
        _perf_tracker    : PerformanceTracker
        _daily_pnl_today : float
        paper            : bool

    TIP — get_trade_history() MUST return list[dict], not list[Order].
    All three current brokers return dicts.  The router attribute-access bug
    (portfolio.py, risk.py) uses t.get("ticker") dict access in several places;
    keeping the dict return type preserves compatibility.

STEP 3 — Wire the broker into ExecutionEngine and L7
    Open engine/execution/execution_engine.py and add a branch in
    ExecutionEngine.__init__() alongside the existing "ibkr" / "paper" paths:

        elif broker_type == "ibkr":
            from engine.execution.ibkr_broker import IBKRBroker
            self.broker = IBKRBroker(initial_cash=initial_nav or 1_000_000.0)

    Then repeat the same addition in
    engine/execution/l7_unified_execution_surface.py inside
    L7UnifiedExecutionSurface.__init__() (L7 silently falls back to paper on
    failure, so wrap in try/except following the existing pattern there).

    Environment variable convention — add to .env / README:
        METADRON_BROKER_TYPE=ibkr

    The shared singleton module (engine/api/shared.py) already reads
    METADRON_BROKER_TYPE and forwards it to ExecutionEngine().

STEP 4 — Validate compliance at import time
    Add this assertion at the bottom of your new broker file:

        # Validate Protocol compliance at import time
        assert isinstance(IBKRBroker(initial_cash=0), BrokerProtocol), (
            "IBKRBroker does not satisfy BrokerProtocol — add missing methods."
        )

    Or write a unit test:

        from engine.execution.broker_protocol import BrokerProtocol
        from engine.execution.ibkr_broker import IBKRBroker

        def test_ibkr_satisfies_protocol():
            broker = IBKRBroker(initial_cash=100_000)
            assert isinstance(broker, BrokerProtocol)

STEP 5 — Update the shared.py BROKER SWAP NOTE
    Edit the docstring at the top of engine/api/shared.py to list the new
    broker type so other developers know it exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IBKR BROKER STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Legacy TradierBroker reference:
    engine/execution/tradier_broker.py

It satisfies BrokerProtocol (same method surface as PaperBroker /
IBKRBroker is the sole broker wired into L7UnifiedExecutionSurface.
L7UnifiedExecutionSurface — there is no broker_type="ibkr" path in either
class's __init__().

To activate TradierBroker, follow Step 3 above with:

    elif broker_type == "ibkr":
        from engine.execution.tradier_broker import TradierBroker
        self.broker = TradierBroker(initial_cash=initial_nav or 1_000_000.0)

And set:
    METADRON_BROKER_TYPE=ibkr
    IBKR_API_KEY=<your-key>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SINGLETON / IMPORT NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All routers should import broker/engine instances from engine/api/shared.py
(not instantiate them directly) to avoid the singleton isolation problem
documented in the audit (Bug 4).  The shared module ensures a single
ExecutionEngine — and therefore a single broker — is shared across all tabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Imported for type hints only — avoids circular imports at runtime.
    # These types are defined in paper_broker.py and imported by all brokers.
    import pandas as pd

    from engine.execution.paper_broker import (
        Order,
        OrderSide,
        OrderType,
        PortfolioState,
        PerformanceTracker,
        Position,
        SignalType,
    )


# ---------------------------------------------------------------------------
# BrokerProtocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BrokerProtocol(Protocol):
    """Structural interface shared by PaperBroker, IBKRBroker, and IBKRBroker (legacy).

    Any object that implements all methods and attributes below satisfies the
    Protocol automatically — no explicit inheritance required.  Because the
    decorator is ``@runtime_checkable``, you can validate compliance at
    runtime:

        from engine.execution.broker_protocol import BrokerProtocol
        assert isinstance(my_broker, BrokerProtocol)

    AI NOTE — Why Protocol instead of ABC?
    ────────────────────────────────────────
    Using ``typing.Protocol`` rather than ``abc.ABC`` means existing broker
    classes do NOT need to be modified (no need to add ``(BrokerABC)`` to
    their class definitions).  All three current brokers satisfy this Protocol
    by structural subtyping — their method signatures already match.

    When asking an LLM to implement a new broker, provide this file as
    context.  The LLM should fill in every method stub with a concrete
    implementation that calls the target brokerage API.

    Required attributes
    ───────────────────
    state            — live PortfolioState (cash, positions, nav, exposures)
    _perf_tracker    — PerformanceTracker driving Sharpe/drawdown metrics
    _daily_pnl_today — running intraday P&L float (used by L7 risk gates)
    paper            — True if this broker instance runs in simulation mode
    """

    # ------------------------------------------------------------------
    # Required instance attributes
    # ------------------------------------------------------------------

    state: "PortfolioState"
    """Live portfolio state: cash, positions dict, nav, exposures.
    Defined in paper_broker.py as a dataclass.  All brokers own their own
    PortfolioState instance and keep it synchronised after every fill."""

    _perf_tracker: "PerformanceTracker"
    """PerformanceTracker instance.  Used by risk.py /risk/metrics to
    compute Sharpe, Sortino, Calmar, Information Ratio, and Treynor
    ratios.  Must expose get_daily_returns() -> pd.Series."""

    _daily_pnl_today: float
    """Intraday P&L accumulated since midnight UTC.  Checked by L7's
    risk engine to enforce the daily loss limit gate.  Reset to 0.0 at
    start of each trading day (see reset_daily_target)."""

    paper: bool
    """True when the broker is running in simulation mode (no real
    orders sent to an exchange).  AlpacaBroker exposes this as
    ``self.paper = os.getenv("IBKR_PAPER_TRADE", "True") == "true"``.
    Used by get_broker_status() and the shared API singleton to label
    the active environment in dashboard headers."""

    # ------------------------------------------------------------------
    # Core trading methods
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: "OrderSide",
        quantity: int,
        order_type: "OrderType" = ...,   # type: ignore[assignment]
        signal_type: "SignalType" = ...,  # type: ignore[assignment]
        limit_price: float | None = None,
        reason: str = "",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> "Order":
        """Submit an order and return the filled Order dataclass.

        AI NOTE:
            In PaperBroker this fills synchronously via MicroPriceModel.
            In AlpacaBroker this submits to the Alpaca SDK, polls for fill
            status (_poll_order_fill), then calls _sync_after_fill.
            A new broker must return an Order dataclass (from paper_broker.py)
            so that the ExecutionEngine pipeline logging stays consistent.
        """
        ...

    def get_trade_history(self) -> list[dict]:
        """Return the full trade log as a list of dicts.

        AI NOTE — IMPORTANT:
            All three current brokers return list[dict], NOT list[Order].
            Each dict is the result of Order.to_dict() plus extra metadata
            (nav_at_fill, cash_after, spread_bps, etc.).
            Router code in portfolio.py and risk.py uses t.get("ticker", "")
            dict access; do NOT change this return type or those endpoints
            will silently break.
        """
        ...

    def get_portfolio_summary(self) -> dict:
        """Return a summary dict of current portfolio state.

        Expected keys (all routers rely on these):
            nav, cash, total_pnl, positions (count), gross_exposure,
            net_exposure, win_count, loss_count, beta, risk_profile
        """
        ...

    def get_all_positions(self) -> "dict[str, Position]":
        """Return all open positions as {ticker: Position}.

        Position is a dataclass from paper_broker.py with fields:
            ticker, quantity, avg_cost, current_price, unrealized_pnl,
            realized_pnl, sector.

        AI NOTE:
            AlpacaBroker syncs positions from the Alpaca SDK on every call.
            A new live broker should do the same — query the exchange and
            update self.state.positions before returning.
        """
        ...

    # ------------------------------------------------------------------
    # NAV / exposure analytics
    # ------------------------------------------------------------------

    def compute_nav(self) -> float:
        """Recompute and return current Net Asset Value.

        cash + sum(position.quantity * current_price for each position).
        AlpacaBroker prefers the account equity from the Alpaca SDK and
        falls back to the local state calculation.
        """
        ...

    def compute_exposures(self) -> dict:
        """Return gross/net/long/short exposures as fractions of NAV.

        Expected keys: gross, net, long, short
        """
        ...

    # ------------------------------------------------------------------
    # Risk / performance metrics
    # ------------------------------------------------------------------

    def get_drawdown(self) -> dict:
        """Return drawdown metrics delegated to _perf_tracker.get_drawdown().

        Expected keys: current_drawdown, max_drawdown, peak_nav, trough_nav
        """
        ...

    def get_performance_metrics(self) -> dict:
        """Return Sharpe ratio, win-rate, avg win/loss, trade frequency.

        AI NOTE:
            This method is a thin wrapper around _perf_tracker methods.
            When building a new broker, reuse PerformanceTracker from
            paper_broker.py rather than reinventing the calculation.
        """
        ...

    def get_daily_pnl(self) -> "pd.DataFrame":
        """Return a DataFrame of daily P&L from _perf_tracker.

        Index: date (datetime), column: daily_pnl (float)
        Used by /risk/metrics for Sortino / Calmar ratio computation.
        """
        ...

    def get_risk_profile(self) -> str:
        """Return current risk profile: 'AGGRESSIVE', 'MODERATE', or 'DEFENSIVE'.

        Risk profile is managed by DailyTargetManager and changes during the
        session as the 5% daily target is approached or exceeded:
            < 5%  → AGGRESSIVE
            ≥ 5%  → MODERATE
            ≥ 6%  → DEFENSIVE
        """
        ...

    def get_leverage_multiplier(self) -> float:
        """Return the current leverage multiplier based on risk profile.

        Standard mapping:
            AGGRESSIVE → 1.0
            MODERATE   → 0.5
            DEFENSIVE  → 0.2

        ExecutionEngine multiplies all TradeAllocator position sizes by this
        value before calling place_order().
        """
        ...

    # ------------------------------------------------------------------
    # Daily target management
    # ------------------------------------------------------------------

    def get_daily_target_state(self) -> dict:
        """Return DailyTargetManager state dict.

        Expected keys: target_pct, current_pnl_pct, target_hit,
                        risk_profile, leverage_multiplier, reset_time
        """
        ...

    def reset_daily_target(self) -> None:
        """Reset the daily compound target tracker at start of a new session.

        AI NOTE:
            Call this at market open (09:30 ET) or via a scheduled cron job.
            The shared singleton in engine/api/shared.py does NOT call this
            automatically — you must integrate it with the live loop
            orchestrator (engine/live_loop_orchestrator.py).
        """
        ...

    # ------------------------------------------------------------------
    # Price refresh
    # ------------------------------------------------------------------

    def refresh_prices(self) -> None:
        """Re-fetch current prices for all held positions and update state.

        PaperBroker uses OpenBB get_adj_close().
        AlpacaBroker uses the Alpaca market data SDK (StockLatestBarRequest).
        A new broker should use whatever market data feed is cheapest/fastest
        for the target exchange.
        """
        ...

    # ------------------------------------------------------------------
    # Reconciliation / export
    # ------------------------------------------------------------------

    def reconcile(self) -> dict:
        """Reconcile local portfolio state against the exchange.

        PaperBroker does an internal position consistency check.
        AlpacaBroker compares local state against live IBKR account data.
        Returns a dict with keys: status, discrepancies, resolved_at.

        AI NOTE:
            For IBKR, this should compare self.state.positions against
            the IBKR account positions endpoint.  Log any discrepancies
            and update local state to match the exchange (exchange is truth).
        """
        ...

    def export_positions_csv(self, filepath: str | None = None) -> str:
        """Export current positions to CSV.

        If filepath is None, returns CSV content as a string.
        If filepath is provided, writes to that file and returns the path.

        AI NOTE:
            The risk.py router's /risk/futures-positions endpoint does NOT
            call this — it uses get_all_positions() directly.  This method
            is used for end-of-day reporting and audit exports.
        """
        ...

    # ------------------------------------------------------------------
    # Live dashboard integration
    # ------------------------------------------------------------------

    def emit_dashboard_state(self, pipeline_state: dict | None = None) -> None:
        """Fire all registered dashboard callbacks with current broker state.

        Called by ExecutionEngine.run() after each pipeline cycle.
        pipeline_state is an optional dict of execution metadata (timing,
        gate results, vote scores) that gets merged into the emitted state.

        AI NOTE:
            This is a push-based mechanism.  The live dashboard WebSocket
            bridge (engine/monitoring/live_dashboard.py) registers a callback
            via register_dashboard_callback() and receives state updates here.
            Ensure your new broker calls all callbacks with a state dict that
            includes at minimum: nav, cash, positions, risk_profile,
            daily_pnl_pct, leverage_multiplier.
        """
        ...

    def get_dashboard_snapshot(self) -> dict:
        """Return the most recently emitted dashboard state dict.

        Used by /portfolio/live to serve the current state without waiting
        for the next emit_dashboard_state() call.  Returns {} if no state
        has been emitted yet.
        """
        ...

    def get_dashboard_history(self, n: int = 100) -> list[dict]:
        """Return the last n emitted dashboard states.

        Used by the time-series charts on the live dashboard.  The deque
        typically has maxlen=1000 so very old states are discarded.

        AI NOTE:
            Do not persist this across restarts — it is in-memory only.
            For long-term historical data, use get_daily_pnl() which reads
            from the PerformanceTracker's persisted daily returns.
        """
        ...

    def register_dashboard_callback(self, callback: object) -> None:
        """Register a callable to receive broker state on every emit.

        Callbacks are called synchronously inside emit_dashboard_state().
        Signature: callback(state: dict) -> None

        Example usage in a WebSocket handler:
            broker.register_dashboard_callback(
                lambda state: ws_manager.broadcast(json.dumps(state))
            )

        AI NOTE:
            Keep callbacks fast — they run in the execution hot path.
            Long-running I/O in a callback will block order placement.
            Use asyncio.create_task() or a queue if the callback is async.
        """
        ...
