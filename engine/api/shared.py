"""
Metadron Capital — Shared API Singletons

All routers import broker/engine instances from here to ensure
consistent state across tabs. No more independent ExecutionEngine()
per router module.

BROKER SWAP NOTE:
    To change the active broker, set METADRON_BROKER_TYPE env var:
        "ibkr"  — IBKRBroker (default, requires IBKR_API_KEY)
        "paper"   — PaperBroker (simulation only)
        "ibkr"  — IBKRBroker (requires IBKR_API_KEY, not yet wired in ExecutionEngine)
        "ibkr"    — IBKRBroker (future — implement broker_protocol.BrokerProtocol)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM SOLVED BY THIS MODULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before this file existed, each router (portfolio.py, risk.py, execution.py)
called ExecutionEngine() independently inside its own lazy singleton function.
This created three separate ExecutionEngine instances — each with its own
broker, its own trade log, and its own PortfolioState — so dashboard tabs
showed inconsistent data (Audit Bug 4).

This module provides a single, process-wide ExecutionEngine.  Every router
that needs the broker, l7 surface, beta corridor, or options engine imports
from here:

    from engine.api.shared import get_broker, get_engine, get_l7, get_beta, get_options

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW BROKER TYPE IS SELECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Read METADRON_BROKER_TYPE from environment (default: "ibkr").
2. Pass as broker_type to ExecutionEngine.__init__().
3. ExecutionEngine selects IBKRBroker based on that value.
4. IBKRBroker is wired as sole broker via L7UnifiedExecutionSurface — see
   engine/execution/broker_protocol.py BROKER SWAP NOTES for wiring steps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FALLBACK BEHAVIOUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If ExecutionEngine() raises (e.g. missing IBKR_API_KEY), this module falls
back to a bare PaperBroker() and logs a WARNING.  This mirrors the behaviour
of the old portfolio.py and risk.py _get_broker() functions so the dashboard
remains usable in dev / CI environments without broker credentials.

To suppress the fallback and force a hard error on missing credentials, set:
    METADRON_BROKER_STRICT=true

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREAD SAFETY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A single threading.Lock (_init_lock) guards all singleton initialisations.
FastAPI runs request handlers in a thread pool, so concurrent first-requests
on startup could race to create multiple instances without this guard.

The lock is only held during the first initialisation of each singleton — once
the object is created it is stored in the module-level variable and subsequent
calls read it without acquiring the lock.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Imported for type annotations only.  Actual imports happen lazily inside
    # the getter functions to avoid slow module-level imports at startup.
    from engine.execution.broker_protocol import BrokerProtocol
    from engine.execution.execution_engine import ExecutionEngine as _ExecutionEngine
    from engine.execution.l7_unified_execution_surface import L7UnifiedExecutionSurface as _L7
    from engine.portfolio.beta_corridor import BetaCorridor as _BetaCorridor
    from engine.execution.options_engine import OptionsEngine as _OptionsEngine
    from engine.execution.decision_matrix import DecisionMatrix as _DecisionMatrix

logger = logging.getLogger("metadron-api.shared")

# ---------------------------------------------------------------------------
# Lock — guards all lazy singleton initialisations
# ---------------------------------------------------------------------------
_init_lock = threading.RLock()  # re-entrant: get_beta()/get_options() call get_engine() while holding it

# ---------------------------------------------------------------------------
# Singleton state variables
# ---------------------------------------------------------------------------
_engine: Optional["_ExecutionEngine"] = None
_l7: Optional["_L7"] = None
_beta: Optional["_BetaCorridor"] = None
_options: Optional["_OptionsEngine"] = None
_decision: Optional["_DecisionMatrix"] = None

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
def _broker_type() -> str:
    """Read active broker type from environment.

    Returns one of: "ibkr" (default), "paper".

    AI NOTE:
        METADRON_BROKER_TYPE is checked on every first-call to _get_engine().
        Changing it after the singleton is created has no effect until the
        process restarts (or _reset_singletons() is called — dev only).
    """
    return os.getenv("METADRON_BROKER_TYPE", "ibkr").lower().strip()


def _strict_mode() -> bool:
    """If True, raise instead of falling back to PaperBroker on init failure."""
    return os.getenv("METADRON_BROKER_STRICT", "false").lower() == "true"


# ---------------------------------------------------------------------------
# ExecutionEngine singleton
# ---------------------------------------------------------------------------
def get_engine() -> "_ExecutionEngine":
    """Return the shared ExecutionEngine instance (lazy, thread-safe).

    AI NOTE — What is ExecutionEngine?
    ────────────────────────────────────
    ExecutionEngine is the main trading pipeline orchestrator.  It owns:
        - self.broker      : the active broker (IBKRBroker)
        - self.l7          : L7UnifiedExecutionSurface (HFT arm)
        - self.macro       : MacroEngine (regime detection)
        - self.cube        : MetadronCube (signal generation)
        - self.alpha       : AlphaOptimizer (signal ranking)
        - self.beta        : BetaCorridor (futures hedging)
        - self.risk_gates  : RiskGateManager (4-gate pre-trade check)
        - self.ensemble    : MLVoteEnsemble (10-tier weighted vote)
        - self.allocator   : TradeAllocator (position sizing)

    BROKER SWAP EFFECT:
        Changing METADRON_BROKER_TYPE= changes which broker ExecutionEngine
        selects in its __init__().  The rest of the pipeline is broker-agnostic
        — it calls self.broker.place_order() regardless of implementation.

    FALLBACK:
        If ExecutionEngine() raises (e.g. missing IBKR credentials) and
        METADRON_BROKER_STRICT is not set, a minimal PaperBroker is wrapped
        in a FallbackEngine stub so callers always receive a valid object.
        This prevents 500 errors on all execution/portfolio/risk endpoints.
    """
    global _engine
    if _engine is not None:
        return _engine

    with _init_lock:
        if _engine is not None:
            return _engine

        broker = _broker_type()
        logger.info("Shared singleton: initialising ExecutionEngine(broker_type=%r)", broker)
        try:
            from engine.execution.execution_engine import ExecutionEngine
            _engine = ExecutionEngine(broker_type=broker)
            logger.info(
                "Shared singleton: ExecutionEngine ready — broker=%s paper=%s",
                type(_engine.broker).__name__,
                getattr(_engine.broker, "paper", "unknown"),
            )
        except Exception as exc:
            if _strict_mode():
                raise
            logger.warning(
                "Shared singleton: ExecutionEngine init failed (%s) — "
                "falling back to bare PaperBroker.  "
                "Set IBKR_HOST / IBKR_PORT to connect. Default: 127.0.0.1:7497 (paper).",
                exc,
            )
            from engine.execution.execution_engine import ExecutionEngine
            _engine = ExecutionEngine(broker_type="paper")

    return _engine


# ---------------------------------------------------------------------------
# Broker accessor
# ---------------------------------------------------------------------------
def get_broker() -> "BrokerProtocol":
    """Return the active broker from the shared ExecutionEngine.

    AI NOTE — What is the broker?
    ──────────────────────────────
    The broker is the lowest-level order-execution object.  It is always one
    IBKRBroker is the sole execution broker.
    All three satisfy BrokerProtocol.

    You can confirm at runtime:
        from engine.execution.broker_protocol import BrokerProtocol
        assert isinstance(get_broker(), BrokerProtocol)

    BROKER SWAP EFFECT:
        The returned broker changes based on METADRON_BROKER_TYPE:
            "alpaca"  → AlpacaBroker (connects to IBKR brokerage API)
            "paper"   → PaperBroker  (fully simulated, no external calls)
            "tradier" → TradierBroker (not yet wired — falls back to paper)
            "ibkr"    → IBKRBroker  (future implementation)

    All routers should call get_broker() instead of instantiating brokers
    directly.  This is the fix for Audit Bug 4 (singleton isolation).

    Usage in a router:
        from engine.api.shared import get_broker

        @router.get("/my-endpoint")
        async def my_endpoint():
            broker = get_broker()
            summary = broker.get_portfolio_summary()
            return summary
    """
    return get_engine().broker


# ---------------------------------------------------------------------------
# L7UnifiedExecutionSurface singleton
# ---------------------------------------------------------------------------
def get_l7() -> Optional["_L7"]:
    """Return the shared L7UnifiedExecutionSurface instance (lazy, thread-safe).

    Returns None if L7 failed to initialise (non-fatal — L7 is optional).

    AI NOTE — What is L7?
    ──────────────────────
    L7 is the "fused continuous execution arm" that unifies:
        - WonderTrader (micro-price + CTA signals)
        - ExchangeCore (order matching ring buffer)
        - AlpacaBroker / PaperBroker (bookkeeping)
        - OptionsEngine (Greeks and hedging)
        - QuantStrategyExecutor (12 technical strategies)
        - BetaCorridor (futures hedging)
        - L7RiskEngine (10-gate pre-trade check)
        - TransactionCostAnalyzer (TCA)
        - SlippageModel
        - ExecutionLearningLoop

    L7's submit_order() is a 10-step pipeline that classifies the product
    type (EQUITY / OPTION / FUTURE / ETF / CRYPTO), runs pre-trade risk,
    routes to the appropriate sub-executor, and logs TCA metrics.

    BROKER SWAP EFFECT:
        L7 internally creates its own broker instance (separate from
        ExecutionEngine's broker) via the same broker_type convention.
        This is a known architectural issue — ideally L7 should receive
        the broker from the shared singleton rather than creating its own.
        Until that refactor is done, L7's broker and ExecutionEngine's broker
        are separate instances pointing to the same IBKR account.

    BUG NOTE (from audit):
        /execution/l7/status returns {} because the router checks for
        l7.get_status() and l7.status() — neither exists.  The correct
        method is l7.get_execution_summary().  Fix: update execution.py
        to call l7.get_execution_summary() directly.
    """
    global _l7
    if _l7 is not None:
        return _l7

    with _init_lock:
        if _l7 is not None:
            return _l7

        logger.info("Shared singleton: initialising L7UnifiedExecutionSurface")
        try:
            from engine.execution.l7_unified_execution_surface import (
                L7UnifiedExecutionSurface,
            )
            _l7 = L7UnifiedExecutionSurface(broker_type=_broker_type())
            logger.info("Shared singleton: L7UnifiedExecutionSurface ready")
        except Exception as exc:
            logger.warning(
                "Shared singleton: L7UnifiedExecutionSurface init failed (%s) — "
                "L7 features will be unavailable.",
                exc,
            )
            _l7 = None  # type: ignore[assignment]  # callers must handle None

    return _l7


# ---------------------------------------------------------------------------
# BetaCorridor singleton
# ---------------------------------------------------------------------------
def get_beta() -> "_BetaCorridor":
    """Return the shared BetaCorridor instance (lazy, thread-safe).

    AI NOTE — What is BetaCorridor?
    ────────────────────────────────
    BetaCorridor manages the portfolio's beta relative to SPY (or a
    configurable benchmark).  It:
        - Computes current portfolio beta from position weights and
          historical correlations (via OpenBB data)
        - Maintains a target beta corridor (e.g. 0.6 – 1.1)
        - Triggers BetaAction (REDUCE / INCREASE / HOLD) when beta drifts
          outside the corridor
        - Drives beta rebalance orders in ExecutionEngine.run() via
          self.broker.place_order(ticker="SPY", ...) — see Audit Bug 5
          (SPY is hardcoded; no config for alternate benchmark)

    Risk endpoints that use BetaCorridor:
        /risk/portfolio  — get_corridor_analytics()
        /risk/beta/stress — stress_test_beta()
        /risk/beta/history — get_beta_history_df()

    BROKER SWAP EFFECT:
        BetaCorridor is broker-agnostic — it reads position weights from
        the PortfolioState it receives and fetches SPY prices via OpenBB.
        Swapping the broker does not affect BetaCorridor behaviour unless
        the new broker's get_all_positions() returns a different Position
        shape (which BrokerProtocol prevents).

    NOTE:
        The shared ExecutionEngine already creates its own BetaCorridor as
        self.beta.  For most use cases, prefer engine.beta from get_engine()
        to avoid a second instance.  This getter exists for routers that
        need BetaCorridor without instantiating the full ExecutionEngine
        (e.g. risk.py during startup before the first pipeline run).
    """
    global _beta
    if _beta is not None:
        return _beta

    with _init_lock:
        if _beta is not None:
            return _beta

        logger.info("Shared singleton: initialising BetaCorridor")
        try:
            # Prefer the beta instance already on the engine to avoid
            # two separate BetaCorridor objects tracking diverging history.
            eng = get_engine()
            if hasattr(eng, "beta") and eng.beta is not None:
                _beta = eng.beta
                logger.info(
                    "Shared singleton: reusing BetaCorridor from ExecutionEngine"
                )
            else:
                from engine.portfolio.beta_corridor import BetaCorridor
                _beta = BetaCorridor()
                logger.info("Shared singleton: BetaCorridor created standalone")
        except Exception as exc:
            logger.warning(
                "Shared singleton: BetaCorridor init failed (%s) — "
                "falling back to standalone BetaCorridor()",
                exc,
            )
            from engine.portfolio.beta_corridor import BetaCorridor
            _beta = BetaCorridor()

    return _beta


# ---------------------------------------------------------------------------
# OptionsEngine singleton
# ---------------------------------------------------------------------------
def get_options() -> "_OptionsEngine":
    """Return the shared OptionsEngine instance (lazy, thread-safe).

    AI NOTE — What is OptionsEngine?
    ──────────────────────────────────
    OptionsEngine manages the options book and provides:
        - Options position tracking (long/short calls and puts)
        - Portfolio Greeks: aggregate delta, gamma, theta, vega, rho
        - Vol surface construction (implied vol by strike/expiry)
        - Hedge requirement computation (delta neutral hedging cost)
        - Regime-based strategy matrix (which options strategy fits
          current macro regime: bull spread, bear spread, straddle, etc.)
        - L7 integration: _execute_option() in L7 routes all OPTION
          product orders through OptionsEngine

    Risk endpoints that use OptionsEngine:
        /risk/greeks             — get_portfolio_greeks()
        /risk/options/hedge      — compute_hedge_requirements()
        /risk/options/strategies — regime_strategy_matrix()
        /risk/options-positions  — .positions attribute

    BROKER SWAP EFFECT:
        OptionsEngine is broker-agnostic at the read level (Greeks, vol
        surface, strategy matrix).  For order execution it relies on
        L7's _execute_option() which calls AlpacaBroker.place_order()
        with appropriate option contract params.  Swapping to IBKR would
        require updating L7._execute_option() to use the IBKR options API.

    NOTE:
        OptionsEngine is a heavyweight object (loads volatility surface
        data on init).  Initialise it once and reuse.  This singleton
        ensures the /risk endpoints and L7 share the same instance.
    """
    global _options
    if _options is not None:
        return _options

    with _init_lock:
        if _options is not None:
            return _options

        logger.info("Shared singleton: initialising OptionsEngine")
        try:
            from engine.execution.options_engine import OptionsEngine
            _options = OptionsEngine()
            logger.info("Shared singleton: OptionsEngine ready")
        except Exception as exc:
            logger.error(
                "Shared singleton: OptionsEngine init failed (%s). "
                "Options-related endpoints will return errors.",
                exc,
            )
            raise  # OptionsEngine failure is not silently recoverable

    return _options


# ---------------------------------------------------------------------------
# DecisionMatrix singleton
# ---------------------------------------------------------------------------
def get_decision() -> "_DecisionMatrix":
    """Return the shared DecisionMatrix instance (lazy, thread-safe)."""
    global _decision
    if _decision is not None:
        return _decision

    with _init_lock:
        if _decision is not None:
            return _decision

        logger.info("Shared singleton: initialising DecisionMatrix")
        try:
            from engine.execution.decision_matrix import DecisionMatrix
            _decision = DecisionMatrix()
            logger.info("Shared singleton: DecisionMatrix ready")
        except Exception as exc:
            logger.error(
                "Shared singleton: DecisionMatrix init failed (%s).", exc,
            )
            raise

    return _decision


# ---------------------------------------------------------------------------
# Development / testing utility
# ---------------------------------------------------------------------------
def _reset_singletons() -> None:
    """Reset all singletons (dev / test use only).

    Calling this forces all getter functions to re-initialise on the next
    call.  Useful in tests that need to switch broker types between test
    cases by manipulating METADRON_BROKER_TYPE between calls.

    WARNING: Never call this in production — the singletons are intentionally
    process-lifetime objects.  Resetting them mid-request will cause
    concurrent requests to see stale or None brokers.

    Usage in tests:
        import os
        os.environ["METADRON_BROKER_TYPE"] = "paper"
        from engine.api.shared import _reset_singletons, get_broker
        _reset_singletons()
        broker = get_broker()
        assert broker.paper is True
    """
    global _engine, _l7, _beta, _options, _decision
    with _init_lock:
        _engine = None
        _l7 = None
        _beta = None
        _options = None
        _decision = None
    logger.warning("Shared singletons reset — all getters will reinitialise on next call")
