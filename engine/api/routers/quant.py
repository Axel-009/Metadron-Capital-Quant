"""
Quant router — QUANT tab
Wraps: QuantStrategyExecutor (12 HFT strategies → L7 execution pipeline),
       AlphaOptimizer (factor model, walk-forward, portfolio optimization),
       PatternRecognitionEngine (candlestick, regime, conviction signals),
       PatternDiscoveryEngine (mirofish + newton law patterns),
       Backtester (momentum, mean-reversion, RV backtests),
       UniverseEngine / OpenBB / IBKR (live price data)

Data flow:
    OpenBB/IBKR OHLCV → PatternRecognition scan → QuantStrategyExecutor (12 HFT strategies)
        → Consensus signal → L7UnifiedExecutionSurface (Stage 6.5)
    Learning: outcomes feed back through PaulOrchestrator → GSD gradient tracking
"""
from fastapi import APIRouter, Query
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("metadron-api.quant")
router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_executor = None
_alpha = None
_pattern = None
_discovery = None
_backtester = None


def _get_executor():
    global _executor
    if _executor is None:
        from engine.execution.quant_strategy_executor import QuantStrategyExecutor
        _executor = QuantStrategyExecutor()
    return _executor


def _get_alpha():
    global _alpha
    if _alpha is None:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        _alpha = AlphaOptimizer()
    return _alpha


def _get_pattern():
    global _pattern
    if _pattern is None:
        from engine.ml.pattern_recognition import PatternRecognitionEngine
        _pattern = PatternRecognitionEngine()
    return _pattern


def _get_discovery():
    global _discovery
    if _discovery is None:
        from engine.signals.pattern_discovery_engine import PatternDiscoveryEngine
        _discovery = PatternDiscoveryEngine()
    return _discovery


def _get_backtester():
    global _backtester
    if _backtester is None:
        from engine.ml.backtester import Backtester
        _backtester = Backtester()
    return _backtester


# ---------------------------------------------------------------------------
# Helper: fetch OHLCV from OpenBB with IBKRBroker fallback
# ---------------------------------------------------------------------------
def _fetch_ohlcv(ticker: str, days: int = 120):
    """Fetch OHLCV data, OpenBB primary, IBKRBroker fallback. No static data."""
    import pandas as pd

    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days + 50)).strftime("%Y-%m-%d")

    # Try OpenBB first
    try:
        from engine.data.openbb_data import get_prices
        df = get_prices(ticker, start=start, end=end)
        if df is not None and not df.empty:
            if hasattr(df.columns, "levels"):
                flat = {}
                for field in ["Open", "High", "Low", "Close", "Volume"]:
                    if field in df.columns.get_level_values(0):
                        col = df[field].iloc[:, 0] if df[field].ndim > 1 else df[field]
                        flat[field] = col
                return pd.DataFrame(flat, index=df.index)
            else:
                renamed = df.rename(columns={c: c.capitalize() for c in df.columns})
                return renamed
    except Exception as e:
        logger.warning(f"OpenBB fetch for {ticker}: {e}")

    # IBKRBroker fallback
    try:
        from engine.data.alpaca_data import get_bars
        df = get_bars(ticker, start=start, end=end)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"IBKR fetch for {ticker}: {e}")

    return pd.DataFrame()


def _infer_sector(ticker: str) -> str:
    """Fallback sector lookup for common tickers."""
    mapping = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
        "GOOGL": "Communication Services", "META": "Communication Services",
        "NFLX": "Communication Services",
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "JPM": "Financials", "V": "Financials", "GS": "Financials",
        "XOM": "Energy", "CVX": "Energy",
        "JNJ": "Health Care", "UNH": "Health Care",
        "PG": "Consumer Staples", "KO": "Consumer Staples",
    }
    return mapping.get(ticker, "Other")


# ═══════════ ENDPOINTS ═══════════

@router.get("/universe")
async def quant_universe():
    """Tradeable ticker universe with live prices from OpenBB/IBKRBroker.

    Returns tickers with sector, current price, and daily change.
    No hardcoded price data — all from live sources.
    """
    try:
        tickers = []

        # Build universe from UniverseEngine if available
        universe_list = []
        try:
            from engine.data.universe_engine import GICS_SECTORS
            for sector, sector_tickers in GICS_SECTORS.items():
                if isinstance(sector_tickers, list):
                    for t in sector_tickers[:2]:
                        if t not in universe_list:
                            universe_list.append(t)
        except Exception:
            pass

        # Ensure core liquid names are included
        core = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
                "JPM", "V", "JNJ", "XOM", "AMD", "CRM", "NFLX", "INTC"]
        for t in core:
            if t not in universe_list:
                universe_list.append(t)

        # Fetch live prices
        import pandas as pd
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")

        for ticker in universe_list[:20]:  # Cap at 20 for speed
            entry = {
                "ticker": ticker,
                "sector": _infer_sector(ticker),
                "price": 0.0,
                "change_pct": 0.0,
                "source": "pending",
            }
            try:
                from engine.data.openbb_data import get_prices
                df = get_prices(ticker, start=start, end=end)
                if df is not None and not df.empty:
                    if hasattr(df.columns, "levels"):
                        close_col = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
                    else:
                        close_col = df.get("Close", df.get("close", df.iloc[:, 0]))
                    latest = float(close_col.iloc[-1])
                    prev = float(close_col.iloc[-2]) if len(close_col) > 1 else latest
                    entry["price"] = round(latest, 2)
                    entry["change_pct"] = round(((latest - prev) / prev) * 100, 2) if prev else 0.0
                    entry["source"] = "openbb"
            except Exception:
                # IBKRBroker fallback
                try:
                    from engine.data.alpaca_data import get_latest_price
                    p = get_latest_price(ticker)
                    if p and p > 0:
                        entry["price"] = round(p, 2)
                        entry["source"] = "ibkr"
                except Exception:
                    pass

            tickers.append(entry)

        return {"tickers": tickers, "total": len(tickers), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/universe error: {e}")
        return {"tickers": [], "error": str(e)}


@router.get("/strategies")
async def quant_strategies(ticker: str = Query("SPY"), days: int = Query(120, ge=30, le=500)):
    """Execute all 12 HFT strategies from QuantStrategyExecutor.

    This is Stage 6.5 of the L7 pipeline — runs after DecisionMatrix approval.
    Returns per-strategy results with signal, direction, confidence,
    stop/target levels, VIX regime, and weighted consensus.
    """
    try:
        import numpy as np

        ohlcv_df = _fetch_ohlcv(ticker, days)
        if ohlcv_df.empty or "Close" not in ohlcv_df.columns:
            return {"ticker": ticker, "strategies": [], "error": "No OHLCV data",
                    "timestamp": datetime.utcnow().isoformat()}

        # Get VIX level for regime gating
        vix = 20.0
        try:
            vix_df = _fetch_ohlcv("^VIX", 10)
            if not vix_df.empty and "Close" in vix_df.columns:
                vix = float(vix_df["Close"].iloc[-1])
        except Exception:
            pass

        # Execute all 12 strategies
        executor = _get_executor()
        result = executor.execute(ticker=ticker, ohlcv=ohlcv_df, vix=vix)

        # Flatten strategy results for frontend
        strategies_flat = []
        for name, s in result.get("strategies", {}).items():
            strategies_flat.append({
                "name": name.replace("_", " ").title(),
                "key": name,
                "direction": s.get("direction", 0),
                "signal": round(s.get("signal", 0.0), 4),
                "confidence": round(s.get("confidence", 0.0) * 100, 1),
                "description": s.get("description", ""),
                "stop_loss": round(s.get("stop_loss", 0.0), 2),
                "take_profit": round(s.get("take_profit", 0.0), 2),
            })

        return {
            "ticker": ticker,
            "regime": result.get("regime", "unknown"),
            "vix": round(vix, 1),
            "scale": round(result.get("scale", 1.0), 2),
            "kill_switch": result.get("kill_switch", False),
            "strategies": strategies_flat,
            "active_count": result.get("active_count", 0),
            "active_names": result.get("active_names", []),
            "consensus_direction": result.get("consensus_direction", 0),
            "consensus_signal": round(result.get("consensus_signal", 0.0), 4),
            "agreement": round(result.get("agreement", 0.0), 3),
            "size_multiplier": round(result.get("size_multiplier", 0.0), 3),
            "stop_loss": round(result.get("stop_loss", 0.0), 2),
            "take_profit": round(result.get("take_profit", 0.0), 2),
            "stop_sources": result.get("stop_sources", []),
            "target_sources": result.get("target_sources", []),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"quant/strategies error: {e}")
        return {"ticker": ticker, "strategies": [], "error": str(e)}


@router.get("/pattern-scan")
async def pattern_scan(ticker: str = Query("SPY"), days: int = Query(252, ge=60, le=500)):
    """Pattern recognition scan — candlestick patterns, regime detection, conviction signals.

    Feeds into L7 execution pipeline and learning loop.
    """
    try:
        ohlcv_df = _fetch_ohlcv(ticker, days)
        if ohlcv_df.empty:
            return {"ticker": ticker, "patterns": {}, "error": "No data",
                    "timestamp": datetime.utcnow().isoformat()}

        engine = _get_pattern()
        result = {}

        # Full scan if available
        if hasattr(engine, "scan_ticker"):
            scan = engine.scan_ticker(ohlcv_df, ticker)
            result["scan"] = scan

        # Analyze for detailed report
        if hasattr(engine, "analyze"):
            analysis = engine.analyze(ticker, period="1y")
            result["analysis"] = analysis

        # High conviction signals
        if hasattr(engine, "get_high_conviction_signals"):
            signals = engine.get_high_conviction_signals()
            result["conviction_signals"] = [
                {
                    "ticker": getattr(s, "ticker", ticker),
                    "direction": getattr(s, "direction", "").name if hasattr(getattr(s, "direction", ""), "name") else str(getattr(s, "direction", "")),
                    "confidence": round(getattr(s, "confidence", 0.0), 3),
                    "entry": round(getattr(s, "entry", 0.0), 2),
                    "stop": round(getattr(s, "stop", 0.0), 2),
                    "target": round(getattr(s, "target", 0.0), 2),
                    "reward_risk": round(getattr(s, "reward_risk", 0.0)(), 2) if callable(getattr(s, "reward_risk", None)) else round(getattr(s, "reward_risk", 0.0), 2),
                    "pattern": getattr(s, "pattern", ""),
                    "regime": getattr(s, "regime", ""),
                }
                for s in (signals[:10] if signals else [])
            ]

        # Pattern discovery integration
        try:
            disc = _get_discovery()
            if hasattr(disc, "get_actionable"):
                actionable = disc.get_actionable(min_confidence=0.5)
                result["discovery_signals"] = [
                    {
                        "ticker": getattr(s, "ticker", ""),
                        "pattern_type": getattr(s, "pattern_type", ""),
                        "direction": getattr(s, "direction", ""),
                        "confidence": round(getattr(s, "confidence", 0.0), 3),
                        "source": getattr(s, "source", ""),
                    }
                    for s in (actionable[:10] if actionable else [])
                ]
        except Exception:
            pass

        return {"ticker": ticker, "patterns": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/pattern-scan error: {e}")
        return {"ticker": ticker, "patterns": {}, "error": str(e)}


@router.get("/execution-log")
async def quant_execution_log(limit: int = Query(50, ge=1, le=200)):
    """Recent execution log from QuantStrategyExecutor — L7 pipeline history."""
    try:
        executor = _get_executor()
        log = executor.get_execution_log()
        recent = log[-limit:][::-1]
        entries = []
        for entry in recent:
            entries.append({
                "ticker": entry.get("ticker", ""),
                "regime": entry.get("regime", ""),
                "consensus_direction": entry.get("consensus_direction", 0),
                "consensus_signal": round(entry.get("consensus_signal", 0.0), 4),
                "active_count": entry.get("active_count", 0),
                "active_names": entry.get("active_names", []),
                "agreement": round(entry.get("agreement", 0.0), 3),
                "size_multiplier": round(entry.get("size_multiplier", 0.0), 3),
                "kill_switch": entry.get("kill_switch", False),
                "stop_loss": round(entry.get("stop_loss", 0.0), 2),
                "take_profit": round(entry.get("take_profit", 0.0), 2),
            })
        return {"log": entries, "total": len(log), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/execution-log error: {e}")
        return {"log": [], "error": str(e)}


@router.get("/factor-model")
async def quant_factor_model():
    """Factor decomposition from AlphaOptimizer.

    Exposes momentum, volatility, quality, and technical factors,
    plus walk-forward OOS Sharpe and retrain status.
    """
    try:
        alpha = _get_alpha()
        result = {}

        # Factor engine details
        if hasattr(alpha, "factor_engine"):
            fe = alpha.factor_engine
            if hasattr(fe, "list_factors"):
                cats = {}
                for cat in ["momentum", "volatility", "technical"]:
                    try:
                        cats[cat] = fe.list_factors(category=cat)
                    except Exception:
                        pass
                result["factors_by_category"] = cats
            if hasattr(fe, "total_factors"):
                result["total_factors"] = fe.total_factors

        # Feature importances
        if hasattr(alpha, "get_feature_importances"):
            fi = alpha.get_feature_importances()
            if fi is not None and hasattr(fi, "to_dict"):
                result["feature_importances"] = fi.head(20).to_dict()

        # OOS Sharpe
        if hasattr(alpha, "get_oos_sharpe"):
            result["oos_sharpe"] = round(alpha.get_oos_sharpe(), 3)

        # Retrain status
        if hasattr(alpha, "should_retrain"):
            result["should_retrain"] = alpha.should_retrain()

        return {"factor_model": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/factor-model error: {e}")
        return {"factor_model": {}, "error": str(e)}


@router.get("/backtest-summary")
async def quant_backtest_summary():
    """Backtest results from Backtester — momentum, mean-reversion, RV strategies."""
    try:
        bt = _get_backtester()
        results = {}

        # Check for cached/recent backtest results
        if hasattr(bt, "results") and bt.results:
            for name, res in bt.results.items():
                if hasattr(res, "summary_dict"):
                    results[name] = res.summary_dict()
                elif isinstance(res, dict):
                    results[name] = res

        return {"backtests": results, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/backtest-summary error: {e}")
        return {"backtests": {}, "error": str(e)}


@router.get("/learning-state")
async def quant_learning_state():
    """Learning state from PaulOrchestrator — how quant patterns feed back into the system.

    Shows gradient tracking, pattern evolution, and how strategy outcomes
    flow through the learning pipeline.
    """
    try:
        result = {}

        # PaulOrchestrator learning state
        try:
            from engine.agents.paul_orchestrator import PaulOrchestrator
            orch = PaulOrchestrator()
            try:
                orch.initialize()
            except Exception:
                pass
            st = orch.status()
            result["orchestrator"] = {
                "gsd_active": st.get("gsd_active", False),
                "paul_active": st.get("paul_active", False),
                "attached_agents": st.get("attached_agents", 0),
            }
        except Exception:
            pass

        # Quant executor learning feedback
        try:
            executor = _get_executor()
            log = executor.get_execution_log()
            if log:
                # Compute learning metrics from execution history
                total_runs = len(log)
                active_avg = sum(e.get("active_count", 0) for e in log) / total_runs if total_runs else 0
                agreement_avg = sum(e.get("agreement", 0) for e in log) / total_runs if total_runs else 0
                kill_count = sum(1 for e in log if e.get("kill_switch", False))
                result["execution_learning"] = {
                    "total_executions": total_runs,
                    "avg_active_strategies": round(active_avg, 1),
                    "avg_agreement": round(agreement_avg, 3),
                    "kill_switch_activations": kill_count,
                    "strategy_consistency": round(1.0 - (kill_count / total_runs), 3) if total_runs else 1.0,
                }
        except Exception:
            pass

        # Pattern recognition learning state
        try:
            engine = _get_pattern()
            if hasattr(engine, "get_audit_log"):
                audit = engine.get_audit_log()
                result["pattern_audit_entries"] = len(audit)
            if hasattr(engine, "get_high_conviction_signals"):
                signals = engine.get_high_conviction_signals()
                result["high_conviction_count"] = len(signals) if signals else 0
        except Exception:
            pass

        return {"learning": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"quant/learning-state error: {e}")
        return {"learning": {}, "error": str(e)}
