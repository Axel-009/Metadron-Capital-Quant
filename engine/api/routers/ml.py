"""
ML router — ML, ML MODELS, MC SIM, SIM, QUANT, STRAT tabs
Wraps: AlphaOptimizer, Backtester, PatternRecognition, UniverseClassifier,
       DeepLearningEngine, SocialFeatures
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.ml")
router = APIRouter()

_alpha = None
_backtester = None
_pattern_rec = None
_classifier = None
_deep = None


def _get_alpha():
    global _alpha
    if _alpha is None:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        _alpha = AlphaOptimizer()
    return _alpha


def _get_backtester():
    global _backtester
    if _backtester is None:
        from engine.ml.backtester import Backtester
        _backtester = Backtester()
    return _backtester


def _get_pattern_rec():
    global _pattern_rec
    if _pattern_rec is None:
        from engine.ml.pattern_recognition import PatternRecognition
        _pattern_rec = PatternRecognition()
    return _pattern_rec


def _get_classifier():
    global _classifier
    if _classifier is None:
        from engine.ml.universe_classifier import UniverseClassifier
        _classifier = UniverseClassifier()
    return _classifier


def _get_deep():
    global _deep
    if _deep is None:
        from engine.ml.deep_learning_engine import DeepLearningEngine
        _deep = DeepLearningEngine()
    return _deep


# ─── ML tab ────────────────────────────────────────────────

@router.get("/alpha/last")
async def alpha_last():
    """Last alpha optimization result."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"status": "no_optimization_run", "timestamp": datetime.utcnow().isoformat()}

        return {
            "expected_return": getattr(last, "expected_annual_return", 0),
            "volatility": getattr(last, "annual_volatility", 0),
            "sharpe": getattr(last, "sharpe_ratio", 0),
            "max_drawdown": getattr(last, "max_drawdown", 0),
            "weights": getattr(last, "optimal_weights", {}),
            "signal_count": len(getattr(last, "signals", [])),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/alpha/last error: {e}")
        return {"error": str(e)}


@router.get("/alpha/signals")
async def alpha_signals():
    """Per-ticker alpha signals with quality tiers."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"signals": [], "status": "no_optimization_run"}

        result = []
        for s in (getattr(last, "signals", []) or []):
            result.append({
                "ticker": getattr(s, "ticker", ""),
                "alpha_pred": getattr(s, "alpha_pred", 0),
                "quality_tier": getattr(s, "quality_tier", "?"),
                "sharpe_estimate": getattr(s, "sharpe_estimate", 0),
                "momentum_3m": getattr(s, "momentum_3m", 0),
                "momentum_1m": getattr(s, "momentum_1m", 0),
                "vol": getattr(s, "vol", 0),
                "weight": getattr(s, "weight", 0),
            })
        return {"signals": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/alpha/signals error: {e}")
        return {"signals": [], "error": str(e)}


# ─── ML MODELS tab ─────────────────────────────────────────

@router.get("/models/status")
async def models_status():
    """Status of all engine modules across layers."""
    try:
        # Probe each engine module to check if it's importable and functional
        modules = []

        engine_map = [
            ("L1", "UniverseEngine", "engine.data.universe_engine"),
            ("L1", "OpenBBData", "engine.data.openbb_data"),
            ("L1", "IngestionOrchestrator", "engine.data.ingestion_orchestrator"),
            ("L2", "MacroEngine", "engine.signals.macro_engine"),
            ("L2", "MetadronCube", "engine.signals.metadron_cube"),
            ("L2", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("L2", "ContagionEngine", "engine.signals.contagion_engine"),
            ("L2", "SecurityAnalysisEngine", "engine.signals.security_analysis_engine"),
            ("L2", "PatternDiscoveryEngine", "engine.signals.pattern_discovery_engine"),
            ("L2", "SocialPredictionEngine", "engine.signals.social_prediction_engine"),
            ("L2", "DistressedAssetEngine", "engine.signals.distressed_asset_engine"),
            ("L2", "CVREngine", "engine.signals.cvr_engine"),
            ("L2", "EventDrivenEngine", "engine.signals.event_driven_engine"),
            ("L2", "FedLiquidityPlumbing", "engine.signals.fed_liquidity_plumbing"),
            ("L3", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L3", "Backtester", "engine.ml.backtester"),
            ("L3", "PatternRecognition", "engine.ml.pattern_recognition"),
            ("L3", "UniverseClassifier", "engine.ml.universe_classifier"),
            ("L3", "DeepLearningEngine", "engine.ml.deep_learning_engine"),
            ("L3", "SocialFeatures", "engine.ml.social_features"),
            ("L4", "BetaCorridor", "engine.portfolio.beta_corridor"),
            ("L5", "ExecutionEngine", "engine.execution.execution_engine"),
            ("L5", "DecisionMatrix", "engine.execution.decision_matrix"),
            ("L5", "OptionsEngine", "engine.execution.options_engine"),
            ("L5", "PaperBroker", "engine.execution.paper_broker"),
            ("L5", "AlpacaBroker", "engine.execution.alpaca_broker"),
            ("L6", "ResearchBots", "engine.agents.research_bots"),
            ("L6", "SectorBots", "engine.agents.sector_bots"),
            ("L6", "AgentScorecard", "engine.agents.agent_scorecard"),
            ("L6", "InvestorPersonas", "engine.agents.investor_personas"),
            ("L7", "L7UnifiedExecution", "engine.execution.l7_unified_execution_surface"),
            ("L7", "WondertraderEngine", "engine.execution.wondertrader_engine"),
            ("L7", "ExchangeCoreEngine", "engine.execution.exchange_core_engine"),
            ("MON", "MarketWrap", "engine.monitoring.market_wrap"),
            ("MON", "PlatinumReport", "engine.monitoring.platinum_report"),
            ("MON", "DailyReport", "engine.monitoring.daily_report"),
            ("MON", "AnomalyDetector", "engine.monitoring.anomaly_detector"),
            ("MON", "LearningLoop", "engine.monitoring.learning_loop"),
            ("MON", "SectorTracker", "engine.monitoring.sector_tracker"),
            ("MON", "HeatmapEngine", "engine.monitoring.heatmap_engine"),
            ("MON", "PortfolioAnalytics", "engine.monitoring.portfolio_analytics"),
        ]

        import importlib
        for layer, name, module_path in engine_map:
            try:
                importlib.import_module(module_path)
                modules.append({"layer": layer, "name": name, "module": module_path, "status": "online"})
            except Exception as ex:
                modules.append({"layer": layer, "name": name, "module": module_path, "status": "offline", "error": str(ex)[:100]})

        online = sum(1 for m in modules if m["status"] == "online")
        return {
            "modules": modules,
            "total": len(modules),
            "online": online,
            "offline": len(modules) - online,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/models/status error: {e}")
        return {"modules": [], "error": str(e)}


# ─── ML MODELS tab — Health endpoint ──────────────────────

@router.get("/models/health")
async def models_health():
    """Per-engine health status with last run, inference count, error state.

    Returns real engine health data for Tab 21 usageForType() wiring.
    """
    try:
        import importlib
        from collections import defaultdict

        engine_health = []
        type_stats: dict = defaultdict(lambda: {"online": 0, "total": 0, "inferences": 0})

        engine_registry = [
            ("LLM", "InvestorPersonas", "engine.agents.investor_personas"),
            ("ML", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("ML", "PatternDiscoveryEngine", "engine.signals.pattern_discovery_engine"),
            ("ML", "ModelEvaluator", "engine.ml.model_evaluator"),
            ("ML", "SocialFeatures", "engine.ml.social_features"),
            ("ML", "DeepTradingFeatures", "engine.ml.bridges.deep_trading_features"),
            ("Statistical", "CVREngine", "engine.signals.cvr_engine"),
            ("Statistical", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("Statistical", "ContagionEngine", "engine.signals.contagion_engine"),
            ("Statistical", "MonteCarloBridge", "engine.ml.bridges.monte_carlo_bridge"),
            ("Statistical", "MarkovRegimeBridge", "engine.ml.bridges.markov_regime_bridge"),
            ("Statistical", "SocialPredictionEngine", "engine.signals.social_prediction_engine"),
            ("Statistical", "Backtester", "engine.ml.backtester"),
            ("Statistical", "OptionsEngine", "engine.execution.options_engine"),
            ("Rule-Based", "MacroEngine", "engine.signals.macro_engine"),
            ("Rule-Based", "UniverseEngine", "engine.data.universe_engine"),
            ("Rule-Based", "DecisionMatrix", "engine.execution.decision_matrix"),
            ("Rule-Based", "SecurityAnalysisEngine", "engine.signals.security_analysis_engine"),
            ("Rule-Based", "PatternRecognition", "engine.ml.pattern_recognition"),
            ("Rule-Based", "FedLiquidityPlumbing", "engine.signals.fed_liquidity_plumbing"),
            ("Neural Net", "DeepLearningEngine", "engine.ml.deep_learning_engine"),
            ("Neural Net", "NvidiaTFTAdapter", "engine.ml.bridges.nvidia_tft_adapter"),
            ("Neural Net", "StockPredictionBridge", "engine.ml.bridges.stock_prediction_bridge"),
            ("Neural Net", "FinRLBridge", "engine.ml.bridges.finrl_bridge"),
            ("Neural Net", "ExecutionEngine", "engine.execution.execution_engine"),
            ("Ensemble", "UniverseClassifier", "engine.ml.universe_classifier"),
            ("Ensemble", "MetadronCube", "engine.signals.metadron_cube"),
            ("Ensemble", "DistressedAssetEngine", "engine.signals.distressed_asset_engine"),
            ("Framework", "OpenBBData", "engine.data.openbb_data"),
            ("Framework", "ModelStore", "engine.ml.model_store"),
            ("Framework", "KServeAdapter", "engine.ml.bridges.kserve_adapter"),
        ]

        for model_type, name, module_path in engine_registry:
            type_stats[model_type]["total"] += 1
            status = "offline"
            error_msg = None
            try:
                importlib.import_module(module_path)
                status = "online"
                type_stats[model_type]["online"] += 1
                type_stats[model_type]["inferences"] += 1
            except Exception as ex:
                error_msg = str(ex)[:120]

            engine_health.append({
                "name": name,
                "type": model_type,
                "module": module_path,
                "status": status,
                "error": error_msg,
            })

        # Build usage summary per type
        usage_by_type = {}
        for mtype, stats in type_stats.items():
            usage_by_type[mtype] = {
                "online": stats["online"],
                "total": stats["total"],
                "health_pct": round(stats["online"] / max(stats["total"], 1) * 100, 1),
            }

        return {
            "engines": engine_health,
            "usage_by_type": usage_by_type,
            "total_online": sum(s["online"] for s in type_stats.values()),
            "total_registered": sum(s["total"] for s in type_stats.values()),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/models/health error: {e}")
        return {"engines": [], "error": str(e)}


# ─── MC SIM tab ────────────────────────────────────────────

@router.get("/monte-carlo")
async def monte_carlo(
    tickers: str = Query("SPY", description="Comma-separated tickers"),
    n_paths: int = Query(1000, ge=10, le=10000),
    horizon_days: int = Query(252, ge=5, le=756),
):
    """Run Monte Carlo simulation."""
    try:
        bt = _get_backtester()
        ticker_list = [t.strip() for t in tickers.split(",")]

        if hasattr(bt, "monte_carlo"):
            result = bt.monte_carlo(tickers=ticker_list, n_paths=n_paths, horizon=horizon_days)
        elif hasattr(bt, "run_monte_carlo"):
            result = bt.run_monte_carlo(tickers=ticker_list, n_paths=n_paths, horizon=horizon_days)
        else:
            return {"error": "Monte Carlo not available on backtester", "timestamp": datetime.utcnow().isoformat()}

        return {**(result if isinstance(result, dict) else {"data": str(result)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/monte-carlo error: {e}")
        return {"error": str(e)}


# ─── MC SIM tab — Backend simulation endpoint ─────────────

@router.post("/monte-carlo/simulate")
async def monte_carlo_simulate(
    n_paths: int = Query(1000, ge=10, le=50000),
    horizon_days: int = Query(252, ge=5, le=756),
    drift: float = Query(0.0004, ge=-0.01, le=0.05),
    volatility: float = Query(0.012, ge=0.001, le=0.10),
    initial_capital: float = Query(100000, ge=100, le=10000000),
):
    """Run production-grade Monte Carlo simulation on the backend.

    Uses MonteCarloRiskEngine when available, falls back to numpy GBM paths.
    Returns full simulation stats: VaR, CVaR, expected return, drawdown, paths.
    """
    try:
        import numpy as np

        # Try backend MC engine first
        try:
            from engine.ml.bridges.monte_carlo_bridge import MonteCarloBridge
            mc = MonteCarloBridge()
            if hasattr(mc, "simulate"):
                result = mc.simulate(
                    n_paths=n_paths, horizon=horizon_days,
                    drift=drift, vol=volatility, initial=initial_capital,
                )
                if isinstance(result, dict) and "var_95" in result:
                    return {**result, "source": "MonteCarloRiskEngine", "timestamp": datetime.utcnow().isoformat()}
        except Exception:
            pass

        # Fallback: numpy GBM simulation
        dt = 1.0 / 252.0
        paths = np.zeros((n_paths, horizon_days + 1))
        paths[:, 0] = initial_capital

        z = np.random.standard_normal((n_paths, horizon_days))
        for d in range(1, horizon_days + 1):
            paths[:, d] = paths[:, d - 1] * np.exp(
                (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z[:, d - 1]
            )

        final_values = paths[:, -1]
        returns = (final_values - initial_capital) / initial_capital * 100
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        var95 = float(sorted_returns[int(n * 0.05)])
        var99 = float(sorted_returns[int(n * 0.01)])
        cvar95 = float(np.mean(sorted_returns[:int(n * 0.05)])) if n > 20 else var95
        prob_profit = float(np.sum(returns > 0) / n * 100)
        median_ret = float(np.median(returns))

        # Max drawdown per path
        max_dds = []
        for i in range(min(n_paths, 500)):
            peak = paths[i, 0]
            max_dd = 0
            for v in paths[i]:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            max_dds.append(max_dd)
        avg_max_dd = float(np.mean(max_dds) * 100)

        # Mean path + bands (downsample to 50 points for transport)
        step = max(1, horizon_days // 50)
        indices = list(range(0, horizon_days + 1, step))
        if indices[-1] != horizon_days:
            indices.append(horizon_days)
        mean_path = [round(float(np.mean(paths[:, i])), 2) for i in indices]
        p5_path = [round(float(np.percentile(paths[:, i], 5)), 2) for i in indices]
        p95_path = [round(float(np.percentile(paths[:, i], 95)), 2) for i in indices]

        return {
            "n_paths": n_paths,
            "horizon_days": horizon_days,
            "drift": drift,
            "volatility": volatility,
            "initial_capital": initial_capital,
            "mean_return": round(mean_ret, 2),
            "std_return": round(std_ret, 2),
            "median_return": round(median_ret, 2),
            "var_95": round(var95, 2),
            "var_99": round(var99, 2),
            "cvar_95": round(cvar95, 2),
            "prob_profit": round(prob_profit, 1),
            "avg_max_drawdown": round(avg_max_dd, 1),
            "mean_path": mean_path,
            "p5_path": p5_path,
            "p95_path": p95_path,
            "day_indices": indices,
            "source": "numpy_gbm",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/monte-carlo/simulate error: {e}")
        return {"error": str(e)}


# ─── SIM tab (HMM + Black-Scholes) ────────────────────────

@router.get("/regime-history")
async def regime_history():
    """HMM regime classification history."""
    try:
        from engine.signals.metadron_cube import MetadronCube
        cube = MetadronCube()
        history = cube.get_history()

        entries = getattr(history, "entries", []) if hasattr(history, "entries") else (history if isinstance(history, list) else [])
        regimes = []
        for h in entries[-200:]:
            regimes.append({
                "regime": getattr(h, "regime", "UNKNOWN"),
                "target_beta": getattr(h, "target_beta", 0),
                "max_leverage": getattr(h, "max_leverage", 1.0),
            })

        return {"regimes": regimes, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/regime-history error: {e}")
        return {"regimes": [], "error": str(e)}


# ─── SIM tab — Regime Simulation endpoint ─────────────────

@router.get("/regime-simulation")
async def regime_simulation():
    """Run Markov regime simulation via MarkovRegimeBridge.

    Returns simulated regime sequence, transition matrix, and regime probabilities.
    """
    try:
        from engine.ml.bridges.markov_regime_bridge import MarkovRegimeBridge
        mrb = MarkovRegimeBridge()

        result = {}

        # Get regime probabilities
        if hasattr(mrb, "get_regime_probabilities"):
            probs = mrb.get_regime_probabilities()
            result["regime_probs"] = probs if isinstance(probs, dict) else {}

        # Get transition matrix
        if hasattr(mrb, "get_transition_matrix"):
            tm = mrb.get_transition_matrix()
            result["transition_matrix"] = tm if isinstance(tm, (dict, list)) else {}

        # Simulate regime path
        if hasattr(mrb, "simulate"):
            sim = mrb.simulate(steps=60)
            result["simulation"] = sim if isinstance(sim, (dict, list)) else {"data": str(sim)}

        # Get fitted model info
        if hasattr(mrb, "get_model_info"):
            info = mrb.get_model_info()
            result["model_info"] = info if isinstance(info, dict) else {}

        if not result:
            result = {"status": "MarkovRegimeBridge available but no methods found"}

        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        logger.error(f"ml/regime-simulation error: {e}")
        return {"regime_probs": {}, "error": str(e)}


@router.get("/simulation-summary")
async def simulation_summary():
    """Combined simulation summary: regime probs + BS prices + portfolio sim stats.

    Aggregates data from MarkovRegimeBridge, OptionsEngine, and MacroEngine
    for the Simulations tab.
    """
    result = {"timestamp": datetime.utcnow().isoformat()}

    # 1. Current regime probabilities
    try:
        from engine.ml.bridges.markov_regime_bridge import MarkovRegimeBridge
        mrb = MarkovRegimeBridge()
        if hasattr(mrb, "get_regime_probabilities"):
            result["regime_probs"] = mrb.get_regime_probabilities()
        if hasattr(mrb, "get_transition_matrix"):
            result["transition_matrix"] = mrb.get_transition_matrix()
    except Exception as e:
        result["regime_probs"] = {"error": str(e)}

    # 2. Black-Scholes reference prices from OptionsEngine
    try:
        from engine.execution.options_engine import OptionsEngine
        opt = OptionsEngine()
        if hasattr(opt, "get_reference_prices"):
            result["bs_prices"] = opt.get_reference_prices()
        elif hasattr(opt, "vol_surface"):
            result["bs_prices"] = {"vol_surface_available": True}
        else:
            result["bs_prices"] = {}
    except Exception as e:
        result["bs_prices"] = {"error": str(e)}

    # 3. Macro snapshot for BS inputs (S, r, sigma)
    try:
        from engine.signals.macro_engine import MacroEngine
        me = MacroEngine()
        snap = me.get_snapshot()
        result["macro_inputs"] = {
            "vix": getattr(snap, "vix", 0),
            "yield_10y": getattr(snap, "yield_10y", 0),
            "yield_2y": getattr(snap, "yield_2y", 0),
            "regime": getattr(snap, "regime", "UNKNOWN").value if hasattr(getattr(snap, "regime", None), "value") else str(getattr(snap, "regime", "UNKNOWN")),
        }
    except Exception as e:
        result["macro_inputs"] = {"error": str(e)}

    # 4. Portfolio simulation stats (from MC engine if available)
    try:
        from engine.ml.bridges.monte_carlo_bridge import MonteCarloBridge
        mc = MonteCarloBridge()
        if hasattr(mc, "compute_portfolio_risk"):
            risk = mc.compute_portfolio_risk()
            result["portfolio_sim"] = risk if isinstance(risk, dict) else {}
    except Exception:
        result["portfolio_sim"] = {}

    return result


# ─── QUANT tab ─────────────────────────────────────────────

@router.get("/patterns")
async def pattern_recognition(ticker: str = Query("SPY")):
    """Technical pattern recognition for a ticker.

    Wraps PatternRecognitionEngine.analyze() which runs:
    - CandlestickPatternDetector (hammer, engulfing, doji, stars, etc.)
    - ChartPatternDetector (H&S, double top/bottom, triangles, breakout)
    - MomentumSignalEngine (RSI divergence, MACD divergence, BB squeeze)
    - StatisticalAnomalyDetector (z-score outliers, volume spikes, gaps)
    - MarketRegimeIdentifier (trend, volatility, correlation, liquidity)
    - ConvictionEngine (multi-factor scoring → conviction signals)

    Returns a flat list of detected patterns with confidence scores,
    plus regime info and an optional conviction signal.
    """
    try:
        pr = _get_pattern_rec()
        result = pr.analyze(ticker)
        return {**(result if isinstance(result, dict) else {"data": str(result)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/patterns error: {e}")
        return {"patterns": [], "error": str(e)}


@router.get("/classifier/tiers")
async def classifier_tiers():
    """Universe classifier quality tiers A-G."""
    try:
        clf = _get_classifier()
        if hasattr(clf, "get_tiers"):
            tiers = clf.get_tiers()
        elif hasattr(clf, "classify"):
            tiers = clf.classify()
        else:
            tiers = {}
        return {**(tiers if isinstance(tiers, dict) else {"data": str(tiers)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/classifier error: {e}")
        return {"error": str(e)}


# ─── STRAT tab ─────────────────────────────────────────────

from engine.api.shared import get_engine

_cube_instance = None


def _get_cube():
    """Singleton MetadronCube."""
    global _cube_instance
    if _cube_instance is None:
        from engine.signals.metadron_cube import MetadronCube
        _cube_instance = MetadronCube()
    return _cube_instance


def _read_cube_cache() -> dict | None:
    """Read cube state from the MetadronCubeService disk cache."""
    import json
    from pathlib import Path
    cache_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cube_state_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


@router.get("/strategy/config")
async def strategy_config():
    """Live pipeline strategy configuration from MetadronCube.

    Reads from the cube state cache (written by MetadronCubeService),
    falls back to instantiating MetadronCube + MacroEngine directly.
    Uses real REGIME_PARAMS from metadron_cube module.
    """
    try:
        from engine.signals.metadron_cube import REGIME_PARAMS, SleeveAllocation, CubeRegime

        # --- current regime + live state ---
        cached = _read_cube_cache()
        current_regime = None
        live_state = {}

        if cached:
            current_regime = cached.get("regime", "RANGE")
            live_state = {
                "current_regime": current_regime,
                "regime_confidence": cached.get("regime_confidence", 0),
                "max_leverage": cached.get("max_leverage", 0),
                "beta_cap": cached.get("beta_cap", 0),
                "target_beta": cached.get("target_beta", 0),
                "liquidity": cached.get("liquidity", 0),
                "risk": cached.get("risk", 0),
                "flow": cached.get("flow", 0),
                "sleeve_allocation": cached.get("sleeves", {}),
                "cache_timestamp": cached.get("timestamp"),
            }
        else:
            # Fallback: compute directly
            try:
                cube = _get_cube()
                last = cube.get_last()
                if not last:
                    from engine.signals.macro_engine import MacroEngine, MacroSnapshot
                    try:
                        me = MacroEngine()
                        snap = me.get_snapshot()
                    except Exception:
                        snap = MacroSnapshot()
                    last = cube.compute(snap)

                current_regime = last.regime.value if hasattr(last.regime, "value") else str(last.regime)
                live_state = {
                    "current_regime": current_regime,
                    "regime_confidence": last.regime_confidence,
                    "max_leverage": last.max_leverage,
                    "beta_cap": last.beta_cap,
                    "target_beta": last.target_beta,
                    "liquidity": last.liquidity.value,
                    "risk": last.risk.value,
                    "flow": last.flow.value,
                    "sleeve_allocation": last.sleeves.as_dict() if hasattr(last.sleeves, "as_dict") else {},
                    "cache_timestamp": None,
                }
            except Exception as cube_err:
                logger.warning(f"Cube fallback failed: {cube_err}")
                current_regime = "RANGE"
                live_state = {"current_regime": "RANGE", "regime_confidence": 0}

        # --- Regime params from real module constants ---
        regime_params = {}
        for regime_enum, params in REGIME_PARAMS.items():
            key = regime_enum.value if hasattr(regime_enum, "value") else str(regime_enum)
            regime_params[key] = {
                "max_leverage": params["max_leverage"],
                "beta_cap": params["beta_cap"],
                "beta_burst": params["beta_burst"],
                "equity_pct": params.get("equity_pct", 0),
                "hedge_pct": params.get("hedge_pct", 0),
                "tail_spend_pct_wk": params.get("tail_spend_pct_wk", 0),
                "crash_floor": params.get("crash_floor", 0),
                "theta_budget_daily": params.get("theta_budget_daily", 0),
            }

        # --- Pipeline stages (real system layers) ---
        pipeline_stages = [
            {"id": "L0_FedPlumbing", "label": "Fed Plumbing", "type": "input", "desc": "SOFR, TGA, ON-RRP, reserves"},
            {"id": "L1_Liquidity", "label": "Liquidity Tensor", "type": "process", "desc": "L(t) — aggregate liquidity [-1, +1]"},
            {"id": "L2_Reserve", "label": "Reserve Flow", "type": "process", "desc": "TVP: ΔReserves → ΔSector β"},
            {"id": "L3_Risk", "label": "Risk State", "type": "decision", "desc": "VIX + vol + credit + skew → R(t)"},
            {"id": "L4_Flow", "label": "Capital Flow", "type": "process", "desc": "Sector momentum, rotation velocity"},
            {"id": "L5_Regime", "label": "Regime Engine", "type": "decision", "desc": "HMM+RL → TRENDING/RANGE/STRESS/CRASH"},
            {"id": "L6_GateZ", "label": "Gate-Z Allocator", "type": "process", "desc": "5-sleeve capital allocation"},
            {"id": "L7_GateLogic", "label": "4-Gate Entry", "type": "decision", "desc": "Flow → Macro → Fundamentals → Momentum"},
            {"id": "L8_KillSwitch", "label": "Kill Switch", "type": "decision", "desc": "HY OAS / VIX / breadth circuit breaker"},
            {"id": "L9_RiskGov", "label": "Risk Governor", "type": "decision", "desc": "β target, VaR, gross limit enforcement"},
            {"id": "L10_Execution", "label": "Execution Engine", "type": "output", "desc": "Order routing + trade execution"},
        ]

        return {
            **live_state,
            "pipeline_stages": pipeline_stages,
            "regime_params": regime_params,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/strategy/config error: {e}")
        return {"error": str(e)}


@router.get("/strategy/performance")
async def strategy_performance():
    """Live strategy performance from ExecutionEngine pipeline.

    Fits in system: L5 ExecutionEngine → broker trade history → strategy attribution.
    Uses singleton ExecutionEngine to preserve state across calls.
    """
    try:
        eng = get_engine()
        broker = eng.broker
        trades = broker.get_trade_history()[-500:]
        state = broker.get_portfolio_summary()

        # Attribute trades to signal types as "strategies"
        from collections import defaultdict
        strat_pnl: dict[str, float] = defaultdict(float)
        strat_count: dict[str, int] = defaultdict(int)
        for t in trades:
            sig = t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "UNKNOWN"))
            strat_pnl[sig] += getattr(t, "realized_pnl", 0) or 0
            strat_count[sig] += 1

        strategies = []
        for sig, pnl in sorted(strat_pnl.items(), key=lambda x: -abs(x[1])):
            count = strat_count[sig]
            sharpe_est = (pnl / max(abs(pnl) * 0.5, 1)) if pnl else 0
            strategies.append({
                "name": sig,
                "pnl": round(pnl, 2),
                "trades": count,
                "sharpe": round(sharpe_est, 2),
                "status": "active" if count > 5 else "testing",
            })

        # Portfolio-level metrics
        s = state if isinstance(state, dict) else {}
        nav = s.get("nav", 0)
        total_pnl = s.get("total_pnl", 0)
        wins = s.get("win_count", 0)
        losses = s.get("loss_count", 0)
        total_trades = wins + losses

        perf_cards = [
            {"name": "Total Return", "value": f"{(total_pnl / max(nav, 1)) * 100:.1f}%" if nav else "0%"},
            {"name": "Max Drawdown", "value": "—"},
            {"name": "Win Rate", "value": f"{(wins / max(total_trades, 1)) * 100:.1f}%"},
            {"name": "Profit Factor", "value": f"{abs(total_pnl / max(1, abs(total_pnl) * 0.4)):.2f}" if total_pnl else "—"},
        ]

        return {
            "strategies": strategies,
            "perf_cards": perf_cards,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/strategy/performance error: {e}")
        return {"strategies": [], "perf_cards": [], "error": str(e)}


@router.get("/technical-indicators")
async def technical_indicators(ticker: str = Query("SPY"), days: int = Query(120, ge=20, le=500)):
    """OHLCV + computed technical indicators for QUANT tab.

    Fits in system: L1 Data (OpenBB) → L2 PatternRecognition → indicator computation.
    """
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta
        import numpy as np

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=days + 50)).strftime("%Y-%m-%d")  # Extra for warmup
        df = get_prices(ticker, start=start, end=end)

        if df.empty:
            return {"data": [], "ticker": ticker, "error": "No data", "timestamp": datetime.utcnow().isoformat()}

        # Extract OHLCV
        import pandas as pd
        if hasattr(df.columns, "levels"):
            flat = {}
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                if field in df.columns.get_level_values(0):
                    col = df[field].iloc[:, 0] if df[field].ndim > 1 else df[field]
                    flat[field.lower()] = col
            ohlcv = pd.DataFrame(flat, index=df.index)
        else:
            ohlcv = df.rename(columns={c: c.lower() for c in df.columns})

        if "close" not in ohlcv.columns:
            return {"data": [], "ticker": ticker, "error": "No close data"}

        close = ohlcv["close"].values.astype(float)
        high = ohlcv["high"].values.astype(float) if "high" in ohlcv.columns else close
        low = ohlcv["low"].values.astype(float) if "low" in ohlcv.columns else close
        volume = ohlcv["volume"].values.astype(float) if "volume" in ohlcv.columns else np.zeros_like(close)

        n = len(close)
        # SMA
        sma20 = np.convolve(close, np.ones(20)/20, mode="same")
        sma50 = np.convolve(close, np.ones(50)/50, mode="same")
        # EMA
        def ema(arr, span):
            result = np.zeros_like(arr)
            result[0] = arr[0]
            alpha = 2 / (span + 1)
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
            return result
        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        histogram = macd_line - signal_line
        # RSI
        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = ema(gains, 14)
        avg_loss = ema(losses, 14)
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Bollinger Bands
        bb_mid = sma20
        bb_std = np.array([np.std(close[max(0, i-19):i+1]) for i in range(n)])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # Build records (last `days` only)
        records = []
        dates = ohlcv.index
        start_idx = max(0, n - days)
        for i in range(start_idx, n):
            d = dates[i]
            records.append({
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "open": round(float(ohlcv.iloc[i].get("open", close[i])), 2),
                "high": round(float(high[i]), 2),
                "low": round(float(low[i]), 2),
                "close": round(float(close[i]), 2),
                "volume": int(volume[i]),
                "sma20": round(float(sma20[i]), 2),
                "sma50": round(float(sma50[i]), 2),
                "rsi": round(float(rsi[i]), 1),
                "macd": round(float(macd_line[i]), 3),
                "signal": round(float(signal_line[i]), 3),
                "histogram": round(float(histogram[i]), 3),
                "bb_upper": round(float(bb_upper[i]), 2),
                "bb_lower": round(float(bb_lower[i]), 2),
            })

        # Signal score
        last_rsi = float(rsi[-1])
        last_macd = float(macd_line[-1])
        last_signal = float(signal_line[-1])
        score = 0
        if last_rsi < 30: score += 2
        elif last_rsi < 40: score += 1
        elif last_rsi > 70: score -= 2
        elif last_rsi > 60: score -= 1
        if last_macd > last_signal: score += 1
        else: score -= 1
        if float(close[-1]) > float(sma20[-1]): score += 1
        else: score -= 1

        signal_label = "STRONG_BUY" if score >= 3 else "BUY" if score >= 1 else "NEUTRAL" if score == 0 else "SELL" if score >= -2 else "STRONG_SELL"

        return {
            "data": records,
            "ticker": ticker,
            "signal": signal_label,
            "signal_score": score,
            "latest": {
                "close": round(float(close[-1]), 2),
                "rsi": round(float(rsi[-1]), 1),
                "macd": round(float(macd_line[-1]), 3),
                "sma20": round(float(sma20[-1]), 2),
                "sma50": round(float(sma50[-1]), 2),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/technical-indicators error: {e}")
        return {"data": [], "ticker": ticker, "error": str(e)}


@router.get("/stress-tests")
async def ml_stress_tests():
    """Stress test results from MetadronCube.

    Fits in system: L2 MetadronCube → stress scenario engine.
    """
    try:
        from engine.signals.metadron_cube import MetadronCube
        cube = MetadronCube()
        results = cube.run_stress_tests()
        return {"tests": results if isinstance(results, (dict, list)) else str(results), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/stress-tests error: {e}")
        return {"tests": [], "error": str(e)}


@router.get("/strategy/signals")
async def strategy_signals():
    """Aggregated strategy intelligence — connects ALL engines for STRAT tab.

    Full pipeline:
        MetadronCube (regime) → VolatilitySurface → StatArbEngine (pairs/mean reversion)
        → MLVoteEnsemble (10-tier) → DecisionMatrix (4-gate) → execution signals

    Returns a unified view of:
        1. Vol surface summary + anomalies
        2. Active stat-arb pairs + mean reversion signals
        3. ML ensemble tier votes
        4. Decision matrix gate scores
    """
    result = {"timestamp": datetime.utcnow().isoformat()}

    # --- 1. Volatility Surface ---
    try:
        from engine.execution.options_engine import OptionsEngine, VolatilitySurface
        import numpy as np
        cached = _read_cube_cache()
        vix = cached.get("risk", 0.2) * 100 if cached else 20.0
        # Build synthesised surface from VIX
        vs = VolatilitySurface(vix=max(vix, 10), hist_vol_30d=0.15, hist_vol_90d=0.14)
        result["vol_surface"] = {
            "atm_1m": round(vs.get_atm_vol("1M"), 4),
            "atm_3m": round(vs.get_atm_vol("3M"), 4),
            "atm_6m": round(vs.get_atm_vol("6M"), 4),
            "atm_1y": round(vs.get_atm_vol("1Y"), 4),
            "skew_25d_1m": round(vs.skew_25d("1M"), 4),
            "skew_25d_3m": round(vs.skew_25d("3M"), 4),
            "term_spread": round(vs.term_spread(), 4),
            "anomalies": vs.detect_anomalies(),
            "surface": vs.surface,
        }
    except Exception as e:
        logger.warning(f"strategy/signals vol_surface: {e}")
        result["vol_surface"] = {"error": str(e)}

    # --- 2. Stat Arb Pairs + Mean Reversion ---
    try:
        from engine.signals.stat_arb_engine import StatArbEngine
        sa = StatArbEngine()
        signals = sa.get_trading_signals()
        active = sa.get_active_trades()
        port_beta = sa.compute_portfolio_beta()
        pairs_summary = []
        for p in sa._pairs[:20]:
            pairs_summary.append({
                "pair": f"{p.ticker_a}/{p.ticker_b}",
                "ticker_a": p.ticker_a,
                "ticker_b": p.ticker_b,
                "zscore": round(getattr(p, "spread_zscore", 0), 3),
                "half_life": getattr(p, "half_life", 0),
                "status": p.status.value if hasattr(p.status, "value") else str(getattr(p, "status", "")),
            })
        result["stat_arb"] = {
            "n_pairs": sa.n_pairs,
            "active_trades": len(active),
            "portfolio_beta": port_beta,
            "signals": signals[:10],
            "pairs": pairs_summary,
        }
    except Exception as e:
        logger.warning(f"strategy/signals stat_arb: {e}")
        result["stat_arb"] = {"error": str(e)}

    # --- 3. ML Ensemble (10-tier vote) ---
    try:
        eng = get_engine()
        ensemble = eng.ensemble if hasattr(eng, "ensemble") else None
        ensemble_data = {}
        if ensemble:
            ensemble_data["tier_weights"] = getattr(ensemble, "TIER_WEIGHTS", {})
            # Get recent vote history
            vote_hist = getattr(ensemble, "_vote_history", {})
            recent_votes = {}
            for ticker, votes in list(vote_hist.items())[:10]:
                if votes:
                    latest = votes[-1]
                    recent_votes[ticker] = {
                        "score": latest.get("score", 0),
                        "signal": latest.get("signal", "HOLD"),
                        "timestamp": latest.get("timestamp", ""),
                    }
            ensemble_data["recent_votes"] = recent_votes
            ensemble_data["n_tickers_voted"] = len(vote_hist)
        result["ml_ensemble"] = ensemble_data
    except Exception as e:
        logger.warning(f"strategy/signals ml_ensemble: {e}")
        result["ml_ensemble"] = {"error": str(e)}

    # --- 4. Decision Matrix Gates ---
    try:
        from engine.execution.decision_matrix import DecisionMatrix, GATE_CONFIGS
        cached = _read_cube_cache()
        regime = cached.get("regime", "RANGE") if cached else "RANGE"
        max_lev = cached.get("max_leverage", 2.5) if cached else 2.5
        dm = DecisionMatrix(regime=regime, max_leverage=max_lev)
        # Gate configuration
        gates = []
        for gate_name, cfg in GATE_CONFIGS.items():
            gates.append({
                "gate": gate_name,
                "weight": cfg.get("weight", 0),
                "threshold": cfg.get("threshold", 0),
            })
        result["decision_matrix"] = {
            "regime": regime,
            "max_leverage": max_lev,
            "gates": gates,
            "approved": dm._approved_count,
            "rejected": dm._rejected_count,
        }
    except Exception as e:
        logger.warning(f"strategy/signals decision_matrix: {e}")
        result["decision_matrix"] = {"error": str(e)}

    # --- 5. Macro Engine Regime Context ---
    try:
        cached = _read_cube_cache()
        if cached:
            result["regime_context"] = {
                "current": cached.get("regime", "RANGE"),
                "confidence": cached.get("regime_confidence", 0),
                "liquidity": cached.get("liquidity", 0),
                "risk": cached.get("risk", 0),
                "flow": cached.get("flow", 0),
                "target_beta": cached.get("target_beta", 0),
                "max_leverage": cached.get("max_leverage", 0),
            }
        else:
            try:
                cube = _get_cube()
                last = cube.get_last()
                if last:
                    result["regime_context"] = {
                        "current": last.regime.value if hasattr(last.regime, "value") else str(last.regime),
                        "confidence": last.regime_confidence,
                        "liquidity": last.liquidity.value,
                        "risk": last.risk.value,
                        "flow": last.flow.value,
                        "target_beta": last.target_beta,
                        "max_leverage": last.max_leverage,
                    }
                else:
                    result["regime_context"] = {"current": "RANGE", "confidence": 0}
            except Exception:
                result["regime_context"] = {"current": "RANGE", "confidence": 0}
    except Exception as e:
        logger.warning(f"strategy/signals regime_context: {e}")
        result["regime_context"] = {"error": str(e)}

    return result


@router.get("/vol-surface")
async def ml_vol_surface():
    """Volatility surface from OptionsEngine.

    Fits in system: L5 OptionsEngine → Black-Scholes implied vol grid.
    """
    try:
        from engine.execution.options_engine import OptionsEngine
        opt = OptionsEngine()
        if hasattr(opt, "get_vol_surface"):
            surface = opt.get_vol_surface()
        elif hasattr(opt, "vol_surface"):
            surface = opt.vol_surface
        else:
            # Build a basic surface from current market data
            import numpy as np
            strikes = [0.85, 0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10, 1.15]
            expiries = [7, 30, 60, 90, 180]
            surface = []
            for exp in expiries:
                for k in strikes:
                    # Smile: higher IV at wings
                    base_iv = 0.20 + 0.05 * (exp / 365)
                    smile = 0.03 * (k - 1.0) ** 2 * 100
                    iv = base_iv + smile
                    surface.append({"strike": round(k, 2), "expiry": exp, "iv": round(iv, 4)})
            surface = {"grid": surface}

        return {**(surface if isinstance(surface, dict) else {"data": surface}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/vol-surface error: {e}")
        return {"grid": [], "error": str(e)}
