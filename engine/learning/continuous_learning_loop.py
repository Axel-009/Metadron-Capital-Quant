"""
Metadron Capital — Continuous Learning Loop Service

PM2-managed 24/7 service that continuously improves the platform.

Updated to use the parallel ensemble endpoint (POST /ensemble) with
ML context so Brain Power can synthesize model outputs with alpha signals,
regime state, pattern discoveries, and RL agent recommendations.
"""

import os, sys, json, time, signal, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("learning-loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PLATFORM_ROOT / "data"
LEARNING_STATE_PATH = DATA_PATH / "learning"
LEARNING_STATE_PATH.mkdir(parents=True, exist_ok=True)

class LearningMetrics:
    def __init__(self):
        self.signal_accuracy = {}
        self.model_performance = {}
        self.agent_rankings = {}
        self.pattern_count = 0
        self.regime_transitions = []
        self.last_retrain = None
        self.total_cycles = 0
    def to_dict(self):
        return {"signal_accuracy": self.signal_accuracy, "model_performance": self.model_performance, "agent_rankings": self.agent_rankings, "pattern_count": self.pattern_count, "regime_transitions": self.regime_transitions[-10:], "last_retrain": self.last_retrain, "total_cycles": self.total_cycles}

class ContinuousLearningLoop:
    def __init__(self):
        self.running = True
        self.metrics = LearningMetrics()
        self.cycle_count = 0
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # ML context state — populated by each task, consumed by ensemble calls
        self._last_alpha_summary = {}
        self._last_regime = {}
        self._last_patterns = {}
        self._last_agent_scores = {}

        self._init_engines()

    def _handle_signal(self, signum, frame):
        self._save_state()
        self.running = False

    def _init_engines(self):
        self.engines = {}
        self._autoresearch = None
        self._graphify = None
        sys.path.insert(0, str(PLATFORM_ROOT))
        for name, (mod, cls) in {"alpha_optimizer": ("engine.ml.alpha_optimizer", "AlphaOptimizer"), "metadron_cube": ("engine.signals.metadron_cube", "MetadronCube"), "universe_classifier": ("engine.ml.universe_classifier", "UniverseClassifier")}.items():
            try: module = __import__(mod, fromlist=[cls]); self.engines[name] = getattr(module, cls); logger.info(f"Loaded {name}")
            except: self.engines[name] = None
        # AutoresearchBridge — tracks training experiments
        try:
            from engine.research.autoresearch_bridge import AutoresearchBridge
            self._autoresearch = AutoresearchBridge()
            logger.info(f"AutoresearchBridge loaded (available={self._autoresearch.is_available()})")
        except Exception as e:
            logger.info(f"AutoresearchBridge unavailable: {e}")
        # GraphifyBridge — codebase knowledge graph
        try:
            from engine.agents.graphify_bridge import GraphifyBridge
            self._graphify = GraphifyBridge()
            logger.info(f"GraphifyBridge loaded (available={self._graphify.is_available()})")
        except Exception as e:
            logger.info(f"GraphifyBridge unavailable: {e}")

        self._last_pattern_recognition = {}
        self._last_stock_predictions = {}
        self._last_graphify = {}              # Separate context — not nested under patterns

    def _is_market_hours(self): now = datetime.now(); return 9 <= now.hour < 16 and now.weekday() < 5
    def _is_overnight(self): now = datetime.now(); return now.hour >= 20 or now.hour < 6

    def _get_ml_context(self) -> dict:
        """Build ML context dict from latest task results for ensemble calls."""
        return {
            "alpha_signals": self._last_alpha_summary,
            "regime": self._last_regime,
            "patterns": self._last_patterns,
            "agent_scores": self._last_agent_scores,
            "pattern_recognition": self._last_pattern_recognition,
            "stock_predictions": self._last_stock_predictions,
            "graphify": self._last_graphify,
        }

    def _task_signal_feedback(self):
        logger.info("Running signal accuracy feedback...")
        try:
            trade_log = DATA_PATH / "trades" / "recent.json"
            if not trade_log.exists(): return
            with open(trade_log) as f: trades = json.load(f)
            signal_hits = {}
            for t in trades:
                sig = t.get("signal_type", "unknown")
                if sig not in signal_hits: signal_hits[sig] = {"wins": 0, "losses": 0, "total_pnl": 0}
                if t.get("pnl", 0) > 0: signal_hits[sig]["wins"] += 1
                else: signal_hits[sig]["losses"] += 1
                signal_hits[sig]["total_pnl"] += t.get("pnl", 0)
            for sig, s in signal_hits.items():
                total = s["wins"] + s["losses"]
                self.metrics.signal_accuracy[sig] = {"accuracy": round(s["wins"]/total, 3) if total else 0, "total_trades": total, "total_pnl": round(s["total_pnl"], 2)}
            # Update ML context with alpha summary
            self._last_alpha_summary = self.metrics.signal_accuracy
        except Exception as e: logger.error(f"Signal feedback failed: {e}")

    def _task_model_retrain(self):
        logger.info("Running model retraining...")
        try:
            retrain_result = {}
            if self.engines.get("alpha_optimizer"):
                self.engines["alpha_optimizer"]()
                self.metrics.last_retrain = datetime.now(timezone.utc).isoformat()
                retrain_result = {
                    "status": "completed",
                    "timestamp": self.metrics.last_retrain,
                    "signal_accuracy": self.metrics.signal_accuracy,
                }

            # POST retrain results to /ensemble so Brain Power can review for anomalies
            try:
                import httpx
                bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")
                payload = {
                    "prompt": (
                        "Alpha model retrain completed. Review the following retrain results "
                        "and flag any anomalies, degraded signal accuracy, or concerning patterns:\n"
                        + json.dumps(retrain_result, indent=2, default=str)
                    ),
                    "task_type": "retrain_review",
                    "max_tokens": 1024,
                    "ml_context": self._get_ml_context(),
                }
                r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
                if r.status_code == 200:
                    review = r.json()
                    logger.info(f"Brain Power retrain review: orchestrator={review.get('orchestrator', 'unknown')}")
            except Exception as e:
                logger.warning(f"Ensemble retrain review unavailable: {e}")

        except Exception as e: logger.error(f"Model retrain failed: {e}")

    def _task_agent_evolution(self):
        logger.info("Running agent evolution...")
        try:
            path = DATA_PATH / "agents" / "scores.json"
            if not path.exists(): return
            with open(path) as f: scores = json.load(f)
            rankings = {}
            for name, s in scores.items():
                rankings[name] = round(0.4*s.get("accuracy",0) + 0.3*min(s.get("sharpe",0)/3.0, 1.0) + 0.3*s.get("hit_rate",0), 3)
            self.metrics.agent_rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
            # Update ML context with agent scores
            self._last_agent_scores = self.metrics.agent_rankings
        except Exception as e: logger.error(f"Agent evolution failed: {e}")

    def _task_pattern_integration(self):
        logger.info("Running pattern integration...")
        try:
            path = PLATFORM_ROOT / "data" / "discoveries" / "latest.json"
            if not path.exists(): return
            with open(path) as f: discoveries = json.load(f)
            self.metrics.pattern_count = len(discoveries.get("patterns", []))
            # Update ML context with pattern discoveries
            self._last_patterns = {
                "count": self.metrics.pattern_count,
                "latest": discoveries.get("patterns", [])[:5],
            }
        except Exception as e: logger.error(f"Pattern integration failed: {e}")

    def _task_regime_calibration(self):
        logger.info("Running regime calibration...")
        try:
            if self.engines.get("metadron_cube"):
                self.engines["metadron_cube"]()
            # Update ML context with regime state
            regime_path = DATA_PATH / "regime" / "current.json"
            if regime_path.exists():
                with open(regime_path) as f:
                    self._last_regime = json.load(f)
            else:
                self._last_regime = {"status": "calibration_complete", "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as e: logger.error(f"Regime calibration failed: {e}")

    def _task_pattern_recognition(self):
        """Run pattern recognition and feed results into ML context for LLM ensemble."""
        logger.info("Running pattern recognition integration...")
        try:
            from engine.ml.pattern_recognition import PatternRecognitionEngine
            pr = PatternRecognitionEngine()
            # Analyze top tickers from recent signals
            trade_log = DATA_PATH / "trades" / "recent.json"
            tickers = ["SPY", "QQQ", "IWM", "NVDA", "AAPL"]
            if trade_log.exists():
                with open(trade_log) as f:
                    trades = json.load(f)
                    tickers = list(set(t.get("ticker", "") for t in trades[:20] if t.get("ticker")))[:10] or tickers
            if hasattr(pr, "analyze"):
                results = pr.analyze(tickers)
                self._last_pattern_recognition = {"tickers_analyzed": len(tickers), "signals": len(results) if results else 0}
            elif hasattr(pr, "detect_patterns"):
                results = pr.detect_patterns(tickers)
                self._last_pattern_recognition = {"tickers_analyzed": len(tickers), "signals": len(results) if results else 0}
            logger.info(f"PatternRecognition: {len(tickers)} tickers analyzed")
        except Exception as e:
            logger.warning(f"Pattern recognition task failed: {e}")

    def _task_stock_prediction(self):
        """Run stock prediction models and feed results into ML context for LLM ensemble."""
        logger.info("Running stock prediction integration...")
        try:
            from engine.ml.bridges.stock_prediction_bridge import StockPredictionBridge
            bridge = StockPredictionBridge()
            tickers = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
            predictions = {}
            for ticker in tickers:
                try:
                    pred = bridge.predict(ticker)
                    if pred:
                        predictions[ticker] = pred
                except Exception:
                    pass
            self._last_stock_predictions = predictions
            logger.info(f"StockPrediction: {len(predictions)} predictions generated")

            # POST to ensemble for Brain Power review
            if predictions:
                try:
                    import httpx
                    bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")
                    payload = {
                        "prompt": (
                            f"Stock prediction update: {len(predictions)} tickers. "
                            "Review predictions and identify high-conviction opportunities:\n"
                            + json.dumps(predictions, indent=2, default=str)
                        ),
                        "task_type": "stock_prediction_review",
                        "max_tokens": 512,
                        "ml_context": self._get_ml_context(),
                    }
                    r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
                    if r.status_code == 200:
                        logger.info("Stock prediction ensemble review completed")
                except Exception as e:
                    logger.warning(f"Stock prediction ensemble review unavailable: {e}")
        except Exception as e:
            logger.warning(f"Stock prediction task failed: {e}")

    def _task_autoresearch_check(self):
        """Check autoresearch experiment status and feed discoveries into ML context."""
        if not self._autoresearch:
            return
        logger.info("Running autoresearch status check...")
        try:
            status = self._autoresearch.get_status()
            if not status.get("available"):
                return
            results = self._autoresearch.read_results()
            if results:
                best_bpb = status.get("best_val_bpb")
                experiment_count = status.get("experiment_count", 0)
                logger.info(
                    f"Autoresearch: {experiment_count} experiments, best_val_bpb={best_bpb}"
                )
                # Feed discoveries into ML context for ensemble calls
                self._last_patterns["autoresearch"] = {
                    "experiment_count": experiment_count,
                    "best_val_bpb": best_bpb,
                    "last_experiment": status.get("last_experiment"),
                    "recent_results": results[-5:],  # last 5 experiments
                }
                # POST discoveries to ensemble for Brain Power analysis
                try:
                    import httpx
                    bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")
                    payload = {
                        "prompt": (
                            f"Autoresearch update: {experiment_count} experiments completed. "
                            f"Best val_bpb={best_bpb}. Review recent results and identify "
                            "any architecture improvements worth integrating into alpha models:\n"
                            + json.dumps(results[-3:], indent=2, default=str)
                        ),
                        "task_type": "autoresearch_review",
                        "max_tokens": 512,
                        "ml_context": self._get_ml_context(),
                    }
                    r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
                    if r.status_code == 200:
                        review = r.json()
                        # Save autoresearch review
                        ar_path = DATA_PATH / "autoresearch"
                        ar_path.mkdir(parents=True, exist_ok=True)
                        with open(ar_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
                            json.dump(review, f, indent=2)
                        logger.info(f"Autoresearch ensemble review saved")
                except Exception as e:
                    logger.warning(f"Autoresearch ensemble review unavailable: {e}")
        except Exception as e:
            logger.error(f"Autoresearch check failed: {e}")

    def _task_graphify_refresh(self):
        """Refresh graphify knowledge graph — fed by autoresearch + agent learning.

        Receives outputs from both autoresearch experiments and agent
        performance/learning. These two combined streams feed graphify
        so the knowledge graph reflects which architectural nodes are
        most active, which agents perform best, and which autoresearch
        experiments are improving model quality.

        The same autoresearch + agent data also feeds LLM review directly
        (parallel path, not through graphify).
        """
        if not self._graphify:
            return
        logger.info("Running graphify knowledge graph refresh...")
        try:
            # Kill switch: graphify bypassed — reroute context through LLM review
            if hasattr(self._graphify, "is_killed") and self._graphify.is_killed:
                logger.info("Graphify KILLED — bypassing to LLM review path (learning loop uninterrupted)")
                self._last_graphify = {
                    "available": False,
                    "kill_switch": True,
                    "cached_god_nodes": len(getattr(self._graphify, "_cached_god_nodes", [])),
                    "rerouted_to": "llm_review",
                }
                return

            if not self._graphify.is_available():
                logger.info("Graphify graph not generated — skipping")
                self._last_graphify = {"available": False}
                return

            god_nodes = self._graphify.get_god_nodes()

            # Build graphify context with agent performance overlay
            agent_performance = {}
            try:
                scores_path = DATA_PATH / "agents" / "scores.json"
                if scores_path.exists():
                    with open(scores_path) as f:
                        agent_performance = json.load(f)
            except Exception:
                pass

            # Pull autoresearch status for graphify context
            autoresearch_status = {}
            if self._autoresearch:
                try:
                    autoresearch_status = self._autoresearch.get_status()
                except Exception:
                    pass

            self._last_graphify = {
                "available": True,
                "god_nodes": [n.get("label", n.get("id", "")) for n in god_nodes[:10]],
                "god_node_count": len(god_nodes),
                "agent_performance": {
                    name: {
                        "accuracy": s.get("accuracy", 0),
                        "sharpe": s.get("sharpe", 0),
                        "rank": s.get("rank", "RECRUIT"),
                    }
                    for name, s in list(agent_performance.items())[:10]
                } if agent_performance else {},
                "autoresearch": {
                    "experiment_count": autoresearch_status.get("experiment_count", 0),
                    "best_val_bpb": autoresearch_status.get("best_val_bpb"),
                    "last_experiment": autoresearch_status.get("last_experiment"),
                } if autoresearch_status.get("available") else {},
                "learning_metrics": {
                    "total_cycles": self.metrics.total_cycles,
                    "signal_accuracy_count": len(self.metrics.signal_accuracy),
                    "pattern_count": self.metrics.pattern_count,
                },
            }
            logger.info(
                f"Graphify: {len(god_nodes)} god nodes + "
                f"{len(agent_performance)} agent scores + "
                f"autoresearch({autoresearch_status.get('experiment_count', 0)} experiments) "
                f"loaded into context"
            )
        except Exception as e:
            logger.error(f"Graphify refresh failed: {e}")

    def _task_llm_market_review(self):
        """Run LLM market review via the parallel ensemble endpoint.

        Posts to /ensemble with full ML context so Brain Power can
        synthesize model outputs with alpha signals, regime state,
        pattern discoveries, and RL agent recommendations.
        """
        logger.info("Running LLM market review via ensemble...")
        try:
            import httpx
            bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")

            narrative_prompt = "Review today's market action."

            payload = {
                "prompt": narrative_prompt,
                "task_type": "narrative",
                "max_tokens": 1024,
                "system_prompt": (
                    "You are the chief strategist at Metadron Capital. Generate a market narrative "
                    "covering: regime_assessment, macro_outlook, sector_rotation, key_risks, "
                    "opportunities, and portfolio_positioning. Be concise and actionable. "
                    "Integrate the ML context provided (alpha signals, regime, patterns, agent scores) "
                    "into your analysis."
                ),
                "ml_context": self._get_ml_context(),
            }

            r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
            if r.status_code == 200:
                narrative_path = DATA_PATH / "narratives"
                narrative_path.mkdir(parents=True, exist_ok=True)
                result = r.json()
                # Save with ensemble metadata
                result["ml_context_snapshot"] = self._get_ml_context()
                with open(narrative_path / f"narrative_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Narrative saved (orchestrator={result.get('orchestrator', 'unknown')})")
        except Exception as e: logger.warning(f"LLM market review unavailable: {e}")

    def _save_state(self):
        with open(LEARNING_STATE_PATH / "learning_state.json", "w") as f:
            json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "cycle_count": self.cycle_count, "metrics": self.metrics.to_dict(), "ml_context": self._get_ml_context()}, f, indent=2)

    def _load_state(self):
        path = LEARNING_STATE_PATH / "learning_state.json"
        if path.exists():
            try:
                with open(path) as f:
                    state = json.load(f)
                    self.cycle_count = state.get("cycle_count", 0)
                    # Restore ML context if available
                    ml_ctx = state.get("ml_context", {})
                    self._last_alpha_summary = ml_ctx.get("alpha_signals", {})
                    self._last_regime = ml_ctx.get("regime", {})
                    self._last_patterns = ml_ctx.get("patterns", {})
                    self._last_agent_scores = ml_ctx.get("agent_scores", {})
            except: pass

    def run(self):
        logger.info("Continuous Learning Loop starting...")
        self._load_state()
        while self.running:
            self.cycle_count += 1
            self.metrics.total_cycles = self.cycle_count
            logger.info(f"Learning Cycle {self.cycle_count}")
            if self._is_market_hours():
                # ── Stream 1: Core signals + ML ──
                self._task_signal_feedback()
                self._task_pattern_recognition()
                self._task_stock_prediction()

                # ── Stream 2: Autoresearch + Agent learning → feeds graphify ──
                self._task_autoresearch_check()
                self._task_agent_evolution()
                self._task_graphify_refresh()

                # (LLM review runs overnight only)
                sleep_time = 300

            elif self._is_overnight():
                # ── Stream 1: Core signals + ML + retraining ──
                self._task_signal_feedback()
                self._task_model_retrain()
                self._task_pattern_integration()
                self._task_regime_calibration()
                self._task_pattern_recognition()
                self._task_stock_prediction()

                # ── Stream 2: Autoresearch + Agent learning → feeds graphify ──
                self._task_autoresearch_check()
                self._task_agent_evolution()
                self._task_graphify_refresh()

                # ── Stream 3: LLM review (fed by autoresearch + agents directly, NOT through graphify) ──
                self._task_llm_market_review()

                sleep_time = 600

            else:
                # ── Stream 1: Core signals + ML ──
                self._task_signal_feedback()
                self._task_pattern_integration()
                self._task_pattern_recognition()
                self._task_stock_prediction()

                # ── Stream 2: Autoresearch + Agent learning → feeds graphify ──
                self._task_autoresearch_check()
                self._task_agent_evolution()
                self._task_graphify_refresh()

                sleep_time = 300
            self._save_state()
            if self.running:
                for _ in range(sleep_time):
                    if not self.running: break
                    time.sleep(1)
        logger.info("Continuous Learning Loop shut down")

if __name__ == "__main__":
    ContinuousLearningLoop().run()
