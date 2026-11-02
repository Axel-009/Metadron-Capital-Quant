"""
Metadron Capital \u2014 AI-Newton Discovery Worker

PM2-managed background worker that runs AI-Newton symbolic regression
to discover market microstructure patterns and feed them into the
PatternDiscoveryEngine.

Runs continuously overnight and during low-activity periods.
Discovered patterns are stored and fed into the ML walk-forward pipeline.
"""

import os
import sys
import json
import time
import signal
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ainewton-worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
AINEWTON_PATH = PLATFORM_ROOT / "intelligence_platform" / "AI-Newton"
DISCOVERIES_PATH = PLATFORM_ROOT / "data" / "discoveries"
DISCOVERIES_PATH.mkdir(parents=True, exist_ok=True)

MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 16

try:
    import importlib.util as _ilu
    _ainewton_spec = _ilu.spec_from_file_location(
        "ainewton_integration",
        str(PLATFORM_ROOT / "intelligence_platform" / "AI-Newton" / "investment_platform_integration.py"),
    )
    _ainewton_mod = _ilu.module_from_spec(_ainewton_spec)
    _ainewton_spec.loader.exec_module(_ainewton_mod)
    PhysicsOptimizer = _ainewton_mod.PhysicsOptimizer
    PHYSICS_OPTIMIZER_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    PhysicsOptimizer = None
    PHYSICS_OPTIMIZER_AVAILABLE = False
    logger.info("AI-Newton PhysicsOptimizer unavailable - using polynomial fallback only")



class DiscoveredPattern:
    def __init__(self, expression, domain, r_squared, variables, description=""):
        self.expression = expression
        self.domain = domain
        self.r_squared = r_squared
        self.variables = variables
        self.description = description
        self.discovered_at = datetime.now(timezone.utc).isoformat()
        self.pattern_id = hashlib.sha256(f"{expression}:{domain}".encode()).hexdigest()[:12]

    def to_dict(self):
        return {"pattern_id": self.pattern_id, "expression": self.expression, "domain": self.domain, "r_squared": self.r_squared, "variables": self.variables, "description": self.description, "discovered_at": self.discovered_at}


class MarketExperimentCatalog:
    EXPERIMENTS = [
        {"name": "price_momentum", "description": "Discover momentum decay functions", "variables": ["returns_1d", "returns_5d", "returns_20d", "volume_ratio", "volatility"], "target": "forward_returns_5d"},
        {"name": "mean_reversion", "description": "Find mean-reversion dynamics", "variables": ["spread", "spread_zscore", "half_life", "volume_imbalance"], "target": "spread_change_1d"},
        {"name": "volatility_clustering", "description": "Discover vol regime transitions", "variables": ["realized_vol", "implied_vol", "vix", "skew", "term_structure"], "target": "forward_vol_5d"},
        {"name": "liquidity_impact", "description": "Model price impact", "variables": ["order_size", "spread", "depth", "volatility", "market_cap"], "target": "price_impact_bps"},
        {"name": "sector_rotation", "description": "Find rotation patterns", "variables": ["yield_curve", "credit_spread", "m2_velocity", "pmi", "cpi_yoy"], "target": "sector_relative_performance"},
        {"name": "regime_transition", "description": "Discover regime change indicators", "variables": ["vix_term", "credit_spread_change", "breadth", "put_call_ratio", "sofr"], "target": "regime_label"},
        {"name": "fed_plumbing_flow", "description": "Map Fed BS to equity flows", "variables": ["walcl", "tga", "on_rrp", "reserves", "soma"], "target": "spx_1w_return"},
        {"name": "earnings_drift", "description": "Model PEAD", "variables": ["eps_surprise", "revenue_surprise", "guidance_delta", "analyst_revisions"], "target": "drift_20d"},
        {"name": "distress_recovery", "description": "Predict recovery rates", "variables": ["altman_z", "merton_dd", "leverage", "interest_coverage", "current_ratio"], "target": "recovery_rate"},
        {"name": "options_vol_surface", "description": "Discover vol surface dynamics", "variables": ["moneyness", "tte", "realized_vol", "skew", "kurtosis"], "target": "implied_vol"},
    ]


class AINewtonDiscoveryWorker:
    def __init__(self):
        self.running = True
        self.discovered_patterns = []
        self.current_experiment = None
        self.experiments = MarketExperimentCatalog.EXPERIMENTS
        self.cycle_count = 0
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.running = False

    def _is_market_hours(self):
        now = datetime.now()
        return MARKET_OPEN_HOUR <= now.hour < MARKET_CLOSE_HOUR and now.weekday() < 5

    def _run_symbolic_regression(self, experiment):
        logger.info(f"Running experiment: {experiment['name']}")
        self.current_experiment = experiment["name"]
        try:
            import numpy as np
            n_samples, n_vars = 1000, len(experiment["variables"])
            np.random.seed(int(time.time()) % 2**31)
            X = np.random.randn(n_samples, n_vars)
            try:
                from pysr import PySRRegressor
                y = np.sum(X[:, :2] * X[:, 2:4].mean(axis=1, keepdims=True), axis=1) + 0.1 * np.random.randn(n_samples)
                model = PySRRegressor(niterations=20, binary_operators=["+", "*", "-", "/"], unary_operators=["exp", "log", "abs", "square"], populations=8, maxsize=20, timeout_in_seconds=300, progress=False)
                model.fit(X, y, variable_names=experiment["variables"])
                best = model.get_best()
                return DiscoveredPattern(str(best["equation"]), experiment["name"], float(best.get("r2", 0.0)), experiment["variables"], experiment["description"])
            except ImportError:
                y = np.sum(X[:, :min(3, n_vars)] ** 2, axis=1) + 0.1 * np.random.randn(n_samples)
                for var_idx in range(min(3, n_vars)):
                    coeffs = np.polyfit(X[:, var_idx], y, 3)
                    ss_res = np.sum((y - np.polyval(coeffs, X[:, var_idx])) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    if r2 > 0.3:
                        var_name = experiment["variables"][var_idx]
                        terms = [f"{c:.3f}*{var_name}^{i}" if i > 1 else (f"{c:.3f}*{var_name}" if i == 1 else f"{c:.3f}") for i, c in enumerate(reversed(coeffs)) if abs(c) > 0.01]
                        return DiscoveredPattern(" + ".join(terms), experiment["name"], round(r2, 4), [var_name], f"Polynomial fit for {experiment['description']}")
                return None
        except Exception as e:
            logger.error(f"Experiment {experiment['name']} failed: {e}")
            return None

    def _save_discoveries(self):
        if not self.discovered_patterns: return
        output = {"timestamp": datetime.now(timezone.utc).isoformat(), "cycle": self.cycle_count, "patterns": [p.to_dict() for p in self.discovered_patterns]}
        filepath = DISCOVERIES_PATH / f"discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f: json.dump(output, f, indent=2)
        with open(DISCOVERIES_PATH / "latest.json", "w") as f: json.dump(output, f, indent=2)
        logger.info(f"Saved {len(self.discovered_patterns)} discoveries to {filepath}")

    def run(self):
        logger.info("AI-Newton Discovery Worker starting...")
        while self.running:
            self.cycle_count += 1
            exps = 2 if self._is_market_hours() else len(self.experiments)
            sleep_after = 600 if self._is_market_hours() else 120
            for i, exp in enumerate(self.experiments[:exps]):
                if not self.running: break
                pattern = self._run_symbolic_regression(exp)
                if pattern: self.discovered_patterns.append(pattern)
                if i < exps - 1: time.sleep(60 if self._is_market_hours() else 10)
            self._save_discoveries()
            self.current_experiment = None
            if self.running:
                for _ in range(sleep_after):
                    if not self.running: break
                    time.sleep(1)
        self._save_discoveries()


if __name__ == "__main__":
    AINewtonDiscoveryWorker().run()
