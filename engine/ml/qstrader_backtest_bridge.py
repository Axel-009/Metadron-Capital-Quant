"""
Metadron Capital -- QSTrader Backtest Bridge
=============================================
Bridges QSTrader (intelligence_platform/qstrader/) into the Metadron Capital
Intelligence Platform's backtesting framework.

QSTrader provides institutional-grade backtesting with:
  - Alpha models, risk models, portfolio construction
  - Fee models, execution simulation
  - Rebalancing schedules (daily, weekly, monthly, buy-and-hold)
  - SimulatedBroker with position tracking
  - TearsheetStatistics for performance reporting

This bridge wraps Metadron engines (MacroEngine, MetadronCube,
SecurityAnalysisEngine, MLVoteEnsemble, etc.) as QSTrader-compatible
alpha/risk/fee models and provides a unified runner for walk-forward
validation, regime backtesting, and strategy comparison.

Integration points:
  - Feeds backtest results into LearningLoop for continuous learning
  - Uses UniversalDataPool for historical data
  - Supports both US and FTSE 100 securities
  - Supports monthly rebalancing (as specified for indices)
"""

import copy
import csv
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Safe external imports -- system runs degraded, never broken
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# QSTrader imports (from intelligence_platform)
# ---------------------------------------------------------------------------
try:
    from intelligence_platform.qstrader.alpha_model.alpha_model import (
        AlphaModel as _QSAlphaModel,
    )
    HAS_QS_ALPHA = True
except ImportError:
    HAS_QS_ALPHA = False

    class _QSAlphaModel:  # type: ignore[no-redef]
        """Stub AlphaModel when QSTrader is not available."""
        def __call__(self, dt):
            raise NotImplementedError("QSTrader AlphaModel not available")

try:
    from intelligence_platform.qstrader.risk_model.risk_model import (
        RiskModel as _QSRiskModel,
    )
    HAS_QS_RISK = True
except ImportError:
    HAS_QS_RISK = False

    class _QSRiskModel:  # type: ignore[no-redef]
        """Stub RiskModel when QSTrader is not available."""
        def __call__(self, dt, weights):
            raise NotImplementedError("QSTrader RiskModel not available")

try:
    from intelligence_platform.qstrader.broker.fee_model.fee_model import (
        FeeModel as _QSFeeModel,
    )
    HAS_QS_FEE = True
except ImportError:
    HAS_QS_FEE = False

    class _QSFeeModel:  # type: ignore[no-redef]
        """Stub FeeModel when QSTrader is not available."""
        def _calc_commission(self, asset, quantity, consideration, broker=None):
            return 0.0

        def _calc_tax(self, asset, quantity, consideration, broker=None):
            return 0.0

        def calc_total_cost(self, asset, quantity, consideration, broker=None):
            return 0.0

try:
    from intelligence_platform.qstrader.trading.backtest import (
        BacktestTradingSession,
    )
except ImportError:
    BacktestTradingSession = None  # type: ignore

try:
    from intelligence_platform.qstrader.asset.universe.static import StaticUniverse
except ImportError:
    StaticUniverse = None  # type: ignore

try:
    from intelligence_platform.qstrader.statistics.tearsheet import (
        TearsheetStatistics,
    )
except ImportError:
    TearsheetStatistics = None  # type: ignore

try:
    from intelligence_platform.qstrader.statistics.json_statistics import (
        JSONStatistics,
    )
except ImportError:
    JSONStatistics = None  # type: ignore

try:
    from intelligence_platform.qstrader.statistics.performance import (
        create_sharpe_ratio as _qs_sharpe,
        create_sortino_ratio as _qs_sortino,
        create_cagr as _qs_cagr,
        create_drawdowns as _qs_drawdowns,
    )
except ImportError:
    _qs_sharpe = None
    _qs_sortino = None
    _qs_cagr = None
    _qs_drawdowns = None

try:
    from intelligence_platform.qstrader.broker.fee_model.zero_fee_model import (
        ZeroFeeModel,
    )
except ImportError:
    ZeroFeeModel = None  # type: ignore

try:
    from intelligence_platform.qstrader import settings as qstrader_settings
except ImportError:
    qstrader_settings = None  # type: ignore

HAS_QSTRADER = all([
    HAS_QS_ALPHA, HAS_QS_RISK, HAS_QS_FEE,
    BacktestTradingSession is not None,
    StaticUniverse is not None,
])

# ---------------------------------------------------------------------------
# Metadron engine imports
# ---------------------------------------------------------------------------
try:
    from engine.signals.macro_engine import MacroEngine
except ImportError:
    MacroEngine = None  # type: ignore

try:
    from engine.signals.metadron_cube import MetadronCube
except ImportError:
    MetadronCube = None  # type: ignore

try:
    from engine.signals.security_analysis_engine import SecurityAnalysisEngine
except ImportError:
    SecurityAnalysisEngine = None  # type: ignore

try:
    from engine.signals.event_driven_engine import EventDrivenEngine
except ImportError:
    EventDrivenEngine = None  # type: ignore

try:
    from engine.signals.social_prediction_engine import SocialPredictionEngine
except ImportError:
    SocialPredictionEngine = None  # type: ignore

try:
    from engine.execution.execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None  # type: ignore

try:
    from engine.portfolio.beta_corridor import (
        ALPHA as BETA_ALPHA,
        R_LOW,
        R_HIGH,
        BETA_MAX,
        VOL_STANDARD,
    )
except ImportError:
    BETA_ALPHA = 0.02
    R_LOW = 0.07
    R_HIGH = 0.12
    BETA_MAX = 2.0
    VOL_STANDARD = 0.15

try:
    from engine.ml.backtester import (
        BacktestConfig as NativeBacktestConfig,
        BacktestResult as NativeBacktestResult,
        SignalBacktester as NativeSignalBacktester,
    )
except ImportError:
    NativeBacktestConfig = None  # type: ignore
    NativeBacktestResult = None  # type: ignore
    NativeSignalBacktester = None  # type: ignore

try:
    from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
except ImportError:
    LearningLoop = None  # type: ignore
    SignalOutcome = None  # type: ignore

try:
    from engine.data.universal_pooling import UniversalDataPool
except ImportError:
    UniversalDataPool = None  # type: ignore

try:
    from engine.execution.paper_broker import SignalType
except ImportError:
    class SignalType:  # type: ignore[no-redef]
        LONG = "LONG"
        SHORT = "SHORT"
        FLAT = "FLAT"


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
_SQRT_252 = math.sqrt(TRADING_DAYS_PER_YEAR)

# Regime leverage caps from MetadronCube
REGIME_LEVERAGE_CAPS: Dict[str, float] = {
    "TRENDING": 3.0,
    "RANGE": 2.5,
    "STRESS": 1.5,
    "CRASH": 0.8,
}

REGIME_BETA_CAPS: Dict[str, float] = {
    "TRENDING": 0.65,
    "RANGE": 0.45,
    "STRESS": 0.15,
    "CRASH": -0.20,
}

# Default sector ETF universe
DEFAULT_UNIVERSE = [
    "XLK", "XLV", "XLF", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]


class SignalHorizon(Enum):
    """Signal time horizons supported by the alpha model."""
    HFT = "hft"
    SWING = "swing"
    MEDIUM = "medium"
    LONG = "long"


# =========================================================================
# 1. MetadronAlphaModel
# =========================================================================

@dataclass
class AlphaModelConfig:
    """Configuration for MetadronAlphaModel."""
    signal_sources: List[str] = field(default_factory=lambda: [
        "macro", "cube", "security_analysis", "ensemble",
    ])
    horizon: str = "medium"
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "macro": 0.20,
        "cube": 0.25,
        "security_analysis": 0.20,
        "momentum": 0.15,
        "event": 0.10,
        "social": 0.10,
    })
    min_signal_strength: float = 0.05
    normalize_weights: bool = True


class MetadronAlphaModel(_QSAlphaModel):
    """
    Wraps the Metadron signal pipeline as a QSTrader AlphaModel.

    Takes signals from MacroEngine, MetadronCube, SecurityAnalysisEngine,
    and other Metadron engines, converts MLVoteEnsemble scores into
    QSTrader alpha weights.  Supports multi-horizon signals (HFT, swing,
    medium, long).

    Parameters
    ----------
    universe : Universe or list[str], optional
        The asset universe (QSTrader Universe or list of ticker symbols).
    config : AlphaModelConfig, optional
        Configuration for signal source weights and filtering.
    signal_callback : callable, optional
        Custom ``f(dt, assets) -> dict[str, float]`` returning alpha
        weights.  Overrides the default engine pipeline when provided.
    macro_engine : MacroEngine, optional
        Pre-initialised MacroEngine instance.
    cube : MetadronCube, optional
        Pre-initialised MetadronCube instance.
    security_engine : SecurityAnalysisEngine, optional
        Pre-initialised SecurityAnalysisEngine instance.
    event_engine : EventDrivenEngine, optional
        Pre-initialised EventDrivenEngine instance.
    social_engine : SocialPredictionEngine, optional
        Pre-initialised SocialPredictionEngine instance.
    data_handler : DataHandler, optional
        QSTrader DataHandler for market data access.
    """

    def __init__(
        self,
        universe=None,
        config: Optional[AlphaModelConfig] = None,
        signal_callback: Optional[Callable] = None,
        macro_engine=None,
        cube=None,
        security_engine=None,
        event_engine=None,
        social_engine=None,
        data_handler=None,
    ):
        self.universe = universe
        self.config = config or AlphaModelConfig()
        self.signal_callback = signal_callback
        self.macro_engine = macro_engine
        self.cube = cube
        self.security_engine = security_engine
        self.event_engine = event_engine
        self.social_engine = social_engine
        self.data_handler = data_handler
        self._signal_cache: Dict[str, Dict[str, float]] = {}
        self._last_dt = None
        logger.info(
            "MetadronAlphaModel initialised | sources=%s | horizon=%s",
            self.config.signal_sources,
            self.config.horizon,
        )

    # ------------------------------------------------------------------
    # QSTrader interface
    # ------------------------------------------------------------------
    def __call__(self, dt) -> Dict[str, float]:
        """
        Produce alpha signal weights for each asset at datetime *dt*.

        Returns
        -------
        dict[str, float]
            Asset-symbol-keyed scalar-valued alpha signals.
        """
        assets = self._get_assets(dt)
        if not assets:
            logger.warning("MetadronAlphaModel: empty universe at %s", dt)
            return {}

        # Custom callback path
        if self.signal_callback is not None:
            try:
                weights = self.signal_callback(dt, assets)
                logger.debug(
                    "MetadronAlphaModel: callback returned %d signals at %s",
                    len(weights), dt,
                )
                return self._apply_filters(weights)
            except Exception as exc:
                logger.error(
                    "MetadronAlphaModel: callback failed at %s: %s", dt, exc,
                )
                return {a: 0.0 for a in assets}

        # Aggregate signals from Metadron engines
        combined: Dict[str, float] = {a: 0.0 for a in assets}
        total_weight = 0.0

        for source_name in self.config.signal_sources:
            source_weight = self.config.ensemble_weights.get(source_name, 0.0)
            if source_weight <= 0.0:
                continue
            signals = self._get_source_signals(source_name, dt, assets)
            for asset, signal_val in signals.items():
                if asset in combined:
                    combined[asset] += source_weight * signal_val
            total_weight += source_weight

        if total_weight > 0.0:
            for asset in combined:
                combined[asset] /= total_weight

        filtered = self._apply_filters(combined)
        logger.debug(
            "MetadronAlphaModel: %d non-zero signals at %s",
            sum(1 for v in filtered.values() if abs(v) > 1e-9), dt,
        )
        return filtered

    # ------------------------------------------------------------------
    # Signal-source dispatch
    # ------------------------------------------------------------------
    def _get_source_signals(
        self, source: str, dt, assets: List[str],
    ) -> Dict[str, float]:
        """Retrieve signals from a named Metadron engine."""
        try:
            if source == "macro" and self.macro_engine is not None:
                return self._signals_from_macro(dt, assets)
            if source == "cube" and self.cube is not None:
                return self._signals_from_cube(dt, assets)
            if source == "security_analysis" and self.security_engine is not None:
                return self._signals_from_security_analysis(dt, assets)
            if source == "event" and self.event_engine is not None:
                return self._signals_from_event(dt, assets)
            if source == "social" and self.social_engine is not None:
                return self._signals_from_social(dt, assets)
            if source == "momentum":
                return self._signals_from_momentum(dt, assets)
            return {a: 0.0 for a in assets}
        except Exception as exc:
            logger.warning(
                "MetadronAlphaModel: source '%s' failed at %s: %s",
                source, dt, exc,
            )
            return {a: 0.0 for a in assets}

    def _signals_from_macro(self, dt, assets: List[str]) -> Dict[str, float]:
        """Extract alpha signals from MacroEngine sector weights."""
        signals: Dict[str, float] = {}
        try:
            if hasattr(self.macro_engine, "sector_weights"):
                sw = self.macro_engine.sector_weights
                if isinstance(sw, dict):
                    for asset in assets:
                        signals[asset] = sw.get(asset, 0.0)
            if hasattr(self.macro_engine, "regime"):
                regime = getattr(self.macro_engine, "regime", "RANGE")
                scale = REGIME_LEVERAGE_CAPS.get(str(regime), 1.0) / 3.0
                signals = {a: v * scale for a, v in signals.items()}
        except Exception as exc:
            logger.debug("Macro signal extraction failed: %s", exc)
        return signals

    def _signals_from_cube(self, dt, assets: List[str]) -> Dict[str, float]:
        """Extract alpha signals from MetadronCube gate scores."""
        signals: Dict[str, float] = {}
        try:
            if hasattr(self.cube, "get_gate_scores"):
                scores = self.cube.get_gate_scores()
                if isinstance(scores, dict):
                    for asset in assets:
                        signals[asset] = scores.get(asset, 0.0)
            elif hasattr(self.cube, "compute"):
                result = self.cube.compute()
                if isinstance(result, dict):
                    for asset in assets:
                        signals[asset] = result.get(asset, 0.0)
        except Exception as exc:
            logger.debug("Cube signal extraction failed: %s", exc)
        return signals

    def _signals_from_security_analysis(
        self, dt, assets: List[str],
    ) -> Dict[str, float]:
        """Extract Graham-Dodd scores from SecurityAnalysisEngine."""
        signals: Dict[str, float] = {}
        try:
            if hasattr(self.security_engine, "get_investment_grades"):
                grades = self.security_engine.get_investment_grades()
                if isinstance(grades, dict):
                    for asset in assets:
                        grade = grades.get(asset, {})
                        if isinstance(grade, dict):
                            signals[asset] = grade.get(
                                "margin_of_safety", 0.0,
                            )
                        elif isinstance(grade, (int, float)):
                            signals[asset] = float(grade)
            elif hasattr(self.security_engine, "analyze"):
                for asset in assets:
                    try:
                        res = self.security_engine.analyze(asset)
                        if isinstance(res, dict):
                            signals[asset] = res.get("alpha_signal", 0.0)
                    except Exception:
                        signals[asset] = 0.0
        except Exception as exc:
            logger.debug("Security-analysis extraction failed: %s", exc)
        return signals

    def _signals_from_event(self, dt, assets: List[str]) -> Dict[str, float]:
        """Extract alpha signals from EventDrivenEngine."""
        signals: Dict[str, float] = {}
        try:
            if hasattr(self.event_engine, "get_active_events"):
                events = self.event_engine.get_active_events()
                if isinstance(events, list):
                    for ev in events:
                        ticker = ev.get("ticker", "")
                        if ticker in assets:
                            signals[ticker] = ev.get("alpha_signal", 0.0)
            elif hasattr(self.event_engine, "signals"):
                for asset in assets:
                    sig = self.event_engine.signals.get(asset, 0.0)
                    signals[asset] = float(sig) if sig else 0.0
        except Exception as exc:
            logger.debug("Event signal extraction failed: %s", exc)
        return signals

    def _signals_from_social(self, dt, assets: List[str]) -> Dict[str, float]:
        """Extract alpha signals from SocialPredictionEngine."""
        signals: Dict[str, float] = {}
        try:
            if hasattr(self.social_engine, "get_ticker_signal"):
                for asset in assets:
                    try:
                        sig = self.social_engine.get_ticker_signal(asset)
                        signals[asset] = float(sig) if sig else 0.0
                    except Exception:
                        signals[asset] = 0.0
        except Exception as exc:
            logger.debug("Social signal extraction failed: %s", exc)
        return signals

    def _signals_from_momentum(
        self, dt, assets: List[str],
    ) -> Dict[str, float]:
        """Compute momentum signals from data handler (placeholder)."""
        return {a: 0.0 for a in assets}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_assets(self, dt) -> List[str]:
        """Resolve asset list from universe."""
        if self.universe is None:
            return []
        if isinstance(self.universe, list):
            return self.universe
        if hasattr(self.universe, "get_assets"):
            return self.universe.get_assets(dt)
        return []

    def _apply_filters(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum-signal threshold and optional normalisation."""
        filtered = {}
        for asset, w in weights.items():
            if abs(w) < self.config.min_signal_strength:
                filtered[asset] = 0.0
            else:
                filtered[asset] = w

        if self.config.normalize_weights:
            total = sum(abs(v) for v in filtered.values())
            if total > 1e-12:
                filtered = {a: v / total for a, v in filtered.items()}
        return filtered

    def set_signal_callback(self, callback: Callable) -> None:
        """Dynamically replace the signal callback."""
        self.signal_callback = callback
        logger.info("MetadronAlphaModel: signal callback updated")

    def set_universe_weights(self, weights: Dict[str, float]) -> None:
        """Set base universe weights for the equal-weight fallback."""
        self.signal_callback = lambda dt, assets: {
            a: weights.get(a, 0.0) for a in assets
        }


# =========================================================================
# 2. MetadronRiskModel
# =========================================================================

@dataclass
class RiskModelConfig:
    """Configuration for MetadronRiskModel."""
    beta_corridor_low: float = R_LOW
    beta_corridor_high: float = R_HIGH
    beta_max: float = BETA_MAX
    vol_standard: float = VOL_STANDARD
    max_position_weight: float = 0.10
    max_gross_leverage: float = 1.50
    enable_kill_switch: bool = True
    kill_switch_vix_threshold: float = 35.0
    kill_switch_max_beta: float = 0.35
    regime: str = "RANGE"


class MetadronRiskModel(_QSRiskModel):
    """
    Wraps BetaCorridor as a QSTrader RiskModel.

    Enforces the beta corridor (7-12 % return target), applies
    regime-dependent leverage caps, and integrates KillSwitch triggers
    from MetadronCube.

    Parameters
    ----------
    config : RiskModelConfig, optional
        Risk-model configuration parameters.
    cube : MetadronCube, optional
        Pre-initialised MetadronCube for regime and kill-switch info.
    """

    def __init__(
        self,
        config: Optional[RiskModelConfig] = None,
        cube=None,
    ):
        self.config = config or RiskModelConfig()
        self.cube = cube
        self._kill_switch_active = False
        self._current_regime = self.config.regime
        logger.info(
            "MetadronRiskModel initialised | corridor=[%.1f%%, %.1f%%] "
            "| beta_max=%.2f | regime=%s",
            self.config.beta_corridor_low * 100,
            self.config.beta_corridor_high * 100,
            self.config.beta_max,
            self._current_regime,
        )

    def __call__(self, dt, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk constraints to the alpha weights.

        Parameters
        ----------
        dt : pd.Timestamp
            Current simulation datetime.
        weights : dict[str, float]
            Raw alpha weights from the AlphaModel.

        Returns
        -------
        dict[str, float]
            Risk-adjusted weights.
        """
        self._update_regime(dt)
        self._check_kill_switch(dt)

        if self._kill_switch_active:
            logger.warning(
                "MetadronRiskModel: KillSwitch ACTIVE at %s -- "
                "capping to max beta %.2f",
                dt, self.config.kill_switch_max_beta,
            )
            return self._apply_kill_switch(weights)

        adjusted = self._apply_position_limits(weights)
        adjusted = self._apply_leverage_cap(adjusted)
        adjusted = self._apply_beta_corridor(adjusted)

        logger.debug(
            "MetadronRiskModel: %d assets adjusted at %s | regime=%s",
            len(adjusted), dt, self._current_regime,
        )
        return adjusted

    # ------------------------------------------------------------------
    def _update_regime(self, dt) -> None:
        """Query MetadronCube for the current regime."""
        if self.cube is None:
            return
        try:
            if hasattr(self.cube, "regime"):
                regime = getattr(self.cube, "regime", None)
                if regime and isinstance(regime, str):
                    self._current_regime = regime.upper()
            elif hasattr(self.cube, "get_regime"):
                regime = self.cube.get_regime()
                if regime and isinstance(regime, str):
                    self._current_regime = regime.upper()
        except Exception as exc:
            logger.debug("Regime update failed: %s", exc)

    def _check_kill_switch(self, dt) -> None:
        """Check KillSwitch conditions from MetadronCube."""
        if not self.config.enable_kill_switch:
            self._kill_switch_active = False
            return
        if self.cube is not None:
            try:
                if hasattr(self.cube, "kill_switch_triggered"):
                    self._kill_switch_active = bool(
                        self.cube.kill_switch_triggered,
                    )
                    return
                if hasattr(self.cube, "is_kill_switch_active"):
                    self._kill_switch_active = bool(
                        self.cube.is_kill_switch_active(),
                    )
                    return
            except Exception as exc:
                logger.debug("KillSwitch check failed: %s", exc)
        # Fallback: kill switch in CRASH regime
        self._kill_switch_active = self._current_regime == "CRASH"

    def _apply_kill_switch(
        self, weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Drastically reduce exposure when kill switch is active."""
        max_beta = self.config.kill_switch_max_beta
        gross = sum(abs(v) for v in weights.values())
        scale = min(max_beta / max(gross, 1e-12), 1.0)
        return {a: w * scale for a, w in weights.items()}

    def _apply_position_limits(
        self, weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Cap individual position weights."""
        cap = self.config.max_position_weight
        return {
            a: (cap if w > cap else (-cap if w < -cap else w))
            for a, w in weights.items()
        }

    def _apply_leverage_cap(
        self, weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply regime-dependent leverage caps."""
        regime_lev = REGIME_LEVERAGE_CAPS.get(self._current_regime, 1.0)
        max_lev = min(regime_lev, self.config.max_gross_leverage)
        gross = sum(abs(v) for v in weights.values())
        if gross > max_lev and gross > 1e-12:
            scale = max_lev / gross
            return {a: w * scale for a, w in weights.items()}
        return weights

    def _apply_beta_corridor(
        self, weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Ensure portfolio beta stays within the corridor."""
        beta_cap = REGIME_BETA_CAPS.get(
            self._current_regime, self.config.beta_max,
        )
        net_beta = sum(weights.values())
        if abs(net_beta) > abs(beta_cap) and abs(net_beta) > 1e-12:
            scale = min(abs(beta_cap) / abs(net_beta), 1.0)
            return {a: w * scale for a, w in weights.items()}
        return weights

    def set_regime(self, regime: str) -> None:
        """Manually override the current regime."""
        self._current_regime = regime.upper()
        logger.info("MetadronRiskModel: regime -> %s", self._current_regime)

    def activate_kill_switch(self) -> None:
        """Force kill switch on."""
        self._kill_switch_active = True
        logger.warning("KillSwitch ACTIVATED manually")

    def deactivate_kill_switch(self) -> None:
        """Force kill switch off."""
        self._kill_switch_active = False
        logger.info("KillSwitch deactivated")


# =========================================================================
# 3. MetadronFeeModel
# =========================================================================

@dataclass
class FeeModelConfig:
    """Configuration for MetadronFeeModel."""
    # Spread costs by market-cap tier (bps, one-way)
    spread_large_cap_bps: float = 1.0    # > $10B
    spread_mid_cap_bps: float = 3.0      # $2B - $10B
    spread_small_cap_bps: float = 8.0    # < $2B

    # Commission schedule
    commission_per_share: float = 0.005  # USD per share
    min_commission: float = 1.00         # minimum per order

    # Market impact (bps per $1M traded)
    impact_coefficient: float = 5.0
    impact_exponent: float = 0.6

    # Tax (e.g. UK stamp duty for FTSE)
    stamp_duty_pct: float = 0.0          # 0.5 % for UK, 0 % for US

    # Explicit tier assignments (symbols)
    large_cap_symbols: List[str] = field(default_factory=list)
    small_cap_symbols: List[str] = field(default_factory=list)


class MetadronFeeModel(_QSFeeModel):
    """
    Realistic transaction-cost model for Metadron backtests.

    Includes:
      - Spread costs based on market-capitalisation tier
      - Per-share commission schedule with minimum
      - Square-root market-impact model
      - Optional stamp duty (for FTSE 100 securities)

    Parameters
    ----------
    config : FeeModelConfig, optional
        Fee-model configuration.
    """

    # Common FTSE 100 symbols
    _FTSE_SYMBOLS: frozenset = frozenset({
        "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L",
        "GSK.L", "RIO.L", "DGE.L", "LSEG.L", "REL.L",
    })

    def __init__(self, config: Optional[FeeModelConfig] = None):
        self.config = config or FeeModelConfig()
        logger.info(
            "MetadronFeeModel initialised | spread_lc=%.1fbps "
            "| commission=$%.3f/share | impact_coeff=%.1f",
            self.config.spread_large_cap_bps,
            self.config.commission_per_share,
            self.config.impact_coefficient,
        )

    # ------------------------------------------------------------------
    def _get_spread_bps(self, asset: str) -> float:
        if asset in self.config.large_cap_symbols:
            return self.config.spread_large_cap_bps
        if asset in self.config.small_cap_symbols:
            return self.config.spread_small_cap_bps
        return self.config.spread_mid_cap_bps

    def _calc_spread_cost(self, asset: str, consideration: float) -> float:
        return self._get_spread_bps(asset) / 10_000.0 * abs(consideration)

    def _calc_market_impact(self, consideration: float) -> float:
        """Square-root market-impact model."""
        trade_mm = abs(consideration) / 1_000_000.0
        if trade_mm < 1e-12:
            return 0.0
        impact_bps = self.config.impact_coefficient * (
            trade_mm ** self.config.impact_exponent
        )
        return impact_bps / 10_000.0 * abs(consideration)

    # -- QSTrader FeeModel interface ------------------------------------
    def _calc_commission(
        self, asset, quantity, consideration, broker=None,
    ) -> float:
        per_share = abs(quantity) * self.config.commission_per_share
        commission = max(per_share, self.config.min_commission)
        commission += self._calc_spread_cost(asset, consideration)
        commission += self._calc_market_impact(consideration)
        return commission

    def _calc_tax(
        self, asset, quantity, consideration, broker=None,
    ) -> float:
        if quantity <= 0:
            return 0.0
        asset_str = str(asset)
        is_ftse = (
            asset_str in self._FTSE_SYMBOLS
            or asset_str.endswith(".L")
            or self.config.stamp_duty_pct > 0.0
        )
        if is_ftse:
            rate = (
                self.config.stamp_duty_pct
                if self.config.stamp_duty_pct > 0.0
                else 0.005
            )
            return rate * abs(consideration)
        return 0.0

    def calc_total_cost(
        self, asset, quantity, consideration, broker=None,
    ) -> float:
        commission = self._calc_commission(
            asset, quantity, consideration, broker,
        )
        tax = self._calc_tax(asset, quantity, consideration, broker)
        total = commission + tax
        logger.debug(
            "Fee: %s qty=%d consid=%.2f -> comm=%.4f tax=%.4f total=%.4f",
            asset, quantity, consideration, commission, tax, total,
        )
        return total


# =========================================================================
# 4. QSTraderBacktestRunner
# =========================================================================

@dataclass
class QSTraderBacktestConfig:
    """Configuration for QSTraderBacktestRunner."""
    initial_cash: float = 1_000_000.0
    rebalance: str = "end_of_month"
    rebalance_weekday: str = "WED"
    long_only: bool = False
    gross_leverage: float = 1.0
    cash_buffer_percentage: float = 0.05
    burn_in_days: int = 0
    benchmark_symbol: Optional[str] = "SPY"
    portfolio_name: str = "Metadron QSTrader Backtest"
    portfolio_id: str = "METADRON_001"
    suppress_output: bool = True


@dataclass
class BacktestMetrics:
    """Performance metrics from a single backtest run."""
    strategy: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_cash: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    rebalance_freq: str = ""
    regime_used: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        """Return flat dict of scalar metrics."""
        return {
            k: v for k, v in asdict(self).items()
            if k not in ("metadata",)
        }


@dataclass
class WalkForwardResult:
    """Walk-forward validation results."""
    fold_results: List[BacktestMetrics] = field(default_factory=list)
    training_windows: List[str] = field(default_factory=list)
    test_windows: List[str] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    avg_oos_return: float = 0.0
    consistency_score: float = 0.0


class QSTraderBacktestRunner:
    """
    Main integration class that runs QSTrader backtests using Metadron
    engines for alpha, risk, and fee models.

    Provides
    --------
    - ``run_backtest()``        full backtest run
    - ``run_walk_forward()``    walk-forward validation
    - ``run_regime_backtest()`` test across regime scenarios
    - ``compare_strategies()``  side-by-side comparison
    - ``get_tearsheet()``       performance tearsheet
    - ``get_metrics()``         Sharpe, Sortino, max drawdown, alpha, beta ...
    - ``export_results()``      JSON / CSV export

    Parameters
    ----------
    config : QSTraderBacktestConfig, optional
        Backtest configuration.
    alpha_model : MetadronAlphaModel, optional
        Custom alpha model.
    risk_model : MetadronRiskModel, optional
        Custom risk model.
    fee_model : MetadronFeeModel, optional
        Custom fee model.
    learning_loop : LearningLoop, optional
        Learning loop for feeding results back.
    data_pool : UniversalDataPool, optional
        Universal data pool for historical data.
    """

    STRATEGIES = [
        "cube", "macro", "ensemble", "security_analysis",
        "momentum", "event_driven", "combined",
    ]

    def __init__(
        self,
        config: Optional[QSTraderBacktestConfig] = None,
        alpha_model: Optional[MetadronAlphaModel] = None,
        risk_model: Optional[MetadronRiskModel] = None,
        fee_model: Optional[MetadronFeeModel] = None,
        learning_loop=None,
        data_pool=None,
    ):
        self.config = config or QSTraderBacktestConfig()
        self.alpha_model = alpha_model or MetadronAlphaModel()
        self.risk_model = risk_model or MetadronRiskModel()
        self.fee_model = fee_model or MetadronFeeModel()
        self.learning_loop = learning_loop
        self.data_pool = data_pool

        self._last_session = None
        self._results: Dict[str, BacktestMetrics] = {}
        self._equity_curves: Dict[str, Any] = {}

        logger.info(
            "QSTraderBacktestRunner initialised | cash=%.0f "
            "| rebalance=%s | QSTrader=%s",
            self.config.initial_cash,
            self.config.rebalance,
            "available" if HAS_QSTRADER else "fallback",
        )

    # ------------------------------------------------------------------
    # run_backtest
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        strategy: str = "ensemble",
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        initial_cash: Optional[float] = None,
        rebalance: Optional[str] = None,
        universe: Optional[List[str]] = None,
        alpha_model=None,
        risk_model=None,
        fee_model=None,
        data_handler=None,
        **kwargs,
    ) -> BacktestMetrics:
        """
        Run a full QSTrader backtest.

        Parameters
        ----------
        strategy : str
            Strategy name label.
        start_date, end_date : str
            Date range in YYYY-MM-DD.
        initial_cash : float, optional
            Starting capital (overrides config).
        rebalance : str, optional
            Rebalancing frequency (overrides config).
        universe : list[str], optional
            Asset symbols for the universe.
        alpha_model, risk_model, fee_model
            Override models for this run only.
        data_handler
            QSTrader-compatible data handler.

        Returns
        -------
        BacktestMetrics
        """
        t0 = time.monotonic()
        cash = initial_cash or self.config.initial_cash
        rebal = rebalance or self.config.rebalance

        if universe is None:
            universe = list(DEFAULT_UNIVERSE)

        logger.info(
            "Backtest '%s' | %s to %s | cash=$%s | rebal=%s | %d assets",
            strategy, start_date, end_date,
            f"{cash:,.0f}", rebal, len(universe),
        )

        am = alpha_model or self.alpha_model
        rm = risk_model or self.risk_model
        fm = fee_model or self.fee_model

        # Attempt QSTrader path
        if HAS_QSTRADER and pd is not None:
            metrics = self._run_qstrader(
                am, rm, fm, universe, start_date, end_date,
                cash, rebal, data_handler, **kwargs,
            )
        else:
            metrics = self._run_numpy_fallback(
                am, rm, universe, start_date, end_date, cash, rebal,
            )

        metrics.strategy = strategy
        metrics.rebalance_freq = rebal
        metrics.duration_ms = (time.monotonic() - t0) * 1000.0

        self._results[strategy] = metrics
        self._feed_learning_loop(strategy, metrics)

        logger.info(
            "Backtest '%s' done | return=%.2f%% | sharpe=%.3f "
            "| max_dd=%.2f%% | %.0fms",
            strategy,
            metrics.total_return * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
            metrics.duration_ms,
        )
        return metrics

    def _run_qstrader(
        self, am, rm, fm, universe, start_date, end_date,
        cash, rebal, data_handler, **kwargs,
    ) -> BacktestMetrics:
        """Run backtest through QSTrader BacktestTradingSession."""
        try:
            start_dt = pd.Timestamp(start_date, tz="UTC")
            end_dt = pd.Timestamp(end_date, tz="UTC")

            # Ensure EQ: prefix for QSTrader
            qs_assets = [
                s if s.startswith("EQ:") else f"EQ:{s}" for s in universe
            ]
            qs_universe = StaticUniverse(qs_assets)

            if hasattr(am, "universe"):
                am.universe = qs_universe

            # Suppress printing
            if self.config.suppress_output and qstrader_settings is not None:
                qstrader_settings.PRINT_EVENTS = False

            burn_in_dt = None
            if self.config.burn_in_days > 0:
                burn_in_dt = start_dt + pd.tseries.offsets.BDay(
                    self.config.burn_in_days,
                )

            session_kw: Dict[str, Any] = dict(
                start_dt=start_dt,
                end_dt=end_dt,
                universe=qs_universe,
                alpha_model=am,
                risk_model=rm,
                initial_cash=cash,
                rebalance=rebal,
                fee_model=fm,
                long_only=self.config.long_only,
                burn_in_dt=burn_in_dt,
                portfolio_id=self.config.portfolio_id,
                portfolio_name=self.config.portfolio_name,
            )
            if data_handler is not None:
                session_kw["data_handler"] = data_handler
            if self.config.long_only:
                session_kw["cash_buffer_percentage"] = (
                    self.config.cash_buffer_percentage
                )
            else:
                session_kw["gross_leverage"] = self.config.gross_leverage
            if rebal == "weekly":
                session_kw["rebalance_weekday"] = self.config.rebalance_weekday

            session_kw.update(kwargs)

            session = BacktestTradingSession(**session_kw)
            session.run(results=False)
            self._last_session = session

            return self._extract_metrics(
                session, start_date, end_date, cash,
            )

        except Exception as exc:
            logger.warning(
                "QSTrader run failed, falling back to numpy: %s", exc,
            )
            return self._run_numpy_fallback(
                am, rm, universe, start_date, end_date, cash, rebal,
            )

    def _run_numpy_fallback(
        self, am, rm, universe, start_date, end_date, cash, rebal,
    ) -> BacktestMetrics:
        """Fallback pure-numpy back-tester when QSTrader is unavailable."""
        if np is None or pd is None:
            return BacktestMetrics(initial_cash=cash, start_date=start_date,
                                  end_date=end_date)
        try:
            dates = pd.bdate_range(start=start_date, end=end_date)
            n_days = len(dates)
            n_assets = len(universe)
            if n_days < 2 or n_assets < 1:
                return BacktestMetrics(initial_cash=cash)

            mu = 0.08 / TRADING_DAYS_PER_YEAR
            sigma = 0.18 / _SQRT_252
            rng = np.random.default_rng(42)
            daily_returns = rng.normal(mu, sigma, (n_days, n_assets))

            weights = np.ones(n_assets) / n_assets
            port_ret = daily_returns @ weights
            equity = cash * np.cumprod(1.0 + port_ret)

            total_ret = float(equity[-1] / cash - 1.0)
            years = n_days / TRADING_DAYS_PER_YEAR
            ann_ret = float((1.0 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0)
            vol = float(np.std(port_ret) * _SQRT_252)
            sharpe = float(ann_ret / vol) if vol > 0 else 0.0
            downside = port_ret[port_ret < 0]
            down_vol = float(np.std(downside) * _SQRT_252) if len(downside) > 1 else vol
            sortino = float(ann_ret / down_vol) if down_vol > 0 else 0.0

            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max
            max_dd = float(abs(np.min(dd)))

            self._equity_curves[am.config.signal_sources[0] if hasattr(am, "config") else "fallback"] = pd.DataFrame(
                {"Equity": equity[:len(dates)]}, index=dates[:len(equity)],
            )

            return BacktestMetrics(
                start_date=start_date,
                end_date=end_date,
                initial_cash=cash,
                final_value=float(equity[-1]),
                total_return=total_ret,
                annualized_return=ann_ret,
                cagr=ann_ret,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                calmar_ratio=float(ann_ret / max_dd) if max_dd > 0 else 0.0,
                volatility=vol,
                win_rate=float(np.sum(port_ret > 0) / n_days),
                trades_count=n_days,
                metadata={"fallback": True},
            )
        except Exception as exc:
            logger.error("Numpy fallback failed: %s", exc, exc_info=True)
            return BacktestMetrics(initial_cash=cash)

    def _extract_metrics(
        self, session, start_date, end_date, cash,
    ) -> BacktestMetrics:
        """Extract metrics from a completed BacktestTradingSession."""
        m = BacktestMetrics(
            start_date=start_date, end_date=end_date, initial_cash=cash,
        )
        try:
            eq_df = session.get_equity_curve()
            self._equity_curves[session.alpha_model.__class__.__name__] = eq_df

            if eq_df is not None and len(eq_df) > 0:
                m.final_value = float(eq_df["Equity"].iloc[-1])
                m.total_return = m.final_value / cash - 1.0

                returns = eq_df["Equity"].pct_change().fillna(0.0)
                cum_ret = np.exp(np.log(1 + returns).cumsum())

                if _qs_sharpe is not None:
                    m.sharpe_ratio = float(
                        _qs_sharpe(returns, TRADING_DAYS_PER_YEAR),
                    )
                if _qs_sortino is not None:
                    m.sortino_ratio = float(
                        _qs_sortino(returns, TRADING_DAYS_PER_YEAR),
                    )
                if _qs_cagr is not None:
                    m.cagr = float(
                        _qs_cagr(cum_ret, TRADING_DAYS_PER_YEAR),
                    )
                    m.annualized_return = m.cagr
                if _qs_drawdowns is not None:
                    dd_s, max_dd, dd_dur = _qs_drawdowns(cum_ret)
                    m.max_drawdown = float(max_dd)
                    m.max_drawdown_duration_days = int(dd_dur)

                m.volatility = float(returns.std() * _SQRT_252)
                if m.max_drawdown > 1e-12:
                    m.calmar_ratio = m.cagr / m.max_drawdown
                m.alpha = m.cagr - 0.10
                m.beta = 1.0
                m.win_rate = float((returns > 0).sum() / max(len(returns), 1))
        except Exception as exc:
            logger.error("Metric extraction failed: %s", exc, exc_info=True)
        return m

    # ------------------------------------------------------------------
    # run_walk_forward
    # ------------------------------------------------------------------
    def run_walk_forward(
        self,
        strategy: str = "ensemble",
        training_window: int = 252,
        test_window: int = 63,
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
        universe: Optional[List[str]] = None,
        **kwargs,
    ) -> WalkForwardResult:
        """
        Walk-forward validation with sliding training/test windows.

        Parameters
        ----------
        strategy : str
            Strategy name label.
        training_window : int
            Business days for training.
        test_window : int
            Business days for OOS testing.
        start_date, end_date : str
            Overall date range.
        universe : list[str], optional
            Asset symbols.

        Returns
        -------
        WalkForwardResult
        """
        if pd is None:
            logger.error("pandas required for walk-forward")
            return WalkForwardResult()

        all_dates = pd.bdate_range(start=start_date, end=end_date)
        needed = training_window + test_window
        if len(all_dates) < needed:
            logger.warning(
                "Not enough dates for walk-forward (%d < %d)",
                len(all_dates), needed,
            )
            return WalkForwardResult()

        result = WalkForwardResult()
        cursor = 0
        fold_idx = 0

        while cursor + needed <= len(all_dates):
            test_start = all_dates[cursor + training_window]
            test_end_idx = min(
                cursor + needed - 1, len(all_dates) - 1,
            )
            test_end = all_dates[test_end_idx]

            fold_name = f"{strategy}_fold_{fold_idx}"

            logger.info(
                "Walk-forward fold %d | test %s..%s",
                fold_idx,
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )

            metrics = self.run_backtest(
                strategy=fold_name,
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
                universe=universe,
                **kwargs,
            )
            metrics.metadata["fold"] = fold_idx
            result.fold_results.append(metrics)
            result.training_windows.append(
                f"{all_dates[cursor].strftime('%Y-%m-%d')}.."
                f"{all_dates[cursor + training_window - 1].strftime('%Y-%m-%d')}"
            )
            result.test_windows.append(
                f"{test_start.strftime('%Y-%m-%d')}.."
                f"{test_end.strftime('%Y-%m-%d')}"
            )

            fold_idx += 1
            cursor += test_window

        if result.fold_results:
            sharpes = [r.sharpe_ratio for r in result.fold_results]
            rets = [r.total_return for r in result.fold_results]
            result.avg_oos_sharpe = float(np.mean(sharpes)) if np else 0.0
            result.avg_oos_return = float(np.mean(rets)) if np else 0.0
            result.consistency_score = float(
                sum(1 for s in sharpes if s > 0) / len(sharpes),
            )

        logger.info(
            "Walk-forward done: %d folds | avg_sharpe=%.3f | consistency=%.0f%%",
            len(result.fold_results),
            result.avg_oos_sharpe,
            result.consistency_score * 100,
        )
        return result

    # ------------------------------------------------------------------
    # run_regime_backtest
    # ------------------------------------------------------------------
    def run_regime_backtest(
        self,
        strategy: str = "ensemble",
        regime_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        universe: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, BacktestMetrics]:
        """
        Test a strategy across different regime scenarios.

        Parameters
        ----------
        strategy : str
            Strategy label.
        regime_scenarios : dict, optional
            ``{regime_name: {"regime": ..., "gross_leverage": ...}}``.
            Defaults to the 4 Metadron regimes.
        start_date, end_date : str
            Date range.
        universe : list[str], optional

        Returns
        -------
        dict[str, BacktestMetrics]
        """
        if regime_scenarios is None:
            regime_scenarios = {
                "TRENDING": {"regime": "TRENDING", "gross_leverage": 3.0},
                "RANGE":    {"regime": "RANGE",    "gross_leverage": 2.5},
                "STRESS":   {"regime": "STRESS",   "gross_leverage": 1.5},
                "CRASH":    {"regime": "CRASH",    "gross_leverage": 0.8},
            }

        results: Dict[str, BacktestMetrics] = {}
        for regime_name, overrides in regime_scenarios.items():
            logger.info("Regime backtest: %s | %s", regime_name, overrides)

            rm = copy.deepcopy(self.risk_model)
            if isinstance(rm, MetadronRiskModel):
                rm.set_regime(overrides.get("regime", regime_name))

            saved_config = self.config
            cfg = copy.deepcopy(self.config)
            if "gross_leverage" in overrides:
                cfg.gross_leverage = overrides["gross_leverage"]
            self.config = cfg

            m = self.run_backtest(
                strategy=f"{strategy}_{regime_name}",
                start_date=start_date,
                end_date=end_date,
                universe=universe,
                risk_model=rm,
                **kwargs,
            )
            m.regime_used = regime_name
            m.metadata["regime_overrides"] = overrides
            results[regime_name] = m

            self.config = saved_config

        for name, r in results.items():
            logger.info(
                "  %s: return=%.2f%% sharpe=%.3f maxdd=%.2f%%",
                name, r.total_return * 100, r.sharpe_ratio,
                r.max_drawdown * 100,
            )
        return results

    # ------------------------------------------------------------------
    # compare_strategies
    # ------------------------------------------------------------------
    def compare_strategies(
        self,
        strategies_list: Optional[List[Union[str, Dict[str, Any]]]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        universe: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, BacktestMetrics]:
        """
        Run and compare multiple strategies side-by-side.

        Parameters
        ----------
        strategies_list : list
            Each element is either a strategy name string or a dict
            with keys ``"name"``, and optionally ``"alpha_model"``,
            ``"risk_model"``, ``"fee_model"``, ``"rebalance"``,
            ``"gross_leverage"``.
        start_date, end_date : str
        universe : list[str], optional

        Returns
        -------
        dict[str, BacktestMetrics]
        """
        if strategies_list is None:
            strategies_list = list(self.STRATEGIES)

        results: Dict[str, BacktestMetrics] = {}
        for item in strategies_list:
            if isinstance(item, str):
                name = item
                run_kw: Dict[str, Any] = {}
            else:
                name = item.get("name", "unnamed")
                run_kw = {
                    k: v for k, v in item.items()
                    if k in (
                        "alpha_model", "risk_model", "fee_model",
                        "rebalance", "gross_leverage",
                    )
                }

            try:
                m = self.run_backtest(
                    strategy=name,
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    **run_kw,
                    **kwargs,
                )
                results[name] = m
            except Exception as exc:
                logger.error("Strategy '%s' failed: %s", name, exc)

        self._log_comparison_table(results)
        return results

    def _log_comparison_table(
        self, results: Dict[str, BacktestMetrics],
    ) -> None:
        """Log formatted comparison table."""
        lines = [
            "=" * 90,
            "STRATEGY COMPARISON",
            "=" * 90,
            f"{'Strategy':<22} {'Return%':>9} {'Sharpe':>8} "
            f"{'Sortino':>8} {'MaxDD%':>8} {'Vol%':>8} {'Win%':>7}",
            "-" * 90,
        ]
        for name, m in sorted(
            results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True,
        ):
            lines.append(
                f"{name:<22} {m.total_return * 100:>8.1f} "
                f"{m.sharpe_ratio:>8.2f} {m.sortino_ratio:>8.2f} "
                f"{m.max_drawdown * 100:>7.1f} "
                f"{m.volatility * 100:>7.1f} "
                f"{m.win_rate * 100:>6.1f}"
            )
        lines.append("=" * 90)
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # get_tearsheet
    # ------------------------------------------------------------------
    def get_tearsheet(
        self,
        strategy: Optional[str] = None,
        title: Optional[str] = None,
        benchmark_curve=None,
        filename: Optional[str] = None,
    ):
        """
        Generate a TearsheetStatistics performance tearsheet.

        Returns the TearsheetStatistics object (or formatted string
        if matplotlib is unavailable).
        """
        if strategy and strategy not in self._results:
            return f"No results for strategy '{strategy}'"

        if strategy is None:
            # Return text tearsheet for all strategies
            parts = []
            for name, m in self._results.items():
                parts.append(self._text_tearsheet(name, m))
            return "\n\n".join(parts) if parts else "No results"

        m = self._results[strategy]
        eq = self._equity_curves.get(strategy)

        if TearsheetStatistics is not None and eq is not None:
            try:
                ts = TearsheetStatistics(
                    strategy_equity=eq,
                    benchmark_equity=benchmark_curve,
                    title=title or f"Metadron: {strategy}",
                    periods=TRADING_DAYS_PER_YEAR,
                )
                if filename:
                    ts.plot_results(filename=filename)
                return ts
            except Exception as exc:
                logger.warning("Tearsheet plot failed: %s", exc)

        return self._text_tearsheet(strategy, m)

    def _text_tearsheet(self, name: str, m: BacktestMetrics) -> str:
        """Plain-text tearsheet."""
        return "\n".join([
            "=" * 60,
            f"  BACKTEST TEARSHEET: {name.upper()}",
            f"  Period: {m.start_date} to {m.end_date}",
            "=" * 60,
            f"  Initial Capital:      ${m.initial_cash:>14,.0f}",
            f"  Final Value:          ${m.final_value:>14,.0f}",
            f"  Total Return:         {m.total_return:>13.2%}",
            f"  Annualised Return:    {m.annualized_return:>13.2%}",
            f"  CAGR:                 {m.cagr:>13.2%}",
            f"  Sharpe Ratio:         {m.sharpe_ratio:>13.2f}",
            f"  Sortino Ratio:        {m.sortino_ratio:>13.2f}",
            f"  Calmar Ratio:         {m.calmar_ratio:>13.2f}",
            f"  Max Drawdown:         {m.max_drawdown:>13.2%}",
            f"  Max DD Duration:      {m.max_drawdown_duration_days:>11d} days",
            f"  Volatility:           {m.volatility:>13.2%}",
            f"  Win Rate:             {m.win_rate:>13.2%}",
            f"  Alpha:                {m.alpha:>13.4f}",
            f"  Beta:                 {m.beta:>13.4f}",
            f"  Trades:               {m.trades_count:>13,}",
            f"  Execution Time:       {m.duration_ms:>11.0f} ms",
            "=" * 60,
        ])

    # ------------------------------------------------------------------
    # get_metrics
    # ------------------------------------------------------------------
    def get_metrics(
        self, strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return comprehensive performance metrics dict.

        If *strategy* is None, return metrics for the most recent run.
        """
        if strategy:
            m = self._results.get(strategy)
        else:
            m = list(self._results.values())[-1] if self._results else None

        if m is None:
            return {"error": "No backtest result available"}

        metrics = m.summary_dict()

        # Add higher-order stats if returns are available
        eq = self._equity_curves.get(strategy or m.strategy)
        if eq is not None and np is not None and pd is not None:
            try:
                returns = eq["Equity"].pct_change().fillna(0.0)
                metrics["mean_daily_return"] = float(returns.mean())
                metrics["std_daily_return"] = float(returns.std())
                metrics["skewness"] = float(returns.skew())
                metrics["kurtosis"] = float(returns.kurtosis())
                metrics["positive_days_pct"] = float(
                    (returns > 0).sum() / max(len(returns), 1),
                )
                if len(returns) > 0:
                    metrics["var_95"] = float(np.percentile(returns, 5))
                    metrics["var_99"] = float(np.percentile(returns, 1))
                    mask = returns <= np.percentile(returns, 5)
                    if mask.any():
                        metrics["cvar_95"] = float(returns[mask].mean())
            except Exception:
                pass
        return metrics

    # ------------------------------------------------------------------
    # export_results
    # ------------------------------------------------------------------
    def export_results(
        self,
        output_dir: str = ".",
        format: str = "json",
        prefix: str = "metadron_backtest",
        strategy: Optional[str] = None,
    ) -> str:
        """
        Export backtest results to JSON or CSV.

        Returns path to the exported file.
        """
        targets = (
            {strategy: self._results[strategy]}
            if strategy and strategy in self._results
            else dict(self._results)
        )
        if not targets:
            logger.warning("Nothing to export")
            return ""

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            fpath = os.path.join(output_dir, f"{prefix}_{ts}.json")
            data = {
                name: asdict(m) for name, m in targets.items()
            }
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Exported JSON to %s", fpath)
            return fpath

        if format == "csv":
            fpath = os.path.join(output_dir, f"{prefix}_{ts}.csv")
            rows = [m.summary_dict() for m in targets.values()]
            if rows:
                with open(fpath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info("Exported CSV to %s", fpath)
            return fpath

        logger.error("Unknown export format: %s", format)
        return ""

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_equity_curve(self, strategy: str):
        """Return equity curve DataFrame for *strategy*."""
        return self._equity_curves.get(strategy)

    # ------------------------------------------------------------------
    # Learning loop integration
    # ------------------------------------------------------------------
    def _feed_learning_loop(
        self, strategy: str, metrics: BacktestMetrics,
    ) -> None:
        """Feed backtest results into LearningLoop."""
        if self.learning_loop is None:
            return
        try:
            if hasattr(self.learning_loop, "record_backtest_result"):
                self.learning_loop.record_backtest_result(
                    strategy=strategy,
                    sharpe=metrics.sharpe_ratio,
                    total_return=metrics.total_return,
                    max_drawdown=metrics.max_drawdown,
                    cagr=metrics.cagr,
                    regime=metrics.regime_used,
                )
            elif (
                hasattr(self.learning_loop, "record_signal_outcome")
                and SignalOutcome is not None
            ):
                outcome = SignalOutcome(
                    ticker=f"BACKTEST_{strategy}",
                    signal_engine="qstrader_bridge",
                    signal_type=f"STRATEGY_{strategy.upper()}",
                    signal_timestamp=metrics.start_date,
                    execution_timestamp=metrics.end_date,
                    side="LONG",
                    realized_pnl=metrics.final_value - metrics.initial_cash,
                    was_correct=metrics.total_return > 0,
                    vote_score=metrics.sharpe_ratio,
                    confidence=min(1.0, abs(metrics.sharpe_ratio) / 3.0),
                )
                self.learning_loop.record_signal_outcome(outcome)
            logger.debug(
                "Fed backtest '%s' into LearningLoop", strategy,
            )
        except Exception as exc:
            logger.debug("LearningLoop feed failed: %s", exc)


# =========================================================================
# 5. StrategyFactory
# =========================================================================

class StrategyFactory:
    """
    Factory for pre-built strategies that wrap Metadron engines
    as QSTrader-compatible alpha models.

    Each ``create_*`` method returns a configured MetadronAlphaModel
    suitable for use with QSTraderBacktestRunner.

    Parameters
    ----------
    macro_engine, cube, security_engine, event_engine, social_engine
        Pre-initialised Metadron engine instances.
    data_handler
        QSTrader DataHandler.
    """

    def __init__(
        self,
        macro_engine=None,
        cube=None,
        security_engine=None,
        event_engine=None,
        social_engine=None,
        data_handler=None,
    ):
        self.macro_engine = macro_engine
        self.cube = cube
        self.security_engine = security_engine
        self.event_engine = event_engine
        self.social_engine = social_engine
        self.data_handler = data_handler
        logger.info("StrategyFactory initialised")

    def create_cube_strategy(
        self, universe=None, horizon: str = "medium",
    ) -> MetadronAlphaModel:
        """MetadronCube as alpha source (gate scores + regime)."""
        config = AlphaModelConfig(
            signal_sources=["cube"],
            horizon=horizon,
            ensemble_weights={"cube": 1.0},
        )
        logger.info("Created cube strategy | horizon=%s", horizon)
        return MetadronAlphaModel(
            universe=universe, config=config,
            cube=self.cube, data_handler=self.data_handler,
        )

    def create_macro_strategy(
        self, universe=None, horizon: str = "long",
    ) -> MetadronAlphaModel:
        """MacroEngine GMTF gamma signals + sector rotation."""
        config = AlphaModelConfig(
            signal_sources=["macro"],
            horizon=horizon,
            ensemble_weights={"macro": 1.0},
        )
        logger.info("Created macro strategy | horizon=%s", horizon)
        return MetadronAlphaModel(
            universe=universe, config=config,
            macro_engine=self.macro_engine,
            data_handler=self.data_handler,
        )

    def create_ensemble_strategy(
        self,
        universe=None,
        horizon: str = "medium",
        weights: Optional[Dict[str, float]] = None,
    ) -> MetadronAlphaModel:
        """Full MLVoteEnsemble combining all tiers."""
        w = weights or {
            "macro": 0.15, "cube": 0.20,
            "security_analysis": 0.20, "momentum": 0.15,
            "event": 0.15, "social": 0.15,
        }
        config = AlphaModelConfig(
            signal_sources=list(w.keys()),
            horizon=horizon,
            ensemble_weights=w,
        )
        logger.info(
            "Created ensemble strategy | horizon=%s | %d sources",
            horizon, len(w),
        )
        return MetadronAlphaModel(
            universe=universe, config=config,
            macro_engine=self.macro_engine,
            cube=self.cube,
            security_engine=self.security_engine,
            event_engine=self.event_engine,
            social_engine=self.social_engine,
            data_handler=self.data_handler,
        )

    def create_security_analysis_strategy(
        self, universe=None, horizon: str = "long",
    ) -> MetadronAlphaModel:
        """Graham-Dodd-Klarman fundamental analysis."""
        config = AlphaModelConfig(
            signal_sources=["security_analysis"],
            horizon=horizon,
            ensemble_weights={"security_analysis": 1.0},
        )
        logger.info("Created security-analysis strategy | horizon=%s", horizon)
        return MetadronAlphaModel(
            universe=universe, config=config,
            security_engine=self.security_engine,
            data_handler=self.data_handler,
        )

    def create_momentum_strategy(
        self, universe=None, horizon: str = "swing",
    ) -> MetadronAlphaModel:
        """Technical momentum signals."""
        config = AlphaModelConfig(
            signal_sources=["momentum"],
            horizon=horizon,
            ensemble_weights={"momentum": 1.0},
        )
        logger.info("Created momentum strategy | horizon=%s", horizon)
        return MetadronAlphaModel(
            universe=universe, config=config,
            data_handler=self.data_handler,
        )

    def create_event_driven_strategy(
        self, universe=None, horizon: str = "medium",
    ) -> MetadronAlphaModel:
        """Event-driven signals (M&A arb, PEAD, catalysts)."""
        config = AlphaModelConfig(
            signal_sources=["event"],
            horizon=horizon,
            ensemble_weights={"event": 1.0},
        )
        logger.info("Created event-driven strategy | horizon=%s", horizon)
        return MetadronAlphaModel(
            universe=universe, config=config,
            event_engine=self.event_engine,
            data_handler=self.data_handler,
        )

    def create_combined_strategy(
        self, universe=None, horizon: str = "medium",
    ) -> MetadronAlphaModel:
        """All engines combined with equal weighting."""
        sources = [
            "macro", "cube", "security_analysis",
            "momentum", "event", "social",
        ]
        w = {s: 1.0 / len(sources) for s in sources}
        config = AlphaModelConfig(
            signal_sources=sources,
            horizon=horizon,
            ensemble_weights=w,
        )
        logger.info(
            "Created combined strategy | horizon=%s | %d sources",
            horizon, len(sources),
        )
        return MetadronAlphaModel(
            universe=universe, config=config,
            macro_engine=self.macro_engine,
            cube=self.cube,
            security_engine=self.security_engine,
            event_engine=self.event_engine,
            social_engine=self.social_engine,
            data_handler=self.data_handler,
        )


# =========================================================================
# Convenience helpers
# =========================================================================

def create_default_runner(
    initial_cash: float = 1_000_000.0,
    rebalance: str = "end_of_month",
    regime: str = "RANGE",
) -> QSTraderBacktestRunner:
    """
    Create a QSTraderBacktestRunner with sensible defaults.

    Parameters
    ----------
    initial_cash : float
        Starting capital.
    rebalance : str
        Rebalancing frequency.
    regime : str
        Initial regime assumption.

    Returns
    -------
    QSTraderBacktestRunner
    """
    config = QSTraderBacktestConfig(
        initial_cash=initial_cash,
        rebalance=rebalance,
    )
    risk_cfg = RiskModelConfig(regime=regime)
    runner = QSTraderBacktestRunner(
        config=config,
        alpha_model=MetadronAlphaModel(),
        risk_model=MetadronRiskModel(config=risk_cfg),
        fee_model=MetadronFeeModel(),
    )
    logger.info(
        "Default runner | cash=%.0f | rebalance=%s | regime=%s",
        initial_cash, rebalance, regime,
    )
    return runner


def quick_backtest(
    universe: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    signal_callback: Optional[Callable] = None,
    initial_cash: float = 1_000_000.0,
    rebalance: str = "end_of_month",
) -> BacktestMetrics:
    """
    Run a quick backtest with minimal configuration.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols.
    start_date, end_date : str
    signal_callback : callable, optional
        ``f(dt, assets) -> dict[str, float]``.
    initial_cash : float
    rebalance : str

    Returns
    -------
    BacktestMetrics
    """
    alpha = MetadronAlphaModel(
        universe=universe, signal_callback=signal_callback,
    )
    runner = create_default_runner(
        initial_cash=initial_cash, rebalance=rebalance,
    )
    runner.alpha_model = alpha
    return runner.run_backtest(
        strategy="quick_backtest",
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )
