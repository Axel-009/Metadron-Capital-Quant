"""
ML Model Bridge Adapters
========================
Signal bridges that produce SignalType values consumed by MLVoteEnsemble
in engine.execution.execution_engine.

All bridges follow the pattern:
  - try/except on external imports
  - graceful degradation with pure-numpy fallbacks
  - produce signals compatible with engine.execution.paper_broker.SignalType
"""

from .finrl_bridge import FinRLBridge
from .nvidia_tft_adapter import NvidiaTFTAdapter
from .monte_carlo_bridge import MonteCarloBridge
from .stock_prediction_bridge import StockPredictionBridge
from .deep_trading_features import DeepTradingFeatureBuilder
from .kserve_adapter import KServeAdapter
from .qlib_bridge import QLIBBridge
from .hedgefund_tracker_bridge import HedgefundTrackerBridge

__all__ = [
    "FinRLBridge",
    "NvidiaTFTAdapter",
    "MonteCarloBridge",
    "StockPredictionBridge",
    "DeepTradingFeatureBuilder",
    "KServeAdapter",
    "QLIBBridge",
    "HedgefundTrackerBridge",
]
