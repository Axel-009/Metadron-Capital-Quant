"""
MiroMomentumEngine — Agent-based market microstructure simulation.

Powered by MiroFish's MarketSimulator (Kyle's Lambda, Heterogeneous Agent
Models, order book dynamics). Runs Monte Carlo agent simulations per security
to produce directional signals, regime classification, and confidence scores.

Pipeline position: L2 → feeds AlphaOptimizer, MLVoteEnsemble (Tier-6)

Data flow:
    Market data (OpenBB) → Calibrate agent population → Simulate price paths
        → Extract signal (direction, confidence, regime) → MiroMomentumSignal

No social media. Pure market microstructure simulation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger("metadron.signals.miro_momentum")

# Import agent sim engine (the actual simulation)
try:
    from .agent_sim_engine import AgentSimEngine, AgentSimSignal
except ImportError:
    AgentSimEngine = None
    AgentSimSignal = None
    logger.warning("AgentSimEngine unavailable — MiroMomentumEngine disabled")


@dataclass
class MiroMomentumSignal:
    """
    Signal output from MiroMomentum agent-based market simulation.

    Consumed by MLVoteEnsemble (Tier-6), AlphaOptimizer, and
    MiroMomentumFeatureBuilder for ML feature extraction.
    """
    ticker: str = ""
    timestamp: str = ""
    sentiment_score: float = 0.0        # [-1, +1] — agent consensus
    engagement: float = 0.0             # [0, 1] — simulation confidence
    momentum: float = 0.0               # Predicted return from MC paths
    consensus_strength: float = 0.0     # Agent agreement level
    regime: str = "random_walk"         # trending/mean_reverting/random_walk
    hurst_exponent: float = 0.5
    miro_momentum_signal: str = "HOLD"  # BUY/SELL/HOLD (mapped from agent direction)
    n_simulations: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "sentiment_score": round(self.sentiment_score, 4),
            "engagement": round(self.engagement, 4),
            "momentum": round(self.momentum, 6),
            "consensus_strength": round(self.consensus_strength, 4),
            "regime": self.regime,
            "hurst_exponent": round(self.hurst_exponent, 4),
            "miro_momentum_signal": self.miro_momentum_signal,
            "n_simulations": self.n_simulations,
        }


class MiroMomentumEngine:
    """
    Agent-based market microstructure simulation engine.

    Runs Monte Carlo agent simulations per security to produce
    directional signals, regime classification, and confidence scores.

    Uses MiroFish's MarketSimulator (Kyle's Lambda, HAM, order book)
    calibrated to real market data from OpenBB.

    No social media. Pure market microstructure.
    """

    def __init__(self, n_simulations: int = 100, simulation_horizon: int = 20):
        if AgentSimEngine is None:
            raise ImportError("AgentSimEngine required for MiroMomentumEngine")

        self._engine = AgentSimEngine(
            n_simulations=n_simulations,
            simulation_horizon=simulation_horizon,
        )
        logger.info("MiroMomentumEngine initialized (agent sim, %d paths)", n_simulations)

    def analyze(self, tickers: Optional[List[str]] = None) -> Dict[str, MiroMomentumSignal]:
        """
        Run agent simulation for tickers and return momentum signals.

        Args:
            tickers: List of tickers to analyze. If None, returns empty.

        Returns:
            Dict mapping ticker → MiroMomentumSignal
        """
        if not tickers:
            return {}

        results = {}
        for ticker in tickers:
            try:
                signal = self._engine.simulate_ticker(ticker)
                sig = MiroMomentumSignal(
                    ticker=signal.ticker,
                    timestamp=signal.timestamp.isoformat(),
                    sentiment_score=signal.agent_consensus,
                    engagement=signal.confidence,
                    momentum=signal.predicted_return,
                    consensus_strength=signal.confidence,
                    regime=signal.regime,
                    hurst_exponent=signal.hurst_exponent,
                    miro_momentum_signal=signal.direction,
                    n_simulations=signal.simulated_paths,
                )
                results[ticker] = sig
            except Exception as e:
                logger.debug("Agent sim failed for %s: %s", ticker, e)
                results[ticker] = MiroMomentumSignal(ticker=ticker, miro_momentum_signal="HOLD")

        return results

    def get_signal(self, ticker: str) -> MiroMomentumSignal:
        """Get signal for a single ticker."""
        results = self.analyze([ticker])
        return results.get(ticker, MiroMomentumSignal(ticker=ticker))

    def get_all_signals(self) -> Dict[str, MiroMomentumSignal]:
        """Get signals for all tickers (called by pipeline)."""
        return {}


# ── Backward compatibility aliases ────────────────────────────────
# These allow existing code that imports the old names to keep working
# while we migrate references.
SocialSnapshot = MiroMomentumSignal
SocialPredictionEngine = MiroMomentumEngine
