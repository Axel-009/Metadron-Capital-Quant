"""
AgentSimEngine — MiroFish agent-based market simulation as a signal source.

Wraps MiroFish's MarketSimulator into a pipeline-compatible signal engine.
Runs per-security agent simulations calibrated to real market data.
Outputs: predicted price direction, regime classification, confidence score.

Feed: L2 Signals → AlphaOptimizer → DecisionMatrix

Architecture:
    OpenBB market data → calibrate agents → simulate → extract signal
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Import MiroFish simulator
try:
    from intelligence_platform.MiroFish.investment_platform_integration import (
        MarketSimulator,
        AgentType,
        MarketAgent,
    )
except ImportError:
    MarketSimulator = None
    AgentType = None
    MarketAgent = None
    logger.warning("MiroFish integration module unavailable — AgentSimEngine disabled")

# Import data layer
try:
    from engine.data.openbb_data import get_adj_close, get_volume
except ImportError:
    get_adj_close = None
    get_volume = None
    logger.warning("OpenBB data layer unavailable — AgentSimEngine will use defaults")


@dataclass
class AgentSimSignal:
    """Output signal from agent simulation for a single ticker."""

    ticker: str
    timestamp: datetime
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # [0, 1]
    predicted_return: float  # Expected return over horizon
    regime: str  # "trending", "mean_reverting", "random_walk"
    hurst_exponent: float
    simulated_paths: int
    price_target: Optional[float] = None
    volatility: float = 0.0
    agent_consensus: float = 0.0  # How much agents agree [-1, +1]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "predicted_return": round(self.predicted_return, 6),
            "regime": self.regime,
            "hurst_exponent": round(self.hurst_exponent, 4),
            "simulated_paths": self.simulated_paths,
            "price_target": round(self.price_target, 2) if self.price_target else None,
            "volatility": round(self.volatility, 4),
            "agent_consensus": round(self.agent_consensus, 4),
        }


@dataclass
class CalibrationParams:
    """Calibration parameters derived from real market data."""

    n_agents: int = 200
    fundamental_value: float = 100.0
    recent_volatility: float = 0.02
    avg_volume: float = 1e6
    momentum_strength: float = 0.5
    mean_reversion_strength: float = 0.5
    noise_level: float = 0.3
    market_maker_participation: float = 0.10

    # Agent population adjustments based on market regime
    momentum_fraction: float = 0.25
    value_fraction: float = 0.20
    market_maker_fraction: float = 0.10
    noise_fraction: float = 0.30
    fundamental_fraction: float = 0.10
    arbitrageur_fraction: float = 0.05


class AgentSimEngine:
    """
    Agent-based market simulation engine.

    For each ticker in the universe:
    1. Pull recent market data from OpenBB (price, volume, volatility)
    2. Calibrate agent population to match real market microstructure
    3. Run N parallel simulations (Monte Carlo)
    4. Aggregate results into a trading signal
    5. Output: AgentSimSignal with direction, confidence, regime

    Calibration Logic:
        - High volatility → more noise traders, wider demand distributions
        - Strong trend → more momentum traders
        - Low volume → fewer agents, higher market impact
        - High volume → more agents, lower market impact per agent
        - Mean-reverting → more value/fundamental agents

    Signal Extraction:
        - Direction: sign(mean predicted return across simulations)
        - Confidence: fraction of simulations agreeing on direction
        - Regime: Hurst exponent from aggregated price paths
    """

    def __init__(
        self,
        n_simulations: int = 100,
        simulation_horizon: int = 20,
        lookback_days: int = 60,
        calibration_window: int = 20,
    ):
        """
        Args:
            n_simulations: Number of Monte Carlo paths per ticker.
            simulation_horizon: Steps ahead to simulate (e.g., 20 = ~1 trading day if step=1min).
            lookback_days: Days of historical data for calibration.
            calibration_window: Recent days for regime detection.
        """
        if MarketSimulator is None:
            raise ImportError("MiroFish integration module not available")

        self.n_simulations = n_simulations
        self.simulation_horizon = simulation_horizon
        self.lookback_days = lookback_days
        self.calibration_window = calibration_window

    def _calibrate_from_market_data(
        self, ticker: str, prices: pd.Series, volumes: pd.Series
    ) -> CalibrationParams:
        """
        Derive simulation parameters from real market data.

        Calibration Rules:
            - Recent volatility → agent noise level and demand variance
            - Price trend (momentum) → momentum trader activity level
            - Mean reversion (autocorrelation) → value investor activity
            - Volume profile → total agent count and market maker participation
            - Fundamental value → anchored to current price with drift
        """
        params = CalibrationParams()

        if len(prices) < 10:
            return params

        returns = np.diff(np.log(prices.values))

        # Volatility calibration
        recent_vol = float(np.std(returns[-self.calibration_window:]))
        params.recent_volatility = max(recent_vol, 0.001)
        params.noise_level = min(recent_vol * 10, 0.5)

        # Trend strength (momentum)
        if len(returns) >= 10:
            short_ma = np.mean(returns[-5:])
            long_ma = np.mean(returns[-20:])
            trend_strength = abs(short_ma - long_ma) / (recent_vol + 1e-10)
            params.momentum_strength = float(np.clip(trend_strength, 0, 1))

            # Mean reversion (negative autocorrelation)
            if len(returns) >= 20:
                autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
                params.mean_reversion_strength = float(max(-autocorr, 0))

        # Volume calibration
        if len(volumes) > 0:
            avg_vol = float(np.mean(volumes))
            params.avg_volume = avg_vol
            # Scale agent count by volume (more volume = more participants)
            params.n_agents = int(np.clip(avg_vol / 10000, 50, 500))
            # Market maker participation scales with volume
            params.market_maker_participation = float(np.clip(avg_vol / 1e7, 0.05, 0.15))

        # Fundamental value: anchored to current price with slight mean-reversion pull
        current_price = float(prices.iloc[-1])
        if len(prices) >= 50:
            long_term_mean = float(np.mean(prices.values[-50:]))
            # Blend current price and long-term mean
            params.fundamental_value = 0.7 * current_price + 0.3 * long_term_mean
        else:
            params.fundamental_value = current_price

        # Adjust agent fractions based on detected regime
        if params.momentum_strength > 0.6:
            # Strong trend: more momentum traders
            params.momentum_fraction = 0.35
            params.noise_fraction = 0.20
            params.value_fraction = 0.15
        elif params.mean_reversion_strength > 0.4:
            # Mean-reverting: more value/fundamental traders
            params.value_fraction = 0.30
            params.fundamental_fraction = 0.15
            params.momentum_fraction = 0.15
            params.noise_fraction = 0.25
        # else: use defaults

        return params

    def _run_single_simulation(
        self, params: CalibrationParams, seed: Optional[int] = None
    ) -> Tuple[List[float], float]:
        """
        Run a single agent-based simulation path.

        Returns:
            Tuple of (price_path, final_return)
        """
        if seed is not None:
            np.random.seed(seed)

        sim = MarketSimulator(n_agents=params.n_agents, initial_price=params.fundamental_value)
        sim.initialize_agents()

        # Generate fundamental value path (slight random walk around anchor)
        fv_path = [params.fundamental_value]
        for _ in range(self.simulation_horizon):
            fv_drift = np.random.normal(0, params.fundamental_value * params.recent_volatility * 0.1)
            fv_path.append(fv_path[-1] + fv_drift)

        # Run simulation
        results = sim.run_simulation(self.simulation_horizon, fv_path)
        price_path = results["price"].values.tolist()

        final_return = (price_path[-1] / price_path[0]) - 1 if price_path[0] > 0 else 0.0

        return price_path, final_return

    def simulate_ticker(self, ticker: str) -> AgentSimSignal:
        """
        Run full Monte Carlo agent simulation for a single ticker.

        Steps:
        1. Fetch market data from OpenBB
        2. Calibrate agent population
        3. Run N simulations in parallel
        4. Aggregate into signal

        Args:
            ticker: Stock ticker symbol.

        Returns:
            AgentSimSignal with direction, confidence, regime.
        """
        now = datetime.now()

        # Fetch market data
        if get_adj_close is not None:
            try:
                prices = get_adj_close(ticker, period=f"{self.lookback_days}d")
                volumes = get_volume(ticker, period=f"{self.lookback_days}d") if get_volume else pd.Series()
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                return self._neutral_signal(ticker, now)
        else:
            return self._neutral_signal(ticker, now)

        if prices is None or len(prices) < 10:
            return self._neutral_signal(ticker, now)

        # Calibrate
        params = self._calibrate_from_market_data(ticker, prices, volumes)

        # Run Monte Carlo simulations
        returns = []
        hurst_values = []
        final_prices = []

        for i in range(self.n_simulations):
            try:
                price_path, final_return = self._run_single_simulation(params, seed=i)
                returns.append(final_return)
                final_prices.append(price_path[-1])

                # Calculate Hurst from this path
                sim = MarketSimulator(n_agents=params.n_agents, initial_price=params.fundamental_value)
                sim.price_history = price_path
                h = sim.calculate_hurst_exponent()
                hurst_values.append(h)
            except Exception as e:
                logger.debug(f"Simulation {i} failed for {ticker}: {e}")
                continue

        if not returns:
            return self._neutral_signal(ticker, now)

        # Aggregate results
        returns = np.array(returns)
        mean_return = float(np.mean(returns))
        median_return = float(np.median(returns))
        std_return = float(np.std(returns))
        mean_hurst = float(np.mean(hurst_values)) if hurst_values else 0.5

        # Direction
        if mean_return > 0.001:
            direction = "BUY"
        elif mean_return < -0.001:
            direction = "SELL"
        else:
            direction = "HOLD"

        # Confidence: fraction of simulations agreeing on direction
        if direction == "BUY":
            confidence = float(np.mean(returns > 0))
        elif direction == "SELL":
            confidence = float(np.mean(returns < 0))
        else:
            confidence = 0.5

        # Regime from Hurst
        if mean_hurst > 0.55:
            regime = "trending"
        elif mean_hurst < 0.45:
            regime = "mean_reverting"
        else:
            regime = "random_walk"

        # Agent consensus: normalized mean return scaled by agreement
        agent_consensus = float(np.clip(mean_return / (std_return + 1e-10), -1, 1))

        # Price target
        current_price = float(prices.iloc[-1])
        price_target = current_price * (1 + mean_return)

        return AgentSimSignal(
            ticker=ticker,
            timestamp=now,
            direction=direction,
            confidence=confidence,
            predicted_return=mean_return,
            regime=regime,
            hurst_exponent=mean_hurst,
            simulated_paths=len(returns),
            price_target=price_target,
            volatility=float(np.std(returns) * np.sqrt(252)),
            agent_consensus=agent_consensus,
        )

    def simulate_universe(self, tickers: List[str]) -> Dict[str, AgentSimSignal]:
        """
        Run agent simulation for all tickers in the universe.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Dict mapping ticker → AgentSimSignal.
        """
        signals = {}
        for ticker in tickers:
            try:
                signals[ticker] = self.simulate_ticker(ticker)
                logger.info(
                    f"{ticker}: {signals[ticker].direction} "
                    f"(confidence={signals[ticker].confidence:.2f}, "
                    f"regime={signals[ticker].regime})"
                )
            except Exception as e:
                logger.error(f"Simulation failed for {ticker}: {e}")
                signals[ticker] = self._neutral_signal(ticker, datetime.now())
        return signals

    def _neutral_signal(self, ticker: str, timestamp: datetime) -> AgentSimSignal:
        """Return a neutral HOLD signal when simulation cannot run."""
        return AgentSimSignal(
            ticker=ticker,
            timestamp=timestamp,
            direction="HOLD",
            confidence=0.5,
            predicted_return=0.0,
            regime="random_walk",
            hurst_exponent=0.5,
            simulated_paths=0,
            price_target=None,
            volatility=0.0,
            agent_consensus=0.0,
        )
