"""
MiroFish integration with Metadron Capital Investment Platform.

MiroFish's multi-agent simulation capabilities are used for:
1. Market simulation -- simulate market scenarios with agent-based models
2. Strategy backtesting -- run agent swarms that simulate trading strategies
3. Risk scenario analysis -- Monte Carlo with agent interactions
4. Portfolio stress testing -- simulate extreme market events

Monte Carlo VaR: VaR_alpha = -mu + z_alpha * sigma (parametric)
    or percentile method: VaR_alpha = -quantile(returns, 1-alpha)
CVaR (Expected Shortfall): CVaR_alpha = E[L | L > VaR_alpha]

Agent-Based Market Simulation:
    Each agent has: wealth, risk_tolerance, strategy, information_set
    Price formation: P_t = f(sum demand_i(P_{t-1}, info_i))
    Market impact: dP = lambda * sign(order) * |order|^delta where delta ~ 0.5

Kyle's Lambda Model:
    P_t = P_{t-1} + lambda * (x_t + u_t)
    Where x_t = informed order flow, u_t = noise trader order flow
    lambda = sigma_v / (2 * sigma_u) (price impact coefficient)

Heterogeneous Agent Models (HAM):
    Chartists: E_c[R_{t+1}] = g * (P_t - P_{t-1}) / P_{t-1}
    Fundamentalists: E_f[R_{t+1}] = phi * (P* - P_t) / P_t
    Fraction switching: n_{c,t} = exp(beta * pi_{c,t-1}) / Z_t

Order Book Dynamics:
    Bid-ask spread: s = 2 * lambda * sigma
    Market depth: D = 1 / lambda
    Price impact: dP/dQ = lambda (Kyle, 1985)

Hurst Exponent for Regime Detection:
    H = log(R/S) / log(n)
    H > 0.5: trending (persistent), H = 0.5: random walk, H < 0.5: mean-reverting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime


class AgentType(Enum):
    MOMENTUM_TRADER = "momentum"
    VALUE_INVESTOR = "value"
    MARKET_MAKER = "market_maker"
    NOISE_TRADER = "noise"
    FUNDAMENTAL_ANALYST = "fundamental"
    ARBITRAGEUR = "arbitrageur"


@dataclass
class MarketAgent:
    """
    Individual market agent with wealth, strategy, and state tracking.

    Agent Utility Function (CRRA):
        U(W) = W^{1-gamma} / (1-gamma) for gamma != 1
        U(W) = ln(W) for gamma = 1
    Where gamma = 1 / risk_tolerance (risk aversion coefficient)

    Optimal Position (Mean-Variance):
        w* = (E[R] - r_f) / (gamma * sigma^2)
    """
    agent_type: AgentType
    wealth: float
    risk_tolerance: float
    position: float = 0.0
    pnl: float = 0.0
    trade_history: List[float] = field(default_factory=list)
    information_quality: float = 0.5  # 0 = no info, 1 = perfect info


class MarketSimulator:
    """
    Agent-based market simulator for Metadron Capital platform.

    Implements a continuous double auction with heterogeneous agents.
    Supports stress testing, scenario analysis, and Monte Carlo simulation.
    """

    def __init__(self, n_agents: int = 100, initial_price: float = 100.0):
        self.agents: List[MarketAgent] = []
        self.price_history: List[float] = [initial_price]
        self.volume_history: List[float] = [0.0]
        self.n_agents = n_agents
        self.order_flow_history: List[float] = []
        self.volatility_history: List[float] = []

    def initialize_agents(self) -> None:
        """
        Create diverse agent population with realistic distribution.

        Population Composition (based on empirical market microstructure):
            - Momentum Traders: 25% (trend followers, CTA-like)
            - Value Investors: 20% (fundamental, mean-reversion)
            - Market Makers: 10% (liquidity providers, delta-neutral)
            - Noise Traders: 30% (retail, uninformed)
            - Fundamental Analysts: 10% (deep research, longer horizon)
            - Arbitrageurs: 5% (stat arb, pairs trading)

        Wealth Distribution: Log-normal (Pareto-like tail)
            W ~ LogNormal(mu=12, sigma=1) => median ~$160K, mean ~$250K
        """
        self.agents = []
        agent_distribution = {
            AgentType.MOMENTUM_TRADER: 0.25,
            AgentType.VALUE_INVESTOR: 0.20,
            AgentType.MARKET_MAKER: 0.10,
            AgentType.NOISE_TRADER: 0.30,
            AgentType.FUNDAMENTAL_ANALYST: 0.10,
            AgentType.ARBITRAGEUR: 0.05,
        }

        info_quality_map = {
            AgentType.MOMENTUM_TRADER: (0.3, 0.15),
            AgentType.VALUE_INVESTOR: (0.6, 0.15),
            AgentType.MARKET_MAKER: (0.5, 0.1),
            AgentType.NOISE_TRADER: (0.1, 0.1),
            AgentType.FUNDAMENTAL_ANALYST: (0.8, 0.1),
            AgentType.ARBITRAGEUR: (0.7, 0.1),
        }

        for agent_type, fraction in agent_distribution.items():
            n = int(self.n_agents * fraction)
            iq_mean, iq_std = info_quality_map[agent_type]
            for _ in range(n):
                self.agents.append(MarketAgent(
                    agent_type=agent_type,
                    wealth=np.random.lognormal(mean=12, sigma=1),
                    risk_tolerance=np.random.uniform(0.1, 0.9),
                    information_quality=np.clip(np.random.normal(iq_mean, iq_std), 0.0, 1.0),
                ))

    def _compute_agent_demand(self, agent: MarketAgent, fundamental_value: float) -> float:
        """
        Compute demand for a single agent based on their type and strategy.

        Demand Functions by Type:
            Momentum: d = tau * W * sign(r) * min(|r| * k, 1)
                where r = recent return, k = sensitivity, tau = risk tolerance
            Value: d = tau * W * (V* - P) / P
                mean-reversion toward fundamental value
            Market Maker: d = -position * reversion_speed
                inventory management, provide liquidity
            Noise: d = tau * W * N(0, sigma_noise)
                random, uninformed trading
            Fundamental: d = tau * W * iq * (V* - P) / P
                like value but scaled by information quality
            Arbitrageur: d = tau * W * (V* - P) / P * speed
                fast convergence trading

        Args:
            agent: MarketAgent instance.
            fundamental_value: Current fundamental/fair value estimate.

        Returns:
            Demand quantity (positive = buy, negative = sell).
        """
        current_price = self.price_history[-1]

        if agent.agent_type == AgentType.MOMENTUM_TRADER:
            if len(self.price_history) >= 2:
                # Use multiple lookback periods for robust momentum
                short_mom = (current_price - self.price_history[-2]) / self.price_history[-2]
                if len(self.price_history) >= 6:
                    med_mom = (current_price - self.price_history[-6]) / self.price_history[-6]
                    momentum = 0.6 * short_mom + 0.4 * med_mom
                else:
                    momentum = short_mom
                demand = (
                    agent.risk_tolerance
                    * agent.wealth
                    * np.sign(momentum)
                    * min(abs(momentum) * 10, 1.0)
                )
            else:
                demand = 0.0

        elif agent.agent_type == AgentType.VALUE_INVESTOR:
            mispricing = (fundamental_value - current_price) / current_price
            # Value investors are patient - scale by sqrt of mispricing for gradual entry
            demand = agent.risk_tolerance * agent.wealth * np.tanh(mispricing * 3)

        elif agent.agent_type == AgentType.MARKET_MAKER:
            # Inventory management: revert position toward zero
            # Also provide liquidity proportional to bid-ask spread opportunity
            inventory_reversion = -agent.position * 0.1
            # Earn spread by being contrarian to recent order flow
            if len(self.order_flow_history) > 0:
                recent_flow = self.order_flow_history[-1]
                contrarian = -np.sign(recent_flow) * agent.wealth * 0.001
            else:
                contrarian = 0.0
            demand = inventory_reversion + contrarian

        elif agent.agent_type == AgentType.NOISE_TRADER:
            # Random demand with occasional herding
            base_noise = np.random.normal(0, 0.01)
            # Herding effect: partially follow recent price trend
            if len(self.price_history) >= 2:
                recent_return = (current_price - self.price_history[-2]) / self.price_history[-2]
                herding = 0.3 * recent_return
            else:
                herding = 0.0
            demand = agent.risk_tolerance * agent.wealth * (base_noise + herding)

        elif agent.agent_type == AgentType.FUNDAMENTAL_ANALYST:
            mispricing = (fundamental_value - current_price) / current_price
            # Scale by information quality - better info leads to larger positions
            demand = (
                agent.risk_tolerance
                * agent.wealth
                * mispricing
                * agent.information_quality
                * 1.5
            )

        elif agent.agent_type == AgentType.ARBITRAGEUR:
            mispricing = (fundamental_value - current_price) / current_price
            # Fast convergence - only trade when mispricing exceeds threshold
            if abs(mispricing) > 0.005:  # 50bps threshold
                demand = agent.risk_tolerance * agent.wealth * mispricing * 2.0
            else:
                demand = 0.0

        else:
            demand = 0.0

        return demand

    def simulate_step(self, fundamental_value: float) -> float:
        """
        Execute single simulation step with agent interactions and price formation.

        Price Formation Process:
            1. Each agent computes demand based on their strategy
            2. Aggregate net order flow: OF = sum(demand_i)
            3. Price impact (Kyle's lambda): dP = lambda * OF
            4. New price: P_t = P_{t-1} * (1 + dP / P_{t-1})
            5. Update agent positions and P&L

        Market Microstructure:
            - Price impact is concave: dP ~ lambda * sign(OF) * |OF|^0.5
            - Volatility clustering via GARCH-like feedback
            - Fat tails emerge from agent heterogeneity

        Args:
            fundamental_value: Current fundamental/fair value.

        Returns:
            New price after this step.
        """
        current_price = self.price_history[-1]
        total_demand = 0.0
        total_volume = 0.0

        for agent in self.agents:
            demand = self._compute_agent_demand(agent, fundamental_value)
            total_demand += demand
            total_volume += abs(demand)

            # Update agent position
            shares_traded = demand / current_price if current_price > 0 else 0.0
            agent.position += shares_traded
            agent.trade_history.append(shares_traded)

        self.order_flow_history.append(total_demand)

        # Price impact: Kyle's lambda model with square-root impact
        lambda_impact = 0.001
        if total_demand != 0:
            # Square-root impact for realistic market microstructure
            impact = lambda_impact * np.sign(total_demand) * np.sqrt(abs(total_demand))
        else:
            impact = 0.0

        # Add small noise for microstructure effects
        microstructure_noise = np.random.normal(0, current_price * 0.0005)

        new_price = max(current_price + impact + microstructure_noise, 0.01)
        self.price_history.append(new_price)
        self.volume_history.append(total_volume)

        # Update agent P&L
        price_change = new_price - current_price
        for agent in self.agents:
            agent.pnl += agent.position * price_change
            agent.wealth += agent.position * price_change

        # Track realized volatility
        if len(self.price_history) >= 3:
            recent_returns = np.diff(np.log(self.price_history[-min(20, len(self.price_history)):]))
            self.volatility_history.append(float(np.std(recent_returns) * np.sqrt(252)))

        return new_price

    def run_simulation(self, n_steps: int, fundamental_values: list) -> pd.DataFrame:
        """
        Run full agent-based market simulation.

        Generates a complete price path with volume, volatility, and order flow data.

        Args:
            n_steps: Number of simulation steps (trading periods).
            fundamental_values: List of fundamental values for each step.
                If shorter than n_steps, the last value is repeated.

        Returns:
            DataFrame with columns: price, volume, order_flow, volatility.
        """
        self.initialize_agents()

        for i in range(n_steps):
            fv = fundamental_values[i] if i < len(fundamental_values) else fundamental_values[-1]
            self.simulate_step(fv)

        # Build results DataFrame
        n_prices = len(self.price_history)
        results = pd.DataFrame({
            "price": self.price_history,
            "volume": self.volume_history + [0.0] * (n_prices - len(self.volume_history)),
            "order_flow": [0.0] + self.order_flow_history + [0.0] * (
                n_prices - len(self.order_flow_history) - 1
            ),
        })

        # Add returns
        results["return"] = results["price"].pct_change()
        results["log_return"] = np.log(results["price"] / results["price"].shift(1))

        # Add rolling volatility
        results["rolling_vol_20"] = results["log_return"].rolling(20).std() * np.sqrt(252)

        return results

    def stress_test(self, shock_magnitude: float, shock_type: str = "price") -> dict:
        """
        Simulate market stress scenario and measure system response.

        Stress Test Types:
            - price: Immediate price drop (flash crash, gap down)
            - volatility: Volatility spike (VIX explosion)
            - liquidity: Liquidity withdrawal (market makers exit)
            - correlation: Correlation spike (contagion)

        Recovery Analysis:
            - Max drawdown from shock
            - Time to 95% recovery
            - Agent wealth distribution post-shock
            - Volatility clustering post-shock

        Extreme Value Theory:
            P(X > x) ~ L(x) * x^{-alpha} for x -> infinity
            Where alpha is the tail index (typically 3-5 for financial returns)

        Args:
            shock_magnitude: Size of shock (e.g., 0.1 = 10% price drop).
            shock_type: Type of stress scenario.

        Returns:
            Dictionary with stress test results.
        """
        if not self.agents:
            self.initialize_agents()

        current_price = self.price_history[-1]
        pre_shock_price = current_price

        # Apply shock
        if shock_type == "price":
            shocked_price = current_price * (1 - shock_magnitude)
        elif shock_type == "volatility":
            # Volatility shock creates large random move
            vol_multiplier = 1 + shock_magnitude * 5
            shocked_price = current_price * np.exp(
                np.random.normal(0, current_price * 0.01 * vol_multiplier)
            )
        elif shock_type == "liquidity":
            # Remove market makers temporarily, creating larger price impact
            active_agents = [a for a in self.agents if a.agent_type != AgentType.MARKET_MAKER]
            panic_selling = sum(
                a.position * a.risk_tolerance * shock_magnitude
                for a in active_agents
                if a.position > 0
            )
            shocked_price = current_price * (1 - 0.001 * abs(panic_selling) / current_price)
            shocked_price = max(shocked_price, current_price * (1 - shock_magnitude * 1.5))
        elif shock_type == "correlation":
            # Correlation shock: all agents behave similarly (herding)
            shocked_price = current_price * (1 - shock_magnitude * 0.8)
        else:
            shocked_price = current_price

        shocked_price = max(shocked_price, 0.01)
        self.price_history.append(shocked_price)

        # Simulate recovery period (50 steps)
        recovery_steps = []
        for _ in range(50):
            p = self.simulate_step(pre_shock_price)
            recovery_steps.append(p)

        # Analyze recovery
        all_post_shock = [shocked_price] + recovery_steps
        max_drawdown = min(all_post_shock) / pre_shock_price - 1

        # Time to 95% recovery
        recovery_threshold = pre_shock_price * 0.95
        recovery_time = len(recovery_steps)  # default: never recovered
        for i, p in enumerate(recovery_steps):
            if p >= recovery_threshold:
                recovery_time = i + 1
                break

        # Agent wealth impact
        wealth_changes = []
        for agent in self.agents:
            wealth_change_pct = (agent.wealth - np.exp(12)) / np.exp(12)  # vs initial median
            wealth_changes.append({
                "type": agent.agent_type.value,
                "wealth_change_pct": round(wealth_change_pct, 4),
                "pnl": round(agent.pnl, 2),
            })

        # Agent survival analysis
        bankrupt_agents = sum(1 for a in self.agents if a.wealth <= 0)
        distressed_agents = sum(
            1 for a in self.agents if 0 < a.wealth < np.exp(12) * 0.2
        )

        # Post-shock volatility
        post_shock_returns = np.diff(np.log(np.array(all_post_shock)))
        post_shock_vol = float(np.std(post_shock_returns) * np.sqrt(252)) if len(post_shock_returns) > 1 else 0.0

        return {
            "shock_type": shock_type,
            "shock_magnitude": shock_magnitude,
            "pre_shock_price": round(pre_shock_price, 2),
            "immediate_shock_price": round(shocked_price, 2),
            "immediate_loss_pct": round((shocked_price / pre_shock_price - 1) * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "recovery_time_steps": recovery_time,
            "final_price": round(recovery_steps[-1], 2),
            "final_recovery_pct": round((recovery_steps[-1] / pre_shock_price - 1) * 100, 2),
            "post_shock_volatility": round(post_shock_vol * 100, 2),
            "bankrupt_agents": bankrupt_agents,
            "distressed_agents": distressed_agents,
            "price_path": [round(p, 2) for p in all_post_shock],
            "agent_impact_summary": {
                agent_type.value: {
                    "avg_pnl": round(float(np.mean([
                        a.pnl for a in self.agents if a.agent_type == agent_type
                    ])), 2),
                    "avg_wealth_change": round(float(np.mean([
                        (a.wealth - np.exp(12)) / np.exp(12)
                        for a in self.agents if a.agent_type == agent_type
                    ])) * 100, 2),
                }
                for agent_type in AgentType
            },
        }

    def monte_carlo_var(
        self,
        portfolio_value: float,
        n_simulations: int = 10000,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> dict:
        """
        Monte Carlo Value at Risk calculation.

        VaR Methodologies:
            1. Parametric (Delta-Normal): VaR = -mu + z_alpha * sigma
               Assumes returns ~ N(mu, sigma^2)

            2. Historical Simulation: VaR = -percentile(returns, (1-alpha)*100)
               Non-parametric, uses actual return distribution

            3. Monte Carlo: Simulate N paths, VaR = -percentile(simulated_PnL, (1-alpha)*100)
               Most flexible, can handle non-linear instruments

        CVaR (Expected Shortfall):
            CVaR = E[L | L > VaR] = mean of losses exceeding VaR
            CVaR >= VaR always (it's the average of the tail)
            CVaR is a coherent risk measure (VaR is not: fails sub-additivity)

        Scaling:
            VaR(T days) = VaR(1 day) * sqrt(T) (under i.i.d. assumption)

        Args:
            portfolio_value: Total portfolio value.
            n_simulations: Number of Monte Carlo paths.
            confidence: Confidence level (e.g., 0.99 for 99% VaR).
            horizon_days: VaR horizon in days.

        Returns:
            Dictionary with VaR, CVaR, and distribution statistics.
        """
        if len(self.price_history) < 2:
            return {
                "var_99": 0.0,
                "cvar_99": 0.0,
                "mean_return": 0.0,
                "volatility": 0.0,
                "message": "Insufficient price history",
            }

        # Calculate historical returns
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))

        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        if sigma == 0:
            sigma = 0.01  # Fallback

        # Scale to horizon
        mu_horizon = mu * horizon_days
        sigma_horizon = sigma * np.sqrt(horizon_days)

        # Monte Carlo simulation
        simulated_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
        simulated_pnl = portfolio_value * (np.exp(simulated_returns) - 1)

        # Parametric VaR
        from scipy.stats import norm
        z_alpha = norm.ppf(confidence)
        parametric_var = portfolio_value * (-mu_horizon + z_alpha * sigma_horizon)

        # Monte Carlo VaR (percentile method)
        mc_var = float(-np.percentile(simulated_pnl, (1 - confidence) * 100))

        # CVaR (Expected Shortfall)
        losses = -simulated_pnl
        var_threshold = np.percentile(losses, confidence * 100)
        tail_losses = losses[losses >= var_threshold]
        mc_cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else mc_var

        # Historical VaR (if enough data)
        if len(returns) >= 30:
            hist_pnl = portfolio_value * (np.exp(returns * horizon_days) - 1)
            historical_var = float(-np.percentile(hist_pnl, (1 - confidence) * 100))
        else:
            historical_var = mc_var

        # Distribution statistics
        skewness = float(pd.Series(simulated_pnl).skew())
        kurtosis = float(pd.Series(simulated_pnl).kurtosis())

        # Worst-case scenarios
        worst_5_pct = float(np.percentile(simulated_pnl, 5))
        worst_1_pct = float(np.percentile(simulated_pnl, 1))
        worst_case = float(np.min(simulated_pnl))

        return {
            "var_99": round(mc_var, 2),
            "cvar_99": round(mc_cvar, 2),
            "parametric_var_99": round(parametric_var, 2),
            "historical_var_99": round(historical_var, 2),
            "mean_return": round(mu * 252, 6),  # Annualized
            "volatility": round(sigma * np.sqrt(252), 6),  # Annualized
            "horizon_days": horizon_days,
            "confidence": confidence,
            "n_simulations": n_simulations,
            "portfolio_value": portfolio_value,
            "distribution": {
                "skewness": round(skewness, 4),
                "excess_kurtosis": round(kurtosis, 4),
                "worst_5pct_pnl": round(worst_5_pct, 2),
                "worst_1pct_pnl": round(worst_1_pct, 2),
                "worst_case_pnl": round(worst_case, 2),
                "best_case_pnl": round(float(np.max(simulated_pnl)), 2),
            },
        }

    def calculate_hurst_exponent(self) -> float:
        """
        Calculate Hurst exponent from price history for regime detection.

        R/S Analysis:
            1. Divide time series into sub-periods of length n
            2. For each sub-period, calculate range R and std dev S
            3. E[R/S] ~ c * n^H as n -> infinity
            4. H = slope of log(R/S) vs log(n)

        Interpretation:
            H > 0.5: Persistent / trending (momentum works)
            H = 0.5: Random walk (no predictability)
            H < 0.5: Anti-persistent / mean-reverting (mean reversion works)

        Returns:
            Hurst exponent as float.
        """
        if len(self.price_history) < 20:
            return 0.5  # Default to random walk with insufficient data

        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))
        n = len(returns)

        # Range of sub-period sizes
        max_k = min(n // 2, 100)
        sizes = list(range(10, max_k + 1, 5))
        if not sizes:
            return 0.5

        rs_values = []
        for size in sizes:
            rs_list = []
            for start in range(0, n - size + 1, size):
                sub = returns[start : start + size]
                mean_sub = np.mean(sub)
                cumdev = np.cumsum(sub - mean_sub)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append((np.log(size), np.log(np.mean(rs_list))))

        if len(rs_values) < 2:
            return 0.5

        log_sizes, log_rs = zip(*rs_values)
        log_sizes = np.array(log_sizes)
        log_rs = np.array(log_rs)

        # Linear regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        hurst = float(coeffs[0])

        return round(np.clip(hurst, 0.0, 1.0), 4)

    def get_simulation_summary(self) -> dict:
        """
        Generate comprehensive summary of the simulation run.

        Returns:
            Dictionary with simulation statistics and agent analysis.
        """
        if len(self.price_history) < 2:
            return {"status": "No simulation data"}

        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))

        # Price statistics
        total_return = (prices[-1] / prices[0]) - 1
        annualized_vol = float(np.std(returns) * np.sqrt(252))
        sharpe = float(np.mean(returns) * 252 / (annualized_vol + 1e-10))

        # Drawdown analysis
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        max_dd = float(np.min(drawdown))

        # Agent performance by type
        agent_performance = {}
        for agent_type in AgentType:
            type_agents = [a for a in self.agents if a.agent_type == agent_type]
            if type_agents:
                pnls = [a.pnl for a in type_agents]
                agent_performance[agent_type.value] = {
                    "count": len(type_agents),
                    "avg_pnl": round(float(np.mean(pnls)), 2),
                    "total_pnl": round(float(np.sum(pnls)), 2),
                    "pnl_std": round(float(np.std(pnls)), 2),
                    "best_pnl": round(float(np.max(pnls)), 2),
                    "worst_pnl": round(float(np.min(pnls)), 2),
                    "pct_profitable": round(
                        sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1
                    ),
                }

        hurst = self.calculate_hurst_exponent()

        return {
            "n_steps": len(prices) - 1,
            "n_agents": len(self.agents),
            "price_start": round(prices[0], 2),
            "price_end": round(prices[-1], 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_volatility": round(annualized_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "hurst_exponent": hurst,
            "regime": (
                "trending" if hurst > 0.55
                else "mean_reverting" if hurst < 0.45
                else "random_walk"
            ),
            "agent_performance": agent_performance,
        }
