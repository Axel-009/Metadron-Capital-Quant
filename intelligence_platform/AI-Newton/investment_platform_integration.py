"""
AI-Newton integration with the Metadron Capital Investment Platform.

Physics-inspired optimization methods applied to portfolio optimization
and strategy parameter tuning across the full universe of assets.

All data sourced exclusively from OpenBB.

Simulated Annealing for Portfolio Optimization:
    Energy function: E(w) = -Sharpe(w) = -(w'mu - r_f) / sqrt(w'Sigma*w)
    Temperature schedule: T(k) = T_0 * alpha^k, alpha in (0.9, 0.999)
    Accept probability: P(accept) = exp(-DeltaE / T)

Genetic Algorithm for Strategy Selection:
    Chromosome: binary vector of strategy inclusion/exclusion
    Fitness: out-of-sample Sharpe ratio
    Crossover: uniform crossover of strategy weights
    Mutation: Gaussian perturbation N(0, sigma_mut)

Particle Swarm for Parameter Optimization:
    v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
    x_i(t+1) = x_i(t) + v_i(t+1)
    Inertia: w decays linearly from 0.9 to 0.4

Gradient Descent Risk Parity:
    Objective: min sum_i (w_i * (Sigma*w)_i - b_i * w'Sigma*w)^2
    where b_i = 1/n (equal risk budget)
    Update: w <- w - lr * grad(objective)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"


@dataclass
class OptimizationResult:
    method: OptimizationMethod
    weights: dict  # symbol -> weight
    sharpe_ratio: float
    expected_return: float
    volatility: float
    max_drawdown: float
    iterations: int
    convergence_history: list = field(default_factory=list)


class PhysicsOptimizer:
    """Physics-inspired optimization for portfolio construction."""

    def simulated_annealing_portfolio(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.04,
        T_0: float = 1.0,
        alpha: float = 0.995,
        max_iter: int = 10000,
        constraints: Optional[dict] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights using simulated annealing.

        Energy function: E(w) = -Sharpe(w)
        Sharpe(w) = (w'mu - r_f) / sqrt(w'Sigma*w)

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns matrix (dates x assets).
        risk_free_rate : float
            Annual risk-free rate.
        T_0 : float
            Initial temperature.
        alpha : float
            Cooling rate.
        max_iter : int
            Maximum iterations.
        constraints : dict, optional
            {"max_weight": 0.10, "min_weight": 0.0, "long_only": True}
        """
        n_assets = returns.shape[1]
        mu = returns.mean().values * 252
        sigma = returns.cov().values * 252
        rf = risk_free_rate

        if constraints is None:
            constraints = {"max_weight": 0.15, "min_weight": 0.0, "long_only": True}

        max_w = constraints.get("max_weight", 0.15)
        min_w = constraints.get("min_weight", 0.0)

        # Initialize with equal weights
        w = np.ones(n_assets) / n_assets
        best_w = w.copy()

        def energy(weights):
            port_ret = weights @ mu
            port_vol = np.sqrt(weights @ sigma @ weights)
            if port_vol < 1e-10:
                return 1e6
            return -(port_ret - rf) / port_vol

        best_energy = energy(w)
        current_energy = best_energy
        history = [best_energy]
        T = T_0

        for k in range(max_iter):
            # Propose neighbor: perturb two random weights
            new_w = w.copy()
            i, j = np.random.choice(n_assets, 2, replace=False)
            delta = np.random.normal(0, 0.02)
            new_w[i] += delta
            new_w[j] -= delta

            # Apply constraints
            new_w = np.clip(new_w, min_w, max_w)
            if new_w.sum() > 0:
                new_w /= new_w.sum()

            new_energy = energy(new_w)
            delta_e = new_energy - current_energy

            # Metropolis criterion
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(T, 1e-10)):
                w = new_w
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_w = w.copy()

            T *= alpha
            if k % 100 == 0:
                history.append(best_energy)

        symbols = returns.columns.tolist()
        port_ret = best_w @ mu
        port_vol = np.sqrt(best_w @ sigma @ best_w)

        return OptimizationResult(
            method=OptimizationMethod.SIMULATED_ANNEALING,
            weights={symbols[i]: float(best_w[i]) for i in range(n_assets)},
            sharpe_ratio=float(-best_energy),
            expected_return=float(port_ret),
            volatility=float(port_vol),
            max_drawdown=self._estimate_max_drawdown(returns, best_w),
            iterations=max_iter,
            convergence_history=history,
        )

    def genetic_algorithm_strategy_selection(
        self,
        strategy_returns: dict,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
    ) -> OptimizationResult:
        """
        Select optimal combination of strategies using genetic algorithm.

        Chromosome: weight vector for each strategy
        Fitness: out-of-sample Sharpe ratio
        Selection: tournament selection (size 3)
        Crossover: uniform crossover
        Mutation: Gaussian perturbation
        """
        strategy_names = list(strategy_returns.keys())
        n_strategies = len(strategy_names)

        # Build returns matrix
        returns_matrix = pd.DataFrame(strategy_returns)
        mu = returns_matrix.mean().values * 252
        sigma_mat = returns_matrix.cov().values * 252

        # Initialize population
        population = np.random.dirichlet(np.ones(n_strategies), population_size)

        def fitness(weights):
            port_ret = weights @ mu
            port_vol = np.sqrt(weights @ sigma_mat @ weights)
            if port_vol < 1e-10:
                return -1e6
            return (port_ret - 0.04) / port_vol

        best_individual = None
        best_fitness = -1e6
        history = []

        for gen in range(generations):
            fitness_scores = np.array([fitness(ind) for ind in population])
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            history.append(best_fitness)

            # Tournament selection
            new_population = [best_individual.copy()]  # elitism
            while len(new_population) < population_size:
                # Tournament select two parents
                candidates = np.random.choice(population_size, 3, replace=False)
                parent1 = population[candidates[np.argmax(fitness_scores[candidates])]]
                candidates = np.random.choice(population_size, 3, replace=False)
                parent2 = population[candidates[np.argmax(fitness_scores[candidates])]]

                # Crossover
                if np.random.random() < crossover_rate:
                    mask = np.random.random(n_strategies) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    child += np.random.normal(0, 0.02, n_strategies)
                    child = np.clip(child, 0, 1)

                child /= child.sum() if child.sum() > 0 else 1
                new_population.append(child)

            population = np.array(new_population[:population_size])

        best_w = best_individual
        port_ret = best_w @ mu
        port_vol = np.sqrt(best_w @ sigma_mat @ best_w)

        return OptimizationResult(
            method=OptimizationMethod.GENETIC_ALGORITHM,
            weights={strategy_names[i]: float(best_w[i]) for i in range(n_strategies)},
            sharpe_ratio=float(best_fitness),
            expected_return=float(port_ret),
            volatility=float(port_vol),
            max_drawdown=self._estimate_max_drawdown(returns_matrix, best_w),
            iterations=generations,
            convergence_history=history,
        )

    def particle_swarm_parameter_optimization(
        self,
        objective_fn: Callable,
        param_ranges: dict,
        n_particles: int = 50,
        max_iter: int = 200,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ) -> dict:
        """
        Optimize strategy parameters using Particle Swarm Optimization.

        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        x_i(t+1) = x_i(t) + v_i(t+1)
        """
        param_names = list(param_ranges.keys())
        n_dims = len(param_names)
        lower = np.array([param_ranges[p][0] for p in param_names])
        upper = np.array([param_ranges[p][1] for p in param_names])

        # Initialize particles
        positions = np.random.uniform(lower, upper, (n_particles, n_dims))
        velocities = np.random.uniform(-(upper - lower) * 0.1, (upper - lower) * 0.1, (n_particles, n_dims))
        pbest_positions = positions.copy()
        pbest_scores = np.full(n_particles, -np.inf)
        gbest_position = positions[0].copy()
        gbest_score = -np.inf

        for iteration in range(max_iter):
            w = w_start - (w_start - w_end) * iteration / max_iter

            for i in range(n_particles):
                params = {param_names[j]: positions[i, j] for j in range(n_dims)}
                score = objective_fn(params)

                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()

            r1, r2 = np.random.random((n_particles, n_dims)), np.random.random((n_particles, n_dims))
            velocities = (w * velocities
                          + c1 * r1 * (pbest_positions - positions)
                          + c2 * r2 * (gbest_position - positions))
            positions = np.clip(positions + velocities, lower, upper)

        return {
            "best_params": {param_names[j]: float(gbest_position[j]) for j in range(n_dims)},
            "best_score": float(gbest_score),
            "iterations": max_iter,
        }

    def gradient_descent_risk_parity(
        self,
        returns: pd.DataFrame,
        target_risk_budget: Optional[np.ndarray] = None,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> OptimizationResult:
        """
        Risk parity via gradient descent.

        Objective: min sum_i (w_i*(Sigma*w)_i / (w'Sigma*w) - b_i)^2
        Equal risk contribution: b_i = 1/n for all i
        """
        n = returns.shape[1]
        sigma = returns.cov().values * 252
        mu = returns.mean().values * 252

        if target_risk_budget is None:
            target_risk_budget = np.ones(n) / n

        w = np.ones(n) / n
        history = []

        for k in range(max_iter):
            port_var = w @ sigma @ w
            if port_var < 1e-12:
                break
            marginal_risk = sigma @ w
            risk_contrib = w * marginal_risk / port_var

            # Gradient of objective
            residuals = risk_contrib - target_risk_budget
            loss = np.sum(residuals ** 2)
            history.append(loss)

            if loss < tol:
                break

            # Numerical gradient
            grad = np.zeros(n)
            eps = 1e-6
            for i in range(n):
                w_plus = w.copy()
                w_plus[i] += eps
                w_plus /= w_plus.sum()
                pv_plus = w_plus @ sigma @ w_plus
                rc_plus = w_plus * (sigma @ w_plus) / pv_plus
                loss_plus = np.sum((rc_plus - target_risk_budget) ** 2)
                grad[i] = (loss_plus - loss) / eps

            w -= lr * grad
            w = np.clip(w, 1e-6, None)
            w /= w.sum()

        symbols = returns.columns.tolist()
        port_ret = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        sharpe = (port_ret - 0.04) / port_vol if port_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.GRADIENT_DESCENT,
            weights={symbols[i]: float(w[i]) for i in range(n)},
            sharpe_ratio=float(sharpe),
            expected_return=float(port_ret),
            volatility=float(port_vol),
            max_drawdown=self._estimate_max_drawdown(returns, w),
            iterations=k + 1,
            convergence_history=history,
        )

    def _estimate_max_drawdown(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Estimate max drawdown from historical returns and weights."""
        port_returns = returns.values @ weights
        cum_returns = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        return float(np.min(drawdowns))
