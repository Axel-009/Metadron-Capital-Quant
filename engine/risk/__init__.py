"""
Engine Risk Module — Monte Carlo risk analysis.

Components:
    - MonteCarloRiskEngine: Per-ticker and portfolio-level VaR, CVaR, stress testing (research)
    - PortfolioMCGate: shared pre-trade Monte Carlo gate (decision matrix + L7 execution)
"""

from .monte_carlo_risk import MonteCarloRiskEngine, TickerRisk, PortfolioRisk
from .portfolio_mc_gate import (
    PortfolioMCGate, Holding, MCResult, get_shared_mc_gate,
    MC_VAR95_MAX, MC_CVAR95_MAX, MC_PROB_PROFIT_MIN,
)

__all__ = [
    "MonteCarloRiskEngine", "TickerRisk", "PortfolioRisk",
    "PortfolioMCGate", "Holding", "MCResult", "get_shared_mc_gate",
    "MC_VAR95_MAX", "MC_CVAR95_MAX", "MC_PROB_PROFIT_MIN",
]
