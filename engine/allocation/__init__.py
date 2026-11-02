"""Allocation Engine — Portfolio allocation rules, kill switch, beta corridor gating, and full universe scan.

Implements the Metadron Capital allocation framework:
- AllocationRules: bucket constraints (IG 30%, HY 20%, ETF 15%, FI+Macro 10%, Options 25% notional, Cash 5%)
- BetaCorridorEngine: HIGH / NEUTRAL / LOW corridor with leverage multiplier
- KillSwitchMonitor: 20% drawdown kill switch
- AllocationEngine: classify, size, aggregate, validate
- FullUniverseScan: 4-universe 150s heartbeat orchestrator (3 cycles/hour)
"""
