"""PatternDiscoveryEngine — L2 Pattern Discovery Stage.

Orchestrates MiroFish (dual simulation) + AI-Newton (symbolic regression)
to discover patterns in the classified universe BEFORE alpha optimization.

Pipeline position:
    L1 Universe (OpenBB) → classified instruments
        ↓
    **L2 Pattern Discovery (Stage 3.2)** ← THIS MODULE
        ├─ MiroFish Dual Simulation
        │   ├─ Mode A: Synthetic Market (emergent clustering, herding, regime shifts)
        │   └─ Mode B: Contagion Network (hidden correlations, divergences, contagion paths)
        ├─ AI-Newton Symbolic Regression
        │   ├─ Conservation laws (ratio stability, power laws)
        │   ├─ Lead-lag relationships
        │   └─ Fair value formulas
        └─ PatternDiscoveryBus → structured output for L3 Alpha
        ↓
    L3 Alpha (QLIB, AlphaOptimizer) — now enriched with discovered patterns

The engine connects to backends in Installation-Back-end-Files via bridges.
"""

import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("metadron.signals.pattern_discovery")

# Locate backend installation
_BACKEND_PATH = Path(__file__).parent.parent.parent.parent / "Installation-Back-end-Files"
if _BACKEND_PATH.exists() and str(_BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PATH))

# Import backends via bridges — guarded per design rule #8
# try/except on ALL external imports — system runs degraded, never broken
try:
    from bridges.mirofish_bridge import MiroFishDualSimulation, DiscoveredPattern
except ImportError:
    MiroFishDualSimulation = None  # type: ignore[assignment,misc]
    DiscoveredPattern = None  # type: ignore[assignment,misc]
    logger.debug("MiroFish bridge not available — pattern discovery will run degraded")

try:
    from bridges.newton_bridge import AINewtonEngine, DiscoveredLaw
except ImportError:
    AINewtonEngine = None  # type: ignore[assignment,misc]
    DiscoveredLaw = None  # type: ignore[assignment,misc]
    logger.debug("AI-Newton bridge not available — symbolic regression disabled")

try:
    from bridges.openbb_bridge import OpenBBBackend
except ImportError:
    OpenBBBackend = None  # type: ignore[assignment,misc]
    logger.debug("OpenBB bridge not available — using direct OpenBB imports")


# ---------------------------------------------------------------------------
# PatternDiscoveryBus — unified output structure
# ---------------------------------------------------------------------------

@dataclass
class DiscoverySignal:
    """Unified signal from any discovery engine, ready for L3 Alpha consumption."""
    source: str              # "mirofish" | "ai_newton" | "mavrock_hlm"
    signal_type: str         # "clustering" | "herding" | "contagion" | "conservation" | "lead_lag" | ...
    tickers: list[str] = field(default_factory=list)
    direction: int = 0       # -1, 0, +1
    strength: float = 0.0    # [0, 1]
    confidence: float = 0.0  # [0, 1]
    formula: str = ""        # AI-Newton symbolic formula (if applicable)
    half_life_days: int = 5
    description: str = ""
    metadata: dict = field(default_factory=dict)


class PatternDiscoveryBus:
    """Collects and aggregates signals from all discovery engines."""

    def __init__(self):
        self._signals: list[DiscoverySignal] = []

    def add_mirofish_patterns(self, patterns: list) -> int:
        """Convert MiroFish DiscoveredPattern to DiscoverySignal."""
        count = 0
        for p in patterns:
            self._signals.append(DiscoverySignal(
                source="mirofish",
                signal_type=p.pattern_type,
                tickers=p.tickers,
                direction=p.direction,
                strength=p.strength,
                confidence=p.confidence,
                half_life_days=p.half_life_days,
                description=p.description,
                metadata=p.metadata,
            ))
            count += 1
        return count

    def add_newton_laws(self, laws: list) -> int:
        """Convert AI-Newton DiscoveredLaw to DiscoverySignal."""
        count = 0
        for law in laws:
            self._signals.append(DiscoverySignal(
                source="ai_newton",
                signal_type=law.law_type,
                tickers=law.tickers,
                direction=law.direction,
                strength=law.strength,
                confidence=law.confidence,
                formula=law.formula,
                description=law.description,
                metadata=law.metadata,
            ))
            count += 1
        return count

    def get_all(self) -> list[DiscoverySignal]:
        """Get all discovery signals sorted by confidence * strength."""
        return sorted(self._signals, key=lambda s: s.confidence * s.strength, reverse=True)

    def get_for_ticker(self, ticker: str) -> list[DiscoverySignal]:
        """Get all discovery signals involving a specific ticker."""
        return [s for s in self._signals if ticker in s.tickers]

    def get_actionable(self, min_confidence: float = 0.5) -> list[DiscoverySignal]:
        """Get signals with trading implications."""
        return [s for s in self._signals
                if s.confidence >= min_confidence and s.direction != 0]

    def get_features_for_alpha(self, tickers: list[str]) -> dict[str, dict]:
        """Convert discovery signals into feature dict for AlphaOptimizer.

        Returns dict of ticker -> feature_dict.
        """
        features = {}
        for ticker in tickers:
            ticker_signals = self.get_for_ticker(ticker)
            if not ticker_signals:
                features[ticker] = {
                    "discovery_direction": 0,
                    "discovery_strength": 0.0,
                    "discovery_count": 0,
                    "mirofish_signal": 0.0,
                    "newton_signal": 0.0,
                }
                continue

            miro_sigs = [s for s in ticker_signals if s.source == "mirofish"]
            newton_sigs = [s for s in ticker_signals if s.source == "ai_newton"]

            # Aggregate direction
            all_dirs = [s.direction * s.confidence for s in ticker_signals if s.direction != 0]
            avg_direction = np.sign(sum(all_dirs)) if all_dirs else 0

            features[ticker] = {
                "discovery_direction": int(avg_direction),
                "discovery_strength": max((s.strength for s in ticker_signals), default=0.0),
                "discovery_count": len(ticker_signals),
                "mirofish_signal": sum(s.direction * s.confidence for s in miro_sigs) / max(len(miro_sigs), 1),
                "newton_signal": sum(s.direction * s.confidence for s in newton_sigs) / max(len(newton_sigs), 1),
                "has_conservation_law": any(s.signal_type == "conservation" for s in newton_sigs),
                "has_lead_lag": any(s.signal_type == "lead_lag" for s in newton_sigs),
                "has_contagion_risk": any(s.signal_type == "contagion" for s in miro_sigs),
                "has_herding": any(s.signal_type == "herding" for s in miro_sigs),
                "has_divergence": any(s.signal_type == "divergence" for s in miro_sigs),
            }

        return features

    def as_dict(self) -> dict:
        """Summary dict for pipeline reporting."""
        by_source = {}
        for s in self._signals:
            by_source.setdefault(s.source, []).append(s)

        return {
            "total_signals": len(self._signals),
            "by_source": {src: len(sigs) for src, sigs in by_source.items()},
            "actionable": len(self.get_actionable()),
            "top_signals": [
                {
                    "source": s.source,
                    "type": s.signal_type,
                    "tickers": s.tickers[:3],
                    "direction": s.direction,
                    "strength": round(s.strength, 3),
                    "confidence": round(s.confidence, 3),
                    "description": s.description[:80],
                }
                for s in self.get_all()[:10]
            ],
        }

    def clear(self):
        self._signals.clear()


# ---------------------------------------------------------------------------
# PatternDiscoveryEngine — main orchestrator
# ---------------------------------------------------------------------------

class PatternDiscoveryEngine:
    """L2 Pattern Discovery Engine.

    Runs MiroFish + AI-Newton on the classified universe and
    feeds structured patterns into the PatternDiscoveryBus for L3.
    """

    def __init__(self):
        self.bus = PatternDiscoveryBus()

        # Initialize backends
        self.mirofish: Optional[MiroFishDualSimulation] = None
        try:
            self.mirofish = MiroFishDualSimulation()
        except Exception as e:
            logger.warning(f"MiroFish init failed: {e}")

        self.newton: Optional[AINewtonEngine] = None
        try:
            self.newton = AINewtonEngine()
        except Exception as e:
            logger.warning(f"AI-Newton init failed: {e}")

        self.openbb: Optional[OpenBBBackend] = None
        try:
            self.openbb = OpenBBBackend()
        except Exception as e:
            logger.warning(f"OpenBB backend init failed: {e}")

        logger.info(f"PatternDiscoveryEngine initialized: "
                    f"MiroFish={'ON' if self.mirofish else 'OFF'} "
                    f"Newton={'ON' if self.newton else 'OFF'} "
                    f"OpenBB={'ON' if self.openbb else 'OFF'}")

    def discover(self, prices: dict[str, pd.Series],
                  fundamentals: Optional[dict[str, dict]] = None) -> PatternDiscoveryBus:
        """Run full pattern discovery on universe data.

        Args:
            prices: Dict of ticker -> close prices.
            fundamentals: Optional dict of ticker -> fundamental data.

        Returns:
            PatternDiscoveryBus with all discovered patterns.
        """
        self.bus.clear()

        # --- MiroFish Dual Simulation ---
        if self.mirofish:
            try:
                logger.info("Running MiroFish dual simulation...")
                patterns = self.mirofish.run_dual(prices)
                n = self.bus.add_mirofish_patterns(patterns)
                logger.info(f"MiroFish produced {n} patterns")
            except Exception as e:
                logger.warning(f"MiroFish discovery failed: {e}")

        # --- AI-Newton Symbolic Regression ---
        if self.newton:
            try:
                logger.info("Running AI-Newton symbolic regression...")
                laws = self.newton.discover(prices, fundamentals)
                n = self.bus.add_newton_laws(laws)
                logger.info(f"AI-Newton discovered {n} laws")
            except Exception as e:
                logger.warning(f"AI-Newton discovery failed: {e}")

        logger.info(f"Pattern Discovery complete: {self.bus.as_dict()}")
        return self.bus

    def get_alpha_features(self, tickers: list[str]) -> dict[str, dict]:
        """Get discovery-based features for AlphaOptimizer enrichment."""
        return self.bus.get_features_for_alpha(tickers)
