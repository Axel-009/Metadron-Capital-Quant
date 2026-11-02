"""Hedgefund Tracker Bridge — Institutional Flow Signals → Engine.

Integrates the hedgefund-tracker sub-repo (SEC 13F/13D/Form 4 analysis)
with the Metadron Capital signal pipeline.  Provides institutional
consensus signals, smart-money positioning, and high-conviction flags.

The bridge exposes:
    1. Quarterly institutional consensus (net buyers, concentration)
    2. High-conviction signal extraction (multiple elite funds accumulating)
    3. Promise Score integration for ML feature engineering

Source repo: intelligence_platform/hedgefund-tracker
Key modules:
    - app/analysis/stocks.py        → quarter_analysis(), stock_analysis()
    - app/ai/agent.py               → AnalystAgent (Promise Score)
    - app/analysis/performance_evaluator.py → PerformanceEvaluator

Usage:
    from engine.ml.bridges.hedgefund_tracker_bridge import HedgefundTrackerBridge
    bridge = HedgefundTrackerBridge()
    signals = bridge.get_institutional_signals("AAPL")
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intelligence Platform: hedgefund-tracker integration
# Provides: SEC 13F/13D analysis, Promise Score, institutional flow signals.
# ---------------------------------------------------------------------------
TRACKER_BASE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "intelligence_platform" / "hedgefund-tracker"
)

try:
    import importlib.util as _ilu

    # --- stocks.py: quarter_analysis, stock_analysis -----------------------
    _stocks_spec = _ilu.spec_from_file_location(
        "hedgefund_tracker_stocks",
        str(TRACKER_BASE / "app" / "analysis" / "stocks.py"),
    )
    _stocks_mod = _ilu.module_from_spec(_stocks_spec)
    _stocks_spec.loader.exec_module(_stocks_mod)

    quarter_analysis = _stocks_mod.quarter_analysis
    stock_analysis = _stocks_mod.stock_analysis
    TRACKER_STOCKS_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    quarter_analysis = None
    stock_analysis = None
    TRACKER_STOCKS_AVAILABLE = False
    logger.info("hedgefund-tracker stocks module unavailable — degraded mode")

try:
    import importlib.util as _ilu

    # --- performance_evaluator.py: PerformanceEvaluator --------------------
    _perf_spec = _ilu.spec_from_file_location(
        "hedgefund_tracker_perf",
        str(TRACKER_BASE / "app" / "analysis" / "performance_evaluator.py"),
    )
    _perf_mod = _ilu.module_from_spec(_perf_spec)
    _perf_spec.loader.exec_module(_perf_mod)

    PerformanceEvaluator = _perf_mod.PerformanceEvaluator
    TRACKER_PERF_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    PerformanceEvaluator = None
    TRACKER_PERF_AVAILABLE = False
    logger.info("hedgefund-tracker PerformanceEvaluator unavailable")

try:
    from engine.execution.paper_broker import SignalType
except Exception:
    class SignalType:
        ML_AGENT_BUY = "ML_AGENT_BUY"
        ML_AGENT_SELL = "ML_AGENT_SELL"
        HOLD = "HOLD"


HEDGEFUND_TRACKER_AVAILABLE = TRACKER_STOCKS_AVAILABLE or TRACKER_PERF_AVAILABLE


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class InstitutionalSignal:
    """Institutional flow signal for a single ticker."""
    ticker: str = ""
    net_buyers: int = 0               # Funds buying minus selling
    buyer_count: int = 0
    seller_count: int = 0
    high_conviction_count: int = 0    # Funds with >3% allocation
    ownership_delta: float = 0.0      # Average change in ownership %
    portfolio_concentration: float = 0.0
    signal_type: str = "HOLD"
    confidence: float = 0.5
    promise_score: float = 0.0        # LLM-weighted composite (0-100)


@dataclass
class InstitutionalConsensus:
    """Aggregate institutional consensus for a quarter."""
    quarter: str = ""
    total_tickers: int = 0
    strong_buys: List[str] = field(default_factory=list)
    strong_sells: List[str] = field(default_factory=list)
    signals: Dict[str, InstitutionalSignal] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class HedgefundTrackerBridge:
    """Bridge adapter connecting hedgefund-tracker to Metadron engine.

    Converts SEC filing analysis (13F/13D/Form 4) into actionable
    institutional flow signals.  Falls back to empty/neutral signals
    when the tracker modules are not available.
    """

    # Thresholds for signal classification
    STRONG_BUY_NET_BUYERS = 3
    STRONG_SELL_NET_BUYERS = -3
    HIGH_CONVICTION_THRESHOLD = 2  # minimum elite funds for conviction boost

    def __init__(self):
        self._perf_evaluator = None
        if TRACKER_PERF_AVAILABLE and PerformanceEvaluator is not None:
            try:
                self._perf_evaluator = PerformanceEvaluator()
            except Exception as exc:
                logger.warning("PerformanceEvaluator init failed: %s", exc)

    # ------------------------------------------------------------------
    # Single-ticker signal
    # ------------------------------------------------------------------

    def get_institutional_signal(
        self, ticker: str, quarter: Optional[str] = None
    ) -> InstitutionalSignal:
        """Get institutional flow signal for a single ticker.

        Returns an InstitutionalSignal with consensus data from 13F
        filings.  Uses latest available quarter if *quarter* is None.
        """
        sig = InstitutionalSignal(ticker=ticker)

        if not TRACKER_STOCKS_AVAILABLE or stock_analysis is None:
            return sig

        try:
            import pandas as pd
            df = stock_analysis(ticker, quarter)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return sig

            # Extract consensus metrics
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                row = df.iloc[0] if len(df) == 1 else df.agg("sum")
                sig.buyer_count = int(row.get("Buyer_Count", 0))
                sig.seller_count = int(row.get("Seller_Count", 0))
                sig.net_buyers = sig.buyer_count - sig.seller_count
                sig.high_conviction_count = int(
                    row.get("High_Conviction_Count", 0)
                )
                sig.ownership_delta = float(
                    row.get("Ownership_Delta_Avg", 0.0)
                )
                sig.portfolio_concentration = float(
                    row.get("Portfolio_Concentration_Avg", 0.0)
                )

            # Classify signal
            sig = self._classify_signal(sig)

        except Exception as exc:
            logger.warning("Institutional signal for %s failed: %s", ticker, exc)

        return sig

    # ------------------------------------------------------------------
    # Quarter consensus
    # ------------------------------------------------------------------

    def get_quarterly_consensus(
        self, quarter: Optional[str] = None
    ) -> InstitutionalConsensus:
        """Get full quarterly institutional consensus.

        Processes all 13F filings for the quarter and produces
        aggregated buy/sell signals per ticker.
        """
        consensus = InstitutionalConsensus(quarter=quarter or "latest")

        if not TRACKER_STOCKS_AVAILABLE or quarter_analysis is None:
            return consensus

        try:
            import pandas as pd
            df = quarter_analysis(quarter)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return consensus

            consensus.total_tickers = len(df)

            for _, row in df.iterrows():
                ticker = str(row.get("Ticker", ""))
                if not ticker:
                    continue

                sig = InstitutionalSignal(
                    ticker=ticker,
                    buyer_count=int(row.get("Buyer_Count", 0)),
                    seller_count=int(row.get("Seller_Count", 0)),
                    high_conviction_count=int(
                        row.get("High_Conviction_Count", 0)
                    ),
                    ownership_delta=float(
                        row.get("Ownership_Delta_Avg", 0.0)
                    ),
                    portfolio_concentration=float(
                        row.get("Portfolio_Concentration_Avg", 0.0)
                    ),
                )
                sig.net_buyers = sig.buyer_count - sig.seller_count
                sig = self._classify_signal(sig)
                consensus.signals[ticker] = sig

                if sig.signal_type == str(SignalType.ML_AGENT_BUY):
                    consensus.strong_buys.append(ticker)
                elif sig.signal_type == str(SignalType.ML_AGENT_SELL):
                    consensus.strong_sells.append(ticker)

        except Exception as exc:
            logger.warning("Quarterly consensus failed: %s", exc)

        return consensus

    # ------------------------------------------------------------------
    # Growth score
    # ------------------------------------------------------------------

    def get_growth_score(self, pct_change: float) -> int:
        """Compute growth score (1-100) from price percentage change."""
        if self._perf_evaluator is not None:
            try:
                return self._perf_evaluator.calculate_growth_score(pct_change)
            except Exception:
                pass
        # Numpy fallback: linear mapping capped at 100
        return max(1, min(100, int(50 + pct_change * 2)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_signal(self, sig: InstitutionalSignal) -> InstitutionalSignal:
        """Classify signal type from institutional consensus metrics."""
        if sig.net_buyers >= self.STRONG_BUY_NET_BUYERS:
            sig.signal_type = str(SignalType.ML_AGENT_BUY)
            sig.confidence = min(
                1.0,
                0.5 + sig.net_buyers * 0.05
                + sig.high_conviction_count * 0.1,
            )
        elif sig.net_buyers <= self.STRONG_SELL_NET_BUYERS:
            sig.signal_type = str(SignalType.ML_AGENT_SELL)
            sig.confidence = min(
                1.0,
                0.5 + abs(sig.net_buyers) * 0.05
                + sig.high_conviction_count * 0.05,
            )
        else:
            sig.signal_type = str(SignalType.HOLD)
            sig.confidence = 0.5

        return sig
