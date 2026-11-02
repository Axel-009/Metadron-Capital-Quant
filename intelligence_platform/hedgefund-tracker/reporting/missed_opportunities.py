"""
Missed Opportunity Report for Hedge Fund Tracker.

Scans the investment universe for trades that could have been profitable
but were not taken. Provides regret analysis, pattern detection, and
opportunity cost quantification.

Mathematical Foundation:
    Opportunity Cost:
        OC_i = max_return_i * hypothetical_size_i
        where max_return_i = (P_peak - P_entry) / P_entry  (for longs)

    Regret Score:
        regret_i = OC_i * probability_of_detection_i * confidence_i
        Ranked descending to surface highest-regret misses.

    Pattern Detection:
        Clustering missed trades by:
            - Asset class
            - Signal type (momentum, mean-reversion, breakout)
            - Time of day / week / month
            - Market regime during miss

    Statistical Significance:
        z = (observed_miss_rate - expected_miss_rate) / sqrt(p*(1-p)/n)
        If z > 1.96, pattern is significant at 95% level.

Usage:
    from missed_opportunities import MissedOpportunityReport, MissedTrade
    import sys; sys.path.insert(0, "..")
    from openbb_universe import AssetClass
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import Counter

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openbb_universe import AssetClass, detect_asset_class, get_full_universe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MissedTrade:
    """Represents a trade that could have been taken but was not."""
    symbol: str
    asset_class: AssetClass
    signal_type: str  # "momentum", "mean_reversion", "breakout", "catalyst"
    signal_date: datetime
    entry_price: float
    peak_price: float
    exit_price: float  # where it ended up after the move
    potential_return: float  # (peak - entry) / entry
    potential_profit: float  # in dollar terms at hypothetical size
    confidence: float  # 0-1 signal confidence
    reason_missed: str
    signal_strength: float  # raw signal value
    holding_period_days: int = 0
    direction: str = "LONG"  # "LONG" or "SHORT"


@dataclass
class MissedPattern:
    """A recurring pattern in missed opportunities."""
    pattern_name: str
    description: str
    frequency: int  # how many times this pattern appeared
    avg_opportunity_cost: float
    total_opportunity_cost: float
    asset_classes_affected: list[str] = field(default_factory=list)
    signal_types: list[str] = field(default_factory=list)
    statistical_significance: float = 0.0  # z-score


# ---------------------------------------------------------------------------
# Missed Opportunity Report Engine
# ---------------------------------------------------------------------------

class MissedOpportunityReport:
    """
    Scans for and analyzes missed trading opportunities.

    Detects opportunities that existed in the universe but were not taken,
    quantifies the opportunity cost, and identifies recurring patterns in
    missed trades.
    """

    def __init__(
        self,
        min_return_threshold: float = 0.05,
        hypothetical_position_size: float = 100_000.0,
        momentum_lookback: int = 21,
        breakout_lookback: int = 20,
        mean_reversion_z_threshold: float = 2.0,
    ):
        """
        Parameters
        ----------
        min_return_threshold : float
            Minimum return to qualify as a "missed opportunity" (default 5%).
        hypothetical_position_size : float
            Hypothetical dollar size for opportunity cost calc.
        momentum_lookback : int
            Days for momentum signal detection.
        breakout_lookback : int
            Days for breakout detection.
        mean_reversion_z_threshold : float
            Z-score threshold for mean-reversion signals.
        """
        self.min_return_threshold = min_return_threshold
        self.hypothetical_size = hypothetical_position_size
        self.momentum_lookback = momentum_lookback
        self.breakout_lookback = breakout_lookback
        self.mr_z_threshold = mean_reversion_z_threshold

    def scan_what_we_missed(
        self,
        universe_data: dict[str, pd.DataFrame],
        date: str,
        current_positions: Optional[set[str]] = None,
        lookforward_days: int = 21,
    ) -> list[MissedTrade]:
        """
        Scan the universe for trades that generated signals but were not taken.

        For each symbol NOT in current_positions:
            1. Check for momentum signal on `date`:
               signal = (P_date / P_{date - lookback}) - 1
               If |signal| > threshold and subsequent return confirms: MISSED

            2. Check for breakout signal:
               If P_date > max(H, breakout_window) or P_date < min(L, breakout_window)
               and subsequent move continues: MISSED

            3. Check for mean-reversion signal:
               z = (P_date - MA(P, lookback)) / std(P, lookback)
               If |z| > threshold and price reverts: MISSED

        Parameters
        ----------
        universe_data : dict[str, pd.DataFrame]
            Symbol -> OHLCV DataFrame (must extend beyond `date`).
        date : str
            The date to check for missed signals (YYYY-MM-DD).
        current_positions : set[str], optional
            Symbols already held (not counted as missed).
        lookforward_days : int
            Days to look forward to measure the opportunity.

        Returns
        -------
        list[MissedTrade]
            List of missed trades, sorted by potential return descending.
        """
        if current_positions is None:
            current_positions = set()

        target_date = pd.Timestamp(date)
        missed_trades = []

        for symbol, df in universe_data.items():
            if symbol in current_positions:
                continue
            if df is None or df.empty or "Close" not in df.columns:
                continue

            # Ensure Date index or column for slicing
            close = df["Close"].dropna()
            if len(close) < self.momentum_lookback + lookforward_days + 5:
                continue

            # Find the index closest to target_date
            if hasattr(close.index, 'get_indexer'):
                try:
                    idx_pos = close.index.get_indexer([target_date], method="nearest")[0]
                except Exception:
                    idx_pos = len(close) - lookforward_days - 1
            else:
                idx_pos = len(close) - lookforward_days - 1

            if idx_pos < self.momentum_lookback or idx_pos + lookforward_days >= len(close):
                continue

            price_at_date = float(close.iloc[idx_pos])
            asset_class = detect_asset_class(symbol)

            # --- Momentum Signal ---
            price_lookback = float(close.iloc[idx_pos - self.momentum_lookback])
            momentum = (price_at_date / price_lookback) - 1.0 if price_lookback > 0 else 0.0

            if abs(momentum) > self.min_return_threshold * 0.5:
                # Look forward to see if momentum continued
                future_prices = close.iloc[idx_pos:idx_pos + lookforward_days + 1]
                if momentum > 0:
                    peak = float(future_prices.max())
                    potential_ret = (peak - price_at_date) / price_at_date
                    direction = "LONG"
                else:
                    trough = float(future_prices.min())
                    potential_ret = (price_at_date - trough) / price_at_date
                    direction = "SHORT"

                if potential_ret >= self.min_return_threshold:
                    missed_trades.append(MissedTrade(
                        symbol=symbol,
                        asset_class=asset_class,
                        signal_type="momentum",
                        signal_date=datetime(target_date.year, target_date.month, target_date.day)
                        if hasattr(target_date, "year") else datetime.now(),
                        entry_price=price_at_date,
                        peak_price=peak if direction == "LONG" else trough,
                        exit_price=float(future_prices.iloc[-1]),
                        potential_return=potential_ret,
                        potential_profit=potential_ret * self.hypothetical_size,
                        confidence=min(abs(momentum) / 0.20, 1.0),
                        reason_missed="Strong momentum signal not acted upon",
                        signal_strength=momentum,
                        holding_period_days=lookforward_days,
                        direction=direction,
                    ))

            # --- Breakout Signal ---
            if idx_pos >= self.breakout_lookback:
                lookback_high = float(close.iloc[idx_pos - self.breakout_lookback:idx_pos].max())
                lookback_low = float(close.iloc[idx_pos - self.breakout_lookback:idx_pos].min())

                future_prices = close.iloc[idx_pos:idx_pos + lookforward_days + 1]

                if price_at_date > lookback_high:
                    peak = float(future_prices.max())
                    potential_ret = (peak - price_at_date) / price_at_date
                    if potential_ret >= self.min_return_threshold:
                        missed_trades.append(MissedTrade(
                            symbol=symbol,
                            asset_class=asset_class,
                            signal_type="breakout",
                            signal_date=datetime(target_date.year, target_date.month, target_date.day)
                            if hasattr(target_date, "year") else datetime.now(),
                            entry_price=price_at_date,
                            peak_price=peak,
                            exit_price=float(future_prices.iloc[-1]),
                            potential_return=potential_ret,
                            potential_profit=potential_ret * self.hypothetical_size,
                            confidence=min(
                                (price_at_date - lookback_high) / (lookback_high * 0.01 + 1e-10), 1.0
                            ),
                            reason_missed="Upside breakout above Donchian channel",
                            signal_strength=(price_at_date - lookback_high) / lookback_high,
                            holding_period_days=lookforward_days,
                            direction="LONG",
                        ))

                elif price_at_date < lookback_low:
                    trough = float(future_prices.min())
                    potential_ret = (price_at_date - trough) / price_at_date
                    if potential_ret >= self.min_return_threshold:
                        missed_trades.append(MissedTrade(
                            symbol=symbol,
                            asset_class=asset_class,
                            signal_type="breakout",
                            signal_date=datetime(target_date.year, target_date.month, target_date.day)
                            if hasattr(target_date, "year") else datetime.now(),
                            entry_price=price_at_date,
                            peak_price=trough,
                            exit_price=float(future_prices.iloc[-1]),
                            potential_return=potential_ret,
                            potential_profit=potential_ret * self.hypothetical_size,
                            confidence=min(
                                (lookback_low - price_at_date) / (lookback_low * 0.01 + 1e-10), 1.0
                            ),
                            reason_missed="Downside breakout below Donchian channel",
                            signal_strength=(lookback_low - price_at_date) / lookback_low,
                            holding_period_days=lookforward_days,
                            direction="SHORT",
                        ))

            # --- Mean Reversion Signal ---
            if idx_pos >= self.momentum_lookback:
                lookback_prices = close.iloc[idx_pos - self.momentum_lookback:idx_pos]
                ma = float(lookback_prices.mean())
                std = float(lookback_prices.std())

                if std > 0:
                    z_score = (price_at_date - ma) / std

                    future_prices = close.iloc[idx_pos:idx_pos + lookforward_days + 1]

                    if z_score > self.mr_z_threshold:
                        # Overextended to upside -> expect reversion down (short opportunity)
                        trough = float(future_prices.min())
                        potential_ret = (price_at_date - trough) / price_at_date
                        if potential_ret >= self.min_return_threshold:
                            missed_trades.append(MissedTrade(
                                symbol=symbol,
                                asset_class=asset_class,
                                signal_type="mean_reversion",
                                signal_date=datetime(target_date.year, target_date.month, target_date.day)
                                if hasattr(target_date, "year") else datetime.now(),
                                entry_price=price_at_date,
                                peak_price=trough,
                                exit_price=float(future_prices.iloc[-1]),
                                potential_return=potential_ret,
                                potential_profit=potential_ret * self.hypothetical_size,
                                confidence=min(abs(z_score) / 4.0, 1.0),
                                reason_missed=f"Mean reversion: z-score={z_score:.2f} (overextended up)",
                                signal_strength=z_score,
                                holding_period_days=lookforward_days,
                                direction="SHORT",
                            ))

                    elif z_score < -self.mr_z_threshold:
                        # Overextended to downside -> expect reversion up (long opportunity)
                        peak = float(future_prices.max())
                        potential_ret = (peak - price_at_date) / price_at_date
                        if potential_ret >= self.min_return_threshold:
                            missed_trades.append(MissedTrade(
                                symbol=symbol,
                                asset_class=asset_class,
                                signal_type="mean_reversion",
                                signal_date=datetime(target_date.year, target_date.month, target_date.day)
                                if hasattr(target_date, "year") else datetime.now(),
                                entry_price=price_at_date,
                                peak_price=peak,
                                exit_price=float(future_prices.iloc[-1]),
                                potential_return=potential_ret,
                                potential_profit=potential_ret * self.hypothetical_size,
                                confidence=min(abs(z_score) / 4.0, 1.0),
                                reason_missed=f"Mean reversion: z-score={z_score:.2f} (overextended down)",
                                signal_strength=z_score,
                                holding_period_days=lookforward_days,
                                direction="LONG",
                            ))

        # Sort by potential return descending
        missed_trades.sort(key=lambda t: t.potential_return, reverse=True)
        return missed_trades

    def calculate_opportunity_cost(self, missed_trades: list[MissedTrade]) -> float:
        """
        Calculate total opportunity cost of missed trades.

        Opportunity Cost:
            OC = sum_i(potential_profit_i)
            where potential_profit_i = potential_return_i * position_size

        Weighted by confidence:
            OC_weighted = sum_i(potential_profit_i * confidence_i)

        Parameters
        ----------
        missed_trades : list[MissedTrade]
            Missed trades from scan_what_we_missed.

        Returns
        -------
        float
            Total opportunity cost (confidence-weighted).
        """
        if not missed_trades:
            return 0.0

        return sum(t.potential_profit * t.confidence for t in missed_trades)

    def rank_by_regret(self, missed_trades: list[MissedTrade]) -> list[MissedTrade]:
        """
        Rank missed trades by regret score.

        Regret Score:
            regret_i = potential_profit_i * confidence_i * (1 / holding_period_i)

        Rationale: Higher profit, higher confidence, and shorter holding
        period (easier to capture) = more regret.

        Parameters
        ----------
        missed_trades : list[MissedTrade]

        Returns
        -------
        list[MissedTrade]
            Sorted by regret score descending.
        """
        def regret_score(trade: MissedTrade) -> float:
            hp = max(trade.holding_period_days, 1)
            return trade.potential_profit * trade.confidence * (1.0 / hp)

        return sorted(missed_trades, key=regret_score, reverse=True)

    def pattern_analysis(
        self, historical_misses: list[MissedTrade]
    ) -> list[MissedPattern]:
        """
        Analyze patterns in historical missed opportunities.

        Detects recurring patterns by:
            1. Asset class concentration:
               Are we consistently missing opportunities in specific asset classes?

            2. Signal type blind spots:
               Are we missing momentum trades more than mean-reversion?

            3. Direction bias:
               Are we missing more longs or shorts?

            4. Size correlation:
               Are we missing the biggest moves?

        Statistical test for each pattern:
            H0: miss rate for pattern = overall miss rate
            z = (p_pattern - p_overall) / sqrt(p_overall * (1-p_overall) / n)
            Significant if |z| > 1.96 (95% confidence)

        Parameters
        ----------
        historical_misses : list[MissedTrade]
            All historical missed trades.

        Returns
        -------
        list[MissedPattern]
            Detected patterns, sorted by total opportunity cost.
        """
        if not historical_misses:
            return []

        patterns = []
        total_misses = len(historical_misses)
        total_oc = sum(t.potential_profit for t in historical_misses)

        # --- Pattern 1: Asset Class Concentration ---
        ac_counts = Counter(t.asset_class.value for t in historical_misses)
        ac_costs = {}
        for t in historical_misses:
            ac_key = t.asset_class.value
            ac_costs[ac_key] = ac_costs.get(ac_key, 0.0) + t.potential_profit

        expected_rate = 1.0 / max(len(ac_counts), 1)
        for ac, count in ac_counts.most_common():
            observed_rate = count / total_misses
            n = total_misses
            if n > 0 and expected_rate > 0 and expected_rate < 1:
                z = (observed_rate - expected_rate) / np.sqrt(
                    expected_rate * (1 - expected_rate) / n
                )
            else:
                z = 0.0

            if abs(z) > 1.0:  # Relaxed threshold for pattern detection
                patterns.append(MissedPattern(
                    pattern_name=f"Asset Class Concentration: {ac}",
                    description=(
                        f"Disproportionately missing {ac} opportunities. "
                        f"Observed {count}/{total_misses} ({observed_rate:.1%}) vs "
                        f"expected {expected_rate:.1%}."
                    ),
                    frequency=count,
                    avg_opportunity_cost=ac_costs.get(ac, 0.0) / max(count, 1),
                    total_opportunity_cost=ac_costs.get(ac, 0.0),
                    asset_classes_affected=[ac],
                    signal_types=list(set(
                        t.signal_type for t in historical_misses
                        if t.asset_class.value == ac
                    )),
                    statistical_significance=z,
                ))

        # --- Pattern 2: Signal Type Blind Spots ---
        sig_counts = Counter(t.signal_type for t in historical_misses)
        sig_costs = {}
        for t in historical_misses:
            sig_costs[t.signal_type] = sig_costs.get(t.signal_type, 0.0) + t.potential_profit

        for sig_type, count in sig_counts.most_common():
            observed_rate = count / total_misses
            expected_rate_sig = 1.0 / max(len(sig_counts), 1)
            n = total_misses
            if n > 0 and expected_rate_sig > 0 and expected_rate_sig < 1:
                z = (observed_rate - expected_rate_sig) / np.sqrt(
                    expected_rate_sig * (1 - expected_rate_sig) / n
                )
            else:
                z = 0.0

            if abs(z) > 1.0:
                patterns.append(MissedPattern(
                    pattern_name=f"Signal Blind Spot: {sig_type}",
                    description=(
                        f"Consistently missing {sig_type} signals. "
                        f"{count}/{total_misses} misses ({observed_rate:.1%})."
                    ),
                    frequency=count,
                    avg_opportunity_cost=sig_costs.get(sig_type, 0.0) / max(count, 1),
                    total_opportunity_cost=sig_costs.get(sig_type, 0.0),
                    asset_classes_affected=list(set(
                        t.asset_class.value for t in historical_misses
                        if t.signal_type == sig_type
                    )),
                    signal_types=[sig_type],
                    statistical_significance=z,
                ))

        # --- Pattern 3: Direction Bias ---
        long_misses = [t for t in historical_misses if t.direction == "LONG"]
        short_misses = [t for t in historical_misses if t.direction == "SHORT"]

        if total_misses > 10:
            long_rate = len(long_misses) / total_misses
            # Test vs 50% expected
            z = (long_rate - 0.5) / np.sqrt(0.25 / total_misses)
            if abs(z) > 1.0:
                dominant = "LONG" if long_rate > 0.5 else "SHORT"
                dominant_list = long_misses if dominant == "LONG" else short_misses
                oc = sum(t.potential_profit for t in dominant_list)
                patterns.append(MissedPattern(
                    pattern_name=f"Direction Bias: Missing {dominant}s",
                    description=(
                        f"Bias toward missing {dominant} trades. "
                        f"Long misses: {len(long_misses)}, Short misses: {len(short_misses)}."
                    ),
                    frequency=len(dominant_list),
                    avg_opportunity_cost=oc / max(len(dominant_list), 1),
                    total_opportunity_cost=oc,
                    asset_classes_affected=list(set(
                        t.asset_class.value for t in dominant_list
                    )),
                    signal_types=list(set(t.signal_type for t in dominant_list)),
                    statistical_significance=z,
                ))

        # --- Pattern 4: Magnitude Blind Spot ---
        returns = [t.potential_return for t in historical_misses]
        if returns:
            median_ret = np.median(returns)
            large_movers = [t for t in historical_misses if t.potential_return > median_ret * 2]
            if len(large_movers) > 3:
                oc = sum(t.potential_profit for t in large_movers)
                patterns.append(MissedPattern(
                    pattern_name="Missing Large Moves",
                    description=(
                        f"Consistently missing the biggest opportunities. "
                        f"{len(large_movers)} trades had return > {median_ret*2:.1%} "
                        f"(2x median of {median_ret:.1%})."
                    ),
                    frequency=len(large_movers),
                    avg_opportunity_cost=oc / max(len(large_movers), 1),
                    total_opportunity_cost=oc,
                    asset_classes_affected=list(set(
                        t.asset_class.value for t in large_movers
                    )),
                    signal_types=list(set(t.signal_type for t in large_movers)),
                    statistical_significance=0.0,
                ))

        # Sort by total opportunity cost
        patterns.sort(key=lambda p: p.total_opportunity_cost, reverse=True)
        return patterns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scanner = MissedOpportunityReport(min_return_threshold=0.03)

    # Synthetic demo
    np.random.seed(42)
    n_days = 100
    symbols = ["AAPL", "NVDA", "GC=F", "BTC-USD", "EURUSD=X", "TLT"]
    universe_data = {}
    for sym in symbols:
        base = {"AAPL": 180, "NVDA": 800, "GC=F": 2000, "BTC-USD": 60000, "EURUSD=X": 1.08, "TLT": 100}[sym]
        vol = {"AAPL": 0.02, "NVDA": 0.03, "GC=F": 0.01, "BTC-USD": 0.04, "EURUSD=X": 0.005, "TLT": 0.008}[sym]
        closes = base * np.exp(np.cumsum(np.random.normal(0.001, vol, n_days)))
        universe_data[sym] = pd.DataFrame({
            "Open": closes * 0.999,
            "High": closes * (1 + np.abs(np.random.normal(0, vol/2, n_days))),
            "Low": closes * (1 - np.abs(np.random.normal(0, vol/2, n_days))),
            "Close": closes,
            "Volume": np.random.randint(1e6, 1e8, n_days),
        })

    print("=== Scanning for Missed Opportunities ===")
    # Use index position ~60% through the data
    missed = scanner.scan_what_we_missed(
        universe_data, date="2024-01-15", current_positions={"AAPL"}
    )
    print(f"  Found {len(missed)} missed trades")

    if missed:
        print("\n=== Top 5 by Potential Return ===")
        for t in missed[:5]:
            print(f"  {t.symbol} ({t.signal_type}): {t.potential_return:.2%} "
                  f"(${t.potential_profit:,.0f}) confidence={t.confidence:.2f}")

        print(f"\n=== Total Opportunity Cost ===")
        oc = scanner.calculate_opportunity_cost(missed)
        print(f"  ${oc:,.0f}")

        print("\n=== Ranked by Regret ===")
        ranked = scanner.rank_by_regret(missed)
        for t in ranked[:5]:
            print(f"  {t.symbol}: {t.signal_type}, return={t.potential_return:.2%}, "
                  f"confidence={t.confidence:.2f}")

        print("\n=== Pattern Analysis ===")
        patterns = scanner.pattern_analysis(missed)
        for p in patterns[:5]:
            print(f"  {p.pattern_name}: freq={p.frequency}, "
                  f"OC=${p.total_opportunity_cost:,.0f}, z={p.statistical_significance:.2f}")
            print(f"    {p.description}")
