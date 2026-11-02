"""
Trade Thesis Generator for Hedge Fund Tracker.

Generates medium-term and long-term investment theses based on macro data,
sector analysis, catalysts, and structural changes. Includes second-derivative
analysis (chain-reaction effects), IPO screening, and distress screening.

Mathematical Foundation:
    Second Derivative Analysis:
        If event E occurs with probability p:
            P(secondary_effect | E) = p * conditional_probability
            Impact chain: E -> S1 -> S2 -> ... -> Sn
            Cumulative impact = product(transmission_rate_i) * initial_impact

    DCF-Based Fair Value:
        V = sum_{t=1}^{T} FCF_t / (1 + WACC)^t + TV / (1 + WACC)^T
        TV = FCF_{T+1} / (WACC - g)   (Gordon growth terminal value)

    Macro Catalyst Scoring:
        score = impact_magnitude * probability * (1 / time_to_event)
        where time_to_event is in trading days

    IPO Valuation:
        Relative: EV/Revenue vs peer group
        Fair value band: [peer_low * revenue, peer_high * revenue]

    Distress Probability (Altman Z-Score):
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Cap / Total Liabilities
        X5 = Sales / Total Assets
        Z < 1.81: Distress zone, Z > 2.99: Safe zone

Usage:
    from thesis_generator import ThesisGenerator, TradeThesis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openbb_universe import AssetClass, detect_asset_class, EQUITY_GICS_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class ThesisTimeframe(Enum):
    """Investment thesis timeframe."""
    SHORT_TERM = "short_term"       # 1-4 weeks
    MEDIUM_TERM = "medium_term"     # 1-6 months
    LONG_TERM = "long_term"         # 6-24 months
    STRUCTURAL = "structural"       # 2-10 years


class ConvictionLevel(Enum):
    """Thesis conviction level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TradeThesis:
    """A complete investment thesis."""
    title: str
    asset_class: AssetClass
    symbols: list[str]
    direction: str  # "LONG" or "SHORT"
    timeframe: ThesisTimeframe
    conviction: ConvictionLevel
    summary: str
    rationale: list[str]
    catalysts: list[str]
    risks: list[str]
    target_return: float
    stop_loss: float
    position_size_pct: float  # of portfolio
    entry_conditions: list[str]
    exit_conditions: list[str]
    second_order_effects: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    score: float = 0.0  # composite thesis quality score


@dataclass
class MacroCatalyst:
    """A macroeconomic catalyst event."""
    event_name: str
    event_type: str  # "rate_decision", "election", "earnings", "data_release"
    expected_date: datetime
    impact_direction: int  # -1, 0, or 1
    impact_magnitude: float  # 0 to 1
    probability: float  # 0 to 1
    affected_asset_classes: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)
    second_order_effects: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SecondaryEffect:
    """A second-derivative (chain reaction) effect from a primary event."""
    primary_event: str
    secondary_effect: str
    transmission_mechanism: str
    affected_symbols: list[str]
    estimated_impact: float  # -1 to 1 (negative = bearish, positive = bullish)
    confidence: float  # 0 to 1
    time_lag_days: int  # how long after primary event
    chain_depth: int  # 1 = direct, 2 = second order, etc.


@dataclass
class IPOThesis:
    """IPO investment thesis."""
    company_name: str
    symbol: str
    expected_date: datetime
    sector: str
    ipo_price_low: float
    ipo_price_high: float
    peer_ev_revenue_range: tuple[float, float]
    estimated_revenue: float
    fair_value_range: tuple[float, float]
    recommendation: str  # "subscribe", "avoid", "wait_for_lockup"
    rationale: list[str]
    risks: list[str]
    score: float = 0.0


@dataclass
class DistressThesis:
    """Distressed investment thesis."""
    symbol: str
    company_name: str
    altman_z_score: float
    distress_probability: float
    current_price: float
    estimated_recovery_value: float
    potential_return: float  # if recovery
    thesis_type: str  # "long_recovery", "short_bankruptcy", "credit_arb"
    rationale: list[str]
    risks: list[str]
    score: float = 0.0


# ---------------------------------------------------------------------------
# Second Derivative Analysis Engine
# ---------------------------------------------------------------------------

# Pre-defined causal chains for common macro events
CAUSAL_CHAINS: dict[str, list[dict[str, Any]]] = {
    "rate_hike": [
        {
            "effect": "Higher mortgage rates",
            "mechanism": "Direct pass-through to lending rates",
            "symbols": ["XHB", "ITB"],  # Homebuilder ETFs
            "impact": -0.6,
            "lag_days": 30,
            "depth": 1,
        },
        {
            "effect": "Housing market slowdown",
            "mechanism": "Higher rates reduce affordability, cooling demand",
            "symbols": ["XHB", "HD", "LOW"],
            "impact": -0.5,
            "lag_days": 90,
            "depth": 2,
        },
        {
            "effect": "Construction materials demand drop",
            "mechanism": "Fewer housing starts -> less lumber, steel, cement demand",
            "symbols": ["LIN", "VMC", "MLM"],
            "impact": -0.4,
            "lag_days": 120,
            "depth": 3,
        },
        {
            "effect": "Construction employment impact",
            "mechanism": "Reduced building activity -> layoffs in construction sector",
            "symbols": [],
            "impact": -0.3,
            "lag_days": 180,
            "depth": 4,
        },
        {
            "effect": "Bank net interest margin improvement",
            "mechanism": "Wider spread between lending and deposit rates",
            "symbols": ["JPM", "BAC", "WFC"],
            "impact": 0.4,
            "lag_days": 60,
            "depth": 1,
        },
        {
            "effect": "Growth stock de-rating",
            "mechanism": "Higher discount rate -> lower present value of future cash flows",
            "symbols": ["QQQ", "ARKK"],
            "impact": -0.5,
            "lag_days": 7,
            "depth": 1,
        },
        {
            "effect": "Dollar strengthening",
            "mechanism": "Higher yields attract foreign capital",
            "symbols": ["EURUSD=X", "GBPUSD=X"],
            "impact": -0.3,
            "lag_days": 14,
            "depth": 1,
        },
        {
            "effect": "EM debt stress",
            "mechanism": "Stronger dollar -> higher USD-denominated debt burden",
            "symbols": ["EEM", "EMB"],
            "impact": -0.4,
            "lag_days": 60,
            "depth": 2,
        },
    ],
    "rate_cut": [
        {
            "effect": "Lower mortgage rates",
            "mechanism": "Direct pass-through to lending rates",
            "symbols": ["XHB", "ITB"],
            "impact": 0.6,
            "lag_days": 30,
            "depth": 1,
        },
        {
            "effect": "Housing market recovery",
            "mechanism": "Lower rates improve affordability",
            "symbols": ["XHB", "HD", "LOW"],
            "impact": 0.5,
            "lag_days": 90,
            "depth": 2,
        },
        {
            "effect": "Growth stock re-rating",
            "mechanism": "Lower discount rate -> higher PV of future cash flows",
            "symbols": ["QQQ", "ARKK"],
            "impact": 0.5,
            "lag_days": 7,
            "depth": 1,
        },
        {
            "effect": "Dollar weakening",
            "mechanism": "Lower yields reduce foreign capital flows",
            "symbols": ["EURUSD=X", "GLD"],
            "impact": 0.3,
            "lag_days": 14,
            "depth": 1,
        },
    ],
    "oil_spike": [
        {
            "effect": "Energy sector rally",
            "mechanism": "Higher revenue for oil producers",
            "symbols": ["XOM", "CVX", "XLE"],
            "impact": 0.7,
            "lag_days": 1,
            "depth": 1,
        },
        {
            "effect": "Transportation cost increase",
            "mechanism": "Higher fuel costs for airlines, shipping",
            "symbols": ["DAL", "UAL", "FDX", "UPS"],
            "impact": -0.5,
            "lag_days": 7,
            "depth": 1,
        },
        {
            "effect": "Consumer discretionary pressure",
            "mechanism": "Higher gas prices reduce consumer spending power",
            "symbols": ["XLY", "MCD", "NKE"],
            "impact": -0.3,
            "lag_days": 30,
            "depth": 2,
        },
        {
            "effect": "Inflation expectations rise",
            "mechanism": "Energy is major CPI component -> higher inflation prints",
            "symbols": ["TIP", "TLT"],
            "impact": -0.3,
            "lag_days": 60,
            "depth": 2,
        },
        {
            "effect": "Renewable energy boost",
            "mechanism": "High oil prices improve economics of alternatives",
            "symbols": ["ICLN", "TAN", "ENPH"],
            "impact": 0.3,
            "lag_days": 30,
            "depth": 2,
        },
    ],
    "recession": [
        {
            "effect": "Equity drawdown",
            "mechanism": "Earnings decline -> multiple contraction",
            "symbols": ["SPY", "QQQ"],
            "impact": -0.7,
            "lag_days": 1,
            "depth": 1,
        },
        {
            "effect": "Flight to safety",
            "mechanism": "Risk-off flows into Treasuries and gold",
            "symbols": ["TLT", "GLD", "GC=F"],
            "impact": 0.6,
            "lag_days": 1,
            "depth": 1,
        },
        {
            "effect": "Credit spreads widen",
            "mechanism": "Default risk increases -> corporate bonds sell off",
            "symbols": ["HYG", "LQD"],
            "impact": -0.5,
            "lag_days": 14,
            "depth": 1,
        },
        {
            "effect": "Consumer staples outperform",
            "mechanism": "Defensive sectors with stable demand",
            "symbols": ["XLP", "PG", "KO", "PEP"],
            "impact": 0.3,
            "lag_days": 7,
            "depth": 1,
        },
        {
            "effect": "Unemployment rise",
            "mechanism": "Companies cut costs -> layoffs",
            "symbols": [],
            "impact": -0.4,
            "lag_days": 90,
            "depth": 2,
        },
        {
            "effect": "Central bank easing",
            "mechanism": "Policy response to economic weakness",
            "symbols": ["TLT", "QQQ"],
            "impact": 0.5,
            "lag_days": 120,
            "depth": 3,
        },
    ],
}


# ---------------------------------------------------------------------------
# Thesis Generator Engine
# ---------------------------------------------------------------------------

class ThesisGenerator:
    """
    Investment thesis generation engine.

    Generates actionable trade theses based on macro analysis, sector data,
    catalysts, and structural changes. Includes second-derivative analysis
    for chain-reaction effects.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_position_size: float = 0.10,
        min_reward_risk_ratio: float = 2.0,
    ):
        """
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate.
        max_position_size : float
            Maximum position size as fraction of portfolio.
        min_reward_risk_ratio : float
            Minimum reward/risk ratio for thesis generation.
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.min_rr = min_reward_risk_ratio

    def generate_medium_term_theses(
        self,
        macro_data: dict[str, pd.DataFrame],
        sector_data: dict[str, dict[str, pd.DataFrame]],
    ) -> list[TradeThesis]:
        """
        Generate medium-term (1-6 month) investment theses.

        Methodology:
            1. Identify sectors with strong relative momentum:
               RS_sector = R_sector / R_market - 1
               If RS > 0.05 (5% outperformance): momentum thesis

            2. Identify sectors with mean-reversion potential:
               z_sector = (R_sector - mean(R_sector, 1y)) / std(R_sector, 1y)
               If z < -2: potential reversion candidate

            3. Cross-reference with macro regime:
               Expansion: favor cyclicals (Industrials, Materials, Discretionary)
               Contraction: favor defensives (Staples, Utilities, Healthcare)

            4. Score and rank theses by:
               score = momentum_signal * 0.4 + value_signal * 0.3 + macro_alignment * 0.3

        Parameters
        ----------
        macro_data : dict[str, pd.DataFrame]
            Macro indicator name -> time series DataFrame.
        sector_data : dict[str, dict[str, pd.DataFrame]]
            GICS sector -> {symbol: OHLCV DataFrame}.

        Returns
        -------
        list[TradeThesis]
            Medium-term theses sorted by score.
        """
        theses = []

        # Calculate sector returns and relative strength
        sector_stats = {}
        all_sector_returns = []

        for sector_name, symbols in sector_data.items():
            sector_returns = []
            for sym, df in symbols.items():
                if df is None or "Close" not in df.columns or len(df) < 126:
                    continue
                close = df["Close"].dropna()
                ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0.0
                ret_6m = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) > 126 else 0.0
                vol = float(np.log(close / close.shift(1)).dropna().std() * np.sqrt(252))
                sector_returns.append({
                    "symbol": sym, "ret_3m": ret_3m, "ret_6m": ret_6m, "vol": vol
                })

            if sector_returns:
                avg_3m = float(np.mean([s["ret_3m"] for s in sector_returns]))
                avg_6m = float(np.mean([s["ret_6m"] for s in sector_returns]))
                avg_vol = float(np.mean([s["vol"] for s in sector_returns]))
                sector_stats[sector_name] = {
                    "avg_3m": avg_3m, "avg_6m": avg_6m, "avg_vol": avg_vol,
                    "symbols": sector_returns,
                }
                all_sector_returns.append(avg_3m)

        if not sector_stats:
            return theses

        market_return = float(np.mean(all_sector_returns))

        # Generate theses for sectors with strong signals
        for sector_name, stats in sector_stats.items():
            relative_strength = stats["avg_3m"] - market_return
            symbols_in_sector = [s["symbol"] for s in stats["symbols"]]

            # --- Momentum Thesis ---
            if relative_strength > 0.03:
                # Strong relative momentum: overweight
                target_return = relative_strength * 2  # expect continuation
                stop_loss = -stats["avg_vol"] * 0.5
                rr = abs(target_return / stop_loss) if stop_loss != 0 else 0

                if rr >= self.min_rr:
                    score = 0.4 * min(relative_strength / 0.10, 1.0)
                    score += 0.3 * max(0, 1 - stats["avg_vol"] / 0.30)
                    score += 0.3 * 0.5  # neutral macro alignment

                    top_symbols = sorted(
                        stats["symbols"], key=lambda x: x["ret_3m"], reverse=True
                    )[:5]

                    theses.append(TradeThesis(
                        title=f"Momentum: Overweight {sector_name}",
                        asset_class=AssetClass.EQUITY,
                        symbols=[s["symbol"] for s in top_symbols],
                        direction="LONG",
                        timeframe=ThesisTimeframe.MEDIUM_TERM,
                        conviction=ConvictionLevel.HIGH if score > 0.7 else ConvictionLevel.MEDIUM,
                        summary=(
                            f"{sector_name} showing strong relative momentum "
                            f"(+{relative_strength:.1%} vs market over 3M). "
                            f"Expect continuation based on trend persistence."
                        ),
                        rationale=[
                            f"3M relative strength: +{relative_strength:.1%}",
                            f"6M return: {stats['avg_6m']:.1%}",
                            f"Sector volatility: {stats['avg_vol']:.1%} (annualized)",
                            "Trend-following historically profitable at 3M horizon",
                        ],
                        catalysts=[
                            "Continued earnings momentum in sector",
                            "Institutional fund flows into sector ETFs",
                        ],
                        risks=[
                            "Momentum reversal on macro shock",
                            "Sector crowding may lead to sharp unwind",
                            f"Volatility: {stats['avg_vol']:.1%} annualized",
                        ],
                        target_return=target_return,
                        stop_loss=stop_loss,
                        position_size_pct=min(
                            self.max_position_size,
                            0.10 / max(stats["avg_vol"], 0.05)
                        ),
                        entry_conditions=[
                            f"Sector continues to outperform market by > 1%",
                            "No major macro catalyst reversing trend",
                        ],
                        exit_conditions=[
                            f"Relative strength turns negative for 5+ days",
                            f"Stop loss at {stop_loss:.1%}",
                            "Take profit at target or 3-month horizon",
                        ],
                        score=score,
                    ))

            # --- Mean Reversion Thesis ---
            elif relative_strength < -0.05:
                # Severely underperforming: potential reversion
                target_return = abs(relative_strength) * 0.7
                stop_loss = relative_strength * 1.5
                rr = abs(target_return / stop_loss) if stop_loss != 0 else 0

                if rr >= self.min_rr:
                    score = 0.4 * min(abs(relative_strength) / 0.15, 1.0)
                    score += 0.3 * max(0, 1 - stats["avg_vol"] / 0.30)
                    score += 0.3 * 0.5

                    oversold_symbols = sorted(
                        stats["symbols"], key=lambda x: x["ret_3m"]
                    )[:5]

                    theses.append(TradeThesis(
                        title=f"Mean Reversion: {sector_name} Recovery",
                        asset_class=AssetClass.EQUITY,
                        symbols=[s["symbol"] for s in oversold_symbols],
                        direction="LONG",
                        timeframe=ThesisTimeframe.MEDIUM_TERM,
                        conviction=ConvictionLevel.MEDIUM,
                        summary=(
                            f"{sector_name} severely underperforming "
                            f"({relative_strength:.1%} vs market). "
                            f"Potential mean-reversion opportunity."
                        ),
                        rationale=[
                            f"3M underperformance: {relative_strength:.1%}",
                            "Historical mean-reversion tendency at extreme levels",
                            "Sector rotation tends to favor laggards over 3-6M",
                        ],
                        catalysts=[
                            "Earnings stabilization",
                            "Sector-specific policy tailwind",
                            "Technical support at current levels",
                        ],
                        risks=[
                            "Value trap: fundamentals may justify underperformance",
                            "Structural decline in sector",
                            "Continued outflows from sector funds",
                        ],
                        target_return=target_return,
                        stop_loss=stop_loss,
                        position_size_pct=min(
                            self.max_position_size * 0.7,
                            0.08 / max(stats["avg_vol"], 0.05)
                        ),
                        entry_conditions=[
                            "Technical base formation visible",
                            "Relative strength turning from negative to positive",
                        ],
                        exit_conditions=[
                            "Relative strength reaches neutral",
                            f"Stop loss at {stop_loss:.1%}",
                        ],
                        score=score,
                    ))

        theses.sort(key=lambda t: t.score, reverse=True)
        return theses

    def generate_long_term_theses(
        self,
        macro_events: list[dict[str, Any]],
        structural_changes: list[dict[str, Any]],
    ) -> list[TradeThesis]:
        """
        Generate long-term (6-24 month) investment theses based on
        structural macro changes.

        Structural themes considered:
            - Demographic shifts (aging, urbanization)
            - Technology disruption (AI, automation, electrification)
            - Policy regime changes (fiscal, monetary, regulatory)
            - Climate/energy transition
            - Deglobalization / reshoring

        Parameters
        ----------
        macro_events : list[dict]
            Each dict: {"event": str, "probability": float, "impact": float,
                        "sectors_affected": list[str], "direction": int}.
        structural_changes : list[dict]
            Each dict: {"theme": str, "beneficiaries": list[str],
                        "losers": list[str], "timeframe_years": int,
                        "conviction": float}.

        Returns
        -------
        list[TradeThesis]
        """
        theses = []

        for change in structural_changes:
            theme = change.get("theme", "Unknown Theme")
            beneficiaries = change.get("beneficiaries", [])
            losers = change.get("losers", [])
            timeframe = change.get("timeframe_years", 5)
            conviction = change.get("conviction", 0.5)

            # Long thesis on beneficiaries
            if beneficiaries:
                conv_level = (
                    ConvictionLevel.VERY_HIGH if conviction > 0.8
                    else ConvictionLevel.HIGH if conviction > 0.6
                    else ConvictionLevel.MEDIUM
                )
                theses.append(TradeThesis(
                    title=f"Structural Long: {theme}",
                    asset_class=AssetClass.EQUITY,
                    symbols=beneficiaries[:10],
                    direction="LONG",
                    timeframe=ThesisTimeframe.LONG_TERM if timeframe <= 3 else ThesisTimeframe.STRUCTURAL,
                    conviction=conv_level,
                    summary=f"Structural tailwind from {theme} over {timeframe}-year horizon.",
                    rationale=[
                        f"Theme: {theme}",
                        f"Expected duration: {timeframe} years",
                        f"Beneficiary companies identified with structural advantage",
                    ],
                    catalysts=[
                        "Policy support or regulatory tailwinds",
                        "Accelerating adoption / demand curves",
                        "Capital investment cycle beginning",
                    ],
                    risks=[
                        "Theme may take longer to materialize than expected",
                        "Competition may erode advantages",
                        "Regulatory risk",
                    ],
                    target_return=0.15 * timeframe,
                    stop_loss=-0.20,
                    position_size_pct=min(self.max_position_size, conviction * 0.10),
                    entry_conditions=[
                        "Accumulate on weakness over 6-12 months",
                        "Start with half position, add on confirmation",
                    ],
                    exit_conditions=[
                        "Theme thesis invalidated by structural change",
                        "Valuation reaches 2x fair value",
                        f"Time horizon: {timeframe} years",
                    ],
                    score=conviction,
                ))

            # Short thesis on losers
            if losers:
                theses.append(TradeThesis(
                    title=f"Structural Short: {theme} Losers",
                    asset_class=AssetClass.EQUITY,
                    symbols=losers[:10],
                    direction="SHORT",
                    timeframe=ThesisTimeframe.LONG_TERM,
                    conviction=ConvictionLevel.MEDIUM,
                    summary=f"Structural headwind from {theme} creates short opportunity.",
                    rationale=[
                        f"Theme: {theme} creates secular decline for these companies",
                        "Market may not fully price structural risk",
                        "Declining moat / competitive position",
                    ],
                    catalysts=[
                        "Earnings disappointment as structural decline accelerates",
                        "Analyst downgrades",
                    ],
                    risks=[
                        "Short squeeze risk",
                        "Company pivot / adaptation",
                        "Borrow cost may be high",
                    ],
                    target_return=0.20,
                    stop_loss=-0.15,
                    position_size_pct=min(self.max_position_size * 0.5, 0.05),
                    entry_conditions=[
                        "Technical breakdown below key support",
                        "Fundamental deterioration confirmed in earnings",
                    ],
                    exit_conditions=[
                        "Company successfully pivots business model",
                        "Stop loss triggered",
                    ],
                    score=conviction * 0.7,
                ))

        # Generate theses from macro events
        for event in macro_events:
            event_name = event.get("event", "Unknown Event")
            probability = event.get("probability", 0.5)
            impact = event.get("impact", 0.0)
            sectors = event.get("sectors_affected", [])
            direction = event.get("direction", 0)

            if abs(impact) < 0.3 or probability < 0.3:
                continue

            trade_direction = "LONG" if direction > 0 else "SHORT"
            target = abs(impact) * probability * 0.5

            theses.append(TradeThesis(
                title=f"Macro Event: {event_name}",
                asset_class=AssetClass.EQUITY,
                symbols=sectors[:5] if sectors else [],
                direction=trade_direction,
                timeframe=ThesisTimeframe.LONG_TERM,
                conviction=(
                    ConvictionLevel.HIGH if probability > 0.7
                    else ConvictionLevel.MEDIUM
                ),
                summary=(
                    f"Positioning for {event_name} "
                    f"(prob={probability:.0%}, impact={impact:+.0%})."
                ),
                rationale=[
                    f"Event probability: {probability:.0%}",
                    f"Expected market impact: {impact:+.0%}",
                    f"Affected sectors: {', '.join(sectors[:5])}",
                ],
                catalysts=[f"{event_name} occurs as expected"],
                risks=["Event does not materialize", "Market already priced in"],
                target_return=target,
                stop_loss=-target * 0.5,
                position_size_pct=min(self.max_position_size, probability * 0.08),
                entry_conditions=["Confirmation of event trajectory"],
                exit_conditions=["Event occurs or probability drops below 20%"],
                score=probability * abs(impact),
            ))

        theses.sort(key=lambda t: t.score, reverse=True)
        return theses

    def detect_macro_catalysts(
        self,
        news: list[dict[str, Any]],
        economic_calendar: list[dict[str, Any]],
    ) -> list[MacroCatalyst]:
        """
        Detect upcoming macro catalysts from news and economic calendar.

        Catalyst Scoring:
            score = impact_magnitude * probability * time_urgency
            time_urgency = 1 / (1 + days_until_event / 30)

        Parameters
        ----------
        news : list[dict]
            Each dict: {"headline": str, "sentiment": float (-1 to 1),
                        "magnitude": float (0 to 1), "categories": list[str]}.
        economic_calendar : list[dict]
            Each dict: {"event": str, "date": str, "type": str,
                        "expected_impact": float, "previous": float,
                        "consensus": float}.

        Returns
        -------
        list[MacroCatalyst]
            Sorted by impact score descending.
        """
        catalysts = []

        # Process economic calendar events
        for event in economic_calendar:
            event_name = event.get("event", "Unknown")
            event_date_str = event.get("date", "")
            event_type = event.get("type", "data_release")
            expected_impact = event.get("expected_impact", 0.5)

            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                event_date = datetime.now() + timedelta(days=30)

            days_until = max((event_date - datetime.now()).days, 1)

            # Determine affected asset classes based on event type
            asset_classes = []
            sectors = []
            direction = 0

            if event_type == "rate_decision":
                asset_classes = ["EQUITY", "FIXED_INCOME", "FX"]
                sectors = ["40 - Financials", "60 - Real Estate"]
                direction = -1 if expected_impact > 0 else 1  # hawkish = bearish for equities
            elif event_type == "employment":
                asset_classes = ["EQUITY", "FIXED_INCOME"]
                sectors = ["25 - Consumer Discretionary", "20 - Industrials"]
                direction = 1 if expected_impact > 0 else -1
            elif event_type == "inflation":
                asset_classes = ["EQUITY", "FIXED_INCOME", "COMMODITY"]
                sectors = ["30 - Consumer Staples", "10 - Energy"]
                direction = -1 if expected_impact > 0 else 1  # higher inflation = bearish
            elif event_type == "gdp":
                asset_classes = ["EQUITY", "COMMODITY"]
                direction = 1 if expected_impact > 0 else -1
            else:
                asset_classes = ["EQUITY"]
                direction = 1 if expected_impact > 0 else -1

            # Look up second order effects
            second_effects = []
            if event_type == "rate_decision" and expected_impact > 0:
                for chain in CAUSAL_CHAINS.get("rate_hike", []):
                    second_effects.append(chain["effect"])
            elif event_type == "rate_decision" and expected_impact < 0:
                for chain in CAUSAL_CHAINS.get("rate_cut", []):
                    second_effects.append(chain["effect"])

            catalysts.append(MacroCatalyst(
                event_name=event_name,
                event_type=event_type,
                expected_date=event_date,
                impact_direction=direction,
                impact_magnitude=abs(expected_impact),
                probability=0.8,  # Calendar events are highly likely
                affected_asset_classes=asset_classes,
                affected_sectors=sectors,
                second_order_effects=second_effects,
                description=(
                    f"{event_name} on {event_date.strftime('%Y-%m-%d')} "
                    f"(expected impact: {expected_impact:+.2f})"
                ),
            ))

        # Process news items into potential catalysts
        for item in news:
            headline = item.get("headline", "")
            sentiment = item.get("sentiment", 0.0)
            magnitude = item.get("magnitude", 0.3)
            categories = item.get("categories", [])

            if abs(sentiment) < 0.3 or magnitude < 0.3:
                continue

            catalysts.append(MacroCatalyst(
                event_name=headline,
                event_type="news",
                expected_date=datetime.now(),
                impact_direction=1 if sentiment > 0 else -1,
                impact_magnitude=magnitude,
                probability=0.6,  # News-derived catalysts have lower certainty
                affected_asset_classes=categories if categories else ["EQUITY"],
                affected_sectors=[],
                second_order_effects=[],
                description=headline,
            ))

        # Sort by impact score
        catalysts.sort(
            key=lambda c: c.impact_magnitude * c.probability,
            reverse=True,
        )
        return catalysts

    def second_derivative_analysis(
        self, primary_event: str
    ) -> list[SecondaryEffect]:
        """
        Analyze second-order (chain reaction) effects of a primary event.

        Chain reaction model:
            Event E -> Effect S1 (depth 1, lag d1)
                    -> Effect S2 (depth 2, lag d2)
                    -> Effect S3 (depth 3, lag d3)

        Cumulative impact at depth n:
            impact_n = initial_impact * product_{i=1}^{n}(transmission_rate_i)
            transmission_rate ~ 0.6-0.8 per link in the chain

        Example chain:
            Rate hike -> Housing slowdown -> Construction materials drop
                      -> Employment impact -> Consumer spending decline

        Parameters
        ----------
        primary_event : str
            Type of primary event. Supported: "rate_hike", "rate_cut",
            "oil_spike", "recession".

        Returns
        -------
        list[SecondaryEffect]
            Chain of secondary effects sorted by depth then impact.
        """
        chain = CAUSAL_CHAINS.get(primary_event, [])
        if not chain:
            logger.warning("No causal chain defined for event: %s", primary_event)
            return []

        effects = []
        for link in chain:
            effects.append(SecondaryEffect(
                primary_event=primary_event,
                secondary_effect=link["effect"],
                transmission_mechanism=link["mechanism"],
                affected_symbols=link["symbols"],
                estimated_impact=link["impact"],
                confidence=max(0.3, 1.0 - link["depth"] * 0.15),
                time_lag_days=link["lag_days"],
                chain_depth=link["depth"],
            ))

        effects.sort(key=lambda e: (e.chain_depth, -abs(e.estimated_impact)))
        return effects

    def ipo_screening(
        self,
        upcoming_ipos: list[dict[str, Any]],
    ) -> list[IPOThesis]:
        """
        Screen upcoming IPOs for investment potential.

        Valuation methodology:
            1. Peer comparison:
               EV/Revenue multiple from comparable public companies.
               Fair value = estimated_revenue * peer_EV_Revenue

            2. Growth premium/discount:
               If growth_rate > peer_avg_growth: apply premium
               premium = (company_growth - peer_growth) / peer_growth * 0.5

            3. Profitability adjustment:
               If unprofitable: apply 20% discount
               If profitable with expanding margins: apply 10% premium

            4. IPO discount:
               Historical first-day pop averages 10-15%.
               Subscribe if IPO price < fair_value * 0.85

        Parameters
        ----------
        upcoming_ipos : list[dict]
            Each dict: {"company": str, "symbol": str, "date": str,
                        "price_low": float, "price_high": float,
                        "revenue": float, "growth_rate": float,
                        "sector": str, "peer_ev_revenue": tuple,
                        "profitable": bool}.

        Returns
        -------
        list[IPOThesis]
            Screened IPOs sorted by score.
        """
        results = []

        for ipo in upcoming_ipos:
            company = ipo.get("company", "Unknown")
            symbol = ipo.get("symbol", "")
            sector = ipo.get("sector", "Technology")
            revenue = ipo.get("revenue", 0.0)
            growth_rate = ipo.get("growth_rate", 0.0)
            price_low = ipo.get("price_low", 0.0)
            price_high = ipo.get("price_high", 0.0)
            peer_ev_rev = ipo.get("peer_ev_revenue", (5.0, 15.0))
            profitable = ipo.get("profitable", False)

            try:
                ipo_date = datetime.strptime(ipo.get("date", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                ipo_date = datetime.now() + timedelta(days=30)

            # Fair value calculation
            base_low = revenue * peer_ev_rev[0]
            base_high = revenue * peer_ev_rev[1]

            # Growth premium
            peer_avg_growth = 0.15  # assume 15% average
            if growth_rate > peer_avg_growth:
                premium = min((growth_rate - peer_avg_growth) / peer_avg_growth * 0.5, 0.5)
            else:
                premium = max((growth_rate - peer_avg_growth) / peer_avg_growth * 0.3, -0.3)

            # Profitability adjustment
            if not profitable:
                prof_adj = -0.20
            else:
                prof_adj = 0.10

            fair_low = base_low * (1 + premium + prof_adj)
            fair_high = base_high * (1 + premium + prof_adj)

            # Midpoint comparison
            ipo_mid = (price_low + price_high) / 2
            fair_mid = (fair_low + fair_high) / 2

            upside = (fair_mid - ipo_mid) / ipo_mid if ipo_mid > 0 else 0
            score = min(max(upside, -1), 1) * 0.5 + min(growth_rate, 1.0) * 0.3

            # Profitability bonus
            if profitable:
                score += 0.2
            score = max(0, min(score, 1.0))

            # Recommendation
            if upside > 0.15:
                recommendation = "subscribe"
            elif upside < -0.10:
                recommendation = "avoid"
            else:
                recommendation = "wait_for_lockup"

            rationale = [
                f"Revenue: ${revenue/1e6:.0f}M, Growth: {growth_rate:.0%}",
                f"Peer EV/Revenue range: {peer_ev_rev[0]:.1f}x - {peer_ev_rev[1]:.1f}x",
                f"Fair value range: ${fair_low/1e6:.0f}M - ${fair_high/1e6:.0f}M",
                f"IPO price range: ${price_low:.0f} - ${price_high:.0f}",
                f"Estimated upside to fair value: {upside:.0%}",
            ]

            risks = ["Lock-up expiry selling pressure (usually 90-180 days post-IPO)"]
            if not profitable:
                risks.append("Not yet profitable - cash burn risk")
            if growth_rate < 0.10:
                risks.append("Low growth rate may not justify IPO premium")

            results.append(IPOThesis(
                company_name=company,
                symbol=symbol,
                expected_date=ipo_date,
                sector=sector,
                ipo_price_low=price_low,
                ipo_price_high=price_high,
                peer_ev_revenue_range=peer_ev_rev,
                estimated_revenue=revenue,
                fair_value_range=(fair_low, fair_high),
                recommendation=recommendation,
                rationale=rationale,
                risks=risks,
                score=score,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def distress_screening(
        self,
        credit_data: list[dict[str, Any]],
    ) -> list[DistressThesis]:
        """
        Screen for distressed investment opportunities.

        Altman Z-Score:
            Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            where:
                X1 = Working Capital / Total Assets
                X2 = Retained Earnings / Total Assets
                X3 = EBIT / Total Assets
                X4 = Market Cap / Total Liabilities
                X5 = Sales / Total Assets

            Interpretation:
                Z > 2.99:  Safe zone (low probability of default)
                1.81 < Z < 2.99: Grey zone (moderate risk)
                Z < 1.81:  Distress zone (high probability of default)

        Distress Probability (simplified logistic model):
            P(distress) = 1 / (1 + exp(Z - 1.81))

        Investment theses:
            Z < 1.0:   Consider short or CDS
            1.0 < Z < 1.81: Potential recovery play if catalysts exist
            Company with improving Z: Long recovery thesis

        Parameters
        ----------
        credit_data : list[dict]
            Each dict: {"symbol": str, "company": str,
                        "working_capital": float, "total_assets": float,
                        "retained_earnings": float, "ebit": float,
                        "market_cap": float, "total_liabilities": float,
                        "sales": float, "current_price": float,
                        "book_value_per_share": float}.

        Returns
        -------
        list[DistressThesis]
            Distressed opportunities sorted by score.
        """
        results = []

        for data in credit_data:
            symbol = data.get("symbol", "")
            company = data.get("company", "")
            total_assets = data.get("total_assets", 1.0)
            total_liabilities = data.get("total_liabilities", 1.0)

            if total_assets <= 0:
                continue

            # Calculate Altman Z-Score components
            x1 = data.get("working_capital", 0.0) / total_assets
            x2 = data.get("retained_earnings", 0.0) / total_assets
            x3 = data.get("ebit", 0.0) / total_assets
            x4 = data.get("market_cap", 0.0) / max(total_liabilities, 1.0)
            x5 = data.get("sales", 0.0) / total_assets

            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

            # Distress probability (logistic)
            distress_prob = 1.0 / (1.0 + np.exp(z_score - 1.81))

            current_price = data.get("current_price", 0.0)
            book_value = data.get("book_value_per_share", 0.0)

            # Estimated recovery value (liquidation basis)
            # Typically 20-40% of book value in distress
            recovery_value = book_value * 0.3

            # Determine thesis type
            if z_score < 1.0:
                thesis_type = "short_bankruptcy"
                potential_return = max(0, (current_price - recovery_value) / current_price)
                direction_text = "Short"
                rationale = [
                    f"Altman Z-Score: {z_score:.2f} (deep distress zone)",
                    f"Distress probability: {distress_prob:.0%}",
                    f"Current price: ${current_price:.2f}, Est. recovery: ${recovery_value:.2f}",
                    "High likelihood of bankruptcy or severe restructuring",
                ]
                risks = [
                    "Short squeeze if rescue financing appears",
                    "Government bailout or intervention",
                    "Activist investor may engineer turnaround",
                ]
            elif z_score < 1.81:
                thesis_type = "long_recovery"
                potential_return = max(0, (book_value - current_price) / current_price)
                direction_text = "Long (recovery)"
                rationale = [
                    f"Altman Z-Score: {z_score:.2f} (distress zone, but not terminal)",
                    f"Distress probability: {distress_prob:.0%}",
                    f"Price/Book: {current_price/book_value:.2f}x" if book_value > 0 else "N/A",
                    "Potential for recovery if management executes turnaround",
                ]
                risks = [
                    "Situation may deteriorate further",
                    "Dilution from emergency capital raise",
                    "Key customer/supplier defection",
                ]
            else:
                # Not distressed enough
                continue

            score = distress_prob * abs(potential_return)
            if z_score < 1.0:
                score *= 1.2  # Higher score for clearer bankruptcy candidates

            results.append(DistressThesis(
                symbol=symbol,
                company_name=company,
                altman_z_score=z_score,
                distress_probability=distress_prob,
                current_price=current_price,
                estimated_recovery_value=recovery_value,
                potential_return=potential_return,
                thesis_type=thesis_type,
                rationale=rationale,
                risks=risks,
                score=score,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = ThesisGenerator()

    # Demo: Second derivative analysis
    print("=== Second Derivative: Rate Hike ===")
    effects = gen.second_derivative_analysis("rate_hike")
    for e in effects:
        print(f"  [Depth {e.chain_depth}] {e.secondary_effect}")
        print(f"    Mechanism: {e.transmission_mechanism}")
        print(f"    Impact: {e.estimated_impact:+.1f}, Lag: {e.time_lag_days}d, "
              f"Confidence: {e.confidence:.0%}")
        if e.affected_symbols:
            print(f"    Symbols: {', '.join(e.affected_symbols)}")

    # Demo: Macro catalysts
    print("\n=== Macro Catalysts ===")
    calendar = [
        {"event": "FOMC Rate Decision", "date": "2026-04-15", "type": "rate_decision", "expected_impact": 0.5},
        {"event": "US CPI Release", "date": "2026-04-10", "type": "inflation", "expected_impact": 0.3},
        {"event": "Non-Farm Payrolls", "date": "2026-04-05", "type": "employment", "expected_impact": 0.4},
    ]
    catalysts = gen.detect_macro_catalysts([], calendar)
    for c in catalysts:
        print(f"  {c.event_name}: impact={c.impact_magnitude:.1f}, "
              f"direction={'bullish' if c.impact_direction > 0 else 'bearish'}")

    # Demo: IPO screening
    print("\n=== IPO Screening ===")
    ipos = [
        {
            "company": "TechCorp AI", "symbol": "TCAI", "date": "2026-05-01",
            "price_low": 20, "price_high": 24, "revenue": 500e6,
            "growth_rate": 0.45, "sector": "Technology",
            "peer_ev_revenue": (8.0, 15.0), "profitable": False,
        },
        {
            "company": "GreenEnergy Co", "symbol": "GRNE", "date": "2026-05-15",
            "price_low": 15, "price_high": 18, "revenue": 200e6,
            "growth_rate": 0.25, "sector": "Utilities",
            "peer_ev_revenue": (3.0, 6.0), "profitable": True,
        },
    ]
    ipo_results = gen.ipo_screening(ipos)
    for ipo in ipo_results:
        print(f"  {ipo.company_name} ({ipo.symbol}): {ipo.recommendation} "
              f"(score={ipo.score:.2f})")
        for r in ipo.rationale:
            print(f"    {r}")

    # Demo: Distress screening
    print("\n=== Distress Screening ===")
    credit = [
        {
            "symbol": "DISTCO", "company": "Distressed Corp",
            "working_capital": -50e6, "total_assets": 500e6,
            "retained_earnings": -200e6, "ebit": -30e6,
            "market_cap": 100e6, "total_liabilities": 400e6,
            "sales": 300e6, "current_price": 5.0, "book_value_per_share": 2.0,
        },
        {
            "symbol": "RECOVCO", "company": "Recovery Inc",
            "working_capital": 10e6, "total_assets": 800e6,
            "retained_earnings": 50e6, "ebit": 20e6,
            "market_cap": 200e6, "total_liabilities": 500e6,
            "sales": 600e6, "current_price": 12.0, "book_value_per_share": 15.0,
        },
    ]
    distress_results = gen.distress_screening(credit)
    for d in distress_results:
        print(f"  {d.company_name} ({d.symbol}): Z={d.altman_z_score:.2f}, "
              f"P(distress)={d.distress_probability:.0%}, type={d.thesis_type}")

    # Demo: Long-term theses
    print("\n=== Long-Term Theses ===")
    structural = [
        {
            "theme": "AI Infrastructure Build-Out",
            "beneficiaries": ["NVDA", "MSFT", "GOOGL", "AMD", "AVGO"],
            "losers": ["legacy_IT_services"],
            "timeframe_years": 5,
            "conviction": 0.85,
        },
        {
            "theme": "Energy Transition",
            "beneficiaries": ["NEE", "ENPH", "FSLR"],
            "losers": ["XOM", "CVX"],
            "timeframe_years": 10,
            "conviction": 0.70,
        },
    ]
    lt_theses = gen.generate_long_term_theses([], structural)
    for t in lt_theses:
        print(f"  {t.title} ({t.direction}): conviction={t.conviction.value}, "
              f"symbols={t.symbols[:5]}")
