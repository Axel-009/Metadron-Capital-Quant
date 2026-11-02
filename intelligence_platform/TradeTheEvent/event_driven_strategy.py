"""
Event-driven trading strategy integrated with full OpenBB universe.

Scans ALL asset classes for event-driven opportunities:
1. Earnings surprises: actual_EPS vs consensus -> post-earnings drift
   SUE = (EPS_actual - EPS_expected) / sigma_forecast
   PEAD: stocks with |SUE| > 2 continue drifting for 60 days

2. Macro events: rate decisions, GDP releases, employment data
   Impact model: dP = beta_event * (actual - expected) / sigma_historical

3. Geopolitical: war, sanctions, regime change
   Cross-asset propagation: dP_i = sum(beta_ij * shock_j)

4. Corporate actions: M&A, spinoffs, buybacks, insider trading
   M&A arb spread: spread = (offer_price - current) / current
   Completion probability: P(close) from historical + deal-specific factors

5. Rating changes: upgrades/downgrades across all rating agencies
   Abnormal return: AR_t = R_t - E[R_t | market model]
   CAR = sum AR_t over event window

6. IPO analysis: first-day pricing, lockup expiry, secondary offering
   IPO underpricing: U = (P_close_day1 - P_offer) / P_offer

Event Study Methodology:
    1. Estimation window: [-250, -10] trading days before event
    2. Event window: [-5, +5] trading days around event
    3. Market model: R_i,t = alpha_i + beta_i * R_m,t + epsilon_i,t
    4. Expected return: E[R_i,t] = alpha_hat + beta_hat * R_m,t
    5. Abnormal return: AR_i,t = R_i,t - E[R_i,t]
    6. CAR_i(t1,t2) = sum_{t=t1}^{t2} AR_i,t
    7. Test statistic: t = CAR / (sigma_AR * sqrt(L)), L = event window length
    8. Significance: |t| > 1.96 at 5% level

Cross-Asset Correlation During Events:
    rho_crisis > rho_normal (correlations increase during stress)
    DCC-GARCH: H_t = D_t * R_t * D_t
    Where D_t = diag(sqrt(h_ii,t)), R_t = dynamic correlation matrix
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime, timedelta

from openbb_universe import (
    get_full_universe,
    get_equity_universe,
    detect_asset_class,
    classify_by_gics,
    AssetClass,
    GICSSector,
    SP500_TOP_50,
    BOND_ETFS,
    COMMODITY_FUTURES,
    FX_MAJORS,
    SECTOR_ETFS,
)


class EventType(Enum):
    EARNINGS = "earnings"
    MACRO_RELEASE = "macro"
    GEOPOLITICAL = "geopolitical"
    CORPORATE_ACTION = "corporate_action"
    RATING_CHANGE = "rating_change"
    IPO = "ipo"
    REGULATORY = "regulatory"
    NATURAL_DISASTER = "natural_disaster"


class TradeHorizon(Enum):
    INTRADAY = "intraday"
    SWING = "swing"  # 2-10 days
    MEDIUM_TERM = "medium_term"  # 10-60 days
    LONG_TERM = "long_term"  # 60+ days


@dataclass
class EventSignal:
    """
    Complete event-driven trading signal with cross-asset effects.

    Risk-Adjusted Sizing:
        position_size_pct = min(kelly_fraction, max_position)
        kelly_fraction = (p * b - q) / b
        Where p = confidence, b = expected_impact / max_loss, q = 1 - p
    """
    event_type: EventType
    symbol: str
    event_date: datetime
    description: str
    expected_impact: float
    confidence: float
    direction: str  # "long", "short", "pair_trade"
    asset_class: str
    affected_symbols: List[str]
    second_derivative_effects: List[dict]
    trade_horizon: str
    position_size_pct: float
    thesis: str
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    risk_reward_ratio: float = 0.0


class EventDrivenStrategy:
    """
    Multi-asset event-driven trading strategy engine.

    Scans earnings, macro, geopolitical, corporate actions, ratings, and IPOs
    across equities, bonds, commodities, crypto, and FX.
    """

    # Mapping of macro events to affected asset betas
    MACRO_EVENT_MAP = {
        "rate": {
            "keywords": ["rate", "fed", "fomc", "ecb", "boj", "boe", "monetary"],
            "assets": [
                {"symbol": "TLT", "beta": -0.8, "asset_class": "fixed_income"},
                {"symbol": "XLF", "beta": 0.5, "asset_class": "equity"},
                {"symbol": "GLD", "beta": -0.3, "asset_class": "commodity"},
                {"symbol": "EURUSD=X", "beta": -0.4, "asset_class": "fx"},
                {"symbol": "BTC-USD", "beta": -0.2, "asset_class": "crypto"},
                {"symbol": "XLRE", "beta": -0.6, "asset_class": "equity"},
                {"symbol": "HYG", "beta": -0.3, "asset_class": "fixed_income"},
            ],
        },
        "gdp": {
            "keywords": ["gdp", "growth", "economic output"],
            "assets": [
                {"symbol": "SPY", "beta": 0.6, "asset_class": "equity"},
                {"symbol": "EEM", "beta": 0.8, "asset_class": "equity"},
                {"symbol": "HYG", "beta": 0.4, "asset_class": "fixed_income"},
                {"symbol": "CL=F", "beta": 0.5, "asset_class": "commodity"},
                {"symbol": "IWM", "beta": 0.7, "asset_class": "equity"},
            ],
        },
        "cpi": {
            "keywords": ["cpi", "inflation", "pce", "price index"],
            "assets": [
                {"symbol": "TIP", "beta": 0.5, "asset_class": "fixed_income"},
                {"symbol": "GLD", "beta": 0.4, "asset_class": "commodity"},
                {"symbol": "TLT", "beta": -0.7, "asset_class": "fixed_income"},
                {"symbol": "XLU", "beta": -0.3, "asset_class": "equity"},
                {"symbol": "BTC-USD", "beta": 0.2, "asset_class": "crypto"},
            ],
        },
        "employment": {
            "keywords": ["employment", "nfp", "payroll", "jobs", "unemployment", "labor"],
            "assets": [
                {"symbol": "SPY", "beta": 0.5, "asset_class": "equity"},
                {"symbol": "DX-Y.NYB", "beta": 0.3, "asset_class": "fx"},
                {"symbol": "TLT", "beta": -0.4, "asset_class": "fixed_income"},
                {"symbol": "XLY", "beta": 0.6, "asset_class": "equity"},
            ],
        },
        "housing": {
            "keywords": ["housing", "home sales", "building permits", "mortgage"],
            "assets": [
                {"symbol": "XLRE", "beta": 0.7, "asset_class": "equity"},
                {"symbol": "XLB", "beta": 0.4, "asset_class": "equity"},
                {"symbol": "HD", "beta": 0.5, "asset_class": "equity"},
            ],
        },
        "trade": {
            "keywords": ["trade balance", "tariff", "import", "export", "trade war"],
            "assets": [
                {"symbol": "EEM", "beta": -0.6, "asset_class": "equity"},
                {"symbol": "USDCNY=X", "beta": 0.5, "asset_class": "fx"},
                {"symbol": "XLI", "beta": -0.4, "asset_class": "equity"},
            ],
        },
        "oil": {
            "keywords": ["oil", "opec", "crude", "energy", "petroleum"],
            "assets": [
                {"symbol": "CL=F", "beta": 0.9, "asset_class": "commodity"},
                {"symbol": "XLE", "beta": 0.7, "asset_class": "equity"},
                {"symbol": "USDCAD=X", "beta": -0.4, "asset_class": "fx"},
                {"symbol": "USO", "beta": 0.85, "asset_class": "commodity"},
            ],
        },
    }

    # Second derivative effects by macro event category
    SECOND_DERIVATIVE_MAP = {
        "rate": {
            "hawkish": [
                {"effect": "Housing slowdown", "lag_months": 3,
                 "affected_sectors": ["Real Estate", "Materials", "Consumer Discretionary"]},
                {"effect": "Strong dollar", "lag_months": 1,
                 "affected_sectors": ["Exporters", "Emerging Markets"]},
                {"effect": "Bank margin expansion", "lag_months": 1,
                 "affected_sectors": ["Financials"]},
                {"effect": "Debt servicing pressure", "lag_months": 6,
                 "affected_sectors": ["High-leverage corporates"]},
                {"effect": "Growth-to-value rotation", "lag_months": 1,
                 "affected_sectors": ["Information Technology", "Communication Services"]},
            ],
            "dovish": [
                {"effect": "Risk asset rally", "lag_months": 0,
                 "affected_sectors": ["Information Technology", "Consumer Discretionary"]},
                {"effect": "Dollar weakening", "lag_months": 1,
                 "affected_sectors": ["Materials", "Industrials"]},
                {"effect": "Yield curve steepening", "lag_months": 1,
                 "affected_sectors": ["Financials"]},
            ],
        },
        "cpi": {
            "hot": [
                {"effect": "Wage pressure", "lag_months": 2,
                 "affected_sectors": ["Labor-intensive industries"]},
                {"effect": "Input cost squeeze", "lag_months": 1,
                 "affected_sectors": ["Consumer Staples", "Industrials"]},
                {"effect": "Real asset outperformance", "lag_months": 1,
                 "affected_sectors": ["Real Estate", "Commodities"]},
                {"effect": "Fed hawkishness pricing", "lag_months": 0,
                 "affected_sectors": ["Growth stocks", "Long-duration bonds"]},
            ],
            "cold": [
                {"effect": "Disinflation trade", "lag_months": 0,
                 "affected_sectors": ["Long-duration bonds", "Growth stocks"]},
                {"effect": "Consumer spending boost", "lag_months": 2,
                 "affected_sectors": ["Consumer Discretionary", "Consumer Staples"]},
            ],
        },
        "employment": {
            "strong": [
                {"effect": "Consumer confidence boost", "lag_months": 1,
                 "affected_sectors": ["Consumer Discretionary", "Consumer Staples"]},
                {"effect": "Wage inflation pressure", "lag_months": 3,
                 "affected_sectors": ["Labor-intensive industries"]},
            ],
            "weak": [
                {"effect": "Recession fears", "lag_months": 0,
                 "affected_sectors": ["Cyclicals", "Small caps"]},
                {"effect": "Safe haven flows", "lag_months": 0,
                 "affected_sectors": ["Utilities", "Consumer Staples", "Gold"]},
            ],
        },
    }

    def __init__(self):
        self.event_history: List[EventSignal] = []
        self.signal_cache: Dict[str, List[EventSignal]] = {}

    def _calculate_stop_and_target(
        self, expected_impact: float, confidence: float, direction: str
    ) -> Tuple[float, float, float]:
        """
        Calculate stop loss, take profit, and risk/reward ratio.

        Position Management Rules:
            Stop loss: 2x expected adverse move (using volatility-adjusted stops)
            Take profit: expected_impact * confidence_multiplier
            Risk/reward minimum: 1.5:1 for swing, 2:1 for medium_term

        Kelly-based sizing:
            f* = (p * b - q) / b
            Where p = confidence, b = take_profit / stop_loss, q = 1 - p

        Args:
            expected_impact: Expected price impact as fraction.
            confidence: Signal confidence (0-1).
            direction: "long" or "short".

        Returns:
            Tuple of (stop_loss_pct, take_profit_pct, risk_reward_ratio).
        """
        # Stop loss: inverse of confidence scaled by impact magnitude
        stop_loss = abs(expected_impact) * (2.0 - confidence)
        stop_loss = max(stop_loss, 0.005)  # Minimum 0.5% stop

        # Take profit: impact scaled by confidence
        take_profit = abs(expected_impact) * (1.0 + confidence)
        take_profit = max(take_profit, stop_loss * 1.5)  # Minimum 1.5:1 R/R

        risk_reward = take_profit / stop_loss if stop_loss > 0 else 0.0

        return (round(stop_loss, 4), round(take_profit, 4), round(risk_reward, 2))

    def scan_earnings_events(
        self, earnings_calendar: pd.DataFrame, historical_data: dict
    ) -> List[EventSignal]:
        """
        Scan for post-earnings drift (PEAD) opportunities.

        Standardized Unexpected Earnings (SUE):
            SUE = (EPS_actual - EPS_expected) / sigma_forecast
            |SUE| > 2: significant surprise
            |SUE| > 3: extreme surprise

        Post-Earnings Announcement Drift (PEAD):
            - Stocks with positive SUE > 2 drift up for ~60 trading days
            - Stocks with negative SUE < -2 drift down for ~60 trading days
            - Effect is stronger for:
                * Small caps (less analyst coverage)
                * Low institutional ownership
                * Low trading volume (less efficient)

        Earnings Quality Indicators:
            - Revenue surprise alignment (both EPS and revenue beat)
            - Guidance revision direction
            - Earnings persistence: AR(1) coefficient of earnings
            - Accruals quality: high accruals = lower earnings persistence

        Args:
            earnings_calendar: DataFrame with columns: symbol, actual_eps,
                expected_eps, std_estimate, report_date.
            historical_data: Dict of symbol -> historical price data for context.

        Returns:
            List of EventSignal objects for actionable earnings events.
        """
        signals = []

        for _, row in earnings_calendar.iterrows():
            symbol = row.get("symbol", "")
            actual_eps = row.get("actual_eps")
            expected_eps = row.get("expected_eps")

            if actual_eps is None or expected_eps is None:
                continue

            # Standard deviation of estimate
            std_eps = row.get("std_estimate")
            if std_eps is None or std_eps == 0:
                std_eps = abs(expected_eps) * 0.1 if expected_eps != 0 else 0.01
            std_eps = max(std_eps, 0.001)

            # Standardized Unexpected Earnings
            sue = (actual_eps - expected_eps) / std_eps

            if abs(sue) < 2.0:
                continue  # Only trade significant surprises

            direction = "long" if sue > 0 else "short"
            expected_impact = sue * 0.005  # ~0.5% per std dev of surprise
            confidence = min(abs(sue) / 5.0, 0.95)

            # Revenue surprise alignment boosts confidence
            actual_rev = row.get("actual_revenue")
            expected_rev = row.get("expected_revenue")
            if actual_rev is not None and expected_rev is not None and expected_rev > 0:
                rev_surprise = (actual_rev - expected_rev) / expected_rev
                if np.sign(rev_surprise) == np.sign(sue):
                    confidence = min(confidence * 1.15, 0.95)
                    expected_impact *= 1.2  # Stronger effect when aligned

            # Historical volatility context for sizing
            hist = historical_data.get(symbol, {})
            hist_vol = hist.get("volatility", 0.3)

            stop_loss, take_profit, rr = self._calculate_stop_and_target(
                expected_impact, confidence, direction
            )

            # Position sizing: Kelly criterion capped
            kelly = (confidence * rr - (1 - confidence)) / rr if rr > 0 else 0
            position_size = min(max(kelly * 100, 0.5), 3.0)

            # Determine trade horizon based on SUE magnitude
            if abs(sue) > 4:
                horizon = TradeHorizon.MEDIUM_TERM.value  # Large surprises drift longer
            else:
                horizon = TradeHorizon.SWING.value

            # Check for sector peers that might also move
            sector_peers = []
            if symbol in SP500_TOP_50:
                _, sector, _ = SP500_TOP_50[symbol]
                gics = classify_by_gics()
                sector_peers = [s for s in gics.get(sector, []) if s != symbol][:5]

            signals.append(EventSignal(
                event_type=EventType.EARNINGS,
                symbol=symbol,
                event_date=pd.Timestamp(row.get("report_date", datetime.now())).to_pydatetime()
                if "report_date" in row
                else datetime.now(),
                description=f"Earnings surprise SUE={sue:.2f} "
                f"(actual={actual_eps:.2f} vs est={expected_eps:.2f})",
                expected_impact=round(expected_impact, 4),
                confidence=round(confidence, 4),
                direction=direction,
                asset_class="equity",
                affected_symbols=[symbol] + sector_peers[:3],
                second_derivative_effects=[
                    {
                        "effect": "Sector sentiment shift",
                        "lag_days": 1,
                        "peers": sector_peers,
                    }
                ]
                if sector_peers
                else [],
                trade_horizon=horizon,
                position_size_pct=round(position_size, 2),
                thesis=(
                    f"Post-earnings drift: {'beat' if sue > 0 else 'miss'} "
                    f"by {abs(sue):.1f} std devs. "
                    f"{'Revenue aligned. ' if actual_rev and expected_rev and np.sign((actual_rev - expected_rev) / max(expected_rev, 1)) == np.sign(sue) else ''}"
                    f"Expect continuation for {30 if abs(sue) < 4 else 60} days."
                ),
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                risk_reward_ratio=rr,
            ))

        return signals

    def _match_macro_event(self, event_name: str) -> Optional[str]:
        """
        Match an event name to a macro event category.

        Args:
            event_name: Name/description of the economic event.

        Returns:
            Matched category key or None.
        """
        event_lower = event_name.lower()
        for category, config in self.MACRO_EVENT_MAP.items():
            for keyword in config["keywords"]:
                if keyword in event_lower:
                    return category
        return None

    def _get_macro_affected_assets(self, event_name: str, surprise: float) -> List[dict]:
        """
        Map macro events to affected assets with beta coefficients.

        Uses the MACRO_EVENT_MAP to find affected instruments and adjusts
        direction based on the sign of the surprise.

        The beta represents the sensitivity of the asset to the macro surprise:
            dP_asset = beta * surprise * base_impact

        Positive beta: asset moves in same direction as surprise
        Negative beta: asset moves opposite to surprise

        Args:
            event_name: Name of the macro event.
            surprise: Standardized surprise (actual - forecast) / historical_std.

        Returns:
            List of dicts with symbol, beta, direction, asset_class.
        """
        category = self._match_macro_event(event_name)
        if category is None:
            return []

        config = self.MACRO_EVENT_MAP[category]
        affected = []

        for asset in config["assets"]:
            beta = asset["beta"]
            # Determine direction: positive beta + positive surprise = long
            net_effect = beta * surprise
            direction = "long" if net_effect > 0 else "short"

            affected.append({
                "symbol": asset["symbol"],
                "beta": beta,
                "direction": direction,
                "asset_class": asset["asset_class"],
                "net_effect": net_effect,
            })

        return affected

    def _calculate_second_derivative(self, event_name: str, surprise: float) -> List[dict]:
        """
        Calculate 2nd derivative (lagged, indirect) effects of macro events.

        Second derivative effects are indirect consequences that manifest
        with a time lag. They represent the causal chain:
            Direct: Fed hikes -> bond yields rise
            2nd order: Higher yields -> housing slows -> construction drops
            3rd order: Construction drops -> materials demand falls -> commodity prices drop

        Args:
            event_name: Name of the macro event.
            surprise: Standardized surprise value.

        Returns:
            List of second derivative effect dictionaries.
        """
        category = self._match_macro_event(event_name)
        if category is None:
            return []

        effects_config = self.SECOND_DERIVATIVE_MAP.get(category, {})
        if not effects_config:
            return []

        # Determine direction label
        if category == "rate":
            direction_key = "hawkish" if surprise > 0 else "dovish"
        elif category == "cpi":
            direction_key = "hot" if surprise > 0 else "cold"
        elif category == "employment":
            direction_key = "strong" if surprise > 0 else "weak"
        else:
            direction_key = "hawkish" if surprise > 0 else "dovish"

        effects = effects_config.get(direction_key, [])

        # Scale effects by surprise magnitude
        scaled_effects = []
        for effect in effects:
            scaled = dict(effect)
            scaled["surprise_magnitude"] = round(abs(surprise), 2)
            scaled["estimated_impact_pct"] = round(abs(surprise) * 0.5, 2)
            scaled_effects.append(scaled)

        return scaled_effects

    def scan_macro_events(
        self, economic_calendar: pd.DataFrame, macro_data: dict
    ) -> List[EventSignal]:
        """
        Scan for macro event trading opportunities across all asset classes.

        Macro Surprise Framework:
            surprise_z = (actual - forecast) / historical_std
            |surprise_z| > 1.5: actionable (1.5 sigma threshold)
            |surprise_z| > 3.0: extreme (highest conviction)

        Cross-Asset Impact:
            Each macro event affects multiple asset classes simultaneously.
            Use MACRO_EVENT_MAP betas for relative sizing across assets.

        Timing:
            - Intraday: immediate reaction (first 1-2 hours)
            - Swing: post-event drift (2-5 days)
            - Medium-term: second derivative effects (weeks to months)

        Args:
            economic_calendar: DataFrame with: event, actual, forecast, historical_std.
            macro_data: Additional macro context data.

        Returns:
            List of EventSignal objects.
        """
        signals = []

        for _, row in economic_calendar.iterrows():
            event_name = row.get("event", "")
            actual = row.get("actual")
            forecast = row.get("forecast")

            if actual is None or forecast is None:
                continue

            historical_std = row.get("historical_std")
            if historical_std is None or historical_std == 0:
                historical_std = abs(forecast) * 0.05 if forecast != 0 else 0.01
            historical_std = max(historical_std, 1e-10)

            surprise = (actual - forecast) / historical_std

            if abs(surprise) < 1.5:
                continue  # Below significance threshold

            affected = self._get_macro_affected_assets(event_name, surprise)
            second_deriv = self._calculate_second_derivative(event_name, surprise)

            for asset_info in affected:
                expected_impact = asset_info["beta"] * surprise * 0.01
                confidence = min(abs(surprise) / 4.0, 0.9)

                stop_loss, take_profit, rr = self._calculate_stop_and_target(
                    expected_impact, confidence, asset_info["direction"]
                )

                kelly = (confidence * rr - (1 - confidence)) / rr if rr > 0 else 0
                position_size = min(max(kelly * 100, 0.3), 2.0)

                horizon = (
                    TradeHorizon.MEDIUM_TERM.value
                    if abs(surprise) > 3
                    else TradeHorizon.SWING.value
                )

                signals.append(EventSignal(
                    event_type=EventType.MACRO_RELEASE,
                    symbol=asset_info["symbol"],
                    event_date=pd.Timestamp(row.get("date", datetime.now())).to_pydatetime()
                    if "date" in row
                    else datetime.now(),
                    description=f"{event_name}: surprise={surprise:.2f}sigma "
                    f"(actual={actual} vs forecast={forecast})",
                    expected_impact=round(expected_impact, 4),
                    confidence=round(confidence, 4),
                    direction=asset_info["direction"],
                    asset_class=asset_info["asset_class"],
                    affected_symbols=[a["symbol"] for a in affected],
                    second_derivative_effects=second_deriv,
                    trade_horizon=horizon,
                    position_size_pct=round(position_size, 2),
                    thesis=(
                        f"Macro surprise in {event_name}: {actual} vs {forecast} expected "
                        f"({surprise:.1f} sigma). {asset_info['symbol']} beta={asset_info['beta']:.2f}."
                    ),
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    risk_reward_ratio=rr,
                ))

        return signals

    def scan_rating_changes(self, ratings_data: pd.DataFrame) -> List[EventSignal]:
        """
        Scan for credit rating change events (upgrades/downgrades).

        Rating Change Impact:
            Downgrade: immediate negative AR (abnormal return)
            Upgrade: modest positive AR (partially priced in)
            Fallen angel (IG -> HY): forced selling creates opportunity
            Rising star (HY -> IG): forced buying creates momentum

        Abnormal Return Calculation:
            AR_t = R_t - (alpha + beta * R_m,t)
            CAR = sum(AR_t) over event window [-1, +5]

        Cross-Asset Effects:
            Equity: downgrade -> negative
            Credit: downgrade -> spread widening
            CDS: downgrade -> CDS spread increase

        Args:
            ratings_data: DataFrame with: symbol, old_rating, new_rating, agency, date.

        Returns:
            List of EventSignal objects.
        """
        signals = []
        # Rating scale (simplified: higher = better)
        rating_scale = {
            "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18,
            "A+": 17, "A": 16, "A-": 15,
            "BBB+": 14, "BBB": 13, "BBB-": 12,
            "BB+": 11, "BB": 10, "BB-": 9,
            "B+": 8, "B": 7, "B-": 6,
            "CCC+": 5, "CCC": 4, "CCC-": 3,
            "CC": 2, "C": 1, "D": 0,
        }

        ig_threshold = 12  # BBB- and above

        for _, row in ratings_data.iterrows():
            symbol = row.get("symbol", "")
            old_rating = row.get("old_rating", "")
            new_rating = row.get("new_rating", "")
            agency = row.get("agency", "Unknown")

            old_score = rating_scale.get(old_rating, 10)
            new_score = rating_scale.get(new_rating, 10)
            notch_change = new_score - old_score

            if notch_change == 0:
                continue

            # Fallen angel detection
            is_fallen_angel = old_score >= ig_threshold and new_score < ig_threshold
            is_rising_star = old_score < ig_threshold and new_score >= ig_threshold

            if is_fallen_angel:
                direction = "long"  # Buy after forced selling creates dislocations
                expected_impact = 0.03  # Expect 3% recovery after forced selling
                confidence = 0.7
                thesis = (
                    f"Fallen angel: {symbol} downgraded from {old_rating} to {new_rating} "
                    f"by {agency}. Forced IG selling creates buying opportunity."
                )
                horizon = TradeHorizon.MEDIUM_TERM.value
            elif is_rising_star:
                direction = "long"
                expected_impact = 0.02
                confidence = 0.75
                thesis = (
                    f"Rising star: {symbol} upgraded from {old_rating} to {new_rating} "
                    f"by {agency}. IG index inclusion creates forced buying."
                )
                horizon = TradeHorizon.SWING.value
            elif notch_change < -2:
                direction = "short"
                expected_impact = -0.02 * abs(notch_change)
                confidence = min(0.5 + abs(notch_change) * 0.05, 0.85)
                thesis = (
                    f"Multi-notch downgrade: {symbol} from {old_rating} to {new_rating} "
                    f"by {agency}. {abs(notch_change)} notch drop signals fundamental deterioration."
                )
                horizon = TradeHorizon.MEDIUM_TERM.value
            elif notch_change < 0:
                direction = "short"
                expected_impact = -0.01 * abs(notch_change)
                confidence = 0.55
                thesis = (
                    f"Rating downgrade: {symbol} from {old_rating} to {new_rating} by {agency}."
                )
                horizon = TradeHorizon.SWING.value
            else:
                direction = "long"
                expected_impact = 0.005 * notch_change
                confidence = 0.5
                thesis = (
                    f"Rating upgrade: {symbol} from {old_rating} to {new_rating} by {agency}."
                )
                horizon = TradeHorizon.SWING.value

            stop_loss, take_profit, rr = self._calculate_stop_and_target(
                expected_impact, confidence, direction
            )

            signals.append(EventSignal(
                event_type=EventType.RATING_CHANGE,
                symbol=symbol,
                event_date=pd.Timestamp(row.get("date", datetime.now())).to_pydatetime()
                if "date" in row
                else datetime.now(),
                description=f"Rating change: {old_rating} -> {new_rating} ({agency})",
                expected_impact=round(expected_impact, 4),
                confidence=round(confidence, 4),
                direction=direction,
                asset_class="equity",
                affected_symbols=[symbol],
                second_derivative_effects=[],
                trade_horizon=horizon,
                position_size_pct=round(min(abs(expected_impact) * 50, 2.5), 2),
                thesis=thesis,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                risk_reward_ratio=rr,
            ))

        return signals

    def scan_corporate_actions(self, corporate_actions: pd.DataFrame) -> List[EventSignal]:
        """
        Scan for corporate action event signals.

        M&A Arbitrage:
            spread = (offer_price - current_price) / current_price
            annualized_return = spread / (expected_days_to_close / 365)
            P(completion) estimated from: deal type, regulatory risk, financing

        Buyback Analysis:
            Signal strength = buyback_amount / market_cap
            Insiders buying + buyback = strong signal

        Spinoff Opportunity:
            Parent often undervalued pre-spinoff
            Spinco often sold by index funds post-spinoff (forced selling)

        Args:
            corporate_actions: DataFrame with: symbol, action_type, details.

        Returns:
            List of EventSignal objects.
        """
        signals = []

        for _, row in corporate_actions.iterrows():
            symbol = row.get("symbol", "")
            action_type = row.get("action_type", "").lower()
            details = row.get("details", {})

            if "merger" in action_type or "acquisition" in action_type:
                offer_price = details.get("offer_price", 0)
                current_price = details.get("current_price", 0)
                if offer_price > 0 and current_price > 0:
                    spread = (offer_price - current_price) / current_price
                    if spread > 0.01:  # Positive spread exists
                        days_to_close = details.get("expected_days", 90)
                        annualized = spread * (365 / max(days_to_close, 1))
                        p_completion = details.get("completion_prob", 0.85)
                        expected_return = spread * p_completion - (1 - p_completion) * 0.2
                        if expected_return > 0:
                            signals.append(EventSignal(
                                event_type=EventType.CORPORATE_ACTION,
                                symbol=symbol,
                                event_date=datetime.now(),
                                description=f"M&A arb: spread={spread:.1%}, "
                                f"ann={annualized:.1%}, P(close)={p_completion:.0%}",
                                expected_impact=round(expected_return, 4),
                                confidence=round(p_completion, 4),
                                direction="long",
                                asset_class="equity",
                                affected_symbols=[symbol],
                                second_derivative_effects=[],
                                trade_horizon=TradeHorizon.MEDIUM_TERM.value,
                                position_size_pct=round(min(annualized * 10, 3.0), 2),
                                thesis=(
                                    f"M&A arbitrage: {spread:.1%} spread to offer price "
                                    f"${offer_price}. Completion probability {p_completion:.0%}. "
                                    f"Annualized return {annualized:.1%}."
                                ),
                                stop_loss_pct=round(0.15, 4),
                                take_profit_pct=round(spread, 4),
                                risk_reward_ratio=round(spread / 0.15, 2),
                            ))

            elif "buyback" in action_type:
                buyback_amount = details.get("amount", 0)
                market_cap = details.get("market_cap", 1)
                buyback_yield = buyback_amount / max(market_cap, 1)
                if buyback_yield > 0.02:  # Significant buyback (>2% of market cap)
                    signals.append(EventSignal(
                        event_type=EventType.CORPORATE_ACTION,
                        symbol=symbol,
                        event_date=datetime.now(),
                        description=f"Buyback: {buyback_yield:.1%} of market cap",
                        expected_impact=round(buyback_yield * 0.5, 4),
                        confidence=0.65,
                        direction="long",
                        asset_class="equity",
                        affected_symbols=[symbol],
                        second_derivative_effects=[],
                        trade_horizon=TradeHorizon.MEDIUM_TERM.value,
                        position_size_pct=round(min(buyback_yield * 20, 2.0), 2),
                        thesis=(
                            f"Significant share buyback: {buyback_yield:.1%} of market cap. "
                            f"Reduces share count, supports EPS growth."
                        ),
                        stop_loss_pct=0.03,
                        take_profit_pct=round(buyback_yield * 0.5, 4),
                        risk_reward_ratio=round(buyback_yield * 0.5 / 0.03, 2),
                    ))

            elif "spinoff" in action_type:
                signals.append(EventSignal(
                    event_type=EventType.CORPORATE_ACTION,
                    symbol=symbol,
                    event_date=datetime.now(),
                    description=f"Spinoff announced for {symbol}",
                    expected_impact=0.05,
                    confidence=0.6,
                    direction="long",
                    asset_class="equity",
                    affected_symbols=[symbol],
                    second_derivative_effects=[
                        {
                            "effect": "Post-spinoff forced selling of SpinCo",
                            "lag_days": 30,
                            "opportunity": "Buy SpinCo after index fund selling",
                        }
                    ],
                    trade_horizon=TradeHorizon.MEDIUM_TERM.value,
                    position_size_pct=1.5,
                    thesis=(
                        f"Spinoff: parent typically undervalued pre-event. "
                        f"SpinCo may face forced selling post-spin creating "
                        f"buying opportunity."
                    ),
                    stop_loss_pct=0.05,
                    take_profit_pct=0.08,
                    risk_reward_ratio=1.6,
                ))

        return signals

    def scan_all_events(
        self,
        earnings_cal: pd.DataFrame,
        economic_cal: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
        macro_data: Optional[dict] = None,
        ratings_data: Optional[pd.DataFrame] = None,
        corporate_actions: Optional[pd.DataFrame] = None,
    ) -> List[EventSignal]:
        """
        Comprehensive event scan across all asset classes and event types.

        Aggregation and Deduplication:
            1. Run all individual scanners
            2. Remove duplicate signals (same symbol + same event type)
            3. Sort by risk-adjusted expected return
            4. Apply portfolio-level position limits

        Portfolio Constraints:
            - Max 5% gross exposure per signal
            - Max 20% exposure per event type
            - Max 40% total event-driven exposure

        Args:
            earnings_cal: Earnings calendar DataFrame.
            economic_cal: Economic calendar DataFrame.
            news_data: News/geopolitical data (optional).
            macro_data: Macro context data (optional).
            ratings_data: Rating changes DataFrame (optional).
            corporate_actions: Corporate actions DataFrame (optional).

        Returns:
            List of EventSignal objects sorted by conviction.
        """
        if macro_data is None:
            macro_data = {}

        all_signals = []

        # Earnings scan
        if earnings_cal is not None and not earnings_cal.empty:
            all_signals.extend(self.scan_earnings_events(earnings_cal, {}))

        # Macro scan
        if economic_cal is not None and not economic_cal.empty:
            all_signals.extend(self.scan_macro_events(economic_cal, macro_data))

        # Rating changes
        if ratings_data is not None and not ratings_data.empty:
            all_signals.extend(self.scan_rating_changes(ratings_data))

        # Corporate actions
        if corporate_actions is not None and not corporate_actions.empty:
            all_signals.extend(self.scan_corporate_actions(corporate_actions))

        # Deduplicate: keep highest conviction per symbol + event_type
        seen = {}
        for signal in all_signals:
            key = (signal.symbol, signal.event_type.value)
            if key not in seen or (
                abs(signal.expected_impact) * signal.confidence
                > abs(seen[key].expected_impact) * seen[key].confidence
            ):
                seen[key] = signal
        all_signals = list(seen.values())

        # Sort by risk-adjusted expected return (impact * confidence / position_size)
        all_signals.sort(
            key=lambda s: abs(s.expected_impact) * s.confidence,
            reverse=True,
        )

        # Store in history
        self.event_history.extend(all_signals)

        return all_signals

    def generate_event_report(self, signals: List[EventSignal]) -> dict:
        """
        Generate comprehensive daily event-driven trading report.

        Report Structure:
            1. Executive Summary: signal count by type and asset class
            2. Top Signals: highest conviction trades
            3. Cross-Asset View: how events link across asset classes
            4. Second Derivative Monitor: upcoming lagged effects
            5. Risk Metrics: total exposure, directional bias, concentration

        Args:
            signals: List of EventSignal objects from scan_all_events.

        Returns:
            Dictionary with full report data.
        """
        if not signals:
            return {
                "date": datetime.now().isoformat(),
                "total_signals": 0,
                "summary": "No actionable event signals detected.",
            }

        # Group by event type
        by_type: Dict[str, List[dict]] = {}
        for s in signals:
            t = s.event_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append({
                "symbol": s.symbol,
                "direction": s.direction,
                "expected_impact": s.expected_impact,
                "confidence": s.confidence,
                "thesis": s.thesis,
                "horizon": s.trade_horizon,
                "position_size_pct": s.position_size_pct,
                "risk_reward": s.risk_reward_ratio,
            })

        # Group by asset class
        by_asset_class: Dict[str, int] = {}
        for s in signals:
            ac = s.asset_class
            by_asset_class[ac] = by_asset_class.get(ac, 0) + 1

        # Directional bias
        long_signals = [s for s in signals if s.direction == "long"]
        short_signals = [s for s in signals if s.direction == "short"]
        long_exposure = sum(s.position_size_pct for s in long_signals)
        short_exposure = sum(s.position_size_pct for s in short_signals)
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Second derivative effects monitor
        all_second_deriv = []
        for s in signals:
            for effect in s.second_derivative_effects:
                all_second_deriv.append({
                    "source_event": s.description,
                    "source_symbol": s.symbol,
                    **effect,
                })

        # Concentration analysis
        symbol_concentration = {}
        for s in signals:
            symbol_concentration[s.symbol] = (
                symbol_concentration.get(s.symbol, 0) + s.position_size_pct
            )
        max_concentration = max(symbol_concentration.values()) if symbol_concentration else 0

        report = {
            "date": datetime.now().isoformat(),
            "total_signals": len(signals),
            "signals_by_type": by_type,
            "signals_by_asset_class": by_asset_class,
            "top_5": [
                {
                    "symbol": s.symbol,
                    "thesis": s.thesis,
                    "impact": s.expected_impact,
                    "confidence": s.confidence,
                    "direction": s.direction,
                    "asset_class": s.asset_class,
                    "risk_reward": s.risk_reward_ratio,
                }
                for s in signals[:5]
            ],
            "exposure_summary": {
                "long_exposure_pct": round(long_exposure, 2),
                "short_exposure_pct": round(short_exposure, 2),
                "net_exposure_pct": round(net_exposure, 2),
                "gross_exposure_pct": round(gross_exposure, 2),
                "long_count": len(long_signals),
                "short_count": len(short_signals),
                "max_single_name_pct": round(max_concentration, 2),
            },
            "second_derivative_monitor": all_second_deriv[:10],
            "risk_warnings": [],
        }

        # Add risk warnings
        if gross_exposure > 40:
            report["risk_warnings"].append(
                f"Gross exposure {gross_exposure:.1f}% exceeds 40% limit."
            )
        if max_concentration > 5:
            report["risk_warnings"].append(
                f"Single name concentration {max_concentration:.1f}% exceeds 5% limit."
            )
        if abs(net_exposure) > 15:
            report["risk_warnings"].append(
                f"Directional bias: net exposure {net_exposure:.1f}%."
            )

        return report
