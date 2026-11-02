"""
Financial distress prediction engine using full universe data from OpenBB.

Altman Z-Score: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value Equity / Total Liabilities
    X5 = Sales / Total Assets
    Z > 2.99: Safe | 1.81-2.99: Grey | Z < 1.81: Distressed

Ohlson O-Score: logit model P(distress) = 1/(1 + exp(-O))
    O = -1.32 - 0.407*log(TA/GNP) + 6.03*(TL/TA) - 1.43*(WC/TA) + 0.076*(CL/CA)
        - 1.72*X - 2.37*(NI/TA) - 1.83*(FFO/TL) + 0.285*Y - 0.521*(dNI)
    Where:
        TA = Total Assets, GNP = Gross National Product deflator
        TL = Total Liabilities, WC = Working Capital
        CL = Current Liabilities, CA = Current Assets
        X = 1 if TL > TA else 0
        NI = Net Income, FFO = Funds From Operations
        Y = 1 if net loss for last 2 years else 0
        dNI = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)

Merton Distance to Default:
    DD = (ln(V/D) + (mu - sigma^2/2)*T) / (sigma*sqrt(T))
    PD = N(-DD)
    Where V = asset value, D = debt face value, mu = asset drift,
    sigma = asset volatility, T = time horizon

Piotroski F-Score (0-9):
    Profitability (4 pts): ROA>0, CFO>0, dROA>0, Accrual(CFO>ROA)
    Leverage (3 pts): dLeverage<0, dLiquidity>0, No dilution
    Efficiency (2 pts): dGrossMargin>0, dAssetTurnover>0

Springate S-Score: S = 1.03*A + 3.07*B + 0.66*C + 0.40*D
    A = Working Capital / Total Assets
    B = EBIT / Total Assets
    C = EBT / Current Liabilities
    D = Sales / Total Assets
    S < 0.862: Distressed

Zmijewski X-Score: X = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) + 0.004*(CA/CL)
    P(distress) = 1/(1 + exp(-X))
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from openbb_universe import (
    get_equity_universe,
    get_full_universe,
    get_fundamentals,
    get_historical,
    classify_by_gics,
    AssetClass,
    GICSSector,
    SP500_TOP_50,
)


class DistressLevel(Enum):
    SAFE = "safe"
    GREY_ZONE = "grey_zone"
    DISTRESSED = "distressed"
    CRITICAL = "critical"


@dataclass
class DistressAssessment:
    """Complete distress assessment for a single security."""
    symbol: str
    z_score: float
    o_score: float
    f_score: int
    distance_to_default: float
    probability_of_default: float
    springate_s: float
    zmijewski_x: float
    distress_level: DistressLevel
    recovery_probability: float
    trade_recommendation: str  # "long_recovery", "short_deteriorating", "avoid"
    thesis: str
    sector: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DistressPredictionEngine:
    """
    Multi-model financial distress prediction engine.

    Combines five complementary models:
    1. Altman Z-Score (discriminant analysis, 1968)
    2. Ohlson O-Score (logistic regression, 1980)
    3. Merton Distance to Default (structural model, 1974)
    4. Piotroski F-Score (accounting-based, 2000)
    5. Springate S-Score (MDA, 1978)
    6. Zmijewski X-Score (probit, 1984)

    Ensemble approach: weighted average of model signals for robust prediction.
    """

    # Default GNP deflator (in billions) for O-Score normalization
    DEFAULT_GNP_DEFLATOR = 25_000_000_000_000  # ~$25T US GDP

    def __init__(self, gnp_deflator: Optional[float] = None):
        self.gnp_deflator = gnp_deflator or self.DEFAULT_GNP_DEFLATOR
        self.assessments: List[DistressAssessment] = []

    def calculate_altman_z(self, financials: dict) -> float:
        """
        Calculate the Altman Z-Score for bankruptcy prediction.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        Interpretation:
            Z > 2.99  -> Safe Zone (low probability of bankruptcy)
            1.81-2.99 -> Grey Zone (moderate risk)
            Z < 1.81  -> Distress Zone (high probability of bankruptcy)

        For private firms (Z'-Score):
            Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
            (X4 uses book value of equity instead of market value)

        Args:
            financials: Dictionary with financial statement data.

        Returns:
            Z-Score as float.
        """
        total_assets = max(financials.get("total_assets", 1), 1)
        total_liabilities = max(financials.get("total_liabilities", 1), 1)

        x1 = financials.get("working_capital", 0) / total_assets
        x2 = financials.get("retained_earnings", 0) / total_assets
        x3 = financials.get("ebit", 0) / total_assets
        x4 = financials.get("market_cap", 0) / total_liabilities
        x5 = financials.get("revenue", 0) / total_assets

        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return round(z_score, 4)

    def calculate_ohlson_o(self, financials: dict, prev_financials: Optional[dict] = None) -> float:
        """
        Calculate the Ohlson O-Score for bankruptcy prediction.

        O = -1.32 - 0.407*log(TA/GNP) + 6.03*(TL/TA) - 1.43*(WC/TA)
            + 0.076*(CL/CA) - 1.72*X - 2.37*(NI/TA) - 1.83*(FFO/TL)
            + 0.285*Y - 0.521*(dNI)

        P(bankruptcy) = 1 / (1 + exp(-O))
        O > 0.5: higher probability of distress

        Args:
            financials: Current period financial data.
            prev_financials: Previous period financial data (for trend variables).

        Returns:
            O-Score as float.
        """
        if prev_financials is None:
            prev_financials = {}

        total_assets = max(financials.get("total_assets", 1), 1)
        total_liabilities = financials.get("total_liabilities", 0)
        working_capital = financials.get("working_capital", 0)
        current_liabilities = max(financials.get("current_liabilities", 1), 1)
        current_assets = max(financials.get("current_assets", 1), 1)
        net_income = financials.get("net_income", 0)
        cfo = financials.get("cfo", 0)

        # Funds from operations (approximation: NI + depreciation ≈ CFO)
        ffo = cfo if cfo != 0 else net_income

        # X = 1 if total liabilities > total assets (negative equity)
        x_indicator = 1.0 if total_liabilities > total_assets else 0.0

        # Y = 1 if net loss in both current and previous year
        prev_ni = prev_financials.get("net_income", 0)
        y_indicator = 1.0 if (net_income < 0 and prev_ni < 0) else 0.0

        # Change in net income (scaled)
        ni_abs = abs(net_income)
        prev_ni_abs = abs(prev_ni)
        denom = ni_abs + prev_ni_abs
        if denom > 0:
            delta_ni = (net_income - prev_ni) / denom
        else:
            delta_ni = 0.0

        # GNP-adjusted size
        log_ta_gnp = np.log(max(total_assets / self.gnp_deflator, 1e-10))

        o_score = (
            -1.32
            - 0.407 * log_ta_gnp
            + 6.03 * (total_liabilities / total_assets)
            - 1.43 * (working_capital / total_assets)
            + 0.076 * (current_liabilities / current_assets)
            - 1.72 * x_indicator
            - 2.37 * (net_income / total_assets)
            - 1.83 * (ffo / max(total_liabilities, 1))
            + 0.285 * y_indicator
            - 0.521 * delta_ni
        )

        return round(o_score, 4)

    def calculate_merton_dd(
        self,
        equity_value: float,
        debt: float,
        equity_vol: float,
        risk_free: float,
        T: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate Merton's Distance to Default and Probability of Default.

        Structural Model (Merton, 1974):
            Firm value V follows geometric Brownian motion: dV = mu*V*dt + sigma_V*V*dW
            Equity = Call option on firm assets: E = V*N(d1) - D*e^{-rT}*N(d2)
            d1 = (ln(V/D) + (r + sigma_V^2/2)*T) / (sigma_V*sqrt(T))
            d2 = d1 - sigma_V*sqrt(T)

        Distance to Default: DD = (ln(V/D) + (mu - sigma_V^2/2)*T) / (sigma_V*sqrt(T))
        Probability of Default: PD = N(-DD)

        The iterative approach to solve for V and sigma_V:
            sigma_V = sigma_E * (E / V) * (1 / N(d1))
            We approximate: sigma_V ≈ sigma_E * E / (E + D)

        Args:
            equity_value: Market value of equity (market cap).
            debt: Face value of debt (total liabilities).
            equity_vol: Annualized equity volatility.
            risk_free: Risk-free rate (annualized).
            T: Time horizon in years (default 1.0).

        Returns:
            Tuple of (distance_to_default, probability_of_default).
        """
        from scipy.stats import norm

        if equity_value <= 0 or debt <= 0 or equity_vol <= 0 or T <= 0:
            return (0.0, 1.0)

        # Approximate asset value and asset volatility
        asset_value = equity_value + debt
        asset_vol = equity_vol * (equity_value / asset_value)

        # Ensure asset_vol is positive and reasonable
        asset_vol = max(asset_vol, 0.01)

        # Distance to Default
        dd_numerator = np.log(asset_value / debt) + (risk_free - 0.5 * asset_vol ** 2) * T
        dd_denominator = asset_vol * np.sqrt(T)
        dd = dd_numerator / dd_denominator

        # Probability of Default
        pd_val = norm.cdf(-dd)

        return (round(float(dd), 4), round(float(pd_val), 6))

    def calculate_f_score(self, financials: dict, prev_financials: dict) -> int:
        """
        Calculate Piotroski F-Score (0-9 scale).

        Profitability Signals (4 points):
            F1: ROA > 0 (positive net income / total assets)
            F2: CFO > 0 (positive operating cash flow)
            F3: Delta ROA > 0 (improving return on assets)
            F4: Accrual: CFO > ROA * Total Assets (quality of earnings)

        Leverage/Liquidity Signals (3 points):
            F5: Delta Leverage < 0 (decreasing long-term debt ratio)
            F6: Delta Liquidity > 0 (improving current ratio)
            F7: No equity dilution (shares outstanding not increased)

        Operating Efficiency Signals (2 points):
            F8: Delta Gross Margin > 0 (improving margins)
            F9: Delta Asset Turnover > 0 (improving efficiency)

        Interpretation:
            8-9: Strong (long candidates)
            5-7: Moderate
            0-4: Weak (short or avoid candidates)

        Args:
            financials: Current period financial data.
            prev_financials: Previous period financial data.

        Returns:
            F-Score as integer (0-9).
        """
        score = 0
        total_assets = max(financials.get("total_assets", 1), 1)
        prev_total_assets = max(prev_financials.get("total_assets", 1), 1)

        # Current ROA
        roa = financials.get("net_income", 0) / total_assets
        prev_roa = prev_financials.get("net_income", 0) / prev_total_assets

        # F1: ROA > 0
        if roa > 0:
            score += 1

        # F2: CFO > 0
        if financials.get("cfo", 0) > 0:
            score += 1

        # F3: Delta ROA > 0
        if roa > prev_roa:
            score += 1

        # F4: Accrual (CFO > net income, indicating earnings quality)
        if financials.get("cfo", 0) > financials.get("net_income", 0):
            score += 1

        # F5: Delta Leverage < 0 (long-term debt / total assets decreasing)
        current_leverage = financials.get("long_term_debt", 0) / total_assets
        prev_leverage = prev_financials.get("long_term_debt", 0) / prev_total_assets
        if current_leverage < prev_leverage:
            score += 1

        # F6: Delta Current Ratio > 0 (improving liquidity)
        if financials.get("current_ratio", 0) > prev_financials.get("current_ratio", 0):
            score += 1

        # F7: No equity dilution
        if financials.get("shares_outstanding", 0) <= prev_financials.get("shares_outstanding", 0):
            score += 1

        # F8: Delta Gross Margin > 0
        current_gm = financials.get("gross_margin", 0)
        prev_gm = prev_financials.get("gross_margin", 0)
        if current_gm > prev_gm:
            score += 1

        # F9: Delta Asset Turnover > 0
        current_at = financials.get("asset_turnover", 0)
        prev_at = prev_financials.get("asset_turnover", 0)
        if current_at > prev_at:
            score += 1

        return score

    def calculate_springate_s(self, financials: dict) -> float:
        """
        Calculate Springate S-Score for distress prediction.

        S = 1.03*A + 3.07*B + 0.66*C + 0.40*D
        Where:
            A = Working Capital / Total Assets
            B = EBIT / Total Assets
            C = EBT / Current Liabilities
            D = Sales / Total Assets

        S < 0.862: Company classified as distressed/failed

        Args:
            financials: Dictionary with financial statement data.

        Returns:
            Springate S-Score as float.
        """
        total_assets = max(financials.get("total_assets", 1), 1)
        current_liabilities = max(financials.get("current_liabilities", 1), 1)

        a = financials.get("working_capital", 0) / total_assets
        b = financials.get("ebit", 0) / total_assets
        # EBT approximation: EBIT - interest expense; use EBIT if interest not available
        ebt = financials.get("ebt", financials.get("ebit", 0))
        c = ebt / current_liabilities
        d = financials.get("revenue", 0) / total_assets

        s_score = 1.03 * a + 3.07 * b + 0.66 * c + 0.40 * d
        return round(s_score, 4)

    def calculate_zmijewski_x(self, financials: dict) -> float:
        """
        Calculate Zmijewski X-Score (probit model).

        X = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) + 0.004*(CA/CL)
        P(distress) = 1/(1 + exp(-X))

        Args:
            financials: Dictionary with financial statement data.

        Returns:
            Zmijewski X-Score as float.
        """
        total_assets = max(financials.get("total_assets", 1), 1)
        total_liabilities = financials.get("total_liabilities", 0)
        current_assets = financials.get("current_assets", 0)
        current_liabilities = max(financials.get("current_liabilities", 1), 1)
        net_income = financials.get("net_income", 0)

        x_score = (
            -4.336
            - 4.513 * (net_income / total_assets)
            + 5.679 * (total_liabilities / total_assets)
            + 0.004 * (current_assets / current_liabilities)
        )
        return round(x_score, 4)

    def _classify_distress_level(
        self, z_score: float, o_score: float, f_score: int, pd_val: float
    ) -> DistressLevel:
        """
        Classify distress level using ensemble of all model outputs.

        Ensemble Logic:
            - Count how many models indicate distress
            - Use weighted voting: Z-Score (0.3), O-Score (0.2), F-Score (0.2), PD (0.3)
            - CRITICAL: 3+ models agree on distress
            - DISTRESSED: 2+ models indicate distress
            - GREY_ZONE: 1 model indicates distress or borderline signals
            - SAFE: No models indicate distress

        Args:
            z_score: Altman Z-Score
            o_score: Ohlson O-Score
            f_score: Piotroski F-Score
            pd_val: Merton probability of default

        Returns:
            DistressLevel enum value.
        """
        distress_signals = 0

        # Z-Score signal
        if z_score < 1.81:
            distress_signals += 2  # Strong signal
        elif z_score < 2.99:
            distress_signals += 1  # Weak signal

        # O-Score signal (probability of bankruptcy)
        o_prob = 1.0 / (1.0 + np.exp(-o_score))
        if o_prob > 0.7:
            distress_signals += 2
        elif o_prob > 0.5:
            distress_signals += 1

        # F-Score signal (low F-Score = weak fundamentals)
        if f_score <= 2:
            distress_signals += 2
        elif f_score <= 4:
            distress_signals += 1

        # Merton PD signal
        if pd_val > 0.3:
            distress_signals += 2
        elif pd_val > 0.1:
            distress_signals += 1

        if distress_signals >= 6:
            return DistressLevel.CRITICAL
        elif distress_signals >= 4:
            return DistressLevel.DISTRESSED
        elif distress_signals >= 2:
            return DistressLevel.GREY_ZONE
        else:
            return DistressLevel.SAFE

    def _calculate_recovery_probability(
        self, f_score: int, z_score: float, prev_z_score: Optional[float] = None
    ) -> float:
        """
        Estimate probability of recovery from distress.

        Recovery indicators:
            1. Rising F-Score (improving fundamentals)
            2. Z-Score trend (improving or stable)
            3. Absolute F-Score level (higher = more likely to recover)

        Simple logistic model:
            P(recovery) = sigmoid(0.3 * f_score + 0.5 * delta_z - 1.5)

        Args:
            f_score: Current Piotroski F-Score.
            z_score: Current Altman Z-Score.
            prev_z_score: Previous period Z-Score (if available).

        Returns:
            Recovery probability as float in [0, 1].
        """
        delta_z = (z_score - prev_z_score) if prev_z_score is not None else 0.0
        logit = 0.3 * f_score + 0.5 * delta_z - 1.5
        recovery_prob = 1.0 / (1.0 + np.exp(-logit))
        return round(float(recovery_prob), 4)

    def _generate_recommendation(
        self, distress_level: DistressLevel, recovery_prob: float, f_score: int, z_score: float
    ) -> Tuple[str, str]:
        """
        Generate trade recommendation and investment thesis.

        Decision Matrix:
            SAFE + high F-score -> "hold" (no distress play)
            GREY_ZONE + rising metrics -> "long_recovery" (turnaround candidate)
            DISTRESSED + high recovery prob -> "long_recovery" (deep value)
            DISTRESSED + low recovery prob -> "short_deteriorating"
            CRITICAL -> "avoid" or "short_deteriorating"

        Args:
            distress_level: Classified distress level.
            recovery_prob: Estimated recovery probability.
            f_score: Piotroski F-Score.
            z_score: Altman Z-Score.

        Returns:
            Tuple of (recommendation, thesis).
        """
        if distress_level == DistressLevel.SAFE:
            return (
                "hold",
                f"Company is financially healthy (Z={z_score:.2f}, F={f_score}). "
                f"No distress-based trade opportunity.",
            )

        if distress_level == DistressLevel.GREY_ZONE:
            if recovery_prob > 0.6 and f_score >= 5:
                return (
                    "long_recovery",
                    f"Grey zone with strong recovery signals (F={f_score}, "
                    f"recovery_prob={recovery_prob:.1%}). Potential turnaround candidate.",
                )
            elif recovery_prob < 0.3:
                return (
                    "short_deteriorating",
                    f"Grey zone with weak recovery outlook (F={f_score}, "
                    f"recovery_prob={recovery_prob:.1%}). Risk of further deterioration.",
                )
            else:
                return (
                    "avoid",
                    f"Grey zone with mixed signals (F={f_score}, "
                    f"recovery_prob={recovery_prob:.1%}). Insufficient conviction.",
                )

        if distress_level == DistressLevel.DISTRESSED:
            if recovery_prob > 0.5 and f_score >= 4:
                return (
                    "long_recovery",
                    f"Distressed but improving fundamentals (Z={z_score:.2f}, F={f_score}, "
                    f"recovery_prob={recovery_prob:.1%}). Deep value recovery play.",
                )
            else:
                return (
                    "short_deteriorating",
                    f"Distressed with weak recovery prospects (Z={z_score:.2f}, F={f_score}, "
                    f"recovery_prob={recovery_prob:.1%}). Further downside likely.",
                )

        # CRITICAL
        if recovery_prob > 0.4 and f_score >= 5:
            return (
                "long_recovery",
                f"Critical distress but showing recovery signs (Z={z_score:.2f}, F={f_score}). "
                f"High-risk/high-reward turnaround. Position size carefully.",
            )
        else:
            return (
                "avoid",
                f"Critical distress with low recovery probability (Z={z_score:.2f}, "
                f"F={f_score}, recovery_prob={recovery_prob:.1%}). Bankruptcy risk elevated.",
            )

    def assess_single(
        self,
        symbol: str,
        financials: dict,
        prev_financials: Optional[dict] = None,
        equity_vol: float = 0.3,
        risk_free: float = 0.05,
        prev_z_score: Optional[float] = None,
    ) -> DistressAssessment:
        """
        Perform complete distress assessment on a single security.

        Runs all models and generates ensemble classification + recommendation.

        Args:
            symbol: Ticker symbol.
            financials: Current period financial data.
            prev_financials: Previous period financial data.
            equity_vol: Annualized equity volatility.
            risk_free: Risk-free rate.
            prev_z_score: Previous period Z-Score for trend analysis.

        Returns:
            DistressAssessment dataclass with all model outputs.
        """
        if prev_financials is None:
            prev_financials = financials

        z_score = self.calculate_altman_z(financials)
        o_score = self.calculate_ohlson_o(financials, prev_financials)
        f_score = self.calculate_f_score(financials, prev_financials)
        springate_s = self.calculate_springate_s(financials)
        zmijewski_x = self.calculate_zmijewski_x(financials)

        equity_value = financials.get("market_cap", 0)
        debt = financials.get("total_liabilities", 0)
        dd, pd_val = self.calculate_merton_dd(equity_value, debt, equity_vol, risk_free)

        distress_level = self._classify_distress_level(z_score, o_score, f_score, pd_val)
        recovery_prob = self._calculate_recovery_probability(f_score, z_score, prev_z_score)
        recommendation, thesis = self._generate_recommendation(
            distress_level, recovery_prob, f_score, z_score
        )

        # Determine sector
        sector = None
        if symbol in SP500_TOP_50:
            _, gics_sector, _ = SP500_TOP_50[symbol]
            sector = gics_sector.value

        assessment = DistressAssessment(
            symbol=symbol,
            z_score=z_score,
            o_score=o_score,
            f_score=f_score,
            distance_to_default=dd,
            probability_of_default=pd_val,
            springate_s=springate_s,
            zmijewski_x=zmijewski_x,
            distress_level=distress_level,
            recovery_probability=recovery_prob,
            trade_recommendation=recommendation,
            thesis=thesis,
            sector=sector,
        )

        self.assessments.append(assessment)
        return assessment

    def scan_universe_for_distress(self, universe_data: Dict[str, dict]) -> List[DistressAssessment]:
        """
        Scan entire equity universe for distressed securities.

        Iterates through all symbols in the provided universe data,
        runs the full assessment pipeline, and returns sorted results.

        Distress Screening Criteria:
            - Z-Score < 2.99 (not clearly safe)
            - OR O-Score probability > 0.3
            - OR F-Score < 5
            - OR Merton PD > 0.05

        Args:
            universe_data: Dict mapping symbol -> financial data dict.
                Each value should contain keys: total_assets, total_liabilities,
                working_capital, retained_earnings, ebit, revenue, market_cap,
                net_income, cfo, current_assets, current_liabilities, etc.

        Returns:
            List of DistressAssessment sorted by distress severity (worst first).
        """
        assessments = []

        for symbol, data in universe_data.items():
            current = data.get("current", data)
            prev = data.get("previous", current)
            equity_vol = data.get("equity_vol", 0.3)

            assessment = self.assess_single(
                symbol=symbol,
                financials=current,
                prev_financials=prev,
                equity_vol=equity_vol,
            )
            assessments.append(assessment)

        # Sort by distress severity: CRITICAL first, then by PD descending
        severity_order = {
            DistressLevel.CRITICAL: 0,
            DistressLevel.DISTRESSED: 1,
            DistressLevel.GREY_ZONE: 2,
            DistressLevel.SAFE: 3,
        }
        assessments.sort(
            key=lambda a: (severity_order.get(a.distress_level, 3), -a.probability_of_default)
        )

        return assessments

    def identify_recovery_candidates(self, distressed: List[DistressAssessment]) -> List[DistressAssessment]:
        """
        Filter distressed securities for recovery/turnaround candidates.

        Recovery Candidate Criteria:
            1. F-Score >= 4 (improving fundamentals)
            2. Recovery probability > 0.4
            3. Not CRITICAL distress (unless F-Score >= 6)
            4. Recommendation is "long_recovery"

        Recovery Catalysts to Look For:
            - Rising F-Score trajectory
            - Narrowing credit spreads
            - Insider buying
            - New management / restructuring
            - Sector tailwinds
            - Asset sales / deleveraging

        Args:
            distressed: List of DistressAssessment objects.

        Returns:
            Filtered list of recovery candidates, sorted by recovery probability.
        """
        candidates = []
        for assessment in distressed:
            is_candidate = False

            if assessment.distress_level == DistressLevel.CRITICAL:
                # Only include critical if very strong F-Score
                if assessment.f_score >= 6 and assessment.recovery_probability > 0.4:
                    is_candidate = True
            elif assessment.distress_level in (DistressLevel.DISTRESSED, DistressLevel.GREY_ZONE):
                if assessment.f_score >= 4 and assessment.recovery_probability > 0.4:
                    is_candidate = True

            if is_candidate:
                candidates.append(assessment)

        candidates.sort(key=lambda a: a.recovery_probability, reverse=True)
        return candidates

    def identify_short_candidates(self, distressed: List[DistressAssessment]) -> List[DistressAssessment]:
        """
        Filter distressed securities for short sale candidates.

        Short Candidate Criteria:
            1. Distress level is DISTRESSED or CRITICAL
            2. F-Score <= 3 (deteriorating fundamentals)
            3. Recovery probability < 0.3
            4. High probability of default (PD > 0.1)
            5. Falling Z-Score (if trend available)

        Short Thesis Signals:
            - Deteriorating Z-Score quarter over quarter
            - Widening CDS spreads
            - Declining revenue and margins
            - Rising leverage ratios
            - Negative operating cash flow
            - Management turnover / auditor changes
            - Going concern qualifications

        Risk Management for Shorts:
            - Position sizing: max 2% of portfolio per short
            - Stop loss: 20% above entry
            - Watch for short squeeze indicators (high short interest, low float)

        Args:
            distressed: List of DistressAssessment objects.

        Returns:
            Filtered list of short candidates, sorted by PD descending.
        """
        candidates = []
        for assessment in distressed:
            if assessment.distress_level in (DistressLevel.DISTRESSED, DistressLevel.CRITICAL):
                if assessment.f_score <= 3 and assessment.recovery_probability < 0.3:
                    candidates.append(assessment)
                elif assessment.probability_of_default > 0.2 and assessment.f_score <= 4:
                    candidates.append(assessment)

        candidates.sort(key=lambda a: a.probability_of_default, reverse=True)
        return candidates

    def generate_distress_report(self, assessments: List[DistressAssessment]) -> dict:
        """
        Generate comprehensive daily distress screening report.

        Report Structure:
            1. Executive Summary: total screened, distressed count, sector breakdown
            2. Critical Alerts: securities in critical zone
            3. Recovery Candidates: turnaround opportunities
            4. Short Candidates: deteriorating securities
            5. Sector Analysis: distress concentration by GICS sector
            6. Model Agreement: where models agree/disagree
            7. Watch List: grey zone securities to monitor

        Args:
            assessments: List of DistressAssessment objects from universe scan.

        Returns:
            Dictionary with full report data.
        """
        if not assessments:
            return {"date": datetime.now().isoformat(), "total_screened": 0, "summary": "No data"}

        # Categorize
        critical = [a for a in assessments if a.distress_level == DistressLevel.CRITICAL]
        distressed = [a for a in assessments if a.distress_level == DistressLevel.DISTRESSED]
        grey_zone = [a for a in assessments if a.distress_level == DistressLevel.GREY_ZONE]
        safe = [a for a in assessments if a.distress_level == DistressLevel.SAFE]

        recovery_candidates = self.identify_recovery_candidates(assessments)
        short_candidates = self.identify_short_candidates(assessments)

        # Sector breakdown
        sector_distress: Dict[str, Dict[str, int]] = {}
        for a in assessments:
            sector = a.sector or "Unknown"
            if sector not in sector_distress:
                sector_distress[sector] = {"total": 0, "distressed": 0, "critical": 0}
            sector_distress[sector]["total"] += 1
            if a.distress_level == DistressLevel.DISTRESSED:
                sector_distress[sector]["distressed"] += 1
            elif a.distress_level == DistressLevel.CRITICAL:
                sector_distress[sector]["critical"] += 1

        # Model agreement analysis
        model_agreement = []
        for a in assessments:
            z_distressed = a.z_score < 1.81
            o_prob = 1.0 / (1.0 + np.exp(-a.o_score))
            o_distressed = o_prob > 0.5
            f_distressed = a.f_score <= 3
            pd_distressed = a.probability_of_default > 0.1

            agree_count = sum([z_distressed, o_distressed, f_distressed, pd_distressed])
            if agree_count >= 3:
                model_agreement.append({
                    "symbol": a.symbol,
                    "models_agreeing": agree_count,
                    "z_score": a.z_score,
                    "o_prob": round(o_prob, 4),
                    "f_score": a.f_score,
                    "pd": a.probability_of_default,
                    "distress_level": a.distress_level.value,
                })

        # Aggregate statistics
        z_scores = [a.z_score for a in assessments]
        f_scores = [a.f_score for a in assessments]
        pds = [a.probability_of_default for a in assessments]

        report = {
            "date": datetime.now().isoformat(),
            "total_screened": len(assessments),
            "executive_summary": {
                "critical_count": len(critical),
                "distressed_count": len(distressed),
                "grey_zone_count": len(grey_zone),
                "safe_count": len(safe),
                "distress_rate": round(
                    (len(critical) + len(distressed)) / max(len(assessments), 1), 4
                ),
            },
            "statistics": {
                "z_score_mean": round(float(np.mean(z_scores)), 4),
                "z_score_median": round(float(np.median(z_scores)), 4),
                "z_score_std": round(float(np.std(z_scores)), 4),
                "f_score_mean": round(float(np.mean(f_scores)), 2),
                "f_score_median": float(np.median(f_scores)),
                "avg_pd": round(float(np.mean(pds)), 6),
                "max_pd": round(float(np.max(pds)), 6),
            },
            "critical_alerts": [
                {
                    "symbol": a.symbol,
                    "z_score": a.z_score,
                    "f_score": a.f_score,
                    "pd": a.probability_of_default,
                    "thesis": a.thesis,
                    "sector": a.sector,
                }
                for a in critical
            ],
            "recovery_candidates": [
                {
                    "symbol": a.symbol,
                    "f_score": a.f_score,
                    "recovery_prob": a.recovery_probability,
                    "z_score": a.z_score,
                    "thesis": a.thesis,
                    "recommendation": a.trade_recommendation,
                }
                for a in recovery_candidates[:10]
            ],
            "short_candidates": [
                {
                    "symbol": a.symbol,
                    "z_score": a.z_score,
                    "f_score": a.f_score,
                    "pd": a.probability_of_default,
                    "thesis": a.thesis,
                }
                for a in short_candidates[:10]
            ],
            "sector_analysis": sector_distress,
            "model_agreement_high_conviction": model_agreement,
            "watch_list": [
                {
                    "symbol": a.symbol,
                    "z_score": a.z_score,
                    "f_score": a.f_score,
                    "distress_level": a.distress_level.value,
                    "recovery_prob": a.recovery_probability,
                }
                for a in grey_zone[:20]
            ],
        }

        return report
