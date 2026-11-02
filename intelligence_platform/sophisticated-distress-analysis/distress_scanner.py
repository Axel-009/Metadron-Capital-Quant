"""
Distress Scanner Module for the Unified Investment Platform.

Scans the entire universe for financially distressed securities that present
trading opportunities -- both long distressed debt and short equity plays.

Implements:
- Altman Z-Score (original manufacturing, Z'-Score for non-manufacturing)
- Merton Model (Distance to Default / Probability of Default)
- Credit Spread Analysis (Z-spread relative value)
- Piotroski F-Score (0-9 fundamental strength)
- Recovery / Short / Event-driven candidate filtering

Usage:
    from distress_scanner import DistressScanner
    scanner = DistressScanner()
    distressed = scanner.scan_distressed_equities(equity_universe)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for scanner outputs
# ---------------------------------------------------------------------------


class DistressZone(str, Enum):
    SAFE = "safe"
    GREY = "grey"
    DISTRESS = "distress"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class AltmanZResult:
    """Altman Z-Score decomposition."""

    z_score: float
    zone: DistressZone
    x1_working_capital_ta: float
    x2_retained_earnings_ta: float
    x3_ebit_ta: float
    x4_market_equity_tl: float
    x5_sales_ta: float


@dataclass
class MertonResult:
    """Merton model (distance to default) output."""

    distance_to_default: float
    probability_of_default: float
    asset_value: float
    asset_volatility: float
    debt_face_value: float
    time_horizon: float


@dataclass
class CreditSpreadResult:
    """Credit spread analysis output."""

    spread_bps: float
    z_spread: float
    historical_mean_bps: float
    historical_std_bps: float
    percentile: float


@dataclass
class PiotroskiResult:
    """Piotroski F-Score decomposition."""

    f_score: int
    # Profitability (4 signals)
    roa_positive: bool
    cfo_positive: bool
    delta_roa_positive: bool
    cfo_gt_roa: bool
    # Leverage / Liquidity (3 signals)
    delta_leverage_negative: bool
    delta_current_positive: bool
    no_dilution: bool
    # Efficiency (2 signals)
    delta_gross_margin_positive: bool
    delta_asset_turnover_positive: bool


@dataclass
class DistressedSecurity:
    """A security flagged as financially distressed."""

    ticker: str
    name: str
    sector: str
    z_score: Optional[AltmanZResult] = None
    merton: Optional[MertonResult] = None
    credit_spread: Optional[CreditSpreadResult] = None
    f_score: Optional[PiotroskiResult] = None
    distress_composite: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DistressedBond:
    """A bond flagged as distressed."""

    ticker: str
    issuer: str
    coupon: float
    maturity: str
    rating: str
    spread_bps: float
    z_spread: float
    yield_to_worst: float
    recovery_estimate: float
    distress_composite: float = 0.0


@dataclass
class RecoveryCandidate:
    """Distressed security showing signs of recovery."""

    security: DistressedSecurity
    f_score_trend: list[int]
    z_score_trend: list[float]
    spread_trend_bps: list[float]
    catalyst: str
    confidence: float


@dataclass
class ShortCandidate:
    """Security with accelerating deterioration."""

    security: DistressedSecurity
    z_score_velocity: float
    spread_widening_bps: float
    days_cash_remaining: Optional[float]
    downgrade_probability: float
    catalyst: str
    confidence: float


@dataclass
class EventTrade:
    """Event-driven trading opportunity."""

    security: DistressedSecurity
    event_type: str  # bankruptcy_emergence, restructuring, activist, spinoff
    event_date: Optional[str]
    expected_catalyst: str
    direction: SignalDirection
    upside_pct: float
    downside_pct: float
    probability: float
    risk_reward: float


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------


def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    total_assets: float,
    sales: float,
) -> AltmanZResult:
    """
    Calculate Altman Z-Score (original 1968 manufacturing formula).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets

    Interpretation:
        Z > 2.99  -> Safe zone
        1.81 < Z <= 2.99 -> Grey zone
        Z <= 1.81 -> Distress zone
    """
    if total_assets == 0:
        raise ValueError("Total assets cannot be zero")
    if total_liabilities == 0:
        total_liabilities = 1e-6  # avoid division by zero

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = DistressZone.SAFE
    elif z > 1.81:
        zone = DistressZone.GREY
    else:
        zone = DistressZone.DISTRESS

    return AltmanZResult(
        z_score=z,
        zone=zone,
        x1_working_capital_ta=x1,
        x2_retained_earnings_ta=x2,
        x3_ebit_ta=x3,
        x4_market_equity_tl=x4,
        x5_sales_ta=x5,
    )


def altman_z_prime_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    book_equity: float,
    total_liabilities: float,
    total_assets: float,
) -> AltmanZResult:
    """
    Altman Z'-Score for non-manufacturing / private firms.

    Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
        X4 uses book value of equity instead of market cap
        X5 is omitted (set to 0) for service firms
    """
    if total_assets == 0:
        raise ValueError("Total assets cannot be zero")
    if total_liabilities == 0:
        total_liabilities = 1e-6

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities
    x5 = 0.0  # excluded for non-manufacturing

    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4

    if z > 2.99:
        zone = DistressZone.SAFE
    elif z > 1.23:
        zone = DistressZone.GREY
    else:
        zone = DistressZone.DISTRESS

    return AltmanZResult(
        z_score=z,
        zone=zone,
        x1_working_capital_ta=x1,
        x2_retained_earnings_ta=x2,
        x3_ebit_ta=x3,
        x4_market_equity_tl=x4,
        x5_sales_ta=x5,
    )


def merton_distance_to_default(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> MertonResult:
    """
    Merton structural model: iteratively solve for asset value and volatility.

    Distance to Default:
        DD = (ln(V/D) + (mu - sigma_A^2/2) * T) / (sigma_A * sqrt(T))

    Probability of Default:
        PD = N(-DD) where N is the standard normal CDF.

    V = asset value, D = debt face value, mu = drift (approx risk-free),
    sigma_A = asset volatility, T = time horizon in years.

    Uses the iterative KMV method:
        E = V * N(d1) - D * exp(-r*T) * N(d2)       (Black-Scholes for equity)
        sigma_E = (V / E) * N(d1) * sigma_A          (equity vol relationship)

    Parameters
    ----------
    equity_value : float
        Current market value of equity (market cap).
    equity_volatility : float
        Annualised equity volatility (e.g. 0.30 for 30%).
    debt_face_value : float
        Face value of total debt (default barrier).
    risk_free_rate : float
        Annualised risk-free rate.
    time_horizon : float
        Time horizon in years (default 1).

    Returns
    -------
    MertonResult
    """
    if equity_value <= 0 or debt_face_value <= 0:
        raise ValueError("Equity value and debt must be positive")

    sqrt_t = math.sqrt(time_horizon)

    # Initial guess: asset value = equity + debt, asset vol = equity vol * E/(E+D)
    v_a = equity_value + debt_face_value
    sigma_a = equity_volatility * equity_value / v_a

    for _ in range(max_iterations):
        d1 = (math.log(v_a / debt_face_value) +
              (risk_free_rate + 0.5 * sigma_a ** 2) * time_horizon) / (sigma_a * sqrt_t)
        d2 = d1 - sigma_a * sqrt_t

        n_d1 = sp_stats.norm.cdf(d1)
        n_d2 = sp_stats.norm.cdf(d2)

        # Black-Scholes equity equation
        e_model = v_a * n_d1 - debt_face_value * math.exp(-risk_free_rate * time_horizon) * n_d2

        # Equity volatility relationship
        if equity_value > 0 and n_d1 > 0:
            sigma_e_model = (v_a / equity_value) * n_d1 * sigma_a
        else:
            sigma_e_model = sigma_a

        # Update asset value using Newton-like step
        v_a_new = equity_value + debt_face_value * math.exp(-risk_free_rate * time_horizon) * n_d2
        sigma_a_new = equity_volatility * equity_value / (v_a_new * n_d1) if (v_a_new * n_d1) > 0 else sigma_a

        # Convergence check
        if abs(v_a_new - v_a) < tolerance and abs(sigma_a_new - sigma_a) < tolerance:
            v_a = v_a_new
            sigma_a = sigma_a_new
            break

        v_a = v_a_new
        sigma_a = max(sigma_a_new, 1e-8)

    # Final distance to default
    dd = (math.log(v_a / debt_face_value) +
          (risk_free_rate - 0.5 * sigma_a ** 2) * time_horizon) / (sigma_a * sqrt_t)
    pd = sp_stats.norm.cdf(-dd)

    return MertonResult(
        distance_to_default=dd,
        probability_of_default=pd,
        asset_value=v_a,
        asset_volatility=sigma_a,
        debt_face_value=debt_face_value,
        time_horizon=time_horizon,
    )


def credit_spread_analysis(
    corporate_yield: float,
    risk_free_yield: float,
    historical_spreads: Optional[np.ndarray] = None,
) -> CreditSpreadResult:
    """
    Credit spread analysis with z-score relative to historical distribution.

    spread = corporate_yield - risk_free_yield
    z_spread = (spread - historical_mean) / historical_std

    Parameters
    ----------
    corporate_yield : float
        Current yield on corporate bond (annualised, e.g. 0.065 for 6.5%).
    risk_free_yield : float
        Current risk-free yield (e.g. 0.04 for 4%).
    historical_spreads : np.ndarray, optional
        Array of historical spread values in bps for z-score calculation.

    Returns
    -------
    CreditSpreadResult
    """
    spread_bps = (corporate_yield - risk_free_yield) * 10_000

    if historical_spreads is not None and len(historical_spreads) > 1:
        hist_mean = float(np.mean(historical_spreads))
        hist_std = float(np.std(historical_spreads, ddof=1))
        if hist_std > 0:
            z_spread = (spread_bps - hist_mean) / hist_std
        else:
            z_spread = 0.0
        percentile = float(sp_stats.percentileofscore(historical_spreads, spread_bps))
    else:
        hist_mean = spread_bps
        hist_std = 0.0
        z_spread = 0.0
        percentile = 50.0

    return CreditSpreadResult(
        spread_bps=spread_bps,
        z_spread=z_spread,
        historical_mean_bps=hist_mean,
        historical_std_bps=hist_std,
        percentile=percentile,
    )


def piotroski_f_score(
    # Current period
    net_income: float,
    cash_from_operations: float,
    roa: float,
    total_assets: float,
    long_term_debt: float,
    current_ratio: float,
    shares_outstanding: float,
    gross_margin: float,
    asset_turnover: float,
    # Prior period
    prior_roa: float,
    prior_long_term_debt: float,
    prior_current_ratio: float,
    prior_shares_outstanding: float,
    prior_gross_margin: float,
    prior_asset_turnover: float,
) -> PiotroskiResult:
    """
    Piotroski F-Score (0-9): composite fundamental strength signal.

    Profitability (4 points):
        1. ROA > 0
        2. Cash flow from operations > 0
        3. Delta ROA > 0 (improving profitability)
        4. CFO > ROA (quality of earnings / accruals)

    Leverage, Liquidity & Source of Funds (3 points):
        5. Delta long-term leverage < 0 (deleveraging)
        6. Delta current ratio > 0 (improving liquidity)
        7. No new share issuance (no dilution)

    Operating Efficiency (2 points):
        8. Delta gross margin > 0
        9. Delta asset turnover > 0

    Parameters
    ----------
    All financial inputs for current and prior period.

    Returns
    -------
    PiotroskiResult with score and decomposition.
    """
    # Profitability
    roa_positive = roa > 0
    cfo_positive = cash_from_operations > 0
    delta_roa_positive = roa > prior_roa
    cfo_gt_roa = (cash_from_operations / total_assets) > roa if total_assets > 0 else False

    # Leverage / Liquidity
    delta_leverage_negative = long_term_debt < prior_long_term_debt
    delta_current_positive = current_ratio > prior_current_ratio
    no_dilution = shares_outstanding <= prior_shares_outstanding

    # Efficiency
    delta_gross_margin_positive = gross_margin > prior_gross_margin
    delta_asset_turnover_positive = asset_turnover > prior_asset_turnover

    score = sum([
        roa_positive,
        cfo_positive,
        delta_roa_positive,
        cfo_gt_roa,
        delta_leverage_negative,
        delta_current_positive,
        no_dilution,
        delta_gross_margin_positive,
        delta_asset_turnover_positive,
    ])

    return PiotroskiResult(
        f_score=score,
        roa_positive=roa_positive,
        cfo_positive=cfo_positive,
        delta_roa_positive=delta_roa_positive,
        cfo_gt_roa=cfo_gt_roa,
        delta_leverage_negative=delta_leverage_negative,
        delta_current_positive=delta_current_positive,
        no_dilution=no_dilution,
        delta_gross_margin_positive=delta_gross_margin_positive,
        delta_asset_turnover_positive=delta_asset_turnover_positive,
    )


# ---------------------------------------------------------------------------
# Composite distress score
# ---------------------------------------------------------------------------


def _compute_distress_composite(
    z_result: Optional[AltmanZResult],
    merton_result: Optional[MertonResult],
    spread_result: Optional[CreditSpreadResult],
    f_result: Optional[PiotroskiResult],
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute a composite distress score in [0, 100].

    Higher = more distressed. Combines Z-score, Merton PD,
    credit spread z-score, and inverse F-score.

    Default weights: z=0.30, merton=0.30, spread=0.20, f_score=0.20
    """
    if weights is None:
        weights = {"z": 0.30, "merton": 0.30, "spread": 0.20, "f_score": 0.20}

    score = 0.0
    total_weight = 0.0

    if z_result is not None:
        # Map z-score to [0, 100]: z=0 -> 100, z=3 -> 0
        z_component = max(0.0, min(100.0, (3.0 - z_result.z_score) / 3.0 * 100))
        score += weights["z"] * z_component
        total_weight += weights["z"]

    if merton_result is not None:
        # PD directly maps to distress: PD * 100
        pd_component = min(100.0, merton_result.probability_of_default * 100)
        score += weights["merton"] * pd_component
        total_weight += weights["merton"]

    if spread_result is not None:
        # z-spread > 2 is very distressed -> cap at 100
        spread_component = max(0.0, min(100.0, spread_result.z_spread * 25 + 50))
        score += weights["spread"] * spread_component
        total_weight += weights["spread"]

    if f_result is not None:
        # F-score 0 -> 100 distress, F-score 9 -> 0 distress
        f_component = (9 - f_result.f_score) / 9 * 100
        score += weights["f_score"] * f_component
        total_weight += weights["f_score"]

    if total_weight > 0:
        score /= total_weight

    return round(score, 2)


# ---------------------------------------------------------------------------
# DistressScanner class
# ---------------------------------------------------------------------------


class DistressScanner:
    """
    Scans entire universe for financially distressed securities that present
    trading opportunities (both long distressed debt and short equity).

    Altman Z-Score: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        Z > 2.99: Safe, 1.81 < Z < 2.99: Grey zone, Z < 1.81: Distress

    Merton Model (Distance to Default):
        DD = (ln(V/D) + (mu - sigma^2/2)*T) / (sigma*sqrt(T))
        PD = N(-DD) where N is cumulative normal
        V = asset value, D = debt, mu = drift, sigma = asset volatility

    Credit Spread Analysis:
        spread = corporate_yield - risk_free_yield
        z_spread = (spread - historical_mean) / historical_std

    Piotroski F-Score (0-9):
        Profitability: ROA>0(1), CFO>0(1), dROA>0(1), CFO>ROA(1)
        Leverage: dLeverage<0(1), dCurrent>0(1), no dilution(1)
        Efficiency: dGrossMargin>0(1), dAssetTurnover>0(1)
    """

    def __init__(
        self,
        z_score_threshold: float = 1.81,
        merton_pd_threshold: float = 0.10,
        spread_threshold_bps: float = 500.0,
        f_score_threshold: int = 3,
        min_rating_for_distress: str = "BB",
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.merton_pd_threshold = merton_pd_threshold
        self.spread_threshold_bps = spread_threshold_bps
        self.f_score_threshold = f_score_threshold
        self.min_rating_for_distress = min_rating_for_distress

        # Rating ordering for comparison
        self._rating_order = {
            "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
            "A+": 5, "A": 6, "A-": 7,
            "BBB+": 8, "BBB": 9, "BBB-": 10,
            "BB+": 11, "BB": 12, "BB-": 13,
            "B+": 14, "B": 15, "B-": 16,
            "CCC+": 17, "CCC": 18, "CCC-": 19,
            "CC": 20, "C": 21, "D": 22,
        }

    def _is_below_rating(self, rating: str, threshold: str) -> bool:
        """Check if rating is below (worse than) threshold."""
        r_ord = self._rating_order.get(rating.upper(), 99)
        t_ord = self._rating_order.get(threshold.upper(), 99)
        return r_ord > t_ord

    def _extract_z_score_inputs(self, fundamentals: dict) -> Optional[dict]:
        """Extract Z-score inputs from fundamental data dict."""
        try:
            bs = fundamentals.get("balance_sheet")
            inc = fundamentals.get("income_statement")
            if bs is None or inc is None or bs.empty or inc.empty:
                return None

            # Take the most recent period
            bs_row = bs.iloc[0] if not bs.empty else {}
            inc_row = inc.iloc[0] if not inc.empty else {}

            total_assets = float(bs_row.get("total_assets", bs_row.get("totalAssets", 0)) or 0)
            if total_assets == 0:
                return None

            current_assets = float(bs_row.get("total_current_assets", bs_row.get("totalCurrentAssets", 0)) or 0)
            current_liabilities = float(bs_row.get("total_current_liabilities", bs_row.get("totalCurrentLiabilities", 0)) or 0)
            total_liabilities = float(bs_row.get("total_liabilities", bs_row.get("totalLiabilities", 0)) or 0)
            retained_earnings = float(bs_row.get("retained_earnings", bs_row.get("retainedEarnings", 0)) or 0)

            ebit = float(inc_row.get("ebit", inc_row.get("operatingIncome", 0)) or 0)
            revenue = float(inc_row.get("revenue", inc_row.get("totalRevenue", 0)) or 0)

            return {
                "working_capital": current_assets - current_liabilities,
                "retained_earnings": retained_earnings,
                "ebit": ebit,
                "total_liabilities": total_liabilities,
                "total_assets": total_assets,
                "sales": revenue,
            }
        except Exception as exc:
            logger.warning("Failed to extract Z-score inputs: %s", exc)
            return None

    def _extract_merton_inputs(
        self, fundamentals: dict, price_data: pd.DataFrame, market_cap: float,
    ) -> Optional[dict]:
        """Extract Merton model inputs."""
        try:
            bs = fundamentals.get("balance_sheet")
            if bs is None or bs.empty:
                return None

            bs_row = bs.iloc[0]
            total_debt = float(
                bs_row.get("total_debt", bs_row.get("totalDebt",
                    bs_row.get("long_term_debt", bs_row.get("longTermDebt", 0)))) or 0
            )
            if total_debt == 0:
                return None

            # Calculate equity volatility from price data
            if price_data is not None and len(price_data) > 30:
                close_col = "Close" if "Close" in price_data.columns else "close"
                if close_col in price_data.columns:
                    returns = price_data[close_col].pct_change().dropna()
                    equity_vol = float(returns.std() * np.sqrt(252))
                else:
                    equity_vol = 0.30
            else:
                equity_vol = 0.30

            return {
                "equity_value": market_cap,
                "equity_volatility": equity_vol,
                "debt_face_value": total_debt,
                "risk_free_rate": 0.045,  # approximate
            }
        except Exception as exc:
            logger.warning("Failed to extract Merton inputs: %s", exc)
            return None

    def _extract_f_score_inputs(self, fundamentals: dict) -> Optional[dict]:
        """Extract Piotroski F-Score inputs from two periods of fundamentals."""
        try:
            bs = fundamentals.get("balance_sheet")
            inc = fundamentals.get("income_statement")
            cf = fundamentals.get("cash_flow")

            if bs is None or inc is None or bs.empty or inc.empty:
                return None
            if len(bs) < 2 or len(inc) < 2:
                return None

            # Current and prior periods
            bs_curr, bs_prev = bs.iloc[0], bs.iloc[1]
            inc_curr, inc_prev = inc.iloc[0], inc.iloc[1]

            total_assets_curr = float(bs_curr.get("total_assets", bs_curr.get("totalAssets", 0)) or 0)
            total_assets_prev = float(bs_prev.get("total_assets", bs_prev.get("totalAssets", 0)) or 0)

            if total_assets_curr == 0 or total_assets_prev == 0:
                return None

            net_income = float(inc_curr.get("net_income", inc_curr.get("netIncome", 0)) or 0)
            net_income_prev = float(inc_prev.get("net_income", inc_prev.get("netIncome", 0)) or 0)
            revenue_curr = float(inc_curr.get("revenue", inc_curr.get("totalRevenue", 0)) or 0)
            revenue_prev = float(inc_prev.get("revenue", inc_prev.get("totalRevenue", 0)) or 0)
            cogs_curr = float(inc_curr.get("cost_of_revenue", inc_curr.get("costOfRevenue", 0)) or 0)
            cogs_prev = float(inc_prev.get("cost_of_revenue", inc_prev.get("costOfRevenue", 0)) or 0)

            cfo = 0.0
            if cf is not None and not cf.empty:
                cf_curr = cf.iloc[0]
                cfo = float(cf_curr.get("operating_cash_flow", cf_curr.get("operatingCashFlow", 0)) or 0)

            roa_curr = net_income / total_assets_curr
            roa_prev = net_income_prev / total_assets_prev

            lt_debt_curr = float(bs_curr.get("long_term_debt", bs_curr.get("longTermDebt", 0)) or 0)
            lt_debt_prev = float(bs_prev.get("long_term_debt", bs_prev.get("longTermDebt", 0)) or 0)

            ca_curr = float(bs_curr.get("total_current_assets", bs_curr.get("totalCurrentAssets", 0)) or 0)
            cl_curr = float(bs_curr.get("total_current_liabilities", bs_curr.get("totalCurrentLiabilities", 0)) or 0)
            ca_prev = float(bs_prev.get("total_current_assets", bs_prev.get("totalCurrentAssets", 0)) or 0)
            cl_prev = float(bs_prev.get("total_current_liabilities", bs_prev.get("totalCurrentLiabilities", 0)) or 0)

            current_ratio_curr = ca_curr / cl_curr if cl_curr > 0 else 0
            current_ratio_prev = ca_prev / cl_prev if cl_prev > 0 else 0

            shares_curr = float(bs_curr.get("common_stock_shares_outstanding",
                                             bs_curr.get("commonStockSharesOutstanding", 0)) or 0)
            shares_prev = float(bs_prev.get("common_stock_shares_outstanding",
                                             bs_prev.get("commonStockSharesOutstanding", 0)) or 0)

            gm_curr = (revenue_curr - cogs_curr) / revenue_curr if revenue_curr > 0 else 0
            gm_prev = (revenue_prev - cogs_prev) / revenue_prev if revenue_prev > 0 else 0

            at_curr = revenue_curr / total_assets_curr
            at_prev = revenue_prev / total_assets_prev

            return {
                "net_income": net_income,
                "cash_from_operations": cfo,
                "roa": roa_curr,
                "total_assets": total_assets_curr,
                "long_term_debt": lt_debt_curr,
                "current_ratio": current_ratio_curr,
                "shares_outstanding": shares_curr,
                "gross_margin": gm_curr,
                "asset_turnover": at_curr,
                "prior_roa": roa_prev,
                "prior_long_term_debt": lt_debt_prev,
                "prior_current_ratio": current_ratio_prev,
                "prior_shares_outstanding": shares_prev,
                "prior_gross_margin": gm_prev,
                "prior_asset_turnover": at_prev,
            }
        except Exception as exc:
            logger.warning("Failed to extract F-score inputs: %s", exc)
            return None

    # -------------------------------------------------------------------
    # Public scanning methods
    # -------------------------------------------------------------------

    def scan_distressed_equities(
        self,
        equity_universe: list,
        fundamentals_cache: Optional[dict[str, dict]] = None,
        price_cache: Optional[dict[str, pd.DataFrame]] = None,
        market_caps: Optional[dict[str, float]] = None,
    ) -> list[DistressedSecurity]:
        """
        Find equities with Z-score < 1.81 or Merton PD > 10%.

        Parameters
        ----------
        equity_universe : list
            List of Security objects (from openbb_data).
        fundamentals_cache : dict, optional
            Pre-fetched fundamentals keyed by ticker.
        price_cache : dict, optional
            Pre-fetched price DataFrames keyed by ticker.
        market_caps : dict, optional
            Market cap by ticker.

        Returns
        -------
        list[DistressedSecurity]
            Securities flagged as distressed, sorted by composite score descending.
        """
        if fundamentals_cache is None:
            fundamentals_cache = {}
        if price_cache is None:
            price_cache = {}
        if market_caps is None:
            market_caps = {}

        distressed: list[DistressedSecurity] = []

        for security in equity_universe:
            ticker = security.ticker if hasattr(security, "ticker") else str(security)
            name = security.name if hasattr(security, "name") else ticker
            sector = (
                security.gics_sector.value
                if hasattr(security, "gics_sector") and security.gics_sector
                else "Unknown"
            )

            fundamentals = fundamentals_cache.get(ticker, {})
            prices = price_cache.get(ticker)
            mcap = market_caps.get(ticker, 0)

            z_result: Optional[AltmanZResult] = None
            merton_result: Optional[MertonResult] = None
            f_result: Optional[PiotroskiResult] = None

            # Z-Score
            z_inputs = self._extract_z_score_inputs(fundamentals)
            if z_inputs and mcap > 0:
                try:
                    z_result = altman_z_score(market_cap=mcap, **z_inputs)
                except Exception as exc:
                    logger.debug("Z-score failed for %s: %s", ticker, exc)

            # Merton
            merton_inputs = self._extract_merton_inputs(fundamentals, prices, mcap)
            if merton_inputs:
                try:
                    merton_result = merton_distance_to_default(**merton_inputs)
                except Exception as exc:
                    logger.debug("Merton failed for %s: %s", ticker, exc)

            # Piotroski F-Score
            f_inputs = self._extract_f_score_inputs(fundamentals)
            if f_inputs:
                try:
                    f_result = piotroski_f_score(**f_inputs)
                except Exception as exc:
                    logger.debug("F-score failed for %s: %s", ticker, exc)

            # Determine if distressed
            is_distressed = False
            if z_result and z_result.zone == DistressZone.DISTRESS:
                is_distressed = True
            if merton_result and merton_result.probability_of_default > self.merton_pd_threshold:
                is_distressed = True
            if f_result and f_result.f_score <= self.f_score_threshold:
                is_distressed = True

            if is_distressed:
                composite = _compute_distress_composite(z_result, merton_result, None, f_result)
                distressed.append(
                    DistressedSecurity(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        z_score=z_result,
                        merton=merton_result,
                        f_score=f_result,
                        distress_composite=composite,
                    )
                )

        # Sort by composite distress score (most distressed first)
        distressed.sort(key=lambda x: x.distress_composite, reverse=True)
        logger.info(
            "Distress scan complete: %d/%d equities flagged",
            len(distressed), len(equity_universe),
        )
        return distressed

    def scan_distressed_bonds(
        self,
        bond_universe: list,
        bond_data: Optional[dict[str, dict]] = None,
        risk_free_rate: float = 0.045,
    ) -> list[DistressedBond]:
        """
        Find bonds with spreads > 500bps or rating below BB.

        Parameters
        ----------
        bond_universe : list
            List of bond Security objects.
        bond_data : dict, optional
            Dict keyed by ticker with bond details (yield, rating, coupon, etc.).
        risk_free_rate : float
            Current risk-free rate for spread calculation.

        Returns
        -------
        list[DistressedBond]
        """
        if bond_data is None:
            bond_data = {}

        distressed: list[DistressedBond] = []

        for bond in bond_universe:
            ticker = bond.ticker if hasattr(bond, "ticker") else str(bond)
            data = bond_data.get(ticker, {})

            bond_yield = data.get("yield", 0.0)
            rating = data.get("rating", "NR")
            coupon = data.get("coupon", 0.0)
            maturity = data.get("maturity", "")
            issuer = data.get("issuer", ticker)

            spread_bps = (bond_yield - risk_free_rate) * 10_000
            hist_spreads = data.get("historical_spreads")
            spread_result = credit_spread_analysis(bond_yield, risk_free_rate, hist_spreads)

            is_distressed = False
            if spread_bps > self.spread_threshold_bps:
                is_distressed = True
            if rating != "NR" and self._is_below_rating(rating, self.min_rating_for_distress):
                is_distressed = True

            if is_distressed:
                # Estimate recovery rate based on seniority and rating
                recovery = self._estimate_recovery(rating, data.get("seniority", "senior_unsecured"))

                composite = min(100.0, spread_bps / 20.0)  # 2000bps = 100
                distressed.append(
                    DistressedBond(
                        ticker=ticker,
                        issuer=issuer,
                        coupon=coupon,
                        maturity=maturity,
                        rating=rating,
                        spread_bps=spread_bps,
                        z_spread=spread_result.z_spread,
                        yield_to_worst=bond_yield,
                        recovery_estimate=recovery,
                        distress_composite=composite,
                    )
                )

        distressed.sort(key=lambda x: x.distress_composite, reverse=True)
        logger.info(
            "Bond distress scan: %d/%d bonds flagged",
            len(distressed), len(bond_universe),
        )
        return distressed

    def _estimate_recovery(self, rating: str, seniority: str) -> float:
        """Estimate recovery rate based on historical averages by seniority."""
        recovery_map = {
            "senior_secured": 0.52,
            "senior_unsecured": 0.37,
            "senior_subordinated": 0.24,
            "subordinated": 0.17,
            "junior_subordinated": 0.11,
        }
        base = recovery_map.get(seniority, 0.30)

        # Adjust slightly by rating
        rating_adj = {
            "BB+": 0.05, "BB": 0.03, "BB-": 0.0,
            "B+": -0.02, "B": -0.05, "B-": -0.08,
            "CCC+": -0.10, "CCC": -0.12, "CCC-": -0.14,
            "CC": -0.16, "C": -0.18, "D": -0.20,
        }
        adj = rating_adj.get(rating, 0.0)
        return max(0.0, min(1.0, base + adj))

    def recovery_candidates(
        self,
        distressed_list: list[DistressedSecurity],
        historical_f_scores: Optional[dict[str, list[int]]] = None,
        historical_z_scores: Optional[dict[str, list[float]]] = None,
        historical_spreads: Optional[dict[str, list[float]]] = None,
    ) -> list[RecoveryCandidate]:
        """
        Filter distressed securities that show signs of improvement:
        - Rising F-score over recent quarters
        - Stabilizing or improving Z-score
        - Narrowing credit spreads / CDS

        Parameters
        ----------
        distressed_list : list[DistressedSecurity]
        historical_f_scores : dict, optional
            Ticker -> list of recent F-scores (oldest first).
        historical_z_scores : dict, optional
            Ticker -> list of recent Z-scores (oldest first).
        historical_spreads : dict, optional
            Ticker -> list of recent spreads in bps (oldest first).

        Returns
        -------
        list[RecoveryCandidate]
        """
        if historical_f_scores is None:
            historical_f_scores = {}
        if historical_z_scores is None:
            historical_z_scores = {}
        if historical_spreads is None:
            historical_spreads = {}

        candidates: list[RecoveryCandidate] = []

        for sec in distressed_list:
            f_trend = historical_f_scores.get(sec.ticker, [])
            z_trend = historical_z_scores.get(sec.ticker, [])
            s_trend = historical_spreads.get(sec.ticker, [])

            signals = 0
            catalysts: list[str] = []

            # F-score improving (last > first, or last 2 increasing)
            if len(f_trend) >= 2 and f_trend[-1] > f_trend[-2]:
                signals += 1
                catalysts.append(f"F-score rising: {f_trend[-2]} -> {f_trend[-1]}")

            # Z-score improving
            if len(z_trend) >= 2 and z_trend[-1] > z_trend[-2]:
                signals += 1
                catalysts.append(
                    f"Z-score improving: {z_trend[-2]:.2f} -> {z_trend[-1]:.2f}"
                )

            # Spread narrowing
            if len(s_trend) >= 2 and s_trend[-1] < s_trend[-2]:
                signals += 1
                catalysts.append(
                    f"Spread narrowing: {s_trend[-2]:.0f}bps -> {s_trend[-1]:.0f}bps"
                )

            if signals >= 2:
                confidence = min(1.0, signals / 3.0 * 0.85)
                candidates.append(
                    RecoveryCandidate(
                        security=sec,
                        f_score_trend=f_trend,
                        z_score_trend=z_trend,
                        spread_trend_bps=s_trend,
                        catalyst="; ".join(catalysts),
                        confidence=round(confidence, 3),
                    )
                )

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        logger.info("Recovery candidates: %d from %d distressed", len(candidates), len(distressed_list))
        return candidates

    def short_candidates(
        self,
        distressed_list: list[DistressedSecurity],
        historical_z_scores: Optional[dict[str, list[float]]] = None,
        historical_spreads: Optional[dict[str, list[float]]] = None,
        cash_burn_rates: Optional[dict[str, float]] = None,
        cash_balances: Optional[dict[str, float]] = None,
    ) -> list[ShortCandidate]:
        """
        Identify short candidates: accelerating deterioration.

        Signals:
        - Falling Z-score trajectory
        - Widening credit spreads
        - High cash burn rate relative to cash balance
        - Potential for downgrade

        Returns
        -------
        list[ShortCandidate]
        """
        if historical_z_scores is None:
            historical_z_scores = {}
        if historical_spreads is None:
            historical_spreads = {}
        if cash_burn_rates is None:
            cash_burn_rates = {}
        if cash_balances is None:
            cash_balances = {}

        candidates: list[ShortCandidate] = []

        for sec in distressed_list:
            z_trend = historical_z_scores.get(sec.ticker, [])
            s_trend = historical_spreads.get(sec.ticker, [])
            burn = cash_burn_rates.get(sec.ticker, 0)
            cash = cash_balances.get(sec.ticker, 0)

            signals = 0
            catalysts: list[str] = []

            # Z-score velocity (rate of decline)
            z_velocity = 0.0
            if len(z_trend) >= 3:
                # Linear regression slope
                x = np.arange(len(z_trend))
                slope, _, _, _, _ = sp_stats.linregress(x, z_trend)
                z_velocity = slope
                if slope < -0.1:
                    signals += 1
                    catalysts.append(f"Z-score declining at {slope:.3f}/quarter")

            # Spread widening
            spread_widening = 0.0
            if len(s_trend) >= 2:
                spread_widening = s_trend[-1] - s_trend[0]
                if spread_widening > 50:
                    signals += 1
                    catalysts.append(f"Spread widened {spread_widening:.0f}bps")

            # Days cash remaining
            days_cash = None
            if burn > 0 and cash > 0:
                days_cash = cash / burn
                if days_cash < 180:
                    signals += 1
                    catalysts.append(f"~{days_cash:.0f} days cash remaining")

            # Downgrade probability heuristic
            downgrade_prob = 0.0
            if sec.z_score and sec.z_score.z_score < 1.0:
                downgrade_prob += 0.3
            if z_velocity < -0.2:
                downgrade_prob += 0.3
            if spread_widening > 100:
                downgrade_prob += 0.2
            if days_cash and days_cash < 90:
                downgrade_prob += 0.2
            downgrade_prob = min(1.0, downgrade_prob)

            if signals >= 1:
                confidence = min(1.0, signals / 4.0 * 0.9 + downgrade_prob * 0.1)
                candidates.append(
                    ShortCandidate(
                        security=sec,
                        z_score_velocity=z_velocity,
                        spread_widening_bps=spread_widening,
                        days_cash_remaining=days_cash,
                        downgrade_probability=round(downgrade_prob, 3),
                        catalyst="; ".join(catalysts),
                        confidence=round(confidence, 3),
                    )
                )

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        logger.info("Short candidates: %d from %d distressed", len(candidates), len(distressed_list))
        return candidates

    def event_driven_opportunities(
        self,
        news: list[dict],
        distressed_list: list[DistressedSecurity],
    ) -> list[EventTrade]:
        """
        Identify event-driven trading opportunities from news catalysts
        combined with distressed securities list.

        Event types:
        - Bankruptcy emergence: company exiting Chapter 11
        - Restructuring: debt exchange, liability management
        - Activist involvement: 13D filings, board seats
        - Management change: new CEO/CFO in distressed company
        - Asset sale / spinoff: unlocking value from parts

        Parameters
        ----------
        news : list[dict]
            News items with keys: ticker, headline, body, date, source.
        distressed_list : list[DistressedSecurity]
            Currently distressed securities.

        Returns
        -------
        list[EventTrade]
        """
        distressed_tickers = {sec.ticker: sec for sec in distressed_list}

        # Keyword patterns for event detection
        event_patterns = {
            "bankruptcy_emergence": [
                "emerge from bankruptcy", "exit chapter 11", "plan of reorganization",
                "creditor agreement", "bankruptcy exit", "restructuring complete",
            ],
            "restructuring": [
                "debt exchange", "liability management", "distressed exchange",
                "debt restructuring", "covenant waiver", "amendment and extension",
                "recapitalization", "balance sheet restructuring",
            ],
            "activist": [
                "13d filing", "activist investor", "board seat", "proxy fight",
                "shareholder proposal", "strategic review", "activist stake",
            ],
            "management_change": [
                "new ceo", "new cfo", "management change", "ceo appointment",
                "turnaround specialist", "interim ceo",
            ],
            "asset_sale": [
                "asset sale", "divestiture", "spinoff", "spin-off",
                "strategic alternatives", "sale process", "non-core assets",
            ],
        }

        # Default risk/reward profiles by event type
        event_profiles = {
            "bankruptcy_emergence": {
                "direction": SignalDirection.LONG, "upside": 0.40, "downside": 0.15, "prob": 0.55,
            },
            "restructuring": {
                "direction": SignalDirection.LONG, "upside": 0.30, "downside": 0.20, "prob": 0.50,
            },
            "activist": {
                "direction": SignalDirection.LONG, "upside": 0.25, "downside": 0.10, "prob": 0.60,
            },
            "management_change": {
                "direction": SignalDirection.LONG, "upside": 0.20, "downside": 0.15, "prob": 0.45,
            },
            "asset_sale": {
                "direction": SignalDirection.LONG, "upside": 0.35, "downside": 0.10, "prob": 0.50,
            },
        }

        trades: list[EventTrade] = []

        for item in news:
            ticker = item.get("ticker", "")
            headline = item.get("headline", "").lower()
            body = item.get("body", "").lower()
            date = item.get("date", "")
            combined_text = headline + " " + body

            if ticker not in distressed_tickers:
                continue

            sec = distressed_tickers[ticker]

            for event_type, patterns in event_patterns.items():
                matched = any(pat in combined_text for pat in patterns)
                if not matched:
                    continue

                profile = event_profiles[event_type]

                # Adjust probability based on distress level
                adj_prob = profile["prob"]
                if sec.distress_composite > 80:
                    adj_prob *= 0.8  # very distressed -> lower recovery probability
                elif sec.distress_composite < 50:
                    adj_prob *= 1.1  # moderately distressed -> higher probability

                upside = profile["upside"]
                downside = profile["downside"]

                if downside > 0:
                    risk_reward = (upside * adj_prob) / (downside * (1 - adj_prob))
                else:
                    risk_reward = float("inf")

                trades.append(
                    EventTrade(
                        security=sec,
                        event_type=event_type,
                        event_date=date,
                        expected_catalyst=item.get("headline", ""),
                        direction=profile["direction"],
                        upside_pct=round(upside * 100, 1),
                        downside_pct=round(downside * 100, 1),
                        probability=round(min(1.0, adj_prob), 3),
                        risk_reward=round(risk_reward, 2),
                    )
                )

        # Sort by risk-reward ratio
        trades.sort(key=lambda x: x.risk_reward, reverse=True)
        logger.info("Event-driven opportunities: %d trades identified", len(trades))
        return trades

    def full_scan(
        self,
        equity_universe: list,
        bond_universe: list,
        fundamentals_cache: Optional[dict] = None,
        price_cache: Optional[dict] = None,
        market_caps: Optional[dict] = None,
        bond_data: Optional[dict] = None,
        news: Optional[list[dict]] = None,
        historical_f_scores: Optional[dict] = None,
        historical_z_scores: Optional[dict] = None,
        historical_spreads: Optional[dict] = None,
    ) -> dict:
        """
        Run the complete distress scanning pipeline.

        Returns
        -------
        dict with keys: distressed_equities, distressed_bonds,
            recovery_candidates, short_candidates, event_trades.
        """
        logger.info("Starting full distress scan...")

        # Equity scan
        distressed_eq = self.scan_distressed_equities(
            equity_universe, fundamentals_cache, price_cache, market_caps,
        )

        # Bond scan
        distressed_bonds = self.scan_distressed_bonds(bond_universe, bond_data)

        # Recovery candidates
        recovery = self.recovery_candidates(
            distressed_eq, historical_f_scores, historical_z_scores, historical_spreads,
        )

        # Short candidates
        shorts = self.short_candidates(distressed_eq, historical_z_scores, historical_spreads)

        # Event-driven
        events: list[EventTrade] = []
        if news:
            events = self.event_driven_opportunities(news, distressed_eq)

        results = {
            "distressed_equities": distressed_eq,
            "distressed_bonds": distressed_bonds,
            "recovery_candidates": recovery,
            "short_candidates": shorts,
            "event_trades": events,
            "scan_timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_equities_scanned": len(equity_universe),
                "distressed_equities": len(distressed_eq),
                "total_bonds_scanned": len(bond_universe),
                "distressed_bonds": len(distressed_bonds),
                "recovery_candidates": len(recovery),
                "short_candidates": len(shorts),
                "event_trades": len(events),
            },
        }

        logger.info("Full scan complete: %s", results["summary"])
        return results


# ---------------------------------------------------------------------------
# Convenience / standalone usage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: calculate Z-score for a hypothetical company
    z = altman_z_score(
        working_capital=500_000,
        retained_earnings=1_200_000,
        ebit=300_000,
        market_cap=5_000_000,
        total_liabilities=3_000_000,
        total_assets=8_000_000,
        sales=6_000_000,
    )
    print(f"Z-Score: {z.z_score:.3f} ({z.zone.value})")

    # Demo: Merton model
    m = merton_distance_to_default(
        equity_value=5_000_000,
        equity_volatility=0.40,
        debt_face_value=3_000_000,
        risk_free_rate=0.045,
    )
    print(f"Distance to Default: {m.distance_to_default:.3f}, PD: {m.probability_of_default:.4f}")

    # Demo: Piotroski F-Score
    f = piotroski_f_score(
        net_income=100_000,
        cash_from_operations=150_000,
        roa=0.05,
        total_assets=2_000_000,
        long_term_debt=500_000,
        current_ratio=1.5,
        shares_outstanding=1_000_000,
        gross_margin=0.35,
        asset_turnover=1.2,
        prior_roa=0.04,
        prior_long_term_debt=600_000,
        prior_current_ratio=1.3,
        prior_shares_outstanding=1_000_000,
        prior_gross_margin=0.33,
        prior_asset_turnover=1.1,
    )
    print(f"F-Score: {f.f_score}/9")

    # Demo: Credit spread
    cs = credit_spread_analysis(
        corporate_yield=0.085,
        risk_free_yield=0.045,
        historical_spreads=np.array([300, 320, 350, 380, 400, 350, 330]),
    )
    print(f"Spread: {cs.spread_bps:.0f}bps, Z-spread: {cs.z_spread:.2f}")
