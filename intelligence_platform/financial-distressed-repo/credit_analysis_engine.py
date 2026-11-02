"""
Credit Analysis Engine for the Unified Investment Platform.

Deep credit analysis using OpenBB data for the entire bond/credit universe.
Implements duration, convexity, DV01, OAS, Z-score, Merton model,
F-score, credit curve analysis, relative value, and transition probabilities.

Usage:
    from credit_analysis_engine import CreditAnalysisEngine
    engine = CreditAnalysisEngine()
    z = engine.calculate_z_score(financials)
    dd = engine.calculate_distance_to_default(equity_data, debt_data)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BondMetrics:
    """Complete bond analytics output."""

    ticker: str
    price: float
    yield_to_maturity: float
    coupon: float
    maturity_years: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    spread_bps: float


@dataclass
class CreditCurve:
    """Credit curve for an issuer across maturities."""

    issuer: str
    maturities: list[float]
    spreads_bps: list[float]
    durations: list[float]
    fitted_slope: float
    fitted_intercept: float
    curve_steepness: float
    is_inverted: bool


@dataclass
class RelativeValuePair:
    """Relative value trade between two bonds."""

    long_ticker: str
    short_ticker: str
    spread_differential_bps: float
    duration_neutral_ratio: float
    expected_convergence_bps: float
    historical_mean_diff_bps: float
    z_score: float
    carry_bps: float
    sector: str


@dataclass
class OASResult:
    """Option-Adjusted Spread calculation result."""

    oas_bps: float
    z_spread_bps: float
    model_price: float
    market_price: float
    option_cost_bps: float


@dataclass
class TransitionMatrix:
    """Rating transition probability matrix."""

    current_rating: str
    probabilities: dict[str, float]  # rating -> probability
    expected_loss: float
    migration_risk: float  # probability of downgrade


# ---------------------------------------------------------------------------
# Bond math primitives
# ---------------------------------------------------------------------------


def bond_price_from_yield(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Calculate bond price from yield to maturity.

    P = sum(C / (1 + y/n)^t for t in 1..n*T) + F / (1 + y/n)^(n*T)

    Parameters
    ----------
    face_value : float
        Par value (typically 100 or 1000).
    coupon_rate : float
        Annual coupon rate (e.g. 0.05 for 5%).
    ytm : float
        Yield to maturity (annualised).
    maturity_years : float
        Years to maturity.
    frequency : int
        Coupon payments per year (default 2 = semi-annual).

    Returns
    -------
    float
        Clean bond price.
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")

    n_periods = int(maturity_years * frequency)
    coupon = face_value * coupon_rate / frequency
    discount = ytm / frequency

    if abs(discount) < 1e-12:
        return coupon * n_periods + face_value

    pv_coupons = 0.0
    for t in range(1, n_periods + 1):
        pv_coupons += coupon / (1 + discount) ** t

    pv_face = face_value / (1 + discount) ** n_periods
    return pv_coupons + pv_face


def yield_from_price(
    face_value: float,
    coupon_rate: float,
    price: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Solve for yield to maturity given a market price (Newton's method via scipy).
    """
    def objective(ytm):
        return bond_price_from_yield(face_value, coupon_rate, ytm, maturity_years, frequency) - price

    try:
        result = sp_optimize.brentq(objective, -0.05, 1.0, xtol=1e-8)
        return float(result)
    except ValueError:
        # Fallback: approximate
        coupon_income = face_value * coupon_rate
        approx = (coupon_income + (face_value - price) / maturity_years) / ((face_value + price) / 2)
        return approx


def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Macaulay Duration.

    D = (1/P) * sum(t * CF_t / (1 + y/n)^t)

    where t is measured in years.
    """
    n_periods = int(maturity_years * frequency)
    coupon = face_value * coupon_rate / frequency
    discount = ytm / frequency
    price = bond_price_from_yield(face_value, coupon_rate, ytm, maturity_years, frequency)

    if price == 0:
        return 0.0

    weighted_sum = 0.0
    for t in range(1, n_periods + 1):
        time_years = t / frequency
        if t < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value
        pv_cf = cf / (1 + discount) ** t
        weighted_sum += time_years * pv_cf

    return weighted_sum / price


def modified_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Modified Duration: D_mod = D_mac / (1 + y/n)

    Measures price sensitivity to yield changes.
    """
    d_mac = macaulay_duration(face_value, coupon_rate, ytm, maturity_years, frequency)
    return d_mac / (1 + ytm / frequency)


def convexity(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Bond Convexity.

    C = (1/P) * sum(t*(t+1) * CF_t / (1 + y/n)^(t+2)) / n^2

    Second-order price sensitivity to yield changes.
    """
    n_periods = int(maturity_years * frequency)
    coupon = face_value * coupon_rate / frequency
    discount = ytm / frequency
    price = bond_price_from_yield(face_value, coupon_rate, ytm, maturity_years, frequency)

    if price == 0:
        return 0.0

    conv_sum = 0.0
    for t in range(1, n_periods + 1):
        if t < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value
        conv_sum += t * (t + 1) * cf / (1 + discount) ** (t + 2)

    return conv_sum / (price * frequency ** 2)


def dv01(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    DV01: Dollar value of a 1 basis point move in yield.

    DV01 = -D_mod * P * 0.0001
    """
    d_mod = modified_duration(face_value, coupon_rate, ytm, maturity_years, frequency)
    price = bond_price_from_yield(face_value, coupon_rate, ytm, maturity_years, frequency)
    return abs(d_mod * price * 0.0001)


def z_spread_from_price(
    market_price: float,
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    spot_rates: list[float],
    frequency: int = 2,
) -> float:
    """
    Calculate Z-spread: constant spread over the spot curve that prices the bond.

    P_market = sum(CF_t / (1 + r_t + z)^t)
    """
    n_periods = int(maturity_years * frequency)
    coupon = face_value * coupon_rate / frequency

    # Extend spot rates if needed
    spots = list(spot_rates)
    while len(spots) < n_periods:
        spots.append(spots[-1] if spots else 0.04)

    def objective(z):
        pv = 0.0
        for t in range(1, n_periods + 1):
            spot = spots[t - 1] / frequency
            cf = coupon if t < n_periods else coupon + face_value
            pv += cf / (1 + spot + z / frequency) ** t
        return pv - market_price

    try:
        result = sp_optimize.brentq(objective, -0.05, 0.50, xtol=1e-8)
        return float(result)
    except ValueError:
        return 0.0


def calculate_oas(
    market_price: float,
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    spot_rates: list[float],
    volatility: float,
    is_callable: bool = False,
    call_price: float = 100.0,
    call_date_years: float = 0.0,
    frequency: int = 2,
    n_paths: int = 1000,
    seed: int = 42,
) -> OASResult:
    """
    Option-Adjusted Spread via Monte Carlo simulation.

    P_market = E[ sum(CF_t / (1 + r_t + OAS)^t) ] under risk-neutral paths.

    For callable bonds, the option cost is the difference between Z-spread and OAS:
        option_cost = z_spread - OAS

    Parameters
    ----------
    market_price : float
        Observed market price.
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    maturity_years : float
        Years to maturity.
    spot_rates : list[float]
        Spot rate curve (annualised).
    volatility : float
        Interest rate volatility for simulation.
    is_callable : bool
        Whether the bond is callable.
    call_price : float
        Call price if callable.
    call_date_years : float
        First call date in years.
    frequency : int
        Coupon frequency.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed.

    Returns
    -------
    OASResult
    """
    rng = np.random.RandomState(seed)
    n_periods = int(maturity_years * frequency)
    coupon = face_value * coupon_rate / frequency
    dt = 1.0 / frequency

    # Extend spots
    spots = list(spot_rates)
    while len(spots) < n_periods:
        spots.append(spots[-1] if spots else 0.04)

    # Z-spread for comparison
    z_spread = z_spread_from_price(
        market_price, face_value, coupon_rate, maturity_years, spot_rates, frequency,
    )

    def price_with_oas(oas: float) -> float:
        total_pv = 0.0
        for path in range(n_paths):
            path_pv = 0.0
            # Generate rate path with log-normal model
            rate_shocks = rng.normal(0, 1, n_periods)
            path_rates = np.zeros(n_periods)
            for t in range(n_periods):
                base_rate = spots[t] / frequency
                shock = volatility * math.sqrt(dt) * rate_shocks[t]
                if t == 0:
                    path_rates[t] = base_rate * math.exp(shock - 0.5 * volatility ** 2 * dt)
                else:
                    path_rates[t] = path_rates[t - 1] * math.exp(shock - 0.5 * volatility ** 2 * dt)

            discount_factor = 1.0
            called = False
            for t in range(1, n_periods + 1):
                r = path_rates[t - 1] + oas / frequency
                discount_factor /= (1 + r)

                if t < n_periods:
                    cf = coupon
                else:
                    cf = coupon + face_value

                # Check call provision
                if is_callable and t >= int(call_date_years * frequency) and not called:
                    # Simple call rule: call if PV of remaining > call price
                    remaining_pv = 0.0
                    remaining_df = 1.0
                    for s in range(t, n_periods):
                        remaining_df /= (1 + path_rates[s] + oas / frequency)
                        rcf = coupon if s < n_periods - 1 else coupon + face_value
                        remaining_pv += rcf * remaining_df
                    if remaining_pv > call_price:
                        path_pv += call_price * discount_factor
                        called = True
                        break

                path_pv += cf * discount_factor

            total_pv += path_pv

        return total_pv / n_paths

    # Solve for OAS
    def objective(oas):
        return price_with_oas(oas) - market_price

    try:
        oas = sp_optimize.brentq(objective, -0.05, 0.30, xtol=1e-6)
    except (ValueError, RuntimeError):
        oas = z_spread  # fallback to z-spread

    model_price = price_with_oas(oas)
    option_cost = z_spread - oas

    return OASResult(
        oas_bps=round(oas * 10_000, 2),
        z_spread_bps=round(z_spread * 10_000, 2),
        model_price=round(model_price, 4),
        market_price=market_price,
        option_cost_bps=round(option_cost * 10_000, 2),
    )


# ---------------------------------------------------------------------------
# Historical transition matrices (Moody's/S&P style)
# ---------------------------------------------------------------------------

# 1-year transition probabilities (approximate, based on historical data)
_TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    "AAA": {"AAA": 0.8793, "AA": 0.1065, "A": 0.0063, "BBB": 0.0006, "BB": 0.0008, "B": 0.0004, "CCC": 0.0000, "D": 0.0000, "NR": 0.0061},
    "AA": {"AAA": 0.0064, "AA": 0.8697, "A": 0.1064, "BBB": 0.0053, "BB": 0.0006, "B": 0.0010, "CCC": 0.0002, "D": 0.0001, "NR": 0.0103},
    "A": {"AAA": 0.0007, "AA": 0.0213, "A": 0.8682, "BBB": 0.0836, "BB": 0.0068, "B": 0.0024, "CCC": 0.0010, "D": 0.0004, "NR": 0.0156},
    "BBB": {"AAA": 0.0002, "AA": 0.0021, "A": 0.0392, "BBB": 0.8479, "BB": 0.0695, "B": 0.0162, "CCC": 0.0062, "D": 0.0020, "NR": 0.0167},
    "BB": {"AAA": 0.0001, "AA": 0.0005, "A": 0.0042, "BBB": 0.0556, "BB": 0.7937, "B": 0.0876, "CCC": 0.0252, "D": 0.0098, "NR": 0.0233},
    "B": {"AAA": 0.0000, "AA": 0.0004, "A": 0.0013, "BBB": 0.0046, "BB": 0.0525, "B": 0.7827, "CCC": 0.0698, "D": 0.0465, "NR": 0.0422},
    "CCC": {"AAA": 0.0000, "AA": 0.0000, "A": 0.0024, "BBB": 0.0058, "BB": 0.0131, "B": 0.1045, "CCC": 0.5548, "D": 0.2529, "NR": 0.0665},
}

# Loss given default by seniority (approximate)
_LGD_BY_SENIORITY: dict[str, float] = {
    "senior_secured_1st": 0.30,
    "senior_secured": 0.40,
    "senior_unsecured": 0.55,
    "subordinated": 0.70,
    "junior_subordinated": 0.85,
}


# ---------------------------------------------------------------------------
# CreditAnalysisEngine class
# ---------------------------------------------------------------------------


class CreditAnalysisEngine:
    """
    Deep credit analysis using OpenBB data for entire bond/credit universe.

    Duration: D = -1/P * dP/dy = sum(t * CF_t * e^(-y*t)) / P
    Modified Duration: D_mod = D / (1 + y/n)
    Convexity: C = (1/P) * d^2P/dy^2 = sum(t*(t+1) * CF_t / (1+y)^(t+2)) / P

    DV01: dollar value of 1bp = -D_mod * P * 0.0001

    OAS (Option-Adjusted Spread):
        P_market = sum E[CF_t / (1 + r_t + OAS)^t] under risk-neutral measure
    """

    def __init__(
        self,
        face_value: float = 100.0,
        frequency: int = 2,
        risk_free_rate: float = 0.045,
    ) -> None:
        self.face_value = face_value
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate

    # -------------------------------------------------------------------
    # Fundamental distress metrics
    # -------------------------------------------------------------------

    def calculate_z_score(self, financials: dict) -> float:
        """
        Calculate Altman Z-Score from financial data.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            X1 = Working Capital / Total Assets
            X2 = Retained Earnings / Total Assets
            X3 = EBIT / Total Assets
            X4 = Market Value of Equity / Total Liabilities
            X5 = Sales / Total Assets

        Parameters
        ----------
        financials : dict
            Must contain: working_capital, retained_earnings, ebit,
            market_cap, total_liabilities, total_assets, sales.

        Returns
        -------
        float
            Altman Z-Score.
        """
        ta = financials["total_assets"]
        if ta == 0:
            raise ValueError("Total assets cannot be zero")

        tl = financials.get("total_liabilities", 1e-6)
        if tl == 0:
            tl = 1e-6

        x1 = financials["working_capital"] / ta
        x2 = financials["retained_earnings"] / ta
        x3 = financials["ebit"] / ta
        x4 = financials["market_cap"] / tl
        x5 = financials["sales"] / ta

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return z

    def calculate_distance_to_default(
        self,
        equity_data: dict,
        debt_data: dict,
        time_horizon: float = 1.0,
    ) -> float:
        """
        Calculate Merton model Distance to Default.

        DD = (ln(V/D) + (mu - sigma^2/2)*T) / (sigma*sqrt(T))

        Uses iterative KMV approach to solve for asset value and volatility.

        Parameters
        ----------
        equity_data : dict
            Must contain: market_cap (float), equity_volatility (float).
        debt_data : dict
            Must contain: total_debt (float).
        time_horizon : float
            Years to default horizon (default 1).

        Returns
        -------
        float
            Distance to Default (number of standard deviations).
        """
        equity_value = equity_data["market_cap"]
        equity_vol = equity_data["equity_volatility"]
        debt = debt_data["total_debt"]

        if equity_value <= 0 or debt <= 0:
            raise ValueError("Equity and debt must be positive")

        sqrt_t = math.sqrt(time_horizon)
        r = self.risk_free_rate

        # Iterative KMV
        v_a = equity_value + debt
        sigma_a = equity_vol * equity_value / v_a

        for _ in range(200):
            d1 = (math.log(v_a / debt) + (r + 0.5 * sigma_a ** 2) * time_horizon) / (sigma_a * sqrt_t)
            d2 = d1 - sigma_a * sqrt_t

            n_d1 = sp_stats.norm.cdf(d1)
            n_d2 = sp_stats.norm.cdf(d2)

            v_a_new = equity_value + debt * math.exp(-r * time_horizon) * n_d2
            sigma_a_new = (
                equity_vol * equity_value / (v_a_new * n_d1)
                if v_a_new * n_d1 > 0
                else sigma_a
            )

            if abs(v_a_new - v_a) < 1e-6 and abs(sigma_a_new - sigma_a) < 1e-6:
                v_a = v_a_new
                sigma_a = sigma_a_new
                break

            v_a = v_a_new
            sigma_a = max(sigma_a_new, 1e-8)

        dd = (math.log(v_a / debt) + (r - 0.5 * sigma_a ** 2) * time_horizon) / (sigma_a * sqrt_t)
        return dd

    def calculate_probability_of_default(
        self,
        equity_data: dict,
        debt_data: dict,
        time_horizon: float = 1.0,
    ) -> float:
        """
        Calculate probability of default using the Merton model.

        PD = N(-DD) where N is the standard normal CDF.

        Returns
        -------
        float
            Probability of default in [0, 1].
        """
        dd = self.calculate_distance_to_default(equity_data, debt_data, time_horizon)
        return float(sp_stats.norm.cdf(-dd))

    def calculate_f_score(self, financials: dict) -> int:
        """
        Calculate Piotroski F-Score (0-9).

        Profitability (4):
            1. ROA > 0
            2. CFO > 0
            3. delta_ROA > 0
            4. CFO/TA > ROA (accruals quality)

        Leverage & Liquidity (3):
            5. delta_leverage < 0
            6. delta_current_ratio > 0
            7. No share dilution

        Efficiency (2):
            8. delta_gross_margin > 0
            9. delta_asset_turnover > 0

        Parameters
        ----------
        financials : dict
            Must contain current and prior period data:
            roa, cfo, total_assets, long_term_debt, current_ratio,
            shares_outstanding, gross_margin, asset_turnover,
            and prior_ prefixed equivalents.

        Returns
        -------
        int
            F-Score in [0, 9].
        """
        score = 0

        # Profitability
        if financials["roa"] > 0:
            score += 1
        if financials["cfo"] > 0:
            score += 1
        if financials["roa"] > financials["prior_roa"]:
            score += 1
        ta = financials["total_assets"]
        if ta > 0 and (financials["cfo"] / ta) > financials["roa"]:
            score += 1

        # Leverage & Liquidity
        if financials["long_term_debt"] < financials["prior_long_term_debt"]:
            score += 1
        if financials["current_ratio"] > financials["prior_current_ratio"]:
            score += 1
        if financials["shares_outstanding"] <= financials["prior_shares_outstanding"]:
            score += 1

        # Efficiency
        if financials["gross_margin"] > financials["prior_gross_margin"]:
            score += 1
        if financials["asset_turnover"] > financials["prior_asset_turnover"]:
            score += 1

        return score

    # -------------------------------------------------------------------
    # Bond analytics
    # -------------------------------------------------------------------

    def full_bond_analytics(
        self,
        ticker: str,
        coupon_rate: float,
        ytm: float,
        maturity_years: float,
        risk_free_yield: Optional[float] = None,
    ) -> BondMetrics:
        """
        Calculate complete bond analytics: price, duration, convexity, DV01, spread.

        Parameters
        ----------
        ticker : str
            Bond identifier.
        coupon_rate : float
            Annual coupon rate.
        ytm : float
            Yield to maturity.
        maturity_years : float
            Years to maturity.
        risk_free_yield : float, optional
            Risk-free yield for spread calculation.

        Returns
        -------
        BondMetrics
        """
        rf = risk_free_yield if risk_free_yield is not None else self.risk_free_rate

        price = bond_price_from_yield(
            self.face_value, coupon_rate, ytm, maturity_years, self.frequency,
        )
        d_mac = macaulay_duration(
            self.face_value, coupon_rate, ytm, maturity_years, self.frequency,
        )
        d_mod = d_mac / (1 + ytm / self.frequency)
        conv = convexity(
            self.face_value, coupon_rate, ytm, maturity_years, self.frequency,
        )
        dv = dv01(
            self.face_value, coupon_rate, ytm, maturity_years, self.frequency,
        )
        spread_bps = (ytm - rf) * 10_000

        return BondMetrics(
            ticker=ticker,
            price=round(price, 4),
            yield_to_maturity=ytm,
            coupon=coupon_rate,
            maturity_years=maturity_years,
            macaulay_duration=round(d_mac, 4),
            modified_duration=round(d_mod, 4),
            convexity=round(conv, 4),
            dv01=round(dv, 6),
            spread_bps=round(spread_bps, 2),
        )

    def price_change_estimate(
        self,
        bond: BondMetrics,
        yield_change_bps: float,
    ) -> float:
        """
        Estimate bond price change for a given yield shift using duration + convexity.

        dP/P approx -D_mod * dy + 0.5 * C * dy^2
        """
        dy = yield_change_bps / 10_000
        pct_change = -bond.modified_duration * dy + 0.5 * bond.convexity * dy ** 2
        return round(bond.price * pct_change, 4)

    # -------------------------------------------------------------------
    # Credit curve and relative value
    # -------------------------------------------------------------------

    def credit_curve_analysis(
        self,
        issuer_bonds: list[dict],
    ) -> CreditCurve:
        """
        Construct and analyze the credit curve for an issuer.

        Takes bonds from the same issuer at different maturities and
        fits a spread curve.

        Parameters
        ----------
        issuer_bonds : list[dict]
            Each dict has: ticker, coupon, ytm, maturity_years, issuer.

        Returns
        -------
        CreditCurve
        """
        if not issuer_bonds:
            raise ValueError("Need at least one bond")

        issuer = issuer_bonds[0].get("issuer", "Unknown")
        maturities: list[float] = []
        spreads: list[float] = []
        durations: list[float] = []

        for bond in sorted(issuer_bonds, key=lambda b: b["maturity_years"]):
            mat = bond["maturity_years"]
            ytm = bond["ytm"]
            cpn = bond["coupon"]

            spread = (ytm - self.risk_free_rate) * 10_000
            d = modified_duration(
                self.face_value, cpn, ytm, mat, self.frequency,
            )

            maturities.append(mat)
            spreads.append(spread)
            durations.append(d)

        # Fit linear regression to spread curve
        if len(maturities) >= 2:
            slope, intercept, _, _, _ = sp_stats.linregress(maturities, spreads)
        else:
            slope = 0.0
            intercept = spreads[0] if spreads else 0.0

        is_inverted = slope < -5.0  # inverted if slope < -5 bps/year
        steepness = (spreads[-1] - spreads[0]) if len(spreads) >= 2 else 0.0

        return CreditCurve(
            issuer=issuer,
            maturities=maturities,
            spreads_bps=spreads,
            durations=durations,
            fitted_slope=round(slope, 2),
            fitted_intercept=round(intercept, 2),
            curve_steepness=round(steepness, 2),
            is_inverted=is_inverted,
        )

    def relative_value_in_credit(
        self,
        sector_bonds: list[dict],
        historical_spreads: Optional[dict[str, np.ndarray]] = None,
    ) -> list[RelativeValuePair]:
        """
        Identify relative value pairs within a credit sector.

        Compares all pairs of bonds within the sector and identifies those with
        spread differentials that deviate significantly from historical norms.

        Parameters
        ----------
        sector_bonds : list[dict]
            Each dict: ticker, coupon, ytm, maturity_years, issuer, rating, sector.
        historical_spreads : dict, optional
            Ticker -> array of historical spreads in bps.

        Returns
        -------
        list[RelativeValuePair]
            Sorted by absolute z-score (most extreme first).
        """
        if historical_spreads is None:
            historical_spreads = {}

        # Calculate metrics for each bond
        bonds_with_metrics: list[dict] = []
        for b in sector_bonds:
            spread = (b["ytm"] - self.risk_free_rate) * 10_000
            d = modified_duration(
                self.face_value, b["coupon"], b["ytm"], b["maturity_years"], self.frequency,
            )
            bonds_with_metrics.append({**b, "spread_bps": spread, "mod_duration": d})

        pairs: list[RelativeValuePair] = []

        for i in range(len(bonds_with_metrics)):
            for j in range(i + 1, len(bonds_with_metrics)):
                b1 = bonds_with_metrics[i]
                b2 = bonds_with_metrics[j]

                # Skip if same issuer (intra-curve, not relative value)
                if b1.get("issuer") == b2.get("issuer"):
                    continue

                spread_diff = b1["spread_bps"] - b2["spread_bps"]

                # Compute historical mean differential if available
                h1 = historical_spreads.get(b1["ticker"])
                h2 = historical_spreads.get(b2["ticker"])

                if h1 is not None and h2 is not None and len(h1) == len(h2):
                    hist_diff = h1 - h2
                    hist_mean = float(np.mean(hist_diff))
                    hist_std = float(np.std(hist_diff, ddof=1))
                    z = (spread_diff - hist_mean) / hist_std if hist_std > 0 else 0.0
                else:
                    hist_mean = spread_diff
                    z = 0.0

                # Duration-neutral ratio
                if b2["mod_duration"] > 0:
                    dn_ratio = b1["mod_duration"] / b2["mod_duration"]
                else:
                    dn_ratio = 1.0

                expected_convergence = spread_diff - hist_mean

                # Carry: difference in running yield (coupon/price)
                p1 = bond_price_from_yield(
                    self.face_value, b1["coupon"], b1["ytm"], b1["maturity_years"], self.frequency,
                )
                p2 = bond_price_from_yield(
                    self.face_value, b2["coupon"], b2["ytm"], b2["maturity_years"], self.frequency,
                )
                carry = 0.0
                if p1 > 0 and p2 > 0:
                    carry = (b1["coupon"] * self.face_value / p1 - b2["coupon"] * self.face_value / p2) * 10_000

                # Determine long/short
                if spread_diff > 0:
                    long_t, short_t = b1["ticker"], b2["ticker"]
                else:
                    long_t, short_t = b2["ticker"], b1["ticker"]
                    spread_diff = -spread_diff
                    expected_convergence = -expected_convergence

                pairs.append(
                    RelativeValuePair(
                        long_ticker=long_t,
                        short_ticker=short_t,
                        spread_differential_bps=round(spread_diff, 2),
                        duration_neutral_ratio=round(dn_ratio, 4),
                        expected_convergence_bps=round(expected_convergence, 2),
                        historical_mean_diff_bps=round(hist_mean, 2),
                        z_score=round(z, 3),
                        carry_bps=round(carry, 2),
                        sector=b1.get("sector", "Unknown"),
                    )
                )

        pairs.sort(key=lambda x: abs(x.z_score), reverse=True)
        return pairs

    # -------------------------------------------------------------------
    # Rating transition
    # -------------------------------------------------------------------

    def transition_probability(self, current_rating: str) -> dict[str, float]:
        """
        Get 1-year rating transition probabilities.

        Based on historical Moody's/S&P average transition matrices.

        Parameters
        ----------
        current_rating : str
            Current rating (simplified: AAA, AA, A, BBB, BB, B, CCC).

        Returns
        -------
        dict[str, float]
            Mapping from target rating to probability.
        """
        # Simplify rating to base
        simplified = current_rating.upper().replace("+", "").replace("-", "")
        if simplified not in _TRANSITION_MATRIX:
            logger.warning("Rating %s not in transition matrix, using BBB", current_rating)
            simplified = "BBB"

        return dict(_TRANSITION_MATRIX[simplified])

    def expected_credit_loss(
        self,
        current_rating: str,
        exposure: float,
        seniority: str = "senior_unsecured",
        time_horizon: int = 1,
    ) -> float:
        """
        Calculate expected credit loss over a time horizon.

        ECL = PD * LGD * EAD

        For multi-year, uses cumulative default probability from the
        transition matrix.

        Parameters
        ----------
        current_rating : str
        exposure : float
            Exposure at default.
        seniority : str
        time_horizon : int
            Years.

        Returns
        -------
        float
            Expected credit loss in dollar terms.
        """
        lgd = _LGD_BY_SENIORITY.get(seniority, 0.55)

        # Cumulative PD via matrix exponentiation (simplified)
        trans = self.transition_probability(current_rating)
        annual_pd = trans.get("D", 0.0)

        # Multi-year cumulative PD (approximate)
        cumulative_pd = 1.0 - (1.0 - annual_pd) ** time_horizon

        ecl = cumulative_pd * lgd * exposure
        return round(ecl, 2)

    def full_credit_analysis(
        self,
        ticker: str,
        bond_info: dict,
        financials: Optional[dict] = None,
        equity_data: Optional[dict] = None,
        debt_data: Optional[dict] = None,
    ) -> dict:
        """
        Run complete credit analysis on a single issuer.

        Parameters
        ----------
        ticker : str
        bond_info : dict
            Keys: coupon, ytm, maturity_years, rating, seniority.
        financials : dict, optional
            For Z-score and F-score calculation.
        equity_data : dict, optional
            For Merton model.
        debt_data : dict, optional
            For Merton model.

        Returns
        -------
        dict with all credit metrics.
        """
        results: dict = {"ticker": ticker, "timestamp": datetime.utcnow().isoformat()}

        # Bond analytics
        metrics = self.full_bond_analytics(
            ticker=ticker,
            coupon_rate=bond_info["coupon"],
            ytm=bond_info["ytm"],
            maturity_years=bond_info["maturity_years"],
        )
        results["bond_metrics"] = metrics

        # Transition probabilities
        rating = bond_info.get("rating", "BBB")
        trans = self.transition_probability(rating)
        results["transition_probabilities"] = trans
        results["default_probability_1y"] = trans.get("D", 0.0)

        # Expected credit loss
        ecl = self.expected_credit_loss(
            rating,
            exposure=self.face_value,
            seniority=bond_info.get("seniority", "senior_unsecured"),
        )
        results["expected_credit_loss"] = ecl

        # Z-Score
        if financials:
            try:
                z = self.calculate_z_score(financials)
                results["z_score"] = z
            except Exception as exc:
                logger.warning("Z-score failed for %s: %s", ticker, exc)

        # Distance to Default
        if equity_data and debt_data:
            try:
                dd = self.calculate_distance_to_default(equity_data, debt_data)
                pd_val = float(sp_stats.norm.cdf(-dd))
                results["distance_to_default"] = dd
                results["merton_pd"] = pd_val
            except Exception as exc:
                logger.warning("Merton failed for %s: %s", ticker, exc)

        # F-Score
        if financials and "roa" in financials:
            try:
                f = self.calculate_f_score(financials)
                results["f_score"] = f
            except Exception as exc:
                logger.warning("F-score failed for %s: %s", ticker, exc)

        return results


# ---------------------------------------------------------------------------
# Convenience / standalone usage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = CreditAnalysisEngine()

    # Bond analytics demo
    metrics = engine.full_bond_analytics(
        ticker="AAPL-5Y",
        coupon_rate=0.045,
        ytm=0.055,
        maturity_years=5.0,
    )
    print(f"Bond: {metrics.ticker}")
    print(f"  Price: {metrics.price}")
    print(f"  Macaulay Duration: {metrics.macaulay_duration}")
    print(f"  Modified Duration: {metrics.modified_duration}")
    print(f"  Convexity: {metrics.convexity}")
    print(f"  DV01: {metrics.dv01}")
    print(f"  Spread: {metrics.spread_bps}bps")

    # Price change estimate
    dP = engine.price_change_estimate(metrics, yield_change_bps=50)
    print(f"  Estimated price change for +50bps: {dP}")

    # Z-Score demo
    z = engine.calculate_z_score({
        "working_capital": 500_000,
        "retained_earnings": 1_200_000,
        "ebit": 300_000,
        "market_cap": 5_000_000,
        "total_liabilities": 3_000_000,
        "total_assets": 8_000_000,
        "sales": 6_000_000,
    })
    print(f"\nZ-Score: {z:.3f}")

    # Distance to Default demo
    dd = engine.calculate_distance_to_default(
        equity_data={"market_cap": 5_000_000, "equity_volatility": 0.40},
        debt_data={"total_debt": 3_000_000},
    )
    print(f"Distance to Default: {dd:.3f}")
    print(f"Probability of Default: {sp_stats.norm.cdf(-dd):.4f}")

    # Transition probabilities
    trans = engine.transition_probability("BB")
    print(f"\nBB Transition Probabilities:")
    for rating, prob in sorted(trans.items(), key=lambda x: -x[1]):
        if prob > 0.001:
            print(f"  {rating}: {prob:.4f}")

    # Relative value demo
    sector_bonds = [
        {"ticker": "FORD-5Y", "coupon": 0.06, "ytm": 0.065, "maturity_years": 5.0,
         "issuer": "Ford", "rating": "BB+", "sector": "Autos"},
        {"ticker": "GM-5Y", "coupon": 0.055, "ytm": 0.058, "maturity_years": 5.0,
         "issuer": "GM", "rating": "BBB-", "sector": "Autos"},
        {"ticker": "FORD-10Y", "coupon": 0.065, "ytm": 0.072, "maturity_years": 10.0,
         "issuer": "Ford", "rating": "BB+", "sector": "Autos"},
    ]
    pairs = engine.relative_value_in_credit(sector_bonds)
    print(f"\nRelative Value Pairs ({len(pairs)}):")
    for p in pairs:
        print(f"  Long {p.long_ticker} / Short {p.short_ticker}: "
              f"diff={p.spread_differential_bps}bps, z={p.z_score}")
