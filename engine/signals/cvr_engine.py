"""CVREngine — Contingent Value Rights Analysis Engine.

Institutional-grade CVR, earn-out, and contingent payment valuation:
    1. Binary Option Model — risk-adjusted PV with deal-collapse hazard
    2. Digital Barrier Option — up-and-in barrier for price/revenue CVRs
    3. Multi-Stage Milestone Tree — Biomedtracker conditional probabilities
    4. Monte Carlo — 10,000 paths, antithetic variates, logit-normal diffusion
    5. Real Options — Black-Scholes adapted for earn-outs and restructuring kickers

Instrument types:
    - Pharma milestone CVRs (FDA approval stages)
    - M&A earn-out CVRs (revenue/EBITDA targets)
    - Restructuring kickers (emergence value triggers)
    - Regulatory approval contingent payments

Dynamic CVR discovery:
    - Scans SEC 8-K filings for merger/acquisition events
    - Scans company news for CVR-creating catalysts
    - Updates underlying prices from live data (IBKR/FMP)
    - Refreshes milestone probabilities from news sentiment

Adjustments:
    - Liquidity discount (illiquid OTC instruments)
    - Counterparty credit risk (acquirer default)
    - Time decay with deal-break hazard rate

Usage:
    from engine.signals.cvr_engine import CVREngine

    cvr = CVREngine()
    results = cvr.analyze()
    report  = cvr.format_cvr_report()
    signals = cvr.get_trading_signals()
"""

import logging
import re
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("metadron.signals.cvr")

# Data layer imports (guarded)
try:
    from ..data.openbb_data import get_company_news, get_company_filings, get_adj_close
except ImportError:
    def get_company_news(*a, **kw): return __import__("pandas").DataFrame()
    def get_company_filings(*a, **kw): return __import__("pandas").DataFrame()
    def get_adj_close(*a, **kw): return __import__("pandas").DataFrame()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CVRType(str, Enum):
    """Type of contingent value right."""
    PHARMA_MILESTONE = "PHARMA_MILESTONE"     # FDA approval milestone
    REVENUE_EARNOUT = "REVENUE_EARNOUT"       # Revenue target earn-out
    EBITDA_EARNOUT = "EBITDA_EARNOUT"         # EBITDA target earn-out
    RESTRUCTURING_KICKER = "RESTRUCTURING"    # Post-restructuring value kicker
    REGULATORY_APPROVAL = "REGULATORY"        # Regulatory milestone
    LITIGATION_CVR = "LITIGATION_CVR"         # Litigation outcome contingent


class CVRSignal(str, Enum):
    """Trading signal for CVR instruments."""
    STRONG_BUY = "STRONG_BUY"    # >30% mispricing (undervalued)
    BUY = "BUY"                  # 10-30% mispricing
    HOLD = "HOLD"                # Fair value ±10%
    SELL = "SELL"                # 10-30% overvalued
    AVOID = "AVOID"              # Illiquid or high counterparty risk


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MilestoneStage:
    """Single stage in a multi-milestone pipeline."""
    name: str                          # e.g. "Phase III", "NDA Filing"
    conditional_prob: float = 0.50     # P(pass | reached this stage)
    time_to_completion_months: int = 12
    cost_usd: float = 0.0
    completed: bool = False


@dataclass
class CVRInstrument:
    """A single CVR instrument to value."""
    ticker: str                       # Underlying or CVR ticker
    name: str
    cvr_type: CVRType
    payment_usd: float                # Per-CVR payment if triggered
    market_price: float               # Current market price per CVR
    trigger_price: float = 0.0        # Barrier price (for price-trigger CVRs)
    acquirer: str = ""                # Acquirer name (counterparty risk)
    acquirer_rating: str = "BBB"      # Credit rating of acquirer
    expiry_months: int = 36           # Time to expiry
    milestones: List[MilestoneStage] = field(default_factory=list)
    underlying_price: float = 0.0     # Current price of underlying asset
    underlying_vol: float = 0.30      # Annual volatility of underlying
    floor_pct: float = 0.0           # Floor payment as % of face


@dataclass
class CVRValuation:
    """Complete valuation of a CVR instrument."""
    ticker: str
    name: str
    cvr_type: CVRType

    # Model valuations
    binary_option_value: float = 0.0
    barrier_option_value: float = 0.0
    milestone_tree_value: float = 0.0
    monte_carlo_value: float = 0.0
    real_option_value: float = 0.0

    # Ensemble
    fair_value: float = 0.0
    market_price: float = 0.0
    mispricing_pct: float = 0.0

    # Adjustments
    liquidity_discount: float = 0.0
    credit_adjustment: float = 0.0
    hazard_rate: float = 0.0

    # Signal
    signal: CVRSignal = CVRSignal.HOLD
    expected_return: float = 0.0
    kelly_fraction: float = 0.0

    # Probability metrics
    trigger_probability: float = 0.0
    expected_payout: float = 0.0
    time_decay_per_month: float = 0.0


# ---------------------------------------------------------------------------
# CVR Catalog
# ---------------------------------------------------------------------------
CVR_CATALOG: List[CVRInstrument] = [
    # Pharma milestone CVR (Biogen Alzheimer's)
    CVRInstrument(
        ticker="BIIB_CVR",
        name="Biogen Alzheimer's Milestone CVR",
        cvr_type=CVRType.PHARMA_MILESTONE,
        payment_usd=2.00,
        market_price=0.85,
        acquirer="Biogen",
        acquirer_rating="A-",
        expiry_months=48,
        milestones=[
            MilestoneStage("Phase III Complete", 0.65, 6, completed=True),
            MilestoneStage("NDA Filing", 0.80, 6),
            MilestoneStage("FDA Review", 0.75, 12),
            MilestoneStage("FDA Approval", 0.70, 6),
        ],
        underlying_price=280.0,
        underlying_vol=0.35,
    ),
    # Merck oncology CVR
    CVRInstrument(
        ticker="MRK_CVR",
        name="Merck Oncology Pipeline CVR",
        cvr_type=CVRType.PHARMA_MILESTONE,
        payment_usd=3.00,
        market_price=1.40,
        acquirer="Merck",
        acquirer_rating="AA-",
        expiry_months=60,
        milestones=[
            MilestoneStage("Phase II Complete", 0.55, 12, completed=True),
            MilestoneStage("Phase III Enrollment", 0.70, 6),
            MilestoneStage("Phase III Results", 0.60, 18),
            MilestoneStage("NDA+Approval", 0.75, 12),
        ],
        underlying_price=120.0,
        underlying_vol=0.25,
    ),
    # PE earn-out CVR
    CVRInstrument(
        ticker="PE_EARNOUT",
        name="PE Portfolio Co Revenue Earn-Out",
        cvr_type=CVRType.REVENUE_EARNOUT,
        payment_usd=5.00,
        market_price=2.10,
        trigger_price=500e6,   # Revenue target $500M
        acquirer="KKR Portfolio",
        acquirer_rating="A",
        expiry_months=36,
        underlying_price=380e6,  # Current revenue run-rate
        underlying_vol=0.20,
    ),
    # Restructuring kicker
    CVRInstrument(
        ticker="RESTRUC_CVR",
        name="Post-Restructuring Value Kicker",
        cvr_type=CVRType.RESTRUCTURING_KICKER,
        payment_usd=1.50,
        market_price=0.45,
        trigger_price=15.0,    # Stock price trigger
        acquirer="Reorganized Entity",
        acquirer_rating="B+",
        expiry_months=24,
        underlying_price=9.50,
        underlying_vol=0.55,
        floor_pct=0.10,
    ),
]


# ---------------------------------------------------------------------------
# Model weights by CVR type
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_BY_TYPE = {
    CVRType.PHARMA_MILESTONE: {
        "binary": 0.10, "barrier": 0.05, "milestone": 0.50,
        "monte_carlo": 0.25, "real_option": 0.10,
    },
    CVRType.REVENUE_EARNOUT: {
        "binary": 0.15, "barrier": 0.30, "milestone": 0.10,
        "monte_carlo": 0.30, "real_option": 0.15,
    },
    CVRType.EBITDA_EARNOUT: {
        "binary": 0.15, "barrier": 0.30, "milestone": 0.10,
        "monte_carlo": 0.30, "real_option": 0.15,
    },
    CVRType.RESTRUCTURING_KICKER: {
        "binary": 0.20, "barrier": 0.25, "milestone": 0.05,
        "monte_carlo": 0.30, "real_option": 0.20,
    },
    CVRType.REGULATORY_APPROVAL: {
        "binary": 0.15, "barrier": 0.05, "milestone": 0.45,
        "monte_carlo": 0.20, "real_option": 0.15,
    },
    CVRType.LITIGATION_CVR: {
        "binary": 0.30, "barrier": 0.05, "milestone": 0.30,
        "monte_carlo": 0.25, "real_option": 0.10,
    },
}

# Credit spread by rating (bps)
CREDIT_SPREADS = {
    "AAA": 30, "AA+": 40, "AA": 50, "AA-": 60,
    "A+": 75, "A": 90, "A-": 110,
    "BBB+": 140, "BBB": 175, "BBB-": 220,
    "BB+": 300, "BB": 375, "BB-": 450,
    "B+": 550, "B": 700, "B-": 900,
    "CCC": 1200,
}


class CVREngine:
    """Institutional-grade Contingent Value Rights valuation engine.

    Provides 5 independent valuation models with type-dependent weighting,
    liquidity/credit adjustments, and Kelly-sized trading signals.
    """

    def __init__(self, catalog: Optional[List[CVRInstrument]] = None,
                 risk_free: float = 0.045, n_mc_paths: int = 10000):
        self.catalog = catalog or CVR_CATALOG
        self.risk_free = risk_free
        self.n_mc_paths = n_mc_paths
        self._results: Dict[str, CVRValuation] = {}
        self._analyzed = False
        # Shared NewsEngine singleton — avoids creating a fresh instance per
        # ticker during scan_for_cvr_events() (same efficiency problem as
        # EventDrivenEngine).  The cached engine shares the newsfilter.io
        # WebSocket connection and the 100-item feed deque.
        self._news_engine = None

    def _get_news_engine(self):
        """Return a cached NewsEngine instance (lazy-initialised once).

        Using a singleton means the newsfilter.io WebSocket connection and
        the 100-item feed deque are shared across all _scan_news() calls
        in a single scan_for_cvr_events() sweep.
        """
        if self._news_engine is None:
            from engine.signals.news_engine import NewsEngine
            self._news_engine = NewsEngine()
        return self._news_engine

    # -----------------------------------------------------------------------
    # Model 1: Binary Option (risk-adjusted PV)
    # -----------------------------------------------------------------------
    def _binary_option(self, inst: CVRInstrument, trigger_prob: float) -> float:
        """Risk-adjusted PV of binary payout.

        V = Payment × P(trigger) × exp(-r×T) × (1 - hazard_discount)

        hazard_discount captures deal-break risk over the CVR lifetime.
        """
        T = inst.expiry_months / 12.0
        discount = np.exp(-self.risk_free * T)

        # Hazard rate for deal collapse (higher for lower-rated acquirers)
        spread_bps = CREDIT_SPREADS.get(inst.acquirer_rating, 300)
        annual_hazard = spread_bps / 10000 * 2.0  # 2x credit spread → hazard
        survival_prob = np.exp(-annual_hazard * T)

        value = inst.payment_usd * trigger_prob * discount * survival_prob

        # Add floor value if applicable
        if inst.floor_pct > 0:
            floor_value = inst.payment_usd * inst.floor_pct * discount * survival_prob
            value = max(value, floor_value)

        return value

    # -----------------------------------------------------------------------
    # Model 2: Digital Barrier Option (up-and-in)
    # -----------------------------------------------------------------------
    def _barrier_option(self, inst: CVRInstrument) -> float:
        """Digital up-and-in barrier option via reflection principle.

        For price/revenue-trigger CVRs:
            V = Payment × [N(d2) + (S/H)^(2μ/σ²) × N(d2')]

        Where H = barrier, S = current, μ = r - σ²/2
        """
        if inst.trigger_price <= 0 or inst.underlying_price <= 0:
            return 0.0

        S = inst.underlying_price
        H = inst.trigger_price
        T = inst.expiry_months / 12.0
        sigma = inst.underlying_vol
        r = self.risk_free

        if T <= 0 or sigma <= 0:
            return 0.0

        # Already breached
        if S >= H:
            return inst.payment_usd * np.exp(-r * T)

        sqrt_T = np.sqrt(T)
        mu = r - 0.5 * sigma**2

        d2 = (np.log(S / H) + mu * T) / (sigma * sqrt_T)
        # Reflection: d2' using image principle
        lam = mu / (sigma**2) if sigma > 0 else 0
        d2_prime = (np.log(H / S) + mu * T) / (sigma * sqrt_T)

        N_d2 = self._norm_cdf(d2)
        ratio = (S / H) ** (2 * lam) if H > 0 else 0
        N_d2p = self._norm_cdf(d2_prime)

        barrier_prob = N_d2 + ratio * N_d2p
        barrier_prob = min(max(barrier_prob, 0.0), 1.0)

        value = inst.payment_usd * barrier_prob * np.exp(-r * T)

        # Floor
        if inst.floor_pct > 0:
            floor_value = inst.payment_usd * inst.floor_pct * np.exp(-r * T)
            value = max(value, floor_value)

        return value

    # -----------------------------------------------------------------------
    # Model 3: Multi-Stage Milestone Tree
    # -----------------------------------------------------------------------
    def _milestone_tree(self, inst: CVRInstrument) -> Tuple[float, float]:
        """Biomedtracker-style conditional probability tree.

        P(all milestones) = ∏ P(stage_i | reached_i)
        Value = Payment × P(all) × exp(-r × total_time)

        Returns (value, cumulative_probability).
        """
        if not inst.milestones:
            return 0.0, 0.0

        cumulative_prob = 1.0
        total_months = 0
        remaining_milestones = [m for m in inst.milestones if not m.completed]

        for milestone in remaining_milestones:
            cumulative_prob *= milestone.conditional_prob
            total_months += milestone.time_to_completion_months

        T = total_months / 12.0
        discount = np.exp(-self.risk_free * T)

        value = inst.payment_usd * cumulative_prob * discount

        # Floor
        if inst.floor_pct > 0:
            floor_value = inst.payment_usd * inst.floor_pct * discount
            value = max(value, floor_value)

        return value, cumulative_prob

    # -----------------------------------------------------------------------
    # Model 4: Monte Carlo (antithetic variates)
    # -----------------------------------------------------------------------
    def _monte_carlo(self, inst: CVRInstrument, trigger_prob: float) -> float:
        """Monte Carlo with logit-normal probability diffusion.

        For milestone CVRs: simulate probability evolution over time
        For barrier CVRs: simulate price paths and check barrier breach

        Uses antithetic variates for variance reduction.
        """
        T = inst.expiry_months / 12.0
        n_steps = max(inst.expiry_months, 12)
        dt = T / n_steps
        n_paths = self.n_mc_paths

        np.random.seed(hash(inst.ticker) % (2**31))

        if inst.trigger_price > 0 and inst.underlying_price > 0:
            # Barrier simulation: GBM price paths
            S = inst.underlying_price
            H = inst.trigger_price
            sigma = inst.underlying_vol
            r = self.risk_free

            z = np.random.randn(n_paths // 2, n_steps)
            z = np.vstack([z, -z])  # Antithetic

            log_S = np.log(S)
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)

            log_prices = np.zeros((n_paths, n_steps + 1))
            log_prices[:, 0] = log_S

            for t in range(n_steps):
                log_prices[:, t + 1] = log_prices[:, t] + drift + diffusion * z[:, t]

            # Check if barrier ever breached
            max_prices = np.exp(log_prices).max(axis=1)
            breach_count = (max_prices >= H).sum()
            breach_prob = breach_count / n_paths

            value = inst.payment_usd * breach_prob * np.exp(-r * T)
        else:
            # Probability diffusion for milestone CVRs
            # Logit-normal diffusion of trigger probability
            if trigger_prob <= 0 or trigger_prob >= 1:
                trigger_prob = max(0.01, min(0.99, trigger_prob))

            logit_p = np.log(trigger_prob / (1 - trigger_prob))
            prob_vol = 0.3  # Volatility of probability estimate

            z = np.random.randn(n_paths // 2)
            z = np.concatenate([z, -z])

            # Diffuse probability over time
            logit_final = logit_p + prob_vol * np.sqrt(T) * z - 0.5 * prob_vol**2 * T
            final_probs = 1.0 / (1.0 + np.exp(-logit_final))

            # Each path: payout if prob > 0.5 (decision threshold)
            payouts = np.where(final_probs > 0.5, inst.payment_usd, 0.0)
            payouts *= final_probs  # Weight by probability

            value = payouts.mean() * np.exp(-self.risk_free * T)

        # Floor
        if inst.floor_pct > 0:
            floor_value = inst.payment_usd * inst.floor_pct * np.exp(-self.risk_free * T)
            value = max(value, floor_value)

        return value

    # -----------------------------------------------------------------------
    # Model 5: Real Options (adapted Black-Scholes)
    # -----------------------------------------------------------------------
    def _real_option(self, inst: CVRInstrument) -> float:
        """Black-Scholes adapted for earn-outs and restructuring kickers.

        Treats CVR as a call option on the underlying value/revenue:
            V = S*N(d1) - K*exp(-rT)*N(d2)

        Scaled to payment amount relative to option moneyness.
        """
        if inst.underlying_price <= 0:
            return 0.0

        S = inst.underlying_price
        K = inst.trigger_price if inst.trigger_price > 0 else S * 1.2
        T = inst.expiry_months / 12.0
        sigma = inst.underlying_vol
        r = self.risk_free

        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        call_value = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)

        # Scale: option_value / (S - K boundary) → fraction of payment
        moneyness = call_value / max(S, 1e-8)
        value = inst.payment_usd * min(moneyness, 1.0)

        # Floor
        if inst.floor_pct > 0:
            floor_value = inst.payment_usd * inst.floor_pct * np.exp(-r * T)
            value = max(value, floor_value)

        return max(value, 0.0)

    # -----------------------------------------------------------------------
    # Adjustments
    # -----------------------------------------------------------------------
    def _liquidity_discount(self, inst: CVRInstrument) -> float:
        """Liquidity discount based on instrument type and market.

        OTC CVRs: 15-25% discount
        Listed CVRs: 5-10% discount
        Restructuring: 20-35% discount
        """
        base_discounts = {
            CVRType.PHARMA_MILESTONE: 0.10,
            CVRType.REVENUE_EARNOUT: 0.18,
            CVRType.EBITDA_EARNOUT: 0.18,
            CVRType.RESTRUCTURING_KICKER: 0.25,
            CVRType.REGULATORY_APPROVAL: 0.12,
            CVRType.LITIGATION_CVR: 0.22,
        }
        return base_discounts.get(inst.cvr_type, 0.15)

    def _credit_adjustment(self, inst: CVRInstrument) -> float:
        """CVA (Credit Value Adjustment) for counterparty risk.

        CVA ≈ spread × T × LGD_acquirer
        """
        spread_bps = CREDIT_SPREADS.get(inst.acquirer_rating, 300)
        T = inst.expiry_months / 12.0
        lgd = 0.40  # Standard LGD assumption

        cva = (spread_bps / 10000) * T * lgd
        return min(cva, 0.50)  # Cap at 50%

    # -----------------------------------------------------------------------
    # Main Analysis
    # -----------------------------------------------------------------------
    def analyze(self) -> Dict[str, CVRValuation]:
        """Run full 5-model ensemble on CVR catalog.

        Calls discover_and_refresh() first to update underlying prices and
        discover new CVR instruments from live news/filings before running
        the 5-model ensemble.  Without this the engine only ever values the
        hardcoded CVR_CATALOG with static prices.
        """
        # Refresh CVR candidates from live news/filings and update prices.
        # Wrapped in try/except so analysis always proceeds even when the
        # data feeds are unavailable (no FMP/IBKR key configured).
        try:
            self.discover_and_refresh()
        except Exception as e:
            logger.warning("CVR discovery failed (using static catalog): %s", e)

        self._results = {}

        for inst in self.catalog:
            val = CVRValuation(
                ticker=inst.ticker,
                name=inst.name,
                cvr_type=inst.cvr_type,
                market_price=inst.market_price,
            )

            # Compute trigger probability from milestones
            milestone_value, trigger_prob = self._milestone_tree(inst)
            if trigger_prob == 0 and inst.trigger_price > 0:
                # Use barrier probability as fallback
                trigger_prob = 0.5  # Prior

            val.trigger_probability = trigger_prob

            # Run all 5 models
            val.binary_option_value = self._binary_option(inst, trigger_prob)
            val.barrier_option_value = self._barrier_option(inst)
            val.milestone_tree_value = milestone_value
            val.monte_carlo_value = self._monte_carlo(inst, trigger_prob)
            val.real_option_value = self._real_option(inst)

            # Ensemble with type-dependent weights
            weights = MODEL_WEIGHTS_BY_TYPE.get(inst.cvr_type, {
                "binary": 0.20, "barrier": 0.20, "milestone": 0.20,
                "monte_carlo": 0.20, "real_option": 0.20,
            })
            model_values = {
                "binary": val.binary_option_value,
                "barrier": val.barrier_option_value,
                "milestone": val.milestone_tree_value,
                "monte_carlo": val.monte_carlo_value,
                "real_option": val.real_option_value,
            }
            raw_fair = sum(model_values[k] * weights[k] for k in weights)

            # Apply adjustments
            val.liquidity_discount = self._liquidity_discount(inst)
            val.credit_adjustment = self._credit_adjustment(inst)
            val.hazard_rate = CREDIT_SPREADS.get(inst.acquirer_rating, 300) / 10000 * 2.0

            val.fair_value = raw_fair * (1 - val.liquidity_discount) * (1 - val.credit_adjustment)
            val.fair_value = max(val.fair_value, 0.0)

            # Mispricing
            if inst.market_price > 0:
                val.mispricing_pct = (val.fair_value - inst.market_price) / inst.market_price
            else:
                val.mispricing_pct = 0.0

            # Expected payout
            val.expected_payout = inst.payment_usd * trigger_prob

            # Time decay
            T = inst.expiry_months / 12.0
            if T > 0:
                val.time_decay_per_month = val.fair_value * self.risk_free / 12.0

            # Signal
            if val.mispricing_pct > 0.30:
                val.signal = CVRSignal.STRONG_BUY
            elif val.mispricing_pct > 0.10:
                val.signal = CVRSignal.BUY
            elif val.mispricing_pct < -0.30:
                val.signal = CVRSignal.AVOID
            elif val.mispricing_pct < -0.10:
                val.signal = CVRSignal.SELL
            else:
                val.signal = CVRSignal.HOLD

            # Expected return & Kelly
            if inst.market_price > 0:
                val.expected_return = (val.expected_payout - inst.market_price) / inst.market_price
                # Kelly: f* = (p*b - q) / b
                p = trigger_prob
                if p > 0 and p < 1:
                    b = (inst.payment_usd - inst.market_price) / max(inst.market_price, 0.01)
                    q = 1 - p
                    if b > 0:
                        val.kelly_fraction = max(0, min((p * b - q) / b, 0.20))

            self._results[inst.ticker] = val

        self._analyzed = True
        return self._results

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------
    def get_trading_signals(self) -> Dict[str, dict]:
        """Return CVR trading signals for pipeline integration."""
        if not self._analyzed:
            self.analyze()
        signals = {}
        for ticker, val in self._results.items():
            signals[ticker] = {
                "signal": val.signal.value,
                "fair_value": val.fair_value,
                "market_price": val.market_price,
                "mispricing_pct": val.mispricing_pct,
                "trigger_prob": val.trigger_probability,
                "expected_return": val.expected_return,
                "kelly_fraction": val.kelly_fraction,
                "cvr_type": val.cvr_type.value,
            }
        return signals

    def get_buy_signals(self) -> List[CVRValuation]:
        """Return CVRs with buy/strong_buy signals."""
        if not self._analyzed:
            self.analyze()
        return [v for v in self._results.values()
                if v.signal in (CVRSignal.STRONG_BUY, CVRSignal.BUY)]

    # -----------------------------------------------------------------------
    # Dynamic CVR Discovery — SEC filings + news feeds
    # -----------------------------------------------------------------------

    # Keywords that indicate CVR-creating events in SEC filings and news
    _MERGER_KEYWORDS = [
        "contingent value right", "cvr", "earn-out", "earnout", "milestone payment",
        "contingent consideration", "contingent payment", "merger agreement",
        "definitive agreement", "acquisition agreement",
    ]
    _PHARMA_KEYWORDS = [
        "fda approval", "nda filing", "phase iii", "phase 3", "pdufa",
        "breakthrough therapy", "accelerated approval", "priority review",
    ]
    _RESTRUCTURING_KEYWORDS = [
        "restructuring", "chapter 11", "emergence", "plan of reorganization",
        "debtor in possession", "creditor recovery",
    ]

    def scan_for_cvr_events(self, tickers: Optional[List[str]] = None) -> List[dict]:
        """Scan SEC 8-K filings and news for CVR-creating events.

        Checks recent 8-K filings and company news for merger/acquisition
        language that indicates new CVR instruments.

        Returns list of discovered event dicts with ticker, event_type,
        headline, date, and confidence score.
        """
        if tickers is None:
            tickers = [inst.ticker.split("_")[0] for inst in self.catalog]
            # Also scan top M&A targets from universe
            try:
                from ..data.universe_engine import get_engine
                ue = get_engine()
                # Pharma, financials, and distressed sectors are CVR-heavy
                for sec in ue.get_all()[:200]:
                    if sec.ticker not in tickers:
                        tickers.append(sec.ticker)
            except Exception:
                pass

        events = []
        for ticker in tickers[:100]:  # Cap at 100 to manage API calls
            try:
                events.extend(self._scan_filings(ticker))
            except Exception as e:
                logger.debug("Filing scan failed for %s: %s", ticker, e)
            try:
                events.extend(self._scan_news(ticker))
            except Exception as e:
                logger.debug("News scan failed for %s: %s", ticker, e)

        # Deduplicate by ticker + event_type
        seen = set()
        unique = []
        for ev in events:
            key = (ev["ticker"], ev["event_type"])
            if key not in seen:
                seen.add(key)
                unique.append(ev)

        logger.info("CVR scan: %d events found across %d tickers", len(unique), len(tickers))
        return unique

    def _scan_filings(self, ticker: str) -> List[dict]:
        """Scan SEC 8-K filings for CVR-related language."""
        events = []
        try:
            filings = get_company_filings(ticker, filing_type="8-K", limit=10)
            if filings.empty:
                return events

            for _, row in filings.iterrows():
                title = str(row.get("title", "") or row.get("description", "") or "").lower()
                date = str(row.get("date", "") or row.get("filing_date", "") or "")

                for kw in self._MERGER_KEYWORDS:
                    if kw in title:
                        events.append({
                            "ticker": ticker,
                            "event_type": "MERGER_CVR",
                            "headline": str(row.get("title", title)),
                            "date": date,
                            "source": "SEC_8K",
                            "keyword_match": kw,
                            "confidence": 0.8 if "contingent value" in title else 0.5,
                        })
                        break

                for kw in self._PHARMA_KEYWORDS:
                    if kw in title:
                        events.append({
                            "ticker": ticker,
                            "event_type": "PHARMA_MILESTONE",
                            "headline": str(row.get("title", title)),
                            "date": date,
                            "source": "SEC_8K",
                            "keyword_match": kw,
                            "confidence": 0.7,
                        })
                        break

                for kw in self._RESTRUCTURING_KEYWORDS:
                    if kw in title:
                        events.append({
                            "ticker": ticker,
                            "event_type": "RESTRUCTURING",
                            "headline": str(row.get("title", title)),
                            "date": date,
                            "source": "SEC_8K",
                            "keyword_match": kw,
                            "confidence": 0.6,
                        })
                        break
        except Exception:
            pass
        return events

    def _scan_news(self, ticker: str) -> List[dict]:
        """Scan company news for CVR-related catalysts.

        Sources: OpenBB (FMP/Tiingo) + NewsEngine (newsfilter.io real-time).
        """
        events = []
        # Enrich with NewsEngine (newsfilter.io) if available.
        # Uses the cached singleton instead of creating a fresh NewsEngine per
        # ticker — prevents 100+ cold-start instantiations per scan sweep.
        try:
            ne = self._get_news_engine()
            for item in ne.get_ticker_news(ticker, limit=5):
                title = item.get("headline", "").lower()
                all_kw = self._MERGER_KEYWORDS + self._PHARMA_KEYWORDS + self._RESTRUCTURING_KEYWORDS
                for kw in all_kw:
                    if kw in title:
                        etype = "PHARMA_MILESTONE" if kw in self._PHARMA_KEYWORDS else "RESTRUCTURING" if kw in self._RESTRUCTURING_KEYWORDS else "MERGER_CVR"
                        events.append({"ticker": ticker, "event_type": etype, "title": item.get("headline", ""), "_source": "NEWSFILTER"})
                        break
        except Exception:
            pass

        try:
            news = get_company_news(ticker, limit=15, provider="fmp")
            if news.empty:
                return events

            for _, row in news.iterrows():
                title = str(row.get("title", "") or "").lower()
                date = str(row.get("date", "") or row.get("published_utc", "") or "")

                all_keywords = self._MERGER_KEYWORDS + self._PHARMA_KEYWORDS + self._RESTRUCTURING_KEYWORDS
                for kw in all_keywords:
                    if kw in title:
                        if kw in self._PHARMA_KEYWORDS:
                            etype = "PHARMA_MILESTONE"
                        elif kw in self._RESTRUCTURING_KEYWORDS:
                            etype = "RESTRUCTURING"
                        else:
                            etype = "MERGER_CVR"
                        events.append({
                            "ticker": ticker,
                            "event_type": etype,
                            "headline": str(row.get("title", "")),
                            "date": date,
                            "source": "NEWS",
                            "keyword_match": kw,
                            "confidence": 0.6,
                        })
                        break
        except Exception:
            pass
        return events

    def refresh_catalog_prices(self):
        """Update underlying prices in the catalog from live data."""
        import pandas as pd
        for inst in self.catalog:
            # Extract underlying ticker from CVR ticker (e.g., BIIB_CVR → BIIB)
            underlying = inst.ticker.split("_")[0]
            try:
                start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                prices = get_adj_close(underlying, start=start)
                if not prices.empty:
                    price = float(
                        prices.iloc[-1].iloc[0]
                        if hasattr(prices.iloc[-1], "iloc")
                        else prices.iloc[-1]
                    )
                    if price > 0:
                        old_price = inst.underlying_price
                        inst.underlying_price = price
                        # Update vol from recent returns
                        if len(prices) > 10:
                            flat = prices.values.flatten() if hasattr(prices, "values") else prices
                            rets = np.diff(np.log(flat[flat > 0]))
                            if len(rets) > 5:
                                inst.underlying_vol = float(np.std(rets) * np.sqrt(252))
                        logger.debug("%s price updated: $%.2f → $%.2f", underlying, old_price, price)
            except Exception as e:
                logger.debug("Price refresh failed for %s: %s", underlying, e)

    def update_milestone_from_news(self, ticker: str, events: List[dict]):
        """Update milestone completion flags based on discovered news events.

        If news/filings indicate a milestone has been reached (e.g., "FDA approval",
        "Phase III results positive"), mark the corresponding milestone as completed
        and adjust the conditional probability upward.
        """
        matching = [inst for inst in self.catalog if inst.ticker.startswith(ticker)]
        if not matching:
            return

        inst = matching[0]
        for event in events:
            headline = event.get("headline", "").lower()
            for ms in inst.milestones:
                if ms.completed:
                    continue
                ms_name_lower = ms.name.lower()
                # Check if headline indicates this milestone was reached
                if any(word in headline for word in ms_name_lower.split()):
                    if event.get("confidence", 0) >= 0.6:
                        ms.completed = True
                        ms.conditional_prob = min(ms.conditional_prob * 1.2, 0.95)
                        logger.info("CVR milestone updated: %s / %s → COMPLETED (from: %s)",
                                    ticker, ms.name, event.get("source", ""))
                        break

    def discover_and_refresh(self):
        """Full CVR discovery cycle: scan events, refresh prices, update milestones.

        Called by the pipeline at Stage 3.85 before analyze().
        """
        # 1. Refresh underlying prices from live data
        self.refresh_catalog_prices()

        # 2. Scan for new CVR events
        events = self.scan_for_cvr_events()

        # 3. Update milestones from discovered events
        for event in events:
            self.update_milestone_from_news(event["ticker"], [event])

        # 4. Mark as needing re-analysis
        self._analyzed = False

        return {
            "events_found": len(events),
            "catalog_size": len(self.catalog),
            "prices_refreshed": True,
        }

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    def format_cvr_report(self) -> str:
        """Generate ASCII CVR analysis report."""
        if not self._analyzed:
            self.analyze()

        lines = [
            "=" * 82,
            "CVR ENGINE — Contingent Value Rights Analysis",
            "=" * 82,
            "",
            f"  {'Ticker':<14} {'Type':<14} {'Fair$':>7} {'Mkt$':>7} "
            f"{'Mispr':>7} {'P(trig)':>8} {'E[R]':>8} {'Kelly':>6} {'Signal':<12}",
            "  " + "-" * 80,
        ]

        for ticker in sorted(self._results, key=lambda t: self._results[t].mispricing_pct, reverse=True):
            v = self._results[ticker]
            lines.append(
                f"  {ticker:<14} {v.cvr_type.value[:12]:<14} "
                f"${v.fair_value:>6.2f} ${v.market_price:>6.2f} "
                f"{v.mispricing_pct:>+6.1%} "
                f"{v.trigger_probability:>7.1%} "
                f"{v.expected_return:>+7.1%} "
                f"{v.kelly_fraction:>5.1%} "
                f"{v.signal.value:<12}"
            )

        # Model detail for each instrument
        lines.extend(["", "  MODEL BREAKDOWN:"])
        for ticker, v in self._results.items():
            lines.append(
                f"    {ticker}: Binary=${v.binary_option_value:.2f} "
                f"Barrier=${v.barrier_option_value:.2f} "
                f"Milestone=${v.milestone_tree_value:.2f} "
                f"MC=${v.monte_carlo_value:.2f} "
                f"RealOpt=${v.real_option_value:.2f}"
            )

        # Adjustments
        lines.extend(["", "  ADJUSTMENTS:"])
        for ticker, v in self._results.items():
            lines.append(
                f"    {ticker}: Liquidity={v.liquidity_discount:.1%} "
                f"CVA={v.credit_adjustment:.1%} "
                f"Hazard={v.hazard_rate:.2%}/yr "
                f"Decay=${v.time_decay_per_month:.3f}/mo"
            )

        lines.extend(["", "=" * 82])
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF (Abramowitz & Stegun)."""
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1.0 if x >= 0 else -1.0
        x_abs = abs(x)
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs / 2)
        return 0.5 * (1.0 + sign * y)
