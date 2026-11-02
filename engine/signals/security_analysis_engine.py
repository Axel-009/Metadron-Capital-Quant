"""SecurityAnalysisEngine — Graham-Dodd-Klarman L2/L2.5 Analysis Framework.

Implements the complete quantitative framework from Security Analysis
(7th Edition, Benjamin Graham, David L. Dodd, Seth A. Klarman editor).

Architecture:
    TOP-DOWN (Macro Direction) — Part I of Security Analysis
        - Interest rate master variable (capitalization rate)
        - Market-level earnings yield vs bond yield
        - CAPE/Shiller cyclically adjusted P/E
        - Credit cycle indicators (spreads, implied default probability)
        - Equity risk premium assessment
        - Speculative component measurement
        - Maximum investment-grade P/E ceiling

    BOTTOM-UP (Security Collection) — Parts II-V of Security Analysis
        - Graham Number (√(22.5 × EPS × BVPS))
        - NCAV/Net-Net (Current Assets − ALL Liabilities)
        - Margin of Safety scoring (≥33% target)
        - Normalized earnings power (5-10yr average)
        - Earnings stability ratio
        - Balance sheet forensics (DSO, inventory, accrual quality)
        - ROIC-WACC spread (7th Ed. primary metric)
        - DuPont ROE decomposition
        - Capitalization structure leverage analysis
        - Coverage ratios (interest coverage, asset coverage)
        - Owner earnings (Buffett/7th Ed.)
        - Comparative analysis across industry peers
        - Investment vs speculation classification
        - Distressed security identification (Mielle framework)
        - Special situations: convertibles, warrants, liquidation surplus

Pipeline Position: Stage 3.1 — after MetadronCube (L2), before PatternDiscovery (L2/3.2)

Dependencies:
    numpy, pandas (no external ML dependencies)

Usage:
    from engine.signals.security_analysis_engine import SecurityAnalysisEngine
    engine = SecurityAnalysisEngine()
    result = engine.analyze(tickers, macro_snapshot, cube_output)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger("metadron.signals.security_analysis")

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False

# --- Mav-Analysis: MultiAssetAnalyzer (technical scan, cross-asset momentum,
#     sector relative value, vol surface, unusual flow detection) ---------------
try:
    import importlib.util as _ilu
    _mav_spec = _ilu.spec_from_file_location(
        "multi_asset_analysis",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
            / "intelligence_platform" / "Mav-Analysis" / "multi_asset_analysis.py"),
    )
    _mav_mod = _ilu.module_from_spec(_mav_spec)
    _mav_spec.loader.exec_module(_mav_mod)
    MultiAssetAnalyzer = _mav_mod.MultiAssetAnalyzer
    MAV_ANALYSIS_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    MultiAssetAnalyzer = None
    MAV_ANALYSIS_AVAILABLE = False
    logger.info("Mav-Analysis MultiAssetAnalyzer unavailable")

# --- stock-chain: AssetClassAnalyzer (cross-asset correlation, risk parity,
#     HMM regime detection, sector rotation, Information Ratio) ----------------
try:
    _sc_spec = _ilu.spec_from_file_location(
        "asset_class_analyzer",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
            / "intelligence_platform" / "stock-chain" / "asset_class_analyzer.py"),
    )
    _sc_mod = _ilu.module_from_spec(_sc_spec)
    _sc_spec.loader.exec_module(_sc_mod)
    AssetClassAnalyzer = _sc_mod.AssetClassAnalyzer
    STOCK_CHAIN_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    AssetClassAnalyzer = None
    STOCK_CHAIN_AVAILABLE = False
    logger.info("stock-chain AssetClassAnalyzer unavailable")


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class SecurityClass(Enum):
    """Graham's security classification taxonomy (Chapter 4)."""
    INVESTMENT_BOND = "investment_bond"
    SPECULATIVE_BOND = "speculative_bond"
    INVESTMENT_PREFERRED = "investment_preferred"
    SPECULATIVE_PREFERRED = "speculative_preferred"
    INVESTMENT_COMMON = "investment_common"
    SPECULATIVE_COMMON = "speculative_common"
    QUASI_INVESTMENT = "quasi_investment"
    DISTRESSED = "distressed"


class MarketValuationRegime(Enum):
    """Top-down market valuation assessment."""
    DEEPLY_UNDERVALUED = "deeply_undervalued"   # CAPE < 12, ERP > 6%
    UNDERVALUED = "undervalued"                 # CAPE 12-16, ERP > 4%
    FAIR_VALUE = "fair_value"                   # CAPE 16-22, ERP 2-4%
    OVERVALUED = "overvalued"                   # CAPE 22-30, ERP 0-2%
    EXTREMELY_OVERVALUED = "extremely_overvalued"  # CAPE > 30, ERP < 0%


class InvestmentGrade(Enum):
    """Graham's investment grade classification."""
    STRONG_INVESTMENT = "strong_investment"     # MoS ≥ 50%, passes all tests
    INVESTMENT = "investment"                   # MoS ≥ 33%, passes core tests
    BORDERLINE = "borderline"                  # MoS 15-33%, passes some tests
    SPECULATIVE = "speculative"                # MoS < 15% or fails key tests
    AVOID = "avoid"                            # Negative MoS or distressed


class SectorType(Enum):
    """Graham's sector classification for coverage standards."""
    INDUSTRIAL = "industrial"
    UTILITY = "utility"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    REAL_ESTATE = "real_estate"
    NATURAL_RESOURCE = "natural_resource"


# Graham's coverage minimums by sector (Chapter 7-11)
COVERAGE_MINIMUMS = {
    SectorType.INDUSTRIAL: {"avg_7yr": 7.0, "single_year": 5.0, "equity_cushion": 1.0},
    SectorType.UTILITY: {"avg_7yr": 4.0, "single_year": 3.0, "equity_cushion": 2.0},
    SectorType.FINANCIAL: {"avg_7yr": 5.0, "single_year": 4.0, "equity_cushion": 1.5},
    SectorType.TECHNOLOGY: {"avg_7yr": 7.0, "single_year": 5.0, "equity_cushion": 1.0},
    SectorType.REAL_ESTATE: {"avg_7yr": 4.0, "single_year": 3.0, "equity_cushion": 1.5},
    SectorType.NATURAL_RESOURCE: {"avg_7yr": 5.0, "single_year": 4.0, "equity_cushion": 1.0},
}

# Graham's asset liquidation rates (Chapter 42)
LIQUIDATION_RATES = {
    "cash": 1.00,
    "govt_securities": 1.00,
    "marketable_securities": 0.85,
    "accounts_receivable": 0.77,
    "inventory_finished": 0.62,
    "inventory_raw": 0.57,
    "prepaid_expenses": 0.37,
    "net_ppe": 0.25,
    "intangibles": 0.00,
    "long_term_investments": 0.82,
}

# GICS sector → Graham sector mapping
GICS_TO_GRAHAM = {
    "Energy": SectorType.NATURAL_RESOURCE,
    "Materials": SectorType.INDUSTRIAL,
    "Industrials": SectorType.INDUSTRIAL,
    "Consumer Discretionary": SectorType.INDUSTRIAL,
    "Consumer Staples": SectorType.INDUSTRIAL,
    "Health Care": SectorType.INDUSTRIAL,
    "Financials": SectorType.FINANCIAL,
    "Information Technology": SectorType.TECHNOLOGY,
    "Communication Services": SectorType.TECHNOLOGY,
    "Utilities": SectorType.UTILITY,
    "Real Estate": SectorType.REAL_ESTATE,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TopDownAssessment:
    """Market-level (top-down) valuation assessment — Graham Part I."""
    regime: MarketValuationRegime = MarketValuationRegime.FAIR_VALUE
    cape_ratio: float = 0.0
    cape_expected_return: float = 0.0
    market_earnings_yield: float = 0.0
    long_bond_yield: float = 0.0
    equity_risk_premium: float = 0.0
    max_investment_pe: float = 20.0
    speculative_pct: float = 0.0         # % of market cap above investment value
    credit_spread: float = 0.0           # HY - IG spread
    implied_default_prob: float = 0.0
    market_pe: float = 0.0
    dividend_yield: float = 0.0
    confidence: float = 0.0              # 0-1 confidence in assessment
    macro_direction: str = "neutral"     # bullish / neutral / bearish
    commentary: str = ""


@dataclass
class BottomUpScore:
    """Per-security bottom-up analysis score — Graham Parts II-V."""
    ticker: str = ""
    # Classification
    security_class: SecurityClass = SecurityClass.SPECULATIVE_COMMON
    investment_grade: InvestmentGrade = InvestmentGrade.SPECULATIVE
    sector_type: SectorType = SectorType.INDUSTRIAL
    # Valuation
    graham_number: float = 0.0
    intrinsic_value: float = 0.0          # best estimate IV
    margin_of_safety: float = 0.0         # (IV - price) / IV
    market_price: float = 0.0
    ncav_per_share: float = 0.0
    ncav_discount: float = 0.0            # (NCAV - price) / NCAV
    pe_normalized: float = 0.0
    pe_x_pb: float = 0.0                  # P/E × P/B (max 22.5)
    earnings_yield: float = 0.0
    dividend_yield: float = 0.0
    # Earnings quality
    normalized_eps: float = 0.0
    eps_stability: float = 0.0            # std/mean (lower = better)
    eps_growth_cagr: float = 0.0
    earnings_trend: str = "stable"        # improving / stable / deteriorating
    # Balance sheet
    book_value_per_share: float = 0.0
    tangible_bvps: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    nwc_per_share: float = 0.0
    # Profitability (7th Ed.)
    roic: float = 0.0
    wacc: float = 0.0
    roic_wacc_spread: float = 0.0         # positive = value creation
    roe: float = 0.0
    dupont_margin: float = 0.0
    dupont_turnover: float = 0.0
    dupont_leverage: float = 0.0
    fcf_margin: float = 0.0
    owner_earnings: float = 0.0           # Buffett/7th Ed.
    economic_profit: float = 0.0          # NOPAT - (WACC × IC)
    # Coverage & leverage
    interest_coverage: float = 0.0
    debt_to_equity: float = 0.0
    equity_cushion_ratio: float = 0.0
    financial_leverage: float = 0.0
    operating_leverage: float = 0.0
    combined_leverage: float = 0.0
    # Forensics
    accrual_quality: float = 0.0          # FCF/NI ratio (>1 = good)
    dso_trend: float = 0.0               # positive = deteriorating
    inventory_trend: float = 0.0          # positive = building
    # Scores
    composite_score: float = 0.0          # 0-100 overall Graham-Dodd score
    passes_two_part_test: bool = False    # safety + satisfactory return
    passes_graham_criteria: bool = False  # all 7 Graham criteria
    # Egan-Jones credit proxy (D/E + CR fundamental analysis)
    egan_jones_tier: str = ""             # A-F based on D/E + CR thresholds
    egan_jones_ig: bool = False           # True if IG (tier A or B)


@dataclass
class ComparativeAnalysis:
    """Cross-security comparative analysis (Chapter 49)."""
    ticker: str = ""
    peer_group: list = field(default_factory=list)
    relative_pe: float = 0.0             # vs peer median
    relative_pb: float = 0.0
    relative_ey: float = 0.0             # earnings yield vs peers
    relative_roic: float = 0.0
    relative_coverage: float = 0.0
    relative_margin: float = 0.0
    exchange_premium: float = 0.0         # Graham's 50% rule
    homogeneity: str = "moderate"         # high / moderate / low


@dataclass
class SecurityAnalysisResult:
    """Complete output from the Security Analysis Engine."""
    # Top-down
    top_down: TopDownAssessment = field(default_factory=TopDownAssessment)
    # Bottom-up per ticker
    bottom_up: dict = field(default_factory=dict)    # ticker → BottomUpScore
    # Comparative
    comparatives: dict = field(default_factory=dict)  # ticker → ComparativeAnalysis
    # Aggregated signals
    investment_universe: list = field(default_factory=list)   # tickers passing investment grade
    speculative_universe: list = field(default_factory=list)  # tickers classified speculative
    distressed_opportunities: list = field(default_factory=list)  # Mielle framework
    net_net_candidates: list = field(default_factory=list)    # NCAV discount > 33%
    # Position sizing adjustments
    margin_of_safety_weights: dict = field(default_factory=dict)  # ticker → weight multiplier
    # Metadata
    analysis_timestamp: str = ""
    tickers_analyzed: int = 0
    investment_grade_count: int = 0
    avg_margin_of_safety: float = 0.0


# ---------------------------------------------------------------------------
# Top-Down Analyzer — Market-Level Graham Framework
# ---------------------------------------------------------------------------

class TopDownAnalyzer:
    """Implements Graham Part I: Market-level analytical frameworks.

    Before any individual security is evaluated, the analyst must establish
    the current macro context. Graham embeds specific mathematical anchors
    for market-level assessment.

    Key inputs from MacroEngine/MetadronCube:
        - 10Y Treasury yield
        - S&P 500 P/E, earnings
        - VIX level
        - Credit spreads (HY-IG)
        - Regime classification
    """

    # Graham's max investment PE by rate environment (Chapter 39)
    RATE_TO_MAX_PE = [
        (0.00, 0.02, 25.0),   # ZIRP: max 25x (7th Ed. warns of danger)
        (0.02, 0.04, 20.0),   # Low rates: Graham's classic 20x ceiling
        (0.04, 0.06, 16.0),   # Moderate: original 1934 standard
        (0.06, 0.08, 14.0),   # Higher: conservative adjustment
        (0.08, 0.15, 10.0),   # High rates: pre-1929 standard
    ]

    def analyze(self, macro_data: dict) -> TopDownAssessment:
        """Run full top-down assessment.

        Args:
            macro_data: dict with keys:
                - treasury_10y: 10Y UST yield
                - sp500_pe: trailing S&P 500 P/E
                - sp500_eps: S&P 500 trailing EPS
                - sp500_level: current S&P 500 index level
                - cape: Shiller CAPE ratio
                - dividend_yield: S&P 500 dividend yield
                - hy_spread: high-yield credit spread (bps)
                - ig_spread: investment-grade spread (bps)
                - vix: VIX level
                - gdp_growth: real GDP growth estimate
                - regime: current macro regime string
        """
        result = TopDownAssessment()

        t10y = macro_data.get("treasury_10y", 0.04)
        sp_pe = macro_data.get("sp500_pe", 20.0)
        sp_eps = macro_data.get("sp500_eps", 0.0)
        sp_level = macro_data.get("sp500_level", 5000.0)
        cape = macro_data.get("cape", 25.0)
        div_yield = macro_data.get("dividend_yield", 0.015)
        hy_spread = macro_data.get("hy_spread", 400.0)
        ig_spread = macro_data.get("ig_spread", 100.0)
        vix = macro_data.get("vix", 20.0)
        gdp_growth = macro_data.get("gdp_growth", 0.02)

        # 2.1 Interest Rate — The Master Variable
        # Capitalization Rate = Bond Yield + Risk Premium
        result.long_bond_yield = t10y

        # Market earnings yield
        result.market_earnings_yield = 1.0 / max(sp_pe, 1.0)
        result.market_pe = sp_pe
        result.dividend_yield = div_yield

        # Equity Risk Premium = Earnings Yield - Bond Yield
        result.equity_risk_premium = result.market_earnings_yield - t10y

        # 2.2 CAPE assessment
        result.cape_ratio = cape
        # CAPE-Based Expected Return ≈ 1/CAPE + GDP growth
        result.cape_expected_return = (1.0 / max(cape, 1.0)) + gdp_growth

        # Maximum Investment-Grade P/E (Graham Chapter 39)
        # "About 20 times average earnings is as high a price as can be paid
        #  in an investment purchase of common stock."
        result.max_investment_pe = self._max_pe_for_rate(t10y)

        # Speculative Component of Market
        # = (Actual P/E - Max Investment P/E) × Earnings / Market Cap
        if sp_pe > result.max_investment_pe:
            result.speculative_pct = (
                (sp_pe - result.max_investment_pe) / sp_pe
            )
        else:
            result.speculative_pct = 0.0

        # 2.3 Credit Cycle Indicators
        result.credit_spread = hy_spread / 100.0  # convert bps to %
        # Implied Default Probability ≈ Spread / (1 - Recovery Rate)
        recovery_rate = 0.40  # standard assumption
        result.implied_default_prob = (hy_spread / 10000.0) / (1.0 - recovery_rate)

        # Determine valuation regime
        result.regime = self._classify_regime(
            cape, result.equity_risk_premium, result.speculative_pct, vix
        )

        # Confidence based on signal agreement
        signals = []
        signals.append(1.0 if result.equity_risk_premium > 0.03 else
                       0.5 if result.equity_risk_premium > 0.0 else 0.0)
        signals.append(1.0 if cape < 18 else 0.5 if cape < 25 else 0.0)
        signals.append(1.0 if result.speculative_pct < 0.1 else
                       0.5 if result.speculative_pct < 0.3 else 0.0)
        signals.append(1.0 if vix < 20 else 0.5 if vix < 30 else 0.0)
        result.confidence = np.mean(signals)

        # Macro direction
        if result.regime in (MarketValuationRegime.DEEPLY_UNDERVALUED,
                             MarketValuationRegime.UNDERVALUED):
            result.macro_direction = "bullish"
        elif result.regime == MarketValuationRegime.FAIR_VALUE:
            result.macro_direction = "neutral"
        else:
            result.macro_direction = "bearish"

        # Commentary
        result.commentary = self._generate_commentary(result)

        logger.info(
            f"TopDown: regime={result.regime.value} CAPE={cape:.1f} "
            f"ERP={result.equity_risk_premium*100:.1f}% "
            f"maxPE={result.max_investment_pe:.0f}x "
            f"spec%={result.speculative_pct*100:.1f}%"
        )

        return result

    def _max_pe_for_rate(self, rate: float) -> float:
        """Graham's rate-adjusted maximum P/E for investment-grade equity."""
        for low, high, max_pe in self.RATE_TO_MAX_PE:
            if low <= rate < high:
                return max_pe
        # Above 15% → very conservative
        return 8.0

    def _classify_regime(self, cape: float, erp: float,
                         spec_pct: float, vix: float) -> MarketValuationRegime:
        """Classify market valuation regime."""
        score = 0.0

        # CAPE scoring
        if cape < 12:
            score += 2.0
        elif cape < 16:
            score += 1.0
        elif cape < 22:
            score += 0.0
        elif cape < 30:
            score -= 1.0
        else:
            score -= 2.0

        # ERP scoring
        if erp > 0.06:
            score += 2.0
        elif erp > 0.04:
            score += 1.0
        elif erp > 0.02:
            score += 0.0
        elif erp > 0.0:
            score -= 1.0
        else:
            score -= 2.0

        # Speculative component
        if spec_pct > 0.30:
            score -= 1.0
        elif spec_pct < 0.05:
            score += 0.5

        # VIX regime
        if vix > 35:
            score += 0.5  # fear = opportunity (Graham contrarian)
        elif vix < 12:
            score -= 0.5  # complacency = danger

        if score >= 3.0:
            return MarketValuationRegime.DEEPLY_UNDERVALUED
        elif score >= 1.0:
            return MarketValuationRegime.UNDERVALUED
        elif score >= -1.0:
            return MarketValuationRegime.FAIR_VALUE
        elif score >= -3.0:
            return MarketValuationRegime.OVERVALUED
        else:
            return MarketValuationRegime.EXTREMELY_OVERVALUED

    def _generate_commentary(self, td: TopDownAssessment) -> str:
        """Generate Graham-style market commentary."""
        parts = []
        if td.equity_risk_premium < 0:
            parts.append(
                "CAUTION: Negative ERP — equities offer no margin of safety "
                "vs bonds. Graham: 'equities no longer investment-grade as "
                "a class versus bonds.'"
            )
        elif td.equity_risk_premium < 0.02:
            parts.append(
                "LOW ERP — equity risk premium thin. Market priced for "
                "perfection with minimal margin of safety."
            )

        if td.cape_ratio > 30:
            parts.append(
                f"CAPE at {td.cape_ratio:.1f}x — historically extreme. "
                "7th Ed. (Klarman): TINA distortions; Graham warns of danger."
            )

        if td.speculative_pct > 0.25:
            parts.append(
                f"Speculative component = {td.speculative_pct*100:.0f}% "
                "of market cap. Price paid for expectations, not facts."
            )

        if td.credit_spread > 6.0:
            parts.append(
                f"Credit spreads wide at {td.credit_spread*100:.0f}bps — "
                "distressed debt opportunities likely (Mielle framework)."
            )

        if td.implied_default_prob > 0.05:
            parts.append(
                f"Implied default rate {td.implied_default_prob*100:.1f}% — "
                "credit cycle stress elevated."
            )

        if not parts:
            parts.append(
                f"Market conditions: {td.regime.value}. "
                f"CAPE={td.cape_ratio:.1f}x, ERP={td.equity_risk_premium*100:.1f}%."
            )

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Bottom-Up Analyzer — Individual Security Graham Framework
# ---------------------------------------------------------------------------

class BottomUpAnalyzer:
    """Implements Graham Parts II-V: Individual security analysis.

    'Security analysis does not seek to determine exactly what the intrinsic
    value is. It needs only to establish that the value is adequate — or
    considerably higher or lower than the market price.'
    — Graham & Dodd, Chapter 1

    Computes all Graham-Dodd valuation metrics, quality tests, and
    classification for each security in the universe.
    """

    # Graham's 7 investment criteria (Intelligent Investor / SA 7th Ed.)
    GRAHAM_CRITERIA = {
        "adequate_size": True,       # Revenue > threshold
        "strong_financial": True,    # Current ratio ≥ 2, debt < NWC
        "earnings_stability": True,  # Positive EPS each of last 10 years
        "dividend_record": True,     # Uninterrupted dividends 20+ years
        "earnings_growth": True,     # Min 33% EPS growth over 10 years
        "moderate_pe": True,         # P/E ≤ 15 (on 3yr avg earnings)
        "moderate_pb": True,         # P/B ≤ 1.5 (or P/E × P/B ≤ 22.5)
    }

    def analyze_security(self, ticker: str, data: dict,
                         top_down: TopDownAssessment) -> BottomUpScore:
        """Full Graham-Dodd analysis of a single security.

        Args:
            ticker: Stock ticker
            data: dict with financial data:
                - price: current market price
                - eps_history: list of annual EPS (oldest→newest, 5-10yrs)
                - eps_ttm: trailing 12-month EPS
                - bvps: book value per share
                - tangible_bvps: tangible BVPS (ex goodwill/intangibles)
                - dps: dividend per share (annual)
                - revenue: annual revenue
                - ebit: annual EBIT
                - net_income: annual net income
                - total_assets: total assets
                - current_assets: current assets
                - current_liabilities: current liabilities
                - total_liabilities: total liabilities
                - total_debt: total debt (short + long)
                - cash: cash and equivalents
                - interest_expense: annual interest expense
                - depreciation: annual D&A
                - capex: annual capital expenditure
                - shares_out: shares outstanding
                - market_cap: market capitalization
                - fcf: free cash flow
                - sector: GICS sector name
                - dso_current: days sales outstanding (current)
                - dso_prior: DSO prior year
                - inventory_current: current inventory
                - inventory_prior: prior year inventory
                - revenue_prior: prior year revenue
            top_down: TopDownAssessment from top-down analyzer
        """
        score = BottomUpScore(ticker=ticker)

        price = data.get("price", 0.0)
        eps_history = data.get("eps_history", [])
        eps_ttm = data.get("eps_ttm", 0.0)
        bvps = data.get("bvps", 0.0)
        tangible_bvps = data.get("tangible_bvps", bvps)
        dps = data.get("dps", 0.0)
        revenue = data.get("revenue", 0.0)
        ebit = data.get("ebit", 0.0)
        net_income = data.get("net_income", 0.0)
        total_assets = data.get("total_assets", 0.0)
        current_assets = data.get("current_assets", 0.0)
        current_liabilities = data.get("current_liabilities", 0.0)
        total_liabilities = data.get("total_liabilities", 0.0)
        total_debt = data.get("total_debt", 0.0)
        cash = data.get("cash", 0.0)
        interest_expense = data.get("interest_expense", 0.0)
        depreciation = data.get("depreciation", 0.0)
        capex = data.get("capex", 0.0)
        shares = data.get("shares_out", 1.0)
        market_cap = data.get("market_cap", price * shares)
        fcf = data.get("fcf", 0.0)
        sector = data.get("sector", "Industrials")

        if price <= 0 or shares <= 0:
            score.investment_grade = InvestmentGrade.AVOID
            return score

        score.market_price = price
        score.sector_type = GICS_TO_GRAHAM.get(sector, SectorType.INDUSTRIAL)

        # ---- 6.1 Core Intrinsic Value ----

        # Normalized EPS (5-10yr average)
        if eps_history and len(eps_history) >= 3:
            score.normalized_eps = float(np.mean(eps_history))
        else:
            score.normalized_eps = eps_ttm

        # 6.3 Graham Number = √(22.5 × EPS × BVPS)
        if score.normalized_eps > 0 and tangible_bvps > 0:
            score.graham_number = np.sqrt(22.5 * score.normalized_eps * tangible_bvps)
        else:
            score.graham_number = 0.0

        # P/E × P/B constraint (max 22.5)
        pe = price / eps_ttm if eps_ttm > 0 else 999.0
        pb = price / tangible_bvps if tangible_bvps > 0 else 999.0
        score.pe_x_pb = pe * pb

        # Normalized P/E
        score.pe_normalized = (
            price / score.normalized_eps if score.normalized_eps > 0 else 999.0
        )

        # Earnings Yield = EPS / Price
        score.earnings_yield = eps_ttm / price if price > 0 else 0.0

        # Dividend Yield
        score.dividend_yield = dps / price if price > 0 else 0.0

        # ---- 8.2 NCAV — Net-Net Formula ----
        # NCAV = Current Assets − ALL Liabilities
        ncav = current_assets - total_liabilities
        score.ncav_per_share = ncav / shares if shares > 0 else 0.0
        if score.ncav_per_share > 0:
            score.ncav_discount = (
                (score.ncav_per_share - price) / score.ncav_per_share
            )
        else:
            score.ncav_discount = -1.0  # negative NCAV

        # ---- 8.1 Book Value ----
        score.book_value_per_share = bvps
        score.tangible_bvps = tangible_bvps

        # ---- 8.3 Working Capital ----
        score.current_ratio = (
            current_assets / current_liabilities
            if current_liabilities > 0 else 999.0
        )
        liquid_assets = cash + data.get("short_term_investments", 0) + \
            data.get("accounts_receivable", current_assets * 0.3)
        score.quick_ratio = (
            liquid_assets / current_liabilities
            if current_liabilities > 0 else 999.0
        )
        nwc = current_assets - current_liabilities
        score.nwc_per_share = nwc / shares if shares > 0 else 0.0

        # ---- 7.2 Earnings Stability & Trend ----
        if eps_history and len(eps_history) >= 3:
            eps_arr = np.array(eps_history, dtype=float)
            eps_mean = np.mean(eps_arr)
            eps_std = np.std(eps_arr)
            score.eps_stability = (
                eps_std / abs(eps_mean) if eps_mean != 0 else 999.0
            )
            # CAGR
            if eps_arr[0] > 0 and eps_arr[-1] > 0 and len(eps_arr) > 1:
                n = len(eps_arr) - 1
                score.eps_growth_cagr = (eps_arr[-1] / eps_arr[0]) ** (1.0 / n) - 1.0
            # Trend
            if len(eps_arr) >= 3:
                recent_avg = np.mean(eps_arr[-3:])
                older_avg = np.mean(eps_arr[:3])
                if older_avg > 0:
                    if recent_avg > older_avg * 1.1:
                        score.earnings_trend = "improving"
                    elif recent_avg < older_avg * 0.9:
                        score.earnings_trend = "deteriorating"
                    else:
                        score.earnings_trend = "stable"

        # ---- 6.6 Return on Capital (7th Edition) ----
        # ROIC = NOPAT / Invested Capital
        equity = total_assets - total_liabilities
        invested_capital = equity + total_debt - cash
        if invested_capital > 0:
            tax_rate = 0.21
            nopat = ebit * (1 - tax_rate)
            score.roic = nopat / invested_capital
        else:
            score.roic = 0.0

        # WACC estimate
        cost_of_equity = top_down.long_bond_yield + 0.05  # ERP ~5%
        cost_of_debt = top_down.long_bond_yield + 0.02  # spread ~2%
        total_capital = equity + total_debt
        if total_capital > 0:
            e_weight = equity / total_capital
            d_weight = total_debt / total_capital
            score.wacc = cost_of_equity * e_weight + cost_of_debt * 0.79 * d_weight
        else:
            score.wacc = cost_of_equity

        # ROIC-WACC spread
        score.roic_wacc_spread = score.roic - score.wacc

        # ROE
        if equity > 0:
            score.roe = net_income / equity
        # DuPont decomposition
        if revenue > 0 and total_assets > 0 and equity > 0:
            score.dupont_margin = net_income / revenue
            score.dupont_turnover = revenue / total_assets
            score.dupont_leverage = total_assets / equity

        # FCF margin
        if revenue > 0:
            score.fcf_margin = fcf / revenue

        # Owner Earnings (Buffett/7th Ed.)
        # = Net Income + D&A - Maintenance CapEx - ΔWC
        maintenance_capex = capex * 0.7  # estimate 70% is maintenance
        score.owner_earnings = net_income + depreciation - maintenance_capex

        # Economic Profit = NOPAT - (WACC × Invested Capital)
        if invested_capital > 0:
            score.economic_profit = (score.roic - score.wacc) * invested_capital

        # ---- 9.2 Coverage Ratios ----
        if interest_expense > 0:
            score.interest_coverage = ebit / interest_expense
        else:
            score.interest_coverage = 999.0  # no debt

        # Debt-to-equity
        if equity > 0:
            score.debt_to_equity = total_debt / equity

        # Egan-Jones credit proxy (fundamental D/E + CR analysis)
        de = score.debt_to_equity
        cr = score.current_ratio
        if de < 0.5 and cr > 1.5:
            score.egan_jones_tier = "A"
        elif de < 1.0 and cr > 1.2:
            score.egan_jones_tier = "B"
        elif de < 2.0 and cr > 0.8:
            score.egan_jones_tier = "C"
        elif de < 3.0 and cr > 0.8:
            score.egan_jones_tier = "D"
        elif de < 5.0 and cr > 0.5:
            score.egan_jones_tier = "E"
        else:
            score.egan_jones_tier = "F"
        score.egan_jones_ig = score.egan_jones_tier in ("A", "B")

        # Equity cushion ratio (Stock-Value Ratio)
        # = Market Cap / Face Value of Debt
        if total_debt > 0:
            score.equity_cushion_ratio = market_cap / total_debt

        # Financial Leverage = EBIT / (EBIT - Interest)
        if ebit > 0 and interest_expense > 0:
            ebit_minus_int = ebit - interest_expense
            score.financial_leverage = (
                ebit / ebit_minus_int if ebit_minus_int > 0 else 999.0
            )
        else:
            score.financial_leverage = 1.0

        # Operating Leverage estimate
        # = Contribution Margin / EBIT (approximate)
        cogs = revenue * 0.6  # rough estimate
        contribution = revenue - cogs
        if ebit > 0:
            score.operating_leverage = contribution / ebit

        score.combined_leverage = score.operating_leverage * score.financial_leverage

        # ---- 8.4 Balance Sheet Forensics ----
        # Accrual quality = FCF / Net Income
        if net_income > 0:
            score.accrual_quality = fcf / net_income
        elif net_income < 0 and fcf > 0:
            score.accrual_quality = 1.5  # positive FCF despite losses = OK
        else:
            score.accrual_quality = 0.0

        # DSO trend (positive = deteriorating)
        dso_current = data.get("dso_current", 0.0)
        dso_prior = data.get("dso_prior", 0.0)
        score.dso_trend = dso_current - dso_prior

        # Inventory trend (positive = building)
        inv_current = data.get("inventory_current", 0.0)
        inv_prior = data.get("inventory_prior", 0.0)
        rev_current = revenue
        rev_prior = data.get("revenue_prior", revenue)
        if rev_prior > 0 and rev_current > 0:
            inv_to_rev_current = inv_current / rev_current
            inv_to_rev_prior = inv_prior / rev_prior
            score.inventory_trend = inv_to_rev_current - inv_to_rev_prior

        # ---- 6.2 Margin of Safety ----
        # Compute intrinsic value as the MEDIAN of multiple methods
        iv_estimates = []

        # Method 1: Graham Number
        if score.graham_number > 0:
            iv_estimates.append(score.graham_number)

        # Method 2: Earning Power Capitalization
        if score.normalized_eps > 0 and score.wacc > 0:
            earning_power_iv = score.normalized_eps / score.wacc
            iv_estimates.append(earning_power_iv)

        # Method 3: Max Investment P/E × Normalized EPS
        if score.normalized_eps > 0:
            max_pe_iv = score.normalized_eps * top_down.max_investment_pe
            iv_estimates.append(max_pe_iv)

        # Method 4: NCAV (most conservative)
        if score.ncav_per_share > 0:
            iv_estimates.append(score.ncav_per_share)

        # Method 5: DCF-lite (7th Ed.)
        if fcf > 0 and shares > 0:
            fcf_per_share = fcf / shares
            growth = min(score.eps_growth_cagr, 0.05) if score.eps_growth_cagr > 0 else 0.02
            discount = max(score.wacc, 0.08)
            if discount > growth:
                dcf_iv = fcf_per_share * (1 + growth) / (discount - growth)
                # Graham: if ≥50% of value is terminal, too speculative
                # So we cap at 15x FCF
                dcf_iv = min(dcf_iv, fcf_per_share * 15)
                iv_estimates.append(dcf_iv)

        if iv_estimates:
            score.intrinsic_value = float(np.median(iv_estimates))
        else:
            score.intrinsic_value = tangible_bvps if tangible_bvps > 0 else 0.0

        # Margin of Safety = (IV - Price) / IV
        if score.intrinsic_value > 0:
            score.margin_of_safety = (
                (score.intrinsic_value - price) / score.intrinsic_value
            )
        else:
            score.margin_of_safety = -1.0

        # ---- Classification ----
        score.security_class = self._classify_security(score, data)
        score.investment_grade = self._grade_investment(score, top_down)
        score.passes_two_part_test = self._two_part_test(score, top_down)
        score.passes_graham_criteria = self._graham_criteria_check(score, data)
        score.composite_score = self._compute_composite(score, top_down)

        return score

    def _classify_security(self, score: BottomUpScore, data: dict) -> SecurityClass:
        """Graham's security classification (Chapter 4)."""
        if score.margin_of_safety >= 0.33 and score.current_ratio >= 2.0:
            if score.eps_stability < 0.5 and score.interest_coverage >= 5.0:
                return SecurityClass.INVESTMENT_COMMON
        if score.ncav_discount >= 0.33:
            return SecurityClass.INVESTMENT_COMMON  # net-net bargain
        if score.margin_of_safety >= 0.15:
            return SecurityClass.QUASI_INVESTMENT
        if score.margin_of_safety < -0.5:
            return SecurityClass.DISTRESSED
        return SecurityClass.SPECULATIVE_COMMON

    def _grade_investment(self, score: BottomUpScore,
                          td: TopDownAssessment) -> InvestmentGrade:
        """Assign investment grade based on Graham composite criteria."""
        mos = score.margin_of_safety
        passes = 0
        total = 0

        # Test 1: Margin of Safety
        total += 1
        if mos >= 0.33:
            passes += 1

        # Test 2: Earnings yield > bond yield
        total += 1
        if score.earnings_yield > td.long_bond_yield:
            passes += 1

        # Test 3: Current ratio ≥ 2
        total += 1
        if score.current_ratio >= 2.0:
            passes += 1

        # Test 4: Earnings stability
        total += 1
        if score.eps_stability < 0.50:
            passes += 1

        # Test 5: P/E × P/B ≤ 22.5
        total += 1
        if score.pe_x_pb <= 22.5:
            passes += 1

        # Test 6: Positive ROIC-WACC spread (7th Ed.)
        total += 1
        if score.roic_wacc_spread > 0:
            passes += 1

        # Test 7: Interest coverage adequate
        total += 1
        mins = COVERAGE_MINIMUMS.get(score.sector_type, COVERAGE_MINIMUMS[SectorType.INDUSTRIAL])
        if score.interest_coverage >= mins["single_year"]:
            passes += 1

        # Test 8: Accrual quality
        total += 1
        if score.accrual_quality >= 0.80:
            passes += 1

        pass_rate = passes / total if total > 0 else 0

        if mos >= 0.50 and pass_rate >= 0.75:
            return InvestmentGrade.STRONG_INVESTMENT
        elif mos >= 0.33 and pass_rate >= 0.60:
            return InvestmentGrade.INVESTMENT
        elif mos >= 0.15 and pass_rate >= 0.40:
            return InvestmentGrade.BORDERLINE
        elif mos >= 0.0:
            return InvestmentGrade.SPECULATIVE
        else:
            return InvestmentGrade.AVOID

    def _two_part_test(self, score: BottomUpScore,
                       td: TopDownAssessment) -> bool:
        """Graham's Two-Part Investment Test (Chapter 4).

        Test 1: Safety of Principal
            Purchase Price ≤ Conservative IV × (1 - MoS)
        Test 2: Satisfactory Return
            Expected Return ≥ Minimum Required Rate
        """
        # Test 1: Safety
        safety = score.margin_of_safety >= 0.15

        # Test 2: Satisfactory return
        min_required = td.long_bond_yield + 0.02  # bond yield + reasonable premium
        expected_return = score.earnings_yield + max(score.eps_growth_cagr, 0)
        satisfactory = expected_return >= min_required

        return safety and satisfactory

    def _graham_criteria_check(self, score: BottomUpScore, data: dict) -> bool:
        """Check all 7 Graham defensive investor criteria."""
        passes = 0

        # 1. Adequate size (revenue > $2B modern equivalent)
        if data.get("revenue", 0) > 2e9:
            passes += 1

        # 2. Strong financial condition (current ratio ≥ 2)
        if score.current_ratio >= 2.0 and score.debt_to_equity < 1.0:
            passes += 1

        # 3. Earnings stability (positive EPS every year)
        eps_hist = data.get("eps_history", [])
        if eps_hist and all(e > 0 for e in eps_hist) and len(eps_hist) >= 5:
            passes += 1

        # 4. Dividend record (continuous dividends)
        if data.get("dps", 0) > 0:
            passes += 1

        # 5. Earnings growth (≥ 33% over 10 years ≈ 2.9% CAGR)
        if score.eps_growth_cagr >= 0.029:
            passes += 1

        # 6. Moderate P/E (≤ 15 on 3yr avg)
        if score.pe_normalized <= 15.0:
            passes += 1

        # 7. Moderate P/B (≤ 1.5 or P/E × P/B ≤ 22.5)
        if score.pe_x_pb <= 22.5:
            passes += 1

        return passes >= 5  # pass 5 of 7 minimum

    def _compute_composite(self, score: BottomUpScore,
                           td: TopDownAssessment) -> float:
        """Compute 0-100 composite Graham-Dodd score.

        Weighted components:
            25% — Margin of Safety
            20% — Earnings Quality (stability, accrual, trend)
            15% — Balance Sheet Strength
            15% — ROIC/Economic Profit (7th Ed.)
            10% — Coverage & Leverage
            10% — Valuation (P/E × P/B constraint)
             5% — Top-Down Adjustment
        """
        components = []

        # 1. Margin of Safety (25%)
        mos_score = np.clip(score.margin_of_safety * 100 + 50, 0, 100)
        components.append(("mos", 0.25, mos_score))

        # 2. Earnings Quality (20%)
        eq_score = 50.0
        if score.eps_stability < 0.20:
            eq_score += 20
        elif score.eps_stability < 0.40:
            eq_score += 10
        elif score.eps_stability > 0.80:
            eq_score -= 20
        if score.accrual_quality >= 1.0:
            eq_score += 15
        elif score.accrual_quality >= 0.8:
            eq_score += 5
        elif score.accrual_quality < 0.5:
            eq_score -= 15
        if score.earnings_trend == "improving":
            eq_score += 10
        elif score.earnings_trend == "deteriorating":
            eq_score -= 10
        eq_score = np.clip(eq_score, 0, 100)
        components.append(("earnings_quality", 0.20, eq_score))

        # 3. Balance Sheet (15%)
        bs_score = 50.0
        if score.current_ratio >= 2.5:
            bs_score += 20
        elif score.current_ratio >= 2.0:
            bs_score += 10
        elif score.current_ratio < 1.0:
            bs_score -= 25
        if score.ncav_per_share > 0:
            bs_score += 15
        if score.debt_to_equity < 0.3:
            bs_score += 10
        elif score.debt_to_equity > 1.5:
            bs_score -= 15
        bs_score = np.clip(bs_score, 0, 100)
        components.append(("balance_sheet", 0.15, bs_score))

        # 4. ROIC/Economic Profit (15%)
        roic_score = 50.0
        if score.roic_wacc_spread > 0.10:
            roic_score += 30
        elif score.roic_wacc_spread > 0.05:
            roic_score += 20
        elif score.roic_wacc_spread > 0:
            roic_score += 10
        elif score.roic_wacc_spread < -0.05:
            roic_score -= 25
        roic_score = np.clip(roic_score, 0, 100)
        components.append(("roic", 0.15, roic_score))

        # 5. Coverage & Leverage (10%)
        cov_score = 50.0
        if score.interest_coverage >= 10.0:
            cov_score += 25
        elif score.interest_coverage >= 7.0:
            cov_score += 15
        elif score.interest_coverage >= 5.0:
            cov_score += 5
        elif score.interest_coverage < 2.0:
            cov_score -= 25
        if score.combined_leverage > 5.0:
            cov_score -= 15
        cov_score = np.clip(cov_score, 0, 100)
        components.append(("coverage", 0.10, cov_score))

        # 6. Valuation (10%)
        val_score = 50.0
        if score.pe_x_pb <= 15:
            val_score += 25
        elif score.pe_x_pb <= 22.5:
            val_score += 10
        elif score.pe_x_pb > 45:
            val_score -= 25
        if score.earnings_yield > td.long_bond_yield + 0.03:
            val_score += 15
        val_score = np.clip(val_score, 0, 100)
        components.append(("valuation", 0.10, val_score))

        # 7. Top-Down Adjustment (5%)
        td_score = 50.0
        if td.regime == MarketValuationRegime.DEEPLY_UNDERVALUED:
            td_score = 90.0
        elif td.regime == MarketValuationRegime.UNDERVALUED:
            td_score = 70.0
        elif td.regime == MarketValuationRegime.OVERVALUED:
            td_score = 30.0
        elif td.regime == MarketValuationRegime.EXTREMELY_OVERVALUED:
            td_score = 10.0
        components.append(("top_down", 0.05, td_score))

        # Weighted composite
        total = sum(w * s for _, w, s in components)
        return round(total, 2)


# ---------------------------------------------------------------------------
# Comparative Analyzer — Graham Chapter 49
# ---------------------------------------------------------------------------

class ComparativeAnalyzer:
    """Graham's Standard Comparative Forms (Chapter 49).

    'The reliability of comparative analysis depends entirely on how
    similar the companies' futures will be.'

    Compares securities within peer groups on standardized metrics.
    """

    def analyze(self, scores: dict, peer_groups: dict) -> dict:
        """Run comparative analysis across peer groups.

        Args:
            scores: ticker → BottomUpScore dict
            peer_groups: ticker → list of peer tickers

        Returns:
            ticker → ComparativeAnalysis dict
        """
        results = {}

        for ticker, peers in peer_groups.items():
            if ticker not in scores:
                continue

            score = scores[ticker]
            peer_scores = [scores[p] for p in peers if p in scores]

            if not peer_scores:
                results[ticker] = ComparativeAnalysis(
                    ticker=ticker, peer_group=peers
                )
                continue

            comp = ComparativeAnalysis(ticker=ticker, peer_group=peers)

            # Compute peer medians
            peer_pe = [s.pe_normalized for s in peer_scores if 0 < s.pe_normalized < 500]
            peer_pb = [s.market_price / s.tangible_bvps for s in peer_scores
                       if s.tangible_bvps > 0]
            peer_ey = [s.earnings_yield for s in peer_scores if s.earnings_yield > 0]
            peer_roic = [s.roic for s in peer_scores if s.roic != 0]
            peer_cov = [s.interest_coverage for s in peer_scores
                        if 0 < s.interest_coverage < 500]
            peer_margin = [s.dupont_margin for s in peer_scores
                          if s.dupont_margin != 0]

            if peer_pe:
                comp.relative_pe = score.pe_normalized / np.median(peer_pe)
            if peer_pb and score.tangible_bvps > 0:
                my_pb = score.market_price / score.tangible_bvps
                comp.relative_pb = my_pb / np.median(peer_pb)
            if peer_ey:
                comp.relative_ey = score.earnings_yield / np.median(peer_ey)
            if peer_roic:
                comp.relative_roic = score.roic / np.median(peer_roic)
            if peer_cov:
                comp.relative_coverage = score.interest_coverage / np.median(peer_cov)
            if peer_margin:
                comp.relative_margin = score.dupont_margin / np.median(peer_margin)

            # Graham's Exchange Rule: "Get at least 50% more for your money"
            if peer_ey and score.earnings_yield > 0:
                best_peer_ey = max(peer_ey)
                comp.exchange_premium = (best_peer_ey / score.earnings_yield) - 1.0

            results[ticker] = comp

        return results


# ---------------------------------------------------------------------------
# Main Engine — SecurityAnalysisEngine
# ---------------------------------------------------------------------------

class SecurityAnalysisEngine:
    """Graham-Dodd-Klarman Security Analysis Engine.

    Pipeline position: Stage 3.1 (L2 → L2.5)
    Sits between MetadronCube and PatternDiscoveryEngine.

    Top-down: Uses macro data to assess market-level investment conditions.
    Bottom-up: Applies Graham-Dodd security analysis to each ticker.

    The engine produces:
        1. Market valuation regime (investment vs speculative environment)
        2. Per-ticker investment grade and margin of safety
        3. Universe filtering (investment-grade vs speculative)
        4. Position sizing adjustments based on MoS
        5. Net-net candidates and distressed opportunities
        6. Comparative analysis across peer groups
    """

    def __init__(self):
        self._top_down = TopDownAnalyzer()
        self._bottom_up = BottomUpAnalyzer()
        self._comparative = ComparativeAnalyzer()
        logger.info("SecurityAnalysisEngine initialized (Graham-Dodd-Klarman L2/L2.5)")

    def analyze(self, tickers: list, macro_data: dict,
                security_data: dict,
                peer_groups: Optional[dict] = None) -> SecurityAnalysisResult:
        """Run full security analysis pipeline.

        Args:
            tickers: list of ticker symbols to analyze
            macro_data: dict of macro-level data (from MacroEngine)
                - treasury_10y, sp500_pe, cape, hy_spread, etc.
            security_data: dict of ticker → security-level financial data
                - Each entry: {price, eps_history, bvps, revenue, ebit, ...}
            peer_groups: optional dict of ticker → list of peer tickers

        Returns:
            SecurityAnalysisResult with full top-down + bottom-up analysis
        """
        result = SecurityAnalysisResult()
        result.analysis_timestamp = datetime.now().isoformat()

        # ---- Phase 1: Top-Down Assessment ----
        result.top_down = self._top_down.analyze(macro_data)

        # ---- Phase 2: Bottom-Up Analysis ----
        for ticker in tickers:
            data = security_data.get(ticker, {})
            if not data:
                continue

            score = self._bottom_up.analyze_security(
                ticker, data, result.top_down
            )
            result.bottom_up[ticker] = score

            # Categorize
            if score.investment_grade in (InvestmentGrade.STRONG_INVESTMENT,
                                          InvestmentGrade.INVESTMENT):
                result.investment_universe.append(ticker)
            elif score.investment_grade == InvestmentGrade.SPECULATIVE:
                result.speculative_universe.append(ticker)

            if score.security_class == SecurityClass.DISTRESSED:
                result.distressed_opportunities.append(ticker)

            if score.ncav_discount >= 0.33:
                result.net_net_candidates.append(ticker)

        result.tickers_analyzed = len(result.bottom_up)

        # ---- Phase 3: Comparative Analysis ----
        if peer_groups:
            result.comparatives = self._comparative.analyze(
                result.bottom_up, peer_groups
            )

        # ---- Phase 4: Position Sizing via Margin of Safety ----
        result.margin_of_safety_weights = self._compute_mos_weights(result)

        # Aggregate stats
        mos_values = [s.margin_of_safety for s in result.bottom_up.values()
                      if s.margin_of_safety > -1.0]
        result.avg_margin_of_safety = float(np.mean(mos_values)) if mos_values else 0.0
        result.investment_grade_count = len(result.investment_universe)

        logger.info(
            f"SecurityAnalysis: {result.tickers_analyzed} analyzed, "
            f"{result.investment_grade_count} investment-grade, "
            f"{len(result.net_net_candidates)} net-nets, "
            f"{len(result.distressed_opportunities)} distressed, "
            f"avg MoS={result.avg_margin_of_safety*100:.1f}%, "
            f"macro={result.top_down.macro_direction}"
        )

        return result

    def _compute_mos_weights(self, result: SecurityAnalysisResult) -> dict:
        """Compute position size multipliers based on margin of safety.

        Graham's principle: higher MoS → larger position justified.
        Klarman's principle: typically begin selling at 10-20% discount to IV.

        Weight multipliers:
            MoS ≥ 50%  → 1.5x (strong conviction, deep value)
            MoS ≥ 33%  → 1.2x (investment grade)
            MoS ≥ 15%  → 1.0x (borderline, standard sizing)
            MoS ≥ 0%   → 0.7x (near fair value, reduce)
            MoS < 0%   → 0.3x (overvalued, minimal or avoid)

        Top-down adjustment:
            Undervalued market → +10% to all weights
            Overvalued market → -15% to all weights
        """
        weights = {}
        td_adj = 1.0
        if result.top_down.regime in (MarketValuationRegime.DEEPLY_UNDERVALUED,
                                      MarketValuationRegime.UNDERVALUED):
            td_adj = 1.10
        elif result.top_down.regime == MarketValuationRegime.OVERVALUED:
            td_adj = 0.85
        elif result.top_down.regime == MarketValuationRegime.EXTREMELY_OVERVALUED:
            td_adj = 0.70

        for ticker, score in result.bottom_up.items():
            mos = score.margin_of_safety
            if mos >= 0.50:
                w = 1.5
            elif mos >= 0.33:
                w = 1.2
            elif mos >= 0.15:
                w = 1.0
            elif mos >= 0.0:
                w = 0.7
            else:
                w = 0.3

            # Quality adjustment
            if score.composite_score >= 75:
                w *= 1.1
            elif score.composite_score < 35:
                w *= 0.8

            # Top-down adjustment
            w *= td_adj

            weights[ticker] = round(min(w, 2.0), 3)

        return weights

    def get_investment_universe(self, result: SecurityAnalysisResult,
                                min_grade: InvestmentGrade = InvestmentGrade.BORDERLINE
                                ) -> list:
        """Get filtered universe by minimum investment grade.

        Args:
            result: SecurityAnalysisResult
            min_grade: minimum acceptable grade

        Returns:
            List of (ticker, composite_score) tuples, sorted by score desc
        """
        grade_order = {
            InvestmentGrade.STRONG_INVESTMENT: 4,
            InvestmentGrade.INVESTMENT: 3,
            InvestmentGrade.BORDERLINE: 2,
            InvestmentGrade.SPECULATIVE: 1,
            InvestmentGrade.AVOID: 0,
        }
        min_order = grade_order.get(min_grade, 0)

        eligible = []
        for ticker, score in result.bottom_up.items():
            if grade_order.get(score.investment_grade, 0) >= min_order:
                eligible.append((ticker, score.composite_score))

        eligible.sort(key=lambda x: x[1], reverse=True)
        return eligible

    def to_alpha_features(self, result: SecurityAnalysisResult,
                           ticker: str) -> dict:
        """Extract features for AlphaOptimizer walk-forward ML.

        Returns dict of numeric features that can be injected into
        the AlphaOptimizer feature matrix.
        """
        score = result.bottom_up.get(ticker)
        if score is None:
            return {}

        return {
            "graham_mos": score.margin_of_safety,
            "graham_composite": score.composite_score / 100.0,
            "graham_ncav_discount": score.ncav_discount,
            "graham_pe_x_pb": min(score.pe_x_pb, 50.0) / 50.0,
            "graham_earnings_yield": score.earnings_yield,
            "graham_eps_stability": min(score.eps_stability, 2.0) / 2.0,
            "graham_current_ratio": min(score.current_ratio, 5.0) / 5.0,
            "graham_roic_spread": np.clip(score.roic_wacc_spread, -0.2, 0.2) / 0.2,
            "graham_accrual_quality": min(score.accrual_quality, 2.0) / 2.0,
            "graham_int_coverage": min(score.interest_coverage, 20.0) / 20.0,
            "graham_mos_weight": result.margin_of_safety_weights.get(ticker, 1.0),
            "graham_macro_direction": (
                1.0 if result.top_down.macro_direction == "bullish" else
                -1.0 if result.top_down.macro_direction == "bearish" else 0.0
            ),
        }

    def to_execution_signal(self, result: SecurityAnalysisResult,
                             ticker: str) -> dict:
        """Convert analysis to execution-compatible signal dict.

        Compatible with MLVoteEnsemble voting format.
        """
        score = result.bottom_up.get(ticker)
        if score is None:
            return {"vote": 0, "confidence": 0.0}

        # Vote based on investment grade
        grade_votes = {
            InvestmentGrade.STRONG_INVESTMENT: 1,
            InvestmentGrade.INVESTMENT: 1,
            InvestmentGrade.BORDERLINE: 0,
            InvestmentGrade.SPECULATIVE: 0,
            InvestmentGrade.AVOID: -1,
        }
        vote = grade_votes.get(score.investment_grade, 0)

        # Enhance vote with MoS
        if score.margin_of_safety >= 0.50 and score.composite_score >= 70:
            vote = 1  # strong buy signal
        elif score.margin_of_safety < -0.20:
            vote = -1  # sell signal

        confidence = score.composite_score / 100.0

        return {
            "vote": vote,
            "confidence": confidence,
            "margin_of_safety": score.margin_of_safety,
            "investment_grade": score.investment_grade.value,
            "composite_score": score.composite_score,
            "intrinsic_value": score.intrinsic_value,
            "security_class": score.security_class.value,
        }

    def run_skill_statement_analysis(self, ticker: str, financials: dict) -> dict:
        """Hook into agent_skills for enhanced financial statement analysis.

        Supplements Graham-Dodd quantitative checks with skill-based analysis
        when the agent_skills module is available.
        """
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            return test_skill(
                "analyzing-financial-statements",
                {"ticker": ticker, "financials": financials},
            )
        except Exception as e:
            logger.debug("Skill statement analysis failed for %s: %s", ticker, e)
            return {}

