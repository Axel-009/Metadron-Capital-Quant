"""GICS Sector Agents -- 11 specialized scoring agents with 8 dimensions each.

Each GICSSectorAgent scores securities across 8 orthogonal dimensions:
    1. Momentum      (price momentum, acceleration)
    2. Value          (P/E, P/B, EV/EBITDA proxy, FCF yield)
    3. Quality        (ROE, debt/equity, earnings stability, margin stability)
    4. Growth         (revenue growth, earnings growth, forward estimates)
    5. Technical      (RSI, MACD, Bollinger, volume trend)
    6. Earnings       (surprise proxy, revision proxy, beat rate)
    7. Sector-Relative (performance vs sector ETF, relative strength)
    8. Risk-Adjusted  (Sharpe, Sortino, Calmar, max drawdown)

Composite score = weighted sum of dimension scores (0-10 each).
Sector-specific weight overrides tune agents for their sector characteristics.

Different from sector_bots.py which uses tier-based signal generation.
This module provides granular, dimension-level scoring for deeper analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..data.universe_engine import GICS_SECTORS, SECTOR_ETFS
except ImportError:
    GICS_SECTORS = {
        10: "Energy", 15: "Materials", 20: "Industrials",
        25: "Consumer Discretionary", 30: "Consumer Staples",
        35: "Health Care", 40: "Financials",
        45: "Information Technology", 50: "Communication Services",
        55: "Utilities", 60: "Real Estate",
    }
    SECTOR_ETFS = {
        "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
        "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
        "Health Care": "XLV", "Financials": "XLF",
        "Information Technology": "XLK", "Communication Services": "XLC",
        "Utilities": "XLU", "Real Estate": "XLRE",
    }

# ═══════════════════════════════════════════════════════════════════════════
# Sector ETF map (canonical ordering)
# ═══════════════════════════════════════════════════════════════════════════
SECTOR_ETF_MAP = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

# Default dimension weights (sum = 1.0)
DEFAULT_WEIGHTS = {
    "momentum": 0.15,
    "value": 0.15,
    "quality": 0.15,
    "growth": 0.12,
    "technical": 0.12,
    "earnings": 0.10,
    "sector_relative": 0.11,
    "risk_adjusted": 0.10,
}

# Sector-specific weight overrides
SECTOR_WEIGHT_OVERRIDES = {
    "Energy": {
        "momentum": 0.20, "value": 0.20, "quality": 0.12,
        "growth": 0.08, "technical": 0.12, "earnings": 0.08,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Materials": {
        "momentum": 0.18, "value": 0.18, "quality": 0.12,
        "growth": 0.10, "technical": 0.12, "earnings": 0.08,
        "sector_relative": 0.12, "risk_adjusted": 0.10,
    },
    "Industrials": {
        "momentum": 0.15, "value": 0.15, "quality": 0.15,
        "growth": 0.13, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Consumer Discretionary": {
        "momentum": 0.18, "value": 0.12, "quality": 0.13,
        "growth": 0.15, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Consumer Staples": {
        "momentum": 0.10, "value": 0.18, "quality": 0.18,
        "growth": 0.08, "technical": 0.10, "earnings": 0.10,
        "sector_relative": 0.11, "risk_adjusted": 0.15,
    },
    "Health Care": {
        "momentum": 0.13, "value": 0.12, "quality": 0.18,
        "growth": 0.15, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Financials": {
        "momentum": 0.14, "value": 0.18, "quality": 0.16,
        "growth": 0.10, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Information Technology": {
        "momentum": 0.18, "value": 0.10, "quality": 0.12,
        "growth": 0.20, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.08,
    },
    "Communication Services": {
        "momentum": 0.16, "value": 0.12, "quality": 0.13,
        "growth": 0.17, "technical": 0.12, "earnings": 0.10,
        "sector_relative": 0.10, "risk_adjusted": 0.10,
    },
    "Utilities": {
        "momentum": 0.10, "value": 0.20, "quality": 0.15,
        "growth": 0.08, "technical": 0.10, "earnings": 0.10,
        "sector_relative": 0.12, "risk_adjusted": 0.15,
    },
    "Real Estate": {
        "momentum": 0.12, "value": 0.18, "quality": 0.14,
        "growth": 0.10, "technical": 0.10, "earnings": 0.10,
        "sector_relative": 0.12, "risk_adjusted": 0.14,
    },
}

# Action thresholds per sector (composite score boundaries)
SECTOR_THRESHOLDS = {
    "Energy":                   {"buy": 6.5, "sell": 3.5},
    "Materials":                {"buy": 6.5, "sell": 3.5},
    "Industrials":              {"buy": 6.2, "sell": 3.8},
    "Consumer Discretionary":   {"buy": 6.3, "sell": 3.5},
    "Consumer Staples":         {"buy": 6.0, "sell": 4.0},
    "Health Care":              {"buy": 6.2, "sell": 3.6},
    "Financials":               {"buy": 6.3, "sell": 3.5},
    "Information Technology":   {"buy": 6.5, "sell": 3.3},
    "Communication Services":   {"buy": 6.3, "sell": 3.5},
    "Utilities":                {"buy": 5.8, "sell": 4.0},
    "Real Estate":              {"buy": 6.0, "sell": 3.8},
}

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False



# ═══════════════════════════════════════════════════════════════════════════
# SectorScoringDimension
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SectorScoringDimension:
    """A single scoring dimension for a security within a GICS sector agent.

    Attributes:
        name:    Dimension name (e.g. 'momentum', 'value').
        score:   Score on a 0-10 scale.
        weight:  Weight in composite calculation (0.0-1.0).
        details: Free-form dict with sub-metric breakdowns.
    """
    name: str = ""
    score: float = 5.0
    weight: float = 0.10
    details: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# GICSSectorAgent — one per GICS sector
# ═══════════════════════════════════════════════════════════════════════════
class GICSSectorAgent:
    """Specialized scoring agent for a single GICS sector.

    Evaluates securities on 8 orthogonal dimensions, producing a composite
    score and a BUY / SELL / HOLD recommendation.

    Parameters
    ----------
    sector_name : str
        GICS sector name (e.g. 'Energy').
    sector_etf : str
        Sector ETF ticker (e.g. 'XLE').
    """

    DIMENSION_NAMES = [
        "momentum", "value", "quality", "growth",
        "technical", "earnings", "sector_relative", "risk_adjusted",
    ]

    def __init__(self, sector_name: str, sector_etf: str):
        self.sector_name = sector_name
        self.sector_etf = sector_etf
        self.weights = dict(
            SECTOR_WEIGHT_OVERRIDES.get(sector_name, DEFAULT_WEIGHTS)
        )
        self.thresholds = SECTOR_THRESHOLDS.get(
            sector_name, {"buy": 6.2, "sell": 3.8}
        )
        self._last_results: list[dict] = []

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def score_security(
        self,
        ticker: str,
        returns: pd.Series,
        fundamentals: dict,
    ) -> dict:
        """Score a single security on all 8 dimensions.

        Parameters
        ----------
        ticker : str
            Security ticker symbol.
        returns : pd.Series
            Daily return series (datetime-indexed).
        fundamentals : dict
            Fundamental data dict with keys like 'trailingPE', 'priceToBook',
            'returnOnEquity', 'debtToEquity', 'revenueGrowth', etc.

        Returns
        -------
        dict with keys:
            ticker, sector, dimensions (list of SectorScoringDimension),
            composite_score (0-10), action ('BUY'/'SELL'/'HOLD'),
            dimension_scores (dict name->score).
        """
        if returns is None or len(returns) < 5:
            return self._empty_result(ticker)

        returns = returns.dropna()
        if len(returns) < 5:
            return self._empty_result(ticker)

        fundamentals = fundamentals or {}

        # Compute each dimension
        dims = []
        dims.append(self._score_momentum(returns))
        dims.append(self._score_value(fundamentals))
        dims.append(self._score_quality(returns, fundamentals))
        dims.append(self._score_growth(fundamentals))
        dims.append(self._score_technical(returns))
        dims.append(self._score_earnings(fundamentals))
        dims.append(self._score_sector_relative(returns))
        dims.append(self._score_risk_adjusted(returns))

        # Composite = weighted sum
        composite = 0.0
        dim_scores = {}
        for d in dims:
            w = self.weights.get(d.name, 0.10)
            d.weight = w
            composite += d.score * w
            dim_scores[d.name] = round(d.score, 3)

        composite = float(np.clip(composite, 0.0, 10.0))

        # Determine action
        if composite >= self.thresholds["buy"]:
            action = "BUY"
        elif composite <= self.thresholds["sell"]:
            action = "SELL"
        else:
            action = "HOLD"

        result = {
            "ticker": ticker,
            "sector": self.sector_name,
            "dimensions": dims,
            "composite_score": round(composite, 4),
            "action": action,
            "dimension_scores": dim_scores,
        }
        return result

    def run_skill_sector_analysis(self, ticker: str, fundamentals: dict) -> dict:
        """Run sector-specific financial analysis skill if available."""
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            return test_skill(
                "analyzing-financial-statements",
                {"ticker": ticker, "sector": self.sector_name, "fundamentals": fundamentals},
            )
        except Exception:
            return {}

    def rank_sector(
        self,
        securities: list,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Score and rank all securities within this sector.

        Parameters
        ----------
        securities : list
            List of ticker strings or objects with .ticker attribute.
        returns_df : pd.DataFrame
            DataFrame where columns are tickers and rows are dates.

        Returns
        -------
        pd.DataFrame
            Ranked DataFrame with columns: ticker, composite_score, action,
            plus one column per dimension score.
        """
        results = []
        for sec in securities:
            ticker = sec.ticker if hasattr(sec, "ticker") else str(sec)
            if ticker in returns_df.columns:
                rets = returns_df[ticker].dropna()
            else:
                rets = pd.Series(dtype=float)
            # Pass empty fundamentals; caller can enrich
            res = self.score_security(ticker, rets, {})
            results.append(res)

        self._last_results = results

        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            row = {
                "ticker": r["ticker"],
                "composite_score": r["composite_score"],
                "action": r["action"],
            }
            row.update(r["dimension_scores"])
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df.index.name = "rank"
        return df

    # -------------------------------------------------------------------
    # Dimension 1: Momentum (weight 0.15 default)
    # -------------------------------------------------------------------
    def _score_momentum(self, returns: pd.Series) -> SectorScoringDimension:
        """1M, 3M, 12M price momentum + momentum acceleration."""
        n = len(returns)

        # 1-month momentum (~21 trading days)
        mom_1m = float(returns.iloc[-21:].sum()) if n >= 21 else float(returns.sum())
        # 3-month momentum (~63 trading days)
        mom_3m = float(returns.iloc[-63:].sum()) if n >= 63 else mom_1m
        # 12-month momentum (~252 trading days)
        mom_12m = float(returns.iloc[-252:].sum()) if n >= 252 else mom_3m

        # Momentum acceleration: 1M annualised vs 3M annualised
        ann_1m = mom_1m * 12.0
        ann_3m = mom_3m * 4.0
        acceleration = ann_1m - ann_3m

        # Map to 0-10 scale
        # Blend: 30% 1M + 30% 3M + 25% 12M + 15% acceleration
        raw = (
            0.30 * self._momentum_to_score(mom_1m)
            + 0.30 * self._momentum_to_score(mom_3m)
            + 0.25 * self._momentum_to_score(mom_12m)
            + 0.15 * self._momentum_to_score(acceleration / 12.0)
        )
        score = float(np.clip(raw, 0.0, 10.0))

        return SectorScoringDimension(
            name="momentum",
            score=score,
            weight=self.weights.get("momentum", 0.15),
            details={
                "mom_1m": round(mom_1m, 5),
                "mom_3m": round(mom_3m, 5),
                "mom_12m": round(mom_12m, 5),
                "acceleration": round(acceleration, 5),
            },
        )

    @staticmethod
    def _momentum_to_score(mom: float) -> float:
        """Convert a momentum return to a 0-10 score.

        -0.30 or worse -> 0, 0.0 -> 5, +0.30 or better -> 10.
        """
        return float(np.clip(5.0 + mom * (50.0 / 3.0), 0.0, 10.0))

    # -------------------------------------------------------------------
    # Dimension 2: Value (weight 0.15 default)
    # -------------------------------------------------------------------
    def _score_value(self, fundamentals: dict) -> SectorScoringDimension:
        """P/E, P/B, EV/EBITDA proxy, FCF yield scoring."""
        pe = fundamentals.get("trailingPE") or fundamentals.get("forwardPE")
        pb = fundamentals.get("priceToBook")
        ev_ebitda = fundamentals.get("enterpriseToEbitda")
        fcf_yield = fundamentals.get("freeCashflowYield")

        sub_scores = []
        details = {}

        # P/E score: lower is better (for value), negatives penalised
        if pe is not None and isinstance(pe, (int, float)):
            details["pe"] = round(pe, 2)
            if pe < 0:
                sub_scores.append(2.0)
            elif pe < 10:
                sub_scores.append(9.0)
            elif pe < 15:
                sub_scores.append(7.5)
            elif pe < 20:
                sub_scores.append(6.0)
            elif pe < 30:
                sub_scores.append(4.5)
            elif pe < 50:
                sub_scores.append(3.0)
            else:
                sub_scores.append(1.5)
        else:
            sub_scores.append(5.0)

        # P/B score: lower is better
        if pb is not None and isinstance(pb, (int, float)):
            details["pb"] = round(pb, 2)
            if pb < 0:
                sub_scores.append(2.0)
            elif pb < 1.0:
                sub_scores.append(9.0)
            elif pb < 2.0:
                sub_scores.append(7.0)
            elif pb < 4.0:
                sub_scores.append(5.5)
            elif pb < 8.0:
                sub_scores.append(3.5)
            else:
                sub_scores.append(2.0)
        else:
            sub_scores.append(5.0)

        # EV/EBITDA proxy: lower is better
        if ev_ebitda is not None and isinstance(ev_ebitda, (int, float)):
            details["ev_ebitda"] = round(ev_ebitda, 2)
            if ev_ebitda < 0:
                sub_scores.append(2.0)
            elif ev_ebitda < 8:
                sub_scores.append(8.5)
            elif ev_ebitda < 12:
                sub_scores.append(7.0)
            elif ev_ebitda < 18:
                sub_scores.append(5.0)
            elif ev_ebitda < 25:
                sub_scores.append(3.5)
            else:
                sub_scores.append(2.0)
        else:
            sub_scores.append(5.0)

        # FCF yield: higher is better
        if fcf_yield is not None and isinstance(fcf_yield, (int, float)):
            details["fcf_yield"] = round(fcf_yield, 4)
            fcf_score = float(np.clip(5.0 + fcf_yield * 50.0, 0.0, 10.0))
            sub_scores.append(fcf_score)
        else:
            sub_scores.append(5.0)

        score = float(np.mean(sub_scores)) if sub_scores else 5.0
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="value",
            score=score,
            weight=self.weights.get("value", 0.15),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 3: Quality (weight 0.15 default)
    # -------------------------------------------------------------------
    def _score_quality(
        self, returns: pd.Series, fundamentals: dict
    ) -> SectorScoringDimension:
        """ROE, debt/equity, earnings stability, margin stability."""
        sub_scores = []
        details = {}

        # ROE: higher is better
        roe = fundamentals.get("returnOnEquity")
        if roe is not None and isinstance(roe, (int, float)):
            details["roe"] = round(roe, 4)
            if roe < 0:
                sub_scores.append(1.5)
            elif roe < 0.05:
                sub_scores.append(3.5)
            elif roe < 0.10:
                sub_scores.append(5.0)
            elif roe < 0.15:
                sub_scores.append(6.5)
            elif roe < 0.25:
                sub_scores.append(8.0)
            else:
                sub_scores.append(9.0)
        else:
            sub_scores.append(5.0)

        # Debt/Equity: lower is better
        de = fundamentals.get("debtToEquity")
        if de is not None and isinstance(de, (int, float)):
            details["debt_to_equity"] = round(de, 2)
            if de < 0:
                sub_scores.append(4.0)  # negative equity
            elif de < 30:
                sub_scores.append(9.0)
            elif de < 60:
                sub_scores.append(7.5)
            elif de < 100:
                sub_scores.append(6.0)
            elif de < 200:
                sub_scores.append(4.0)
            else:
                sub_scores.append(2.0)
        else:
            sub_scores.append(5.0)

        # Earnings stability: stddev of rolling 63-day returns as proxy
        if len(returns) >= 126:
            rolling_vol = returns.rolling(63).std().dropna()
            if len(rolling_vol) >= 2:
                vol_of_vol = float(rolling_vol.std())
                details["vol_of_vol"] = round(vol_of_vol, 6)
                # Lower vol-of-vol = more stable earnings proxy
                stability = float(np.clip(8.0 - vol_of_vol * 500.0, 1.0, 9.5))
                sub_scores.append(stability)
            else:
                sub_scores.append(5.0)
        else:
            sub_scores.append(5.0)

        # Margin stability: profit margin level as proxy
        margin = fundamentals.get("profitMargins")
        if margin is not None and isinstance(margin, (int, float)):
            details["profit_margin"] = round(margin, 4)
            if margin < 0:
                sub_scores.append(2.0)
            elif margin < 0.05:
                sub_scores.append(4.0)
            elif margin < 0.10:
                sub_scores.append(5.5)
            elif margin < 0.20:
                sub_scores.append(7.0)
            elif margin < 0.35:
                sub_scores.append(8.5)
            else:
                sub_scores.append(9.0)
        else:
            sub_scores.append(5.0)

        score = float(np.mean(sub_scores)) if sub_scores else 5.0
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="quality",
            score=score,
            weight=self.weights.get("quality", 0.15),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 4: Growth (weight 0.12 default)
    # -------------------------------------------------------------------
    def _score_growth(self, fundamentals: dict) -> SectorScoringDimension:
        """Revenue growth, earnings growth, forward estimates proxy."""
        sub_scores = []
        details = {}

        # Revenue growth
        rev_growth = fundamentals.get("revenueGrowth")
        if rev_growth is not None and isinstance(rev_growth, (int, float)):
            details["revenue_growth"] = round(rev_growth, 4)
            g_score = float(np.clip(5.0 + rev_growth * 20.0, 0.0, 10.0))
            sub_scores.append(g_score)
        else:
            sub_scores.append(5.0)

        # Earnings growth
        earn_growth = fundamentals.get("earningsGrowth") or fundamentals.get(
            "earningsQuarterlyGrowth"
        )
        if earn_growth is not None and isinstance(earn_growth, (int, float)):
            details["earnings_growth"] = round(earn_growth, 4)
            e_score = float(np.clip(5.0 + earn_growth * 15.0, 0.0, 10.0))
            sub_scores.append(e_score)
        else:
            sub_scores.append(5.0)

        # Forward estimates proxy: forward PE vs trailing PE
        fwd_pe = fundamentals.get("forwardPE")
        trail_pe = fundamentals.get("trailingPE")
        if (
            fwd_pe is not None
            and trail_pe is not None
            and isinstance(fwd_pe, (int, float))
            and isinstance(trail_pe, (int, float))
            and trail_pe > 0
            and fwd_pe > 0
        ):
            # If forward PE < trailing PE, earnings expected to grow
            peg_ratio = fwd_pe / trail_pe
            details["fwd_trail_pe_ratio"] = round(peg_ratio, 3)
            if peg_ratio < 0.70:
                sub_scores.append(9.0)
            elif peg_ratio < 0.85:
                sub_scores.append(7.5)
            elif peg_ratio < 1.00:
                sub_scores.append(6.0)
            elif peg_ratio < 1.15:
                sub_scores.append(4.5)
            else:
                sub_scores.append(3.0)
        else:
            sub_scores.append(5.0)

        score = float(np.mean(sub_scores)) if sub_scores else 5.0
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="growth",
            score=score,
            weight=self.weights.get("growth", 0.12),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 5: Technical (weight 0.12 default)
    # -------------------------------------------------------------------
    def _score_technical(self, returns: pd.Series) -> SectorScoringDimension:
        """RSI, MACD signal, Bollinger %B, volume trend proxy."""
        prices = (1.0 + returns).cumprod()
        arr = prices.values.astype(float)
        details = {}

        # --- RSI (14-period) ---
        rsi_val = self._compute_rsi(arr, period=14)
        details["rsi"] = round(rsi_val, 2)
        # Oversold (<30) -> bullish (high score), overbought (>70) -> bearish
        if rsi_val < 30:
            rsi_score = 8.0 + (30.0 - rsi_val) / 15.0
        elif rsi_val < 45:
            rsi_score = 6.5
        elif rsi_val < 55:
            rsi_score = 5.0
        elif rsi_val < 70:
            rsi_score = 3.5
        else:
            rsi_score = 2.0 - (rsi_val - 70.0) / 30.0
        rsi_score = float(np.clip(rsi_score, 0.0, 10.0))

        # --- MACD signal ---
        macd_line, signal_line, histogram = self._compute_macd(arr)
        details["macd_histogram"] = round(histogram, 6)
        if histogram > 0 and macd_line > signal_line:
            macd_score = float(np.clip(5.0 + abs(histogram) * 1000.0, 5.0, 9.0))
        elif histogram < 0 and macd_line < signal_line:
            macd_score = float(np.clip(5.0 - abs(histogram) * 1000.0, 1.0, 5.0))
        else:
            macd_score = 5.0

        # --- Bollinger %B (20-period, 2 std) ---
        pct_b = self._compute_bollinger_pctb(arr, period=20)
        details["bollinger_pct_b"] = round(pct_b, 4)
        # %B near 0 -> oversold (bullish), near 1 -> overbought (bearish)
        bb_score = float(np.clip(8.0 - pct_b * 6.0, 1.0, 9.5))

        # --- Volume trend proxy (using return magnitude trend) ---
        if len(returns) >= 42:
            recent_vol = float(np.mean(np.abs(returns.iloc[-21:])))
            prior_vol = float(np.mean(np.abs(returns.iloc[-42:-21])))
            if prior_vol > 0:
                vol_trend = recent_vol / prior_vol
                details["vol_trend_ratio"] = round(vol_trend, 3)
                # Rising volume with positive momentum = bullish
                recent_mom = float(returns.iloc[-21:].sum())
                if recent_mom > 0 and vol_trend > 1.0:
                    vt_score = float(np.clip(5.0 + vol_trend * 2.0, 5.0, 9.0))
                elif recent_mom < 0 and vol_trend > 1.0:
                    vt_score = float(np.clip(5.0 - vol_trend * 2.0, 1.0, 5.0))
                else:
                    vt_score = 5.0
            else:
                vt_score = 5.0
        else:
            vt_score = 5.0

        # Blend: RSI 25%, MACD 30%, BB 25%, Volume trend 20%
        score = 0.25 * rsi_score + 0.30 * macd_score + 0.25 * bb_score + 0.20 * vt_score
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="technical",
            score=score,
            weight=self.weights.get("technical", 0.12),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 6: Earnings (weight 0.10 default)
    # -------------------------------------------------------------------
    def _score_earnings(self, fundamentals: dict) -> SectorScoringDimension:
        """Earnings surprise proxy, revision proxy, beat rate estimate."""
        sub_scores = []
        details = {}

        # Earnings surprise proxy
        surprise = fundamentals.get("earningsSurprise") or fundamentals.get(
            "earningsQuarterlyGrowth"
        )
        if surprise is not None and isinstance(surprise, (int, float)):
            details["earnings_surprise"] = round(surprise, 4)
            s_score = float(np.clip(5.0 + surprise * 25.0, 0.0, 10.0))
            sub_scores.append(s_score)
        else:
            sub_scores.append(5.0)

        # Revision proxy: forward PE improvement
        fwd_pe = fundamentals.get("forwardPE")
        trail_pe = fundamentals.get("trailingPE")
        if (
            fwd_pe is not None
            and trail_pe is not None
            and isinstance(fwd_pe, (int, float))
            and isinstance(trail_pe, (int, float))
            and trail_pe > 0
        ):
            revision = (trail_pe - fwd_pe) / trail_pe
            details["revision_proxy"] = round(revision, 4)
            r_score = float(np.clip(5.0 + revision * 20.0, 0.0, 10.0))
            sub_scores.append(r_score)
        else:
            sub_scores.append(5.0)

        # Beat rate proxy: positive quarterly growth suggests beating expectations
        qtr_growth = fundamentals.get("earningsQuarterlyGrowth")
        if qtr_growth is not None and isinstance(qtr_growth, (int, float)):
            details["quarterly_growth"] = round(qtr_growth, 4)
            if qtr_growth > 0.10:
                sub_scores.append(8.0)
            elif qtr_growth > 0.0:
                sub_scores.append(6.5)
            elif qtr_growth > -0.10:
                sub_scores.append(4.0)
            else:
                sub_scores.append(2.5)
        else:
            sub_scores.append(5.0)

        score = float(np.mean(sub_scores)) if sub_scores else 5.0
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="earnings",
            score=score,
            weight=self.weights.get("earnings", 0.10),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 7: Sector-Relative (weight 0.11 default)
    # -------------------------------------------------------------------
    def _score_sector_relative(
        self, returns: pd.Series
    ) -> SectorScoringDimension:
        """Performance vs sector ETF proxy, relative strength.

        Without live ETF data, uses return characteristics as a proxy.
        If sector ETF returns are available in the series index alignment,
        this computes true relative performance.
        """
        details = {}
        n = len(returns)

        # Proxy: compare security stats to neutral benchmarks
        # 1-month relative performance proxy
        mom_1m = float(returns.iloc[-21:].sum()) if n >= 21 else float(returns.sum())
        # Assume sector avg ~0.8% monthly -> 0.008
        sector_avg_1m = 0.008
        rel_perf_1m = mom_1m - sector_avg_1m
        details["rel_perf_1m"] = round(rel_perf_1m, 5)

        # 3-month relative performance proxy
        mom_3m = float(returns.iloc[-63:].sum()) if n >= 63 else mom_1m
        sector_avg_3m = 0.024
        rel_perf_3m = mom_3m - sector_avg_3m
        details["rel_perf_3m"] = round(rel_perf_3m, 5)

        # Relative strength score
        rs_score_1m = float(np.clip(5.0 + rel_perf_1m * 40.0, 0.0, 10.0))
        rs_score_3m = float(np.clip(5.0 + rel_perf_3m * 15.0, 0.0, 10.0))

        # Relative strength trend (improving or deteriorating)
        if n >= 63:
            rs_recent = float(returns.iloc[-21:].mean())
            rs_prior = float(returns.iloc[-63:-21].mean())
            if rs_prior != 0:
                rs_trend = rs_recent / abs(rs_prior) if rs_prior != 0 else 1.0
            else:
                rs_trend = 1.0
            details["rs_trend"] = round(rs_trend, 4)
            trend_score = float(np.clip(5.0 + (rs_trend - 1.0) * 10.0, 0.0, 10.0))
        else:
            trend_score = 5.0

        # Blend: 40% 1M relative, 35% 3M relative, 25% trend
        score = 0.40 * rs_score_1m + 0.35 * rs_score_3m + 0.25 * trend_score
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="sector_relative",
            score=score,
            weight=self.weights.get("sector_relative", 0.11),
            details=details,
        )

    # -------------------------------------------------------------------
    # Dimension 8: Risk-Adjusted (weight 0.10 default)
    # -------------------------------------------------------------------
    def _score_risk_adjusted(self, returns: pd.Series) -> SectorScoringDimension:
        """Sharpe, Sortino, Calmar, max drawdown scoring."""
        details = {}
        n = len(returns)
        ann_factor = 252.0

        # --- Sharpe ratio ---
        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        sharpe = (mean_ret * ann_factor) / (std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0.0
        details["sharpe"] = round(sharpe, 4)

        # --- Sortino ratio ---
        downside = returns[returns < 0]
        down_std = float(downside.std()) if len(downside) > 1 else std_ret
        sortino = (mean_ret * ann_factor) / (down_std * np.sqrt(ann_factor)) if down_std > 0 else 0.0
        details["sortino"] = round(sortino, 4)

        # --- Max drawdown ---
        cum = (1.0 + returns).cumprod()
        running_max = cum.cummax()
        drawdowns = (cum - running_max) / running_max
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        details["max_drawdown"] = round(max_dd, 5)

        # --- Calmar ratio ---
        ann_ret = mean_ret * ann_factor
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0
        details["calmar"] = round(calmar, 4)

        # Score each sub-metric on 0-10
        # Sharpe: <-0.5 -> 1, 0 -> 4, 1.0 -> 6, 2.0 -> 8, 3.0+ -> 10
        sharpe_score = float(np.clip(4.0 + sharpe * 2.0, 0.0, 10.0))

        # Sortino: similar but slightly more generous
        sortino_score = float(np.clip(4.0 + sortino * 1.5, 0.0, 10.0))

        # Calmar: 0 -> 4, 1 -> 6, 3+ -> 10
        calmar_score = float(np.clip(4.0 + calmar * 2.0, 0.0, 10.0))

        # Max drawdown: 0% -> 10, -10% -> 7, -25% -> 4, -50%+ -> 1
        dd_score = float(np.clip(10.0 + max_dd * 18.0, 0.0, 10.0))

        # Blend: Sharpe 30%, Sortino 25%, Calmar 20%, Max DD 25%
        score = (
            0.30 * sharpe_score
            + 0.25 * sortino_score
            + 0.20 * calmar_score
            + 0.25 * dd_score
        )
        score = float(np.clip(score, 0.0, 10.0))

        return SectorScoringDimension(
            name="risk_adjusted",
            score=score,
            weight=self.weights.get("risk_adjusted", 0.10),
            details=details,
        )

    # -------------------------------------------------------------------
    # Technical indicator helpers (pure numpy)
    # -------------------------------------------------------------------
    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI using Wilder smoothing."""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2.0 / (period + 1)
        out = np.empty_like(data, dtype=float)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _compute_macd(
        prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """Compute MACD line, signal line, histogram."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        alpha_f = 2.0 / (fast + 1)
        alpha_s = 2.0 / (slow + 1)
        alpha_sig = 2.0 / (signal + 1)
        # EMA fast
        ema_f = np.empty(len(prices), dtype=float)
        ema_f[0] = prices[0]
        for i in range(1, len(prices)):
            ema_f[i] = alpha_f * prices[i] + (1.0 - alpha_f) * ema_f[i - 1]
        # EMA slow
        ema_s = np.empty(len(prices), dtype=float)
        ema_s[0] = prices[0]
        for i in range(1, len(prices)):
            ema_s[i] = alpha_s * prices[i] + (1.0 - alpha_s) * ema_s[i - 1]
        macd_line = ema_f - ema_s
        # Signal line
        sig_line = np.empty(len(macd_line), dtype=float)
        sig_line[0] = macd_line[0]
        for i in range(1, len(macd_line)):
            sig_line[i] = alpha_sig * macd_line[i] + (1.0 - alpha_sig) * sig_line[i - 1]
        hist = macd_line - sig_line
        return float(macd_line[-1]), float(sig_line[-1]), float(hist[-1])

    @staticmethod
    def _compute_bollinger_pctb(
        prices: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> float:
        """Compute Bollinger %B."""
        if len(prices) < period:
            return 0.5
        window = prices[-period:]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if std == 0:
            return 0.5
        upper = mid + num_std * std
        lower = mid - num_std * std
        pct_b = (prices[-1] - lower) / (upper - lower)
        return float(np.clip(pct_b, -0.5, 1.5))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _empty_result(self, ticker: str) -> dict:
        """Return a neutral result when data is insufficient."""
        dims = []
        dim_scores = {}
        for name in self.DIMENSION_NAMES:
            d = SectorScoringDimension(
                name=name, score=5.0, weight=self.weights.get(name, 0.10)
            )
            dims.append(d)
            dim_scores[name] = 5.0
        return {
            "ticker": ticker,
            "sector": self.sector_name,
            "dimensions": dims,
            "composite_score": 5.0,
            "action": "HOLD",
            "dimension_scores": dim_scores,
        }


# ═══════════════════════════════════════════════════════════════════════════
# GICSSectorAgentManager — orchestrates all 11 sector agents
# ═══════════════════════════════════════════════════════════════════════════
class GICSSectorAgentManager:
    """Manages 11 GICS sector agents and aggregates their results.

    Creates one GICSSectorAgent per GICS sector, runs scoring across the
    full equity universe, and provides ranking / reporting utilities.
    """

    def __init__(self):
        self.agents: dict[str, GICSSectorAgent] = {}
        self._results: dict[str, pd.DataFrame] = {}
        self._security_results: dict[str, list[dict]] = {}
        self._init_agents()

    def _init_agents(self):
        """Create one agent per GICS sector."""
        for sector_name, etf in SECTOR_ETF_MAP.items():
            self.agents[sector_name] = GICSSectorAgent(
                sector_name=sector_name, sector_etf=etf
            )

    def run_all_sectors(
        self,
        universe: dict,
        returns_df: pd.DataFrame,
    ) -> dict:
        """Run all 11 sector agents across the universe.

        Parameters
        ----------
        universe : dict
            Mapping of sector_name -> list of tickers (or Security objects).
        returns_df : pd.DataFrame
            DataFrame with tickers as columns and dates as index.

        Returns
        -------
        dict
            Mapping of sector_name -> pd.DataFrame of ranked results.
        """
        self._results = {}
        self._security_results = {}

        for sector_name, agent in self.agents.items():
            securities = universe.get(sector_name, [])
            if not securities:
                self._results[sector_name] = pd.DataFrame()
                self._security_results[sector_name] = []
                continue

            ranked = agent.rank_sector(securities, returns_df)
            self._results[sector_name] = ranked
            self._security_results[sector_name] = agent._last_results

        return dict(self._results)

    def get_top_picks(self, n_per_sector: int = 3) -> list:
        """Get the top N picks from each sector.

        Returns
        -------
        list of dict
            Each dict has: ticker, sector, composite_score, action.
            Sorted by composite_score descending across all sectors.
        """
        picks = []
        for sector_name, df in self._results.items():
            if df is None or df.empty:
                continue
            buy_df = df[df["action"] == "BUY"].head(n_per_sector)
            if buy_df.empty:
                # Fall back to top N regardless of action
                buy_df = df.head(n_per_sector)
            for _, row in buy_df.iterrows():
                picks.append({
                    "ticker": row["ticker"],
                    "sector": sector_name,
                    "composite_score": row["composite_score"],
                    "action": row["action"],
                })

        picks.sort(key=lambda x: x["composite_score"], reverse=True)
        return picks

    def get_sector_rankings(self) -> dict:
        """Rank sectors by average composite score.

        Returns
        -------
        dict
            Mapping of sector_name -> {avg_score, n_securities, n_buys,
            n_sells, n_holds, top_ticker}.
        """
        rankings = {}
        for sector_name, df in self._results.items():
            if df is None or df.empty:
                rankings[sector_name] = {
                    "avg_score": 0.0,
                    "n_securities": 0,
                    "n_buys": 0,
                    "n_sells": 0,
                    "n_holds": 0,
                    "top_ticker": None,
                }
                continue

            avg = float(df["composite_score"].mean())
            n_buys = int((df["action"] == "BUY").sum())
            n_sells = int((df["action"] == "SELL").sum())
            n_holds = int((df["action"] == "HOLD").sum())
            top_ticker = str(df.iloc[0]["ticker"]) if len(df) > 0 else None

            rankings[sector_name] = {
                "avg_score": round(avg, 4),
                "n_securities": len(df),
                "n_buys": n_buys,
                "n_sells": n_sells,
                "n_holds": n_holds,
                "top_ticker": top_ticker,
            }

        # Sort by avg_score descending
        rankings = dict(
            sorted(rankings.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        )
        return rankings

    def format_report(self) -> str:
        """Generate an ASCII report of all sector agent results.

        Returns
        -------
        str
            Multi-section ASCII report with sector rankings, top picks,
            and dimension breakdowns.
        """
        lines = []
        sep = "=" * 105
        thin = "-" * 105

        lines.append(sep)
        lines.append("  GICS SECTOR AGENT REPORT -- 8-Dimension Scoring".center(105))
        lines.append(sep)
        lines.append("")

        # Section 1: Sector Rankings
        lines.append("  SECTOR RANKINGS (by average composite score)")
        lines.append(thin)
        lines.append(
            f"  {'Rank':<5} {'Sector':<28} {'Avg Score':>10} "
            f"{'#Sec':>5} {'BUY':>5} {'SELL':>5} {'HOLD':>5} {'Top Pick':<10}"
        )
        lines.append(thin)

        rankings = self.get_sector_rankings()
        for rank, (sector, data) in enumerate(rankings.items(), 1):
            top = data["top_ticker"] or "---"
            lines.append(
                f"  {rank:<5} {sector:<28} {data['avg_score']:>10.4f} "
                f"{data['n_securities']:>5} {data['n_buys']:>5} "
                f"{data['n_sells']:>5} {data['n_holds']:>5} {top:<10}"
            )
        lines.append(thin)
        lines.append("")

        # Section 2: Top Picks
        top_picks = self.get_top_picks(n_per_sector=3)
        lines.append("  TOP PICKS (best 3 per sector, sorted by composite)")
        lines.append(thin)
        lines.append(
            f"  {'#':<4} {'Ticker':<10} {'Sector':<28} "
            f"{'Score':>8} {'Action':<6}"
        )
        lines.append(thin)
        for i, pick in enumerate(top_picks[:30], 1):
            lines.append(
                f"  {i:<4} {pick['ticker']:<10} {pick['sector']:<28} "
                f"{pick['composite_score']:>8.4f} {pick['action']:<6}"
            )
        lines.append(thin)
        lines.append("")

        # Section 3: Per-sector dimension breakdown (top 3 per sector)
        lines.append("  DIMENSION BREAKDOWN (top 3 per sector)")
        lines.append(thin)
        dim_header = "  {:<8} {:<24}".format("Ticker", "Sector")
        for d in GICSSectorAgent.DIMENSION_NAMES:
            abbr = d[:4].upper()
            dim_header += f" {abbr:>6}"
        dim_header += " {:>8}".format("COMP")
        lines.append(dim_header)
        lines.append(thin)

        for sector_name in SECTOR_ETF_MAP:
            df = self._results.get(sector_name)
            if df is None or df.empty:
                continue
            top_df = df.head(3)
            for _, row in top_df.iterrows():
                row_str = f"  {row['ticker']:<8} {sector_name:<24}"
                for d in GICSSectorAgent.DIMENSION_NAMES:
                    val = row.get(d, 5.0)
                    row_str += f" {val:>6.2f}"
                row_str += f" {row['composite_score']:>8.4f}"
                lines.append(row_str)

        lines.append(thin)

        # Summary statistics
        total_sec = sum(d["n_securities"] for d in rankings.values())
        total_buys = sum(d["n_buys"] for d in rankings.values())
        total_sells = sum(d["n_sells"] for d in rankings.values())
        total_holds = sum(d["n_holds"] for d in rankings.values())
        overall_avg = (
            float(np.mean([d["avg_score"] for d in rankings.values() if d["n_securities"] > 0]))
            if any(d["n_securities"] > 0 for d in rankings.values())
            else 0.0
        )

        lines.append("")
        lines.append(
            f"  Total: {total_sec} securities | "
            f"BUY: {total_buys} | SELL: {total_sells} | HOLD: {total_holds} | "
            f"Overall Avg Score: {overall_avg:.4f}"
        )
        lines.append(sep)

        return "\n".join(lines)
