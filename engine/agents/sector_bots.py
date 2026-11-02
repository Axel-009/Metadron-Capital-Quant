"""Sector Micro-Bots — 11 specialized GICS sector agents.

Each bot runs within its sector, learning and improving to maximise alpha.
Bots are scored weekly: 40% accuracy + 30% Sharpe + 30% hit rate.

Tiers:
    TIER_1 Generals    — Sharpe >2.0, accuracy >80%
    TIER_2 Captains    — Sharpe >1.5, accuracy >55%
    TIER_3 Lieutenants — Sharpe >1.0, accuracy >50%
    TIER_4 Recruits    — below thresholds
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from ..data.universe_engine import GICS_SECTORS, SECTOR_ETFS, Security
from ..data.yahoo_data import get_returns, get_adj_close, get_fundamentals
from ..ml.alpha_optimizer import classify_quality

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False



# ---------------------------------------------------------------------------
# Agent tiers
# ---------------------------------------------------------------------------
class AgentTier:
    GENERAL = "TIER_1_General"
    CAPTAIN = "TIER_2_Captain"
    LIEUTENANT = "TIER_3_Lieutenant"
    RECRUIT = "TIER_4_Recruit"


TIER_THRESHOLDS = {
    AgentTier.GENERAL: {"min_sharpe": 2.0, "min_accuracy": 0.80},
    AgentTier.CAPTAIN: {"min_sharpe": 1.5, "min_accuracy": 0.55},
    AgentTier.LIEUTENANT: {"min_sharpe": 1.0, "min_accuracy": 0.50},
    AgentTier.RECRUIT: {"min_sharpe": -999, "min_accuracy": 0.0},
}


@dataclass
class BotScore:
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0
    composite: float = 0.0    # 40% acc + 30% sharpe + 30% hit
    tier: str = AgentTier.RECRUIT
    total_signals: int = 0
    correct_signals: int = 0
    consecutive_top_weeks: int = 0
    consecutive_bottom_weeks: int = 0


@dataclass
class SectorSignal:
    ticker: str
    sector: str
    direction: str = "HOLD"   # BUY / SELL / HOLD
    confidence: float = 0.0
    quality_tier: str = "D"
    alpha_estimate: float = 0.0
    momentum: float = 0.0
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Technical Analyzer — pure-numpy technical indicators
# ---------------------------------------------------------------------------
class TechnicalAnalyzer:
    """Compute common technical indicators from a price series.

    All calculations use numpy only; no TA-Lib dependency required.
    Input: pandas Series of adjusted close prices, datetime-indexed.
    """

    def __init__(self, prices: pd.Series):
        self.prices = prices.dropna()
        self._arr = self.prices.values.astype(float)

    # -- RSI (Wilder smoothing, 14-period default) --------------------------
    def rsi(self, period: int = 14) -> float:
        """Return the latest RSI value (0-100)."""
        if len(self._arr) < period + 1:
            return 50.0  # neutral fallback
        deltas = np.diff(self._arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Wilder exponential moving average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    # -- MACD (12/26/9) -----------------------------------------------------
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Return (macd_line, signal_line, histogram, cross_signal).

        cross_signal: +1 bullish cross, -1 bearish cross, 0 no cross.
        """
        if len(self._arr) < slow + signal:
            return 0.0, 0.0, 0.0, 0
        ema_fast = self._ema(self._arr, fast)
        ema_slow = self._ema(self._arr, slow)
        macd_line = ema_fast - ema_slow
        sig_line = self._ema(macd_line, signal)
        hist = macd_line - sig_line
        # Detect cross in last 2 bars
        cross = 0
        if len(hist) >= 2:
            if hist[-1] > 0 and hist[-2] <= 0:
                cross = 1   # bullish
            elif hist[-1] < 0 and hist[-2] >= 0:
                cross = -1  # bearish
        return float(macd_line[-1]), float(sig_line[-1]), float(hist[-1]), cross

    # -- Bollinger Band %B ---------------------------------------------------
    def bollinger_pct_b(self, period: int = 20, num_std: float = 2.0) -> float:
        """Return %B: 0 = lower band, 1 = upper band, 0.5 = midline."""
        if len(self._arr) < period:
            return 0.5
        window = self._arr[-period:]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if std == 0:
            return 0.5
        upper = mid + num_std * std
        lower = mid - num_std * std
        pct_b = (self._arr[-1] - lower) / (upper - lower)
        return float(np.clip(pct_b, -0.5, 1.5))

    # -- Moving average crossover (golden/death cross) ----------------------
    def ma_crossover(self, fast_period: int = 50, slow_period: int = 200) -> str:
        """Return 'GOLDEN', 'DEATH', or 'NEUTRAL'."""
        if len(self._arr) < slow_period:
            return "NEUTRAL"
        ma_fast = np.mean(self._arr[-fast_period:])
        ma_slow = np.mean(self._arr[-slow_period:])
        prev_fast = np.mean(self._arr[-fast_period - 1:-1])
        prev_slow = np.mean(self._arr[-slow_period - 1:-1])
        if ma_fast > ma_slow and prev_fast <= prev_slow:
            return "GOLDEN"
        elif ma_fast < ma_slow and prev_fast >= prev_slow:
            return "DEATH"
        elif ma_fast > ma_slow:
            return "ABOVE"
        else:
            return "BELOW"

    # -- Volume-weighted momentum -------------------------------------------
    def volume_weighted_momentum(self, volumes: pd.Series, period: int = 21) -> float:
        """Momentum weighted by relative volume. Falls back to simple momentum."""
        try:
            v = volumes.values.astype(float)
            p = self._arr
            n = min(period, len(p), len(v))
            if n < 5:
                return 0.0
            rets = np.diff(p[-n:]) / p[-n:-1]
            vols = v[-n + 1:] if len(v) >= n else np.ones(n - 1)
            if np.sum(vols) == 0:
                return float(np.sum(rets))
            weights = vols / np.sum(vols)
            return float(np.sum(rets * weights))
        except Exception:
            if len(self._arr) < period:
                return 0.0
            return float((self._arr[-1] / self._arr[-period] - 1.0))

    # -- Composite technical score -------------------------------------------
    def composite_score(self, volumes: Optional[pd.Series] = None) -> float:
        """Combine indicators into a single score in [-1, +1].

        Weighting: RSI 20%, MACD 25%, BB 15%, MA-cross 20%, vol-mom 20%.
        """
        # RSI component: oversold(>0) neutral(0) overbought(<0)
        rsi_val = self.rsi()
        if rsi_val < 30:
            rsi_score = (30 - rsi_val) / 30.0         # 0..+1 oversold=bullish
        elif rsi_val > 70:
            rsi_score = (70 - rsi_val) / 30.0          # -1..0 overbought=bearish
        else:
            rsi_score = 0.0

        # MACD component
        _, _, hist, cross = self.macd()
        macd_score = np.clip(hist * 100, -1, 1)
        if cross == 1:
            macd_score = max(macd_score, 0.5)
        elif cross == -1:
            macd_score = min(macd_score, -0.5)

        # Bollinger %B component
        pct_b = self.bollinger_pct_b()
        bb_score = 1.0 - 2.0 * pct_b   # low %B -> bullish, high %B -> bearish

        # MA crossover component
        cross_label = self.ma_crossover()
        ma_map = {"GOLDEN": 1.0, "ABOVE": 0.3, "NEUTRAL": 0.0, "BELOW": -0.3, "DEATH": -1.0}
        ma_score = ma_map.get(cross_label, 0.0)

        # Volume-weighted momentum
        vwm = self.volume_weighted_momentum(volumes) if volumes is not None else 0.0
        vwm_score = float(np.clip(vwm * 10, -1, 1))

        composite = (
            0.20 * rsi_score
            + 0.25 * macd_score
            + 0.15 * bb_score
            + 0.20 * ma_score
            + 0.20 * vwm_score
        )
        return float(np.clip(composite, -1.0, 1.0))

    # -- Internal helpers ----------------------------------------------------
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average via numpy."""
        alpha = 2.0 / (period + 1)
        out = np.empty_like(data, dtype=float)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out


# ---------------------------------------------------------------------------
# Fundamental Scorer — lightweight fundamental scoring
# ---------------------------------------------------------------------------
class FundamentalScorer:
    """Score a ticker on fundamental metrics using get_fundamentals().

    All methods return a score in [-1, +1] with safe fallbacks.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._data = {}
        try:
            raw = get_fundamentals(ticker)
            if isinstance(raw, dict):
                self._data = raw
        except Exception:
            self._data = {}

    def pe_score(self) -> float:
        """Score P/E ratio: low PE bullish, high PE bearish."""
        pe = self._data.get("trailingPE") or self._data.get("forwardPE")
        if pe is None or not isinstance(pe, (int, float)):
            return 0.0
        if pe < 0:
            return -0.5   # negative earnings
        if pe < 12:
            return 0.8
        if pe < 18:
            return 0.4
        if pe < 25:
            return 0.0
        if pe < 40:
            return -0.3
        return -0.7       # very expensive

    def revenue_growth_score(self) -> float:
        """Score revenue growth rate."""
        growth = self._data.get("revenueGrowth")
        if growth is None or not isinstance(growth, (int, float)):
            return 0.0
        return float(np.clip(growth * 2.0, -1.0, 1.0))

    def earnings_surprise_score(self) -> float:
        """Score latest earnings surprise (beat/miss)."""
        surprise = self._data.get("earningsSurprise") or self._data.get("earningsQuarterlyGrowth")
        if surprise is None or not isinstance(surprise, (int, float)):
            return 0.0
        return float(np.clip(surprise * 3.0, -1.0, 1.0))

    def dividend_yield_score(self) -> float:
        """Assess dividend yield: moderate yield is positive, very high may signal risk."""
        dy = self._data.get("dividendYield")
        if dy is None or not isinstance(dy, (int, float)):
            return 0.0
        if dy < 0:
            return 0.0
        if dy < 0.01:
            return 0.0     # negligible
        if dy < 0.03:
            return 0.3     # healthy
        if dy < 0.06:
            return 0.5     # attractive
        return 0.1          # very high yield — possible value trap

    def composite_score(self) -> float:
        """Weighted fundamental composite in [-1, +1].

        Weights: PE 30%, revenue growth 30%, earnings surprise 25%, dividend 15%.
        """
        score = (
            0.30 * self.pe_score()
            + 0.30 * self.revenue_growth_score()
            + 0.25 * self.earnings_surprise_score()
            + 0.15 * self.dividend_yield_score()
        )
        return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Sector Relative Strength — cross-sector analysis
# ---------------------------------------------------------------------------
class SectorRelativeStrength:
    """Measure sector performance relative to SPY and other sectors.

    Used by SectorBotManager for rotation signals and heatmap data.
    """

    BENCHMARK = "SPY"

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._sector_returns: dict[str, pd.Series] = {}
        self._spy_returns: Optional[pd.Series] = None

    def load(self, sectors_etfs: dict[str, str]):
        """Load return series for each sector ETF and the benchmark."""
        start = (pd.Timestamp.now() - pd.Timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")
        try:
            spy = get_returns(self.BENCHMARK, start=start)
            if isinstance(spy, pd.DataFrame) and not spy.empty:
                self._spy_returns = spy.iloc[:, 0].dropna()
            elif isinstance(spy, pd.Series):
                self._spy_returns = spy.dropna()
        except Exception:
            self._spy_returns = None

        for sector, etf in sectors_etfs.items():
            if not etf:
                continue
            try:
                r = get_returns(etf, start=start)
                if isinstance(r, pd.DataFrame) and not r.empty:
                    self._sector_returns[sector] = r.iloc[:, 0].dropna()
                elif isinstance(r, pd.Series):
                    self._sector_returns[sector] = r.dropna()
            except Exception:
                continue

    def relative_performance(self, sector: str, window_days: int = 21) -> float:
        """Sector excess return vs SPY over window_days."""
        sr = self._sector_returns.get(sector)
        if sr is None or self._spy_returns is None:
            return 0.0
        n = min(window_days, len(sr), len(self._spy_returns))
        if n < 5:
            return 0.0
        sec_ret = float(sr.iloc[-n:].sum())
        spy_ret = float(self._spy_returns.iloc[-n:].sum())
        return sec_ret - spy_ret

    def relative_strength_multi(self, sector: str) -> dict:
        """Return relative strength over 1w, 1m, 3m windows."""
        return {
            "1w": self.relative_performance(sector, 5),
            "1m": self.relative_performance(sector, 21),
            "3m": self.relative_performance(sector, 63),
        }

    def rotation_momentum(self, sector: str) -> float:
        """Sector rotation momentum: acceleration of relative performance.

        Positive = improving relative strength (rotate IN).
        Negative = deteriorating relative strength (rotate OUT).
        """
        rs_1m = self.relative_performance(sector, 21)
        rs_3m = self.relative_performance(sector, 63)
        # Annualised 1m minus annualised 3m -> acceleration
        return (rs_1m * 12) - (rs_3m * 4)

    def sector_correlation(self, sector_a: str, sector_b: str, window: int = 63) -> float:
        """Rolling correlation between two sector ETFs."""
        ra = self._sector_returns.get(sector_a)
        rb = self._sector_returns.get(sector_b)
        if ra is None or rb is None:
            return 0.0
        n = min(window, len(ra), len(rb))
        if n < 10:
            return 0.0
        a = ra.iloc[-n:].values
        b = rb.iloc[-n:].values
        # Align lengths
        m = min(len(a), len(b))
        a, b = a[-m:], b[-m:]
        std_a, std_b = np.std(a), np.std(b)
        if std_a == 0 or std_b == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def mean_reversion_signal(self, sector_a: str, sector_b: str) -> dict:
        """Mean-reversion signal for a sector pair.

        Returns dict with spread z-score and recommended action.
        """
        ra = self._sector_returns.get(sector_a)
        rb = self._sector_returns.get(sector_b)
        if ra is None or rb is None:
            return {"z_score": 0.0, "action": "NEUTRAL", "spread": 0.0}
        n = min(63, len(ra), len(rb))
        if n < 20:
            return {"z_score": 0.0, "action": "NEUTRAL", "spread": 0.0}
        a_cum = np.cumsum(ra.iloc[-n:].values)
        b_cum = np.cumsum(rb.iloc[-n:].values)
        m = min(len(a_cum), len(b_cum))
        spread = a_cum[-m:] - b_cum[-m:]
        mu = np.mean(spread)
        sigma = np.std(spread)
        if sigma == 0:
            return {"z_score": 0.0, "action": "NEUTRAL", "spread": float(spread[-1])}
        z = (spread[-1] - mu) / sigma
        action = "NEUTRAL"
        if z > 1.5:
            action = f"SHORT_{sector_a}_LONG_{sector_b}"
        elif z < -1.5:
            action = f"LONG_{sector_a}_SHORT_{sector_b}"
        return {"z_score": float(z), "action": action, "spread": float(spread[-1])}

    def get_all_relative_strength(self) -> dict[str, dict]:
        """Return relative strength data for all loaded sectors."""
        result = {}
        for sector in self._sector_returns:
            rs = self.relative_strength_multi(sector)
            rs["rotation_momentum"] = self.rotation_momentum(sector)
            result[sector] = rs
        return result


# ---------------------------------------------------------------------------
# Sector Bot
# ---------------------------------------------------------------------------
@dataclass
class SectorBot:
    """Micro-bot specialised for a single GICS sector."""
    sector: str
    gics_code: int = 0
    etf: str = ""
    score: BotScore = field(default_factory=BotScore)
    signals: list = field(default_factory=list)
    is_active: bool = True

    def analyze(self, tickers: list[str], lookback_days: int = 252) -> list[SectorSignal]:
        """Analyse all tickers in this sector and generate signals."""
        self.signals = []
        if not tickers:
            return self.signals

        start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        for ticker in tickers:
            signal = self._analyze_single(ticker, start)
            if signal:
                self.signals.append(signal)

        # Sort by alpha estimate descending
        self.signals.sort(key=lambda s: s.alpha_estimate, reverse=True)
        return self.signals

    def _analyze_single(self, ticker: str, start: str) -> Optional[SectorSignal]:
        """Single-stock analysis within this sector, combining momentum,
        technical, and fundamental scores."""
        try:
            rets = get_returns(ticker, start=start)
            if isinstance(rets, pd.DataFrame) and not rets.empty:
                r = rets.iloc[:, 0].dropna()
            elif isinstance(rets, pd.Series):
                r = rets.dropna()
            else:
                return None

            if len(r) < 21:
                return None

            # Momentum signals
            mom_1m = float(r.iloc[-21:].sum())
            mom_3m = float(r.iloc[-63:].sum()) if len(r) >= 63 else mom_1m
            vol = float(r.std() * np.sqrt(252))
            sharpe = (float(r.mean() * 252) / vol) if vol > 0 else 0.0

            # Quality tier
            tier = classify_quality(sharpe, mom_3m)

            # Technical score
            tech_score = self._technical_score(ticker, start)

            # Fundamental score
            fund_score = self._fundamental_score(ticker)

            # Alpha estimate: blend momentum, technical, and fundamental
            base_alpha = mom_3m + sharpe * 0.01
            enhanced_alpha = (
                0.50 * base_alpha
                + 0.30 * tech_score * 0.05    # scale tech to alpha-like magnitude
                + 0.20 * fund_score * 0.05
            )
            alpha = enhanced_alpha

            # Direction with conviction from combined scores
            conviction = self.compute_conviction(alpha, tech_score, fund_score, tier)

            if conviction > 0.5 and tier in ("A", "B", "C"):
                direction = "BUY"
                confidence = min(1.0, conviction)
            elif conviction < -0.3 and tier in ("F", "G"):
                direction = "SELL"
                confidence = min(1.0, abs(conviction))
            else:
                direction = "HOLD"
                confidence = 0.3

            reasoning = (
                f"Mom3m={mom_3m:.3f} Sharpe={sharpe:.2f} Vol={vol:.2%} "
                f"Tier={tier} Tech={tech_score:+.2f} Fund={fund_score:+.2f}"
            )

            return SectorSignal(
                ticker=ticker,
                sector=self.sector,
                direction=direction,
                confidence=confidence,
                quality_tier=tier,
                alpha_estimate=alpha,
                momentum=mom_3m,
                reasoning=reasoning,
            )

        except Exception:
            return None

    def _technical_score(self, ticker: str, start: str) -> float:
        """Compute composite technical score via TechnicalAnalyzer."""
        try:
            prices = get_adj_close(ticker, start=start)
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                p = prices.iloc[:, 0].dropna()
            elif isinstance(prices, pd.Series):
                p = prices.dropna()
            else:
                return 0.0
            if len(p) < 30:
                return 0.0
            ta = TechnicalAnalyzer(p)
            return ta.composite_score()
        except Exception:
            return 0.0

    def _fundamental_score(self, ticker: str) -> float:
        """Compute composite fundamental score via FundamentalScorer."""
        try:
            fs = FundamentalScorer(ticker)
            return fs.composite_score()
        except Exception:
            return 0.0

    def compute_conviction(
        self,
        alpha: float,
        tech_score: float,
        fund_score: float,
        quality_tier: str,
    ) -> float:
        """Compute conviction level for a signal.

        Returns value in [-1, +1] combining alpha, technical, and fundamental.
        """
        tier_bonus = {"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.0,
                      "E": -0.1, "F": -0.2, "G": -0.3}
        bonus = tier_bonus.get(quality_tier, 0.0)
        raw = alpha * 10.0 + 0.3 * tech_score + 0.2 * fund_score + bonus
        return float(np.clip(raw, -1.0, 1.0))

    def run_skill_analysis(self, ticker: str) -> dict:
        """Run financial-statement analysis skill for a ticker if available."""
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            return test_skill("analyzing-financial-statements", {"ticker": ticker, "sector": self.sector})
        except Exception:
            return {}

    def get_buy_signals(self) -> list[SectorSignal]:
        return [s for s in self.signals if s.direction == "BUY"]

    def get_sell_signals(self) -> list[SectorSignal]:
        return [s for s in self.signals if s.direction == "SELL"]

    def get_sector_summary(self) -> dict:
        """Return a summary dict for this sector bot's latest analysis."""
        buys = self.get_buy_signals()
        sells = self.get_sell_signals()
        holds = [s for s in self.signals if s.direction == "HOLD"]
        avg_alpha = float(np.mean([s.alpha_estimate for s in self.signals])) if self.signals else 0.0
        avg_conf = float(np.mean([s.confidence for s in self.signals])) if self.signals else 0.0
        return {
            "sector": self.sector,
            "etf": self.etf,
            "total_signals": len(self.signals),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "hold_count": len(holds),
            "avg_alpha": round(avg_alpha, 5),
            "avg_confidence": round(avg_conf, 3),
            "top_pick": buys[0].ticker if buys else None,
            "worst_pick": sells[0].ticker if sells else None,
            "tier": self.score.tier,
            "is_active": self.is_active,
        }


# ---------------------------------------------------------------------------
# Agent Scorecard
# ---------------------------------------------------------------------------
class AgentScorecard:
    """Track and rank all 11 sector bots + any additional agents.

    Weekly score: 40% accuracy + 30% Sharpe + 30% hit rate.
    Promotion after 4 consecutive top weeks; demotion after 2 bottom weeks.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/agent_scorecard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scores: dict[str, BotScore] = {}

    def update_score(
        self,
        agent_name: str,
        accuracy: float,
        sharpe: float,
        hit_rate: float,
        is_top_this_week: bool = False,
        is_bottom_this_week: bool = False,
    ):
        """Update an agent's weekly score."""
        score = self.scores.get(agent_name, BotScore())
        score.accuracy = accuracy
        score.sharpe = sharpe
        score.hit_rate = hit_rate
        score.composite = 0.40 * accuracy + 0.30 * min(sharpe / 3.0, 1.0) + 0.30 * hit_rate

        # Promotion/demotion
        if is_top_this_week:
            score.consecutive_top_weeks += 1
            score.consecutive_bottom_weeks = 0
        elif is_bottom_this_week:
            score.consecutive_bottom_weeks += 1
            score.consecutive_top_weeks = 0
        else:
            score.consecutive_top_weeks = max(0, score.consecutive_top_weeks)
            score.consecutive_bottom_weeks = max(0, score.consecutive_bottom_weeks)

        # Assign tier
        if score.consecutive_top_weeks >= 4 and sharpe > 2.0 and accuracy > 0.80:
            score.tier = AgentTier.GENERAL
        elif sharpe > 1.5 and accuracy > 0.55:
            score.tier = AgentTier.CAPTAIN
        elif sharpe > 1.0 and accuracy > 0.50:
            score.tier = AgentTier.LIEUTENANT
        else:
            score.tier = AgentTier.RECRUIT

        # Demotion override
        if score.consecutive_bottom_weeks >= 2 and score.tier != AgentTier.RECRUIT:
            tier_order = [AgentTier.GENERAL, AgentTier.CAPTAIN, AgentTier.LIEUTENANT, AgentTier.RECRUIT]
            idx = tier_order.index(score.tier)
            if idx < len(tier_order) - 1:
                score.tier = tier_order[idx + 1]

        self.scores[agent_name] = score

    def get_leaderboard(self) -> list[tuple[str, BotScore]]:
        """Sorted leaderboard by composite score."""
        return sorted(
            self.scores.items(),
            key=lambda x: x[1].composite,
            reverse=True,
        )

    def save(self):
        """Persist scorecard to JSON."""
        week = datetime.now().strftime("%Y%W")
        data = {name: asdict(score) for name, score in self.scores.items()}
        (self.log_dir / f"{week}.json").write_text(json.dumps(data, indent=2))
        # Also save leaderboard
        lb = [{"rank": i+1, "agent": name, **asdict(score)}
              for i, (name, score) in enumerate(self.get_leaderboard())]
        (self.log_dir / "leaderboard.json").write_text(json.dumps(lb, indent=2))

    def print_leaderboard(self) -> str:
        """ASCII box-drawing leaderboard."""
        lines = []
        lines.append("┌─────┬────────────────────────────┬──────────┬─────────┬──────────┬───────────┐")
        lines.append("│Rank │ Agent                      │ Score    │ Sharpe  │ Accuracy │ Tier      │")
        lines.append("├─────┼────────────────────────────┼──────────┼─────────┼──────────┼───────────┤")
        for i, (name, score) in enumerate(self.get_leaderboard()):
            lines.append(
                f"│ {i+1:<3} │ {name:<26} │ {score.composite:.4f}  │ {score.sharpe:>6.2f}  │ {score.accuracy:>7.1%}  │ {score.tier:<9} │"
            )
        lines.append("└─────┴────────────────────────────┴──────────┴─────────┴──────────┴───────────┘")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sector Bot Manager
# ---------------------------------------------------------------------------
class SectorBotManager:
    """Manages all 11 GICS sector micro-bots."""

    def __init__(self):
        self.bots: dict[str, SectorBot] = {}
        self.scorecard = AgentScorecard()
        self._relative_strength: Optional[SectorRelativeStrength] = None
        self._init_bots()

    def _init_bots(self):
        """Create one bot per GICS sector."""
        for code, sector in GICS_SECTORS.items():
            etf = SECTOR_ETFS.get(sector, "")
            self.bots[sector] = SectorBot(
                sector=sector,
                gics_code=code,
                etf=etf,
            )

    def run_all(self, universe: dict[str, list] = None) -> dict[str, list[SectorSignal]]:
        """Run all 11 sector bots. Returns sector -> signals."""
        results = {}
        for sector, bot in self.bots.items():
            if universe and sector in universe:
                tickers = [s.ticker if hasattr(s, 'ticker') else s for s in universe[sector]]
            else:
                tickers = []
            signals = bot.analyze(tickers)
            results[sector] = signals
        return results

    def get_all_buy_signals(self) -> list[SectorSignal]:
        """Aggregate buy signals across all sectors."""
        buys = []
        for bot in self.bots.values():
            buys.extend(bot.get_buy_signals())
        return sorted(buys, key=lambda s: s.alpha_estimate, reverse=True)

    def get_all_sell_signals(self) -> list[SectorSignal]:
        sells = []
        for bot in self.bots.values():
            sells.extend(bot.get_sell_signals())
        return sells

    def get_sector_rotation_signals(self) -> list[dict]:
        """Return sector rotation recommendations based on relative strength.

        Each recommendation: {sector, etf, direction, momentum, strength_1m, strength_3m}.
        """
        srs = self._ensure_relative_strength()
        all_rs = srs.get_all_relative_strength()
        recommendations = []
        for sector, rs_data in all_rs.items():
            mom = rs_data.get("rotation_momentum", 0.0)
            s1m = rs_data.get("1m", 0.0)
            s3m = rs_data.get("3m", 0.0)
            if mom > 0.5:
                direction = "OVERWEIGHT"
            elif mom < -0.5:
                direction = "UNDERWEIGHT"
            else:
                direction = "NEUTRAL"
            etf = SECTOR_ETFS.get(sector, "")
            recommendations.append({
                "sector": sector,
                "etf": etf,
                "direction": direction,
                "rotation_momentum": round(mom, 4),
                "rel_strength_1m": round(s1m, 4),
                "rel_strength_3m": round(s3m, 4),
            })
        recommendations.sort(key=lambda x: x["rotation_momentum"], reverse=True)
        return recommendations

    def get_top_picks(self, n: int = 5) -> list[SectorSignal]:
        """Return the best N picks across all sectors by alpha estimate."""
        all_buys = self.get_all_buy_signals()
        return all_buys[:n]

    def get_sector_heatmap_data(self) -> dict[str, dict]:
        """Return data suitable for rendering a sector heatmap.

        Keys are sector names; values contain signal counts, avg alpha,
        relative strength, and a color score in [-1, +1].
        """
        srs = self._ensure_relative_strength()
        all_rs = srs.get_all_relative_strength()
        heatmap = {}
        for sector, bot in self.bots.items():
            summary = bot.get_sector_summary()
            rs_data = all_rs.get(sector, {})
            rel_1m = rs_data.get("1m", 0.0)
            rot_mom = rs_data.get("rotation_momentum", 0.0)
            # Color score: blend of avg_alpha direction and relative strength
            alpha_dir = 1.0 if summary["avg_alpha"] > 0 else (-1.0 if summary["avg_alpha"] < 0 else 0.0)
            color_score = float(np.clip(
                0.4 * alpha_dir + 0.3 * np.clip(rel_1m * 20, -1, 1) + 0.3 * np.clip(rot_mom, -1, 1),
                -1.0, 1.0
            ))
            heatmap[sector] = {
                "etf": bot.etf,
                "buy_count": summary["buy_count"],
                "sell_count": summary["sell_count"],
                "hold_count": summary["hold_count"],
                "avg_alpha": summary["avg_alpha"],
                "rel_strength_1m": round(rel_1m, 4),
                "rotation_momentum": round(rot_mom, 4),
                "color_score": round(color_score, 3),
                "top_pick": summary["top_pick"],
            }
        return heatmap

    def print_sector_dashboard(self) -> str:
        """Return an ASCII dashboard summarising all sectors."""
        lines = []
        lines.append("=" * 100)
        lines.append("  SECTOR DASHBOARD".center(100))
        lines.append("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S").center(96))
        lines.append("=" * 100)
        lines.append(
            f"{'Sector':<28} {'ETF':<6} {'Buys':>5} {'Sells':>5} "
            f"{'Holds':>5} {'AvgAlpha':>10} {'Tier':<16} {'TopPick':<8}"
        )
        lines.append("-" * 100)
        for sector in sorted(self.bots.keys()):
            bot = self.bots[sector]
            s = bot.get_sector_summary()
            top = s["top_pick"] or "---"
            lines.append(
                f"{s['sector']:<28} {s['etf']:<6} {s['buy_count']:>5} {s['sell_count']:>5} "
                f"{s['hold_count']:>5} {s['avg_alpha']:>+10.5f} {s['tier']:<16} {top:<8}"
            )
        lines.append("-" * 100)

        # Aggregate stats
        all_buys = self.get_all_buy_signals()
        all_sells = self.get_all_sell_signals()
        total_signals = sum(len(b.signals) for b in self.bots.values())
        lines.append(
            f"  Total signals: {total_signals}  |  Buys: {len(all_buys)}  |  "
            f"Sells: {len(all_sells)}  |  Holds: {total_signals - len(all_buys) - len(all_sells)}"
        )

        if all_buys:
            top5 = all_buys[:5]
            lines.append("")
            lines.append("  TOP 5 PICKS:")
            for i, sig in enumerate(top5):
                lines.append(
                    f"    {i+1}. {sig.ticker:<8} ({sig.sector:<24}) "
                    f"alpha={sig.alpha_estimate:+.4f}  conf={sig.confidence:.2f}  {sig.quality_tier}"
                )

        lines.append("=" * 100)
        return "\n".join(lines)

    # -- Internal helpers ----------------------------------------------------
    def _ensure_relative_strength(self) -> SectorRelativeStrength:
        """Lazy-load the SectorRelativeStrength analyzer."""
        if self._relative_strength is None:
            self._relative_strength = SectorRelativeStrength()
            etfs = {sector: bot.etf for sector, bot in self.bots.items() if bot.etf}
            try:
                self._relative_strength.load(etfs)
            except Exception:
                pass
        return self._relative_strength
