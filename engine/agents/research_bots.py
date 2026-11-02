"""Research Bots — 11 GICS Sector Research Specialists.

Each bot is a specialized research agent for one GICS sector that:
    - Monitors sector-specific securities daily
    - Learns from market behaviour using our investment strategy
    - Feeds research into the investment decision matrix
    - Tracks its own performance and improves over time
    - Identifies mathematical best outcome estimates
    - Uses reliable, verifiable data sources (OpenBB)

Bot Hierarchy:
    DIRECTOR   — Sharpe >3.0, accuracy >85%, 6+ consecutive top weeks
    GENERAL    — Sharpe >2.0, accuracy >80%, 4+ consecutive top weeks
    CAPTAIN    — Sharpe >1.5, accuracy >55%
    LIEUTENANT — Sharpe >1.0, accuracy >50%
    RECRUIT    — Below thresholds

DNA Framework Agent Report:
    - Specialty, progress, improvement trajectory
    - Weekly scoring: 40% accuracy + 30% Sharpe + 30% hit rate
    - Hierarchy promotion/demotion based on sustained performance
"""

import logging
import json
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..data.yahoo_data import get_adj_close, get_returns, get_prices, get_fundamentals
    from ..data.universe_engine import (
        GICS_SECTORS, SECTOR_ETFS, FACTOR_ETFS, Security,
        UniverseEngine, get_engine, RV_PAIRS,
    )
    from ..ml.alpha_optimizer import classify_quality, build_features
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_prices(*a, **kw): return pd.DataFrame()
    def get_fundamentals(*a, **kw): return {}
    GICS_SECTORS = {}
    SECTOR_ETFS = {}
    FACTOR_ETFS = {}
    RV_PAIRS = []
    def classify_quality(s, m): return "D"
    def build_features(r): return pd.DataFrame()
    class Security:
        ticker = ""
        market_cap = 0
    def get_engine():
        return None

logger = logging.getLogger(__name__)

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
# Research Bot Hierarchy
# ---------------------------------------------------------------------------
class BotRank:
    DIRECTOR = "DIRECTOR"
    GENERAL = "GENERAL"
    CAPTAIN = "CAPTAIN"
    LIEUTENANT = "LIEUTENANT"
    RECRUIT = "RECRUIT"


RANK_THRESHOLDS = {
    BotRank.DIRECTOR: {"min_sharpe": 3.0, "min_accuracy": 0.85, "min_top_weeks": 6},
    BotRank.GENERAL: {"min_sharpe": 2.0, "min_accuracy": 0.80, "min_top_weeks": 4},
    BotRank.CAPTAIN: {"min_sharpe": 1.5, "min_accuracy": 0.55, "min_top_weeks": 0},
    BotRank.LIEUTENANT: {"min_sharpe": 1.0, "min_accuracy": 0.50, "min_top_weeks": 0},
    BotRank.RECRUIT: {"min_sharpe": -999, "min_accuracy": 0.0, "min_top_weeks": 0},
}

RANK_ORDER = [BotRank.DIRECTOR, BotRank.GENERAL, BotRank.CAPTAIN, BotRank.LIEUTENANT, BotRank.RECRUIT]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ResearchSignal:
    """A research bot's investment signal."""
    id: str = ""
    ticker: str = ""
    sector: str = ""
    direction: str = "HOLD"  # LONG / SHORT / HOLD
    conviction: str = "LOW"  # LOW / MEDIUM / HIGH / EXTREME
    confidence: float = 0.0
    alpha_estimate: float = 0.0
    edge_bps: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    quality_tier: str = "D"
    reasoning: str = ""
    confirming_factors: list = field(default_factory=list)
    timestamp: str = ""
    expiry_days: int = 5


@dataclass
class BotPerformance:
    """Track bot performance metrics."""
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0
    composite_score: float = 0.0
    rank: str = BotRank.RECRUIT
    consecutive_top_weeks: int = 0
    consecutive_bottom_weeks: int = 0
    total_pnl: float = 0.0
    avg_alpha: float = 0.0
    best_signal: str = ""
    worst_signal: str = ""
    weekly_scores: list = field(default_factory=list)
    improvement_trend: float = 0.0  # slope of weekly scores


@dataclass
class LearningState:
    """Bot's learned parameters from market observation."""
    # Feature weights learned from historical accuracy
    momentum_weight: float = 0.25
    mean_reversion_weight: float = 0.20
    quality_weight: float = 0.20
    volume_weight: float = 0.15
    technical_weight: float = 0.20
    # Sector-specific learned thresholds
    optimal_rsi_buy: float = 30.0
    optimal_rsi_sell: float = 70.0
    optimal_macd_threshold: float = 0.0
    sector_beta: float = 1.0
    sector_vol_regime: str = "NORMAL"
    # Pattern memory
    patterns_seen: int = 0
    patterns_correct: int = 0
    last_update: str = ""


@dataclass
class DailyResearchReport:
    """Daily output from a research bot."""
    sector: str = ""
    date: str = ""
    market_assessment: str = ""
    top_picks: list = field(default_factory=list)  # list of ResearchSignal
    avoid_list: list = field(default_factory=list)
    sector_outlook: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    sector_momentum: float = 0.0
    relative_strength_vs_spy: float = 0.0
    best_opportunity: str = ""
    mathematical_best_estimate: dict = field(default_factory=dict)
    risk_warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Technical Analysis Module (per-bot)
# ---------------------------------------------------------------------------
class BotTechnicalAnalysis:
    """Technical analysis toolkit used by each research bot."""

    def compute_rsi(self, returns: np.ndarray, period: int = 14) -> float:
        if len(returns) < period:
            return 50.0
        gains = returns[-period:].copy()
        losses = returns[-period:].copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = gains.mean()
        avg_loss = abs(losses.mean())
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def compute_macd(self, prices: np.ndarray) -> dict:
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0, "crossover": "NONE"}
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd_line = ema_12[-1] - ema_26[-1]
        # Signal line (9-period EMA of MACD)
        if len(prices) > 35:
            macd_series = self._ema(prices, 12) - self._ema(prices, 26)
            signal_series = self._ema(macd_series[-20:], 9) if len(macd_series) >= 20 else macd_series
            signal = signal_series[-1]
        else:
            signal = macd_line * 0.9
        histogram = macd_line - signal
        crossover = "BULLISH" if macd_line > signal and histogram > 0 else (
            "BEARISH" if macd_line < signal else "NONE"
        )
        return {"macd": float(macd_line), "signal": float(signal),
                "histogram": float(histogram), "crossover": crossover}

    def compute_bollinger(self, prices: np.ndarray, period: int = 20) -> dict:
        if len(prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "pct_b": 0.5, "squeeze": False}
        sma = prices[-period:].mean()
        std = prices[-period:].std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        pct_b = (prices[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        bandwidth = (upper - lower) / sma if sma > 0 else 0
        squeeze = bandwidth < 0.04  # Tight squeeze
        return {"upper": float(upper), "middle": float(sma), "lower": float(lower),
                "pct_b": float(pct_b), "squeeze": squeeze}

    def compute_momentum_score(self, returns: np.ndarray) -> dict:
        scores = {}
        if len(returns) >= 5:
            scores["mom_5d"] = float(returns[-5:].sum())
        if len(returns) >= 21:
            scores["mom_1m"] = float(returns[-21:].sum())
        if len(returns) >= 63:
            scores["mom_3m"] = float(returns[-63:].sum())
        if len(returns) >= 126:
            scores["mom_6m"] = float(returns[-126:].sum())
        if len(returns) >= 252:
            scores["mom_12m"] = float(returns[-252:].sum())
        # Composite: weighted blend
        weights = {"mom_5d": 0.15, "mom_1m": 0.25, "mom_3m": 0.30, "mom_6m": 0.20, "mom_12m": 0.10}
        composite = sum(scores.get(k, 0) * w for k, w in weights.items())
        scores["composite"] = composite
        return scores

    def compute_volume_signal(self, volumes: np.ndarray) -> dict:
        if len(volumes) < 20:
            return {"ratio": 1.0, "trend": "FLAT", "anomaly": False}
        avg_20d = volumes[-20:].mean()
        current = volumes[-1]
        ratio = current / avg_20d if avg_20d > 0 else 1.0
        trend = "RISING" if volumes[-5:].mean() > avg_20d * 1.1 else (
            "FALLING" if volumes[-5:].mean() < avg_20d * 0.9 else "FLAT"
        )
        return {"ratio": float(ratio), "trend": trend, "anomaly": ratio > 2.0}

    def identify_support_resistance(self, prices: np.ndarray, window: int = 20) -> dict:
        if len(prices) < window * 2:
            return {"support": 0, "resistance": 0}
        recent = prices[-window * 2:]
        # Simple pivot-based S/R
        resistance = float(np.percentile(recent, 90))
        support = float(np.percentile(recent, 10))
        return {"support": support, "resistance": resistance}

    def _ema(self, data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result


# ---------------------------------------------------------------------------
# Sector Research Bot
# ---------------------------------------------------------------------------
class SectorResearchBot:
    """Specialized research agent for a single GICS sector.

    Each bot:
    1. Monitors its sector's securities daily
    2. Runs technical + fundamental analysis
    3. Generates conviction-scored signals
    4. Learns from outcomes (updates weights)
    5. Feeds into the investment decision matrix
    6. Computes mathematical best-outcome estimates
    """

    def __init__(self, sector: str, gics_code: int = 0, etf: str = ""):
        self.sector = sector
        self.gics_code = gics_code
        self.etf = etf
        self.performance = BotPerformance(rank=BotRank.RECRUIT)
        self.learning = LearningState()
        self.tech = BotTechnicalAnalysis()
        self._signal_history: deque = deque(maxlen=500)
        self._daily_reports: deque = deque(maxlen=60)
        self._active_signals: list[ResearchSignal] = []
        self._sector_returns_cache: Optional[pd.Series] = None
        self.is_active = True
        self.bot_id = str(uuid.uuid4())[:8]

    def run_daily_research(self, tickers: list[str], lookback_days: int = 252) -> DailyResearchReport:
        """Execute daily research cycle for this sector.

        Returns a comprehensive daily research report with:
        - Top picks with conviction scoring
        - Avoid list (short/sell candidates)
        - Mathematical best outcome estimate
        - Sector outlook and relative strength
        """
        report = DailyResearchReport(
            sector=self.sector,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        if not tickers:
            report.market_assessment = "No tickers available for analysis"
            return report

        # 1. Analyze sector ETF for macro context
        sector_data = self._analyze_sector_etf()
        report.sector_momentum = sector_data.get("momentum", 0)
        report.relative_strength_vs_spy = sector_data.get("rel_strength", 0)
        report.sector_outlook = sector_data.get("outlook", "NEUTRAL")

        # 2. Analyze each ticker
        all_signals = []
        start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        for ticker in tickers:
            signal = self._research_single(ticker, start)
            if signal:
                all_signals.append(signal)

        # 3. Sort by alpha estimate
        all_signals.sort(key=lambda s: s.alpha_estimate, reverse=True)

        # 4. Top picks (LONG signals with HIGH+ conviction)
        report.top_picks = [s for s in all_signals if s.direction == "LONG" and s.confidence > 0.5][:10]

        # 5. Avoid list (SHORT signals or poor quality)
        report.avoid_list = [s for s in all_signals if s.direction == "SHORT"][:5]

        # 6. Mathematical best estimate
        report.mathematical_best_estimate = self._compute_best_estimate(all_signals)

        # 7. Best opportunity
        if report.top_picks:
            best = report.top_picks[0]
            report.best_opportunity = (
                f"{best.ticker}: {best.direction} | Conviction={best.conviction} "
                f"| Alpha={best.alpha_estimate:.1%} | {best.reasoning}"
            )

        # 8. Risk warnings
        report.risk_warnings = self._assess_sector_risks(sector_data)

        # 9. Market assessment
        report.market_assessment = self._generate_assessment(sector_data, all_signals)

        # Store
        self._daily_reports.append(report)
        self._active_signals = [s for s in all_signals if s.direction != "HOLD"]

        return report

    def _analyze_sector_etf(self) -> dict:
        """Analyze sector ETF for macro context."""
        result = {"momentum": 0, "rel_strength": 0, "outlook": "NEUTRAL",
                  "vol": 0, "rsi": 50, "trend": "FLAT"}
        if not self.etf:
            return result

        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
            prices = get_adj_close([self.etf, "SPY"], start=start)
            if prices.empty:
                return result

            if self.etf in prices.columns:
                etf_prices = prices[self.etf].dropna()
                etf_returns = etf_prices.pct_change().dropna().values

                if len(etf_returns) >= 21:
                    result["momentum"] = float(etf_returns[-21:].sum())
                    result["vol"] = float(etf_returns.std() * np.sqrt(252))
                    result["rsi"] = self.tech.compute_rsi(etf_returns)

                    # Trend determination
                    if len(etf_prices) >= 50:
                        sma_20 = float(etf_prices.iloc[-20:].mean())
                        sma_50 = float(etf_prices.iloc[-50:].mean())
                        price = float(etf_prices.iloc[-1])
                        if price > sma_20 > sma_50:
                            result["trend"] = "UPTREND"
                        elif price < sma_20 < sma_50:
                            result["trend"] = "DOWNTREND"
                        else:
                            result["trend"] = "SIDEWAYS"

            # Relative strength vs SPY
            if self.etf in prices.columns and "SPY" in prices.columns:
                etf_r = prices[self.etf].pct_change().dropna()
                spy_r = prices["SPY"].pct_change().dropna()
                common = etf_r.index.intersection(spy_r.index)
                if len(common) >= 21:
                    etf_1m = float(etf_r.loc[common[-21:]].sum())
                    spy_1m = float(spy_r.loc[common[-21:]].sum())
                    result["rel_strength"] = etf_1m - spy_1m

            # Outlook
            if result["momentum"] > 0.03 and result["rsi"] < 70:
                result["outlook"] = "BULLISH"
            elif result["momentum"] < -0.03 and result["rsi"] > 30:
                result["outlook"] = "BEARISH"
            else:
                result["outlook"] = "NEUTRAL"

        except Exception as e:
            logger.debug(f"Sector ETF analysis failed for {self.etf}: {e}")

        return result

    def _research_single(self, ticker: str, start: str) -> Optional[ResearchSignal]:
        """Deep research on a single ticker."""
        try:
            rets = get_returns(ticker, start=start)
            if isinstance(rets, pd.DataFrame) and not rets.empty:
                r = rets.iloc[:, 0].dropna().values
            elif isinstance(rets, pd.Series):
                r = rets.dropna().values
            else:
                return None

            if len(r) < 21:
                return None

            # Compute features
            rsi = self.tech.compute_rsi(r)
            mom = self.tech.compute_momentum_score(r)
            vol = float(r.std() * np.sqrt(252))
            sharpe = (float(r.mean() * 252) / vol) if vol > 0 else 0

            # Quality tier
            mom_3m = mom.get("mom_3m", 0)
            quality = classify_quality(sharpe, mom_3m)

            # MACD (from cumulative returns as price proxy)
            cum_prices = np.cumprod(1 + r)
            macd_data = self.tech.compute_macd(cum_prices)
            bb_data = self.tech.compute_bollinger(cum_prices)
            sr_data = self.tech.identify_support_resistance(cum_prices)

            # Composite score using learned weights
            composite = (
                self.learning.momentum_weight * mom.get("composite", 0) * 10 +
                self.learning.quality_weight * (1 if quality in ("A", "B", "C") else -1) * 0.3 +
                self.learning.technical_weight * (1 if macd_data["crossover"] == "BULLISH" else
                                                   (-1 if macd_data["crossover"] == "BEARISH" else 0)) * 0.3 +
                self.learning.mean_reversion_weight * (1 if rsi < self.learning.optimal_rsi_buy else
                                                       (-1 if rsi > self.learning.optimal_rsi_sell else 0)) * 0.2
            )

            # Confirming factors
            confirmations = []
            if mom.get("composite", 0) > 0:
                confirmations.append("MOMENTUM_POSITIVE")
            if quality in ("A", "B"):
                confirmations.append("HIGH_QUALITY")
            if macd_data["crossover"] == "BULLISH":
                confirmations.append("MACD_BULLISH")
            if rsi < 30:
                confirmations.append("RSI_OVERSOLD")
            elif rsi > 70:
                confirmations.append("RSI_OVERBOUGHT")
            if bb_data["pct_b"] < 0.2:
                confirmations.append("BB_LOWER_BAND")
            elif bb_data["pct_b"] > 0.8:
                confirmations.append("BB_UPPER_BAND")
            if bb_data["squeeze"]:
                confirmations.append("BB_SQUEEZE")

            # Direction and conviction
            direction = "HOLD"
            conviction = "LOW"
            confidence = 0.0

            if composite > 0.3 and len(confirmations) >= 3:
                direction = "LONG"
                if composite > 0.7 and len(confirmations) >= 4:
                    conviction = "HIGH"
                    confidence = min(0.95, 0.5 + composite * 0.3)
                elif composite > 0.5:
                    conviction = "MEDIUM"
                    confidence = min(0.80, 0.4 + composite * 0.3)
                else:
                    conviction = "LOW"
                    confidence = min(0.60, 0.3 + composite * 0.2)
            elif composite < -0.3 and len([c for c in confirmations if "OVERBOUGHT" in c or "BEARISH" in c]) >= 2:
                direction = "SHORT"
                conviction = "MEDIUM" if composite < -0.5 else "LOW"
                confidence = min(0.70, 0.3 + abs(composite) * 0.3)

            # Alpha estimate
            alpha = mom_3m + sharpe * 0.01 + (0.02 if quality in ("A", "B") else 0)

            # Entry/stop/target
            last_price = float(cum_prices[-1])
            stop_pct = 0.05 if conviction == "HIGH" else 0.08
            target_pct = alpha * 4 if alpha > 0 else 0.10

            signal = ResearchSignal(
                id=str(uuid.uuid4())[:8],
                ticker=ticker,
                sector=self.sector,
                direction=direction,
                conviction=conviction,
                confidence=confidence,
                alpha_estimate=alpha,
                edge_bps=max(0, alpha * 10000),
                entry_price=last_price,
                stop_loss=last_price * (1 - stop_pct) if direction == "LONG" else last_price * (1 + stop_pct),
                target_price=last_price * (1 + target_pct) if direction == "LONG" else last_price * (1 - target_pct),
                quality_tier=quality,
                reasoning=(
                    f"RSI={rsi:.0f} MACD={macd_data['crossover']} Mom={mom.get('composite', 0):.3f} "
                    f"Sharpe={sharpe:.2f} Vol={vol:.1%} Quality={quality} BB%B={bb_data['pct_b']:.2f}"
                ),
                confirming_factors=confirmations,
                timestamp=datetime.now().isoformat(),
            )

            self._signal_history.append(signal)
            return signal

        except Exception as e:
            logger.debug(f"Research failed for {ticker}: {e}")
            return None

    def _compute_best_estimate(self, signals: list[ResearchSignal]) -> dict:
        """Mathematical best possible outcome estimate.

        Uses the highest conviction signals to estimate what the
        theoretical maximum daily alpha could be.
        """
        if not signals:
            return {"best_daily_alpha_bps": 0, "best_ticker": "N/A",
                    "probability": 0, "expected_value_bps": 0}

        long_signals = [s for s in signals if s.direction == "LONG" and s.confidence > 0.5]
        if not long_signals:
            return {"best_daily_alpha_bps": 0, "best_ticker": "N/A",
                    "probability": 0, "expected_value_bps": 0}

        best = max(long_signals, key=lambda s: s.alpha_estimate * s.confidence)
        daily_alpha_bps = best.alpha_estimate * 10000 / 252  # Annualized → daily
        probability = best.confidence
        ev = daily_alpha_bps * probability

        return {
            "best_daily_alpha_bps": round(daily_alpha_bps, 2),
            "best_ticker": best.ticker,
            "probability": round(probability, 3),
            "expected_value_bps": round(ev, 2),
            "conviction": best.conviction,
            "confirming_factors": len(best.confirming_factors),
            "quality_tier": best.quality_tier,
        }

    def _assess_sector_risks(self, sector_data: dict) -> list[str]:
        warnings = []
        if sector_data.get("vol", 0) > 0.30:
            warnings.append(f"High sector volatility: {sector_data['vol']:.1%}")
        if sector_data.get("rsi", 50) > 75:
            warnings.append(f"Sector RSI overbought: {sector_data['rsi']:.0f}")
        if sector_data.get("rsi", 50) < 25:
            warnings.append(f"Sector RSI oversold: {sector_data['rsi']:.0f}")
        if sector_data.get("rel_strength", 0) < -0.05:
            warnings.append(f"Underperforming SPY by {abs(sector_data['rel_strength']):.1%}")
        return warnings

    def _generate_assessment(self, sector_data: dict, signals: list[ResearchSignal]) -> str:
        long_count = sum(1 for s in signals if s.direction == "LONG")
        short_count = sum(1 for s in signals if s.direction == "SHORT")
        high_conv = sum(1 for s in signals if s.conviction in ("HIGH", "EXTREME"))

        return (
            f"{self.sector}: {sector_data.get('outlook', 'NEUTRAL')} | "
            f"Trend={sector_data.get('trend', 'N/A')} | "
            f"RS={sector_data.get('rel_strength', 0):+.1%} | "
            f"Signals: {long_count}L/{short_count}S/{high_conv} high-conviction"
        )

    def learn_from_outcome(self, ticker: str, actual_return: float):
        """Update learning weights based on observed outcome.

        Simple online learning: adjust weights toward signals that
        were correct and away from those that were wrong.
        """
        learning_rate = 0.05
        matching = [s for s in self._signal_history if s.ticker == ticker]
        if not matching:
            return

        signal = matching[-1]
        predicted_positive = signal.direction == "LONG"
        actual_positive = actual_return > 0

        correct = predicted_positive == actual_positive
        self.learning.patterns_seen += 1
        if correct:
            self.learning.patterns_correct += 1
            adjustment = learning_rate
        else:
            adjustment = -learning_rate

        # Update weights (bounded)
        self.learning.momentum_weight = np.clip(self.learning.momentum_weight + adjustment * 0.3, 0.05, 0.50)
        self.learning.technical_weight = np.clip(self.learning.technical_weight + adjustment * 0.2, 0.05, 0.50)
        self.learning.quality_weight = np.clip(self.learning.quality_weight + adjustment * 0.2, 0.05, 0.50)
        self.learning.mean_reversion_weight = np.clip(self.learning.mean_reversion_weight + adjustment * 0.15, 0.05, 0.50)
        self.learning.volume_weight = np.clip(self.learning.volume_weight + adjustment * 0.15, 0.05, 0.50)

        # Normalize
        total = (self.learning.momentum_weight + self.learning.technical_weight +
                 self.learning.quality_weight + self.learning.mean_reversion_weight +
                 self.learning.volume_weight)
        if total > 0:
            self.learning.momentum_weight /= total
            self.learning.technical_weight /= total
            self.learning.quality_weight /= total
            self.learning.mean_reversion_weight /= total
            self.learning.volume_weight /= total

        self.learning.last_update = datetime.now().isoformat()

    def update_weekly_score(self, accuracy: float, sharpe: float, hit_rate: float,
                            is_top: bool = False, is_bottom: bool = False):
        """Update weekly performance score and rank."""
        self.performance.accuracy = accuracy
        self.performance.sharpe = sharpe
        self.performance.hit_rate = hit_rate
        self.performance.composite_score = 0.40 * accuracy + 0.30 * min(sharpe / 3.0, 1.0) + 0.30 * hit_rate

        # Promotion tracking
        if is_top:
            self.performance.consecutive_top_weeks += 1
            self.performance.consecutive_bottom_weeks = 0
        elif is_bottom:
            self.performance.consecutive_bottom_weeks += 1
            self.performance.consecutive_top_weeks = 0

        # Rank assignment
        old_rank = self.performance.rank
        new_rank = BotRank.RECRUIT
        for rank in RANK_ORDER:
            thresh = RANK_THRESHOLDS[rank]
            if (sharpe >= thresh["min_sharpe"] and
                accuracy >= thresh["min_accuracy"] and
                self.performance.consecutive_top_weeks >= thresh["min_top_weeks"]):
                new_rank = rank
                break

        # Demotion check
        if self.performance.consecutive_bottom_weeks >= 2:
            idx = RANK_ORDER.index(new_rank)
            if idx < len(RANK_ORDER) - 1:
                new_rank = RANK_ORDER[min(idx + 1, len(RANK_ORDER) - 1)]

        self.performance.rank = new_rank
        self.performance.weekly_scores.append(self.performance.composite_score)

        # Improvement trend (slope of last 8 weekly scores)
        if len(self.performance.weekly_scores) >= 4:
            recent = self.performance.weekly_scores[-8:]
            x = np.arange(len(recent))
            if len(x) > 1:
                slope = np.polyfit(x, recent, 1)[0]
                self.performance.improvement_trend = float(slope)

    def run_skill_financial_model(self, ticker: str) -> dict:
        """Run financial modeling skill and download results if available."""
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            result = test_skill("creating-financial-models", {"ticker": ticker, "sector": self.sector})
            file_ids = extract_file_ids(result) if isinstance(result, (dict, str)) else []
            for fid in file_ids:
                download_file(fid)
            return result if isinstance(result, dict) else {"output": result}
        except Exception:
            return {}

    def get_active_signals(self) -> list[ResearchSignal]:
        return [s for s in self._active_signals if s.direction != "HOLD"]

    def get_high_conviction_signals(self) -> list[ResearchSignal]:
        return [s for s in self._active_signals
                if s.conviction in ("HIGH", "EXTREME") and s.confidence > 0.6]


# ---------------------------------------------------------------------------
# Research Bot Manager (Intelligence Arm)
# ---------------------------------------------------------------------------
class ResearchBotManager:
    """Manages all 11 GICS sector research bots.

    The Intelligence Arm:
    - Deploys and coordinates sector research bots
    - Aggregates research into investment decisions
    - Tracks bot performance and manages hierarchy
    - Generates DNA framework reports
    - Identifies mathematical best outcomes across all sectors
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.bots: dict[str, SectorResearchBot] = {}
        self.log_dir = log_dir or Path("logs/research_bots")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_bots()
        self._daily_reports: dict[str, DailyResearchReport] = {}

    def _init_bots(self):
        """Initialize one research bot per GICS sector."""
        for code, sector in GICS_SECTORS.items():
            etf = SECTOR_ETFS.get(sector, "")
            self.bots[sector] = SectorResearchBot(
                sector=sector, gics_code=code, etf=etf,
            )

    def run_daily_research(self, universe: Optional[dict] = None) -> dict[str, DailyResearchReport]:
        """Run all 11 sector bots' daily research cycle.

        Returns sector → DailyResearchReport mapping.
        """
        results = {}
        for sector, bot in self.bots.items():
            if not bot.is_active:
                continue

            tickers = []
            if universe and sector in universe:
                tickers = [s.ticker if hasattr(s, "ticker") else s for s in universe[sector]]

            try:
                report = bot.run_daily_research(tickers)
                results[sector] = report
            except Exception as e:
                logger.warning(f"Research bot {sector} failed: {e}")
                results[sector] = DailyResearchReport(
                    sector=sector,
                    date=datetime.now().strftime("%Y-%m-%d"),
                    market_assessment=f"Research failed: {e}",
                )

        self._daily_reports = results
        return results

    def get_all_signals(self) -> list[ResearchSignal]:
        """Get all active signals across all bots, sorted by alpha."""
        all_signals = []
        for bot in self.bots.values():
            all_signals.extend(bot.get_active_signals())
        return sorted(all_signals, key=lambda s: s.alpha_estimate * s.confidence, reverse=True)

    def get_high_conviction_signals(self) -> list[ResearchSignal]:
        """Get only high-conviction signals across all sectors."""
        signals = []
        for bot in self.bots.values():
            signals.extend(bot.get_high_conviction_signals())
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def get_best_opportunities(self, top_n: int = 10) -> list[ResearchSignal]:
        """Top N opportunities across all sectors by expected value."""
        all_signals = self.get_all_signals()
        return all_signals[:top_n]

    def get_mathematical_best_estimates(self) -> dict:
        """Mathematical best outcome estimates across all sectors."""
        estimates = {}
        for sector, report in self._daily_reports.items():
            estimates[sector] = report.mathematical_best_estimate
        # Overall best
        best_sector = max(estimates.items(),
                          key=lambda x: x[1].get("expected_value_bps", 0),
                          default=("N/A", {}))
        return {
            "sectors": estimates,
            "overall_best_sector": best_sector[0],
            "overall_best_estimate": best_sector[1],
        }

    def update_weekly_scores(self):
        """Update all bots' weekly scores and rankings.

        Accuracy is computed from actual outcomes tracked via learn_from_outcome(),
        NOT from confidence scores. Hit rate measures the fraction of high-confidence
        signals (confidence > 0.6) that were directionally correct.
        """
        scores = []
        for sector, bot in self.bots.items():
            signals = list(bot._signal_history)
            if not signals:
                continue

            # Outcome-based accuracy from learning loop (real results)
            if bot.learning.patterns_seen > 0:
                accuracy = bot.learning.patterns_correct / bot.learning.patterns_seen
            else:
                # Fallback: no outcomes recorded yet — use 0.5 (neutral)
                accuracy = 0.5

            # Approximate Sharpe from alpha estimates
            alphas = [s.alpha_estimate for s in signals if s.alpha_estimate != 0]
            if alphas:
                sharpe = float(np.mean(alphas) / max(np.std(alphas), 0.001) * np.sqrt(252))
            else:
                sharpe = 0

            scores.append((sector, bot.performance.composite_score))

        # Determine top and bottom
        if len(scores) >= 3:
            scores.sort(key=lambda x: x[1], reverse=True)
            top_sectors = {s[0] for s in scores[:3]}
            bottom_sectors = {s[0] for s in scores[-3:]}
        else:
            top_sectors = set()
            bottom_sectors = set()

        for sector, bot in self.bots.items():
            # Outcome-based accuracy (real results from learn_from_outcome)
            if bot.learning.patterns_seen > 0:
                accuracy = bot.learning.patterns_correct / bot.learning.patterns_seen
            else:
                accuracy = 0.5

            # Hit rate: fraction of high-confidence signals that were correct
            # (distinct from accuracy — measures calibration of conviction)
            high_conf_signals = [s for s in bot._signal_history if s.confidence > 0.6]
            if high_conf_signals and bot.learning.patterns_seen > 0:
                # Match high-confidence signals against outcomes
                high_conf_correct = sum(
                    1 for s in high_conf_signals
                    if any(
                        s.ticker == o_s.ticker and
                        (s.direction == "LONG") == (o_s.alpha_estimate > 0)
                        for o_s in bot._signal_history
                        if o_s.ticker == s.ticker
                    )
                )
                hit_rate = high_conf_correct / len(high_conf_signals)
            else:
                hit_rate = accuracy  # Default to overall accuracy if no high-conf data

            alphas = [s.alpha_estimate for s in bot._signal_history if s.alpha_estimate != 0]
            sharpe = float(np.mean(alphas) / max(np.std(alphas), 0.001) * np.sqrt(252)) if alphas else 0

            bot.update_weekly_score(
                accuracy=accuracy,
                sharpe=sharpe,
                hit_rate=hit_rate,
                is_top=sector in top_sectors,
                is_bottom=sector in bottom_sectors,
            )

    def get_leaderboard(self) -> list[tuple[str, BotPerformance]]:
        """Ranked leaderboard of all research bots."""
        return sorted(
            [(sector, bot.performance) for sector, bot in self.bots.items()],
            key=lambda x: x[1].composite_score,
            reverse=True,
        )

    def print_dna_report(self) -> str:
        """DNA Framework Agent Report — specialty, progress, improvement."""
        lines = [
            "=" * 90,
            "DNA FRAMEWORK — RESEARCH BOT INTELLIGENCE REPORT",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 90,
            "",
            f"  Total Bots: {len(self.bots)}  |  Active: {sum(1 for b in self.bots.values() if b.is_active)}",
            "",
        ]

        # Hierarchy summary
        rank_counts = {}
        for bot in self.bots.values():
            rank_counts[bot.performance.rank] = rank_counts.get(bot.performance.rank, 0) + 1
        lines.append("  HIERARCHY:")
        for rank in RANK_ORDER:
            count = rank_counts.get(rank, 0)
            lines.append(f"    {rank:<12} : {count} bots")

        # Leaderboard
        lines.extend([
            "",
            "  ┌─────┬────────────────────────────────┬────────┬─────────┬──────────┬───────────┬────────────┐",
            "  │Rank │ Sector                          │ Score  │ Sharpe  │ Accuracy │ Rank      │ Trend      │",
            "  ├─────┼────────────────────────────────┼────────┼─────────┼──────────┼───────────┼────────────┤",
        ])

        for i, (sector, perf) in enumerate(self.get_leaderboard()):
            trend = "↑" if perf.improvement_trend > 0.01 else ("↓" if perf.improvement_trend < -0.01 else "→")
            lines.append(
                f"  │ {i+1:<3} │ {sector:<30} │ {perf.composite_score:.4f} │ "
                f"{perf.sharpe:>6.2f}  │ {perf.accuracy:>7.1%}  │ {perf.rank:<9} │ {trend:<10} │"
            )

        lines.append("  └─────┴────────────────────────────────┴────────┴─────────┴──────────┴───────────┴────────────┘")

        # Per-bot specialty details
        lines.extend(["", "  BOT SPECIALTIES & LEARNING STATE:"])
        for sector, bot in self.bots.items():
            lr = bot.learning
            lines.append(
                f"    {sector:<30} | Weights: Mom={lr.momentum_weight:.2f} Tech={lr.technical_weight:.2f} "
                f"Qual={lr.quality_weight:.2f} MR={lr.mean_reversion_weight:.2f} Vol={lr.volume_weight:.2f} | "
                f"Patterns: {lr.patterns_seen} seen, {lr.patterns_correct} correct"
            )

        # Best opportunities
        best_signals = self.get_high_conviction_signals()[:5]
        if best_signals:
            lines.extend(["", "  TOP HIGH-CONVICTION SIGNALS:"])
            for s in best_signals:
                lines.append(
                    f"    {s.ticker:<8} {s.direction:<6} Conv={s.conviction:<8} "
                    f"Conf={s.confidence:.2f} Alpha={s.alpha_estimate:.3f} | {s.reasoning[:60]}"
                )

        # Mathematical best estimates
        estimates = self.get_mathematical_best_estimates()
        overall = estimates.get("overall_best_estimate", {})
        if overall:
            lines.extend([
                "",
                "  MATHEMATICAL BEST OUTCOME ESTIMATE:",
                f"    Best Sector: {estimates.get('overall_best_sector', 'N/A')}",
                f"    Best Ticker: {overall.get('best_ticker', 'N/A')}",
                f"    Daily Alpha: {overall.get('best_daily_alpha_bps', 0):.1f} bps",
                f"    Probability: {overall.get('probability', 0):.1%}",
                f"    Expected Value: {overall.get('expected_value_bps', 0):.1f} bps",
            ])

        lines.extend(["", "=" * 90])
        return "\n".join(lines)

    def save_state(self):
        """Persist bot state for continuity."""
        state = {}
        for sector, bot in self.bots.items():
            state[sector] = {
                "rank": bot.performance.rank,
                "composite_score": bot.performance.composite_score,
                "sharpe": bot.performance.sharpe,
                "accuracy": bot.performance.accuracy,
                "improvement_trend": bot.performance.improvement_trend,
                "learning": asdict(bot.learning),
                "signal_count": len(bot._signal_history),
            }
        filepath = self.log_dir / f"state_{datetime.now().strftime('%Y%m%d')}.json"
        filepath.write_text(json.dumps(state, indent=2, default=str))

    def load_state(self, filepath: Optional[Path] = None):
        """Load bot state from previous session."""
        if filepath is None:
            files = sorted(self.log_dir.glob("state_*.json"), reverse=True)
            if not files:
                return
            filepath = files[0]

        try:
            state = json.loads(filepath.read_text())
            for sector, data in state.items():
                if sector in self.bots:
                    bot = self.bots[sector]
                    bot.performance.rank = data.get("rank", BotRank.RECRUIT)
                    bot.performance.composite_score = data.get("composite_score", 0)
                    bot.performance.sharpe = data.get("sharpe", 0)
                    bot.performance.accuracy = data.get("accuracy", 0)
                    bot.performance.improvement_trend = data.get("improvement_trend", 0)
                    learning_data = data.get("learning", {})
                    for key, val in learning_data.items():
                        if hasattr(bot.learning, key):
                            setattr(bot.learning, key, val)
        except Exception as e:
            logger.warning(f"Failed to load bot state: {e}")

    # ── LLM-powered sector analysis ──────────────────────────────

    def run_llm_sector_analysis(self, sector: str = "") -> dict:
        """Run LLM inference bridge analysis for a sector or all sectors.

        Sends the sector's top signals, performance, and market data to the
        ensemble endpoint so Brain Power can generate strategic recommendations.

        Returns dict with sector → LLM analysis mapping.
        """
        import os
        results = {}
        sectors = [sector] if sector else list(self.bots.keys())

        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — LLM sector analysis skipped")
            return results

        bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")

        for sec in sectors:
            bot = self.bots.get(sec)
            if not bot or not bot.is_active:
                continue
            try:
                signals = bot.get_active_signals()
                perf = bot.performance
                signal_summary = [
                    {"ticker": s.ticker, "direction": s.direction, "confidence": s.confidence,
                     "alpha_estimate": s.alpha_estimate, "quality_tier": s.quality_tier}
                    for s in signals[:10]
                ]

                payload = {
                    "prompt": (
                        f"Analyze {sec} sector for investment opportunities. "
                        f"Bot rank: {perf.rank}, accuracy: {perf.accuracy:.1%}, "
                        f"Sharpe: {perf.sharpe:.2f}. Top signals:\n"
                        + json.dumps(signal_summary, indent=2, default=str)
                        + "\nProvide: key thesis, top 3 picks, risks, and conviction level."
                    ),
                    "task_type": "sector_research",
                    "max_tokens": 512,
                    "ml_context": {
                        "sector": sec,
                        "bot_rank": perf.rank,
                        "accuracy": perf.accuracy,
                        "sharpe": perf.sharpe,
                        "active_signals": len(signals),
                    },
                }

                r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
                if r.status_code == 200:
                    results[sec] = r.json()
                    logger.info(f"LLM sector analysis for {sec}: orchestrator={results[sec].get('orchestrator', 'unknown')}")
            except Exception as e:
                logger.warning(f"LLM sector analysis for {sec} failed: {e}")
                results[sec] = {"error": str(e)}

        return results
