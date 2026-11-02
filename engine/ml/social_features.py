"""SocialFeatures — Feature engineering for social prediction signals.

Layer 3 module that transforms raw SocialSnapshot data into ML-ready
features for the AlphaOptimizer and ML Vote Ensemble.

Features generated:
    1. Sentiment momentum (EMA of sentiment over simulation windows)
    2. Engagement velocity features (acceleration/deceleration)
    3. Influence-weighted consensus
    4. Narrative regime (trending/reversing/stable)
    5. Cross-topic correlation features
    6. Agent behavior clustering features

Used by:
    - MLVoteEnsemble Tier-6 (social sentiment voter)
    - AlphaOptimizer (additional factor for walk-forward ML)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SocialFeatureVector:
    """ML-ready feature vector from social prediction data."""
    ticker: str = ""

    # Sentiment features
    sentiment_raw: float = 0.0           # raw sentiment [-1, +1]
    sentiment_ema_fast: float = 0.0      # 3-window EMA
    sentiment_ema_slow: float = 0.0      # 10-window EMA
    sentiment_macd: float = 0.0          # fast - slow
    sentiment_zscore: float = 0.0        # z-score vs history

    # Engagement features
    engagement_rate: float = 0.0         # actions per agent per hour
    engagement_acceleration: float = 0.0 # 2nd derivative
    viral_score: float = 0.0            # normalized viral velocity

    # Consensus features
    consensus_strength: float = 0.0      # how aligned agents are [0, 1]
    influence_gini: float = 0.0          # inequality of influence [0, 1]
    stance_entropy: float = 0.0          # entropy of stance distribution

    # Narrative features
    narrative_regime: str = "stable"     # trending/reversing/stable
    narrative_score: float = 0.0         # [-1, +1] direction

    # Composite
    social_alpha: float = 0.0           # composite alpha signal
    social_confidence: float = 0.0       # confidence in signal [0, 1]


class SocialFeatureBuilder:
    """Build ML features from social prediction snapshots.

    Maintains a rolling history of snapshots to compute time-series
    features like momentum, MACD, and z-scores.
    """

    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self._history: list[dict] = []  # list of snapshot dicts

    def add_snapshot(self, snapshot_dict: dict) -> None:
        """Add a social snapshot to history."""
        self._history.append(snapshot_dict)
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

    def build_features(self, ticker: str, current_snapshot: dict) -> SocialFeatureVector:
        """Build feature vector for a specific ticker."""
        fv = SocialFeatureVector(ticker=ticker)

        # Get ticker-specific sentiment
        ticker_signals = current_snapshot.get("ticker_signals", {})
        overall = current_snapshot.get("overall_sentiment", 0.0)
        ticker_sentiment = ticker_signals.get(ticker, overall)

        fv.sentiment_raw = ticker_sentiment

        # Historical sentiment for time-series features
        hist_sentiments = []
        for snap in self._history:
            ts = snap.get("ticker_signals", {})
            hist_sentiments.append(ts.get(ticker, snap.get("overall_sentiment", 0.0)))

        if hist_sentiments:
            fv.sentiment_ema_fast = self._ema(hist_sentiments, span=3)
            fv.sentiment_ema_slow = self._ema(hist_sentiments, span=min(10, len(hist_sentiments)))
            fv.sentiment_macd = fv.sentiment_ema_fast - fv.sentiment_ema_slow

            if len(hist_sentiments) >= 3:
                mean = np.mean(hist_sentiments)
                std = np.std(hist_sentiments)
                fv.sentiment_zscore = (ticker_sentiment - mean) / std if std > 0 else 0.0

        # Engagement features
        total_actions = current_snapshot.get("total_actions", 0)
        total_agents = current_snapshot.get("total_agents", 1)
        sim_hours = max(1, total_actions // max(1, total_agents))
        fv.engagement_rate = total_actions / max(1, total_agents * sim_hours)
        fv.viral_score = min(1.0, current_snapshot.get("viral_velocity", 0.0) / 10.0)

        # Engagement acceleration (compare current to previous)
        if len(self._history) >= 2:
            prev = self._history[-2]
            prev_actions = prev.get("total_actions", 0)
            fv.engagement_acceleration = (total_actions - prev_actions) / max(1, prev_actions)

        # Consensus features
        n_bull = current_snapshot.get("agents_bullish", 0)
        n_bear = current_snapshot.get("agents_bearish", 0)
        n_neut = current_snapshot.get("agents_neutral", 0)
        total = n_bull + n_bear + n_neut

        if total > 0:
            # Consensus = max share
            fv.consensus_strength = max(n_bull, n_bear, n_neut) / total

            # Stance entropy
            probs = np.array([n_bull, n_bear, n_neut], dtype=float) / total
            probs = probs[probs > 0]
            fv.stance_entropy = float(-np.sum(probs * np.log2(probs))) / np.log2(3)  # normalize to [0,1]

        # Influence inequality (Gini from concentration)
        fv.influence_gini = current_snapshot.get("influence_concentration", 0.0)

        # Narrative regime
        trend = current_snapshot.get("sentiment_trend", 0.0)
        dominance = current_snapshot.get("narrative_dominance", 0.0)

        if abs(trend) > 0.2:
            fv.narrative_regime = "trending"
            fv.narrative_score = trend
        elif abs(trend) > 0.1 and dominance > 0.6:
            fv.narrative_regime = "reversing"
            fv.narrative_score = -trend  # counter-trend
        else:
            fv.narrative_regime = "stable"
            fv.narrative_score = 0.0

        # Composite social alpha
        fv.social_alpha = (
            0.35 * fv.sentiment_raw
            + 0.20 * fv.sentiment_macd
            + 0.15 * fv.viral_score * np.sign(fv.sentiment_raw)
            + 0.15 * fv.narrative_score
            + 0.15 * (fv.consensus_strength - 0.5) * np.sign(fv.sentiment_raw)
        )

        fv.social_confidence = current_snapshot.get("sentiment_confidence", 0.0) * (
            0.5 + 0.5 * fv.consensus_strength
        )
        fv.social_confidence = max(0.0, min(1.0, fv.social_confidence))

        return fv

    def build_feature_dict(self, ticker: str, current_snapshot: dict) -> dict:
        """Build features as dict (for integration with DeepTradingFeatures)."""
        fv = self.build_features(ticker, current_snapshot)
        return {
            "social_sentiment_raw": fv.sentiment_raw,
            "social_sentiment_ema_fast": fv.sentiment_ema_fast,
            "social_sentiment_ema_slow": fv.sentiment_ema_slow,
            "social_sentiment_macd": fv.sentiment_macd,
            "social_sentiment_zscore": fv.sentiment_zscore,
            "social_engagement_rate": fv.engagement_rate,
            "social_engagement_accel": fv.engagement_acceleration,
            "social_viral_score": fv.viral_score,
            "social_consensus": fv.consensus_strength,
            "social_influence_gini": fv.influence_gini,
            "social_stance_entropy": fv.stance_entropy,
            "social_narrative_score": fv.narrative_score,
            "social_alpha": fv.social_alpha,
            "social_confidence": fv.social_confidence,
        }

    @staticmethod
    def _ema(values: list[float], span: int) -> float:
        """Exponential moving average of a list."""
        if not values:
            return 0.0
        alpha = 2.0 / (span + 1)
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result
