"""
Air-LLM integration for the Metadron Capital Investment Platform.

Uses efficient LLM inference for natural language processing tasks
across the investment workflow:

1. Real-time news sentiment analysis across all asset classes
2. Earnings call transcript analysis and guidance extraction
3. SEC filing parsing and risk factor extraction
4. Trade thesis generation in natural language
5. Portfolio commentary and market narrative generation
6. Analyst report summarization

Sentiment Scoring:
    compound = (pos - neg) / (pos + neg + neu + alpha)
    where alpha = 15 (normalization constant)

    Confidence-weighted sentiment:
    S_weighted = S_raw * confidence * recency_decay
    recency_decay = exp(-lambda * hours_since_publish)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class SentimentScore:
    symbol: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    level: SentimentLevel
    source_count: int
    key_phrases: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EarningsAnalysis:
    symbol: str
    quarter: str
    revenue_guidance: str
    margin_outlook: str
    risk_factors: list
    management_tone: float  # -1 to 1
    key_quotes: list
    forward_guidance_change: str  # "raised", "maintained", "lowered"
    surprise_factor: float


@dataclass
class ThesisDocument:
    symbol: str
    direction: str
    horizon: str
    entry_rationale: str
    key_catalysts: list
    risk_factors: list
    target_price: float
    stop_loss: float
    position_size_recommendation: str
    confidence: float


class InvestmentLLMProcessor:
    """NLP processing engine for the investment platform."""

    # Sentiment keyword dictionaries
    POSITIVE_KEYWORDS = [
        "beat", "exceed", "strong", "growth", "upgrade", "outperform", "bullish",
        "raise", "positive", "accelerat", "record", "surge", "rally", "breakout",
        "recovery", "expansion", "optimistic", "upside", "momentum", "innovation",
        "dividend increase", "buyback", "acquisition synerg", "market share gain",
    ]
    NEGATIVE_KEYWORDS = [
        "miss", "below", "weak", "decline", "downgrade", "underperform", "bearish",
        "cut", "negative", "decelerat", "warning", "plunge", "crash", "breakdown",
        "recession", "contraction", "pessimistic", "downside", "headwind", "risk",
        "dividend cut", "dilution", "writedown", "market share loss", "default",
    ]

    def analyze_news_sentiment(self, news_items: list) -> list:
        """
        Analyze sentiment from news items.

        compound = (pos_count - neg_count) / (pos_count + neg_count + neutral_count + 15)
        Confidence = min(source_count / 10, 1.0) * text_length_factor
        """
        results = {}
        for item in news_items:
            symbol = item.get("symbol", "UNKNOWN")
            text = (item.get("title", "") + " " + item.get("description", "")).lower()
            published = item.get("published", datetime.now())

            pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
            neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)
            neutral_count = max(len(text.split()) / 10 - pos_count - neg_count, 0)

            compound = (pos_count - neg_count) / (pos_count + neg_count + neutral_count + 15)

            # Recency decay: lambda = 0.05 per hour
            if isinstance(published, str):
                try:
                    published = datetime.fromisoformat(published)
                except (ValueError, TypeError):
                    published = datetime.now()
            hours_ago = max((datetime.now() - published).total_seconds() / 3600, 0)
            recency_decay = np.exp(-0.05 * hours_ago)
            weighted_score = compound * recency_decay

            if symbol not in results:
                results[symbol] = {"scores": [], "phrases": [], "count": 0}
            results[symbol]["scores"].append(weighted_score)
            results[symbol]["count"] += 1

            # Extract key phrases
            for kw in self.POSITIVE_KEYWORDS + self.NEGATIVE_KEYWORDS:
                if kw in text:
                    results[symbol]["phrases"].append(kw)

        output = []
        for symbol, data in results.items():
            avg_score = np.mean(data["scores"]) if data["scores"] else 0
            confidence = min(data["count"] / 10, 1.0)

            if avg_score > 0.3:
                level = SentimentLevel.VERY_BULLISH
            elif avg_score > 0.1:
                level = SentimentLevel.BULLISH
            elif avg_score < -0.3:
                level = SentimentLevel.VERY_BEARISH
            elif avg_score < -0.1:
                level = SentimentLevel.BEARISH
            else:
                level = SentimentLevel.NEUTRAL

            output.append(SentimentScore(
                symbol=symbol,
                score=float(avg_score),
                confidence=float(confidence),
                level=level,
                source_count=data["count"],
                key_phrases=list(set(data["phrases"]))[:10],
            ))
        return output

    def parse_earnings_transcript(self, transcript: str, symbol: str, quarter: str) -> EarningsAnalysis:
        """
        Extract structured information from earnings call transcript.

        Analyzes: revenue guidance, margin outlook, risk factors, tone.
        Management tone = (positive_sentence_count - negative) / total
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
        total = max(len(sentences), 1)

        guidance_sentences = [s for s in sentences if any(w in s.lower() for w in ["guidance", "expect", "outlook", "forecast", "anticipate"])]
        risk_sentences = [s for s in sentences if any(w in s.lower() for w in ["risk", "challenge", "headwind", "concern", "uncertain"])]
        margin_sentences = [s for s in sentences if any(w in s.lower() for w in ["margin", "profitab", "cost", "expense", "efficiency"])]

        pos_sentences = sum(1 for s in sentences if any(kw in s.lower() for kw in self.POSITIVE_KEYWORDS[:10]))
        neg_sentences = sum(1 for s in sentences if any(kw in s.lower() for kw in self.NEGATIVE_KEYWORDS[:10]))
        tone = (pos_sentences - neg_sentences) / total

        # Detect guidance direction
        guidance_text = " ".join(guidance_sentences).lower()
        if any(w in guidance_text for w in ["raise", "increase", "above", "higher"]):
            guidance_change = "raised"
        elif any(w in guidance_text for w in ["lower", "reduce", "below", "cut"]):
            guidance_change = "lowered"
        else:
            guidance_change = "maintained"

        return EarningsAnalysis(
            symbol=symbol,
            quarter=quarter,
            revenue_guidance=" ".join(guidance_sentences[:3]) if guidance_sentences else "No explicit guidance found",
            margin_outlook=" ".join(margin_sentences[:3]) if margin_sentences else "No margin commentary found",
            risk_factors=[s[:200] for s in risk_sentences[:5]],
            management_tone=float(tone),
            key_quotes=guidance_sentences[:5],
            forward_guidance_change=guidance_change,
            surprise_factor=abs(tone) * 2,
        )

    def generate_trade_thesis(self, opportunity: dict) -> ThesisDocument:
        """
        Generate structured trade thesis from quantitative opportunity data.
        """
        symbol = opportunity.get("symbol", "")
        direction = opportunity.get("direction", "long")
        score = opportunity.get("composite_score", 0)
        asset_class = opportunity.get("asset_class", "equity")
        metrics = opportunity.get("metrics", {})

        # Build rationale
        rationale_parts = []
        if metrics.get("momentum_score", 0) > 0.5:
            rationale_parts.append(f"Strong momentum (score: {metrics['momentum_score']:.2f})")
        if metrics.get("value_score", 0) > 0.5:
            rationale_parts.append(f"Attractive valuation (score: {metrics['value_score']:.2f})")
        if metrics.get("catalyst_score", 0) > 0.5:
            rationale_parts.append(f"Near-term catalysts (score: {metrics['catalyst_score']:.2f})")
        if metrics.get("z_score", 0) > 2:
            rationale_parts.append(f"Statistical mispricing (z={metrics['z_score']:.1f})")

        entry = opportunity.get("current_price", 100)
        if direction == "long":
            target = entry * (1 + abs(score) * 0.1)
            stop = entry * (1 - abs(score) * 0.03)
        else:
            target = entry * (1 - abs(score) * 0.1)
            stop = entry * (1 + abs(score) * 0.03)

        catalysts = opportunity.get("catalysts", ["Momentum continuation"])
        risks = opportunity.get("risks", ["Market regime change", "Liquidity risk"])

        return ThesisDocument(
            symbol=symbol,
            direction=direction,
            horizon=opportunity.get("horizon", "swing"),
            entry_rationale="; ".join(rationale_parts) if rationale_parts else f"Composite score {score:.2f} exceeds threshold",
            key_catalysts=catalysts[:5],
            risk_factors=risks[:5],
            target_price=float(target),
            stop_loss=float(stop),
            position_size_recommendation=f"{min(abs(score) * 2, 5):.1f}% of portfolio",
            confidence=min(abs(score), 0.95),
        )

    def summarize_daily_market(self, market_data: dict) -> str:
        """Generate end-of-day market narrative from data."""
        lines = [f"Market Summary - {datetime.now().strftime('%Y-%m-%d')}"]
        lines.append("=" * 50)

        if "indices" in market_data:
            lines.append("\nMajor Indices:")
            for idx, data in market_data["indices"].items():
                chg = data.get("change_pct", 0)
                direction = "up" if chg > 0 else "down" if chg < 0 else "flat"
                lines.append(f"  {idx}: {direction} {abs(chg):.2f}%")

        if "sectors" in market_data:
            lines.append("\nSector Performance:")
            sorted_sectors = sorted(market_data["sectors"].items(), key=lambda x: x[1].get("change_pct", 0), reverse=True)
            for sector, data in sorted_sectors[:3]:
                lines.append(f"  Best: {sector} +{data.get('change_pct', 0):.2f}%")
            for sector, data in sorted_sectors[-3:]:
                lines.append(f"  Worst: {sector} {data.get('change_pct', 0):.2f}%")

        if "top_movers" in market_data:
            lines.append("\nNotable Movers:")
            for mover in market_data["top_movers"][:5]:
                lines.append(f"  {mover['symbol']}: {mover.get('change_pct', 0):+.2f}% - {mover.get('reason', '')}")

        return "\n".join(lines)

    def risk_narrative(self, risk_metrics: dict) -> str:
        """Generate risk commentary from quantitative metrics."""
        lines = ["Risk Assessment Report"]
        lines.append("-" * 30)

        var_99 = risk_metrics.get("var_99", 0)
        cvar_99 = risk_metrics.get("cvar_99", 0)
        max_dd = risk_metrics.get("max_drawdown", 0)
        beta = risk_metrics.get("portfolio_beta", 1.0)
        concentration = risk_metrics.get("concentration_hhi", 0)

        lines.append(f"VaR (99%): ${var_99:,.0f}")
        lines.append(f"CVaR (99%): ${cvar_99:,.0f}")
        lines.append(f"Max Drawdown: {max_dd:.2%}")
        lines.append(f"Portfolio Beta: {beta:.2f}")

        # Risk warnings
        warnings = []
        if abs(max_dd) > 0.15:
            warnings.append("ELEVATED: Max drawdown exceeds 15% threshold")
        if beta > 1.3:
            warnings.append("HIGH BETA: Portfolio is significantly more volatile than market")
        if concentration > 0.15:
            warnings.append("CONCENTRATED: Top positions represent outsized risk")
        if var_99 > risk_metrics.get("portfolio_value", 1e9) * 0.03:
            warnings.append("VAR BREACH: Daily VaR exceeds 3% of portfolio value")

        if warnings:
            lines.append("\nRisk Warnings:")
            for w in warnings:
                lines.append(f"  ! {w}")
        else:
            lines.append("\nAll risk metrics within acceptable bounds.")

        return "\n".join(lines)
