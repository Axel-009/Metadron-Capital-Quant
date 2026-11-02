"""Metadron Capital — News Intelligence Engine.

Two-source news pipeline:
  1. PRIMARY: newsfilter.io real-time WebSocket (10,000+ sources: Reuters,
     Bloomberg, WSJ, SEC, Seeking Alpha). Streams via socket.io.
  2. FALLBACK: OpenBB news (Tiingo, Benzinga, FMP) — used when newsfilter
     is unavailable or for per-ticker company news.

Fits in system architecture:
    L1 Data (newsfilter.io + OpenBB news)
    → L2 NewsEngine (sentiment scoring + urgency categorization)
    → EventDrivenEngine (catalyst detection)
    → WRAP tab LiveNewsFeed (real-time display)
    → MacroEngine (headline sentiment as macro signal)

Usage:
    from engine.signals.news_engine import NewsEngine
    engine = NewsEngine()
    feed = engine.get_live_feed(limit=20)
    ticker_news = engine.get_ticker_news("AAPL", limit=10)
"""

import logging
import re
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

logger = logging.getLogger("metadron.scan.news_engine")

# ─── Sentiment keywords ────────────────────────────────────

BULLISH_KEYWORDS = {
    "beats", "surges", "soars", "rally", "record", "upgrade", "raises",
    "bullish", "outperform", "exceeds", "accelerates", "strong", "boom",
    "buyback", "dividend", "growth", "momentum", "breakout", "deal",
    "approval", "launch", "expands", "profit", "gains", "optimism",
    "dovish", "cut", "easing", "stimulus", "recovery",
}

BEARISH_KEYWORDS = {
    "misses", "drops", "plunges", "crash", "warning", "downgrade",
    "bearish", "underperform", "slows", "declines", "weak", "loss",
    "lawsuit", "recall", "investigation", "default", "bankruptcy",
    "layoffs", "recession", "tariff", "sanctions", "hawkish", "hike",
    "inflation", "overvalued", "sell-off", "selloff", "concern", "risk",
}

BREAKING_KEYWORDS = {
    "breaking", "just in", "alert", "urgent", "emergency", "flash",
    "fed decision", "rate decision", "surprise", "shock", "crash",
    "halt", "suspended", "war", "invasion",
}

HOT_KEYWORDS = {
    "ai", "nvidia", "gpu", "semiconductor", "apple", "microsoft",
    "earnings", "ipo", "merger", "acquisition", "billion", "trillion",
    "crypto", "bitcoin", "tesla", "openai", "copilot",
}


@dataclass
class NewsItem:
    """A single processed news item."""
    id: str
    timestamp: str
    source: str
    headline: str
    tickers: List[str] = field(default_factory=list)
    sentiment: str = "neutral"  # bullish, bearish, neutral
    category: str = "top"       # hot, top, breaking
    sentiment_score: float = 0.0  # -1.0 to +1.0
    url: str = ""
    raw_date: str = ""


class NewsFilterBridge:
    """Bridge to newsfilter.io real-time WebSocket API.

    Connects via socket.io to stream financial news from 10,000+ sources
    (Reuters, Bloomberg, WSJ, SEC, Seeking Alpha, etc.).
    Articles arrive pre-mapped to company tickers.

    Source: /news-engine/ folder (npm package: realtime-newsapi)
    """

    NEWSFILTER_URL = "https://api.newsfilter.io"

    def __init__(self):
        self._connected = False
        self._articles: deque = deque(maxlen=200)
        self._sio = None

    def connect(self):
        """Connect to newsfilter.io WebSocket."""
        try:
            import socketio
            self._sio = socketio.Client()

            @self._sio.on("connect")
            def on_connect():
                self._connected = True
                self._sio.emit("action", {"type": "subscribe", "filterId": "all"})
                logger.info("NewsFilter.io connected — streaming live news")

            @self._sio.on("articles")
            def on_articles(data):
                articles = data.get("articles", []) if isinstance(data, dict) else []
                for a in articles:
                    self._articles.appendleft(a)

            @self._sio.on("disconnect")
            def on_disconnect():
                self._connected = False
                logger.warning("NewsFilter.io disconnected")

            self._sio.connect(self.NEWSFILTER_URL, transports=["websocket"])
        except ImportError:
            logger.info("python-socketio not installed — newsfilter.io bridge unavailable (pip install python-socketio[client])")
        except Exception as e:
            logger.warning(f"NewsFilter.io connection failed: {e}")

    def disconnect(self):
        if self._sio and self._connected:
            self._sio.disconnect()

    def get_articles(self, limit: int = 30) -> list:
        """Get buffered articles from the WebSocket stream."""
        return list(self._articles)[:limit]

    @property
    def is_connected(self) -> bool:
        return self._connected


class NewsEngine:
    """Live news intelligence engine.

    Primary: newsfilter.io WebSocket (10,000+ sources, real-time)
    Fallback: OpenBB news providers (see API KEY STATUS below)

    Fetches news, scores sentiment, categorizes urgency,
    and maintains a rolling feed for the WRAP tab.

    API KEY STATUS (as of deployment):
        - FMP_API_KEY:      CONFIGURED — primary data provider, used for quotes/fundamentals
        - Tiingo:           NOT CONFIGURED — but included in OpenBB free tier, so some
                            basic news may still flow through OpenBB's Tiingo provider
                            without an explicit key. Acceptable as a fallback source.
        - Benzinga:         NOT CONFIGURED — requires paid API key. OpenBB news calls
                            with provider='benzinga' will silently return empty.
        - newsfilter.io:    OPTIONAL — WebSocket primary. Works without API key for
                            basic feed; premium key unlocks full real-time stream.

    Data flow (single source of truth for all news on the platform):
        newsfilter.io (primary, real-time)
            ↓ fallback
        OpenBB get_world_news() / get_company_news() (FMP default)
            ↓ consumed by
        /macro/news, /signals/news/live, EventDrivenEngine, CVREngine
    """

    def __init__(self):
        self._feed: deque[NewsItem] = deque(maxlen=100)
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(seconds=60)  # Min 60s between fetches
        self._seen_ids: set = set()
        self._newsfilter: Optional[NewsFilterBridge] = None

        # Try to connect to newsfilter.io
        try:
            self._newsfilter = NewsFilterBridge()
            self._newsfilter.connect()
        except Exception:
            pass

    def _score_sentiment(self, headline: str) -> tuple[str, float]:
        """Score headline sentiment using keyword matching.

        Returns (label, score) where score is -1.0 to +1.0.
        """
        words = set(re.findall(r'\b\w+\b', headline.lower()))
        bull = len(words & BULLISH_KEYWORDS)
        bear = len(words & BEARISH_KEYWORDS)

        total = bull + bear
        if total == 0:
            return "neutral", 0.0

        score = (bull - bear) / total
        if score > 0.2:
            return "bullish", min(score, 1.0)
        elif score < -0.2:
            return "bearish", max(score, -1.0)
        return "neutral", score

    def _categorize(self, headline: str) -> str:
        """Categorize headline urgency: breaking > hot > top."""
        lower = headline.lower()
        if any(kw in lower for kw in BREAKING_KEYWORDS):
            return "breaking"
        if any(kw in lower for kw in HOT_KEYWORDS):
            return "hot"
        return "top"

    def _extract_tickers(self, headline: str, symbols: Optional[List[str]] = None) -> List[str]:
        """Extract ticker symbols from headline text."""
        tickers = []
        # Match $TICKER pattern
        tickers.extend(re.findall(r'\$([A-Z]{1,5})\b', headline))
        # Match known symbols in uppercase words
        words = re.findall(r'\b([A-Z]{2,5})\b', headline)
        known = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM",
                 "BAC", "GS", "XOM", "UNH", "V", "SPY", "QQQ", "IWM", "TLT",
                 "GLD", "VIX", "DIA", "AVGO", "CRM", "NFLX", "AMD", "INTC",
                 "PFE", "MRK", "JNJ", "KO", "PEP", "WMT", "HD", "BA", "CAT"}
        for w in words:
            if w in known and w not in tickers:
                tickers.append(w)
        if symbols:
            for s in symbols:
                if s.upper() not in tickers:
                    tickers.append(s.upper())
        return tickers[:5]

    def _make_id(self, headline: str, source: str) -> str:
        """Generate deterministic ID for dedup."""
        return hashlib.md5(f"{headline[:80]}:{source}".encode()).hexdigest()[:12]

    def _fetch_from_newsfilter(self) -> List[NewsItem]:
        """Fetch from newsfilter.io WebSocket buffer (primary source)."""
        if not self._newsfilter or not self._newsfilter.is_connected:
            return []

        items = []
        for article in self._newsfilter.get_articles(limit=30):
            headline = article.get("title", "")
            if not headline:
                continue
            source = article.get("source", article.get("provider", "NewsFilter"))
            date_str = article.get("publishedAt", article.get("date", ""))
            symbols = article.get("symbols", article.get("tickers", []))
            if isinstance(symbols, str):
                symbols = symbols.split(",")
            url = article.get("url", article.get("link", ""))

            nid = self._make_id(headline, source)
            if nid in self._seen_ids:
                continue
            self._seen_ids.add(nid)

            sentiment, score = self._score_sentiment(headline)
            category = self._categorize(headline)
            tickers = self._extract_tickers(headline, symbols)

            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now()
                delta = datetime.now() - dt.replace(tzinfo=None)
                ts = "just now" if delta.total_seconds() < 60 else f"{int(delta.total_seconds() / 60)}m ago" if delta.total_seconds() < 3600 else f"{int(delta.total_seconds() / 3600)}h ago"
            except Exception:
                ts = ""

            items.append(NewsItem(
                id=nid, timestamp=ts, source=source, headline=headline,
                tickers=tickers, sentiment=sentiment, sentiment_score=score,
                category=category, url=url, raw_date=date_str,
            ))

        if items:
            logger.info(f"NewsFilter.io: {len(items)} new articles")
        return items

    def _fetch_from_openbb(self, limit: int = 30) -> List[NewsItem]:
        """Fetch fresh news from OpenBB (fallback source)."""
        items = []
        try:
            from engine.data.openbb_data import get_world_news, get_company_news

            # World news
            df = get_world_news(limit=limit)
            if hasattr(df, "iterrows"):
                for _, row in df.iterrows():
                    headline = str(row.get("title", row.get("headline", "")))
                    if not headline:
                        continue
                    source = str(row.get("source", row.get("publisher", "OpenBB")))
                    date_str = str(row.get("date", row.get("published", "")))
                    symbols = []
                    if "symbols" in row and row["symbols"]:
                        symbols = row["symbols"] if isinstance(row["symbols"], list) else str(row["symbols"]).split(",")
                    url = str(row.get("url", row.get("link", "")))

                    nid = self._make_id(headline, source)
                    if nid in self._seen_ids:
                        continue
                    self._seen_ids.add(nid)

                    sentiment, score = self._score_sentiment(headline)
                    category = self._categorize(headline)
                    tickers = self._extract_tickers(headline, symbols)

                    # Format timestamp
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now()
                        delta = datetime.now() - dt.replace(tzinfo=None)
                        if delta.total_seconds() < 60:
                            ts = "just now"
                        elif delta.total_seconds() < 3600:
                            ts = f"{int(delta.total_seconds() / 60)}m ago"
                        elif delta.total_seconds() < 86400:
                            ts = f"{int(delta.total_seconds() / 3600)}h ago"
                        else:
                            ts = f"{int(delta.days)}d ago"
                    except Exception:
                        ts = date_str[:16] if date_str else ""

                    items.append(NewsItem(
                        id=nid,
                        timestamp=ts,
                        source=source,
                        headline=headline,
                        tickers=tickers,
                        sentiment=sentiment,
                        sentiment_score=score,
                        category=category,
                        url=url,
                        raw_date=date_str,
                    ))

            # Company news for top portfolio tickers
            for ticker in ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]:
                try:
                    cdf = get_company_news(ticker, limit=3)
                    if hasattr(cdf, "iterrows"):
                        for _, row in cdf.iterrows():
                            headline = str(row.get("title", row.get("headline", "")))
                            if not headline:
                                continue
                            nid = self._make_id(headline, ticker)
                            if nid in self._seen_ids:
                                continue
                            self._seen_ids.add(nid)

                            sentiment, score = self._score_sentiment(headline)
                            category = self._categorize(headline)

                            items.append(NewsItem(
                                id=nid,
                                timestamp="",
                                source=str(row.get("source", row.get("publisher", ""))),
                                headline=headline,
                                tickers=[ticker],
                                sentiment=sentiment,
                                sentiment_score=score,
                                category=category,
                            ))
                except Exception:
                    continue

        except ImportError:
            logger.warning("OpenBB not available for news fetching")
        except Exception as e:
            logger.error(f"News fetch error: {e}")

        return items

    def refresh(self, limit: int = 30) -> int:
        """Refresh the news feed.

        Priority chain:
            1. newsfilter.io WebSocket (real-time, 10k+ sources)
            2. OpenBB get_world_news + get_company_news (FMP provider)
            3. Market-state fallback — generates headlines from live FMP
               quote data (SPY, VIX, DXY, US10Y) so the feed is never empty

        Returns count of new items added.
        """
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return 0

        # Primary: newsfilter.io real-time stream
        items = self._fetch_from_newsfilter()

        # Fallback 1: OpenBB/FMP news if newsfilter has no data
        if not items:
            items = self._fetch_from_openbb(limit=limit)

        # Fallback 2: generate market-state headlines from live quote data
        # so the news feed is NEVER empty when the engine is running
        if not items and len(self._feed) == 0:
            items = self._generate_market_state_headlines()

        new_count = 0
        for item in items:
            if item.id not in {i.id for i in self._feed}:
                self._feed.appendleft(item)
                new_count += 1

        self._last_fetch = now
        if new_count:
            logger.info(f"News refresh: {new_count} new items ({len(self._feed)} total)")
        return new_count

    def _generate_market_state_headlines(self) -> List[NewsItem]:
        """Last-resort fallback: build headlines from live FMP market data.

        When both newsfilter.io and OpenBB news are unavailable (no API keys,
        network issues, etc.), we still show the user something real by
        fetching SPY/VIX/DXY quotes and generating factual market-state
        headlines from them.
        """
        items = []
        try:
            from engine.data.openbb_data import get_quote
            quotes = get_quote(["SPY", "QQQ"])
            if not quotes:
                return items

            now_str = "just now"
            for q in quotes:
                ticker = q.get("symbol", "")
                price = q.get("price", 0)
                change = q.get("change", q.get("changesPercentage", 0))
                if not ticker or not price:
                    continue

                direction = "up" if change and change > 0 else "down"
                pct = abs(change) if isinstance(change, (int, float)) else 0
                headline = f"{ticker} trading at ${price:.2f}, {direction} {pct:.2f}% — FMP live quote"
                sentiment = "bullish" if direction == "up" else "bearish"

                items.append(NewsItem(
                    id=self._make_id(headline, "FMP"),
                    timestamp=now_str,
                    source="FMP Market Data",
                    headline=headline,
                    tickers=[ticker],
                    sentiment=sentiment,
                    sentiment_score=0.6 if sentiment == "bullish" else -0.6,
                    category="top",
                ))

            if items:
                logger.info(f"News fallback: generated {len(items)} market-state headlines from FMP quotes")
        except Exception as e:
            logger.warning(f"Market-state headline generation failed: {e}")

        return items

    def get_live_feed(self, limit: int = 20) -> List[dict]:
        """Get the live news feed for the WRAP tab.

        Auto-refreshes from OpenBB if stale. Returns list of dicts
        matching the frontend LiveNewsItem interface.
        """
        self.refresh()

        items = []
        for item in list(self._feed)[:limit]:
            items.append({
                "id": item.id,
                "timestamp": item.timestamp,
                "source": item.source,
                "headline": item.headline,
                "tickers": item.tickers,
                "sentiment": item.sentiment,
                "category": item.category,
                "sentiment_score": item.sentiment_score,
            })
        return items

    def get_ticker_news(self, ticker: str, limit: int = 10) -> List[dict]:
        """Get news for a specific ticker."""
        self.refresh()
        items = [i for i in self._feed if ticker.upper() in i.tickers]
        return [
            {
                "id": i.id, "timestamp": i.timestamp, "source": i.source,
                "headline": i.headline, "tickers": i.tickers,
                "sentiment": i.sentiment, "category": i.category,
            }
            for i in items[:limit]
        ]

    # ── MiroMomentum news-triggered simulation ─────────────────

    def run_miro_on_news_tickers(self) -> dict:
        """Run MiroMomentumEngine on tickers flagged by recent news.

        Extracts unique tickers from the live feed, runs agent-based
        market simulation on each, and returns enriched signals that
        combine news sentiment with MiroMomentum direction/confidence.

        Returns dict mapping ticker → {news_sentiment, miro_signal, combined_score}.
        This flows directly into EventDrivenEngine and CVREngine.
        """
        self.refresh()

        # Collect unique tickers from recent news
        news_tickers = set()
        ticker_sentiment = {}
        for item in list(self._feed)[:50]:
            for t in item.tickers:
                news_tickers.add(t)
                # Track aggregate sentiment per ticker
                if t not in ticker_sentiment:
                    ticker_sentiment[t] = {"scores": [], "headlines": 0}
                ticker_sentiment[t]["scores"].append(item.sentiment_score)
                ticker_sentiment[t]["headlines"] += 1

        if not news_tickers:
            return {}

        # Run MiroMomentumEngine on news-flagged tickers
        miro_signals = {}
        try:
            from engine.signals.social_prediction_engine import MiroMomentumEngine
            miro = MiroMomentumEngine(n_simulations=50, simulation_horizon=15)
            miro_results = miro.analyze(tickers=list(news_tickers))
            miro_signals = miro_results
        except Exception as e:
            logger.warning("MiroMomentum news branch unavailable: %s", e)

        # Combine news sentiment + MiroMomentum into enriched signals
        enriched = {}
        for ticker in news_tickers:
            sent = ticker_sentiment.get(ticker, {})
            scores = sent.get("scores", [])
            avg_sentiment = sum(scores) / len(scores) if scores else 0.0

            miro_sig = miro_signals.get(ticker)
            miro_direction = "HOLD"
            miro_confidence = 0.5
            miro_momentum = 0.0
            if miro_sig:
                miro_direction = miro_sig.miro_momentum_signal
                miro_confidence = miro_sig.consensus_strength
                miro_momentum = miro_sig.momentum

            # Combined score: 40% news sentiment + 60% MiroMomentum
            combined = 0.40 * avg_sentiment + 0.60 * (miro_momentum * 10)  # scale momentum

            enriched[ticker] = {
                "ticker": ticker,
                "news_sentiment": round(avg_sentiment, 3),
                "news_headlines": sent.get("headlines", 0),
                "miro_direction": miro_direction,
                "miro_confidence": round(miro_confidence, 3),
                "miro_momentum": round(miro_momentum, 4),
                "combined_score": round(combined, 3),
                "signal": "BUY" if combined > 0.1 else ("SELL" if combined < -0.1 else "HOLD"),
            }

        logger.info(
            "News+MiroMomentum: %d tickers analyzed, %d BUY, %d SELL, %d HOLD",
            len(enriched),
            sum(1 for v in enriched.values() if v["signal"] == "BUY"),
            sum(1 for v in enriched.values() if v["signal"] == "SELL"),
            sum(1 for v in enriched.values() if v["signal"] == "HOLD"),
        )
        return enriched

    def get_sentiment_summary(self) -> dict:
        """Aggregate sentiment across all recent news."""
        if not self._feed:
            return {"bullish": 0, "bearish": 0, "neutral": 0, "avg_score": 0}

        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        total_score = 0
        for item in self._feed:
            counts[item.sentiment] = counts.get(item.sentiment, 0) + 1
            total_score += item.sentiment_score

        return {
            **counts,
            "total": len(self._feed),
            "avg_score": round(total_score / len(self._feed), 3),
        }
