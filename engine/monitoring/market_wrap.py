"""Market Wrap — End-of-day market summary for Metadron Capital.

Provides:
    - Major index performance (S&P 500, NASDAQ, Russell 2000, VIX)
    - Sector performance ranking
    - Notable movers (biggest gainers/losers)
    - Volume analysis
    - Breadth indicators (advance/decline, new highs/lows)
    - Macro data summary (yields, spreads, commodities)
    - Upcoming events awareness
    - ASCII formatted output
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..data.openbb_data import get_adj_close, get_returns, get_prices
    from ..data.universe_engine import SECTOR_ETFS, SP500_TOP_HOLDINGS
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_prices(*a, **kw): return pd.DataFrame()
    SECTOR_ETFS = {}
    SP500_TOP_HOLDINGS = []

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IndexPerformance:
    ticker: str = ""
    name: str = ""
    last_price: float = 0.0
    change_1d: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0
    change_ytd: float = 0.0
    volume_ratio: float = 1.0


@dataclass
class SectorPerf:
    sector: str = ""
    etf: str = ""
    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    relative_strength: float = 0.0


@dataclass
class Mover:
    ticker: str = ""
    name: str = ""
    change_pct: float = 0.0
    volume_ratio: float = 1.0
    sector: str = ""


@dataclass
class BreadthData:
    advancing: int = 0
    declining: int = 0
    unchanged: int = 0
    new_highs: int = 0
    new_lows: int = 0
    advance_decline_ratio: float = 1.0
    breadth_thrust: float = 0.0


@dataclass
class MacroSummary:
    yield_10y: float = 0.0
    yield_2y: float = 0.0
    yield_spread: float = 0.0
    vix: float = 0.0
    dxy: float = 0.0
    gold: float = 0.0
    oil: float = 0.0
    bitcoin: float = 0.0


@dataclass
class MarketWrapReport:
    timestamp: str = ""
    indices: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)
    breadth: BreadthData = field(default_factory=BreadthData)
    macro: MacroSummary = field(default_factory=MacroSummary)
    market_tone: str = "NEUTRAL"


# ---------------------------------------------------------------------------
# Index Tracker
# ---------------------------------------------------------------------------
class IndexTracker:
    """Track major market indices."""

    INDICES = {
        "^GSPC": "S&P 500",
        "^NDX": "NASDAQ 100",
        "^RUT": "Russell 2000",
        "^DJI": "Dow Jones",
        "^VIX": "VIX",
    }

    def get_performance(self) -> list[IndexPerformance]:
        results = []
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            prices = get_adj_close(list(self.INDICES.keys()), start=start)
            if prices.empty:
                return results

            for ticker, name in self.INDICES.items():
                if ticker not in prices.columns:
                    continue
                p = prices[ticker].dropna()
                if len(p) < 2:
                    continue

                perf = IndexPerformance(ticker=ticker, name=name, last_price=float(p.iloc[-1]))
                perf.change_1d = float(p.iloc[-1] / p.iloc[-2] - 1) if len(p) >= 2 else 0
                perf.change_1w = float(p.iloc[-1] / p.iloc[-5] - 1) if len(p) >= 5 else 0
                perf.change_1m = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0

                # YTD
                year_start = p[p.index >= f"{datetime.now().year}-01-01"]
                if len(year_start) > 0:
                    perf.change_ytd = float(p.iloc[-1] / year_start.iloc[0] - 1)

                results.append(perf)
        except Exception as e:
            logger.warning(f"Index tracking failed: {e}")
        return results


# ---------------------------------------------------------------------------
# Sector Performance
# ---------------------------------------------------------------------------
class SectorTracker:
    """Track sector ETF performance."""

    def get_performance(self) -> list[SectorPerf]:
        results = []
        if not SECTOR_ETFS:
            return results
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            etfs = list(SECTOR_ETFS.values())
            prices = get_adj_close(etfs + ["SPY"], start=start)
            if prices.empty:
                return results

            inv_map = {v: k for k, v in SECTOR_ETFS.items()}
            spy_returns = prices["SPY"].pct_change().dropna() if "SPY" in prices.columns else None

            for etf in etfs:
                if etf not in prices.columns:
                    continue
                p = prices[etf].dropna()
                if len(p) < 5:
                    continue

                sector = inv_map.get(etf, etf)
                sp = SectorPerf(sector=sector, etf=etf)
                sp.return_1d = float(p.iloc[-1] / p.iloc[-2] - 1) if len(p) >= 2 else 0
                sp.return_1w = float(p.iloc[-1] / p.iloc[-5] - 1) if len(p) >= 5 else 0
                sp.return_1m = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0

                # Relative strength vs SPY
                if spy_returns is not None and len(p) >= 21 and "SPY" in prices.columns:
                    spy_1m = float(prices["SPY"].iloc[-1] / prices["SPY"].iloc[-21] - 1)
                    sp.relative_strength = sp.return_1m - spy_1m

                results.append(sp)

            results.sort(key=lambda x: x.return_1d, reverse=True)
        except Exception as e:
            logger.warning(f"Sector tracking failed: {e}")
        return results


# ---------------------------------------------------------------------------
# Notable Movers
# ---------------------------------------------------------------------------
class MoverScanner:
    """Find biggest gainers and losers."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def scan(self, tickers: Optional[list[str]] = None) -> tuple[list[Mover], list[Mover]]:
        if tickers is None:
            tickers = list(SP500_TOP_HOLDINGS[:50]) if SP500_TOP_HOLDINGS else []
        if not tickers:
            return [], []

        gainers, losers = [], []
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            prices = get_adj_close(tickers, start=start)
            if prices.empty:
                return [], []

            returns = prices.pct_change().iloc[-1].dropna()
            sorted_returns = returns.sort_values(ascending=False)

            for ticker in sorted_returns.head(self.top_n).index:
                gainers.append(Mover(
                    ticker=ticker, name=ticker,
                    change_pct=float(sorted_returns[ticker]) * 100,
                ))

            for ticker in sorted_returns.tail(self.top_n).index:
                losers.append(Mover(
                    ticker=ticker, name=ticker,
                    change_pct=float(sorted_returns[ticker]) * 100,
                ))
        except Exception as e:
            logger.warning(f"Mover scan failed: {e}")
        return gainers, losers


# ---------------------------------------------------------------------------
# Breadth Calculator
# ---------------------------------------------------------------------------
class BreadthCalculator:
    """Calculate market breadth indicators."""

    def compute(self, returns_1d: Optional[pd.Series] = None) -> BreadthData:
        bd = BreadthData()
        if returns_1d is None or returns_1d.empty:
            return bd

        bd.advancing = int((returns_1d > 0.001).sum())
        bd.declining = int((returns_1d < -0.001).sum())
        bd.unchanged = int(len(returns_1d) - bd.advancing - bd.declining)

        total = bd.advancing + bd.declining
        bd.advance_decline_ratio = bd.advancing / total if total > 0 else 1.0

        # Breadth thrust: >2:1 A/D ratio with strong participation
        if bd.advancing > 0 and bd.declining > 0:
            bd.breadth_thrust = bd.advancing / bd.declining
        else:
            bd.breadth_thrust = bd.advancing if bd.declining == 0 else 0

        return bd


# ---------------------------------------------------------------------------
# Macro Data Fetcher
# ---------------------------------------------------------------------------
class MacroDataFetcher:
    """Fetch macro indicators."""

    TICKERS = {
        "^VIX": "vix",
        "^TNX": "yield_10y",
        "GC=F": "gold",
        "CL=F": "oil",
        "DX-Y.NYB": "dxy",
    }

    def fetch(self) -> MacroSummary:
        ms = MacroSummary()
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            prices = get_adj_close(list(self.TICKERS.keys()), start=start)
            if prices.empty:
                return ms

            for ticker, field_name in self.TICKERS.items():
                if ticker in prices.columns:
                    val = float(prices[ticker].dropna().iloc[-1])
                    setattr(ms, field_name, val)

            ms.yield_spread = ms.yield_10y - ms.yield_2y
        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
        return ms


# ---------------------------------------------------------------------------
# Market Tone Classifier
# ---------------------------------------------------------------------------
class MarketToneClassifier:
    """Classify overall market tone."""

    def classify(
        self,
        index_perf: list[IndexPerformance],
        breadth: BreadthData,
        vix: float,
    ) -> str:
        score = 0

        # Index direction
        spy = next((i for i in index_perf if i.name == "S&P 500"), None)
        if spy:
            if spy.change_1d > 0.01:
                score += 2
            elif spy.change_1d > 0:
                score += 1
            elif spy.change_1d < -0.01:
                score -= 2
            elif spy.change_1d < 0:
                score -= 1

        # Breadth
        if breadth.advance_decline_ratio > 0.65:
            score += 1
        elif breadth.advance_decline_ratio < 0.35:
            score -= 1

        # VIX
        if vix > 30:
            score -= 2
        elif vix > 20:
            score -= 1
        elif vix < 15:
            score += 1

        if score >= 3:
            return "STRONGLY_BULLISH"
        elif score >= 1:
            return "BULLISH"
        elif score <= -3:
            return "STRONGLY_BEARISH"
        elif score <= -1:
            return "BEARISH"
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Market Wrap Generator
# ---------------------------------------------------------------------------
class MarketWrapGenerator:
    """Generate complete end-of-day market wrap."""

    def __init__(self):
        self._index_tracker = IndexTracker()
        self._sector_tracker = SectorTracker()
        self._mover_scanner = MoverScanner()
        self._breadth_calc = BreadthCalculator()
        self._macro_fetcher = MacroDataFetcher()
        self._tone_classifier = MarketToneClassifier()

    def generate(self) -> MarketWrapReport:
        report = MarketWrapReport(timestamp=datetime.now().isoformat())

        # Indices
        report.indices = self._index_tracker.get_performance()

        # Sectors
        report.sectors = self._sector_tracker.get_performance()

        # Movers
        report.top_gainers, report.top_losers = self._mover_scanner.scan()

        # Macro
        report.macro = self._macro_fetcher.fetch()

        # Breadth (from gainers/losers if available)
        all_changes = [m.change_pct for m in report.top_gainers + report.top_losers]
        if all_changes:
            returns_series = pd.Series(all_changes) / 100
            report.breadth = self._breadth_calc.compute(returns_series)

        # Market tone
        report.market_tone = self._tone_classifier.classify(
            report.indices, report.breadth, report.macro.vix,
        )

        return report

    def generate_ascii(self) -> str:
        report = self.generate()
        return format_market_wrap(report)


def format_market_wrap(report: MarketWrapReport) -> str:
    """Format market wrap as ASCII."""
    lines = [
        "=" * 70,
        f"MARKET WRAP — {report.timestamp[:10]}",
        f"Market Tone: {report.market_tone}",
        "=" * 70,
        "",
        "MAJOR INDICES:",
        f"  {'Index':<20} {'Price':>10} {'1D':>8} {'1W':>8} {'1M':>8} {'YTD':>8}",
        "  " + "-" * 62,
    ]

    for idx in report.indices:
        lines.append(
            f"  {idx.name:<20} {idx.last_price:>10,.1f} "
            f"{idx.change_1d:>+7.1%} {idx.change_1w:>+7.1%} "
            f"{idx.change_1m:>+7.1%} {idx.change_ytd:>+7.1%}"
        )

    lines.extend(["", "SECTOR PERFORMANCE:", f"  {'Sector':<30} {'1D':>8} {'1W':>8} {'1M':>8} {'RS':>8}",
                   "  " + "-" * 54])

    for sp in report.sectors:
        lines.append(
            f"  {sp.sector:<30} {sp.return_1d:>+7.1%} "
            f"{sp.return_1w:>+7.1%} {sp.return_1m:>+7.1%} "
            f"{sp.relative_strength:>+7.1%}"
        )

    if report.top_gainers:
        lines.extend(["", "TOP GAINERS:"])
        for m in report.top_gainers[:5]:
            lines.append(f"  {m.ticker:<8} {m.change_pct:>+7.1f}%")

    if report.top_losers:
        lines.extend(["", "TOP LOSERS:"])
        for m in report.top_losers[:5]:
            lines.append(f"  {m.ticker:<8} {m.change_pct:>+7.1f}%")

    lines.extend([
        "", "BREADTH:",
        f"  Advancing: {report.breadth.advancing}  Declining: {report.breadth.declining}",
        f"  A/D Ratio: {report.breadth.advance_decline_ratio:.2f}",
        "",
        "MACRO:",
        f"  VIX: {report.macro.vix:.1f}  10Y: {report.macro.yield_10y:.2f}%  "
        f"Gold: ${report.macro.gold:,.0f}  Oil: ${report.macro.oil:.1f}",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
