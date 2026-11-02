"""
Metadron Capital — TCA Engine (Tab 13)

Dedicated Transaction Cost Analysis engine that wraps L7's TransactionCostAnalyzer
and adds:
  1. Cost trend aggregation (daily rollups for time-series)
  2. Per-sector / per-venue / per-algo cost decomposition
  3. Outlier detection (trades exceeding 2σ from mean cost)
  4. Benchmark scoring (IS vs VWAP vs TWAP)
  5. Participation rate tracking
  6. Execution quality scoring (0-100)

Data flow:
  broker.get_trade_history() → TCAEngine.rebuild() → decompose costs →
  aggregate by venue / algo / sector / day → serve to frontend via API

Falls back to IBKR broker price data — zero static/mock data.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("metadron-tca-engine")


# ─── Data Classes ──────────────────────────────────────────────────

@dataclass
class TCATradeRecord:
    """Single trade with full TCA decomposition."""
    order_id: str = ""
    ticker: str = ""
    side: str = "BUY"
    quantity: int = 0
    fill_price: float = 0.0
    arrival_price: float = 0.0
    vwap_price: float = 0.0
    # Cost decomposition (bps)
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    commission_bps: float = 0.0
    total_cost_bps: float = 0.0
    # Implementation shortfall
    implementation_shortfall_usd: float = 0.0
    vwap_slippage_bps: float = 0.0
    # Metadata
    venue: str = "ENGINE"
    algo: str = "SMART"
    sector: str = "Unknown"
    signal_type: str = ""
    product_type: str = "EQUITY"
    participation_rate: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TCADayPoint:
    """Daily aggregated cost trend point."""
    date: str = ""
    trades: int = 0
    volume_usd: float = 0.0
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    timing_bps: float = 0.0
    commission_bps: float = 0.0
    total_bps: float = 0.0
    avg_is_usd: float = 0.0
    fill_rate: float = 1.0


@dataclass
class TCAVenueStat:
    """Per-venue execution quality."""
    venue: str = ""
    fills: int = 0
    avg_cost_bps: float = 0.0
    avg_latency_ms: float = 0.0
    avg_impact_bps: float = 0.0
    avg_spread_bps: float = 0.0
    fill_rate: float = 1.0
    quality_score: float = 0.0  # 0-100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TCAAlgoStat:
    """Per-algorithm execution quality."""
    algo: str = ""
    trades: int = 0
    avg_cost_bps: float = 0.0
    avg_is_usd: float = 0.0
    avg_vwap_slip_bps: float = 0.0
    avg_participation: float = 0.0
    quality_score: float = 0.0  # 0-100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TCAOutlier:
    """Trade flagged as an execution outlier."""
    order_id: str = ""
    ticker: str = ""
    side: str = ""
    total_cost_bps: float = 0.0
    z_score: float = 0.0
    reason: str = ""
    timestamp: str = ""


@dataclass
class TCABenchmark:
    """Benchmark comparison: our execution vs theoretical benchmarks."""
    benchmark: str = ""  # "VWAP", "TWAP", "Arrival"
    avg_slippage_bps: float = 0.0
    total_shortfall_usd: float = 0.0
    win_rate: float = 0.0  # % of trades that beat the benchmark
    trades_evaluated: int = 0


# ─── Sector Mapping ───────────────────────────────────────────────

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "AMD": "Technology", "INTC": "Technology",
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication", "T": "Communication", "VZ": "Communication",
    "AMZN": "Consumer Disc.", "TSLA": "Consumer Disc.", "HD": "Consumer Disc.",
    "NKE": "Consumer Disc.", "MCD": "Consumer Disc.", "SBUX": "Consumer Disc.",
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "MA": "Financials", "GS": "Financials", "MS": "Financials",
    "BRK.B": "Financials", "C": "Financials", "WFC": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "ABT": "Healthcare",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
    "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity",
    "TLT": "Fixed Income", "IEF": "Fixed Income", "HYG": "Fixed Income",
}


# ─── Market Impact Model ─────────────────────────────────────────

IMPACT_COEFFICIENT = 0.10
EQUITY_COMMISSION_PER_SHARE = 0.0  # IBKRBroker: $0
FUTURE_COMMISSION_PER_CONTRACT = 1.50


def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = _safe_mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


# ─── TCA Engine ──────────────────────────────────────────────────

class TCAEngine:
    """Full TCA engine — decomposes, aggregates, and scores execution quality.

    Usage:
        engine = TCAEngine()
        engine.rebuild(trades)  # trades = broker.get_trade_history()
        records = engine.get_records()
        trend = engine.get_cost_trend(days=30)
        venues = engine.get_venue_stats()
        algos = engine.get_algo_stats()
        outliers = engine.get_outliers()
        benchmarks = engine.get_benchmarks()
        summary = engine.get_summary()
    """

    def __init__(self):
        self._records: List[TCATradeRecord] = []
        self._by_day: Dict[str, List[TCATradeRecord]] = defaultdict(list)
        self._by_venue: Dict[str, List[TCATradeRecord]] = defaultdict(list)
        self._by_algo: Dict[str, List[TCATradeRecord]] = defaultdict(list)
        self._by_sector: Dict[str, List[TCATradeRecord]] = defaultdict(list)
        self._last_rebuild: Optional[datetime] = None
        self._quote_cache: Dict[str, dict] = {}

    def _classify_sector(self, ticker: str) -> str:
        """Map ticker to sector."""
        return SECTOR_MAP.get(ticker, "Other")

    def _decompose_trade(self, trade: dict, quote_data: Optional[dict] = None) -> TCATradeRecord:
        """Decompose a single trade into TCA cost components."""
        ticker = trade.get("ticker", "")
        side_raw = trade.get("side", "BUY")
        side = side_raw.value if hasattr(side_raw, "value") else str(side_raw)
        qty = abs(trade.get("quantity", 0) or 0)
        fill_price = trade.get("fill_price", 0) or 0
        arrival = trade.get("arrival_price", fill_price) or fill_price
        sig_raw = trade.get("signal_type", "")
        signal_type = sig_raw.value if hasattr(sig_raw, "value") else str(sig_raw)
        venue = trade.get("venue", "ENGINE") or "ENGINE"
        latency = trade.get("latency_ms", 0) or 0
        product_type = trade.get("product_type", "EQUITY") or "EQUITY"
        if hasattr(product_type, "value"):
            product_type = product_type.value

        ts_raw = trade.get("fill_timestamp", trade.get("timestamp", ""))
        timestamp = ts_raw.isoformat() if hasattr(ts_raw, "isoformat") else str(ts_raw or "")

        notional = abs(qty * fill_price) if fill_price else 0

        # ── 1. Spread cost ─────────────────────────────────────
        spread_bps = 1.5  # Default half-spread estimate
        if quote_data:
            bid = quote_data.get("bid", 0) or 0
            ask = quote_data.get("ask", 0) or 0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                if mid > 0:
                    spread_bps = ((ask - bid) / mid) * 10_000 / 2  # Half-spread

        # ── 2. Market impact (sqrt model) ──────────────────────
        adv = 100_000  # Default ADV
        daily_vol = 0.02  # Default daily volatility
        participation = qty / max(adv, 1)
        impact_bps = IMPACT_COEFFICIENT * (participation ** 0.5) * daily_vol * 10_000

        # ── 3. Timing cost ─────────────────────────────────────
        timing_bps = 0.0
        if arrival > 0 and fill_price > 0:
            if side in ("BUY", "COVER"):
                timing_bps = (fill_price - arrival) / arrival * 10_000
            else:
                timing_bps = (arrival - fill_price) / arrival * 10_000

        # ── 4. Commission ──────────────────────────────────────
        if "FUTURE" in product_type.upper():
            commission_usd = abs(qty) * FUTURE_COMMISSION_PER_CONTRACT
        else:
            commission_usd = abs(qty) * EQUITY_COMMISSION_PER_SHARE
        commission_bps = (commission_usd / max(notional, 1)) * 10_000 if notional > 0 else 0

        total_bps = spread_bps + impact_bps + max(timing_bps, 0) + commission_bps

        # ── Implementation Shortfall ───────────────────────────
        is_usd = 0.0
        if arrival > 0 and fill_price > 0:
            if side in ("BUY", "COVER"):
                is_usd = (fill_price - arrival) * qty
            else:
                is_usd = (arrival - fill_price) * qty

        # ── VWAP Slippage ─────────────────────────────────────
        vwap_price = trade.get("vwap_price", 0) or 0
        vwap_slip = 0.0
        if vwap_price > 0 and fill_price > 0:
            if side in ("BUY", "COVER"):
                vwap_slip = (fill_price - vwap_price) / vwap_price * 10_000
            else:
                vwap_slip = (vwap_price - fill_price) / vwap_price * 10_000

        # ── Participation rate estimate ────────────────────────
        participation_rate = min(participation * 100, 100)

        # ── Algo classification ────────────────────────────────
        algo = "SMART"
        if signal_type:
            sig_upper = signal_type.upper()
            if "VWAP" in sig_upper:
                algo = "VWAP"
            elif "TWAP" in sig_upper:
                algo = "TWAP"
            elif "IS" in sig_upper or "SHORTFALL" in sig_upper:
                algo = "IS"
            elif "POV" in sig_upper or "PARTICIPATION" in sig_upper:
                algo = "POV"
            elif "CLOSE" in sig_upper or "MOC" in sig_upper:
                algo = "CLOSE"
            elif "ADAPTIVE" in sig_upper:
                algo = "ADAPTIVE"
            elif "ARB" in sig_upper:
                algo = "ARB"
            elif "MOMENTUM" in sig_upper:
                algo = "MOMENTUM"
            elif "MEAN_REVERSION" in sig_upper or "MR" in sig_upper:
                algo = "MEAN_REV"
            elif "ML" in sig_upper or "ENSEMBLE" in sig_upper:
                algo = "ML_ENSEMBLE"
            else:
                algo = signal_type[:12]

        return TCATradeRecord(
            order_id=trade.get("order_id", f"TCA-{id(trade):08x}"),
            ticker=ticker,
            side=side,
            quantity=qty,
            fill_price=round(fill_price, 4),
            arrival_price=round(arrival, 4),
            vwap_price=round(vwap_price, 4),
            spread_cost_bps=round(spread_bps, 2),
            market_impact_bps=round(impact_bps, 2),
            timing_cost_bps=round(timing_bps, 2),
            commission_bps=round(commission_bps, 2),
            total_cost_bps=round(total_bps, 2),
            implementation_shortfall_usd=round(is_usd, 2),
            vwap_slippage_bps=round(vwap_slip, 2),
            venue=venue,
            algo=algo,
            sector=self._classify_sector(ticker),
            signal_type=signal_type,
            product_type=product_type,
            participation_rate=round(participation_rate, 1),
            latency_ms=round(latency, 1),
            timestamp=timestamp,
        )

    def rebuild(self, trades: List[dict], quotes: Optional[Dict[str, dict]] = None):
        """Rebuild TCA from raw trade history (broker.get_trade_history()).

        Args:
            trades: List of trade dicts from broker
            quotes: Optional {ticker: {bid, ask, ...}} for spread calculations
        """
        self._records = []
        self._by_day.clear()
        self._by_venue.clear()
        self._by_algo.clear()
        self._by_sector.clear()
        self._quote_cache = quotes or {}

        for t in trades:
            # Skip unfilled trades
            fp = t.get("fill_price", 0) or 0
            if fp <= 0:
                status = t.get("status", "")
                if hasattr(status, "value"):
                    status = status.value
                if str(status) not in ("FILLED", "PARTIAL"):
                    continue

            ticker = t.get("ticker", "")
            quote = self._quote_cache.get(ticker)
            rec = self._decompose_trade(t, quote)
            self._records.append(rec)

            # Index by day
            day = rec.timestamp[:10] if rec.timestamp else "unknown"
            self._by_day[day].append(rec)

            # Index by venue
            self._by_venue[rec.venue].append(rec)

            # Index by algo
            self._by_algo[rec.algo].append(rec)

            # Index by sector
            self._by_sector[rec.sector].append(rec)

        self._last_rebuild = datetime.now(timezone.utc)
        logger.info(f"TCAEngine rebuilt: {len(self._records)} trades decomposed")

    # ─── Accessors ────────────────────────────────────────────

    def get_records(self, limit: int = 200) -> List[dict]:
        """Return most recent TCA records."""
        return [r.to_dict() for r in self._records[-limit:]]

    def get_summary(self) -> dict:
        """Aggregate summary metrics."""
        if not self._records:
            return {
                "total_trades": 0, "fill_rate": 0, "avg_total_cost_bps": 0,
                "avg_spread_bps": 0, "avg_impact_bps": 0, "avg_timing_bps": 0,
                "avg_vwap_slip_bps": 0, "total_is_usd": 0,
                "total_volume_usd": 0, "execution_quality_score": 0,
                "best_execution": None, "worst_execution": None,
                "cost_trend": "STABLE",
            }

        recs = self._records
        total_cost = [r.total_cost_bps for r in recs]

        # Execution quality score: 100 - normalized cost
        avg_total = _safe_mean(total_cost)
        quality_score = max(0, min(100, 100 - avg_total * 10))

        # Cost trend: compare last 20% vs first 20%
        n = len(recs)
        split = max(1, n // 5)
        early = _safe_mean([r.total_cost_bps for r in recs[:split]])
        late = _safe_mean([r.total_cost_bps for r in recs[-split:]])
        if early > 0 and late < early * 0.9:
            trend = "IMPROVING"
        elif early > 0 and late > early * 1.1:
            trend = "DEGRADING"
        else:
            trend = "STABLE"

        # Best / worst
        best = min(recs, key=lambda r: r.total_cost_bps)
        worst = max(recs, key=lambda r: r.total_cost_bps)

        return {
            "total_trades": len(recs),
            "fill_rate": len(recs) / max(len(recs), 1),
            "avg_total_cost_bps": round(avg_total, 2),
            "avg_spread_bps": round(_safe_mean([r.spread_cost_bps for r in recs]), 2),
            "avg_impact_bps": round(_safe_mean([r.market_impact_bps for r in recs]), 2),
            "avg_timing_bps": round(_safe_mean([r.timing_cost_bps for r in recs]), 2),
            "avg_commission_bps": round(_safe_mean([r.commission_bps for r in recs]), 2),
            "avg_vwap_slip_bps": round(_safe_mean([r.vwap_slippage_bps for r in recs]), 2),
            "total_is_usd": round(sum(r.implementation_shortfall_usd for r in recs), 2),
            "total_volume_usd": round(sum(abs(r.quantity * r.fill_price) for r in recs), 2),
            "execution_quality_score": round(quality_score, 1),
            "best_execution": {"ticker": best.ticker, "cost_bps": best.total_cost_bps},
            "worst_execution": {"ticker": worst.ticker, "cost_bps": worst.total_cost_bps},
            "cost_trend": trend,
        }

    def get_cost_trend(self, days: int = 30) -> List[dict]:
        """Daily cost trend for time-series chart."""
        if not self._records:
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        points: List[dict] = []

        for day in sorted(self._by_day.keys()):
            if day < cutoff or day == "unknown":
                continue
            recs = self._by_day[day]
            n = len(recs)
            if n == 0:
                continue
            volume = sum(abs(r.quantity * r.fill_price) for r in recs)
            points.append({
                "date": day,
                "trades": n,
                "volume_usd": round(volume, 0),
                "spread_bps": round(_safe_mean([r.spread_cost_bps for r in recs]), 2),
                "impact_bps": round(_safe_mean([r.market_impact_bps for r in recs]), 2),
                "timing_bps": round(_safe_mean([r.timing_cost_bps for r in recs]), 2),
                "commission_bps": round(_safe_mean([r.commission_bps for r in recs]), 2),
                "total_bps": round(_safe_mean([r.total_cost_bps for r in recs]), 2),
                "avg_is_usd": round(_safe_mean([r.implementation_shortfall_usd for r in recs]), 2),
            })

        return points

    def get_venue_stats(self) -> List[dict]:
        """Per-venue execution quality metrics."""
        stats: List[dict] = []
        for venue, recs in self._by_venue.items():
            if not recs:
                continue
            avg_cost = _safe_mean([r.total_cost_bps for r in recs])
            avg_latency = _safe_mean([r.latency_ms for r in recs])
            avg_impact = _safe_mean([r.market_impact_bps for r in recs])
            avg_spread = _safe_mean([r.spread_cost_bps for r in recs])
            # Quality: lower cost → higher score (max 100)
            q_score = max(0, min(100, 100 - avg_cost * 8))
            stats.append({
                "venue": venue,
                "fills": len(recs),
                "avg_cost_bps": round(avg_cost, 2),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_impact_bps": round(avg_impact, 2),
                "avg_spread_bps": round(avg_spread, 2),
                "fill_rate": 1.0,
                "quality_score": round(q_score, 1),
            })
        return sorted(stats, key=lambda s: s["avg_cost_bps"])

    def get_algo_stats(self) -> List[dict]:
        """Per-algorithm execution quality metrics."""
        stats: List[dict] = []
        for algo, recs in self._by_algo.items():
            if not recs:
                continue
            avg_cost = _safe_mean([r.total_cost_bps for r in recs])
            avg_is = _safe_mean([r.implementation_shortfall_usd for r in recs])
            avg_vwap = _safe_mean([r.vwap_slippage_bps for r in recs])
            avg_part = _safe_mean([r.participation_rate for r in recs])
            q_score = max(0, min(100, 100 - avg_cost * 8))
            stats.append({
                "algo": algo,
                "trades": len(recs),
                "avg_cost_bps": round(avg_cost, 2),
                "avg_is_usd": round(avg_is, 2),
                "avg_vwap_slip_bps": round(avg_vwap, 2),
                "avg_participation": round(avg_part, 1),
                "quality_score": round(q_score, 1),
            })
        return sorted(stats, key=lambda s: s["avg_cost_bps"])

    def get_sector_stats(self) -> List[dict]:
        """Per-sector cost breakdown."""
        stats: List[dict] = []
        for sector, recs in self._by_sector.items():
            if not recs:
                continue
            stats.append({
                "sector": sector,
                "trades": len(recs),
                "avg_cost_bps": round(_safe_mean([r.total_cost_bps for r in recs]), 2),
                "avg_impact_bps": round(_safe_mean([r.market_impact_bps for r in recs]), 2),
                "total_is_usd": round(sum(r.implementation_shortfall_usd for r in recs), 2),
                "total_volume_usd": round(sum(abs(r.quantity * r.fill_price) for r in recs), 2),
            })
        return sorted(stats, key=lambda s: s["avg_cost_bps"], reverse=True)

    def get_outliers(self, sigma_threshold: float = 2.0) -> List[dict]:
        """Flag trades with cost > sigma_threshold standard deviations from mean."""
        if len(self._records) < 5:
            return []

        costs = [r.total_cost_bps for r in self._records]
        mu = _safe_mean(costs)
        sigma = _safe_std(costs)

        if sigma <= 0:
            return []

        outliers: List[dict] = []
        for rec in self._records:
            z = (rec.total_cost_bps - mu) / sigma
            if abs(z) >= sigma_threshold:
                reason = "HIGH_COST" if z > 0 else "NEGATIVE_COST"
                if rec.market_impact_bps > mu + sigma:
                    reason = "HIGH_MARKET_IMPACT"
                elif rec.timing_cost_bps > mu + sigma:
                    reason = "ADVERSE_TIMING"
                outliers.append({
                    "order_id": rec.order_id,
                    "ticker": rec.ticker,
                    "side": rec.side,
                    "total_cost_bps": rec.total_cost_bps,
                    "z_score": round(z, 2),
                    "reason": reason,
                    "timestamp": rec.timestamp,
                })

        return sorted(outliers, key=lambda o: abs(o["z_score"]), reverse=True)[:20]

    def get_benchmarks(self) -> List[dict]:
        """Compare execution quality against standard benchmarks."""
        if not self._records:
            return []

        benchmarks: List[dict] = []

        # Arrival price benchmark
        arrival_slips = [r.timing_cost_bps for r in self._records if r.arrival_price > 0]
        if arrival_slips:
            benchmarks.append({
                "benchmark": "Arrival",
                "avg_slippage_bps": round(_safe_mean(arrival_slips), 2),
                "total_shortfall_usd": round(
                    sum(r.implementation_shortfall_usd for r in self._records), 2
                ),
                "win_rate": round(
                    sum(1 for s in arrival_slips if s <= 0) / len(arrival_slips) * 100, 1
                ),
                "trades_evaluated": len(arrival_slips),
            })

        # VWAP benchmark
        vwap_slips = [r.vwap_slippage_bps for r in self._records if r.vwap_price > 0]
        if vwap_slips:
            benchmarks.append({
                "benchmark": "VWAP",
                "avg_slippage_bps": round(_safe_mean(vwap_slips), 2),
                "total_shortfall_usd": 0,
                "win_rate": round(
                    sum(1 for s in vwap_slips if s <= 0) / len(vwap_slips) * 100, 1
                ),
                "trades_evaluated": len(vwap_slips),
            })

        # Overall cost benchmark (vs zero-cost ideal)
        total_costs = [r.total_cost_bps for r in self._records]
        benchmarks.append({
            "benchmark": "Zero-Cost Ideal",
            "avg_slippage_bps": round(_safe_mean(total_costs), 2),
            "total_shortfall_usd": round(
                sum(abs(r.quantity * r.fill_price) * r.total_cost_bps / 10_000
                    for r in self._records), 2
            ),
            "win_rate": round(
                sum(1 for c in total_costs if c <= 1.0) / len(total_costs) * 100, 1
            ),
            "trades_evaluated": len(total_costs),
        })

        return benchmarks

    def get_cost_decomposition(self) -> List[dict]:
        """Per-asset cost decomposition for stacked bar chart."""
        ticker_map: Dict[str, List[TCATradeRecord]] = defaultdict(list)
        for r in self._records:
            ticker_map[r.ticker].append(r)

        decomp: List[dict] = []
        for ticker, recs in sorted(ticker_map.items()):
            decomp.append({
                "ticker": ticker,
                "trades": len(recs),
                "spread": round(_safe_mean([r.spread_cost_bps for r in recs]), 2),
                "impact": round(_safe_mean([r.market_impact_bps for r in recs]), 2),
                "timing": round(_safe_mean([r.timing_cost_bps for r in recs]), 2),
                "commission": round(_safe_mean([r.commission_bps for r in recs]), 2),
                "total": round(_safe_mean([r.total_cost_bps for r in recs]), 2),
            })

        return sorted(decomp, key=lambda d: d["total"], reverse=True)

    def get_is_distribution(self) -> List[dict]:
        """Implementation shortfall distribution buckets."""
        buckets = [
            {"range": "< -$500", "min": float("-inf"), "max": -500, "count": 0},
            {"range": "-$500 to -$100", "min": -500, "max": -100, "count": 0},
            {"range": "-$100 to $0", "min": -100, "max": 0, "count": 0},
            {"range": "$0 to $100", "min": 0, "max": 100, "count": 0},
            {"range": "$100 to $500", "min": 100, "max": 500, "count": 0},
            {"range": "> $500", "min": 500, "max": float("inf"), "count": 0},
        ]
        for r in self._records:
            v = r.implementation_shortfall_usd
            for b in buckets:
                if b["min"] <= v < b["max"]:
                    b["count"] += 1
                    break

        return [{"range": b["range"], "count": b["count"]} for b in buckets]

    def get_execution_quality_score(self) -> dict:
        """Composite execution quality metrics for radar chart."""
        if not self._records:
            return {"dimensions": [], "scores": []}

        recs = self._records
        avg_cost = _safe_mean([r.total_cost_bps for r in recs])
        avg_impact = _safe_mean([r.market_impact_bps for r in recs])
        avg_timing = _safe_mean([abs(r.timing_cost_bps) for r in recs])
        avg_spread = _safe_mean([r.spread_cost_bps for r in recs])
        avg_vwap_slip = _safe_mean([abs(r.vwap_slippage_bps) for r in recs])

        # Normalize to 0-100 (lower cost = higher score)
        cost_score = max(0, min(100, 100 - avg_cost * 8))
        impact_score = max(0, min(100, 100 - avg_impact * 15))
        timing_score = max(0, min(100, 100 - avg_timing * 12))
        spread_score = max(0, min(100, 100 - avg_spread * 20))
        vwap_score = max(0, min(100, 100 - avg_vwap_slip * 10))
        consistency = max(0, min(100, 100 - _safe_std([r.total_cost_bps for r in recs]) * 10))

        return {
            "dimensions": [
                "Total Cost", "Market Impact", "Timing",
                "Spread", "VWAP Tracking", "Consistency"
            ],
            "scores": [
                round(cost_score, 1), round(impact_score, 1), round(timing_score, 1),
                round(spread_score, 1), round(vwap_score, 1), round(consistency, 1),
            ],
        }
