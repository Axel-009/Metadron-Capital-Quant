"""
sector_tracker.py — Sector Tracking Engine for Metadron Capital

Tracks all 11 GICS sectors with performance data, flow signals,
heatmap generation, rotation detection, missed opportunity logging,
and error tracking. All data sourced via OpenBB.
"""

import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..data.yahoo_data import get_adj_close, get_returns, get_prices
    from ..data.universe_engine import SECTOR_ETFS, GICS_SECTORS, SP500_TOP_HOLDINGS
except ImportError:
    def get_adj_close(ticker: str, period: str = "1y") -> Optional[np.ndarray]:
        """Stub: returns None when data layer unavailable."""
        return None

    def get_returns(ticker: str, period: str = "1y") -> Optional[np.ndarray]:
        """Stub: returns None when data layer unavailable."""
        return None

    def get_prices(ticker: str, period: str = "1y") -> Optional[dict]:
        """Stub: returns None when data layer unavailable."""
        return None

    SECTOR_ETFS = {
        "Information Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Health Care": "XLV",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
    }

    GICS_SECTORS = list(SECTOR_ETFS.keys())

    SP500_TOP_HOLDINGS = {
        "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN"],
        "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY"],
        "Health Care": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
        "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL", "MO", "KHC"],
        "Industrials": ["CAT", "UNP", "HON", "UPS", "BA", "RTX", "DE", "GE", "LMT", "MMM"],
        "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "NUE", "VMC", "MLM"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
        "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
        "Communication Services": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR"],
    }

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_TICKER = "SPY"

TIMEFRAME_DAYS = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": None,  # computed dynamically
}

HEAT_THRESHOLDS = {
    "HOT": 2.0,
    "WARM": 0.0,
    "COOL": -2.0,
    # anything below COOL is COLD
}

CYCLICAL_SECTORS = [
    "Information Technology",
    "Consumer Discretionary",
    "Financials",
    "Industrials",
    "Materials",
    "Energy",
]

DEFENSIVE_SECTORS = [
    "Consumer Staples",
    "Health Care",
    "Utilities",
    "Real Estate",
    "Communication Services",
]

ROTATION_PAIRS = [
    ("XLK", "XLU"),   # Tech vs Utilities (risk-on / risk-off)
    ("XLY", "XLP"),   # Discretionary vs Staples
    ("XLF", "XLU"),   # Financials vs Utilities
    ("XLI", "XLV"),   # Industrials vs Health Care
    ("XLE", "XLRE"),  # Energy vs Real Estate
]


# =========================================================================
# 1. SectorPerformanceTracker
# =========================================================================

class SectorPerformanceTracker:
    """Track performance, relative strength, momentum, and breadth
    for all 11 GICS sectors via their corresponding ETFs."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._last_refresh: Optional[datetime.datetime] = None

    # ----- helpers -----

    @staticmethod
    def _pct_change(prices: np.ndarray, n: int) -> Optional[float]:
        """Return percentage change over the last *n* trading days."""
        if prices is None or len(prices) <= n:
            return None
        return float((prices[-1] / prices[-n - 1]) - 1.0) * 100.0

    @staticmethod
    def _ytd_return(prices: np.ndarray) -> Optional[float]:
        """Return YTD percentage change.  Assumes *prices* covers >= 1 year."""
        if prices is None or len(prices) < 2:
            return None
        today = datetime.date.today()
        ytd_days = (today - datetime.date(today.year, 1, 1)).days
        trading_days = int(ytd_days * 252 / 365)
        trading_days = min(trading_days, len(prices) - 1)
        if trading_days < 1:
            return None
        return float((prices[-1] / prices[-trading_days - 1]) - 1.0) * 100.0

    def _fetch_sector_prices(self, period: str = "1y") -> Dict[str, Optional[np.ndarray]]:
        """Fetch adjusted close prices for every sector ETF and the benchmark."""
        tickers = dict(SECTOR_ETFS)
        tickers["SPY"] = "SPY"
        result: Dict[str, Optional[np.ndarray]] = {}
        for label, ticker in tickers.items():
            result[label] = get_adj_close(ticker, period)
        return result

    # ----- public API -----

    def compute_performance(self, period: str = "1y") -> Dict[str, Dict[str, Optional[float]]]:
        """Return a dict keyed by sector name with sub-dict of timeframe returns."""
        prices_map = self._fetch_sector_prices(period)
        results: Dict[str, Dict[str, Optional[float]]] = {}
        for sector in GICS_SECTORS:
            prices = prices_map.get(sector)
            perf: Dict[str, Optional[float]] = {}
            for tf_label, tf_days in TIMEFRAME_DAYS.items():
                if tf_label == "YTD":
                    perf[tf_label] = self._ytd_return(prices)
                else:
                    perf[tf_label] = self._pct_change(prices, tf_days) if prices is not None else None
            results[sector] = perf
        return results

    def compute_relative_strength(self, period: str = "1y") -> Dict[str, Dict[str, Optional[float]]]:
        """Return sector returns minus SPY returns for each timeframe."""
        prices_map = self._fetch_sector_prices(period)
        spy_prices = prices_map.get("SPY")
        spy_perf: Dict[str, Optional[float]] = {}
        for tf_label, tf_days in TIMEFRAME_DAYS.items():
            if tf_label == "YTD":
                spy_perf[tf_label] = self._ytd_return(spy_prices)
            else:
                spy_perf[tf_label] = self._pct_change(spy_prices, tf_days) if spy_prices is not None else None

        results: Dict[str, Dict[str, Optional[float]]] = {}
        for sector in GICS_SECTORS:
            prices = prices_map.get(sector)
            rs: Dict[str, Optional[float]] = {}
            for tf_label, tf_days in TIMEFRAME_DAYS.items():
                if tf_label == "YTD":
                    sec_ret = self._ytd_return(prices)
                else:
                    sec_ret = self._pct_change(prices, tf_days) if prices is not None else None
                spy_ret = spy_perf.get(tf_label)
                if sec_ret is not None and spy_ret is not None:
                    rs[tf_label] = round(sec_ret - spy_ret, 4)
                else:
                    rs[tf_label] = None
            results[sector] = rs
        return results

    def momentum_score(self, period: str = "1y") -> Dict[str, Optional[float]]:
        """Composite momentum score: weighted average of multi-timeframe returns.
        Weights: 1D=5%, 1W=10%, 1M=20%, 3M=25%, 6M=25%, YTD=15%.
        """
        weights = {"1D": 0.05, "1W": 0.10, "1M": 0.20, "3M": 0.25, "6M": 0.25, "YTD": 0.15}
        perf = self.compute_performance(period)
        scores: Dict[str, Optional[float]] = {}
        for sector, tf_map in perf.items():
            total_w = 0.0
            weighted_sum = 0.0
            for tf, w in weights.items():
                val = tf_map.get(tf)
                if val is not None:
                    weighted_sum += val * w
                    total_w += w
            scores[sector] = round(weighted_sum / total_w, 4) if total_w > 0 else None
        return scores

    def volume_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Compare latest volume to 20-day average for each sector ETF."""
        results: Dict[str, Dict[str, Any]] = {}
        for sector, etf in SECTOR_ETFS.items():
            data = get_prices(etf, period="3mo")
            if data is None or len(data.get("volume", [])) < 21:
                results[sector] = {"status": "NO_DATA"}
                continue
            vol = data["volume"]
            avg_20 = float(np.mean(vol[-21:-1]))
            latest = float(vol[-1])
            ratio = latest / avg_20 if avg_20 > 0 else 0.0
            results[sector] = {
                "latest_volume": latest,
                "avg_20d_volume": round(avg_20, 0),
                "volume_ratio": round(ratio, 2),
                "signal": "ABOVE_AVG" if ratio > 1.0 else "BELOW_AVG",
            }
        return results

    def sector_breadth(self) -> Dict[str, Dict[str, Any]]:
        """Percentage of top holdings within each sector that are positive today."""
        results: Dict[str, Dict[str, Any]] = {}
        for sector in GICS_SECTORS:
            holdings = SP500_TOP_HOLDINGS.get(sector, [])
            if not holdings:
                results[sector] = {"pct_positive": None, "positive": 0, "total": 0}
                continue
            pos_count = 0
            total_count = 0
            for ticker in holdings:
                prices = get_adj_close(ticker, period="5d")
                if prices is not None and len(prices) >= 2:
                    total_count += 1
                    if prices[-1] > prices[-2]:
                        pos_count += 1
            pct = round(pos_count / total_count * 100.0, 1) if total_count > 0 else None
            results[sector] = {
                "pct_positive": pct,
                "positive": pos_count,
                "total": total_count,
            }
        return results


# =========================================================================
# 2. SectorFlowAnalyzer
# =========================================================================

class SectorFlowAnalyzer:
    """Estimate money flow via ETF volume * price direction.
    This is an approximation — true fund-flow data requires premium feeds."""

    @staticmethod
    def _compute_flow(data: Optional[dict], lookback: int = 5) -> Optional[np.ndarray]:
        """Dollar-volume signed by direction of close-to-close move."""
        if data is None:
            return None
        close = data.get("close")
        volume = data.get("volume")
        if close is None or volume is None or len(close) < lookback + 1:
            return None
        direction = np.sign(np.diff(close))  # +1 / 0 / -1
        dollar_vol = close[1:] * volume[1:]
        return direction * dollar_vol

    def net_flow(self, lookback: int = 5) -> Dict[str, Dict[str, Any]]:
        """Net flow signal for each sector over the last *lookback* days."""
        results: Dict[str, Dict[str, Any]] = {}
        for sector, etf in SECTOR_ETFS.items():
            data = get_prices(etf, period="3mo")
            flow = self._compute_flow(data, lookback)
            if flow is None or len(flow) < lookback:
                results[sector] = {"signal": "NO_DATA", "net_flow": None}
                continue
            recent = flow[-lookback:]
            net = float(np.sum(recent))
            avg_magnitude = float(np.mean(np.abs(recent)))
            if avg_magnitude == 0:
                signal = "NEUTRAL"
            elif net / avg_magnitude > 1.5:
                signal = "INFLOW"
            elif net / avg_magnitude < -1.5:
                signal = "OUTFLOW"
            else:
                signal = "NEUTRAL"
            results[sector] = {
                "signal": signal,
                "net_flow": round(net, 0),
                "avg_daily_magnitude": round(avg_magnitude, 0),
            }
        return results

    def flow_momentum(self, lookback: int = 10) -> Dict[str, str]:
        """Is flow accelerating or decelerating?  Compare first-half vs second-half."""
        results: Dict[str, str] = {}
        for sector, etf in SECTOR_ETFS.items():
            data = get_prices(etf, period="3mo")
            flow = self._compute_flow(data, lookback)
            if flow is None or len(flow) < lookback:
                results[sector] = "NO_DATA"
                continue
            recent = flow[-lookback:]
            half = lookback // 2
            first_half = float(np.sum(recent[:half]))
            second_half = float(np.sum(recent[half:]))
            if abs(first_half) < 1e-9:
                results[sector] = "STABLE"
            elif second_half / abs(first_half) > 1.2:
                results[sector] = "ACCELERATING"
            elif second_half / abs(first_half) < 0.8:
                results[sector] = "DECELERATING"
            else:
                results[sector] = "STABLE"
        return results

    def flow_divergence(self, lookback: int = 5) -> Dict[str, Dict[str, Any]]:
        """Detect divergence: price rising + outflow = bearish warning,
        price falling + inflow = bullish signal."""
        results: Dict[str, Dict[str, Any]] = {}
        for sector, etf in SECTOR_ETFS.items():
            data = get_prices(etf, period="3mo")
            if data is None or len(data.get("close", [])) < lookback + 1:
                results[sector] = {"divergence": "NO_DATA"}
                continue
            close = data["close"]
            price_chg = float(close[-1] / close[-lookback - 1] - 1.0)
            flow = self._compute_flow(data, lookback)
            if flow is None or len(flow) < lookback:
                results[sector] = {"divergence": "NO_DATA"}
                continue
            net = float(np.sum(flow[-lookback:]))
            if price_chg > 0.005 and net < 0:
                div = "BEARISH_DIVERGENCE"
            elif price_chg < -0.005 and net > 0:
                div = "BULLISH_DIVERGENCE"
            else:
                div = "NONE"
            results[sector] = {
                "divergence": div,
                "price_change_pct": round(price_chg * 100, 2),
                "net_flow": round(net, 0),
            }
        return results

    def cross_sector_ranking(self, lookback: int = 5) -> List[Tuple[str, float]]:
        """Rank sectors by net flow (highest inflow first)."""
        flows = self.net_flow(lookback)
        ranked: List[Tuple[str, float]] = []
        for sector, info in flows.items():
            nf = info.get("net_flow")
            if nf is not None:
                ranked.append((sector, nf))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# =========================================================================
# 3. SectorHeatmapGenerator
# =========================================================================

class SectorHeatmapGenerator:
    """Generate ASCII heatmaps of sector/security performance."""

    HEAT_SYMBOLS = {
        "HOT": "+++",
        "WARM": " + ",
        "COOL": " - ",
        "COLD": "---",
    }

    HEAT_COLORS = {
        "HOT": "[HOT]",
        "WARM": "[WRM]",
        "COOL": "[COL]",
        "COLD": "[CLD]",
    }

    @staticmethod
    def _classify_heat(pct: float) -> str:
        if pct >= HEAT_THRESHOLDS["HOT"]:
            return "HOT"
        elif pct >= HEAT_THRESHOLDS["WARM"]:
            return "WARM"
        elif pct >= HEAT_THRESHOLDS["COOL"]:
            return "COOL"
        else:
            return "COLD"

    def _fetch_ticker_return(self, ticker: str, lookback_days: int) -> Optional[float]:
        """Return pct change over *lookback_days* trading days."""
        prices = get_adj_close(ticker, period="3mo")
        if prices is None or len(prices) <= lookback_days:
            return None
        return float((prices[-1] / prices[-lookback_days - 1]) - 1.0) * 100.0

    def sector_heat(self, view: str = "1D") -> Dict[str, Dict[str, Any]]:
        """Compute heat level for each sector ETF.
        *view*: '1D', '1W', '1M'."""
        lookback_map = {"1D": 1, "1W": 5, "1M": 21}
        lb = lookback_map.get(view, 1)
        results: Dict[str, Dict[str, Any]] = {}
        for sector, etf in SECTOR_ETFS.items():
            ret = self._fetch_ticker_return(etf, lb)
            if ret is None:
                results[sector] = {"heat": "NO_DATA", "return_pct": None}
            else:
                results[sector] = {
                    "heat": self._classify_heat(ret),
                    "return_pct": round(ret, 2),
                }
        return results

    def holdings_heat(self, sector: str, view: str = "1D") -> List[Dict[str, Any]]:
        """Heat for individual holdings within a sector."""
        lookback_map = {"1D": 1, "1W": 5, "1M": 21}
        lb = lookback_map.get(view, 1)
        holdings = SP500_TOP_HOLDINGS.get(sector, [])
        result: List[Dict[str, Any]] = []
        for ticker in holdings:
            ret = self._fetch_ticker_return(ticker, lb)
            if ret is None:
                result.append({"ticker": ticker, "heat": "NO_DATA", "return_pct": None})
            else:
                result.append({
                    "ticker": ticker,
                    "heat": self._classify_heat(ret),
                    "return_pct": round(ret, 2),
                })
        result.sort(key=lambda x: x.get("return_pct") or 0.0, reverse=True)
        return result

    def generate_ascii_heatmap(self, view: str = "1D") -> str:
        """Full ASCII heatmap across all sectors and their top holdings."""
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append(f"  SECTOR HEATMAP  |  View: {view}  |  {datetime.date.today().isoformat()}")
        lines.append("=" * 78)
        lines.append("")
        lines.append(f"  {'SECTOR':<30s} {'RET%':>7s}  {'HEAT':>5s}  HOLDINGS")
        lines.append("-" * 78)

        sector_data = self.sector_heat(view)
        for sector in GICS_SECTORS:
            info = sector_data.get(sector, {})
            ret = info.get("return_pct")
            heat = info.get("heat", "NO_DATA")
            ret_str = f"{ret:+.2f}%" if ret is not None else "  N/A "
            heat_tag = self.HEAT_COLORS.get(heat, "[N/A]")

            # Build mini-bar for holdings
            holdings = SP500_TOP_HOLDINGS.get(sector, [])
            holding_parts: List[str] = []
            lookback_map = {"1D": 1, "1W": 5, "1M": 21}
            lb = lookback_map.get(view, 1)
            for ticker in holdings[:6]:
                h_ret = self._fetch_ticker_return(ticker, lb)
                if h_ret is not None:
                    h_heat = self._classify_heat(h_ret)
                    sym = self.HEAT_SYMBOLS.get(h_heat, " ? ")
                    holding_parts.append(f"{ticker}:{sym}")
                else:
                    holding_parts.append(f"{ticker}: ? ")

            lines.append(f"  {sector:<30s} {ret_str:>7s}  {heat_tag:>5s}  {' '.join(holding_parts)}")

        lines.append("")
        lines.append("  Legend: [HOT] > +2%  |  [WRM] 0..+2%  |  [COL] -2..0%  |  [CLD] < -2%")
        lines.append("=" * 78)
        return "\n".join(lines)

    def generate_sector_detail(self, sector: str, view: str = "1D") -> str:
        """Detailed ASCII breakdown for a single sector."""
        lines: List[str] = []
        lines.append(f"  SECTOR DETAIL: {sector}")
        lines.append("-" * 50)
        etf = SECTOR_ETFS.get(sector, "???")
        lookback_map = {"1D": 1, "1W": 5, "1M": 21}
        lb = lookback_map.get(view, 1)
        etf_ret = self._fetch_ticker_return(etf, lb)
        etf_heat = self._classify_heat(etf_ret) if etf_ret is not None else "NO_DATA"
        lines.append(f"  ETF: {etf}  |  Return: {etf_ret:+.2f}%  |  Heat: {etf_heat}" if etf_ret is not None
                      else f"  ETF: {etf}  |  Return: N/A  |  Heat: NO_DATA")
        lines.append("")
        lines.append(f"  {'TICKER':<8s} {'RETURN':>8s}  {'HEAT':<6s}  BAR")
        lines.append("  " + "-" * 46)

        holdings = self.holdings_heat(sector, view)
        for h in holdings:
            ticker = h["ticker"]
            ret = h.get("return_pct")
            heat = h.get("heat", "NO_DATA")
            if ret is not None:
                bar_len = int(min(max(ret, -10), 10))
                if bar_len >= 0:
                    bar = "|" + "#" * bar_len
                else:
                    bar = "#" * abs(bar_len) + "|"
                lines.append(f"  {ticker:<8s} {ret:>+7.2f}%  {heat:<6s}  {bar}")
            else:
                lines.append(f"  {ticker:<8s}     N/A   {'N/A':<6s}")

        lines.append("-" * 50)
        return "\n".join(lines)


# =========================================================================
# 4. SectorRotationDetector
# =========================================================================

class SectorRotationDetector:
    """Detect rotation between cyclical and defensive sectors and
    identify early/mid/late cycle behaviour."""

    def __init__(self) -> None:
        self._perf_tracker = SectorPerformanceTracker()

    def cyclical_vs_defensive(self, lookback_days: int = 21) -> Dict[str, Any]:
        """Average return of cyclical sectors vs defensive sectors."""
        prices_map = {s: get_adj_close(SECTOR_ETFS[s], period="6mo") for s in GICS_SECTORS}
        cyc_rets: List[float] = []
        def_rets: List[float] = []
        for sector, prices in prices_map.items():
            if prices is None or len(prices) <= lookback_days:
                continue
            ret = float((prices[-1] / prices[-lookback_days - 1]) - 1.0) * 100.0
            if sector in CYCLICAL_SECTORS:
                cyc_rets.append(ret)
            elif sector in DEFENSIVE_SECTORS:
                def_rets.append(ret)

        cyc_avg = float(np.mean(cyc_rets)) if cyc_rets else 0.0
        def_avg = float(np.mean(def_rets)) if def_rets else 0.0
        spread = cyc_avg - def_avg

        if spread > 3.0:
            regime = "RISK_ON"
        elif spread < -3.0:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        return {
            "cyclical_avg_return": round(cyc_avg, 2),
            "defensive_avg_return": round(def_avg, 2),
            "spread": round(spread, 2),
            "regime": regime,
        }

    def pair_ratios(self, lookback_days: int = 21) -> List[Dict[str, Any]]:
        """Ratio analysis on canonical rotation pairs."""
        results: List[Dict[str, Any]] = []
        for risk_on_etf, risk_off_etf in ROTATION_PAIRS:
            p_on = get_adj_close(risk_on_etf, period="6mo")
            p_off = get_adj_close(risk_off_etf, period="6mo")
            if p_on is None or p_off is None or len(p_on) <= lookback_days or len(p_off) <= lookback_days:
                results.append({
                    "pair": f"{risk_on_etf}/{risk_off_etf}",
                    "signal": "NO_DATA",
                })
                continue
            ratio_now = float(p_on[-1] / p_off[-1])
            ratio_prev = float(p_on[-lookback_days - 1] / p_off[-lookback_days - 1])
            ratio_change = ratio_now / ratio_prev - 1.0
            if ratio_change > 0.02:
                sig = "RISK_ON"
            elif ratio_change < -0.02:
                sig = "RISK_OFF"
            else:
                sig = "NEUTRAL"
            results.append({
                "pair": f"{risk_on_etf}/{risk_off_etf}",
                "ratio_now": round(ratio_now, 4),
                "ratio_prev": round(ratio_prev, 4),
                "ratio_change_pct": round(ratio_change * 100, 2),
                "signal": sig,
            })
        return results

    def rotation_velocity(self, window: int = 5) -> Dict[str, Any]:
        """Measure how fast rotation is occurring (daily spread changes)."""
        spreads: List[float] = []
        for day_offset in range(window, 0, -1):
            prices_map = {s: get_adj_close(SECTOR_ETFS[s], period="6mo") for s in GICS_SECTORS}
            cyc_rets: List[float] = []
            def_rets: List[float] = []
            for sector, prices in prices_map.items():
                if prices is None or len(prices) <= day_offset + 1:
                    continue
                idx = -day_offset if day_offset > 0 else -1
                prev_idx = idx - 1
                if abs(prev_idx) >= len(prices):
                    continue
                ret = float((prices[idx] / prices[prev_idx]) - 1.0) * 100.0
                if sector in CYCLICAL_SECTORS:
                    cyc_rets.append(ret)
                elif sector in DEFENSIVE_SECTORS:
                    def_rets.append(ret)
            cyc_avg = float(np.mean(cyc_rets)) if cyc_rets else 0.0
            def_avg = float(np.mean(def_rets)) if def_rets else 0.0
            spreads.append(cyc_avg - def_avg)

        if len(spreads) < 2:
            return {"velocity": 0.0, "direction": "UNKNOWN"}

        velocity = float(np.std(spreads))
        trend = float(np.mean(np.diff(spreads)))
        if trend > 0.2:
            direction = "TOWARD_CYCLICAL"
        elif trend < -0.2:
            direction = "TOWARD_DEFENSIVE"
        else:
            direction = "STABLE"

        return {
            "velocity": round(velocity, 4),
            "trend": round(trend, 4),
            "direction": direction,
        }

    def cycle_phase(self) -> Dict[str, Any]:
        """Estimate economic cycle phase from sector leadership patterns.
        Early: Financials + Tech leading.  Mid: Industrials + Energy.
        Late: Energy + Materials.  Recession: Staples + Utilities + Health Care.
        """
        scores = self._perf_tracker.momentum_score()
        if not scores or all(v is None for v in scores.values()):
            return {"phase": "UNKNOWN", "confidence": 0.0}

        sorted_sectors = sorted(
            [(s, sc) for s, sc in scores.items() if sc is not None],
            key=lambda x: x[1], reverse=True,
        )
        if len(sorted_sectors) < 3:
            return {"phase": "UNKNOWN", "confidence": 0.0}

        top_3 = {s for s, _ in sorted_sectors[:3]}

        early_set = {"Financials", "Information Technology", "Consumer Discretionary"}
        mid_set = {"Industrials", "Energy", "Materials"}
        late_set = {"Energy", "Materials", "Health Care"}
        recession_set = {"Consumer Staples", "Utilities", "Health Care"}

        overlaps = {
            "EARLY_CYCLE": len(top_3 & early_set),
            "MID_CYCLE": len(top_3 & mid_set),
            "LATE_CYCLE": len(top_3 & late_set),
            "RECESSION": len(top_3 & recession_set),
        }
        best_phase = max(overlaps, key=overlaps.get)  # type: ignore[arg-type]
        confidence = round(overlaps[best_phase] / 3.0, 2)

        return {
            "phase": best_phase,
            "confidence": confidence,
            "top_3_sectors": [s for s, _ in sorted_sectors[:3]],
            "overlap_scores": overlaps,
        }


# =========================================================================
# 5. MissedOpportunityLogger
# =========================================================================

class MissedOpportunityLogger:
    """Track large movers that were not in the portfolio."""

    MOVE_THRESHOLD_PCT = 20.0  # flag moves > 20%

    def __init__(self) -> None:
        self._log: List[Dict[str, Any]] = []

    def scan_large_movers(
        self,
        portfolio_tickers: List[str],
        universe: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan today's moves across the universe and log any that exceeded
        the threshold but were NOT in the portfolio."""
        if universe is None:
            universe = SP500_TOP_HOLDINGS

        portfolio_set = set(t.upper() for t in portfolio_tickers)
        missed: List[Dict[str, Any]] = []

        for sector, tickers in universe.items():
            for ticker in tickers:
                prices = get_adj_close(ticker, period="5d")
                if prices is None or len(prices) < 2:
                    continue
                daily_ret = float((prices[-1] / prices[-2]) - 1.0) * 100.0
                if abs(daily_ret) < self.MOVE_THRESHOLD_PCT:
                    continue
                if ticker.upper() in portfolio_set:
                    continue  # we held it — not missed

                reason = self._classify_miss(ticker, portfolio_set, universe)
                entry = {
                    "date": datetime.date.today().isoformat(),
                    "ticker": ticker,
                    "sector": sector,
                    "daily_return_pct": round(daily_ret, 2),
                    "direction": "UP" if daily_ret > 0 else "DOWN",
                    "reason_missed": reason,
                }
                missed.append(entry)
                self._log.append(entry)

        return missed

    @staticmethod
    def _classify_miss(
        ticker: str,
        portfolio_set: set,
        universe: Dict[str, List[str]],
    ) -> str:
        """Heuristic classification of why a name was missed."""
        all_universe = set()
        for tickers in universe.values():
            all_universe.update(t.upper() for t in tickers)
        if ticker.upper() not in all_universe:
            return "NOT_IN_UNIVERSE"
        if ticker.upper() in portfolio_set:
            return "HELD"
        return "NO_SIGNAL"

    def pattern_analysis(self) -> Dict[str, Any]:
        """Analyze logged missed opportunities for recurring patterns."""
        if not self._log:
            return {"total_missed": 0, "by_sector": {}, "by_reason": {}, "avg_move": 0.0}
        by_sector: Dict[str, int] = defaultdict(int)
        by_reason: Dict[str, int] = defaultdict(int)
        moves: List[float] = []
        for entry in self._log:
            by_sector[entry["sector"]] += 1
            by_reason[entry["reason_missed"]] += 1
            moves.append(abs(entry["daily_return_pct"]))
        return {
            "total_missed": len(self._log),
            "by_sector": dict(by_sector),
            "by_reason": dict(by_reason),
            "avg_move": round(float(np.mean(moves)), 2) if moves else 0.0,
        }

    def get_log(self) -> List[Dict[str, Any]]:
        """Return full log of missed opportunities."""
        return list(self._log)

    def clear(self) -> None:
        self._log.clear()


# =========================================================================
# 6. ErrorLogger
# =========================================================================

class ErrorLogger:
    """Centralized error tracking for the sector tracking subsystem."""

    CATEGORIES = ("DATA_ERROR", "SIGNAL_ERROR", "EXECUTION_ERROR", "RISK_GATE")
    SEVERITIES = ("INFO", "WARNING", "ERROR", "CRITICAL")

    def __init__(self) -> None:
        self._errors: List[Dict[str, Any]] = []

    def log(
        self,
        message: str,
        category: str = "DATA_ERROR",
        severity: str = "WARNING",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if category not in self.CATEGORIES:
            category = "DATA_ERROR"
        if severity not in self.SEVERITIES:
            severity = "WARNING"
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "category": category,
            "severity": severity,
            "message": message,
            "context": context or {},
        }
        self._errors.append(entry)
        logger.log(
            getattr(logging, severity, logging.WARNING),
            "[%s] %s — %s",
            category,
            severity,
            message,
        )

    def daily_summary(self) -> Dict[str, Any]:
        """Summarize today's errors."""
        today_str = datetime.date.today().isoformat()
        today_errors = [
            e for e in self._errors if e["timestamp"].startswith(today_str)
        ]
        by_cat: Dict[str, int] = defaultdict(int)
        by_sev: Dict[str, int] = defaultdict(int)
        for e in today_errors:
            by_cat[e["category"]] += 1
            by_sev[e["severity"]] += 1
        return {
            "date": today_str,
            "total": len(today_errors),
            "by_category": dict(by_cat),
            "by_severity": dict(by_sev),
            "errors": today_errors,
        }

    def get_errors(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out = self._errors
        if category:
            out = [e for e in out if e["category"] == category]
        if severity:
            out = [e for e in out if e["severity"] == severity]
        return out

    def clear(self) -> None:
        self._errors.clear()


# =========================================================================
# 7. SectorTrackingEngine (master orchestrator)
# =========================================================================

class SectorTrackingEngine:
    """Master class that composes all sub-components and exposes
    high-level sector tracking operations."""

    def __init__(self) -> None:
        self.performance = SectorPerformanceTracker()
        self.flow = SectorFlowAnalyzer()
        self.heatmap = SectorHeatmapGenerator()
        self.rotation = SectorRotationDetector()
        self.missed = MissedOpportunityLogger()
        self.errors = ErrorLogger()

    # ----- public API -----

    def generate_daily_wrap(self) -> Dict[str, Any]:
        """Full daily sector wrap with all major metrics."""
        try:
            perf = self.performance.compute_performance()
        except Exception as exc:
            self.errors.log(f"Performance fetch failed: {exc}", "DATA_ERROR", "ERROR")
            perf = {}

        try:
            rel_strength = self.performance.compute_relative_strength()
        except Exception as exc:
            self.errors.log(f"Relative strength failed: {exc}", "DATA_ERROR", "ERROR")
            rel_strength = {}

        try:
            momentum = self.performance.momentum_score()
        except Exception as exc:
            self.errors.log(f"Momentum score failed: {exc}", "SIGNAL_ERROR", "ERROR")
            momentum = {}

        try:
            flows = self.flow.net_flow()
        except Exception as exc:
            self.errors.log(f"Flow analysis failed: {exc}", "DATA_ERROR", "ERROR")
            flows = {}

        try:
            rotation = self.rotation.cyclical_vs_defensive()
        except Exception as exc:
            self.errors.log(f"Rotation detection failed: {exc}", "SIGNAL_ERROR", "ERROR")
            rotation = {}

        try:
            phase = self.rotation.cycle_phase()
        except Exception as exc:
            self.errors.log(f"Cycle phase failed: {exc}", "SIGNAL_ERROR", "ERROR")
            phase = {}

        return {
            "date": datetime.date.today().isoformat(),
            "performance": perf,
            "relative_strength": rel_strength,
            "momentum_scores": momentum,
            "flow_signals": flows,
            "rotation": rotation,
            "cycle_phase": phase,
            "error_summary": self.errors.daily_summary(),
        }

    def generate_heatmap(self, view: str = "1D") -> str:
        """ASCII heatmap string."""
        try:
            return self.heatmap.generate_ascii_heatmap(view)
        except Exception as exc:
            self.errors.log(f"Heatmap generation failed: {exc}", "DATA_ERROR", "ERROR")
            return f"[HEATMAP UNAVAILABLE — {exc}]"

    def get_sector_rankings(self) -> List[Tuple[str, Optional[float]]]:
        """Sectors ranked by composite momentum score (best first)."""
        try:
            scores = self.performance.momentum_score()
        except Exception as exc:
            self.errors.log(f"Ranking failed: {exc}", "SIGNAL_ERROR", "ERROR")
            return []
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1] if x[1] is not None else -9999,
            reverse=True,
        )
        return ranked

    def get_missed_opportunities(
        self, portfolio_tickers: List[str]
    ) -> List[Dict[str, Any]]:
        """Scan for large movers not in portfolio."""
        try:
            return self.missed.scan_large_movers(portfolio_tickers)
        except Exception as exc:
            self.errors.log(f"Missed opportunity scan failed: {exc}", "DATA_ERROR", "ERROR")
            return []

    def get_flow_signals(self) -> Dict[str, Dict[str, Any]]:
        """Current flow state for all sectors."""
        try:
            return self.flow.net_flow()
        except Exception as exc:
            self.errors.log(f"Flow signals failed: {exc}", "DATA_ERROR", "ERROR")
            return {}

    def get_rotation_signals(self) -> Dict[str, Any]:
        """Rotation analysis bundle."""
        try:
            return {
                "cyclical_vs_defensive": self.rotation.cyclical_vs_defensive(),
                "pair_ratios": self.rotation.pair_ratios(),
                "velocity": self.rotation.rotation_velocity(),
                "cycle_phase": self.rotation.cycle_phase(),
            }
        except Exception as exc:
            self.errors.log(f"Rotation signals failed: {exc}", "SIGNAL_ERROR", "ERROR")
            return {}

    def get_error_summary(self) -> Dict[str, Any]:
        """Today's error summary."""
        return self.errors.daily_summary()

    def format_sector_dashboard(self) -> str:
        """Full ASCII dashboard combining rankings, flows, rotation, errors."""
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append(f"  METADRON CAPITAL — SECTOR DASHBOARD  |  {datetime.date.today().isoformat()}")
        lines.append("=" * 78)

        # -- Rankings --
        lines.append("")
        lines.append("  SECTOR RANKINGS (by momentum score)")
        lines.append("  " + "-" * 50)
        rankings = self.get_sector_rankings()
        for rank, (sector, score) in enumerate(rankings, start=1):
            score_str = f"{score:+.2f}" if score is not None else " N/A"
            lines.append(f"  {rank:>2d}. {sector:<32s} {score_str}")

        # -- Flow Signals --
        lines.append("")
        lines.append("  FLOW SIGNALS")
        lines.append("  " + "-" * 50)
        flows = self.get_flow_signals()
        for sector in GICS_SECTORS:
            info = flows.get(sector, {})
            sig = info.get("signal", "N/A")
            lines.append(f"  {sector:<32s} {sig}")

        # -- Rotation --
        lines.append("")
        lines.append("  ROTATION")
        lines.append("  " + "-" * 50)
        rot = self.get_rotation_signals()
        cvd = rot.get("cyclical_vs_defensive", {})
        lines.append(f"  Regime:  {cvd.get('regime', 'N/A')}")
        lines.append(f"  Spread:  {cvd.get('spread', 'N/A')}")
        phase = rot.get("cycle_phase", {})
        lines.append(f"  Phase:   {phase.get('phase', 'N/A')}  (confidence {phase.get('confidence', 0):.0%})")
        vel = rot.get("velocity", {})
        lines.append(f"  Direction: {vel.get('direction', 'N/A')}  |  Velocity: {vel.get('velocity', 'N/A')}")

        pairs = rot.get("pair_ratios", [])
        if pairs:
            lines.append("")
            lines.append("  PAIR RATIOS")
            lines.append("  " + "-" * 50)
            for p in pairs:
                lines.append(f"  {p.get('pair', '???'):<14s}  chg: {p.get('ratio_change_pct', 'N/A'):>6}%  {p.get('signal', 'N/A')}")

        # -- Errors --
        lines.append("")
        lines.append("  ERROR SUMMARY")
        lines.append("  " + "-" * 50)
        errs = self.get_error_summary()
        lines.append(f"  Total today: {errs.get('total', 0)}")
        by_sev = errs.get("by_severity", {})
        if by_sev:
            parts = [f"{k}: {v}" for k, v in by_sev.items()]
            lines.append(f"  By severity: {', '.join(parts)}")

        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)


# =========================================================================
# CLI convenience
# =========================================================================

if __name__ == "__main__":
    engine = SectorTrackingEngine()
    print(engine.format_sector_dashboard())
    print()
    print(engine.generate_heatmap("1D"))
