"""Metadron Capital — Evening Backtester.

Post-market analysis engine designed to run after 4:00 PM ET.
Performs:
- Mispricing detection (z-scores, pairs divergence, fair value)
- Relative value analysis (sector RS, momentum, mean reversion)
- Correlation & pattern detection (rolling corr, breakdown alerts)
- Strategy backtesting (walk-forward on top signals)

Results saved to logs/backtest/YYYY-MM-DD_evening.json

Usage:
    from engine.ml.evening_backtester import EveningBacktester
    bt = EveningBacktester()
    results = bt.run_evening_backtest()
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("metadron.ml.evening_backtester")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
BACKTEST_DIR = LOGS_DIR / "backtest"

# ─── Data imports with graceful degradation ───────────────

try:
    from engine.data.yahoo_data import get_adj_close, get_returns, get_fundamentals
except ImportError:
    def get_adj_close(*a, **kw): return pd.DataFrame()
    def get_returns(*a, **kw): return pd.DataFrame()
    def get_fundamentals(*a, **kw): return {}

try:
    from engine.data.universe_engine import SECTOR_ETFS, RV_PAIRS
except ImportError:
    SECTOR_ETFS = {}
    RV_PAIRS = []

try:
    from engine.ml.pattern_recognition import PatternRecognitionEngine
except ImportError:
    PatternRecognitionEngine = None

try:
    from engine.signals.macro_engine import MacroEngine
except ImportError:
    MacroEngine = None

# ─── Constants ────────────────────────────────────────────

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLC", "XLB", "XLRE",
    "TLT", "GLD", "SLV", "USO",
]

TRADING_DAYS_PER_YEAR = 252
Z_SCORE_THRESHOLD = 2.0
CORR_BREAKDOWN_THRESHOLD_HIGH = 0.8
CORR_BREAKDOWN_THRESHOLD_LOW = 0.4


class EveningBacktester:
    """Post-market analysis engine for mispricing, RV, and pattern detection."""

    def __init__(self, universe: list = None, lookback_days: int = 252):
        self.universe = universe or DEFAULT_UNIVERSE
        self.lookback_days = lookback_days
        self._results_cache = None
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Mispricing Detection ─────────────────────────────

    def detect_mispricings(self, prices: pd.DataFrame, returns: pd.DataFrame) -> list:
        """Identify securities trading >2 std devs from rolling mean.

        Also checks pairs divergence and fair value for available tickers.
        """
        mispricings = []

        if prices.empty:
            return mispricings

        # Z-score analysis: rolling 20-day mean/std
        for ticker in prices.columns:
            try:
                series = prices[ticker].dropna()
                if len(series) < 30:
                    continue

                rolling_mean = series.rolling(20).mean()
                rolling_std = series.rolling(20).std()

                if rolling_std.iloc[-1] == 0 or pd.isna(rolling_std.iloc[-1]):
                    continue

                z_score = (series.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

                if abs(z_score) > Z_SCORE_THRESHOLD:
                    current_price = float(series.iloc[-1])
                    fair_value = float(rolling_mean.iloc[-1])
                    deviation = (current_price - fair_value) / fair_value * 100

                    mispricings.append({
                        "ticker": ticker,
                        "current_price": round(current_price, 2),
                        "fair_value": round(fair_value, 2),
                        "z_score": round(float(z_score), 3),
                        "deviation_pct": round(deviation, 2),
                        "signal": "SHORT" if z_score > 0 else "LONG",
                        "confidence": min(round(abs(float(z_score)) / 4.0, 3), 1.0),
                        "source": "z_score",
                    })
            except Exception as e:
                logger.debug(f"Z-score calc failed for {ticker}: {e}")

        # Pairs divergence
        for a, b in RV_PAIRS:
            if a in prices.columns and b in prices.columns:
                try:
                    spread = prices[a] / prices[b]
                    spread = spread.dropna()
                    if len(spread) < 30:
                        continue

                    spread_mean = spread.rolling(20).mean().iloc[-1]
                    spread_std = spread.rolling(20).std().iloc[-1]

                    if spread_std == 0 or pd.isna(spread_std):
                        continue

                    spread_z = (spread.iloc[-1] - spread_mean) / spread_std

                    if abs(spread_z) > Z_SCORE_THRESHOLD:
                        mispricings.append({
                            "ticker": f"{a}/{b}",
                            "current_price": round(float(spread.iloc[-1]), 4),
                            "fair_value": round(float(spread_mean), 4),
                            "z_score": round(float(spread_z), 3),
                            "deviation_pct": round((float(spread.iloc[-1]) - float(spread_mean)) / float(spread_mean) * 100, 2),
                            "signal": f"LONG {b}/SHORT {a}" if spread_z > 0 else f"LONG {a}/SHORT {b}",
                            "confidence": min(round(abs(float(spread_z)) / 4.0, 3), 1.0),
                            "source": "pairs_divergence",
                        })
                except Exception as e:
                    logger.debug(f"Pairs calc failed for {a}/{b}: {e}")

        return sorted(mispricings, key=lambda x: abs(x["z_score"]), reverse=True)

    # ─── Relative Value Analysis ──────────────────────────

    def analyze_relative_value(self, returns: pd.DataFrame) -> list:
        """Cross-sector relative strength, momentum, and mean reversion."""
        rv_signals = []

        if returns.empty:
            return rv_signals

        # Cross-sector relative strength (12-1 month momentum)
        sector_etfs = list(SECTOR_ETFS.values())
        available = [t for t in sector_etfs if t in returns.columns]

        for ticker in available:
            try:
                r = returns[ticker].dropna()
                if len(r) < TRADING_DAYS_PER_YEAR:
                    continue

                # 12-month momentum minus 1-month (standard momentum factor)
                mom_12m = float((1 + r.iloc[-TRADING_DAYS_PER_YEAR:]).prod() - 1)
                mom_1m = float((1 + r.iloc[-21:]).prod() - 1)
                momentum = mom_12m - mom_1m

                # RSI (14-day)
                delta = r.iloc[-14:]
                gain = delta.clip(lower=0).mean()
                loss = (-delta.clip(upper=0)).mean()
                rs = gain / loss if loss != 0 else 100
                rsi = 100 - (100 / (1 + rs))

                signal = "NEUTRAL"
                if rsi > 70:
                    signal = "OVERBOUGHT"
                elif rsi < 30:
                    signal = "OVERSOLD"

                # Sector name lookup
                sector_name = next((k for k, v in SECTOR_ETFS.items() if v == ticker), ticker)

                rv_signals.append({
                    "ticker": ticker,
                    "sector": sector_name,
                    "momentum_12_1": round(momentum * 100, 2),
                    "rsi_14": round(float(rsi), 1),
                    "signal": signal,
                    "relative_strength": round(mom_12m * 100, 2),
                    "source": "sector_rs",
                })
            except Exception as e:
                logger.debug(f"RV calc failed for {ticker}: {e}")

        # Pairs trading signals
        for a, b in RV_PAIRS:
            if a in returns.columns and b in returns.columns:
                try:
                    r_a = returns[a].dropna()
                    r_b = returns[b].dropna()
                    common = r_a.index.intersection(r_b.index)
                    if len(common) < 60:
                        continue

                    corr = r_a.loc[common[-60:]].corr(r_b.loc[common[-60:]])
                    spread_return = (r_a.loc[common[-20:]] - r_b.loc[common[-20:]]).sum()

                    rv_signals.append({
                        "ticker": f"{a}/{b}",
                        "sector": "Pairs",
                        "momentum_12_1": round(float(spread_return) * 100, 2),
                        "rsi_14": 0,
                        "signal": "LONG_SPREAD" if spread_return < -0.03 else "SHORT_SPREAD" if spread_return > 0.03 else "NEUTRAL",
                        "relative_strength": round(float(corr) * 100, 1),
                        "source": "pairs_rv",
                    })
                except Exception as e:
                    logger.debug(f"Pairs RV failed for {a}/{b}: {e}")

        return sorted(rv_signals, key=lambda x: abs(x["momentum_12_1"]), reverse=True)

    # ─── Correlation & Pattern Detection ──────────────────

    def detect_correlations(self, returns: pd.DataFrame) -> dict:
        """Rolling correlation matrix and breakdown alerts."""
        result = {
            "current_30d": {},
            "current_60d": {},
            "current_90d": {},
            "breakdowns": [],
        }

        if returns.empty:
            return result

        # Rolling correlation matrices
        for window, key in [(30, "current_30d"), (60, "current_60d"), (90, "current_90d")]:
            try:
                recent = returns.iloc[-window:]
                if len(recent) >= window // 2:
                    corr = recent.corr()
                    # Convert to serializable format — top triangle only
                    corr_dict = {}
                    for i, col_a in enumerate(corr.columns):
                        for j, col_b in enumerate(corr.columns):
                            if i < j:
                                val = corr.iloc[i, j]
                                if not pd.isna(val):
                                    corr_dict[f"{col_a}/{col_b}"] = round(float(val), 3)
                    result[key] = corr_dict
            except Exception as e:
                logger.debug(f"{window}d correlation calc failed: {e}")

        # Breakdown alerts: pairs historically >0.8 corr but currently <0.4
        try:
            if len(returns) >= 252:
                long_corr = returns.iloc[-252:].corr()
                short_corr = returns.iloc[-30:].corr()

                for i, col_a in enumerate(long_corr.columns):
                    for j, col_b in enumerate(long_corr.columns):
                        if i >= j:
                            continue
                        lc = long_corr.iloc[i, j]
                        sc = short_corr.iloc[i, j] if col_a in short_corr.columns and col_b in short_corr.columns else lc
                        if pd.isna(lc) or pd.isna(sc):
                            continue
                        if lc > CORR_BREAKDOWN_THRESHOLD_HIGH and sc < CORR_BREAKDOWN_THRESHOLD_LOW:
                            result["breakdowns"].append({
                                "pair": f"{col_a}/{col_b}",
                                "long_term_corr": round(float(lc), 3),
                                "current_corr": round(float(sc), 3),
                                "delta": round(float(lc - sc), 3),
                                "alert": "CORRELATION_BREAKDOWN",
                            })
        except Exception as e:
            logger.debug(f"Breakdown detection failed: {e}")

        return result

    def detect_patterns(self, prices: pd.DataFrame) -> list:
        """Use PatternRecognitionEngine for technical pattern detection."""
        patterns = []

        if PatternRecognitionEngine is None or prices.empty:
            return patterns

        try:
            pre = PatternRecognitionEngine()
            for ticker in prices.columns[:20]:  # Limit to avoid slow processing
                try:
                    series = prices[ticker].dropna()
                    if len(series) < 50:
                        continue

                    if hasattr(pre, "scan_ticker"):
                        results = pre.scan_ticker(ticker, series)
                        if results:
                            for r in (results if isinstance(results, list) else [results]):
                                if isinstance(r, dict):
                                    patterns.append(r)
                                elif hasattr(r, "__dict__"):
                                    pat = {}
                                    pat["ticker"] = ticker
                                    pat["pattern_type"] = str(getattr(r, "pattern_type", "unknown"))
                                    pat["direction"] = str(getattr(r, "direction", "UNKNOWN"))
                                    pat["confidence"] = float(getattr(r, "confidence", 0))
                                    pat["entry_price"] = float(getattr(r, "entry_price", 0))
                                    patterns.append(pat)
                except Exception as e:
                    logger.debug(f"Pattern scan failed for {ticker}: {e}")
        except Exception as e:
            logger.debug(f"PatternRecognitionEngine unavailable: {e}")

        return patterns

    # ─── Strategy Backtesting ─────────────────────────────

    def backtest_signals(self, mispricings: list, returns: pd.DataFrame) -> list:
        """Walk-forward backtest on identified mispricings."""
        results = []

        if returns.empty or not mispricings:
            return results

        for signal in mispricings[:10]:  # Top 10 signals only
            ticker = signal.get("ticker", "")
            if "/" in ticker:
                continue  # Skip pair signals for simple backtest

            if ticker not in returns.columns:
                continue

            try:
                r = returns[ticker].dropna()
                if len(r) < 60:
                    continue

                # Simple mean-reversion backtest: trade in signal direction
                # Use last 60 days of returns
                test_r = r.iloc[-60:]
                direction = 1 if signal.get("signal") == "LONG" else -1

                strategy_returns = test_r * direction
                cumulative = (1 + strategy_returns).cumprod()

                total_return = float(cumulative.iloc[-1] - 1)
                annual_return = total_return * (TRADING_DAYS_PER_YEAR / len(test_r))
                vol = float(strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                sharpe = annual_return / vol if vol > 0 else 0

                # Benchmark: buy and hold
                bh_cumulative = (1 + test_r).cumprod()
                bh_return = float(bh_cumulative.iloc[-1] - 1)

                # Max drawdown
                peak = cumulative.expanding(min_periods=1).max()
                dd = (cumulative - peak) / peak
                max_dd = float(dd.min())

                # Win rate
                wins = (strategy_returns > 0).sum()
                total = len(strategy_returns)
                win_rate = float(wins / total) if total > 0 else 0

                # Profit factor
                gross_profit = float(strategy_returns[strategy_returns > 0].sum())
                gross_loss = float(abs(strategy_returns[strategy_returns < 0].sum()))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

                results.append({
                    "ticker": ticker,
                    "strategy": f"MeanReversion_{signal.get('source', 'unknown')}",
                    "direction": signal.get("signal", ""),
                    "sharpe": round(sharpe, 3),
                    "total_return": round(total_return * 100, 2),
                    "annual_return": round(annual_return * 100, 2),
                    "max_drawdown": round(max_dd * 100, 2),
                    "win_rate": round(win_rate * 100, 1),
                    "profit_factor": round(profit_factor, 2),
                    "vs_benchmark": round((total_return - bh_return) * 100, 2),
                    "test_days": len(test_r),
                })
            except Exception as e:
                logger.debug(f"Backtest failed for {ticker}: {e}")

        return sorted(results, key=lambda x: x.get("sharpe", 0), reverse=True)

    # ─── Main Entry Point ─────────────────────────────────

    def run_evening_backtest(self, date_str: Optional[str] = None) -> dict:
        """Run full post-market analysis.

        Args:
            date_str: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Complete backtest results dict.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        logger.info(f"Starting evening backtest for {date_str}")

        # Fetch data
        start_date = (datetime.now() - timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")

        # Expand universe with RV pair tickers
        all_tickers = list(set(
            self.universe +
            [a for a, b in RV_PAIRS] +
            [b for a, b in RV_PAIRS]
        ))

        prices = pd.DataFrame()
        returns = pd.DataFrame()

        try:
            prices = get_adj_close(all_tickers, start=start_date)
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")

        try:
            returns = get_returns(all_tickers, start=start_date)
        except Exception as e:
            logger.warning(f"Returns fetch failed: {e}")

        # Run all analyses
        mispricings = self.detect_mispricings(prices, returns)
        relative_value = self.analyze_relative_value(returns)
        correlations = self.detect_correlations(returns)
        patterns = self.detect_patterns(prices)
        backtests = self.backtest_signals(mispricings, returns)

        # Get regime context
        regime = "UNKNOWN"
        try:
            if MacroEngine is not None:
                me = MacroEngine()
                snapshot = me.get_snapshot() if hasattr(me, "get_snapshot") else None
                if snapshot:
                    regime = getattr(snapshot, "regime", "UNKNOWN")
                    if hasattr(regime, "value"):
                        regime = regime.value
        except Exception:
            pass

        # Compute opportunities (aggregate high-conviction signals)
        opportunities = self._aggregate_opportunities(mispricings, relative_value, patterns, backtests)

        # Summary
        high_conviction = [o for o in opportunities if o.get("confidence", 0) > 0.6]
        expected_returns = [o.get("expected_return", 0) for o in opportunities if o.get("expected_return")]

        result = {
            "date": date_str,
            "mispricings": mispricings,
            "relative_value": relative_value,
            "correlations": correlations,
            "patterns": patterns,
            "backtests": backtests,
            "regime": regime,
            "opportunities": opportunities,
            "summary": {
                "total_opportunities": len(opportunities),
                "high_conviction": len(high_conviction),
                "avg_expected_return": round(np.mean(expected_returns), 3) if expected_returns else 0,
                "total_mispricings": len(mispricings),
                "total_rv_signals": len(relative_value),
                "total_patterns": len(patterns),
                "correlation_breakdowns": len(correlations.get("breakdowns", [])),
            },
            "generated_at": datetime.now().isoformat(),
        }

        # Save to file
        try:
            output_path = BACKTEST_DIR / f"{date_str}_evening.json"
            output_path.write_text(json.dumps(result, indent=2, default=str))
            logger.info(f"Evening backtest saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

        self._results_cache = result
        return result

    def _aggregate_opportunities(self, mispricings, rv_signals, patterns, backtests) -> list:
        """Combine signals into unified opportunity list."""
        opps = []

        # From mispricings
        for m in mispricings:
            if abs(m.get("z_score", 0)) > Z_SCORE_THRESHOLD:
                opps.append({
                    "source": "mispricing",
                    "ticker": m["ticker"],
                    "direction": m.get("signal", ""),
                    "expected_return": abs(m.get("deviation_pct", 0)),
                    "confidence": m.get("confidence", 0),
                    "detail": f"Z-score: {m.get('z_score', 0):.2f}",
                })

        # From RV signals with extreme RSI
        for rv in rv_signals:
            if rv.get("signal") in ("OVERBOUGHT", "OVERSOLD"):
                direction = "SHORT" if rv["signal"] == "OVERBOUGHT" else "LONG"
                opps.append({
                    "source": "relative_value",
                    "ticker": rv["ticker"],
                    "direction": direction,
                    "expected_return": abs(rv.get("momentum_12_1", 0)),
                    "confidence": 0.5,
                    "detail": f"RSI: {rv.get('rsi_14', 0):.0f}, Mom: {rv.get('momentum_12_1', 0):.1f}%",
                })

        # From patterns
        for p in patterns:
            direction = p.get("direction", "UNKNOWN")
            if isinstance(direction, str) and direction not in ("UNKNOWN", ""):
                opps.append({
                    "source": "pattern",
                    "ticker": p.get("ticker", ""),
                    "direction": direction,
                    "expected_return": 0,
                    "confidence": p.get("confidence", 0),
                    "detail": f"Pattern: {p.get('pattern_type', 'unknown')}",
                })

        # From backtests with positive Sharpe
        for bt in backtests:
            if bt.get("sharpe", 0) > 0.5:
                opps.append({
                    "source": "backtest",
                    "ticker": bt["ticker"],
                    "direction": bt.get("direction", ""),
                    "expected_return": bt.get("annual_return", 0),
                    "confidence": min(bt.get("sharpe", 0) / 2.0, 1.0),
                    "detail": f"Sharpe: {bt.get('sharpe', 0):.2f}, WR: {bt.get('win_rate', 0):.0f}%",
                })

        return sorted(opps, key=lambda x: x.get("confidence", 0), reverse=True)

    # ─── Cached Result Access ─────────────────────────────

    def get_latest_results(self) -> dict:
        """Get most recent backtest results, from cache or file."""
        if self._results_cache:
            return self._results_cache

        # Try to load most recent file
        if BACKTEST_DIR.exists():
            files = sorted(BACKTEST_DIR.glob("*_evening.json"), reverse=True)
            if files:
                try:
                    return json.loads(files[0].read_text())
                except Exception:
                    pass

        return {}

    def get_backtest_history(self, days: int = 7) -> list:
        """List recent backtest dates with summary stats."""
        history = []
        if not BACKTEST_DIR.exists():
            return history

        files = sorted(BACKTEST_DIR.glob("*_evening.json"), reverse=True)
        for f in files[:days]:
            try:
                data = json.loads(f.read_text())
                history.append({
                    "date": data.get("date", f.stem.replace("_evening", "")),
                    "summary": data.get("summary", {}),
                    "regime": data.get("regime", "UNKNOWN"),
                    "generated_at": data.get("generated_at", ""),
                })
            except Exception:
                pass

        return history

    def get_backtest_by_date(self, date_str: str) -> dict:
        """Get full results for a specific date."""
        path = BACKTEST_DIR / f"{date_str}_evening.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}
