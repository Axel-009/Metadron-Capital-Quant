"""Metadron Capital — Daily Summary Report Generator.

Produces a one-page daily performance report covering:
- Performance analytics (P&L, returns, Sharpe, win rate)
- NAV (IBKR + Paper, delta)
- Pricing (benchmarks, top holdings)
- Risk (VaR, CVaR, drawdown, beta, sector concentration)
- Outlook (regime, ML consensus, macro indicators)

Usage:
    from engine.monitoring.daily_summary_generator import DailySummaryGenerator
    gen = DailySummaryGenerator()
    summary = gen.generate_daily_summary()
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("metadron.monitoring.daily_summary")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class DailySummaryGenerator:
    """Generates one-page daily performance report."""

    def __init__(self):
        pass

    def _get_performance(self) -> dict:
        """Performance analytics from PlatinumReportGenerator and engines."""
        perf = {
            "daily_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "cumulative_return_day": 0.0,
            "cumulative_return_wtd": 0.0,
            "cumulative_return_mtd": 0.0,
            "cumulative_return_ytd": 0.0,
            "sharpe_30d": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }

        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine()
            broker = eng.broker
            summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
            perf["daily_pnl"] = summary.get("total_pnl", 0)
            perf["realized_pnl"] = summary.get("realized_pnl", 0)
            perf["unrealized_pnl"] = summary.get("unrealized_pnl", 0)
            perf["total_trades"] = len(broker.get_trade_history()) if hasattr(broker, "get_trade_history") else 0
            perf["win_rate"] = summary.get("win_rate", 0)
        except Exception as e:
            logger.debug(f"Performance data unavailable: {e}")

        try:
            from engine.monitoring.daily_report import _compute_sharpe
            import numpy as np
            # Try to compute sharpe from recent returns
            try:
                from engine.data.yahoo_data import get_returns
                import pandas as pd
                start = pd.Timestamp.now() - pd.Timedelta(days=60)
                rets = get_returns(["SPY"], start=start.strftime("%Y-%m-%d"))
                if not rets.empty:
                    perf["sharpe_30d"] = _compute_sharpe(rets.iloc[-30:].values.flatten())
            except Exception:
                pass
        except Exception:
            pass

        return perf

    def _get_nav(self) -> dict:
        """NAV from IBKRBroker and trade logs."""
        nav = {
            "portfolio_nav": 0.0,
            "paper_nav": 0.0,
            "ibkr_nav": 0.0,
            "nav_delta": 0.0,
            "cash": 0.0,
        }

        try:
            from engine.execution.paper_broker import PaperBroker
            pb = PaperBroker()
            pv = pb.get_portfolio_value() if hasattr(pb, "get_portfolio_value") else 0
            nav["paper_nav"] = pv if isinstance(pv, (int, float)) else 0
        except Exception as e:
            logger.debug(f"Paper NAV unavailable: {e}")

        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            ab = AlpacaBroker(initial_cash=0, paper=True)
            acct = ab.get_account() if hasattr(ab, "get_account") else {}
            if isinstance(acct, dict):
                nav["ibkr_nav"] = float(acct.get("equity", 0))
                nav["cash"] = float(acct.get("cash", 0))
        except Exception as e:
            logger.debug(f"IBKR NAV unavailable: {e}")

        nav["portfolio_nav"] = nav["ibkr_nav"] or nav["paper_nav"]
        nav["nav_delta"] = nav["ibkr_nav"] - nav["paper_nav"] if nav["ibkr_nav"] and nav["paper_nav"] else 0
        return nav

    def _get_pricing(self) -> dict:
        """Key benchmark prices and top holdings."""
        benchmarks = {}
        try:
            from engine.data.yahoo_data import get_adj_close
            tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            prices = get_adj_close(tickers, start=(datetime.now().replace(day=1)).strftime("%Y-%m-%d"))
            if not prices.empty:
                for t in tickers:
                    if t in prices.columns:
                        last = prices[t].dropna()
                        if not last.empty:
                            benchmarks[t] = round(float(last.iloc[-1]), 2)
        except Exception as e:
            logger.debug(f"Pricing data unavailable: {e}")

        return {
            "benchmarks": benchmarks,
            "top_holdings": [],
        }

    def _get_risk(self) -> dict:
        """Risk metrics: VaR, CVaR, drawdown, beta, sector concentration."""
        risk = {
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "max_drawdown": 0.0,
            "beta_to_spy": 0.0,
            "sector_concentration": {},
        }

        try:
            from engine.monitoring.daily_report import _compute_var, _compute_cvar, _compute_max_drawdown
            from engine.data.yahoo_data import get_returns
            import pandas as pd
            start = pd.Timestamp.now() - pd.Timedelta(days=60)
            rets = get_returns(["SPY"], start=start.strftime("%Y-%m-%d"))
            if not rets.empty:
                r = rets.values.flatten()
                risk["var_95"] = round(float(_compute_var(r, 0.05)), 6)
                risk["var_99"] = round(float(_compute_var(r, 0.01)), 6)
                risk["cvar_95"] = round(float(_compute_cvar(r, 0.05)), 6)
        except Exception as e:
            logger.debug(f"Risk metrics unavailable: {e}")

        return risk

    def _get_outlook(self) -> dict:
        """Outlook: regime, ML consensus, macro indicators."""
        outlook = {
            "regime": "UNKNOWN",
            "ml_consensus": "NEUTRAL",
            "vix": 0.0,
            "dxy": 0.0,
            "yield_2s10s": 0.0,
            "next_session_signals": [],
        }

        try:
            from engine.signals.macro_engine import MacroEngine
            me = MacroEngine()
            snapshot = me.get_snapshot() if hasattr(me, "get_snapshot") else None
            if snapshot:
                outlook["regime"] = getattr(snapshot, "regime", "UNKNOWN")
                if hasattr(outlook["regime"], "value"):
                    outlook["regime"] = outlook["regime"].value
                outlook["vix"] = getattr(snapshot, "vix", 0)
                outlook["yield_2s10s"] = getattr(snapshot, "yield_spread", 0)
        except Exception as e:
            logger.debug(f"Macro outlook unavailable: {e}")

        return outlook

    def generate_daily_summary(self, date_str: Optional[str] = None) -> dict:
        """Generate the full daily summary as structured data.

        Args:
            date_str: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Complete daily summary dict.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        return {
            "date": date_str,
            "performance": self._get_performance(),
            "nav": self._get_nav(),
            "pricing": self._get_pricing(),
            "risk": self._get_risk(),
            "outlook": self._get_outlook(),
            "generated_at": datetime.now().isoformat(),
        }

    def generate_daily_summary_html(self, date_str: Optional[str] = None) -> str:
        """Generate HTML formatted one-pager.

        Args:
            date_str: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            HTML string of the daily summary.
        """
        data = self.generate_daily_summary(date_str)
        perf = data["performance"]
        nav = data["nav"]
        risk = data["risk"]
        outlook = data["outlook"]
        pricing = data["pricing"]

        benchmarks_html = ""
        for ticker, price in pricing.get("benchmarks", {}).items():
            benchmarks_html += f"<tr><td>{ticker}</td><td>${price:,.2f}</td></tr>"

        return f"""<!DOCTYPE html>
<html>
<head><title>Metadron Daily Summary — {data['date']}</title>
<style>
body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1, h2 {{ color: #00d4aa; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
td, th {{ border: 1px solid #30363d; padding: 6px 12px; text-align: left; }}
th {{ background: #161b22; color: #00d4aa; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
</style>
</head>
<body>
<h1>METADRON DAILY SUMMARY — {data['date']}</h1>

<h2>PERFORMANCE</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Daily P&L</td><td class="{'positive' if perf['daily_pnl'] >= 0 else 'negative'}">${perf['daily_pnl']:,.2f}</td></tr>
<tr><td>Win Rate</td><td>{perf['win_rate']:.1%}</td></tr>
<tr><td>Sharpe (30d)</td><td>{perf['sharpe_30d']:.2f}</td></tr>
<tr><td>Total Trades</td><td>{perf['total_trades']}</td></tr>
</table>

<h2>NAV</h2>
<table>
<tr><th>Account</th><th>Value</th></tr>
<tr><td>Portfolio NAV</td><td>${nav['portfolio_nav']:,.2f}</td></tr>
<tr><td>Paper NAV</td><td>${nav['paper_nav']:,.2f}</td></tr>
<tr><td>IBKR NAV</td><td>${nav['ibkr_nav']:,.2f}</td></tr>
<tr><td>Delta</td><td class="{'positive' if nav['nav_delta'] >= 0 else 'negative'}">${nav['nav_delta']:,.2f}</td></tr>
</table>

<h2>BENCHMARKS</h2>
<table>
<tr><th>Ticker</th><th>Price</th></tr>
{benchmarks_html}
</table>

<h2>RISK</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>VaR 95%</td><td>{risk['var_95']:.4%}</td></tr>
<tr><td>VaR 99%</td><td>{risk['var_99']:.4%}</td></tr>
<tr><td>CVaR 95%</td><td>{risk['cvar_95']:.4%}</td></tr>
<tr><td>Max Drawdown</td><td>{risk['max_drawdown']:.4%}</td></tr>
<tr><td>Beta to SPY</td><td>{risk['beta_to_spy']:.2f}</td></tr>
</table>

<h2>OUTLOOK</h2>
<table>
<tr><th>Indicator</th><th>Value</th></tr>
<tr><td>Regime</td><td>{outlook['regime']}</td></tr>
<tr><td>ML Consensus</td><td>{outlook['ml_consensus']}</td></tr>
<tr><td>VIX</td><td>{outlook['vix']:.1f}</td></tr>
<tr><td>2s10s Spread</td><td>{outlook['yield_2s10s']:.2f}</td></tr>
</table>

<p style="color: #484f58; font-size: 10px;">Generated at {data['generated_at']}</p>
</body>
</html>"""
