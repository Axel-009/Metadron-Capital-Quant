"""
Risk router — RISK tab
Wraps: BetaCorridor, DecisionMatrix, OptionsEngine
"""
# BROKER SWAP NOTE: This router accesses broker via get_broker() which
# tries ExecutionEngine (IBKRBroker) with PaperBroker fallback.
# Risk metrics use broker._perf_tracker for daily returns (Sharpe, Sortino, etc.)
# Risk alerts use BetaCorridor + OptionsEngine (broker-independent).
# When adding IBKR: ensure _perf_tracker is populated with daily NAV snapshots.
from fastapi import APIRouter
from datetime import datetime
import logging

from engine.api.shared import get_broker, get_engine, get_beta, get_options, get_decision

logger = logging.getLogger("metadron-api.risk")
router = APIRouter()


@router.get("/portfolio")
async def risk_portfolio():
    """Full risk metrics: VaR, beta, drawdown, Sharpe, Sortino."""
    try:
        beta = get_beta()
        analytics = beta.get_corridor_analytics()
        history = beta.get_history()
        latest = history[-1] if history else None

        result = {
            "current_beta": latest.current_beta if latest and hasattr(latest, "current_beta") else 0,
            "target_beta": latest.target_beta if latest and hasattr(latest, "target_beta") else 0,
            "corridor_position": latest.corridor_position if latest and hasattr(latest, "corridor_position") else "UNKNOWN",
            "analytics": analytics if isinstance(analytics, dict) else {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return result
    except Exception as e:
        logger.error(f"risk/portfolio error: {e}")
        return {"error": str(e)}


@router.get("/greeks")
async def risk_greeks():
    """Aggregate portfolio Greeks: delta, gamma, theta, vega, rho."""
    try:
        opt = get_options()
        greeks = opt.get_portfolio_greeks()
        return {**greeks, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/greeks error: {e}")
        return {"error": str(e)}


@router.get("/options/hedge")
async def risk_options_hedge():
    """Current hedge requirements and cost drag."""
    try:
        opt = get_options()
        hedge = opt.compute_hedge_requirements()
        return {**(hedge if isinstance(hedge, dict) else {}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options/hedge error: {e}")
        return {"error": str(e)}


@router.get("/options/strategies")
async def risk_options_strategies():
    """Regime → strategy mapping."""
    try:
        opt = get_options()
        matrix = opt.regime_strategy_matrix()
        return {"strategies": matrix, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options/strategies error: {e}")
        return {"error": str(e)}


@router.get("/beta/stress")
async def risk_beta_stress():
    """Beta under stress scenarios."""
    try:
        beta = get_beta()
        df = beta.stress_test_beta()
        result = df.to_dict(orient="records") if hasattr(df, "to_dict") else []
        return {"scenarios": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/beta/stress error: {e}")
        return {"scenarios": [], "error": str(e)}


@router.get("/beta/history")
async def risk_beta_history():
    """Historical beta time-series."""
    try:
        beta = get_beta()
        df = beta.get_beta_history_df()
        if hasattr(df, "to_dict"):
            records = df.tail(200).reset_index().to_dict(orient="records")
            # Convert timestamps to strings
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return {"history": records, "timestamp": datetime.utcnow().isoformat()}
        return {"history": [], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/beta/history error: {e}")
        return {"history": [], "error": str(e)}


@router.get("/alerts")
async def risk_alerts():
    """Top risk alerts derived from current engine state."""
    try:
        beta = get_beta()
        opt = get_options()

        alerts = []
        # Beta corridor alerts
        history = beta.get_history()
        if history:
            latest = history[-1]
            b = latest.current_beta if hasattr(latest, "current_beta") else 0
            t = latest.target_beta if hasattr(latest, "target_beta") else 0
            pos = latest.corridor_position if hasattr(latest, "corridor_position") else "UNKNOWN"
            if pos == "ABOVE":
                alerts.append({"name": "Beta Above Corridor", "value": f"β={b:.2f} (target {t:.2f})", "severity": "high"})
            elif pos == "BELOW":
                alerts.append({"name": "Beta Below Corridor", "value": f"β={b:.2f} (target {t:.2f})", "severity": "medium"})

        # Options concentration
        greeks = opt.get_portfolio_greeks()
        if isinstance(greeks, dict):
            delta = greeks.get("delta", 0)
            if abs(delta) > 0.8:
                alerts.append({"name": "High Delta Exposure", "value": f"Δ={delta:.2f}", "severity": "high"})
            theta = greeks.get("theta", 0)
            if theta < -50:
                alerts.append({"name": "Theta Bleed", "value": f"Θ={theta:.1f}/day", "severity": "medium"})

        # Sector concentration from beta analytics
        analytics = beta.get_corridor_analytics()
        if isinstance(analytics, dict):
            vol_regime = analytics.get("vol_regime", "")
            if vol_regime == "HIGH":
                alerts.append({"name": "Elevated Volatility", "value": f"Vol regime: {vol_regime}", "severity": "high"})

        # Pad with default if empty
        if not alerts:
            alerts.append({"name": "No Active Alerts", "value": "System nominal", "severity": "low"})

        # Always include portfolio risk context even when no alerts fire
        try:
            broker = get_broker()
            summary = broker.get_portfolio_summary()
            dd = broker.get_drawdown() if hasattr(broker, "get_drawdown") else {}
            # Add context alerts for key metrics
            max_dd = dd.get("max_drawdown", 0)
            if max_dd > 0.15:
                alerts.append({"name": "Drawdown Warning", "value": f"{max_dd*100:.1f}% from peak", "severity": "high"})
            elif max_dd > 0.08:
                alerts.append({"name": "Drawdown Elevated", "value": f"{max_dd*100:.1f}% from peak", "severity": "medium"})

            nav = summary.get("nav", 0)
            if nav > 0:
                gross_exp = summary.get("gross_exposure", 0)
                if gross_exp > 0.9:
                    alerts.append({"name": "High Gross Exposure", "value": f"{gross_exp*100:.0f}%", "severity": "high"})
        except Exception:
            pass

        return {"alerts": alerts, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/alerts error: {e}")
        return {"alerts": [], "error": str(e)}


@router.get("/order-distribution")
async def order_distribution():
    """Order type distribution from recent trades."""
    try:
        beta = get_beta()  # Use beta to get broker access
        # Get trades from the execution engine broker
        try:
            eng = get_engine()
            trades = eng.broker.get_trade_history()[-200:]
        except Exception:
            trades = []

        if not trades:
            return {"distribution": [], "timestamp": datetime.utcnow().isoformat()}

        # Count signal types as order distribution
        type_counts: dict[str, int] = {}
        for t in trades:
            raw_sig = t.get("signal_type", "UNKNOWN") if isinstance(t, dict) else getattr(t, "signal_type", "UNKNOWN")
            sig = raw_sig.value if hasattr(raw_sig, "value") else str(raw_sig)
            type_counts[sig] = type_counts.get(sig, 0) + 1

        total = sum(type_counts.values()) or 1
        colors = ["#00d4aa", "#58a6ff", "#f85149", "#bc8cff", "#d29922", "#4ecdc4", "#3fb950"]
        distribution = []
        for i, (name, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
            distribution.append({
                "name": name,
                "value": round(count / total * 100, 1),
                "count": count,
                "color": colors[i % len(colors)],
            })

        return {"distribution": distribution, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/order-distribution error: {e}")
        return {"distribution": [], "error": str(e)}


@router.get("/metrics")
async def risk_metrics():
    """Portfolio risk metrics: Sharpe, Sortino, Calmar, Information, Treynor.

    All ratios are computed from actual daily returns (NAV series) via the
    broker's PerformanceTracker.  No Sharpe-multiplier approximations.
    """
    try:
        import numpy as np

        beta = get_beta()
        broker = get_broker()
        history = beta.get_history()

        # ── Pull real daily returns from the broker's NAV series ──────────
        returns = np.array([], dtype=np.float64)
        if hasattr(broker, "_perf_tracker"):
            returns = broker._perf_tracker.get_daily_returns()

        TRADING_DAYS = 252
        daily_rf = 0.05 / TRADING_DAYS  # ~5 % risk-free default

        # ── Sharpe ────────────────────────────────────────────────────────
        if len(returns) >= 2:
            excess = returns - daily_rf
            sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS))
        else:
            sharpe = 0.0

        # ── Sortino (downside deviation uses only negative excess returns) ─
        if len(returns) >= 2:
            excess = returns - daily_rf
            downside = excess[excess < 0]
            downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
            sortino = float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS)) if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        # ── Max Drawdown (from running-max NAV) ───────────────────────────
        dd_info = broker.get_drawdown() if hasattr(broker, "get_drawdown") else {}
        max_dd = dd_info.get("max_drawdown", 0.0)        # fraction, e.g. 0.08
        max_dd_pct = max_dd * 100                         # percentage for display

        # ── Calmar (annualised return / max drawdown) ─────────────────────
        if len(returns) >= 2 and max_dd > 0:
            annualized_return = float(np.mean(returns) * TRADING_DAYS)
            calmar = annualized_return / max_dd
        else:
            calmar = 0.0

        # ── Information Ratio (excess return over benchmark / tracking err)
        #    Benchmark = SPY returns via BetaCorridor history (Rm field).
        if len(returns) >= 2 and history and len(history) >= 2:
            bench_rets = np.array(
                [h.Rm for h in history if hasattr(h, "Rm")], dtype=np.float64
            )
            # Align lengths (history may differ from daily NAV count)
            min_len = min(len(returns), len(bench_rets))
            if min_len >= 2:
                port_tail = returns[-min_len:]
                bench_tail = bench_rets[-min_len:]
                active = port_tail - bench_tail
                tracking_err = float(np.std(active, ddof=1))
                info_ratio = float(np.mean(active) / tracking_err * np.sqrt(TRADING_DAYS)) if tracking_err > 0 else 0.0
            else:
                info_ratio = 0.0
        else:
            info_ratio = 0.0

        # ── Treynor (annualised excess return / portfolio beta) ───────────
        betas = [h.current_beta for h in history if hasattr(h, "current_beta")] if history else []
        avg_beta = float(np.mean(betas)) if betas else 0.0
        if len(returns) >= 2 and avg_beta != 0:
            annualized_excess = float((np.mean(returns) - daily_rf) * TRADING_DAYS)
            treynor = annualized_excess / avg_beta
        else:
            treynor = 0.0

        # ── Build response (same shape as before) ─────────────────────────
        metrics = [
            {"name": "Sharpe Ratio",      "value": f"{sharpe:.2f}",      "status": "good" if sharpe > 1.5 else "warning" if sharpe > 0 else "bad"},
            {"name": "Sortino Ratio",     "value": f"{sortino:.2f}",     "status": "good" if sortino > 2 else "warning" if sortino > 0 else "bad"},
            {"name": "Max Drawdown",      "value": f"{max_dd_pct:.2f}%", "status": "warning" if max_dd_pct > 20 else "good"},
            {"name": "Calmar Ratio",      "value": f"{calmar:.2f}",      "status": "good" if calmar > 1 else "neutral"},
            {"name": "Information Ratio", "value": f"{info_ratio:.2f}",  "status": "good" if info_ratio > 0.5 else "neutral"},
            {"name": "Treynor Ratio",     "value": f"{treynor:.1f}%",    "status": "good" if treynor > 0 else "bad"},
        ]
        return {"metrics": metrics, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/metrics error: {e}")
        return {"metrics": [], "error": str(e)}


@router.get("/fills")
async def risk_fills():
    """Recent order fills from broker."""
    try:
        eng = get_engine()
        trades = eng.broker.get_trade_history()[-20:]
        fills = []
        for t in trades:
            ts = t.get("fill_timestamp", "") if isinstance(t, dict) else getattr(t, "fill_timestamp", "")
            time_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:8] if ts else ""
            raw_side = t.get("side", "") if isinstance(t, dict) else getattr(t, "side", "")
            side_str = raw_side.value if hasattr(raw_side, "value") else str(raw_side)
            raw_sig = t.get("signal_type", "") if isinstance(t, dict) else getattr(t, "signal_type", "")
            sig_str = raw_sig.value if hasattr(raw_sig, "value") else str(raw_sig)
            fill_price = t.get("fill_price", 0) if isinstance(t, dict) else getattr(t, "fill_price", 0)
            fills.append({
                "time": time_str,
                "pair": t.get("ticker", "") if isinstance(t, dict) else getattr(t, "ticker", ""),
                "side": side_str,
                "qty": t.get("quantity", 0) if isinstance(t, dict) else getattr(t, "quantity", 0),
                "price": fill_price,
                "status": "FILLED" if fill_price and fill_price > 0 else "NO FILL",
                "strategy": sig_str,
            })
        return {"fills": fills, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/fills error: {e}")
        return {"fills": [], "error": str(e)}


@router.get("/options-positions")
async def risk_options_positions():
    """Options positions from OptionsEngine."""
    try:
        opt = get_options()
        greeks = opt.get_portfolio_greeks()
        positions = []
        # Get positions if available
        if hasattr(opt, "positions"):
            for p in opt.positions:
                positions.append({
                    "ticker": getattr(p, "underlying", ""),
                    "type": getattr(p, "option_type", "").upper(),
                    "strike": getattr(p, "strike", 0),
                    "expiry": str(getattr(p, "expiry", "")),
                    "qty": getattr(p, "quantity", 0),
                    "delta": getattr(p, "greeks", {}).get("delta", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "gamma": getattr(p, "greeks", {}).get("gamma", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "theta": getattr(p, "greeks", {}).get("theta", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "vega": getattr(p, "greeks", {}).get("vega", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "pnl": getattr(p, "pnl", 0),
                })
        agg = {
            "delta": greeks.get("delta", 0) if isinstance(greeks, dict) else 0,
            "gamma": greeks.get("gamma", 0) if isinstance(greeks, dict) else 0,
            "theta": greeks.get("theta", 0) if isinstance(greeks, dict) else 0,
            "vega": greeks.get("vega", 0) if isinstance(greeks, dict) else 0,
            "total_pnl": sum(p.get("pnl", 0) for p in positions),
        }
        return {"positions": positions, "aggregate": agg, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options-positions error: {e}")
        return {"positions": [], "aggregate": {}, "error": str(e)}


@router.get("/futures-positions")
async def risk_futures_positions():
    """Futures positions from broker."""
    try:
        broker = get_broker()
        positions = broker.get_all_positions()
        futures = []
        total_pnl = 0
        total_margin = 0
        total_notional = 0
        for ticker, pos in positions.items():
            # Filter for futures-like tickers (ES, NQ, CL, GC, ZB, ZN, 6E)
            if not any(ticker.startswith(f) for f in ["ES", "NQ", "YM", "CL", "GC", "ZB", "ZN", "6E", "RTY", "VX"]):
                continue
            pnl = getattr(pos, "unrealized_pnl", 0)
            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            entry = getattr(pos, "avg_cost", 0)
            notional = abs(qty * price * 50)  # Approximate multiplier
            margin = abs(qty) * 12000  # Approximate margin
            futures.append({
                "contract": ticker,
                "side": "LONG" if qty > 0 else "SHORT",
                "qty": abs(qty),
                "entry": round(entry, 2),
                "last": round(price, 2),
                "pnl": round(pnl, 2),
                "margin": round(margin, 2),
                "notional": round(notional, 2),
            })
            total_pnl += pnl
            total_margin += margin
            total_notional += notional

        return {
            "positions": futures,
            "totals": {"pnl": round(total_pnl, 2), "margin": round(total_margin, 2), "notional": round(total_notional, 2)},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"risk/futures-positions error: {e}")
        return {"positions": [], "totals": {}, "error": str(e)}


@router.get("/margin")
async def risk_margin():
    """Margin information from broker."""
    try:
        broker = get_broker()
        state = broker.get_portfolio_summary()
        s = state if isinstance(state, dict) else {}
        nav = s.get("nav", 0)
        cash = s.get("cash", 0)
        gross = s.get("gross_exposure", 0)

        # Derive margin metrics from portfolio state
        margin_used = gross * 0.3 if gross else 0  # ~30% margin on gross
        margin_available = nav - margin_used
        utilization = (margin_used / nav * 100) if nav > 0 else 0
        buying_power = nav * 4  # Reg-T 4x for pattern day traders
        maintenance = margin_used * 0.75

        return {
            "margin": {
                "regT": round(nav * 0.5, 2),
                "portfolioMargin": round(margin_used, 2),
                "marginUsed": round(margin_used, 2),
                "marginAvailable": round(margin_available, 2),
                "maintenanceMargin": round(maintenance, 2),
                "utilizationPct": round(utilization, 1),
                "buyingPower": round(buying_power, 2),
                "sma": round(margin_available * 0.5, 2),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"risk/margin error: {e}")
        return {"margin": {}, "error": str(e)}


@router.get("/liquidity-scoring")
async def risk_liquidity_scoring():
    """Liquidity risk scoring for portfolio positions."""
    try:
        broker = get_broker()
        positions = broker.get_all_positions()
        scoring = []
        for ticker, pos in positions.items():
            # TODO: Add avg_volume field to Position dataclass or fetch from OpenBB cache
            vol = getattr(pos, "avg_volume", None) or getattr(pos, "volume", None) or 0
            price = getattr(pos, "current_price", 0) or 1
            qty = abs(getattr(pos, "quantity", 0))
            # Simple liquidity score: higher volume = more liquid
            score = min(100, int(vol / 1000)) if vol else 50
            adv_str = f"{vol / 1e6:.1f}M" if vol > 1e6 else f"{vol / 1e3:.0f}K" if vol > 0 else "—"
            impact = "Low" if score > 80 else "Medium" if score > 50 else "High"
            spread_est = 0.01 if score > 80 else 0.05 if score > 50 else 0.15
            scoring.append({
                "asset": ticker,
                "score": score,
                "adv": adv_str,
                "spread": f"{spread_est:.2f}%",
                "impact": impact,
            })

        scoring.sort(key=lambda x: -x["score"])
        return {"scoring": scoring[:10], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/liquidity-scoring error: {e}")
        return {"scoring": [], "error": str(e)}
