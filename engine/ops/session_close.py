"""Post-session file generation for Metadron Capital.

Runs after market close (16:00 ET) to generate:
1. Daily transaction log (TX) — all trades executed during the session
2. Machine learning log — model predictions vs outcomes
3. Learning loop snapshot — signal accuracy, tier weight adjustments
4. Reconciliation log — IBKR vs trade log broker position comparison
5. Error/alert log — all engine errors during the session

These files feed the ARCHIVE tab and provide audit trail for the system.

Usage:
    from engine.ops.session_close import generate_session_files
    generate_session_files()
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("metadron.ops.session_close")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def _ensure_dirs():
    """Create all log directories if they don't exist."""
    for subdir in ["transactions", "reconciliation", "learning", "ml_models", "errors", "platinum", "portfolio", "reports"]:
        (LOGS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def generate_tx_log() -> Path:
    """Generate daily transaction log from broker trade history.

    Source: ExecutionEngine → IBKRBroker.get_trades()
    Output: logs/transactions/TX_{YYYYMMDD}.json
    """
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y%m%d")
    output = LOGS_DIR / "transactions" / f"TX_{date_str}.json"

    try:
        from engine.execution.execution_engine import ExecutionEngine
        eng = ExecutionEngine()
        trades = eng.broker.get_trades(limit=5000)

        records = []
        for t in trades:
            records.append({
                "id": str(getattr(t, "id", "")),
                "ticker": getattr(t, "ticker", ""),
                "side": t.side.value if hasattr(t.side, "value") else str(getattr(t, "side", "")),
                "quantity": getattr(t, "quantity", 0),
                "fill_price": getattr(t, "fill_price", 0),
                "signal_type": t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "")),
                "timestamp": t.fill_timestamp.isoformat() if hasattr(t, "fill_timestamp") and t.fill_timestamp else "",
                "reason": getattr(t, "reason", ""),
            })

        payload = {
            "date": date_str,
            "total_trades": len(records),
            "trades": records,
            "generated_at": datetime.now().isoformat(),
        }
        output.write_text(json.dumps(payload, indent=2))
        logger.info(f"TX log generated: {output} ({len(records)} trades)")
        return output
    except Exception as e:
        logger.error(f"TX log generation failed: {e}")
        output.write_text(json.dumps({"date": date_str, "error": str(e), "trades": []}))
        return output


def generate_recon_log() -> Path:
    """Generate reconciliation log comparing IBKRBroker vs trade log.

    The only expected difference should be futures positions (no futures broker
    on IBKR). All equity/ETF/options positions should match.

    Source: AlpacaBroker.get_positions() vs PaperBroker.get_positions()
    Output: logs/reconciliation/RECON_{YYYYMMDD}.json
    """
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y%m%d")
    output = LOGS_DIR / "reconciliation" / f"RECON_{date_str}.json"

    try:
        # Get trade log positions
        from engine.execution.paper_broker import PaperBroker
        paper = PaperBroker()
        paper_positions = paper.get_all_positions()

        # Get IBKRBroker positions
        broker_positions = {}
        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            alpaca = AlpacaBroker(initial_cash=0, paper=True)
            broker_positions = ibkr.get_positions()
        except Exception as e:
            logger.warning(f"IBKR broker unavailable for recon: {e}")

        # Compare positions
        all_tickers = set(list(paper_positions.keys()) + list(broker_positions.keys()))
        FUTURES_PREFIXES = ("ES", "NQ", "YM", "CL", "GC", "ZB", "ZN", "6E", "RTY", "VX")

        matches = []
        mismatches = []
        paper_only = []
        ibkr_only = []

        for ticker in sorted(all_tickers):
            is_futures = any(ticker.startswith(p) for p in FUTURES_PREFIXES)
            in_paper = ticker in paper_positions
            in_ibkr = ticker in broker_positions

            if in_paper and in_ibkr:
                p = paper_positions[ticker]
                a = broker_positions[ticker]
                p_qty = getattr(p, "quantity", 0) if hasattr(p, "quantity") else p.get("quantity", 0)
                a_qty = a.get("quantity", getattr(a, "quantity", 0)) if isinstance(a, dict) else getattr(a, "quantity", 0)

                if p_qty == a_qty:
                    matches.append({"ticker": ticker, "qty": p_qty, "status": "MATCHED"})
                else:
                    mismatches.append({
                        "ticker": ticker,
                        "paper_qty": p_qty,
                        "ibkr_qty": a_qty,
                        "delta": p_qty - a_qty,
                        "is_futures": is_futures,
                        "status": "EXPECTED_DIFF" if is_futures else "MISMATCH",
                    })
            elif in_paper and not in_ibkr:
                p = paper_positions[ticker]
                p_qty = getattr(p, "quantity", 0) if hasattr(p, "quantity") else p.get("quantity", 0)
                paper_only.append({
                    "ticker": ticker, "qty": p_qty,
                    "is_futures": is_futures,
                    "status": "EXPECTED_PAPER_ONLY" if is_futures else "PAPER_ONLY",
                })
            elif in_ibkr and not in_paper:
                a = broker_positions[ticker]
                a_qty = a.get("quantity", getattr(a, "quantity", 0)) if isinstance(a, dict) else getattr(a, "quantity", 0)
                ibkr_only.append({"ticker": ticker, "qty": a_qty, "status": "IBKR_ONLY"})

        # NAV comparison
        paper_nav = paper.compute_nav()
        ibkr_nav = 0
        try:
            ibkr_nav = ibkr.compute_nav()
        except Exception:
            pass

        payload = {
            "date": date_str,
            "matches": matches,
            "mismatches": mismatches,
            "paper_only": paper_only,
            "ibkr_only": ibkr_only,
            "summary": {
                "total_positions": len(all_tickers),
                "matched": len(matches),
                "mismatched": len(mismatches),
                "paper_only": len(paper_only),
                "ibkr_only": len(ibkr_only),
                "futures_expected_diffs": sum(1 for m in mismatches if m.get("is_futures")) + sum(1 for p in paper_only if p.get("is_futures")),
                "paper_nav": paper_nav,
                "ibkr_nav": ibkr_nav,
                "nav_delta": paper_nav - ibkr_nav,
                "nav_delta_pct": ((paper_nav - ibkr_nav) / max(ibkr_nav, 1)) * 100 if ibkr_nav else 0,
            },
            "generated_at": datetime.now().isoformat(),
        }
        output.write_text(json.dumps(payload, indent=2))
        logger.info(f"Recon log generated: {output} (matched={len(matches)}, mismatches={len(mismatches)})")
        return output
    except Exception as e:
        logger.error(f"Recon log generation failed: {e}")
        output.write_text(json.dumps({"date": date_str, "error": str(e)}))
        return output


def generate_learning_log() -> Path:
    """Generate learning loop snapshot.

    Source: LearningLoop state — engine accuracies, tier weights, regime calibration
    Output: logs/learning/LEARNING_{YYYYMMDD}.json
    """
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y%m%d")
    output = LOGS_DIR / "learning" / f"LEARNING_{date_str}.json"

    try:
        from engine.monitoring.learning_loop import LearningLoop
        ll = LearningLoop()
        if hasattr(ll, "persist_snapshot"):
            ll.persist_snapshot()

        state = {}
        if hasattr(ll, "get_state"):
            state = ll.get_state()
        elif hasattr(ll, "summary"):
            state = ll.summary()

        payload = {
            "date": date_str,
            "state": state if isinstance(state, dict) else str(state),
            "generated_at": datetime.now().isoformat(),
        }
        output.write_text(json.dumps(payload, indent=2, default=str))
        logger.info(f"Learning log generated: {output}")
        return output
    except Exception as e:
        logger.error(f"Learning log generation failed: {e}")
        output.write_text(json.dumps({"date": date_str, "error": str(e)}))
        return output


def generate_ml_log() -> Path:
    """Generate ML model performance log.

    Source: All ML engines — import status, accuracy metrics
    Output: logs/ml_models/ML_{YYYYMMDD}.json
    """
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y%m%d")
    output = LOGS_DIR / "ml_models" / f"ML_{date_str}.json"

    try:
        import importlib
        engine_map = [
            ("L1", "UniverseEngine", "engine.data.universe_engine"),
            ("L2", "MacroEngine", "engine.signals.macro_engine"),
            ("L2", "MetadronCube", "engine.signals.metadron_cube"),
            ("L2", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("L3", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L4", "BetaCorridor", "engine.portfolio.beta_corridor"),
            ("L5", "ExecutionEngine", "engine.execution.execution_engine"),
            ("L6", "ResearchBots", "engine.agents.research_bots"),
            ("L7", "L7Execution", "engine.execution.l7_unified_execution_surface"),
        ]

        modules = []
        for layer, name, path in engine_map:
            try:
                importlib.import_module(path)
                modules.append({"layer": layer, "name": name, "status": "online"})
            except Exception as ex:
                modules.append({"layer": layer, "name": name, "status": "offline", "error": str(ex)[:200]})

        online = sum(1 for m in modules if m["status"] == "online")
        payload = {
            "date": date_str,
            "modules": modules,
            "online": online,
            "total": len(modules),
            "generated_at": datetime.now().isoformat(),
        }
        output.write_text(json.dumps(payload, indent=2))
        logger.info(f"ML log generated: {output} ({online}/{len(modules)} online)")
        return output
    except Exception as e:
        logger.error(f"ML log generation failed: {e}")
        output.write_text(json.dumps({"date": date_str, "error": str(e)}))
        return output


def generate_error_log() -> Path:
    """Collect all engine errors from the current session into a single log.

    Source: Python logging handlers + engine exception traces
    Output: logs/errors/ERRORS_{YYYYMMDD}.json
    """
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y%m%d")
    output = LOGS_DIR / "errors" / f"ERRORS_{date_str}.json"

    try:
        # Collect from centralized error handler
        from engine.ops.error_logger import get_session_errors
        errors = get_session_errors()
    except ImportError:
        errors = []

    payload = {
        "date": date_str,
        "errors": errors,
        "total": len(errors),
        "generated_at": datetime.now().isoformat(),
    }
    output.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Error log generated: {output} ({len(errors)} errors)")
    return output


def generate_session_files() -> dict:
    """Generate all post-session files. Called by run_close.py after market close.

    Returns dict of generated file paths.
    """
    logger.info("Starting post-session file generation...")
    _ensure_dirs()

    files = {}
    files["tx_log"] = str(generate_tx_log())
    files["recon_log"] = str(generate_recon_log())
    files["learning_log"] = str(generate_learning_log())
    files["ml_log"] = str(generate_ml_log())
    files["error_log"] = str(generate_error_log())

    logger.info(f"Post-session files generated: {list(files.keys())}")
    return files
