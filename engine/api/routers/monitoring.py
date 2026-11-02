"""
Monitoring router — REPORTS, TECH, ARCHIVE tabs
Wraps: PlatinumReport, DailyReport, PortfolioReport, AnomalyDetector,
       SectorTracker, LearningLoop, MemoryMonitor
"""
from fastapi import APIRouter, Query
from datetime import datetime
from pathlib import Path
import logging
import os

logger = logging.getLogger("metadron-api.monitoring")
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_anomaly = None
_sector_tracker = None
_learning = None


def _get_anomaly():
    global _anomaly
    if _anomaly is None:
        from engine.monitoring.anomaly_detector import AnomalyDetector
        _anomaly = AnomalyDetector()
    return _anomaly


def _get_sector_tracker():
    global _sector_tracker
    if _sector_tracker is None:
        from engine.monitoring.sector_tracker import SectorTracker
        _sector_tracker = SectorTracker()
    return _sector_tracker


def _get_learning():
    global _learning
    if _learning is None:
        from engine.monitoring.learning_loop import LearningLoop
        _learning = LearningLoop()
    return _learning


# ─── REPORTS tab ───────────────────────────────────────────

@router.get("/reports/list")
async def reports_list():
    """Available reports and their last generation timestamps."""
    try:
        logs_dir = PROJECT_ROOT / "logs"
        reports = []

        report_types = [
            ("platinum", "Platinum Report", "Comprehensive portfolio overview with attribution analysis and risk decomposition for C-suite stakeholders.", "Executive"),
            ("daily", "Daily P&L Report", "Detailed daily profit & loss breakdown by strategy, sector, and individual position.", "Operations"),
            ("portfolio", "Portfolio Analytics", "Factor exposure analysis, correlation matrices, and regime classification report.", "Research"),
            ("risk", "Risk Dashboard", "VaR analysis, stress test results, Greeks exposure, and liquidity risk assessment.", "Risk"),
            ("execution", "Execution Quality", "Trade execution analysis: slippage, fill rates, market impact, and broker comparison.", "Trading"),
            ("investor", "Monthly Investor", "Monthly performance letter with NAV history, benchmark comparison, and market outlook.", "Investor"),
            ("compliance", "Compliance Report", "Regulatory compliance checks, position limits, and concentration risk analysis.", "Compliance"),
            ("ml-model", "ML Model Report", "Model performance metrics, feature importance, and prediction accuracy analysis.", "Research"),
        ]

        for folder, name, desc, rtype in report_types:
            report_dir = logs_dir / folder
            last_file = None
            last_time = None

            if report_dir.exists():
                files = sorted(report_dir.glob("*.txt"), key=os.path.getmtime, reverse=True)
                if not files:
                    files = sorted(report_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
                if files:
                    last_file = files[0].name
                    last_time = datetime.fromtimestamp(os.path.getmtime(files[0])).isoformat()

            reports.append({
                "id": folder,
                "name": name,
                "type": rtype,
                "description": desc,
                "last_generated": last_time,
                "last_file": last_file,
                "status": "ready" if last_file else "not_generated",
            })

        return {"reports": reports, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/reports/list error: {e}")
        return {"reports": [], "error": str(e)}


@router.get("/reports/generate")
async def generate_report(report_type: str = Query("platinum")):
    """Generate a report on demand.

    Supported types: platinum, daily, portfolio, risk, execution,
    investor, compliance, ml-model.
    """
    try:
        result = None
        log_dir = None

        if report_type == "platinum":
            from engine.monitoring.platinum_report import PlatinumReportGenerator
            gen = PlatinumReportGenerator()
            result = gen.generate_close_report() if hasattr(gen, "generate_close_report") else str(gen.generate() if hasattr(gen, "generate") else "Not available")
            log_dir = PROJECT_ROOT / "logs" / "platinum"

        elif report_type == "daily":
            from engine.monitoring.daily_report import generate_close_report, save_report
            report_dict = generate_close_report({})
            path = save_report(report_dict)
            result = f"Daily report generated: {path}"
            log_dir = None  # save_report already writes

        elif report_type == "portfolio":
            from engine.monitoring.portfolio_report import PortfolioReportGenerator
            gen = PortfolioReportGenerator()
            result = gen.generate_close_report() if hasattr(gen, "generate_close_report") else "Not available"
            log_dir = PROJECT_ROOT / "logs" / "portfolio"

        elif report_type == "risk":
            from engine.monitoring.risk_report import RiskReportGenerator
            gen = RiskReportGenerator()
            result = gen.generate()
            log_dir = PROJECT_ROOT / "logs" / "risk"

        elif report_type == "execution":
            from engine.api.shared import get_engine
            eng = get_engine()
            result = eng.format_execution_report() if hasattr(eng, "format_execution_report") else "Not available"
            log_dir = PROJECT_ROOT / "logs" / "execution"

        elif report_type == "investor":
            from engine.monitoring.investor_report import InvestorReportGenerator
            gen = InvestorReportGenerator()
            result = gen.generate()
            log_dir = PROJECT_ROOT / "logs" / "investor"

        elif report_type == "compliance":
            from engine.monitoring.compliance_report import ComplianceReportGenerator
            gen = ComplianceReportGenerator()
            result = gen.generate()
            log_dir = PROJECT_ROOT / "logs" / "compliance"

        elif report_type == "ml-model":
            try:
                from engine.monitoring.learning_loop import LearningLoop
                ll = LearningLoop()
                result = ll.format_learning_report() if hasattr(ll, "format_learning_report") else "Not available"
            except Exception:
                result = "ML model report: learning loop unavailable"
            log_dir = PROJECT_ROOT / "logs" / "ml-model"

        else:
            return {"error": f"Unknown report type: {report_type}", "valid_types": ["platinum", "daily", "portfolio", "risk", "execution", "investor", "compliance", "ml-model"]}

        # Persist to log directory
        if log_dir and result:
            log_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
            (log_dir / fname).write_text(result if isinstance(result, str) else str(result))

        return {
            "report_type": report_type,
            "content": result if isinstance(result, str) else str(result),
            "status": "generated",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"monitoring/reports/generate error: {e}")
        return {"error": str(e)}


# ─── TECH tab ──────────────────────────────────────────────

@router.get("/engines/health")
async def engines_health():
    """All engine health: status, latency, memory, errors."""
    try:
        import importlib
        import time

        engines = [
            ("L1", "UniverseEngine", "engine.data.universe_engine"),
            ("L2", "MacroEngine", "engine.signals.macro_engine"),
            ("L2", "MetadronCube", "engine.signals.metadron_cube"),
            ("L2", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("L3", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L4", "BetaCorridor", "engine.portfolio.beta_corridor"),
            ("L5", "ExecutionEngine", "engine.execution.execution_engine"),
            ("L5", "OptionsEngine", "engine.execution.options_engine"),
            ("L5", "VolatilitySurface", "engine.execution.options_engine"),
            ("L5", "DecisionMatrix", "engine.execution.decision_matrix"),
            ("L5", "MLVoteEnsemble", "engine.execution.execution_engine"),
            ("L6", "ResearchBots", "engine.agents.research_bots"),
            ("L6", "PatternRecognition", "engine.signals.pattern_recognition"),
            ("L7", "L7Execution", "engine.execution.l7_unified_execution_surface"),
            ("L4", "FuturesEngine", "engine.api.routers.futures"),
            ("L5", "TCAEngine", "engine.execution.tca_engine"),
            ("L6", "AgentScorecard", "engine.agents.agent_scorecard"),
            ("L6", "AgentMonitor", "engine.agents.agent_monitor"),
            ("L6", "PaulOrchestrator", "engine.agents.paul_orchestrator"),
            ("L6", "EnforcementEngine", "engine.agents.enforcement_engine"),
            ("L6", "DynamicAgentFactory", "engine.agents.dynamic_agent_factory"),
            ("L6", "SectorBots", "engine.agents.sector_bots"),
            ("L6", "InvestorPersonas", "engine.agents.investor_personas"),
            ("L6.5", "QuantStrategyExecutor", "engine.execution.quant_strategy_executor"),
            ("L6.5", "PatternRecognitionEngine", "engine.ml.pattern_recognition"),
            ("L6.5", "PatternDiscoveryEngine", "engine.signals.pattern_discovery_engine"),
            ("L6.5", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L6.5", "Backtester", "engine.ml.backtester"),
            ("MON", "AnomalyDetector", "engine.monitoring.anomaly_detector"),
        ]

        results = []
        for layer, name, module_path in engines:
            start = time.time()
            try:
                importlib.import_module(module_path)
                latency = (time.time() - start) * 1000  # ms
                results.append({
                    "id": layer, "name": name, "level": layer,
                    "status": "online", "latency": round(latency, 1),
                    "errors": 0,
                })
            except Exception as ex:
                results.append({
                    "id": layer, "name": name, "level": layer,
                    "status": "offline", "latency": 0,
                    "errors": 1, "error_msg": str(ex)[:100],
                })

        return {"engines": results, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/engines/health error: {e}")
        return {"engines": [], "error": str(e)}


@router.get("/anomalies")
async def anomalies():
    """Statistical anomaly scanner results."""
    try:
        ad = _get_anomaly()
        if hasattr(ad, "scan"):
            results = ad.scan()
        elif hasattr(ad, "detect"):
            results = ad.detect()
        else:
            results = []

        if isinstance(results, list):
            return {"anomalies": results, "timestamp": datetime.utcnow().isoformat()}
        return {"anomalies": results if isinstance(results, dict) else str(results), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/anomalies error: {e}")
        return {"anomalies": [], "error": str(e)}


@router.get("/sectors")
async def sector_tracking():
    """Sector performance + missed opportunities."""
    try:
        st = _get_sector_tracker()
        if hasattr(st, "get_tracking"):
            data = st.get_tracking()
        elif hasattr(st, "track"):
            data = st.track()
        else:
            data = {}
        return {**(data if isinstance(data, dict) else {"data": str(data)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/sectors error: {e}")
        return {"error": str(e)}


@router.get("/learning-loop")
async def learning_loop():
    """Closed-loop feedback: signal accuracy → tier weights."""
    try:
        ll = _get_learning()
        if hasattr(ll, "get_state"):
            state = ll.get_state()
        elif hasattr(ll, "summary"):
            state = ll.summary()
        else:
            state = {}
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/learning-loop error: {e}")
        return {"error": str(e)}


# ─── Centralized errors (TECH tab reads this) ─────────────

@router.get("/errors")
async def engine_errors(limit: int = Query(30, ge=1, le=200)):
    """All engine errors from the current session. TECH tab primary error source."""
    try:
        from engine.ops.error_logger import get_recent_errors, get_error_counts
        errors = get_recent_errors(limit)
        counts = get_error_counts()
        return {"errors": errors, "counts": counts, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/errors error: {e}")
        return {"errors": [], "counts": {}, "error": str(e)}


@router.get("/session-close")
async def trigger_session_close():
    """Trigger post-session file generation (TX logs, recon, learning, ML, errors).

    Normally called by run_close.py, but available via API for manual trigger.
    """
    try:
        from engine.ops.session_close import generate_session_files
        files = generate_session_files()
        return {"files": files, "status": "generated", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/session-close error: {e}")
        return {"error": str(e)}


# ─── ARCHIVE tab ───────────────────────────────────────────

@router.get("/archive/files")
async def archive_files(
    folder: str = Query("logs", description="Subfolder to list"),
    limit: int = Query(50, ge=1, le=200),
):
    """List archived files: TX logs, reconciliation, learning loop."""
    try:
        base = PROJECT_ROOT / folder
        if not base.exists() or not str(base.resolve()).startswith(str(PROJECT_ROOT)):
            return {"files": [], "error": "Invalid folder"}

        files = []
        for f in sorted(base.rglob("*"), key=os.path.getmtime, reverse=True):
            if f.is_file() and not f.name.startswith("."):
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(PROJECT_ROOT)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
                if len(files) >= limit:
                    break

        return {"files": files, "folder": folder, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/archive/files error: {e}")
        return {"files": [], "error": str(e)}


@router.get("/archive/read")
async def archive_read(path: str = Query(..., description="Relative file path")):
    """Read an archived file's content."""
    try:
        file_path = PROJECT_ROOT / path
        if not file_path.exists() or not str(file_path.resolve()).startswith(str(PROJECT_ROOT)):
            return {"error": "File not found or access denied"}

        content = file_path.read_text(errors="replace")[:50000]  # Cap at 50KB
        return {
            "path": path,
            "content": content,
            "size": file_path.stat().st_size,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"monitoring/archive/read error: {e}")
        return {"error": str(e)}


@router.get("/vps-metrics")
async def vps_metrics():
    """VPS system metrics: CPU, memory, disk, network."""
    try:
        import shutil

        # Disk usage
        disk = shutil.disk_usage("/")
        disk_total_gb = disk.total / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_pct = (disk.used / disk.total * 100) if disk.total else 0

        # Memory (from /proc/meminfo if available)
        mem_total = 0
        mem_avail = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) / 1024  # MB
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1]) / 1024
        except Exception:
            pass
        mem_used = mem_total - mem_avail
        mem_pct = (mem_used / mem_total * 100) if mem_total else 0

        # CPU load
        cpu_load = 0.0
        try:
            with open("/proc/loadavg") as f:
                cpu_load = float(f.read().split()[0])
        except Exception:
            pass

        return {
            "metrics": [
                {"name": "Primary", "cpu": round(cpu_load * 10, 1), "memory": round(mem_pct, 1), "disk": round(disk_pct, 1), "network": "—"},
            ],
            "system": {
                "disk_total_gb": round(disk_total_gb, 1),
                "disk_used_gb": round(disk_used_gb, 1),
                "mem_total_mb": round(mem_total, 0),
                "mem_used_mb": round(mem_used, 0),
                "cpu_load": round(cpu_load, 2),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"monitoring/vps-metrics error: {e}")
        return {"metrics": [], "error": str(e)}


@router.get("/logs")
async def log_stream(limit: int = Query(20, ge=1, le=100)):
    """Recent log messages from engine log files."""
    try:
        log_files = sorted(PROJECT_ROOT.glob("logs/**/*.log"), key=os.path.getmtime, reverse=True)
        if not log_files:
            log_files = sorted(PROJECT_ROOT.glob("logs/**/*.txt"), key=os.path.getmtime, reverse=True)

        messages = []
        for lf in log_files[:3]:  # Check last 3 log files
            try:
                lines = lf.read_text(errors="replace").strip().split("\n")
                for line in lines[-limit:]:
                    if line.strip():
                        # Parse log level if present
                        level = "info"
                        if "ERROR" in line.upper():
                            level = "error"
                        elif "WARN" in line.upper():
                            level = "warn"
                        elif "DEBUG" in line.upper():
                            level = "debug"
                        messages.append({
                            "time": line[:19] if len(line) > 19 else "",
                            "level": level,
                            "message": line,
                            "source": lf.name,
                        })
            except Exception:
                continue

        # Sort by time descending, take limit
        messages.sort(key=lambda m: m["time"], reverse=True)
        return {"messages": messages[:limit], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/logs error: {e}")
        return {"messages": [], "error": str(e)}
