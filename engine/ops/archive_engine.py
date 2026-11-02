"""Metadron Capital — Archive Engine.

Collects daily broker logs, tech logs, error logs, TX logs, pattern recognition
files into a dated archive structure: logs/archive/YYYY/MM/DD/

Usage:
    from engine.ops.archive_engine import ArchiveEngine
    engine = ArchiveEngine()
    engine.archive_daily()           # archive today's files
    engine.get_archive_dates(2026, 4)  # list available dates
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("metadron.ops.archive_engine")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"


class ArchiveEngine:
    """Collects and archives daily system artifacts."""

    def __init__(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create base archive directory."""
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _date_dir(self, date_str: str) -> Path:
        """Return the archive directory for a given date string (YYYY-MM-DD)."""
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            d = datetime.now()
        p = ARCHIVE_DIR / str(d.year) / f"{d.month:02d}" / f"{d.day:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ─── Collectors ───────────────────────────────────────────

    def _collect_broker_trades(self) -> dict:
        """Collect merged trade log from Paper + IBKRBrokers."""
        trades = []
        # trade log
        try:
            from engine.execution.paper_broker import PaperBroker
            pb = PaperBroker()
            paper_trades = pb.get_trade_history() if hasattr(pb, "get_trade_history") else []
            for t in paper_trades:
                rec = dict(t) if isinstance(t, dict) else {}
                rec["source"] = "paper"
                trades.append(rec)
        except Exception as e:
            logger.warning(f"trade log trade collection failed: {e}")

        # IBKRBroker
        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            ab = AlpacaBroker(initial_cash=0, paper=True)
            ibkr_trades = ab.get_order_history() if hasattr(ab, "get_order_history") else []
            for t in ibkr_trades:
                rec = dict(t) if isinstance(t, dict) else {}
                rec["source"] = "ibkr"
                trades.append(rec)
        except Exception as e:
            logger.debug(f"IBKR broker trade collection skipped: {e}")

        return {
            "total_trades": len(trades),
            "paper_count": sum(1 for t in trades if t.get("source") == "paper"),
            "ibkr_count": sum(1 for t in trades if t.get("source") == "ibkr"),
            "trades": trades,
            "collected_at": datetime.now().isoformat(),
        }

    def _collect_tech_system(self) -> dict:
        """Collect system health snapshot."""
        result = {
            "uptime": "unknown",
            "engine_status": {},
            "collected_at": datetime.now().isoformat(),
        }
        try:
            import os
            result["pid"] = os.getpid()
            result["python_version"] = os.sys.version
        except Exception:
            pass
        return result

    def _collect_errors(self) -> dict:
        """Collect all errors from the session."""
        errors = []
        try:
            from engine.ops.error_logger import get_session_errors
            errors = get_session_errors()
        except Exception as e:
            logger.warning(f"Error collection failed: {e}")

        return {
            "total_errors": len(errors),
            "errors": errors,
            "collected_at": datetime.now().isoformat(),
        }

    def _collect_transactions(self) -> dict:
        """Collect or copy TX log from session_close output."""
        tx_data = {"trades": [], "total_trades": 0}
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            tx_path = LOGS_DIR / "transactions" / f"TX_{date_str}.json"
            if tx_path.exists():
                tx_data = json.loads(tx_path.read_text())
            else:
                from engine.ops.session_close import generate_tx_log
                generate_tx_log()
                if tx_path.exists():
                    tx_data = json.loads(tx_path.read_text())
        except Exception as e:
            logger.warning(f"TX log collection failed: {e}")

        tx_data["collected_at"] = datetime.now().isoformat()
        return tx_data

    def _collect_patterns(self) -> dict:
        """Collect pattern recognition results and trend signals."""
        patterns = []
        discoveries = []

        try:
            from engine.ml.pattern_recognition import PatternRecognitionEngine
            pre = PatternRecognitionEngine()
            if hasattr(pre, "get_active_patterns"):
                raw = pre.get_active_patterns()
                for p in (raw if isinstance(raw, list) else []):
                    if isinstance(p, dict):
                        patterns.append(p)
                    elif hasattr(p, "__dict__"):
                        patterns.append({k: str(v) for k, v in p.__dict__.items()})
        except Exception as e:
            logger.debug(f"PatternRecognitionEngine collection skipped: {e}")

        try:
            from engine.signals.pattern_discovery_engine import PatternDiscoveryEngine
            pde = PatternDiscoveryEngine()
            if hasattr(pde, "get_recent_signals"):
                raw = pde.get_recent_signals()
                for d in (raw if isinstance(raw, list) else []):
                    if isinstance(d, dict):
                        discoveries.append(d)
                    elif hasattr(d, "__dict__"):
                        discoveries.append({k: str(v) for k, v in d.__dict__.items()})
        except Exception as e:
            logger.debug(f"PatternDiscoveryEngine collection skipped: {e}")

        return {
            "patterns_count": len(patterns),
            "discoveries_count": len(discoveries),
            "patterns": patterns,
            "discoveries": discoveries,
            "collected_at": datetime.now().isoformat(),
        }

    # ─── Public Methods ───────────────────────────────────────

    def archive_daily(self, date_str: Optional[str] = None) -> dict:
        """Run full daily archival.

        Args:
            date_str: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            dict with archive results including file paths and counts.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        target = self._date_dir(date_str)
        files_written = {}

        collectors = {
            "broker_trades.json": self._collect_broker_trades,
            "tech_system.json": self._collect_tech_system,
            "errors.json": self._collect_errors,
            "transactions.json": self._collect_transactions,
            "patterns.json": self._collect_patterns,
        }

        for filename, collector in collectors.items():
            try:
                data = collector()
                path = target / filename
                path.write_text(json.dumps(data, indent=2, default=str))
                files_written[filename] = str(path)
            except Exception as e:
                logger.error(f"Failed to archive {filename}: {e}")
                files_written[filename] = f"ERROR: {e}"

        # Daily summary (generated separately)
        try:
            from engine.monitoring.daily_summary_generator import DailySummaryGenerator
            gen = DailySummaryGenerator()
            summary = gen.generate_daily_summary(date_str)
            path = target / "daily_summary.json"
            path.write_text(json.dumps(summary, indent=2, default=str))
            files_written["daily_summary.json"] = str(path)
        except Exception as e:
            logger.warning(f"Daily summary generation failed: {e}")
            files_written["daily_summary.json"] = f"ERROR: {e}"

        logger.info(f"Archive completed for {date_str}: {len(files_written)} files → {target}")
        return {
            "date": date_str,
            "directory": str(target),
            "files": files_written,
            "files_count": len([v for v in files_written.values() if not v.startswith("ERROR")]),
            "archived_at": datetime.now().isoformat(),
        }

    def get_archive_dates(self, year: Optional[int] = None, month: Optional[int] = None) -> list:
        """List available archive dates.

        Args:
            year: Filter by year. Defaults to all years.
            month: Filter by month (requires year).

        Returns:
            List of dicts with date and file count.
        """
        dates = []
        if not ARCHIVE_DIR.exists():
            return dates

        for year_dir in sorted(ARCHIVE_DIR.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                y = int(year_dir.name)
            except ValueError:
                continue
            if year is not None and y != year:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    m = int(month_dir.name)
                except ValueError:
                    continue
                if month is not None and m != month:
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    try:
                        d = int(day_dir.name)
                    except ValueError:
                        continue
                    files = [f.name for f in day_dir.iterdir() if f.is_file()]
                    dates.append({
                        "date": f"{y:04d}-{m:02d}-{d:02d}",
                        "files_count": len(files),
                        "files": files,
                    })

        return dates

    def get_archive_by_date(self, date_str: str) -> dict:
        """Return all files for a given date.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            dict with file names and their contents.
        """
        target = self._date_dir(date_str)
        result = {"date": date_str, "files": {}}

        if not target.exists():
            return result

        for f in sorted(target.iterdir()):
            if f.is_file() and f.suffix == ".json":
                try:
                    result["files"][f.name] = json.loads(f.read_text())
                except Exception:
                    result["files"][f.name] = {"error": "Failed to parse"}

        return result

    def get_archive_file(self, date_str: str, filename: str) -> dict:
        """Return specific file content from an archive date.

        Args:
            date_str: Date in YYYY-MM-DD format.
            filename: Name of the file (e.g. broker_trades.json).

        Returns:
            File content as dict, or error dict.
        """
        target = self._date_dir(date_str) / filename
        if not target.exists():
            return {"error": f"File not found: {filename} for {date_str}"}
        try:
            return json.loads(target.read_text())
        except Exception as e:
            return {"error": f"Failed to parse {filename}: {e}"}

    def get_archive_summary(self, year: int, month: int) -> dict:
        """Monthly summary stats.

        Args:
            year: Year (e.g. 2026).
            month: Month (1-12).

        Returns:
            dict with aggregated monthly statistics.
        """
        dates = self.get_archive_dates(year=year, month=month)
        total_trades = 0
        total_errors = 0
        total_patterns = 0
        total_files = 0

        for d in dates:
            date_str = d["date"]
            total_files += d["files_count"]

            # Read individual file summaries
            target = self._date_dir(date_str)
            for fname in ["broker_trades.json", "errors.json", "patterns.json"]:
                fpath = target / fname
                if fpath.exists():
                    try:
                        data = json.loads(fpath.read_text())
                        if fname == "broker_trades.json":
                            total_trades += data.get("total_trades", 0)
                        elif fname == "errors.json":
                            total_errors += data.get("total_errors", 0)
                        elif fname == "patterns.json":
                            total_patterns += data.get("patterns_count", 0)
                    except Exception:
                        pass

        return {
            "year": year,
            "month": month,
            "days_archived": len(dates),
            "total_files": total_files,
            "total_trades": total_trades,
            "total_errors": total_errors,
            "total_patterns": total_patterns,
            "dates": [d["date"] for d in dates],
        }
