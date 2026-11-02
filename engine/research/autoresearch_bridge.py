"""
Bridge for karpathy/autoresearch autonomous training loop.
Tracks val_bpb experiments, exposes results to agents.
Ref: https://github.com/karpathy/autoresearch
"""
from __future__ import annotations
import logging
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

_AUTORESEARCH_ROOT = Path(__file__).resolve().parent / "autoresearch"
_RESULTS_TSV = _AUTORESEARCH_ROOT / "results.tsv"


class AutoresearchBridge:
    """Access autoresearch training loop results and status."""

    def is_available(self) -> bool:
        return _AUTORESEARCH_ROOT.exists() and (_AUTORESEARCH_ROOT / "train.py").exists()

    def get_status(self) -> dict:
        try:
            results = self.read_results()
            return {
                "available": self.is_available(),
                "path": str(_AUTORESEARCH_ROOT),
                "has_results": len(results) > 0,
                "experiment_count": len(results),
                "last_experiment": results[-1].get("tag") if results else None,
                "best_val_bpb": min(
                    (float(r.get("val_bpb", 999)) for r in results if r.get("val_bpb")),
                    default=None,
                ),
            }
        except Exception as e:
            logger.warning("AutoresearchBridge.get_status error: %s", e)
            return {"available": False, "error": str(e)}

    def read_results(self) -> list:
        if not _RESULTS_TSV.exists():
            return []
        try:
            with open(_RESULTS_TSV, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                return list(reader)
        except Exception as e:
            logger.warning("AutoresearchBridge.read_results error: %s", e)
            return []

    def get_program_md(self) -> str:
        p = _AUTORESEARCH_ROOT / "program.md"
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""
