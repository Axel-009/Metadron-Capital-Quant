"""
Model evaluation utilities for the Metadron Capital framework.

Direct port of T3.1/Code/src/utils/models_utils.py evaluation logic,
adapted to the Metadron Capital engine structure. Provides per-class
and aggregate scoring, confusion matrices, and tier-weighted error
analysis for quality-grade prediction models (tiers A-G).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False


from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class ModelEvaluator:
    """Comprehensive model evaluation for classification tasks.

    Supports fine-grained per-class metrics and tier-aware distance
    weighting for ordinal quality grades (A through G).
    """

    TIER_ORDER: List[str] = ["A", "B", "C", "D", "E", "F", "G"]
    TIER_INDEX: Dict[str, int] = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
    }

    # ------------------------------------------------------------------ #
    #  Per-class (micro) metrics                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def micro_scores(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[Any]] = None,
    ) -> Dict[Any, Dict[str, float]]:
        """Per-class Precision, Recall, F1.

        Returns ``{class_label: {precision, recall, f1, support}}``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
        return {
            label: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
            for i, label in enumerate(labels)
        }

    # ------------------------------------------------------------------ #
    #  Macro / aggregate metrics                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def macro_scores(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Balanced accuracy, macro-F1, weighted-F1.

        Returns ``{balanced_accuracy, macro_f1, weighted_f1}``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return {
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }

    # ------------------------------------------------------------------ #
    #  Confusion matrix                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Standard confusion matrix of shape ``(n_classes, n_classes)``."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        return sk_confusion_matrix(y_true, y_pred, labels=labels)

    # ------------------------------------------------------------------ #
    #  Tier-weighted confusion                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def tier_weighted_confusion(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Confusion matrix with tier-aware distance weighting.

        Quality tiers A-G map to indices 0-6.  Distance penalty per
        prediction is ``|predicted_idx - actual_idx|``:
        A predicted as G = distance 6 (worst), A as B = distance 1 (minor).

        Returns ``{confusion_matrix, weighted_error,
        mean_absolute_tier_error, tier_accuracy}``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = cls.confusion_matrix(y_true, y_pred, labels=cls.TIER_ORDER)

        distances: List[int] = []
        for t, p in zip(y_true, y_pred):
            t_idx = cls.TIER_INDEX.get(str(t))
            p_idx = cls.TIER_INDEX.get(str(p))
            if t_idx is not None and p_idx is not None:
                distances.append(abs(p_idx - t_idx))
            else:
                logger.warning("Unknown tier: true=%s, pred=%s", t, p)

        n = len(distances)
        weighted_error = float(sum(distances)) if distances else 0.0
        mean_abs_error = float(np.mean(distances)) if distances else 0.0
        tier_accuracy = (
            float(sum(1 for d in distances if d == 0) / n)
            if n > 0 else 0.0
        )
        return {
            "confusion_matrix": cm,
            "weighted_error": weighted_error,
            "mean_absolute_tier_error": mean_abs_error,
            "tier_accuracy": tier_accuracy,
        }

    # ------------------------------------------------------------------ #
    #  Combined evaluation                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run all evaluations; return combined report.

        Returns ``{micro, macro, confusion, tier_weighted}``.
        ``tier_weighted`` is ``None`` when labels are not valid tiers.
        """
        result: Dict[str, Any] = {
            "micro": cls.micro_scores(y_true, y_pred, labels=labels),
            "macro": cls.macro_scores(y_true, y_pred),
            "confusion": cls.confusion_matrix(y_true, y_pred, labels=labels),
        }
        all_labels = set(np.asarray(y_true)) | set(np.asarray(y_pred))
        if all_labels <= set(cls.TIER_ORDER):
            result["tier_weighted"] = cls.tier_weighted_confusion(y_true, y_pred)
        else:
            logger.info(
                "Skipping tier-weighted confusion: labels %s not in TIER_ORDER.",
                all_labels,
            )
            result["tier_weighted"] = None
        return result

    # ------------------------------------------------------------------ #
    #  Pretty-print                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def format_report(cls, eval_result: Dict[str, Any]) -> str:
        """Format an evaluation dict as a human-readable ASCII report."""
        lines: List[str] = []
        sep = "=" * 60

        # Macro scores
        lines.append(sep)
        lines.append("  MACRO SCORES")
        lines.append(sep)
        macro = eval_result.get("macro", {})
        for key in ("balanced_accuracy", "macro_f1", "weighted_f1"):
            lines.append(f"  {key:<25s}: {macro.get(key, 0.0):.4f}")

        # Per-class scores
        lines.append("")
        lines.append(sep)
        lines.append("  PER-CLASS SCORES")
        lines.append(sep)
        lines.append(
            f"  {'Class':<10s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Sup':>8s}"
        )
        lines.append("  " + "-" * 46)
        micro = eval_result.get("micro", {})
        for label in sorted(micro.keys(), key=str):
            m = micro[label]
            lines.append(
                f"  {str(label):<10s} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8d}"
            )

        # Confusion matrix
        lines.append("")
        lines.append(sep)
        lines.append("  CONFUSION MATRIX")
        lines.append(sep)
        cm = eval_result.get("confusion")
        if cm is not None:
            for row in cm:
                lines.append("  " + "  ".join(f"{v:>5d}" for v in row))

        # Tier-weighted metrics
        tw = eval_result.get("tier_weighted")
        if tw is not None:
            lines.append("")
            lines.append(sep)
            lines.append("  TIER-WEIGHTED METRICS")
            lines.append(sep)
            lines.append(f"  {'weighted_error':<30s}: {tw['weighted_error']:.2f}")
            lines.append(
                f"  {'mean_absolute_tier_error':<30s}: "
                f"{tw['mean_absolute_tier_error']:.4f}"
            )
            lines.append(f"  {'tier_accuracy':<30s}: {tw['tier_accuracy']:.4f}")

        lines.append(sep)
        return "\n".join(lines)

    @classmethod
    def export_evaluation_skill(
        cls,
        eval_result: Dict[str, Any],
        skill_name: str = "model-evaluation-export",
    ) -> dict:
        """Export evaluation results via agent_skills and download artifacts.

        Returns skill output dict or empty dict if unavailable.
        """
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            result = test_skill(skill_name, {"evaluation": eval_result})
            file_ids = extract_file_ids(result) if isinstance(result, (dict, str)) else []
            downloaded = download_all_files(file_ids) if file_ids else []
            return {"skill_result": result, "downloaded_files": downloaded}
        except Exception:
            logger.debug("Skill export failed")
            return {}
