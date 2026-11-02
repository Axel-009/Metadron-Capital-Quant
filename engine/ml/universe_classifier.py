"""UniverseClassifier — XGBoost quality-tier classifier (A-G).

Implements a soft-voting 4-model ensemble that classifies securities into
quality tiers based on fundamental and quantitative features.  Hyperparameters
follow T3.1 spec from the MODERATE-Project calibration:
    n_estimators=120, max_depth=6, learning_rate=0.1, gamma=0,
    reg_lambda=10, colsample_bylevel=0.5

Tiers
-----
A  Sharpe ≥ 2.0 AND momentum > 15%
B  Sharpe ≥ 1.5 AND momentum > 10%
C  Sharpe ≥ 1.0 AND momentum > 5%
D  Sharpe ≥ 0.5 AND momentum > 0%
E  Sharpe ≥ 0.0
F  Sharpe < 0.0 AND momentum > -10%
G  Everything else (distressed / avoid)

Reconciliation Engine
---------------------
Compares top-down observable tier (from price/vol) with bottom-up fundamental
tier.  Divergence > 2 grades triggers a FLAG for manual review or alpha
opportunity detection (fallen-angel / rising-star signals).
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger("metadron.intelligence.universe_classifier")

# ---------------------------------------------------------------------------
# Quality Tier Enum
# ---------------------------------------------------------------------------

class QualityTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


TIER_ORDER = list(QualityTier)
TIER_INDEX = {t: i for i, t in enumerate(TIER_ORDER)}


def classify_tier(sharpe: float, momentum: float) -> QualityTier:
    """Rule-based tier assignment from Sharpe and momentum."""
    if sharpe >= 2.0 and momentum > 0.15:
        return QualityTier.A
    elif sharpe >= 1.5 and momentum > 0.10:
        return QualityTier.B
    elif sharpe >= 1.0 and momentum > 0.05:
        return QualityTier.C
    elif sharpe >= 0.5 and momentum > 0.0:
        return QualityTier.D
    elif sharpe >= 0.0:
        return QualityTier.E
    elif momentum > -0.10:
        return QualityTier.F
    else:
        return QualityTier.G


# ---------------------------------------------------------------------------
# FundamentalsStore
# ---------------------------------------------------------------------------

@dataclass
class FundamentalRecord:
    """Cached fundamental data for a single ticker."""
    ticker: str
    roe: float = 0.0
    de_ratio: float = 0.0
    interest_coverage: float = 0.0
    current_ratio: float = 0.0
    revenue_growth: float = 0.0
    earnings_stability: float = 0.5
    free_cash_flow_yield: float = 0.0
    gross_margin: float = 0.0
    piotroski_f: int = 5
    altman_z: float = 3.0
    sharpe: float = 0.0
    momentum_3m: float = 0.0
    momentum_6m: float = 0.0
    volatility: float = 0.20
    max_drawdown: float = -0.10
    beta: float = 1.0
    updated: str = ""


class FundamentalsStore:
    """In-memory cache of fundamental data per ticker.

    Loads from a JSON file if available, otherwise initialises empty.
    """

    def __init__(self, cache_path: Optional[str] = None):
        self._cache_path = Path(cache_path) if cache_path else Path("data/fundamentals_cache.json")
        self._records: Dict[str, FundamentalRecord] = {}
        self._load()

    def _load(self):
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text())
                for ticker, data in raw.items():
                    self._records[ticker] = FundamentalRecord(ticker=ticker, **{
                        k: v for k, v in data.items() if k != "ticker"
                    })
                logger.info("FundamentalsStore loaded %d records", len(self._records))
            except Exception as e:
                logger.warning("FundamentalsStore load failed: %s", e)

    def save(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {t: asdict(r) for t, r in self._records.items()}
        self._cache_path.write_text(json.dumps(data, indent=2))

    def get(self, ticker: str) -> Optional[FundamentalRecord]:
        return self._records.get(ticker)

    def put(self, record: FundamentalRecord):
        record.updated = datetime.now().isoformat()
        self._records[record.ticker] = record

    def tickers(self) -> List[str]:
        return list(self._records.keys())

    def to_dataframe(self) -> Any:
        """Return records as a pandas DataFrame (if pandas available)."""
        if pd is None:
            return self._records
        rows = [asdict(r) for r in self._records.values()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def compute_from_returns(self, ticker: str, returns: Any) -> FundamentalRecord:
        """Derive fundamental proxies from a return series.

        Parameters
        ----------
        ticker : str
        returns : pd.Series  (daily returns, DatetimeIndex)
        """
        if pd is None or returns is None or len(returns) < 20:
            return FundamentalRecord(ticker=ticker)

        ann_ret = float(returns.mean() * 252)
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        mom_3m = float(returns.iloc[-63:].sum()) if len(returns) >= 63 else float(returns.sum())
        mom_6m = float(returns.iloc[-126:].sum()) if len(returns) >= 126 else float(returns.sum())

        cum = (1 + returns).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min())

        # ROE proxy: return efficiency
        abs_mean = float(returns.abs().mean())
        roe_proxy = float(returns.mean()) / abs_mean if abs_mean > 0 else 0.0

        # D/E proxy: downside/upside vol ratio
        up_vol = float(returns[returns > 0].std()) if (returns > 0).sum() > 5 else 0.01
        down_vol = float(returns[returns < 0].std()) if (returns < 0).sum() > 5 else 0.01
        de_proxy = down_vol / up_vol if up_vol > 0 else 1.0

        # Earnings stability proxy
        monthly = returns.resample("ME").sum() if len(returns) > 21 else returns
        if len(monthly) > 3:
            ac = monthly.autocorr(lag=1)
            earn_stab = 1.0 - abs(float(ac)) if not np.isnan(ac) else 0.5
        else:
            earn_stab = 0.5

        # Interest coverage proxy (Sharpe as proxy)
        ic_proxy = max(sharpe, 0) * 5.0

        record = FundamentalRecord(
            ticker=ticker,
            roe=roe_proxy,
            de_ratio=de_proxy,
            interest_coverage=ic_proxy,
            earnings_stability=earn_stab,
            sharpe=sharpe,
            momentum_3m=mom_3m,
            momentum_6m=mom_6m,
            volatility=ann_vol,
            max_drawdown=max_dd,
            beta=1.0,
        )
        self.put(record)
        return record


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "sharpe", "momentum_3m", "momentum_6m", "volatility", "max_drawdown",
    "roe", "de_ratio", "interest_coverage", "current_ratio",
    "revenue_growth", "earnings_stability", "free_cash_flow_yield",
    "gross_margin", "piotroski_f", "altman_z", "beta",
]


def record_to_features(rec: FundamentalRecord) -> np.ndarray:
    """Convert a FundamentalRecord to a 16-element feature vector."""
    return np.array([
        rec.sharpe,
        rec.momentum_3m,
        rec.momentum_6m,
        rec.volatility,
        rec.max_drawdown,
        rec.roe,
        rec.de_ratio,
        rec.interest_coverage,
        rec.current_ratio,
        rec.revenue_growth,
        rec.earnings_stability,
        rec.free_cash_flow_yield,
        rec.gross_margin,
        rec.piotroski_f,
        rec.altman_z,
        rec.beta,
    ], dtype=np.float64)


def record_to_label(rec: FundamentalRecord) -> int:
    """Derive tier label index from rule-based classification."""
    tier = classify_tier(rec.sharpe, rec.momentum_3m)
    return TIER_INDEX[tier]


# ---------------------------------------------------------------------------
# UniverseClassifier
# ---------------------------------------------------------------------------

class UniverseClassifier:
    """Soft-voting 4-model ensemble for quality tier classification.

    Models:
        1. GaussianNB         — fast probabilistic baseline
        2. GradientBoosting   — sequential boosting
        3. RandomForest       — bagging diversity
        4. XGBoost            — T3.1 hyperparams (primary)

    Falls back to pure rule-based classification if sklearn/xgboost
    are not installed.
    """

    # T3.1 XGBoost hyperparameters (from MODERATE-Project calibration)
    XGB_PARAMS = {
        "n_estimators": 120,
        "max_depth": 6,
        "learning_rate": 0.1,
        "gamma": 0,
        "reg_lambda": 10,
        "colsample_bylevel": 0.5,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }

    # Ensemble voting weights
    MODEL_WEIGHTS = {
        "gaussian_nb": 0.15,
        "gradient_boosting": 0.25,
        "random_forest": 0.25,
        "xgboost": 0.35,
    }

    def __init__(self, fundamentals_store: Optional[FundamentalsStore] = None):
        self.store = fundamentals_store or FundamentalsStore()
        self._models: Dict[str, Any] = {}
        self._is_trained = False
        self._n_classes = len(QualityTier)
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._reconciliation_flags: List[Dict[str, Any]] = []

        self._init_models()

    def _init_models(self):
        """Initialise ensemble models (graceful fallback if missing)."""
        if HAS_SKLEARN:
            self._models["gaussian_nb"] = GaussianNB()
            self._models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42,
            )
            self._models["random_forest"] = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42,
            )

        if HAS_XGB:
            self._models["xgboost"] = XGBClassifier(**self.XGB_PARAMS)
        elif HAS_SKLEARN:
            # Fallback: use another GBM if XGBoost unavailable
            self._models["xgboost"] = GradientBoostingClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.1,
                random_state=42,
            )

    @property
    def is_ml_available(self) -> bool:
        return len(self._models) > 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """Train the ensemble.

        If X/y not provided, generates training data from the
        FundamentalsStore using rule-based labels.
        """
        if X is None or y is None:
            X, y = self._build_training_set()

        if X is None or len(X) < 10:
            logger.warning("Insufficient training data (%d samples), using rule-based only",
                           len(X) if X is not None else 0)
            return

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        for name, model in self._models.items():
            try:
                model.fit(X, y_encoded)
                logger.info("Trained %s on %d samples (labels: %s)", name, len(X), le.classes_.tolist())
            except Exception as e:
                logger.warning("Failed to train %s: %s", name, e)

        self._is_trained = True

    def _build_training_set(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build X, y from FundamentalsStore records."""
        records = [self.store.get(t) for t in self.store.tickers()]
        records = [r for r in records if r is not None]

        if not records:
            return None, None

        X = np.array([record_to_features(r) for r in records])
        y = np.array([record_to_label(r) for r in records])

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        return X, y

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def classify(self, ticker: str, record: Optional[FundamentalRecord] = None) -> Dict[str, Any]:
        """Classify a single ticker into a quality tier.

        Returns dict with:
            tier: QualityTier
            tier_label: str (A-G)
            confidence: float (0-1)
            probabilities: dict[tier -> probability]
            method: str ("ensemble" or "rule_based")
            credit_quality_score: float (0-1)
        """
        if record is None:
            record = self.store.get(ticker)
        if record is None:
            record = FundamentalRecord(ticker=ticker)

        # Rule-based classification (always available)
        rule_tier = classify_tier(record.sharpe, record.momentum_3m)
        rule_idx = TIER_INDEX[rule_tier]

        # ML ensemble prediction (if trained)
        if self._is_trained and self._models:
            features = record_to_features(record).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            proba_sum = np.zeros(self._n_classes)
            weight_sum = 0.0

            for name, model in self._models.items():
                try:
                    weight = self.MODEL_WEIGHTS.get(name, 0.25)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)[0]
                        # Pad if model doesn't predict all classes
                        if len(proba) < self._n_classes:
                            padded = np.zeros(self._n_classes)
                            padded[:len(proba)] = proba
                            proba = padded
                        proba_sum += weight * proba
                    else:
                        pred = model.predict(features)[0]
                        one_hot = np.zeros(self._n_classes)
                        one_hot[int(pred)] = 1.0
                        proba_sum += weight * one_hot
                    weight_sum += weight
                except Exception as e:
                    logger.debug("Model %s prediction failed: %s", name, e)

            if weight_sum > 0:
                proba_sum /= weight_sum
                ml_idx = int(np.argmax(proba_sum))
                confidence = float(proba_sum[ml_idx])
                ml_tier = TIER_ORDER[ml_idx]
                method = "ensemble"
            else:
                ml_tier = rule_tier
                ml_idx = rule_idx
                confidence = 0.5
                proba_sum = np.zeros(self._n_classes)
                proba_sum[rule_idx] = 1.0
                method = "rule_based"
        else:
            ml_tier = rule_tier
            ml_idx = rule_idx
            confidence = 0.5
            proba_sum = np.zeros(self._n_classes)
            proba_sum[rule_idx] = 1.0
            method = "rule_based"

        # Credit quality score: map tier to [0, 1]
        credit_quality_score = 1.0 - (ml_idx / max(self._n_classes - 1, 1))

        result = {
            "ticker": ticker,
            "tier": ml_tier,
            "tier_label": ml_tier.value,
            "tier_index": ml_idx,
            "confidence": confidence,
            "probabilities": {t.value: float(proba_sum[i]) for i, t in enumerate(TIER_ORDER)},
            "method": method,
            "credit_quality_score": credit_quality_score,
            "sharpe": record.sharpe,
            "momentum_3m": record.momentum_3m,
        }

        self._predictions[ticker] = result
        return result

    def classify_universe(self, tickers: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Classify all tickers (or a subset) in the FundamentalsStore."""
        if tickers is None:
            tickers = self.store.tickers()

        results = {}
        for ticker in tickers:
            results[ticker] = self.classify(ticker)

        return results

    # ------------------------------------------------------------------
    # Reconciliation Engine
    # ------------------------------------------------------------------

    def reconcile(self, ticker: str,
                  observable_tier: Optional[QualityTier] = None) -> Dict[str, Any]:
        """Compare top-down observable tier with bottom-up fundamental tier.

        A divergence of >2 grades triggers a FLAG.

        Parameters
        ----------
        ticker : str
        observable_tier : QualityTier, optional
            If not provided, derived from recent price performance.
        """
        record = self.store.get(ticker)
        if record is None:
            return {"ticker": ticker, "divergence": 0, "flag": False}

        # Bottom-up: ML/rule-based tier
        pred = self._predictions.get(ticker)
        if pred is None:
            pred = self.classify(ticker, record)
        bottom_up_idx = pred["tier_index"]

        # Top-down: observable tier (from Sharpe/momentum)
        if observable_tier is not None:
            top_down_idx = TIER_INDEX[observable_tier]
        else:
            top_down_idx = record_to_label(record)

        divergence = abs(top_down_idx - bottom_up_idx)
        flag = divergence >= 2

        direction = ""
        if top_down_idx < bottom_up_idx:
            direction = "rising_star"  # observable better than fundamental
        elif top_down_idx > bottom_up_idx:
            direction = "fallen_angel"  # fundamental better than observable

        result = {
            "ticker": ticker,
            "top_down_tier": TIER_ORDER[top_down_idx].value,
            "bottom_up_tier": TIER_ORDER[bottom_up_idx].value,
            "divergence": divergence,
            "direction": direction,
            "flag": flag,
        }

        if flag:
            self._reconciliation_flags.append(result)

        return result

    def reconcile_universe(self, tickers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run reconciliation across the universe."""
        if tickers is None:
            tickers = self.store.tickers()

        results = []
        for ticker in tickers:
            results.append(self.reconcile(ticker))

        return results

    def get_flagged(self) -> List[Dict[str, Any]]:
        """Return tickers flagged by reconciliation (divergence >= 2)."""
        return self._reconciliation_flags.copy()

    def get_rising_stars(self) -> List[Dict[str, Any]]:
        """Tickers where observable tier is better than fundamental."""
        return [f for f in self._reconciliation_flags if f["direction"] == "rising_star"]

    def get_fallen_angels(self) -> List[Dict[str, Any]]:
        """Tickers where fundamental tier is better than observable."""
        return [f for f in self._reconciliation_flags if f["direction"] == "fallen_angel"]

    # ------------------------------------------------------------------
    # Credit Quality Integration
    # ------------------------------------------------------------------

    def get_credit_scores(self) -> Dict[str, Dict[str, Any]]:
        """Return credit quality scores for all classified tickers.

        Format compatible with ExecutionEngine.MLVoteEnsemble.set_credit_scores().
        """
        scores = {}
        for ticker, pred in self._predictions.items():
            scores[ticker] = {
                "credit_quality_score": pred["credit_quality_score"],
                "tier": pred["tier_label"],
                "confidence": pred["confidence"],
            }
        return scores

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_report(self) -> str:
        """Generate a text report of classifications."""
        lines = [
            "=" * 60,
            "UNIVERSE CLASSIFIER — QUALITY TIER REPORT",
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Method: {'Ensemble (4-model)' if self._is_trained else 'Rule-based'}",
            f"  Tickers classified: {len(self._predictions)}",
            "=" * 60,
            "",
        ]

        # Tier distribution
        tier_counts = {t.value: 0 for t in QualityTier}
        for pred in self._predictions.values():
            tier_counts[pred["tier_label"]] += 1

        lines.append("TIER DISTRIBUTION:")
        for tier_val, count in tier_counts.items():
            bar = "#" * count
            lines.append(f"  {tier_val}: {count:>4}  {bar}")
        lines.append("")

        # Top classifications
        sorted_preds = sorted(
            self._predictions.values(),
            key=lambda p: p["credit_quality_score"],
            reverse=True,
        )

        lines.append("TOP 10 (by credit quality score):")
        lines.append(f"  {'Ticker':<10} {'Tier':<6} {'CQS':>6} {'Conf':>6} {'Sharpe':>8} {'Mom3M':>8}")
        lines.append("  " + "-" * 50)
        for pred in sorted_preds[:10]:
            lines.append(
                f"  {pred['ticker']:<10} {pred['tier_label']:<6} "
                f"{pred['credit_quality_score']:>6.3f} {pred['confidence']:>6.3f} "
                f"{pred.get('sharpe', 0):>8.3f} {pred.get('momentum_3m', 0):>8.3f}"
            )
        lines.append("")

        # Bottom 5
        lines.append("BOTTOM 5 (weakest credit):")
        for pred in sorted_preds[-5:]:
            lines.append(
                f"  {pred['ticker']:<10} {pred['tier_label']:<6} "
                f"{pred['credit_quality_score']:>6.3f}"
            )
        lines.append("")

        # Reconciliation flags
        if self._reconciliation_flags:
            lines.append(f"RECONCILIATION FLAGS ({len(self._reconciliation_flags)}):")
            for flag in self._reconciliation_flags:
                lines.append(
                    f"  {flag['ticker']:<10} TD={flag['top_down_tier']} "
                    f"BU={flag['bottom_up_tier']} Δ={flag['divergence']} "
                    f"({flag['direction']})"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[str] = None):
        """Save predictions and flags to JSON."""
        save_path = Path(path) if path else Path("data/classifier_state.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "timestamp": datetime.now().isoformat(),
            "is_trained": self._is_trained,
            "n_predictions": len(self._predictions),
            "predictions": self._predictions,
            "reconciliation_flags": self._reconciliation_flags,
        }
        save_path.write_text(json.dumps(state, indent=2, default=str))
        logger.info("Classifier state saved to %s", save_path)

    def load_state(self, path: Optional[str] = None):
        """Load predictions from JSON (does NOT restore models)."""
        load_path = Path(path) if path else Path("data/classifier_state.json")
        if not load_path.exists():
            return

        try:
            state = json.loads(load_path.read_text())
            self._predictions = state.get("predictions", {})
            self._reconciliation_flags = state.get("reconciliation_flags", [])
            logger.info("Classifier state loaded: %d predictions", len(self._predictions))
        except Exception as e:
            logger.warning("Failed to load classifier state: %s", e)


# ---------------------------------------------------------------------------
# CreditQualityClassifier — 6-factor weighted scoring model
# ---------------------------------------------------------------------------

class CreditQualityClassifier:
    """Maps securities to implied credit ratings (AAA-D).

    Uses a 6-factor weighted model:
        1. Interest Coverage Ratio  (25%)
        2. Debt/Equity Ratio        (20%)
        3. ROE Proxy                (15%)
        4. Earnings Stability       (15%)
        5. Altman Z-Score           (15%)
        6. Free Cash Flow Yield     (10%)

    Output: credit_quality_score in [0.0, 1.0] and implied CreditRating.
    """

    FACTOR_WEIGHTS = {
        "interest_coverage": 0.25,
        "de_ratio": 0.20,
        "roe": 0.15,
        "earnings_stability": 0.15,
        "altman_z": 0.15,
        "fcf_yield": 0.10,
    }

    # Rating thresholds on credit_quality_score
    RATING_THRESHOLDS = [
        (0.95, "AAA"),
        (0.90, "AA"),
        (0.80, "A"),
        (0.70, "BBB"),
        (0.55, "BB"),
        (0.40, "B"),
        (0.25, "CCC"),
        (0.15, "CC"),
        (0.05, "C"),
        (0.00, "D"),
    ]

    def score(self, record: FundamentalRecord) -> Dict[str, Any]:
        """Compute credit quality score and implied rating.

        Parameters
        ----------
        record : FundamentalRecord

        Returns
        -------
        dict with credit_quality_score, credit_rating, factor_scores
        """
        # Normalise each factor to [0, 1]
        ic_score = min(max(record.interest_coverage / 10.0, 0), 1)
        de_score = 1.0 / (1.0 + max(record.de_ratio, 0))
        roe_score = min(max(record.roe, 0), 1)
        es_score = min(max(record.earnings_stability, 0), 1)
        az_score = min(max((record.altman_z - 1.0) / 4.0, 0), 1)
        fcf_score = min(max((record.free_cash_flow_yield + 0.05) / 0.15, 0), 1)

        factor_scores = {
            "interest_coverage": ic_score,
            "de_ratio": de_score,
            "roe": roe_score,
            "earnings_stability": es_score,
            "altman_z": az_score,
            "fcf_yield": fcf_score,
        }

        # Weighted sum
        credit_quality_score = sum(
            self.FACTOR_WEIGHTS[k] * factor_scores[k]
            for k in self.FACTOR_WEIGHTS
        )
        credit_quality_score = min(max(credit_quality_score, 0), 1)

        # Map to rating
        credit_rating = "D"
        for threshold, rating in self.RATING_THRESHOLDS:
            if credit_quality_score >= threshold:
                credit_rating = rating
                break

        return {
            "credit_quality_score": credit_quality_score,
            "credit_rating": credit_rating,
            "factor_scores": factor_scores,
        }

    def score_universe(self, store: FundamentalsStore) -> Dict[str, Dict[str, Any]]:
        """Score all tickers in a FundamentalsStore."""
        results = {}
        for ticker in store.tickers():
            record = store.get(ticker)
            if record is not None:
                results[ticker] = self.score(record)
                results[ticker]["ticker"] = ticker
        return results
