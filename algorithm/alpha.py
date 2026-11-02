"""Alpha models — walk-forward ML, quality-tier ensemble, chart patterns, Kronos, model scorecard.

All free / open source, run inside the algorithm; optional libraries are guarded (numpy, scikit-learn,
xgboost) and every model degrades to "no forecast" rather than failing.

    patterns()         breakout to a 52-week high on rising volume, golden cross (50 > 200 in the last 10 days),
                       oversold bounce (RSI14 < 30 turning up), breakdown below the 200-day → score in [−1, 1]
    WalkForward        weekly samples (features at t, label = 21-day forward return, filled when it matures);
                       refit every REFIT_WEEKS on the last MAX_SAMPLES: ridge (numpy closed form) + gradient
                       boosting (scikit-learn) when available; forecast = mean of fitted models, annualised
    QualityEnsemble    4 classifiers (Gaussian NB, gradient boosting, random forest, XGBoost) — soft vote for
                       P(top tercile of 63-day forward return) from fundamentals features
    kronos_alpha()     reads ObjectStore metadron/kronos.json written by agents/kronos_agent.py (CPU inference)
    Scorecard          per-model rank IC on matured forecasts (trailing 12 periods); weight ∝ max(IC, 0); a model
                       with IC ≤ 0 for 4 straight periods gets weight 0 (ablation rule, architecture point 5b)
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:          # pragma: no cover
    np = None
try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    SKLEARN = True
except Exception:          # pragma: no cover
    SKLEARN = False
try:
    from xgboost import XGBClassifier
    XGB = True
except Exception:          # pragma: no cover
    XGB = False

HORIZON_DAYS = 21
REFIT_WEEKS = 4
MAX_SAMPLES = 6000
MIN_SAMPLES = 300
RIDGE_L2 = 1.0
FEATURES = ("mom_12_1", "mom_3m", "rev_1m", "vol_63", "composite", "mos", "roic_wacc", "edge", "pattern")
Q_FEATURES = ("composite", "mos", "roic_wacc", "accruals", "credit_ord", "log_pepb", "ey")


def _rets(px):
    return [b / a - 1 for a, b in zip(px[:-1], px[1:]) if a > 0]


def rsi(px: List[float], n: int = 14) -> Optional[float]:
    if len(px) < n + 1:
        return None
    r = _rets(px[-(n + 1):])
    up = sum(x for x in r if x > 0) / n
    dn = -sum(x for x in r if x < 0) / n
    return 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)


def patterns(px: List[float], vol: Optional[List[float]] = None) -> Tuple[float, List[str]]:
    if len(px) < 210:
        return 0.0, []
    s, tags = 0.0, []
    hi252 = max(px[-252:])
    vol_up = bool(vol) and len(vol) >= 60 and sum(vol[-5:]) / 5 > 1.5 * sum(vol[-60:]) / 60
    if px[-1] >= 0.98 * hi252 and vol_up:
        s += 0.6; tags.append("breakout 52w high on volume")
    ma = lambda n, k=0: sum(px[len(px) - n - k:len(px) - k]) / n
    if ma(50) > ma(200) and any(ma(50, k) <= ma(200, k) for k in range(1, 11)):
        s += 0.4; tags.append("golden cross")
    r_now, r_prev = rsi(px), rsi(px[:-1])
    if r_prev is not None and r_now is not None and r_prev < 30 <= r_now:
        s += 0.3; tags.append("oversold bounce")
    if px[-1] < ma(200) and px[-2] >= ma(200, 1):
        s -= 0.6; tags.append("breakdown below 200d")
    return max(-1.0, min(1.0, s)), tags


def price_features(px: List[float]) -> Dict[str, float]:
    f = {"mom_12_1": 0.0, "mom_3m": 0.0, "rev_1m": 0.0, "vol_63": 0.3}
    if len(px) >= 253:
        f["mom_12_1"] = px[-22] / px[-253] - 1
    if len(px) >= 64:
        f["mom_3m"] = px[-1] / px[-64] - 1
        r = _rets(px[-64:])
        m = sum(r) / len(r)
        f["vol_63"] = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1) * 252)
    if len(px) >= 22:
        f["rev_1m"] = px[-1] / px[-22] - 1
    return f


def _finite(v) -> float:
    try:
        v = float(v)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


class WalkForward:
    def __init__(self, features=FEATURES, horizon_steps: int = 4):
        self.features = features
        self.horizon_steps = horizon_steps          # weekly samples → 4 steps ≈ 21 trading days
        self.pending: deque = deque()               # (step, ticker, x, price)
        self.X: deque = deque(maxlen=MAX_SAMPLES)
        self.y: deque = deque(maxlen=MAX_SAMPLES)
        self.step_n = 0
        self.weeks_since_fit = REFIT_WEEKS
        self.coef = None
        self.mu = None
        self.sd = None
        self.gbm = None

    def vector(self, feats: Dict[str, float]) -> List[float]:
        return [_finite(feats.get(k)) for k in self.features]

    def observe(self, rows: Dict[str, Tuple[Dict[str, float], float]]) -> List[Tuple[str, float]]:
        """rows: {ticker: (features, price)} at this weekly step. Returns matured (ticker, realised return)."""
        self.step_n += 1
        matured = []
        while self.pending and self.step_n - self.pending[0][0] >= self.horizon_steps:
            st, t, x, p0 = self.pending.popleft()
            if t in rows and p0 > 0:
                r = rows[t][1] / p0 - 1
                self.X.append(x)
                self.y.append(r)
                matured.append((t, r))
        for t, (feats, p) in rows.items():
            if p and p > 0:
                self.pending.append((self.step_n, t, self.vector(feats), p))
        self.weeks_since_fit += 1
        if self.weeks_since_fit >= REFIT_WEEKS:
            self.fit()
        return matured

    def fit(self) -> bool:
        if np is None or len(self.y) < MIN_SAMPLES:
            return False
        X = np.asarray(self.X, dtype=float)
        y = np.asarray(self.y, dtype=float)
        y = np.clip(y, -0.5, 0.5)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        Z1 = np.hstack([np.ones((len(Z), 1)), Z])
        reg = RIDGE_L2 * np.eye(Z1.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(Z1.T @ Z1 + reg, Z1.T @ y)
        if SKLEARN:
            try:
                self.gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, random_state=7).fit(Z, y)
            except Exception:
                self.gbm = None
        self.weeks_since_fit = 0
        return True

    def predict(self, feats: Dict[str, float]) -> Optional[float]:
        """Annualised forecast of the 21-day return, or None before the first fit."""
        if self.coef is None or np is None:
            return None
        z = (np.asarray(self.vector(feats)) - self.mu) / self.sd
        preds = [float(self.coef[0] + z @ self.coef[1:])]
        if self.gbm is not None:
            preds.append(float(self.gbm.predict(z.reshape(1, -1))[0]))
        return sum(preds) / len(preds) * (252 / HORIZON_DAYS)


class QualityEnsemble:
    """Soft vote of 4 classifiers for P(top tercile of forward 63-day return)."""

    def __init__(self, horizon_steps: int = 13):
        self.wf = WalkForward(Q_FEATURES, horizon_steps)
        self.models = []

    def observe(self, rows):
        matured = self.wf.observe(rows)
        if self.wf.weeks_since_fit == 0:
            self.fit()
        return matured

    def fit(self) -> bool:
        if not SKLEARN or np is None or len(self.wf.y) < MIN_SAMPLES:
            return False
        X = (np.asarray(self.wf.X) - self.wf.mu) / self.wf.sd
        y = np.asarray(self.wf.y)
        lab = (y >= np.quantile(y, 2 / 3)).astype(int)
        if lab.sum() < 10 or lab.sum() > len(lab) - 10:
            return False
        cands = [GaussianNB(), GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=7),
                 RandomForestClassifier(n_estimators=200, max_depth=6, random_state=7, n_jobs=1)]
        if XGB:
            cands.append(XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, verbosity=0))
        self.models = []
        for m in cands:
            try:
                self.models.append(m.fit(X, lab))
            except Exception:
                continue
        return bool(self.models)

    def prob_top(self, feats: Dict[str, float]) -> Optional[float]:
        if not self.models or self.wf.mu is None:
            return None
        z = ((np.asarray(self.wf.vector(feats)) - self.wf.mu) / self.wf.sd).reshape(1, -1)
        ps = []
        for m in self.models:
            try:
                ps.append(float(m.predict_proba(z)[0][1]))
            except Exception:
                continue
        return sum(ps) / len(ps) if ps else None


def quality_features(score) -> Dict[str, float]:
    tiers = "ABCDEF"
    return {"composite": score.composite / 100, "mos": max(-1.0, score.margin_of_safety),
            "roic_wacc": score.roic_wacc, "accruals": min(score.accrual_quality, 3.0),
            "credit_ord": tiers.index(score.capital.credit_tier) if score.capital.credit_tier in tiers else 3,
            "log_pepb": math.log(max(min(score.pe_x_pb, 999.0), 1e-3)), "ey": score.earnings_yield}


def kronos_alpha(raw: Optional[str]) -> Dict[str, float]:
    """{ticker: annualised expected return} from metadron/kronos.json."""
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        h = float(obj.get("horizon_days", HORIZON_DAYS))
        return {t: float(v) * 252 / h for t, v in obj.get("forecasts", {}).items()}
    except Exception:
        return {}


def rank_ic(pred: List[float], real: List[float]) -> Optional[float]:
    n = len(pred)
    if n < 5:
        return None
    rk = lambda v: {i: r for r, i in enumerate(sorted(range(n), key=lambda i: v[i]))}
    a, b = rk(pred), rk(real)
    d2 = sum((a[i] - b[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


class Scorecard:
    """Per-model IC on matured forecasts; weight ∝ max(mean IC, 0); 4 periods of IC ≤ 0 → weight 0."""

    def __init__(self, models=("score", "ml", "quality", "kronos"), window: int = 12):
        self.ics = {m: deque(maxlen=window) for m in models}
        self.bad_streak = {m: 0 for m in models}
        self.open: Dict[str, Dict[str, float]] = {m: {} for m in models}      # pending forecasts this period

    def record(self, model: str, ticker: str, forecast: Optional[float]):
        if forecast is not None and model in self.open:
            self.open[model][ticker] = forecast

    def settle(self, realised: Dict[str, float]) -> None:
        for m, fc in self.open.items():
            common = [t for t in fc if t in realised]
            ic = rank_ic([fc[t] for t in common], [realised[t] for t in common])
            if ic is not None:
                self.ics[m].append(ic)
                self.bad_streak[m] = self.bad_streak[m] + 1 if ic <= 0 else 0
            self.open[m] = {}

    def weights(self) -> Dict[str, float]:
        raw = {}
        for m, v in self.ics.items():
            if self.bad_streak[m] >= 4:
                raw[m] = 0.0
            elif not v:
                raw[m] = 1.0 if m == "score" else 0.0          # unscored models start at 0 until proven
            else:
                raw[m] = max(sum(v) / len(v), 0.0)
        if sum(raw.values()) <= 0:
            raw["score"] = 1.0
        tot = sum(raw.values())
        return {m: w / tot for m, w in raw.items()}

    def report(self) -> Dict[str, dict]:
        return {m: {"ic_mean": (sum(v) / len(v)) if v else None, "n": len(v), "bad_streak": self.bad_streak[m]}
                for m, v in self.ics.items()}
