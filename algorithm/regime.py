"""Market regime — Gaussian HMM on SPY / volatility / VIX / credit, with a rules fallback.

Regimes: TRENDING · RANGE · STRESS · CRASH (same names as the risk budget and MetadronCube).

    features  20-day SPY return, 20-day realised vol (daily), VIX/100, credit proxy
              (HYG − LQD 21-day relative return), each standardised for the fit
    model     4-state Gaussian HMM (hmmlearn), refit at most every REFIT_DAYS on the last
              FIT_WINDOW days; the current regime = forward-backward posterior of the last day
    mapping   ABSOLUTE, per state, from its mean 20-day return r20 and annualised vol (max of realised
              and VIX) — a calm market with 4 calm states stays calm (rank-based labels would force one
              state to CRASH):
                  CRASH     vol ≥ 40%  or r20 ≤ −15%
                  STRESS    vol ≥ 25%  or r20 ≤ −7%
                  TRENDING  r20 ≥ +0.4% (≈ +5%/yr) and vol < 18%
                  RANGE     otherwise
    fallback  if hmmlearn is missing or the fit fails: transparent threshold rules (design rule 2)

The HMM classifies the MARKET level only; theme crashes are rotation.py's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:                                    # guarded (design rule 2)
    import numpy as np
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:                       # pragma: no cover
    np = None
    GaussianHMM = None
    HMM_AVAILABLE = False

REGIMES = ("TRENDING", "RANGE", "STRESS", "CRASH")
FIT_WINDOW = 750
REFIT_DAYS = 21
MIN_OBS = 120
POSTERIOR_WINDOW = 120


@dataclass
class RegimeState:
    regime: str
    probs: Dict[str, float]
    confidence: float
    method: str                          # "hmm" | "rules"
    detail: dict = field(default_factory=dict)


def _features(spy: List[float], vix: Optional[List[float]] = None,
              hyg: Optional[List[float]] = None, lqd: Optional[List[float]] = None) -> List[List[float]]:
    rets = [b / a - 1 for a, b in zip(spy[:-1], spy[1:]) if a > 0]
    rows = []
    for i in range(20, len(rets)):
        w = rets[i - 19:i + 1]
        m = sum(w) / len(w)
        vol = math.sqrt(sum((x - m) ** 2 for x in w) / (len(w) - 1))
        row = [sum(w), vol]
        if vix is not None:
            j = len(vix) - (len(rets) - i)
            row.append(vix[j] / 100 if 0 <= j < len(vix) else 0.2)
        if hyg is not None and lqd is not None:
            j = len(hyg) - (len(rets) - i)
            k = len(lqd) - (len(rets) - i)
            if j >= 21 and k >= 21:
                row.append((hyg[j] / hyg[j - 21]) - (lqd[k] / lqd[k - 21]))
            else:
                row.append(0.0)
        rows.append(row)
    return rows


def rules_regime(spy: List[float], vix: Optional[float] = None) -> RegimeState:
    """Transparent fallback: trailing 63-day return and 20-day vol (+ VIX if given)."""
    rets = [b / a - 1 for a, b in zip(spy[:-1], spy[1:]) if a > 0]
    if len(rets) < 63:
        return RegimeState("RANGE", {r: 0.25 for r in REGIMES}, 0.25, "rules", {"reason": "insufficient data"})
    r63 = spy[-1] / spy[-64] - 1
    w = rets[-20:]
    m = sum(w) / 20
    vol = math.sqrt(sum((x - m) ** 2 for x in w) / 19 * 252)
    v = vix if vix is not None else vol * 100
    if r63 < -0.15 or v > 40:
        reg = "CRASH"
    elif r63 < -0.05 or v > 28:
        reg = "STRESS"
    elif r63 > 0.04 and v < 20:
        reg = "TRENDING"
    else:
        reg = "RANGE"
    probs = {r: (0.7 if r == reg else 0.1) for r in REGIMES}
    return RegimeState(reg, probs, 0.7, "rules", {"r63": r63, "vol": vol, "vix": v})


def label_state(r20: float, vol: float) -> str:
    """Absolute regime label from a state's mean 20-day return and annualised volatility."""
    if vol >= 0.40 or r20 <= -0.15:
        return "CRASH"
    if vol >= 0.25 or r20 <= -0.07:
        return "STRESS"
    if r20 >= 0.004 and vol < 0.18:
        return "TRENDING"
    return "RANGE"


class RegimeModel:
    """Stateful HMM wrapper: refit every REFIT_DAYS, classify every call."""

    def __init__(self, n_states: int = 4, seed: int = 7):
        self.n_states = n_states
        self.seed = seed
        self.model = None
        self.state_map: Dict[int, str] = {}
        self.mu = None
        self.sd = None
        self.days_since_fit = REFIT_DAYS

    def _fit(self, X):
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=200,
                            random_state=self.seed)
        model.fit(X)
        raw = model.means_ * self.sd + self.mu            # back to original units
        self.state_map = {}
        for s_ in range(self.n_states):
            trend = float(raw[s_, 0])                     # 20-day return
            vol = float(raw[s_, 1]) * math.sqrt(252)
            if raw.shape[1] > 2:
                vol = max(vol, float(raw[s_, 2]))           # VIX/100 is already annualised
            self.state_map[s_] = label_state(trend, vol)
        self.model = model

    def classify(self, spy: List[float], vix: Optional[List[float]] = None,
                 hyg: Optional[List[float]] = None, lqd: Optional[List[float]] = None) -> RegimeState:
        last_vix = vix[-1] if vix else None
        if not HMM_AVAILABLE:
            return rules_regime(spy, last_vix)
        rows = _features(spy[-(FIT_WINDOW + 22):], vix, hyg, lqd)
        if len(rows) < MIN_OBS:
            return rules_regime(spy, last_vix)
        X = np.asarray(rows, dtype=float)
        try:
            self.days_since_fit += 1
            if self.model is None or self.days_since_fit >= REFIT_DAYS or self.mu is None \
                    or self.mu.shape[0] != X.shape[1]:
                self.mu = X.mean(axis=0)
                self.sd = X.std(axis=0) + 1e-9
                self._fit((X - self.mu) / self.sd)
                self.days_since_fit = 0
            Z = (X[-POSTERIOR_WINDOW:] - self.mu) / self.sd
            post = self.model.predict_proba(Z)[-1]
        except Exception as exc:
            st = rules_regime(spy, last_vix)
            st.detail["hmm_error"] = str(exc)
            return st
        probs = {r: 0.0 for r in REGIMES}
        for s, p in enumerate(post):
            probs[self.state_map.get(s, "RANGE")] += float(p)
        reg = max(probs, key=probs.get)
        return RegimeState(reg, probs, probs[reg], "hmm", {"n_obs": len(rows)})
