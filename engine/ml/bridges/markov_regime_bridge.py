"""Markov Regime Bridge — hmmlearn HMM → MetadronCube RegimeEngine.

Integrates the hmmlearn Hidden Markov Model library with MetadronCube's
RegimeEngine for data-driven regime detection. Replaces the hand-tuned
Markov transition matrix with a statistically fitted Gaussian HMM.

The bridge:
    1. Fits a GaussianHMM on market observables (returns, vol, spreads)
    2. Maps HMM hidden states → CubeRegime (TRENDING/RANGE/STRESS/CRASH)
    3. Provides transition probability matrix learned from data
    4. Outputs regime probabilities for confidence scoring

Integration Points:
    - MetadronCube.RegimeEngine (engine/signals/metadron_cube.py:485)
      → Replaces hardcoded TRANSITION_PROBS with fitted probabilities
      → Adds posterior regime probabilities for confidence

    - MacroEngine (engine/signals/macro_engine.py)
      → HMM regime as confirmatory signal alongside GMTF

Source repo: repos/markov-model (https://github.com/Axel-009/markov-model)
             Contains hmmlearn library source (GaussianHMM, GMMHMM, etc.)

Usage:
    from engine.ml.bridges.markov_regime_bridge import MarkovRegimeBridge
    bridge = MarkovRegimeBridge(n_regimes=4)
    bridge.fit(returns, volatility, spreads)
    regime, confidence = bridge.predict_regime(latest_obs)
    transition_matrix = bridge.get_transition_matrix()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# hmmlearn from local source or installed package
MARKOV_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "repos" / "markov-model"

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:  # rule 8 — degrade to threshold regimes, never crash
    GaussianHMM = None
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed — MarkovRegimeBridge uses threshold fallback")


# CubeRegime mapping (must match engine/signals/macro_engine.py)
REGIME_NAMES = ["TRENDING", "RANGE", "STRESS", "CRASH"]


@dataclass
class RegimePrediction:
    """Output from HMM regime prediction."""
    regime: str = "RANGE"               # TRENDING/RANGE/STRESS/CRASH
    regime_index: int = 1               # 0-3
    confidence: float = 0.5             # [0, 1]
    state_probabilities: list[float] = field(default_factory=lambda: [0.25] * 4)
    transition_matrix: Optional[np.ndarray] = None
    log_likelihood: float = 0.0
    metadata: dict = field(default_factory=dict)


class MarkovRegimeBridge:
    """HMM-based market regime detection bridge for MetadronCube.

    Fits a Gaussian Hidden Markov Model on market observables and maps
    the hidden states to Metadron Capital's 4-regime framework.

    Observables used:
        1. Market returns (SPY daily returns)
        2. Realized volatility (20-day rolling std)
        3. Credit spreads (HY-IG or proxy)
        4. VIX level
        5. Yield curve slope (10Y-2Y)

    State mapping heuristic:
        - Highest mean return, lowest vol → TRENDING
        - Moderate return, moderate vol → RANGE
        - Negative return, high vol → STRESS
        - Most negative return, highest vol → CRASH
    """

    def __init__(self, n_regimes: int = 4, covariance_type: str = "full",
                 n_iter: int = 100, random_state: int = 42):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
        self._fitted = False
        self._state_map: dict[int, str] = {}
        self._feature_names: list[str] = []

        self._model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        ) if GaussianHMM is not None else None
        logger.info(f"MarkovRegimeBridge initialized (n_regimes={n_regimes})")

    def fit(self, returns: np.ndarray, volatility: np.ndarray,
            credit_spread: Optional[np.ndarray] = None,
            vix: Optional[np.ndarray] = None,
            yield_slope: Optional[np.ndarray] = None) -> dict:
        """Fit HMM on market observables.

        Args:
            returns: Daily returns array (T,)
            volatility: Realized volatility array (T,)
            credit_spread: Credit spread array (T,) optional
            vix: VIX levels (T,) optional
            yield_slope: 10Y-2Y spread (T,) optional

        Returns:
            Dict with fitting metrics.
        """
        # Build observation matrix
        features = [returns, volatility]
        self._feature_names = ["returns", "volatility"]

        if credit_spread is not None:
            features.append(credit_spread)
            self._feature_names.append("credit_spread")
        if vix is not None:
            features.append(vix)
            self._feature_names.append("vix")
        if yield_slope is not None:
            features.append(yield_slope)
            self._feature_names.append("yield_slope")

        # Align lengths
        min_len = min(len(f) for f in features)
        X = np.column_stack([f[-min_len:] for f in features])

        # Remove NaN rows
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]

        if len(X) < 50:
            logger.warning(f"Insufficient data for HMM fitting ({len(X)} obs)")
            return {"error": "insufficient data", "n_obs": len(X)}
        if self._model is None:
            return {"error": "hmmlearn unavailable", "n_obs": len(X)}

        try:
            self._model.fit(X)
            self._fitted = True

            # Map HMM states to regime names based on return/vol characteristics
            self._map_states_to_regimes(X)

            score = self._model.score(X)
            logger.info(f"HMM fitted: {len(X)} obs, log-likelihood={score:.2f}")
            return {
                "n_obs": len(X),
                "n_features": X.shape[1],
                "log_likelihood": float(score),
                "converged": self._model.monitor_.converged,
                "n_iter": self._model.monitor_.iter,
                "state_map": self._state_map,
            }
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")
            return {"error": str(e)}

    def predict_regime(self, returns: float, volatility: float,
                        credit_spread: Optional[float] = None,
                        vix: Optional[float] = None,
                        yield_slope: Optional[float] = None) -> RegimePrediction:
        """Predict current regime from latest observables.

        Returns RegimePrediction with regime name and confidence.
        """
        if not self._fitted:
            return RegimePrediction(regime="RANGE", confidence=0.3,
                                    metadata={"error": "model not fitted"})

        # Build observation vector
        obs = [returns, volatility]
        if "credit_spread" in self._feature_names and credit_spread is not None:
            obs.append(credit_spread)
        if "vix" in self._feature_names and vix is not None:
            obs.append(vix)
        if "yield_slope" in self._feature_names and yield_slope is not None:
            obs.append(yield_slope)

        X = np.array([obs])

        try:
            state_probs = self._model.predict_proba(X)[0]
            best_state = int(np.argmax(state_probs))
            regime_name = self._state_map.get(best_state, "RANGE")
            confidence = float(state_probs[best_state])

            # Map probabilities to regime order
            regime_probs = [0.0] * 4
            for state_idx, prob in enumerate(state_probs):
                regime = self._state_map.get(state_idx, "RANGE")
                regime_idx = REGIME_NAMES.index(regime) if regime in REGIME_NAMES else 1
                regime_probs[regime_idx] += prob

            return RegimePrediction(
                regime=regime_name,
                regime_index=REGIME_NAMES.index(regime_name) if regime_name in REGIME_NAMES else 1,
                confidence=confidence,
                state_probabilities=regime_probs,
                transition_matrix=self._model.transmat_,
                log_likelihood=float(self._model.score(X)),
            )
        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            regime = self._threshold_regime(returns, volatility)
            return RegimePrediction(
                regime=regime,
                regime_index=REGIME_NAMES.index(regime),
                confidence=0.5,
                metadata={"method": "threshold_fallback"},
            )

    @property
    def fitted(self) -> bool:
        return self._fitted

    def predict_regime_sequence(self, X: np.ndarray) -> RegimePrediction:
        """Predict the CURRENT regime from a recent observation sequence.

        Uses the forward-backward posterior of the last row, so the HMM's
        transition dynamics (state persistence) are used — unlike
        predict_regime(), which scores one isolated observation.

        X: (T, n_features) in the same column order used by fit().
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            return RegimePrediction(regime="RANGE", confidence=0.3,
                                    metadata={"error": "empty sequence"})
        X = X[~np.any(np.isnan(X), axis=1)]
        if not self._fitted or self._model is None or len(X) == 0:
            last = X[-1] if len(X) else [0.0, 0.0]
            regime = self._threshold_regime(float(last[0]), float(last[1]))
            return RegimePrediction(regime=regime, regime_index=REGIME_NAMES.index(regime),
                                    confidence=0.5, metadata={"method": "threshold_fallback"})
        try:
            post = self._model.predict_proba(X)[-1]
            regime_probs = [0.0] * 4
            for state_idx, prob in enumerate(post):
                regime = self._state_map.get(state_idx, "RANGE")
                regime_probs[REGIME_NAMES.index(regime) if regime in REGIME_NAMES else 1] += float(prob)
            best = int(np.argmax(regime_probs))
            return RegimePrediction(
                regime=REGIME_NAMES[best], regime_index=best,
                confidence=float(regime_probs[best]),
                state_probabilities=regime_probs,
                transition_matrix=self._model.transmat_,
                log_likelihood=float(self._model.score(X)),
                metadata={"method": "hmm_sequence", "n_obs": int(len(X))},
            )
        except Exception as e:
            logger.warning(f"HMM sequence prediction failed: {e}")
            last = X[-1]
            regime = self._threshold_regime(float(last[0]), float(last[1]))
            return RegimePrediction(regime=regime, regime_index=REGIME_NAMES.index(regime),
                                    confidence=0.5, metadata={"method": "threshold_fallback"})

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Get the learned transition probability matrix.

        Returns 4x4 matrix indexed by REGIME_NAMES order.
        Can be used to replace MetadronCube RegimeEngine.TRANSITION_PROBS.
        """
        if self._model is None or not self._fitted:
            return None

        try:
            raw_transmat = self._model.transmat_
            # Reorder to match REGIME_NAMES order
            n = len(REGIME_NAMES)
            ordered = np.zeros((n, n))
            for i, ri in enumerate(REGIME_NAMES):
                for j, rj in enumerate(REGIME_NAMES):
                    si = self._reverse_state_map().get(ri)
                    sj = self._reverse_state_map().get(rj)
                    if si is not None and sj is not None:
                        ordered[i, j] = raw_transmat[si, sj]
            # Normalize rows
            row_sums = ordered.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            ordered = ordered / row_sums
            return ordered
        except Exception as e:
            logger.warning(f"Failed to extract transition matrix: {e}")
            return None

    def get_transition_probs_dict(self) -> dict:
        """Get transition matrix as nested dict compatible with RegimeEngine.TRANSITION_PROBS.

        Returns dict matching MetadronCube format:
            {CubeRegime.X: {CubeRegime.Y: prob, ...}, ...}
        """
        matrix = self.get_transition_matrix()
        if matrix is None:
            return {}

        result = {}
        for i, from_regime in enumerate(REGIME_NAMES):
            result[from_regime] = {}
            for j, to_regime in enumerate(REGIME_NAMES):
                result[from_regime][to_regime] = float(matrix[i, j])
        return result

    # --- Private helpers ---

    def _map_states_to_regimes(self, X: np.ndarray):
        """Map HMM hidden states to regime names by return/vol characteristics."""
        if self._model is None:
            return

        states = self._model.predict(X)
        means = self._model.means_

        # Return is feature 0, volatility is feature 1
        state_return_means = {s: means[s, 0] for s in range(self.n_regimes)}
        state_vol_means = {s: means[s, 1] for s in range(self.n_regimes)}

        # Sort states by return (descending) and vol (ascending)
        # Highest return + lowest vol = TRENDING
        # Most negative return + highest vol = CRASH
        scored = {}
        for s in range(self.n_regimes):
            scored[s] = state_return_means[s] - 0.5 * state_vol_means[s]

        sorted_states = sorted(scored.keys(), key=lambda s: scored[s], reverse=True)

        regime_assignment = REGIME_NAMES[:self.n_regimes]
        self._state_map = {sorted_states[i]: regime_assignment[i]
                           for i in range(min(len(sorted_states), len(regime_assignment)))}

        logger.info(f"HMM state mapping: {self._state_map}")

    def _reverse_state_map(self) -> dict[str, int]:
        """Reverse mapping: regime_name → state_index."""
        return {v: k for k, v in self._state_map.items()}

    @staticmethod
    def _threshold_regime(returns: float, volatility: float) -> str:
        """Simple threshold-based regime detection (numpy fallback)."""
        if returns > 0.001 and volatility < 0.015:
            return "TRENDING"
        elif returns < -0.005 and volatility > 0.025:
            return "CRASH"
        elif returns < -0.002 or volatility > 0.02:
            return "STRESS"
        else:
            return "RANGE"
