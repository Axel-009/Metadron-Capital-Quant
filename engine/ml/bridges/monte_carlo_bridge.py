"""
ARIMA(1,1,1) + 1000-path Monte Carlo simulation bridge.
Produces MC_BUY / MC_SELL signals for MLVoteEnsemble.
Fully pure-numpy implementation -- no statsmodels dependency.
"""

import logging

import numpy as np

try:
    from engine.execution.paper_broker import SignalType
except Exception:
    class SignalType:
        MC_BUY = "MC_BUY"
        MC_SELL = "MC_SELL"

logger = logging.getLogger(__name__)


class MonteCarloBridge:
    """Monte Carlo simulation bridge with AR(1) return dynamics."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    # ------------------------------------------------------------------
    # AR(1) parameter estimation via numpy least squares
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_ar1(returns: np.ndarray) -> tuple[float, float, float]:
        """
        Fit AR(1) model: r_t = mu + phi * (r_{t-1} - mu) + eps

        Returns (mu, phi, sigma) where sigma is the residual std dev.
        """
        if len(returns) < 10:
            mu = float(np.mean(returns)) if len(returns) > 0 else 0.0
            sigma = float(np.std(returns)) if len(returns) > 1 else 0.01
            return mu, 0.0, max(sigma, 1e-8)

        y = returns[1:]
        x = returns[:-1]

        # OLS: y = a + b * x  =>  mu = a / (1 - b), phi = b
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            mu = float(np.mean(returns))
            sigma = float(np.std(returns))
            return mu, 0.0, max(sigma, 1e-8)

        a, phi = float(beta[0]), float(beta[1])

        # Clamp phi to stationary region
        phi = float(np.clip(phi, -0.99, 0.99))

        mu = a / (1.0 - phi) if abs(1.0 - phi) > 1e-10 else float(np.mean(returns))

        residuals = y - (a + phi * x)
        sigma = float(np.std(residuals))
        sigma = max(sigma, 1e-8)

        return mu, phi, sigma

    # ------------------------------------------------------------------
    # Monte Carlo path generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_paths(
        mu: float,
        phi: float,
        sigma: float,
        last_return: float,
        n_paths: int,
        horizon: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Generate Monte Carlo return paths.

        r_t = mu + phi * (r_{t-1} - mu) + sigma * noise

        Returns array of shape (n_paths, horizon).
        """
        paths = np.empty((n_paths, horizon), dtype=np.float64)
        noise = rng.standard_normal((n_paths, horizon))

        prev = np.full(n_paths, last_return, dtype=np.float64)
        for t in range(horizon):
            current = mu + phi * (prev - mu) + sigma * noise[:, t]
            paths[:, t] = current
            prev = current

        return paths

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_statistics(paths: np.ndarray) -> dict:
        """Compute aggregate statistics across all Monte Carlo paths."""
        # Cumulative returns per path
        cum_returns = np.sum(paths, axis=1)

        mean_return = float(np.mean(cum_returns))
        var_95 = float(np.percentile(cum_returns, 5))
        paths_positive_pct = float(np.mean(cum_returns > 0))

        return {
            "mean_return": mean_return,
            "var_95": var_95,
            "paths_positive_pct": paths_positive_pct,
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(stats: dict) -> float:
        """
        Confidence derived from path agreement and magnitude.
        High confidence when most paths agree on direction.
        """
        pos_pct = stats["paths_positive_pct"]
        direction_strength = abs(pos_pct - 0.5) * 2.0  # 0 to 1
        magnitude = min(abs(stats["mean_return"]) * 50.0, 1.0)
        confidence = 0.6 * direction_strength + 0.4 * magnitude
        return float(np.clip(confidence, 0.0, 1.0))

    def simulate(
        self,
        ticker: str,
        returns: np.ndarray,
        n_paths: int = 1000,
        horizon: int = 21,
    ) -> dict:
        """
        Run Monte Carlo simulation for the given ticker.

        Returns
        -------
        dict with keys: signal, mean_return, var_95, paths_positive_pct, confidence
        """
        rng = np.random.RandomState(self.seed)

        if len(returns) < 2:
            return {
                "signal": SignalType.MC_BUY,
                "mean_return": 0.0,
                "var_95": 0.0,
                "paths_positive_pct": 0.5,
                "confidence": 0.0,
            }

        mu, phi, sigma = self._fit_ar1(returns)
        last_return = float(returns[-1])

        paths = self._generate_paths(mu, phi, sigma, last_return, n_paths, horizon, rng)
        stats = self._compute_statistics(paths)
        confidence = self._compute_confidence(stats)

        if stats["mean_return"] > 0:
            signal = SignalType.MC_BUY
        else:
            signal = SignalType.MC_SELL

        return {
            "signal": signal,
            "mean_return": stats["mean_return"],
            "var_95": stats["var_95"],
            "paths_positive_pct": stats["paths_positive_pct"],
            "confidence": confidence,
        }
