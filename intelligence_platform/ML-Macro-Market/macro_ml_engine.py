"""
Macro-driven ML Engine for regime detection and investment implications.

Identifies macroeconomic regime changes using Hidden Markov Models,
factor models, and nowcasting techniques, then generates investment
implications across all asset classes.

Regime Detection (Hidden Markov Model):
    States: expansion, peak, contraction, trough
    Transition matrix P(S_t | S_{t-1}) estimated via Baum-Welch algorithm
    Emission probabilities: GDP_growth, unemployment_change, yield_curve_slope

Factor Models:
    Fama-French 5-factor:
        R_i - R_f = alpha + beta_MKT*(R_m - R_f) + beta_SMB*SMB
                    + beta_HML*HML + beta_RMW*RMW + beta_CMA*CMA

    Macro factors:
        GDP_surprise, inflation_surprise, rate_surprise, USD_change

Nowcasting:
    Dynamic Factor Model for GDP nowcasting
    Principal Component Regression on high-frequency indicators

Usage:
    from macro_ml_engine import MacroMLEngine
    engine = MacroMLEngine()
    regime = engine.detect_regime(macro_data)
    allocation = engine.optimal_asset_allocation_for_regime(regime)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openbb_universe import (
    AssetClass,
    detect_asset_class,
    get_full_universe,
    get_historical,
    get_multiple,
    MACRO_INDICATORS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes & enums
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Macroeconomic regime states."""
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"


@dataclass
class CascadingEffect:
    """Second-derivative effect of a macroeconomic shock."""
    order: int              # 1 = primary, 2 = secondary, 3 = tertiary
    description: str
    affected_asset_class: AssetClass
    expected_impact: float  # standardised impact score [-1, 1]
    confidence: float       # confidence in the causal chain [0, 1]
    lag_days: int           # expected lag before effect manifests
    transmission_channel: str


@dataclass
class TradeThesis:
    """A macro-driven trade idea."""
    asset_class: AssetClass
    direction: str          # "long" or "short"
    symbol: str
    rationale: str
    expected_return: float
    confidence: float
    horizon_days: int
    risk_factors: List[str] = field(default_factory=list)
    regime: Optional[MarketRegime] = None


@dataclass
class RegimeTransition:
    """Probability distribution over possible regime transitions."""
    current_regime: MarketRegime
    transition_probabilities: Dict[MarketRegime, float]
    most_likely_next: MarketRegime
    confidence: float
    leading_indicators_signal: Dict[str, float]


# ---------------------------------------------------------------------------
# Hidden Markov Model for Regime Detection
# ---------------------------------------------------------------------------

class HiddenMarkovRegimeModel:
    """
    Gaussian Hidden Markov Model for macroeconomic regime detection.

    States: {expansion, peak, contraction, trough}
    Observations: multivariate Gaussian emissions

    Model parameters:
        pi   : initial state distribution (1 x K)
        A    : transition matrix (K x K), A[i,j] = P(S_t=j | S_{t-1}=i)
        mu_k : emission mean for state k (1 x D)
        Sigma_k : emission covariance for state k (D x D)

    Estimation: Baum-Welch (EM algorithm)
        E-step: Forward-backward algorithm
            alpha_t(i) = P(O_1..O_t, S_t=i)
            beta_t(i)  = P(O_{t+1}..O_T | S_t=i)
            gamma_t(i) = P(S_t=i | O_1..O_T)
            xi_t(i,j)  = P(S_t=i, S_{t+1}=j | O_1..O_T)

        M-step: Update parameters
            pi_i     = gamma_1(i)
            A[i,j]   = sum_t(xi_t(i,j)) / sum_t(gamma_t(i))
            mu_k     = sum_t(gamma_t(k) * O_t) / sum_t(gamma_t(k))
            Sigma_k  = sum_t(gamma_t(k) * (O_t-mu_k)(O_t-mu_k)^T) / sum_t(gamma_t(k))

    Viterbi decoding for most-likely state sequence:
        delta_t(j) = max_i [delta_{t-1}(i) * A[i,j]] * b_j(O_t)
    """

    N_STATES = 4  # expansion, peak, contraction, trough
    STATE_NAMES = [MarketRegime.EXPANSION, MarketRegime.PEAK,
                   MarketRegime.CONTRACTION, MarketRegime.TROUGH]

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        # Model parameters (initialised in fit())
        self.pi: Optional[np.ndarray] = None         # (K,)
        self.A: Optional[np.ndarray] = None           # (K, K)
        self.means: Optional[np.ndarray] = None       # (K, D)
        self.covars: Optional[np.ndarray] = None      # (K, D, D)
        self._is_fitted = False

    def _init_params(self, n_features: int) -> None:
        """Initialise model parameters."""
        K = self.N_STATES
        self.pi = np.ones(K) / K

        # Transition matrix: slight preference for staying in current state
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        # Normalise rows
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.means = self.rng.randn(K, n_features) * 0.5
        self.covars = np.array([np.eye(n_features) for _ in range(K)])

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, covar: np.ndarray) -> float:
        """
        Multivariate Gaussian probability density.

        p(x | mu, Sigma) = (2*pi)^(-D/2) * |Sigma|^(-1/2)
                           * exp(-0.5 * (x - mu)^T Sigma^{-1} (x - mu))
        """
        D = len(x)
        diff = x - mean
        try:
            covar_inv = np.linalg.inv(covar)
            covar_det = np.linalg.det(covar)
        except np.linalg.LinAlgError:
            covar_reg = covar + 1e-6 * np.eye(D)
            covar_inv = np.linalg.inv(covar_reg)
            covar_det = np.linalg.det(covar_reg)

        covar_det = max(covar_det, 1e-300)
        norm_const = (2 * math.pi) ** (-D / 2) * covar_det ** (-0.5)
        exponent = -0.5 * diff @ covar_inv @ diff
        return norm_const * math.exp(min(exponent, 500))

    def _emission_probs(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute emission probabilities B[t, k] = P(O_t | S_t = k).

        Returns
        -------
        np.ndarray of shape (T, K)
        """
        T = len(observations)
        K = self.N_STATES
        B = np.zeros((T, K))
        for t in range(T):
            for k in range(K):
                B[t, k] = self._gaussian_pdf(observations[t], self.means[k], self.covars[k])
        # Floor to prevent underflow
        B = np.maximum(B, 1e-300)
        return B

    def _forward(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm.

        alpha_t(i) = P(O_1, ..., O_t, S_t = i)

        alpha_1(i) = pi_i * B[1, i]
        alpha_t(j) = [sum_i alpha_{t-1}(i) * A[i,j]] * B[t, j]

        Returns (alpha, scale_factors) for numerical stability.
        """
        T, K = B.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]

        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * B[t, j]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm.

        beta_t(i) = P(O_{t+1}, ..., O_T | S_t = i)

        beta_T(i) = 1
        beta_t(i) = sum_j A[i,j] * B[t+1, j] * beta_{t+1}(j)
        """
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                beta[t, i] = np.sum(self.A[i] * B[t + 1] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta

    def _e_step(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        E-step of Baum-Welch.

        Compute:
            gamma_t(i) = P(S_t = i | O_1..O_T)
            xi_t(i,j)  = P(S_t = i, S_{t+1} = j | O_1..O_T)
        """
        B = self._emission_probs(observations)
        alpha, scale = self._forward(B)
        beta = self._backward(B, scale)

        T, K = B.shape

        # gamma_t(i) = alpha_t(i) * beta_t(i) / sum_j(alpha_t(j) * beta_t(j))
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma /= gamma_sum

        # xi_t(i,j) = alpha_t(i) * A[i,j] * B[t+1,j] * beta_{t+1}(j) / normaliser
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = alpha[t, i] * self.A[i, j] * B[t + 1, j] * beta[t + 1, j]
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        # Log-likelihood = sum(log(scale))
        log_likelihood = np.sum(np.log(np.maximum(scale, 1e-300)))

        return gamma, xi, log_likelihood

    def _m_step(self, observations: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """
        M-step of Baum-Welch. Update model parameters.

        pi_i     = gamma_1(i)
        A[i,j]   = sum_t(xi_t(i,j)) / sum_t(gamma_t(i))
        mu_k     = sum_t(gamma_t(k) * O_t) / sum_t(gamma_t(k))
        Sigma_k  = sum_t(gamma_t(k) * (O_t - mu_k)(O_t - mu_k)^T) / sum_t(gamma_t(k))
        """
        T, K = gamma.shape
        D = observations.shape[1]

        # Initial state distribution
        self.pi = gamma[0] / gamma[0].sum()

        # Transition matrix
        for i in range(K):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                for j in range(K):
                    self.A[i, j] = xi[:, i, j].sum() / denom
            else:
                self.A[i] = 1.0 / K
        # Re-normalise
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission parameters
        for k in range(K):
            gamma_k = gamma[:, k]
            gamma_sum = gamma_k.sum()
            if gamma_sum > 0:
                # Mean: mu_k = sum(gamma_t(k) * O_t) / sum(gamma_t(k))
                self.means[k] = (gamma_k[:, np.newaxis] * observations).sum(axis=0) / gamma_sum

                # Covariance: Sigma_k
                diff = observations - self.means[k]
                weighted_diff = gamma_k[:, np.newaxis] * diff
                self.covars[k] = (weighted_diff.T @ diff) / gamma_sum
                # Regularise
                self.covars[k] += 1e-4 * np.eye(D)

    def fit(self, observations: np.ndarray) -> "HiddenMarkovRegimeModel":
        """
        Fit HMM via Baum-Welch (EM) algorithm.

        Parameters
        ----------
        observations : np.ndarray of shape (T, D)
            Time series of macro indicators.

        Returns
        -------
        self
        """
        T, D = observations.shape
        self._init_params(D)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            gamma, xi, log_likelihood = self._e_step(observations)
            self._m_step(observations, gamma, xi)

            improvement = log_likelihood - prev_ll
            logger.debug("Baum-Welch iter %d: LL=%.4f, improvement=%.6f",
                         iteration, log_likelihood, improvement)

            if abs(improvement) < self.tol and iteration > 5:
                logger.info("Baum-Welch converged at iteration %d (LL=%.4f)",
                            iteration, log_likelihood)
                break
            prev_ll = log_likelihood

        self._is_fitted = True
        return self

    def decode(self, observations: np.ndarray) -> List[MarketRegime]:
        """
        Viterbi decoding for most-likely state sequence.

        delta_t(j) = max_i [delta_{t-1}(i) * A[i,j]] * b_j(O_t)
        psi_t(j)   = argmax_i [delta_{t-1}(i) * A[i,j]]

        Backtrack: S*_t = psi_{t+1}(S*_{t+1})
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        B = self._emission_probs(observations)
        T, K = B.shape

        # Log-space for numerical stability
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(B, 1e-300))

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(K):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return [self.STATE_NAMES[s] for s in states]

    def predict_next_state(self, current_state: MarketRegime) -> Dict[MarketRegime, float]:
        """
        Predict next state probabilities: P(S_{t+1} | S_t = current_state).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        idx = self.STATE_NAMES.index(current_state)
        probs = self.A[idx]
        return {self.STATE_NAMES[j]: float(probs[j]) for j in range(self.N_STATES)}


# ---------------------------------------------------------------------------
# Factor Models
# ---------------------------------------------------------------------------

class FactorModel:
    """
    Fama-French 5-factor + macro factor model.

    Fama-French 5-factor:
        R_i - R_f = alpha + beta_MKT * (R_m - R_f)
                    + beta_SMB * SMB
                    + beta_HML * HML
                    + beta_RMW * RMW
                    + beta_CMA * CMA
                    + epsilon

    Macro factors:
        R_i - R_f = alpha + beta_GDP * GDP_surprise
                    + beta_INF * inflation_surprise
                    + beta_RATE * rate_surprise
                    + beta_USD * USD_change
                    + epsilon

    Estimation: OLS
        beta = (X^T X)^{-1} X^T y
        se(beta) = sqrt(diag(sigma^2 * (X^T X)^{-1}))
        t-stat = beta / se(beta)
        R^2 = 1 - SS_res / SS_tot
    """

    FAMA_FRENCH_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    MACRO_FACTORS = ["GDP_surprise", "inflation_surprise", "rate_surprise", "USD_change"]

    def __init__(self):
        self.betas: Dict[str, float] = {}
        self.alpha: float = 0.0
        self.r_squared: float = 0.0
        self.t_stats: Dict[str, float] = {}
        self.residual_std: float = 0.0
        self._is_fitted = False

    def fit(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None,
    ) -> "FactorModel":
        """
        Fit factor model via OLS.

        beta = (X^T X)^{-1} X^T y

        Parameters
        ----------
        returns : pd.Series
            Asset excess returns (or total if risk_free_rate provided).
        factor_returns : pd.DataFrame
            Factor return series (columns = factor names).
        risk_free_rate : pd.Series, optional
            Risk-free rate series. If provided, excess returns = returns - Rf.

        Returns
        -------
        self
        """
        # Align indices
        common_idx = returns.index.intersection(factor_returns.index)
        y = returns.loc[common_idx].values
        X = factor_returns.loc[common_idx].values

        if risk_free_rate is not None:
            rf = risk_free_rate.reindex(common_idx).fillna(0).values
            y = y - rf

        # Add intercept
        n = len(y)
        X_with_intercept = np.column_stack([np.ones(n), X])

        # OLS: beta = (X^T X)^{-1} X^T y
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            beta_hat = XtX_inv @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            XtX_reg = X_with_intercept.T @ X_with_intercept + 1e-6 * np.eye(X_with_intercept.shape[1])
            XtX_inv = np.linalg.inv(XtX_reg)
            beta_hat = XtX_inv @ X_with_intercept.T @ y

        self.alpha = float(beta_hat[0])

        factor_names = list(factor_returns.columns)
        self.betas = {name: float(beta_hat[i + 1]) for i, name in enumerate(factor_names)}

        # Residuals and R^2
        y_hat = X_with_intercept @ beta_hat
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = float(1 - ss_res / max(ss_tot, 1e-10))

        # Standard errors: se(beta) = sqrt(diag(sigma^2 * (X^T X)^{-1}))
        sigma2 = ss_res / max(n - X_with_intercept.shape[1], 1)
        self.residual_std = float(np.sqrt(sigma2))
        se = np.sqrt(np.diag(sigma2 * XtX_inv))

        # t-statistics: t = beta / se(beta)
        self.t_stats = {"alpha": float(beta_hat[0] / max(se[0], 1e-10))}
        for i, name in enumerate(factor_names):
            self.t_stats[name] = float(beta_hat[i + 1] / max(se[i + 1], 1e-10))

        self._is_fitted = True
        logger.info("Factor model fitted: R^2=%.4f, alpha=%.6f", self.r_squared, self.alpha)
        return self

    def predict(self, factor_returns: pd.DataFrame) -> pd.Series:
        """Predict returns given factor values."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        predicted = pd.Series(self.alpha, index=factor_returns.index)
        for name, beta in self.betas.items():
            if name in factor_returns.columns:
                predicted += beta * factor_returns[name]
        return predicted

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Return factor exposures with significance."""
        return {
            "alpha": self.alpha,
            "betas": self.betas,
            "t_stats": self.t_stats,
            "r_squared": self.r_squared,
            "residual_std": self.residual_std,
        }


# ---------------------------------------------------------------------------
# Nowcasting via Principal Component Regression
# ---------------------------------------------------------------------------

class NowcastingEngine:
    """
    GDP nowcasting using Dynamic Factor Model / Principal Component Regression.

    Method:
        1. Standardise high-frequency indicators: z_i = (x_i - mu_i) / sigma_i
        2. Extract principal components: Z = U S V^T (SVD)
           PC_k = V[:, k] (first K components explaining >90% variance)
        3. Regress GDP on PCs: GDP_t = alpha + sum(beta_k * PC_k,t) + epsilon
        4. Nowcast: GDP_now = alpha + sum(beta_k * PC_k,now)

    Variance explained by k-th component:
        VE_k = lambda_k / sum(lambda_j)
        where lambda_k = S[k]^2 / (n-1)
    """

    def __init__(self, n_components: int = 5, variance_threshold: float = 0.90):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_std: Optional[np.ndarray] = None
        self.regression_betas: Optional[np.ndarray] = None
        self.regression_alpha: float = 0.0
        self.variance_explained: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        indicators: pd.DataFrame,
        target_gdp: pd.Series,
    ) -> "NowcastingEngine":
        """
        Fit nowcasting model via PCR.

        Parameters
        ----------
        indicators : pd.DataFrame
            High-frequency indicators (monthly/weekly).
        target_gdp : pd.Series
            GDP growth (quarterly, aligned to indicator dates).

        Returns
        -------
        self
        """
        # Align
        common_idx = indicators.index.intersection(target_gdp.index)
        X = indicators.loc[common_idx].values
        y = target_gdp.loc[common_idx].values

        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            logger.warning("Insufficient data for nowcasting (%d observations)", len(X))
            return self

        # Standardise: z_i = (x_i - mu) / sigma
        self.pca_mean = X.mean(axis=0)
        self.pca_std = X.std(axis=0)
        self.pca_std[self.pca_std == 0] = 1.0
        Z = (X - self.pca_mean) / self.pca_std

        # SVD: Z = U S V^T
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)

        # Variance explained
        eigenvalues = S ** 2 / (len(Z) - 1)
        total_var = eigenvalues.sum()
        self.variance_explained = eigenvalues / total_var
        cumulative_var = np.cumsum(self.variance_explained)

        # Determine number of components
        n_comp = min(
            self.n_components,
            int(np.searchsorted(cumulative_var, self.variance_threshold) + 1),
            len(S),
        )

        logger.info("PCR: using %d components (%.1f%% variance explained)",
                     n_comp, cumulative_var[n_comp - 1] * 100)

        # Principal components: PC = Z @ V^T[:n_comp]^T = Z @ V[:n_comp].T
        self.pca_components = Vt[:n_comp].T  # (D, n_comp)
        PCs = Z @ self.pca_components  # (T, n_comp)

        # Regress GDP on PCs: y = alpha + beta^T * PC
        PC_with_intercept = np.column_stack([np.ones(len(PCs)), PCs])
        beta_hat = np.linalg.lstsq(PC_with_intercept, y, rcond=None)[0]
        self.regression_alpha = float(beta_hat[0])
        self.regression_betas = beta_hat[1:]

        self._is_fitted = True
        return self

    def nowcast(self, current_indicators: pd.Series) -> Dict[str, float]:
        """
        Generate GDP nowcast from current indicator values.

        Returns
        -------
        dict
            {"gdp_nowcast": float, "confidence_interval": (low, high)}
        """
        if not self._is_fitted:
            raise RuntimeError("Nowcasting model not fitted.")

        x = current_indicators.values.reshape(1, -1)
        z = (x - self.pca_mean) / self.pca_std
        pcs = z @ self.pca_components  # (1, n_comp)
        gdp_nowcast = float(self.regression_alpha + pcs @ self.regression_betas)

        # Simple confidence interval based on model uncertainty
        ci_width = abs(gdp_nowcast) * 0.2  # 20% relative width
        return {
            "gdp_nowcast": gdp_nowcast,
            "confidence_interval_low": gdp_nowcast - ci_width,
            "confidence_interval_high": gdp_nowcast + ci_width,
        }


# ---------------------------------------------------------------------------
# MacroMLEngine -- main orchestrator
# ---------------------------------------------------------------------------

class MacroMLEngine:
    """
    Macro-driven ML engine that identifies regime changes and their
    investment implications.

    Integrates:
    - Hidden Markov Model for regime detection
    - Fama-French 5-factor + macro factor models
    - GDP nowcasting via Principal Component Regression
    - Second-derivative effect analysis
    - Regime-optimal asset allocation
    - Trade idea generation
    """

    # Optimal allocations per regime (research-based priors)
    # Allocation weights sum to 1.0 within each regime
    REGIME_ALLOCATIONS: Dict[MarketRegime, Dict[AssetClass, float]] = {
        MarketRegime.EXPANSION: {
            AssetClass.EQUITY: 0.55,
            AssetClass.BOND: 0.10,
            AssetClass.COMMODITY: 0.10,
            AssetClass.CRYPTO: 0.05,
            AssetClass.FX: 0.05,
            AssetClass.ETF: 0.10,
            AssetClass.INDEX: 0.05,
        },
        MarketRegime.PEAK: {
            AssetClass.EQUITY: 0.30,
            AssetClass.BOND: 0.25,
            AssetClass.COMMODITY: 0.15,
            AssetClass.CRYPTO: 0.02,
            AssetClass.FX: 0.08,
            AssetClass.ETF: 0.15,
            AssetClass.INDEX: 0.05,
        },
        MarketRegime.CONTRACTION: {
            AssetClass.EQUITY: 0.10,
            AssetClass.BOND: 0.40,
            AssetClass.COMMODITY: 0.15,
            AssetClass.CRYPTO: 0.01,
            AssetClass.FX: 0.09,
            AssetClass.ETF: 0.20,
            AssetClass.INDEX: 0.05,
        },
        MarketRegime.TROUGH: {
            AssetClass.EQUITY: 0.40,
            AssetClass.BOND: 0.20,
            AssetClass.COMMODITY: 0.10,
            AssetClass.CRYPTO: 0.05,
            AssetClass.FX: 0.05,
            AssetClass.ETF: 0.15,
            AssetClass.INDEX: 0.05,
        },
    }

    def __init__(self):
        self.hmm = HiddenMarkovRegimeModel()
        self.factor_model = FactorModel()
        self.nowcaster = NowcastingEngine()
        self._current_regime: Optional[MarketRegime] = None
        self._regime_history: List[MarketRegime] = []

    def detect_regime(
        self,
        macro_indicators: pd.DataFrame,
    ) -> MarketRegime:
        """
        Detect current macroeconomic regime using HMM.

        Emission variables:
        - GDP growth (QoQ annualised)
        - Unemployment rate change
        - Yield curve slope (10Y - 2Y)

        Parameters
        ----------
        macro_indicators : pd.DataFrame
            Columns should include GDP growth, unemployment change,
            yield curve slope.

        Returns
        -------
        MarketRegime
        """
        # Standardise observations
        obs = macro_indicators.values.astype(float)
        mask = ~np.isnan(obs).any(axis=1)
        obs_clean = obs[mask]

        if len(obs_clean) < 20:
            logger.warning("Insufficient macro data for regime detection (%d obs)", len(obs_clean))
            return MarketRegime.EXPANSION  # default

        # Standardise
        obs_mean = obs_clean.mean(axis=0)
        obs_std = obs_clean.std(axis=0)
        obs_std[obs_std == 0] = 1.0
        obs_standardised = (obs_clean - obs_mean) / obs_std

        # Fit HMM and decode
        self.hmm.fit(obs_standardised)
        regimes = self.hmm.decode(obs_standardised)

        self._regime_history = regimes
        self._current_regime = regimes[-1]

        logger.info("Detected regime: %s (from %d observations)",
                     self._current_regime.value, len(obs_clean))
        return self._current_regime

    def predict_regime_transition(
        self,
        current_regime: MarketRegime,
        leading_indicators: Optional[pd.DataFrame] = None,
    ) -> RegimeTransition:
        """
        Predict probability of regime transitions.

        Uses the HMM transition matrix: P(S_{t+1} | S_t)
        Optionally adjusted by leading indicator signals.

        Parameters
        ----------
        current_regime : MarketRegime
        leading_indicators : pd.DataFrame, optional
            Leading indicators to adjust transition probabilities.

        Returns
        -------
        RegimeTransition
        """
        if not self.hmm._is_fitted:
            # Use prior transition probabilities
            base_probs = {
                MarketRegime.EXPANSION: {MarketRegime.EXPANSION: 0.85, MarketRegime.PEAK: 0.10,
                                         MarketRegime.CONTRACTION: 0.03, MarketRegime.TROUGH: 0.02},
                MarketRegime.PEAK: {MarketRegime.EXPANSION: 0.10, MarketRegime.PEAK: 0.50,
                                    MarketRegime.CONTRACTION: 0.35, MarketRegime.TROUGH: 0.05},
                MarketRegime.CONTRACTION: {MarketRegime.EXPANSION: 0.05, MarketRegime.PEAK: 0.02,
                                           MarketRegime.CONTRACTION: 0.70, MarketRegime.TROUGH: 0.23},
                MarketRegime.TROUGH: {MarketRegime.EXPANSION: 0.35, MarketRegime.PEAK: 0.05,
                                      MarketRegime.CONTRACTION: 0.10, MarketRegime.TROUGH: 0.50},
            }
            probs = base_probs[current_regime]
        else:
            probs = self.hmm.predict_next_state(current_regime)

        # Adjust with leading indicators if provided
        li_signals: Dict[str, float] = {}
        if leading_indicators is not None and not leading_indicators.empty:
            for col in leading_indicators.columns:
                latest = leading_indicators[col].dropna().iloc[-1]
                mean = leading_indicators[col].mean()
                std = leading_indicators[col].std()
                z_score = (latest - mean) / max(std, 1e-10)
                li_signals[col] = float(z_score)

            # Aggregate signal
            avg_signal = np.mean(list(li_signals.values()))
            if avg_signal > 1.0:
                # Positive signal: boost expansion probability
                probs[MarketRegime.EXPANSION] *= 1.2
                probs[MarketRegime.CONTRACTION] *= 0.8
            elif avg_signal < -1.0:
                # Negative signal: boost contraction probability
                probs[MarketRegime.CONTRACTION] *= 1.2
                probs[MarketRegime.EXPANSION] *= 0.8

            # Re-normalise
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}

        most_likely = max(probs, key=probs.get)
        confidence = probs[most_likely]

        return RegimeTransition(
            current_regime=current_regime,
            transition_probabilities=probs,
            most_likely_next=most_likely,
            confidence=confidence,
            leading_indicators_signal=li_signals,
        )

    def factor_exposure_analysis(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Analyse portfolio factor exposures using Fama-French 5-factor model.

        R_i - R_f = alpha + beta_MKT*(R_m-R_f) + beta_SMB*SMB
                    + beta_HML*HML + beta_RMW*RMW + beta_CMA*CMA

        Parameters
        ----------
        portfolio_returns : pd.Series
        factor_returns : pd.DataFrame
        risk_free_rate : pd.Series, optional

        Returns
        -------
        dict
            Factor exposures, t-statistics, R-squared, alpha.
        """
        self.factor_model.fit(portfolio_returns, factor_returns, risk_free_rate)
        return self.factor_model.get_exposure_summary()

    def macro_event_impact(
        self,
        event_type: str,
        severity: float,
    ) -> Dict[AssetClass, float]:
        """
        Estimate impact of a macro event on each asset class.

        Impact model:
            impact_i = sensitivity_i * severity * regime_modifier

        Sensitivity matrix (empirical estimates):
            Asset class sensitivities to common macro shocks.

        Parameters
        ----------
        event_type : str
            One of: "rate_hike", "rate_cut", "inflation_shock", "growth_shock",
                    "credit_crisis", "geopolitical", "pandemic", "fiscal_stimulus"
        severity : float
            Severity on scale [-1, 1] (negative = contractionary).

        Returns
        -------
        dict
            AssetClass -> expected return impact (annualised %).
        """
        # Empirical sensitivity matrix: event_type -> {asset_class: sensitivity}
        sensitivity_matrix: Dict[str, Dict[AssetClass, float]] = {
            "rate_hike": {
                AssetClass.EQUITY: -0.15, AssetClass.BOND: -0.25,
                AssetClass.COMMODITY: -0.05, AssetClass.CRYPTO: -0.20,
                AssetClass.FX: 0.10, AssetClass.ETF: -0.12, AssetClass.INDEX: -0.13,
            },
            "rate_cut": {
                AssetClass.EQUITY: 0.15, AssetClass.BOND: 0.20,
                AssetClass.COMMODITY: 0.08, AssetClass.CRYPTO: 0.18,
                AssetClass.FX: -0.08, AssetClass.ETF: 0.12, AssetClass.INDEX: 0.14,
            },
            "inflation_shock": {
                AssetClass.EQUITY: -0.10, AssetClass.BOND: -0.30,
                AssetClass.COMMODITY: 0.25, AssetClass.CRYPTO: 0.05,
                AssetClass.FX: -0.05, AssetClass.ETF: -0.08, AssetClass.INDEX: -0.10,
            },
            "growth_shock": {
                AssetClass.EQUITY: -0.25, AssetClass.BOND: 0.15,
                AssetClass.COMMODITY: -0.15, AssetClass.CRYPTO: -0.25,
                AssetClass.FX: -0.10, AssetClass.ETF: -0.20, AssetClass.INDEX: -0.22,
            },
            "credit_crisis": {
                AssetClass.EQUITY: -0.35, AssetClass.BOND: 0.10,
                AssetClass.COMMODITY: -0.20, AssetClass.CRYPTO: -0.40,
                AssetClass.FX: 0.05, AssetClass.ETF: -0.30, AssetClass.INDEX: -0.32,
            },
            "geopolitical": {
                AssetClass.EQUITY: -0.12, AssetClass.BOND: 0.08,
                AssetClass.COMMODITY: 0.15, AssetClass.CRYPTO: -0.10,
                AssetClass.FX: 0.05, AssetClass.ETF: -0.10, AssetClass.INDEX: -0.11,
            },
            "pandemic": {
                AssetClass.EQUITY: -0.30, AssetClass.BOND: 0.20,
                AssetClass.COMMODITY: -0.25, AssetClass.CRYPTO: -0.15,
                AssetClass.FX: 0.08, AssetClass.ETF: -0.25, AssetClass.INDEX: -0.28,
            },
            "fiscal_stimulus": {
                AssetClass.EQUITY: 0.20, AssetClass.BOND: -0.10,
                AssetClass.COMMODITY: 0.12, AssetClass.CRYPTO: 0.15,
                AssetClass.FX: -0.05, AssetClass.ETF: 0.15, AssetClass.INDEX: 0.18,
            },
        }

        if event_type not in sensitivity_matrix:
            logger.warning("Unknown event type: %s. Using neutral impact.", event_type)
            return {ac: 0.0 for ac in AssetClass}

        sensitivities = sensitivity_matrix[event_type]

        # Regime modifier: amplify in contraction, dampen in expansion
        regime_modifier = 1.0
        if self._current_regime == MarketRegime.CONTRACTION:
            regime_modifier = 1.3  # amplified reaction
        elif self._current_regime == MarketRegime.EXPANSION:
            regime_modifier = 0.8  # dampened reaction
        elif self._current_regime == MarketRegime.TROUGH:
            regime_modifier = 1.1

        impacts: Dict[AssetClass, float] = {}
        for ac, sensitivity in sensitivities.items():
            # impact = sensitivity * severity * regime_modifier
            impacts[ac] = round(sensitivity * severity * regime_modifier, 4)

        return impacts

    def second_derivative_effects(
        self,
        primary_shock: str,
        severity: float = 1.0,
    ) -> List[CascadingEffect]:
        """
        Model second and third-order cascading effects of a macro shock.

        Transmission channels:
        1. Direct price impact (1st order)
        2. Portfolio rebalancing flows (2nd order)
        3. Liquidity/credit tightening (2nd order)
        4. Earnings revision chain (3rd order)
        5. Sentiment contagion (3rd order)

        Parameters
        ----------
        primary_shock : str
            Type of primary shock.
        severity : float
            Magnitude on [-1, 1] scale.

        Returns
        -------
        list of CascadingEffect
        """
        effects: List[CascadingEffect] = []

        # Cascading effect templates
        shock_chains: Dict[str, List[Dict[str, Any]]] = {
            "rate_hike": [
                {"order": 1, "desc": "Bond prices decline as yields rise",
                 "ac": AssetClass.BOND, "impact": -0.25, "lag": 0, "channel": "direct_price"},
                {"order": 1, "desc": "Growth stocks derate on higher discount rate",
                 "ac": AssetClass.EQUITY, "impact": -0.15, "lag": 1, "channel": "direct_price"},
                {"order": 2, "desc": "Margin calls force equity liquidation",
                 "ac": AssetClass.EQUITY, "impact": -0.08, "lag": 5, "channel": "portfolio_rebalancing"},
                {"order": 2, "desc": "USD strengthens, pressuring commodity prices",
                 "ac": AssetClass.COMMODITY, "impact": -0.10, "lag": 3, "channel": "fx_transmission"},
                {"order": 2, "desc": "Crypto sell-off as risk appetite falls",
                 "ac": AssetClass.CRYPTO, "impact": -0.20, "lag": 2, "channel": "risk_sentiment"},
                {"order": 3, "desc": "EM sovereign spreads widen, capital outflows",
                 "ac": AssetClass.BOND, "impact": -0.05, "lag": 15, "channel": "credit_tightening"},
                {"order": 3, "desc": "Corporate earnings downgrades from higher debt costs",
                 "ac": AssetClass.EQUITY, "impact": -0.10, "lag": 30, "channel": "earnings_revision"},
                {"order": 3, "desc": "Housing market slowdown feeds back to consumer sentiment",
                 "ac": AssetClass.ETF, "impact": -0.05, "lag": 60, "channel": "sentiment_contagion"},
            ],
            "inflation_shock": [
                {"order": 1, "desc": "Real bond yields spike, nominal bonds sell off",
                 "ac": AssetClass.BOND, "impact": -0.30, "lag": 0, "channel": "direct_price"},
                {"order": 1, "desc": "Commodities rally on inflation pass-through",
                 "ac": AssetClass.COMMODITY, "impact": 0.25, "lag": 0, "channel": "direct_price"},
                {"order": 2, "desc": "Equities reprice on margin compression",
                 "ac": AssetClass.EQUITY, "impact": -0.12, "lag": 5, "channel": "earnings_revision"},
                {"order": 2, "desc": "Central bank hawkishness reprices FX carry trades",
                 "ac": AssetClass.FX, "impact": -0.08, "lag": 7, "channel": "policy_transmission"},
                {"order": 3, "desc": "Consumer spending declines, cyclical sectors hit",
                 "ac": AssetClass.ETF, "impact": -0.10, "lag": 30, "channel": "demand_destruction"},
                {"order": 3, "desc": "Wage-price spiral risk reprices long-duration assets",
                 "ac": AssetClass.BOND, "impact": -0.08, "lag": 60, "channel": "expectations_channel"},
            ],
            "growth_shock": [
                {"order": 1, "desc": "Equities decline on earnings expectations",
                 "ac": AssetClass.EQUITY, "impact": -0.25, "lag": 0, "channel": "direct_price"},
                {"order": 1, "desc": "Flight to quality into treasuries",
                 "ac": AssetClass.BOND, "impact": 0.15, "lag": 0, "channel": "safe_haven"},
                {"order": 2, "desc": "Industrial commodities decline on demand outlook",
                 "ac": AssetClass.COMMODITY, "impact": -0.18, "lag": 5, "channel": "demand_destruction"},
                {"order": 2, "desc": "Credit spreads widen, corporate bonds sell off",
                 "ac": AssetClass.BOND, "impact": -0.10, "lag": 10, "channel": "credit_tightening"},
                {"order": 2, "desc": "Crypto correlation with risk assets drives sell-off",
                 "ac": AssetClass.CRYPTO, "impact": -0.25, "lag": 3, "channel": "risk_sentiment"},
                {"order": 3, "desc": "Bank lending tightens, small-cap impact",
                 "ac": AssetClass.EQUITY, "impact": -0.08, "lag": 30, "channel": "credit_channel"},
                {"order": 3, "desc": "Dollar strengthens as growth differential widens",
                 "ac": AssetClass.FX, "impact": 0.10, "lag": 20, "channel": "fx_transmission"},
            ],
            "credit_crisis": [
                {"order": 1, "desc": "Credit spreads explode, HY bonds crash",
                 "ac": AssetClass.BOND, "impact": -0.35, "lag": 0, "channel": "direct_price"},
                {"order": 1, "desc": "Equities gap down on systemic risk",
                 "ac": AssetClass.EQUITY, "impact": -0.30, "lag": 0, "channel": "direct_price"},
                {"order": 2, "desc": "Forced deleveraging across all risk assets",
                 "ac": AssetClass.ETF, "impact": -0.25, "lag": 3, "channel": "portfolio_rebalancing"},
                {"order": 2, "desc": "Gold rallies on safe-haven flows",
                 "ac": AssetClass.COMMODITY, "impact": 0.15, "lag": 2, "channel": "safe_haven"},
                {"order": 2, "desc": "Crypto flash crash from liquidation cascades",
                 "ac": AssetClass.CRYPTO, "impact": -0.40, "lag": 1, "channel": "liquidity_crisis"},
                {"order": 3, "desc": "Sovereign risk repricing in weak economies",
                 "ac": AssetClass.BOND, "impact": -0.15, "lag": 20, "channel": "contagion"},
                {"order": 3, "desc": "FX volatility spike as carry trades unwind",
                 "ac": AssetClass.FX, "impact": -0.20, "lag": 10, "channel": "fx_transmission"},
            ],
        }

        chains = shock_chains.get(primary_shock, [])
        if not chains:
            # Generic shock with attenuating impact
            for order in [1, 2, 3]:
                for ac in AssetClass:
                    effects.append(CascadingEffect(
                        order=order,
                        description=f"Order-{order} {primary_shock} impact on {ac.value}",
                        affected_asset_class=ac,
                        expected_impact=severity * (-0.1) * (0.5 ** (order - 1)),
                        confidence=0.5 / order,
                        lag_days=order * 10,
                        transmission_channel="generic",
                    ))
            return effects

        for chain in chains:
            # Scale by severity, attenuate confidence by order
            effects.append(CascadingEffect(
                order=chain["order"],
                description=chain["desc"],
                affected_asset_class=chain["ac"],
                expected_impact=round(chain["impact"] * severity, 4),
                confidence=round(max(0.3, 0.9 - 0.2 * (chain["order"] - 1)), 2),
                lag_days=chain["lag"],
                transmission_channel=chain["channel"],
            ))

        effects.sort(key=lambda e: (e.order, -abs(e.expected_impact)))
        return effects

    def optimal_asset_allocation_for_regime(
        self,
        regime: MarketRegime,
    ) -> Dict[AssetClass, float]:
        """
        Return optimal asset allocation weights for a given regime.

        Based on historical regime performance analysis:
        - Expansion: overweight equities, underweight bonds
        - Peak: reduce equity, increase commodities & defensive
        - Contraction: overweight bonds, underweight risky assets
        - Trough: begin increasing equity, maintain bond floor

        Parameters
        ----------
        regime : MarketRegime

        Returns
        -------
        dict
            AssetClass -> allocation weight (sums to 1.0).
        """
        allocation = self.REGIME_ALLOCATIONS.get(regime)
        if allocation is None:
            # Default balanced
            n = len(AssetClass)
            allocation = {ac: 1.0 / n for ac in AssetClass}

        # Verify weights sum to 1.0
        total = sum(allocation.values())
        if abs(total - 1.0) > 1e-6:
            allocation = {k: v / total for k, v in allocation.items()}

        return allocation

    def generate_macro_trade_ideas(
        self,
        regime: MarketRegime,
        regime_transition_prob: Optional[Dict[MarketRegime, float]] = None,
    ) -> List[TradeThesis]:
        """
        Generate actionable trade ideas based on current regime and transition probabilities.

        Trade idea generation logic:
        1. Identify overweight/underweight shifts from current to expected regime
        2. Select highest-conviction symbols within favoured asset classes
        3. Compute expected return from regime transition
        4. Risk-adjust by transition probability confidence

        Parameters
        ----------
        regime : MarketRegime
        regime_transition_prob : dict, optional
            Probability of transitioning to each regime.

        Returns
        -------
        list of TradeThesis
        """
        ideas: List[TradeThesis] = []

        # Current regime allocation
        current_alloc = self.REGIME_ALLOCATIONS[regime]

        # Determine expected next regime
        if regime_transition_prob:
            next_regime = max(regime_transition_prob, key=regime_transition_prob.get)
            transition_conf = regime_transition_prob[next_regime]
        else:
            next_regime = regime
            transition_conf = 0.5

        next_alloc = self.REGIME_ALLOCATIONS[next_regime]

        # Identify allocation shifts
        universe = get_full_universe()
        for ac in AssetClass:
            delta = next_alloc.get(ac, 0) - current_alloc.get(ac, 0)
            symbols = universe.get(ac, [])

            if abs(delta) < 0.03 or not symbols:
                continue  # Skip small shifts

            direction = "long" if delta > 0 else "short"
            # Pick representative symbol
            symbol = symbols[0]

            expected_return = abs(delta) * 0.5  # simplified estimate
            confidence = min(transition_conf * (1.0 + abs(delta) * 2), 0.95)

            risk_factors = []
            if regime == MarketRegime.CONTRACTION:
                risk_factors.append("recession_deepening")
            if regime == MarketRegime.PEAK:
                risk_factors.append("policy_error")
            if ac == AssetClass.CRYPTO:
                risk_factors.append("regulatory_risk")
            risk_factors.append("model_uncertainty")

            rationale = (
                f"Regime transition {regime.value} -> {next_regime.value} "
                f"(p={transition_conf:.2f}) suggests {direction} {ac.value}. "
                f"Allocation shift: {delta:+.1%}."
            )

            ideas.append(TradeThesis(
                asset_class=ac,
                direction=direction,
                symbol=symbol,
                rationale=rationale,
                expected_return=round(expected_return, 4),
                confidence=round(confidence, 3),
                horizon_days=63,  # quarterly horizon for macro trades
                risk_factors=risk_factors,
                regime=next_regime,
            ))

        # Sort by confidence * expected_return (risk-adjusted conviction)
        ideas.sort(key=lambda t: t.confidence * t.expected_return, reverse=True)
        return ideas


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print("=== Macro ML Engine ===")
    print()

    engine = MacroMLEngine()

    print("Regime types:", [r.value for r in MarketRegime])
    print()

    print("Regime allocations:")
    for regime, alloc in MacroMLEngine.REGIME_ALLOCATIONS.items():
        parts = [f"{ac.value}={w:.0%}" for ac, w in alloc.items()]
        print(f"  {regime.value}: {', '.join(parts)}")
    print()

    print("Mathematical formulas:")
    print("  HMM Baum-Welch:")
    print("    alpha_t(i) = [sum_j alpha_{t-1}(j) * A[j,i]] * B[t,i]")
    print("    gamma_t(i) = alpha_t(i) * beta_t(i) / P(O)")
    print("    A[i,j] = sum_t(xi_t(i,j)) / sum_t(gamma_t(i))")
    print()
    print("  Fama-French 5-factor:")
    print("    R_i - R_f = alpha + beta_MKT*(R_m-R_f) + beta_SMB*SMB")
    print("                + beta_HML*HML + beta_RMW*RMW + beta_CMA*CMA")
    print()
    print("  Nowcasting PCR:")
    print("    z_i = (x_i - mu) / sigma")
    print("    PC_k via SVD: Z = U S V^T")
    print("    GDP_now = alpha + sum(beta_k * PC_k)")
    print()
    print("  Viterbi decoding:")
    print("    delta_t(j) = max_i [delta_{t-1}(i) * A[i,j]] * b_j(O_t)")
    print()

    # Demo: second derivative effects
    print("Second derivative effects for 'rate_hike':")
    effects = engine.second_derivative_effects("rate_hike", severity=0.8)
    for e in effects[:5]:
        print(f"  Order {e.order}: {e.description}")
        print(f"    Impact: {e.expected_impact:+.2%}, Lag: {e.lag_days}d, Channel: {e.transmission_channel}")

    print()
    print("Trade ideas for CONTRACTION regime:")
    ideas = engine.generate_macro_trade_ideas(
        MarketRegime.CONTRACTION,
        {MarketRegime.TROUGH: 0.35, MarketRegime.CONTRACTION: 0.55,
         MarketRegime.EXPANSION: 0.08, MarketRegime.PEAK: 0.02},
    )
    for idea in ideas[:3]:
        print(f"  {idea.direction.upper()} {idea.symbol} ({idea.asset_class.value})")
        print(f"    {idea.rationale}")
        print(f"    Expected return: {idea.expected_return:.2%}, Confidence: {idea.confidence:.1%}")
