"""Statistical Arbitrage Engine — Medallion-style mean-reversion & cointegration.

Implements market-neutral pair trading via:
    - Cointegrated pair detection (Engle-Granger simplified ADF)
    - OLS hedge-ratio estimation
    - Ornstein-Uhlenbeck half-life for mean-reversion speed
    - Factor residual alpha decomposition (5-factor: MKT, SMB, HML, MOM, QUAL)
    - Z-score signal generation with entry/exit/stop thresholds
    - Portfolio beta neutralisation (target: sum(beta) ~ 0)

Signal thresholds:
    Entry long  : z < -2.0  (spread compressed — buy the spread)
    Entry short : z > +2.0  (spread expanded  — sell the spread)
    Exit        : |z| < 0.5 (mean reverted)
    Stop-loss   : |z| > 4.0 (regime break / structural divergence)

All data via OpenBB. Pure Python + numpy. No external stat libraries required.
try/except on ALL external imports — system runs degraded, never broken.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger("metadron.signals.stat_arb")

# ═══════════════════════════════════════════════════════════════════════════
# Import RV pairs from universe engine — graceful fallback
# ═══════════════════════════════════════════════════════════════════════════
try:
    from engine.data.universe_engine import RV_PAIR_MAP, RV_PAIRS
except ImportError:
    try:
        from ..data.universe_engine import RV_PAIR_MAP, RV_PAIRS
    except ImportError:
        logger.warning("universe_engine not available — using DEFAULT_PAIRS fallback")
        RV_PAIRS = [
            ("AAPL", "MSFT"), ("GOOGL", "META"), ("AMZN", "WMT"),
            ("JPM", "BAC"), ("GS", "MS"), ("XOM", "CVX"),
            ("PFE", "MRK"), ("JNJ", "ABT"), ("KO", "PEP"),
            ("HD", "LOW"), ("V", "MA"), ("DIS", "CMCSA"),
            ("UNH", "CI"), ("BA", "LMT"), ("CAT", "DE"),
            ("NVDA", "AMD"), ("CRM", "ORCL"), ("NFLX", "DIS"),
            ("COST", "TGT"), ("UPS", "FDX"), ("NEE", "DUK"),
            ("SPG", "O"), ("SLB", "HAL"), ("MCD", "SBUX"),
            ("NKE", "LULU"), ("TSLA", "F"),
        ]
        RV_PAIR_MAP = {}
        for _a, _b in RV_PAIRS:
            RV_PAIR_MAP[_a] = _b
            RV_PAIR_MAP[_b] = _a

DEFAULT_PAIRS = {
    "tech":       [("AAPL", "MSFT"), ("NVDA", "AMD"), ("CRM", "ORCL"), ("GOOGL", "META")],
    "financials": [("JPM", "BAC"), ("GS", "MS"), ("V", "MA")],
    "energy":     [("XOM", "CVX"), ("SLB", "HAL")],
    "healthcare": [("PFE", "MRK"), ("JNJ", "ABT"), ("UNH", "CI")],
    "consumer":   [("KO", "PEP"), ("HD", "LOW"), ("MCD", "SBUX"),
                   ("NKE", "LULU"), ("COST", "TGT")],
    "industrial": [("BA", "LMT"), ("CAT", "DE"), ("UPS", "FDX")],
    "media":      [("DIS", "CMCSA"), ("NFLX", "DIS")],
    "utilities":  [("NEE", "DUK")],
    "reits":      [("SPG", "O")],
    "auto":       [("TSLA", "F")],
    "staples":    [("AMZN", "WMT")],
}


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Constants
# ═══════════════════════════════════════════════════════════════════════════
class SignalDirection(str, Enum):
    """Trade direction for a stat-arb pair."""
    LONG_A_SHORT_B = "LONG_A_SHORT_B"
    SHORT_A_LONG_B = "SHORT_A_LONG_B"
    EXIT = "EXIT"
    STOP = "STOP"
    NEUTRAL = "NEUTRAL"


class PairStatus(str, Enum):
    """Lifecycle status of a cointegrated pair."""
    COINTEGRATED = "COINTEGRATED"
    DECOUPLED = "DECOUPLED"
    MONITORING = "MONITORING"
    TRADING = "TRADING"
    STOPPED = "STOPPED"


# Z-score thresholds (Medallion-calibrated)
ZSCORE_ENTRY_LONG = -2.0       # spread compressed — buy the spread
ZSCORE_ENTRY_SHORT = 2.0       # spread expanded  — sell the spread
ZSCORE_EXIT = 0.5              # mean reverted — close position
ZSCORE_STOP = 4.0              # regime break — emergency exit

# Cointegration thresholds
COINT_PVALUE_THRESHOLD = 0.05  # Engle-Granger significance
MIN_CORRELATION = 0.50         # minimum Pearson correlation for pair candidacy
MAX_HALF_LIFE = 120            # days — reject pairs with slow reversion
MIN_HALF_LIFE = 2              # days — reject noise pairs
MIN_LOOKBACK = 60              # minimum price observations

# Factor names
FACTOR_NAMES = ["MARKET", "SIZE", "VALUE", "MOMENTUM", "QUALITY"]


# ═══════════════════════════════════════════════════════════════════════════
# PairStatistics — core output dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PairStatistics:
    """Complete statistical profile for a cointegrated pair.

    Attributes
    ----------
    ticker_a : str          — first leg ticker
    ticker_b : str          — second leg ticker (hedge leg)
    correlation : float     — Pearson correlation of returns
    cointegration_pvalue : float — Engle-Granger ADF p-value
    half_life : float       — Ornstein-Uhlenbeck mean-reversion speed (days)
    spread_zscore : float   — current z-score of the spread
    hedge_ratio : float     — OLS beta (units of B per unit of A)
    signal_strength : float — composite signal strength [0, 1]
    status : str            — pair lifecycle status
    timestamp : str         — ISO 8601 timestamp
    """
    ticker_a: str = ""
    ticker_b: str = ""
    correlation: float = 0.0
    cointegration_pvalue: float = 1.0
    half_life: float = 0.0
    spread_zscore: float = 0.0
    hedge_ratio: float = 0.0
    signal_strength: float = 0.0
    status: str = PairStatus.MONITORING.value
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_cointegrated(self) -> bool:
        return self.cointegration_pvalue < COINT_PVALUE_THRESHOLD

    @property
    def is_tradeable(self) -> bool:
        return (
            self.is_cointegrated
            and MIN_HALF_LIFE <= self.half_life <= MAX_HALF_LIFE
            and abs(self.correlation) >= MIN_CORRELATION
        )

    @property
    def direction(self) -> SignalDirection:
        if abs(self.spread_zscore) > ZSCORE_STOP:
            return SignalDirection.STOP
        if self.spread_zscore < ZSCORE_ENTRY_LONG:
            return SignalDirection.LONG_A_SHORT_B
        if self.spread_zscore > ZSCORE_ENTRY_SHORT:
            return SignalDirection.SHORT_A_LONG_B
        if abs(self.spread_zscore) < ZSCORE_EXIT:
            return SignalDirection.EXIT
        return SignalDirection.NEUTRAL


# ═══════════════════════════════════════════════════════════════════════════
# Simplified ADF Test — Engle-Granger cointegration
# ═══════════════════════════════════════════════════════════════════════════
def _adf_test_statistic(series: np.ndarray) -> float:
    """Compute the ADF test statistic for a time series.

    Uses the regression:  delta_y(t) = alpha + gamma * y(t-1) + epsilon
    The ADF statistic is the t-statistic on gamma.

    Parameters
    ----------
    series : np.ndarray
        Time series to test for a unit root.

    Returns
    -------
    float
        ADF t-statistic.  More negative = stronger rejection of unit root.
    """
    if len(series) < 10:
        return 0.0

    y = np.asarray(series, dtype=np.float64)
    dy = np.diff(y)                       # delta y
    y_lag = y[:-1]                        # y(t-1)

    n = len(dy)
    # Design matrix: [y(t-1), 1]
    X = np.column_stack([y_lag, np.ones(n)])

    # OLS: beta = (X'X)^{-1} X'dy
    try:
        XtX = X.T @ X
        Xty = X.T @ dy
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return 0.0

    gamma = beta[0]
    residuals = dy - X @ beta
    sigma2 = float(np.sum(residuals ** 2)) / max(n - 2, 1)

    # Standard error of gamma
    try:
        cov_matrix = sigma2 * np.linalg.inv(XtX)
        se_gamma = float(np.sqrt(abs(cov_matrix[0, 0])))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0

    if se_gamma < 1e-15:
        return 0.0

    return float(gamma / se_gamma)


def _adf_pvalue(adf_stat: float, n: int) -> float:
    """Approximate p-value for ADF test using MacKinnon critical values.

    Uses interpolation of standard critical values for the no-trend case:
        1%  : -3.43
        5%  : -2.86
        10% : -2.57

    Parameters
    ----------
    adf_stat : float
        ADF t-statistic.
    n : int
        Number of observations.

    Returns
    -------
    float
        Approximate p-value in [0, 1].
    """
    # MacKinnon critical values (constant, no trend, n >= 50)
    # These are conservative approximations
    critical_values = [
        (-3.96, 0.001),
        (-3.43, 0.01),
        (-2.86, 0.05),
        (-2.57, 0.10),
        (-1.94, 0.25),
        (-1.62, 0.50),
        (-0.50, 0.75),
        (0.00,  0.90),
        (0.50,  0.95),
        (1.50,  0.99),
    ]

    if adf_stat <= critical_values[0][0]:
        return critical_values[0][1]
    if adf_stat >= critical_values[-1][0]:
        return critical_values[-1][1]

    # Linear interpolation between critical values
    for i in range(len(critical_values) - 1):
        cv_lo, pv_lo = critical_values[i]
        cv_hi, pv_hi = critical_values[i + 1]
        if cv_lo <= adf_stat <= cv_hi:
            frac = (adf_stat - cv_lo) / (cv_hi - cv_lo) if cv_hi != cv_lo else 0.0
            return pv_lo + frac * (pv_hi - pv_lo)

    return 0.50


def engle_granger_test(prices_a: np.ndarray, prices_b: np.ndarray) -> tuple:
    """Engle-Granger two-step cointegration test.

    Step 1: Regress prices_a on prices_b to get residuals.
    Step 2: Test residuals for stationarity with ADF.

    Parameters
    ----------
    prices_a, prices_b : np.ndarray
        Price series for the two assets.

    Returns
    -------
    tuple
        (adf_statistic, p_value, hedge_ratio, residuals)
    """
    a = np.asarray(prices_a, dtype=np.float64)
    b = np.asarray(prices_b, dtype=np.float64)

    n = min(len(a), len(b))
    if n < MIN_LOOKBACK:
        return (0.0, 1.0, 0.0, np.array([]))

    a = a[-n:]
    b = b[-n:]

    # Step 1: OLS regression  a = alpha + beta * b + epsilon
    X = np.column_stack([b, np.ones(n)])
    try:
        beta = np.linalg.lstsq(X, a, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (0.0, 1.0, 0.0, np.array([]))

    hedge_ratio = float(beta[0])
    residuals = a - X @ beta

    # Step 2: ADF test on residuals
    adf_stat = _adf_test_statistic(residuals)
    p_value = _adf_pvalue(adf_stat, n)

    return (adf_stat, p_value, hedge_ratio, residuals)


# ═══════════════════════════════════════════════════════════════════════════
# CointegratedPair — manages a single pair's analytics
# ═══════════════════════════════════════════════════════════════════════════
class CointegratedPair:
    """Analytics engine for a single cointegrated pair.

    Computes hedge ratio, spread, z-score, half-life, and generates
    entry/exit signals based on z-score thresholds.
    """

    def __init__(self, ticker_a: str, ticker_b: str):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self._hedge_ratio: float = 0.0
        self._spread: Optional[np.ndarray] = None
        self._zscore: float = 0.0
        self._half_life: float = 0.0
        self._correlation: float = 0.0
        self._coint_pvalue: float = 1.0
        self._adf_stat: float = 0.0
        self._signal_strength: float = 0.0
        self._status: PairStatus = PairStatus.MONITORING
        self._last_update: str = ""
        self._residuals: Optional[np.ndarray] = None

    @property
    def pair_id(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"

    def compute_hedge_ratio(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """Compute OLS hedge ratio: units of B to hold per unit of A.

        Regression: prices_a = alpha + beta * prices_b + epsilon
        The hedge ratio beta tells us how many shares of B to short
        for every share of A we go long (or vice versa).

        Parameters
        ----------
        prices_a : np.ndarray
            Price series for ticker_a.
        prices_b : np.ndarray
            Price series for ticker_b.

        Returns
        -------
        float
            OLS hedge ratio (beta coefficient).
        """
        a = np.asarray(prices_a, dtype=np.float64)
        b = np.asarray(prices_b, dtype=np.float64)

        n = min(len(a), len(b))
        if n < 2:
            return 0.0

        a = a[-n:]
        b = b[-n:]

        # OLS: a = alpha + beta * b
        X = np.column_stack([b, np.ones(n)])
        try:
            result = np.linalg.lstsq(X, a, rcond=None)
            beta = result[0]
            self._hedge_ratio = float(beta[0])
        except np.linalg.LinAlgError:
            self._hedge_ratio = 0.0

        return self._hedge_ratio

    def compute_spread(self, prices_a: np.ndarray, prices_b: np.ndarray,
                       hedge_ratio: Optional[float] = None) -> np.ndarray:
        """Compute the spread: S(t) = P_a(t) - hedge_ratio * P_b(t).

        The spread is the residual after removing the linear relationship.
        A stationary spread indicates cointegration.

        Parameters
        ----------
        prices_a : np.ndarray
            Price series for ticker_a.
        prices_b : np.ndarray
            Price series for ticker_b.
        hedge_ratio : float, optional
            Override hedge ratio (uses stored value if None).

        Returns
        -------
        np.ndarray
            The spread time series.
        """
        a = np.asarray(prices_a, dtype=np.float64)
        b = np.asarray(prices_b, dtype=np.float64)

        n = min(len(a), len(b))
        a = a[-n:]
        b = b[-n:]

        hr = hedge_ratio if hedge_ratio is not None else self._hedge_ratio
        self._spread = a - hr * b
        return self._spread

    def compute_zscore(self, spread: np.ndarray, lookback: int = 60) -> float:
        """Compute the z-score of the current spread value.

        z = (spread_current - mean(spread[-lookback:])) / std(spread[-lookback:])

        Parameters
        ----------
        spread : np.ndarray
            Spread time series.
        lookback : int
            Rolling window for mean/std estimation.

        Returns
        -------
        float
            Current z-score of the spread.
        """
        s = np.asarray(spread, dtype=np.float64)

        if len(s) < lookback:
            lookback = len(s)
        if lookback < 2:
            self._zscore = 0.0
            return 0.0

        window = s[-lookback:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))

        if std < 1e-12:
            self._zscore = 0.0
            return 0.0

        self._zscore = float((s[-1] - mean) / std)
        return self._zscore

    def compute_half_life(self, spread: np.ndarray) -> float:
        """Estimate Ornstein-Uhlenbeck half-life of mean reversion.

        Model: dS(t) = theta * (mu - S(t)) * dt + sigma * dW(t)
        Regression: delta_S(t) = lambda * S(t-1) + epsilon
        Half-life = -ln(2) / lambda

        A smaller half-life means faster mean reversion and thus
        more attractive for stat-arb trading.

        Parameters
        ----------
        spread : np.ndarray
            Spread time series.

        Returns
        -------
        float
            Half-life in periods (days). Returns MAX_HALF_LIFE * 2
            if the spread is not mean-reverting.
        """
        s = np.asarray(spread, dtype=np.float64)

        if len(s) < 10:
            self._half_life = MAX_HALF_LIFE * 2
            return self._half_life

        # AR(1) regression: delta_s = lambda * s_lag + intercept
        delta_s = np.diff(s)
        s_lag = s[:-1]

        n = len(delta_s)
        X = np.column_stack([s_lag, np.ones(n)])

        try:
            result = np.linalg.lstsq(X, delta_s, rcond=None)
            lam = float(result[0][0])
        except np.linalg.LinAlgError:
            self._half_life = MAX_HALF_LIFE * 2
            return self._half_life

        # lambda must be negative for mean reversion
        if lam >= 0:
            self._half_life = MAX_HALF_LIFE * 2
            return self._half_life

        self._half_life = float(-np.log(2) / lam)

        # Bound the result
        if self._half_life < 0:
            self._half_life = MAX_HALF_LIFE * 2

        return self._half_life

    def _compute_signal_strength(self) -> float:
        """Composite signal strength combining z-score, half-life, and correlation.

        Components (equal-weighted):
            1. Z-score intensity: how far beyond the entry threshold
            2. Half-life attractiveness: faster reversion = stronger signal
            3. Correlation stability: higher correlation = more stable pair
            4. Cointegration strength: lower p-value = stronger relationship

        Returns
        -------
        float
            Signal strength in [0, 1].
        """
        components = []

        # Z-score component: scales from 0 at |z|=2 to 1 at |z|=4
        z_abs = abs(self._zscore)
        if z_abs >= ZSCORE_ENTRY_SHORT:
            z_component = min(1.0, (z_abs - ZSCORE_ENTRY_SHORT) / (ZSCORE_STOP - ZSCORE_ENTRY_SHORT))
        else:
            z_component = 0.0
        components.append(z_component)

        # Half-life component: faster = better, max at 5 days, min at MAX_HALF_LIFE
        if MIN_HALF_LIFE <= self._half_life <= MAX_HALF_LIFE:
            hl_component = 1.0 - (self._half_life - MIN_HALF_LIFE) / (MAX_HALF_LIFE - MIN_HALF_LIFE)
        else:
            hl_component = 0.0
        components.append(hl_component)

        # Correlation component
        corr_component = max(0.0, min(1.0, abs(self._correlation)))
        components.append(corr_component)

        # Cointegration component: lower p-value = better
        if self._coint_pvalue < COINT_PVALUE_THRESHOLD:
            coint_component = 1.0 - (self._coint_pvalue / COINT_PVALUE_THRESHOLD)
        else:
            coint_component = 0.0
        components.append(coint_component)

        self._signal_strength = float(np.mean(components))
        return self._signal_strength

    def full_analysis(self, prices_a: np.ndarray, prices_b: np.ndarray,
                      lookback: int = 60) -> PairStatistics:
        """Run the complete analytical pipeline for this pair.

        Sequence:
            1. Compute correlation
            2. Engle-Granger cointegration test
            3. Compute hedge ratio
            4. Compute spread
            5. Compute z-score
            6. Compute half-life
            7. Compute signal strength
            8. Determine status

        Parameters
        ----------
        prices_a, prices_b : np.ndarray
            Price series for the two tickers.
        lookback : int
            Lookback window for z-score computation.

        Returns
        -------
        PairStatistics
            Complete statistical profile for the pair.
        """
        a = np.asarray(prices_a, dtype=np.float64)
        b = np.asarray(prices_b, dtype=np.float64)

        n = min(len(a), len(b))
        if n < MIN_LOOKBACK:
            return PairStatistics(
                ticker_a=self.ticker_a, ticker_b=self.ticker_b,
                status=PairStatus.MONITORING.value,
            )

        a = a[-n:]
        b = b[-n:]

        # 1. Correlation (returns-based)
        if n > 1:
            ret_a = np.diff(a) / a[:-1]
            ret_b = np.diff(b) / b[:-1]
            # Guard against zero variance
            std_a = np.std(ret_a)
            std_b = np.std(ret_b)
            if std_a > 1e-12 and std_b > 1e-12:
                self._correlation = float(np.corrcoef(ret_a, ret_b)[0, 1])
            else:
                self._correlation = 0.0
        else:
            self._correlation = 0.0

        # 2. Engle-Granger cointegration test
        adf_stat, p_value, hedge_ratio, residuals = engle_granger_test(a, b)
        self._adf_stat = adf_stat
        self._coint_pvalue = p_value
        self._hedge_ratio = hedge_ratio
        self._residuals = residuals

        # 3. Spread
        spread = self.compute_spread(a, b, hedge_ratio)

        # 4. Z-score
        self.compute_zscore(spread, lookback)

        # 5. Half-life
        self.compute_half_life(spread)

        # 6. Signal strength
        self._compute_signal_strength()

        # 7. Status
        if abs(self._zscore) > ZSCORE_STOP:
            self._status = PairStatus.STOPPED
        elif p_value < COINT_PVALUE_THRESHOLD and MIN_HALF_LIFE <= self._half_life <= MAX_HALF_LIFE:
            if abs(self._zscore) >= abs(ZSCORE_ENTRY_LONG):
                self._status = PairStatus.TRADING
            else:
                self._status = PairStatus.COINTEGRATED
        else:
            self._status = PairStatus.DECOUPLED

        self._last_update = datetime.now().isoformat()

        return PairStatistics(
            ticker_a=self.ticker_a,
            ticker_b=self.ticker_b,
            correlation=round(self._correlation, 4),
            cointegration_pvalue=round(self._coint_pvalue, 6),
            half_life=round(self._half_life, 2),
            spread_zscore=round(self._zscore, 4),
            hedge_ratio=round(self._hedge_ratio, 4),
            signal_strength=round(self._signal_strength, 4),
            status=self._status.value,
        )

    def get_signal(self) -> dict:
        """Generate entry/exit signal based on current z-score thresholds.

        Thresholds (Medallion-calibrated):
            Entry long  : z < -2.0  → LONG_A_SHORT_B
            Entry short : z > +2.0  → SHORT_A_LONG_B
            Exit        : |z| < 0.5 → EXIT (take profit / mean reverted)
            Stop        : |z| > 4.0 → STOP (regime break)

        Returns
        -------
        dict
            Signal with direction, z-score, strength, and sizing info.
        """
        z = self._zscore
        direction = SignalDirection.NEUTRAL

        if abs(z) > ZSCORE_STOP:
            direction = SignalDirection.STOP
        elif z < ZSCORE_ENTRY_LONG:
            direction = SignalDirection.LONG_A_SHORT_B
        elif z > ZSCORE_ENTRY_SHORT:
            direction = SignalDirection.SHORT_A_LONG_B
        elif abs(z) < ZSCORE_EXIT:
            direction = SignalDirection.EXIT

        # Position sizing: scale inversely with z-score distance from threshold
        # Max size at |z| = 3.0 (midpoint), reduce at extremes
        if direction in (SignalDirection.LONG_A_SHORT_B, SignalDirection.SHORT_A_LONG_B):
            # Size scales with confidence: 0.5 at entry to 1.0 at |z|=3
            raw_size = min(1.0, (abs(z) - abs(ZSCORE_ENTRY_LONG)) / 2.0)
            position_size = max(0.1, raw_size)
        elif direction == SignalDirection.STOP:
            position_size = 0.0
        elif direction == SignalDirection.EXIT:
            position_size = 0.0
        else:
            position_size = 0.0

        return {
            "pair": self.pair_id,
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "direction": direction.value,
            "zscore": round(z, 4),
            "hedge_ratio": round(self._hedge_ratio, 4),
            "half_life": round(self._half_life, 2),
            "correlation": round(self._correlation, 4),
            "cointegration_pvalue": round(self._coint_pvalue, 6),
            "signal_strength": round(self._signal_strength, 4),
            "position_size": round(position_size, 4),
            "status": self._status.value,
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# FactorResidualModel — 5-factor alpha decomposition
# ═══════════════════════════════════════════════════════════════════════════
class FactorResidualModel:
    """Market-neutral alpha extraction via factor residual decomposition.

    Decomposes individual security returns into factor exposures (betas)
    and the residual (alpha). The residual is the idiosyncratic return
    not explained by systematic factors — this is the stat-arb alpha.

    Factors
    -------
    MARKET   : broad market return (SPY proxy)
    SIZE     : small minus big (IWM - SPY proxy)
    VALUE    : value minus growth (VLUE - VUG proxy)
    MOMENTUM : winners minus losers (MTUM proxy)
    QUALITY  : quality minus junk (QUAL proxy)

    Usage: decompose returns to find securities with significant positive
    residual alpha, then trade those in a market-neutral portfolio.
    """

    FACTORS = FACTOR_NAMES

    def __init__(self):
        self._factor_betas: dict[str, np.ndarray] = {}
        self._residuals: dict[str, np.ndarray] = {}
        self._r_squared: dict[str, float] = {}
        self._alpha_scores: dict[str, float] = {}

    def decompose(self, returns: np.ndarray, factor_returns: np.ndarray) -> dict:
        """Decompose asset returns into factor loadings and residual alpha.

        Model: R_i = alpha_i + sum(beta_k * F_k) + epsilon_i

        Parameters
        ----------
        returns : np.ndarray
            Asset return series (T,) or (T, N) for multiple assets.
        factor_returns : np.ndarray
            Factor return matrix (T, K) where K = number of factors.

        Returns
        -------
        dict
            Keys: 'betas' (factor loadings), 'alpha' (intercept),
                  'residual' (epsilon), 'r_squared', 'factor_names'
        """
        r = np.asarray(returns, dtype=np.float64)
        f = np.asarray(factor_returns, dtype=np.float64)

        # Handle 1-D returns (single asset)
        if r.ndim == 1:
            return self._decompose_single(r, f)

        # Multi-asset decomposition
        results = []
        n_assets = r.shape[1] if r.ndim == 2 else 1
        for i in range(n_assets):
            asset_returns = r[:, i]
            result = self._decompose_single(asset_returns, f)
            results.append(result)

        return {
            "assets": results,
            "n_assets": n_assets,
            "n_factors": f.shape[1] if f.ndim == 2 else 1,
            "factor_names": list(self.FACTORS[:f.shape[1] if f.ndim == 2 else 1]),
        }

    def _decompose_single(self, returns: np.ndarray, factor_returns: np.ndarray) -> dict:
        """Decompose a single asset's returns against factor returns.

        Parameters
        ----------
        returns : np.ndarray
            Single asset return series (T,).
        factor_returns : np.ndarray
            Factor return matrix (T, K).

        Returns
        -------
        dict
            Factor decomposition results.
        """
        r = np.asarray(returns, dtype=np.float64)
        f = np.asarray(factor_returns, dtype=np.float64)

        if f.ndim == 1:
            f = f.reshape(-1, 1)

        n = min(len(r), len(f))
        if n < 10:
            k = f.shape[1]
            return {
                "betas": np.zeros(k),
                "alpha": 0.0,
                "residual": np.array([]),
                "r_squared": 0.0,
                "factor_names": list(self.FACTORS[:k]),
            }

        r = r[-n:]
        f = f[-n:]

        k = f.shape[1]

        # OLS with intercept: R = alpha + beta * F + epsilon
        X = np.column_stack([f, np.ones(n)])

        try:
            result = np.linalg.lstsq(X, r, rcond=None)
            coeffs = result[0]
        except np.linalg.LinAlgError:
            return {
                "betas": np.zeros(k),
                "alpha": 0.0,
                "residual": np.zeros(n),
                "r_squared": 0.0,
                "factor_names": list(self.FACTORS[:k]),
            }

        betas = coeffs[:k]
        alpha = float(coeffs[k])       # intercept = alpha
        fitted = X @ coeffs
        residual = r - fitted

        # R-squared
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((r - np.mean(r)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        return {
            "betas": betas,
            "alpha": alpha,
            "residual": residual,
            "r_squared": r_squared,
            "factor_names": list(self.FACTORS[:k]),
        }

    def get_residual_signals(self, universe_returns: np.ndarray,
                             factor_returns: np.ndarray,
                             ticker_names: Optional[list] = None) -> list:
        """Rank securities by residual alpha for market-neutral trading.

        Process:
            1. Decompose each security's returns
            2. Extract annualised residual alpha
            3. Rank by alpha magnitude
            4. Return top/bottom quintiles as long/short candidates

        Parameters
        ----------
        universe_returns : np.ndarray
            Returns matrix (T, N) for N securities.
        factor_returns : np.ndarray
            Factor returns matrix (T, K).
        ticker_names : list, optional
            Ticker symbols corresponding to columns in universe_returns.

        Returns
        -------
        list[dict]
            Ranked list of residual alpha signals.
        """
        u = np.asarray(universe_returns, dtype=np.float64)
        f = np.asarray(factor_returns, dtype=np.float64)

        if u.ndim == 1:
            u = u.reshape(-1, 1)

        n_assets = u.shape[1]
        if ticker_names is None:
            ticker_names = [f"ASSET_{i}" for i in range(n_assets)]

        signals = []
        for i in range(n_assets):
            result = self._decompose_single(u[:, i], f)
            annualised_alpha = result["alpha"] * 252  # daily to annual

            # Residual volatility (annualised)
            if len(result["residual"]) > 1:
                residual_vol = float(np.std(result["residual"], ddof=1)) * np.sqrt(252)
            else:
                residual_vol = 0.0

            # Information ratio = alpha / residual_vol
            ir = annualised_alpha / residual_vol if residual_vol > 1e-12 else 0.0

            # Market beta (first factor loading)
            market_beta = float(result["betas"][0]) if len(result["betas"]) > 0 else 0.0

            signals.append({
                "ticker": ticker_names[i],
                "alpha_annual": round(annualised_alpha, 6),
                "residual_vol": round(residual_vol, 6),
                "information_ratio": round(ir, 4),
                "market_beta": round(market_beta, 4),
                "r_squared": round(result["r_squared"], 4),
                "factor_betas": {
                    name: round(float(b), 4)
                    for name, b in zip(result["factor_names"], result["betas"])
                },
                "direction": "LONG" if annualised_alpha > 0 else "SHORT",
            })

            # Store for later retrieval
            self._alpha_scores[ticker_names[i]] = annualised_alpha
            self._r_squared[ticker_names[i]] = result["r_squared"]

        # Sort by absolute alpha (strongest signals first)
        signals.sort(key=lambda x: abs(x["alpha_annual"]), reverse=True)

        return signals

    def get_long_short_portfolio(self, signals: list, top_n: int = 5) -> dict:
        """Construct a long-short portfolio from residual alpha signals.

        Takes the top N positive-alpha securities as longs and
        top N negative-alpha securities as shorts.

        Parameters
        ----------
        signals : list[dict]
            Output from get_residual_signals().
        top_n : int
            Number of securities per side.

        Returns
        -------
        dict
            Long-short portfolio specification.
        """
        longs = [s for s in signals if s["alpha_annual"] > 0]
        shorts = [s for s in signals if s["alpha_annual"] < 0]

        # Sort longs by alpha (highest first), shorts by alpha (most negative first)
        longs.sort(key=lambda x: x["alpha_annual"], reverse=True)
        shorts.sort(key=lambda x: x["alpha_annual"])

        selected_longs = longs[:top_n]
        selected_shorts = shorts[:top_n]

        # Equal-weight within each side
        n_long = len(selected_longs)
        n_short = len(selected_shorts)

        long_weight = 0.5 / n_long if n_long > 0 else 0.0
        short_weight = -0.5 / n_short if n_short > 0 else 0.0

        portfolio = []
        total_beta = 0.0

        for s in selected_longs:
            weight = long_weight
            portfolio.append({
                "ticker": s["ticker"],
                "weight": round(weight, 4),
                "side": "LONG",
                "alpha": s["alpha_annual"],
                "beta": s["market_beta"],
            })
            total_beta += weight * s["market_beta"]

        for s in selected_shorts:
            weight = short_weight
            portfolio.append({
                "ticker": s["ticker"],
                "weight": round(weight, 4),
                "side": "SHORT",
                "alpha": s["alpha_annual"],
                "beta": s["market_beta"],
            })
            total_beta += weight * s["market_beta"]

        return {
            "positions": portfolio,
            "n_long": n_long,
            "n_short": n_short,
            "portfolio_beta": round(total_beta, 4),
            "is_beta_neutral": abs(total_beta) < 0.10,
        }


# ═══════════════════════════════════════════════════════════════════════════
# StatArbEngine — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════
class StatArbEngine:
    """Medallion-style Statistical Arbitrage Engine.

    Orchestrates the complete stat-arb pipeline:
        1. Scan universe for cointegrated pairs
        2. Compute pair statistics (hedge ratio, spread, z-score, half-life)
        3. Generate entry/exit signals with position sizes
        4. Monitor portfolio beta neutrality (target: sum(beta) ~ 0)
        5. Track active trades and P&L

    The engine pre-loads 26 relative-value pairs from the universe engine
    covering all major GICS sectors.

    Usage
    -----
    >>> engine = StatArbEngine()
    >>> stats = engine.scan_pairs(price_data)
    >>> signals = engine.get_trading_signals()
    >>> beta = engine.compute_portfolio_beta()
    >>> print(engine.format_stat_arb_report())
    """

    def __init__(self):
        # Initialize pairs — try dynamic discovery, fall back to static RV_PAIRS
        self._pairs: list[CointegratedPair] = []
        self._pair_stats: list[PairStatistics] = []
        self._active_trades: list[dict] = []
        self._trade_history: list[dict] = []
        self._factor_model = FactorResidualModel()
        self._portfolio_beta: float = 0.0
        self._last_scan: str = ""
        self._initialized: bool = False

        # Seed with static pairs, then extend dynamically
        for ticker_a, ticker_b in RV_PAIRS:
            self._pairs.append(CointegratedPair(ticker_a, ticker_b))

        logger.info(f"StatArbEngine initialized with {len(self._pairs)} seed pairs")

    def discover_pairs(self, min_correlation: float = 0.70, max_pairs: int = 50) -> int:
        """Dynamically discover cointegrated pairs from the live universe.

        Scans same-sector tickers for high correlation and adds new pairs
        to the engine. Uses OpenBB/FMP price data for correlation analysis.

        Returns number of new pairs discovered.
        """
        try:
            from ..data.universe_engine import get_engine
            from ..data.openbb_data import get_adj_close
            import pandas as pd
        except ImportError:
            logger.debug("Cannot discover pairs — imports unavailable")
            return 0

        ue = get_engine()
        sectors = ue.get_sectors() if hasattr(ue, "get_sectors") else {}
        existing_keys = {(p.ticker_a, p.ticker_b) for p in self._pairs}
        existing_keys.update({(p.ticker_b, p.ticker_a) for p in self._pairs})
        added = 0

        # Scan within each sector for correlated pairs
        for sector_name, tickers in sectors.items():
            if len(tickers) < 2:
                continue
            # Take top 15 by liquidity/market cap per sector
            candidates = tickers[:15] if isinstance(tickers, list) else list(tickers)[:15]
            if len(candidates) < 2:
                continue

            try:
                start = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
                prices = get_adj_close(candidates, start=start)
                if prices.empty or len(prices) < 60:
                    continue
                returns = prices.pct_change().dropna()
                if returns.empty:
                    continue

                # Compute pairwise correlation
                corr = returns.corr()
                for i, t1 in enumerate(corr.columns):
                    for j, t2 in enumerate(corr.columns):
                        if j <= i:
                            continue
                        c = corr.iloc[i, j]
                        if abs(c) >= min_correlation and (t1, t2) not in existing_keys:
                            self._pairs.append(CointegratedPair(t1, t2))
                            existing_keys.add((t1, t2))
                            existing_keys.add((t2, t1))
                            added += 1
                            if added >= max_pairs:
                                break
                    if added >= max_pairs:
                        break
            except Exception as e:
                logger.debug("Pair discovery failed for sector %s: %s", sector_name, e)

            if added >= max_pairs:
                break

        if added > 0:
            logger.info("Discovered %d new pairs (total: %d)", added, len(self._pairs))
        return added

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    @property
    def n_active(self) -> int:
        return len(self._active_trades)

    def scan_pairs(self, price_data: dict, lookback: int = 60) -> list:
        """Scan all pairs and compute full statistics.

        Parameters
        ----------
        price_data : dict
            Mapping of ticker -> np.ndarray of prices.
            Must contain prices for both legs of each pair.
        lookback : int
            Lookback window for z-score computation.

        Returns
        -------
        list[PairStatistics]
            Statistics for all pairs with sufficient data.
        """
        self._pair_stats = []
        scanned = 0
        skipped = 0

        for pair in self._pairs:
            prices_a = price_data.get(pair.ticker_a)
            prices_b = price_data.get(pair.ticker_b)

            if prices_a is None or prices_b is None:
                skipped += 1
                logger.debug(f"Skipping {pair.pair_id}: missing price data")
                continue

            pa = np.asarray(prices_a, dtype=np.float64)
            pb = np.asarray(prices_b, dtype=np.float64)

            if len(pa) < MIN_LOOKBACK or len(pb) < MIN_LOOKBACK:
                skipped += 1
                logger.debug(f"Skipping {pair.pair_id}: insufficient data "
                             f"({len(pa)}/{len(pb)} < {MIN_LOOKBACK})")
                continue

            stats = pair.full_analysis(pa, pb, lookback)
            self._pair_stats.append(stats)
            scanned += 1

        self._last_scan = datetime.now().isoformat()
        self._initialized = True

        logger.info(f"Scanned {scanned} pairs, skipped {skipped}")
        return list(self._pair_stats)

    def get_trading_signals(self) -> list:
        """Generate entry/exit signals with position sizes for all active pairs.

        Only emits signals for pairs that are:
            - Cointegrated (p < 0.05)
            - Have valid half-life (2 < HL < 120 days)
            - Have z-score beyond entry/exit thresholds

        Returns
        -------
        list[dict]
            Trading signals with direction, sizing, and metadata.
        """
        signals = []
        new_active = []

        for pair, stats in zip(self._pairs, self._pair_stats):
            signal = pair.get_signal()

            if signal["direction"] == SignalDirection.NEUTRAL.value:
                continue

            if not stats.is_tradeable and signal["direction"] not in (
                SignalDirection.EXIT.value, SignalDirection.STOP.value
            ):
                continue

            # Kelly-inspired position sizing
            # size = edge / odds, capped at signal_strength
            edge = abs(signal["zscore"]) - abs(ZSCORE_ENTRY_LONG)
            kelly_fraction = min(0.25, max(0.0, edge / ZSCORE_STOP))
            signal["kelly_size"] = round(kelly_fraction, 4)

            # Sector classification
            signal["sector"] = self._classify_pair_sector(pair.ticker_a, pair.ticker_b)

            signals.append(signal)

            # Track active trades
            if signal["direction"] in (
                SignalDirection.LONG_A_SHORT_B.value,
                SignalDirection.SHORT_A_LONG_B.value,
            ):
                trade = {
                    "pair": pair.pair_id,
                    "ticker_a": pair.ticker_a,
                    "ticker_b": pair.ticker_b,
                    "direction": signal["direction"],
                    "entry_zscore": signal["zscore"],
                    "hedge_ratio": signal["hedge_ratio"],
                    "position_size": signal["position_size"],
                    "kelly_size": signal["kelly_size"],
                    "entry_time": datetime.now().isoformat(),
                    "status": "OPEN",
                    "pnl": 0.0,
                }
                new_active.append(trade)

        # Update active trades: close exits/stops, add new entries
        closed = []
        for trade in self._active_trades:
            pair_id = trade["pair"]
            # Check if there is an exit/stop signal for this pair
            exit_signal = next(
                (s for s in signals
                 if s["pair"] == pair_id
                 and s["direction"] in (SignalDirection.EXIT.value, SignalDirection.STOP.value)),
                None,
            )
            if exit_signal:
                trade["status"] = "CLOSED"
                trade["exit_time"] = datetime.now().isoformat()
                trade["exit_zscore"] = exit_signal["zscore"]
                trade["exit_reason"] = exit_signal["direction"]
                closed.append(trade)
                self._trade_history.append(trade)
            else:
                new_active.append(trade)

        self._active_trades = [t for t in new_active if t["status"] == "OPEN"]

        return signals

    def compute_portfolio_beta(self) -> float:
        """Compute net portfolio beta across all active stat-arb positions.

        For a market-neutral portfolio, the target is sum(beta) ~ 0.
        Each pair contributes: beta_pair = weight_a * beta_a + weight_b * beta_b
        where weight_b is negative (short leg).

        Returns
        -------
        float
            Net portfolio beta. Should be close to 0 for market-neutral.
        """
        if not self._active_trades:
            self._portfolio_beta = 0.0
            return 0.0

        total_beta = 0.0
        for trade in self._active_trades:
            # Pairs are approximately market-neutral by construction
            # Net beta per pair ~ hedge_ratio deviation from perfect hedge
            hr = trade.get("hedge_ratio", 1.0)
            size = trade.get("position_size", 0.0)

            # Assume beta_a ~ 1.0, beta_b ~ 1.0 (large-cap equity pairs)
            # Long-short pair beta ~ size * (1 - hr) for LONG_A_SHORT_B
            if trade["direction"] == SignalDirection.LONG_A_SHORT_B.value:
                pair_beta = size * (1.0 - hr)
            elif trade["direction"] == SignalDirection.SHORT_A_LONG_B.value:
                pair_beta = size * (hr - 1.0)
            else:
                pair_beta = 0.0

            total_beta += pair_beta

        self._portfolio_beta = round(total_beta, 4)
        return self._portfolio_beta

    def get_active_trades(self) -> list:
        """Return all currently active (open) stat-arb trades.

        Returns
        -------
        list[dict]
            Active trades with entry info, direction, sizing.
        """
        return list(self._active_trades)

    def get_trade_history(self) -> list:
        """Return closed trade history.

        Returns
        -------
        list[dict]
            Closed trades with entry/exit info and P&L.
        """
        return list(self._trade_history)

    def get_cointegrated_pairs(self) -> list:
        """Return pairs that pass the cointegration test.

        Returns
        -------
        list[PairStatistics]
            Only pairs with p-value < 0.05 and valid half-life.
        """
        return [s for s in self._pair_stats if s.is_tradeable]

    def get_pair_statistics(self) -> list:
        """Return all pair statistics from the last scan.

        Returns
        -------
        list[PairStatistics]
            Complete statistics for all scanned pairs.
        """
        return list(self._pair_stats)

    def _classify_pair_sector(self, ticker_a: str, ticker_b: str) -> str:
        """Classify a pair into its sector grouping.

        Uses the DEFAULT_PAIRS sector mapping for quick lookup.

        Parameters
        ----------
        ticker_a, ticker_b : str
            Pair tickers.

        Returns
        -------
        str
            Sector name or 'unknown'.
        """
        pair_tuple = (ticker_a, ticker_b)
        pair_tuple_rev = (ticker_b, ticker_a)

        for sector, pairs in DEFAULT_PAIRS.items():
            if pair_tuple in pairs or pair_tuple_rev in pairs:
                return sector
        return "unknown"

    def format_stat_arb_report(self) -> str:
        """Generate an ASCII dashboard of the stat-arb engine state.

        Sections:
            1. Header with summary stats
            2. Pair scanner results
            3. Active signals
            4. Portfolio beta
            5. Trade history summary

        Returns
        -------
        str
            Multi-line ASCII report.
        """
        lines = []
        w = 78  # report width

        # ── Header ──────────────────────────────────────────────────────
        lines.append("=" * w)
        lines.append("  METADRON CAPITAL — STATISTICAL ARBITRAGE ENGINE")
        lines.append("  Medallion-Style Mean-Reversion + Cointegration")
        lines.append("=" * w)
        lines.append(f"  Scan Time    : {self._last_scan or 'NOT RUN'}")
        lines.append(f"  Total Pairs  : {self.n_pairs}")
        lines.append(f"  Scanned      : {len(self._pair_stats)}")

        n_coint = len([s for s in self._pair_stats if s.is_cointegrated])
        n_trade = len([s for s in self._pair_stats if s.is_tradeable])
        n_active = len(self._active_trades)

        lines.append(f"  Cointegrated : {n_coint}")
        lines.append(f"  Tradeable    : {n_trade}")
        lines.append(f"  Active Trades: {n_active}")
        lines.append(f"  Portfolio B  : {self._portfolio_beta:+.4f}")
        lines.append("")

        # ── Pair Scanner ────────────────────────────────────────────────
        lines.append("-" * w)
        lines.append("  PAIR SCANNER RESULTS")
        lines.append("-" * w)
        header = (
            f"  {'Pair':<16} {'Corr':>6} {'Coint-p':>9} {'HL':>6} "
            f"{'Z-Score':>8} {'HR':>6} {'Str':>5} {'Status':<12}"
        )
        lines.append(header)
        lines.append("  " + "-" * (w - 4))

        for stats in sorted(self._pair_stats, key=lambda s: abs(s.spread_zscore), reverse=True):
            pair_name = f"{stats.ticker_a}/{stats.ticker_b}"
            coint_marker = "*" if stats.is_cointegrated else " "

            # Z-score visual indicator
            z = stats.spread_zscore
            if abs(z) > ZSCORE_STOP:
                z_indicator = "!!!"
            elif abs(z) > ZSCORE_ENTRY_SHORT:
                z_indicator = " >>"
            elif abs(z) < ZSCORE_EXIT:
                z_indicator = " =="
            else:
                z_indicator = "   "

            line = (
                f"  {pair_name:<16} {stats.correlation:>6.3f} "
                f"{stats.cointegration_pvalue:>8.4f}{coint_marker} "
                f"{stats.half_life:>5.1f}d {z:>+7.3f}{z_indicator} "
                f"{stats.hedge_ratio:>6.3f} {stats.signal_strength:>5.3f} "
                f"{stats.status:<12}"
            )
            lines.append(line)

        lines.append("")

        # ── Active Signals ──────────────────────────────────────────────
        active_signals = [
            s for s in self._pair_stats
            if s.direction in (SignalDirection.LONG_A_SHORT_B, SignalDirection.SHORT_A_LONG_B)
            and s.is_tradeable
        ]

        lines.append("-" * w)
        lines.append("  ACTIVE SIGNALS")
        lines.append("-" * w)

        if active_signals:
            for stats in active_signals:
                pair_name = f"{stats.ticker_a}/{stats.ticker_b}"
                direction = stats.direction.value
                lines.append(
                    f"  {pair_name:<16} {direction:<22} "
                    f"z={stats.spread_zscore:>+7.3f}  "
                    f"str={stats.signal_strength:.3f}  "
                    f"HL={stats.half_life:.0f}d"
                )
        else:
            lines.append("  No active entry signals at this time.")

        lines.append("")

        # ── Active Trades ───────────────────────────────────────────────
        lines.append("-" * w)
        lines.append("  ACTIVE TRADES")
        lines.append("-" * w)

        if self._active_trades:
            for trade in self._active_trades:
                lines.append(
                    f"  {trade['pair']:<16} {trade['direction']:<22} "
                    f"entry_z={trade.get('entry_zscore', 0):>+7.3f}  "
                    f"size={trade.get('position_size', 0):.3f}  "
                    f"kelly={trade.get('kelly_size', 0):.3f}"
                )
        else:
            lines.append("  No active trades.")

        lines.append("")

        # ── Portfolio Beta ──────────────────────────────────────────────
        lines.append("-" * w)
        lines.append("  PORTFOLIO BETA NEUTRALITY")
        lines.append("-" * w)

        beta = self._portfolio_beta
        beta_bar_width = 40
        # Visualise beta on a scale of -0.5 to +0.5
        beta_clamped = max(-0.5, min(0.5, beta))
        centre = beta_bar_width // 2
        pos = int(centre + beta_clamped * beta_bar_width)
        pos = max(0, min(beta_bar_width - 1, pos))

        bar = ["-"] * beta_bar_width
        bar[centre] = "|"
        bar[pos] = "X"

        beta_status = "NEUTRAL" if abs(beta) < 0.05 else (
            "LONG-BIASED" if beta > 0 else "SHORT-BIASED"
        )

        lines.append(f"  Net Beta: {beta:>+.4f}  ({beta_status})")
        lines.append(f"  [{' '.join(bar)}]")
        lines.append(f"  {'Short':>6}{' ' * (beta_bar_width * 2 - 14)}{'Long':>6}")
        lines.append("")

        # ── Trade History Summary ───────────────────────────────────────
        lines.append("-" * w)
        lines.append("  TRADE HISTORY")
        lines.append("-" * w)

        n_closed = len(self._trade_history)
        if n_closed > 0:
            exits = [t for t in self._trade_history if t.get("exit_reason") == SignalDirection.EXIT.value]
            stops = [t for t in self._trade_history if t.get("exit_reason") == SignalDirection.STOP.value]
            lines.append(f"  Total Closed : {n_closed}")
            lines.append(f"  Mean-Reverted: {len(exits)}")
            lines.append(f"  Stopped Out  : {len(stops)}")
        else:
            lines.append("  No closed trades yet.")

        lines.append("")

        # ── Cointegration Summary ───────────────────────────────────────
        lines.append("-" * w)
        lines.append("  COINTEGRATION SUMMARY BY SECTOR")
        lines.append("-" * w)

        sector_stats: dict[str, list] = {}
        for stats in self._pair_stats:
            sector = self._classify_pair_sector(stats.ticker_a, stats.ticker_b)
            if sector not in sector_stats:
                sector_stats[sector] = []
            sector_stats[sector].append(stats)

        for sector in sorted(sector_stats.keys()):
            pairs_in_sector = sector_stats[sector]
            n_total = len(pairs_in_sector)
            n_coint_sec = len([s for s in pairs_in_sector if s.is_cointegrated])
            avg_corr = np.mean([s.correlation for s in pairs_in_sector]) if pairs_in_sector else 0
            lines.append(
                f"  {sector:<14} {n_coint_sec}/{n_total} cointegrated  "
                f"avg_corr={avg_corr:>+.3f}"
            )

        lines.append("")
        lines.append("=" * w)
        lines.append(f"  Target: sum(beta) ~ 0  |  Current: {self._portfolio_beta:+.4f}")
        lines.append("=" * w)

        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a machine-readable summary of the engine state.

        Returns
        -------
        dict
            Summary statistics for programmatic consumption.
        """
        n_coint = len([s for s in self._pair_stats if s.is_cointegrated])
        n_tradeable = len([s for s in self._pair_stats if s.is_tradeable])
        n_entry = len([
            s for s in self._pair_stats
            if s.direction in (SignalDirection.LONG_A_SHORT_B, SignalDirection.SHORT_A_LONG_B)
            and s.is_tradeable
        ])

        return {
            "engine": "StatArbEngine",
            "version": "1.0.0",
            "total_pairs": self.n_pairs,
            "scanned_pairs": len(self._pair_stats),
            "cointegrated_pairs": n_coint,
            "tradeable_pairs": n_tradeable,
            "entry_signals": n_entry,
            "active_trades": len(self._active_trades),
            "closed_trades": len(self._trade_history),
            "portfolio_beta": self._portfolio_beta,
            "is_beta_neutral": abs(self._portfolio_beta) < 0.05,
            "last_scan": self._last_scan,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════
def generate_synthetic_prices(n_days: int = 252, n_pairs: int = 5,
                              seed: int = 42) -> dict:
    """Generate synthetic cointegrated price pairs for testing.

    Creates pairs where:
        P_a(t) = drift + beta * P_b(t) + OU_process(t)

    The OU process ensures the spread is mean-reverting.

    Parameters
    ----------
    n_days : int
        Number of trading days.
    n_pairs : int
        Number of pairs to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping of ticker -> price array.
    """
    rng = np.random.default_rng(seed)
    price_data = {}

    pair_tickers = RV_PAIRS[:n_pairs]

    for ticker_a, ticker_b in pair_tickers:
        # Base price for leg B
        base_price = 100.0 + rng.uniform(-20, 50)
        returns_b = rng.normal(0.0003, 0.015, n_days)
        prices_b = base_price * np.cumprod(1 + returns_b)

        # Cointegrated leg A: beta * B + OU noise
        beta = rng.uniform(0.7, 1.3)
        alpha = rng.uniform(-10, 10)
        theta = rng.uniform(0.02, 0.15)  # mean-reversion speed
        sigma_ou = rng.uniform(1.0, 5.0)

        # Simulate OU process
        ou = np.zeros(n_days)
        for t in range(1, n_days):
            ou[t] = ou[t - 1] + theta * (0 - ou[t - 1]) + sigma_ou * rng.normal()

        prices_a = alpha + beta * prices_b + ou

        # Ensure positive prices
        prices_a = np.maximum(prices_a, 1.0)
        prices_b = np.maximum(prices_b, 1.0)

        price_data[ticker_a] = prices_a
        price_data[ticker_b] = prices_b

    return price_data


def generate_synthetic_factor_returns(n_days: int = 252, seed: int = 42) -> np.ndarray:
    """Generate synthetic factor returns for testing the factor model.

    Creates 5 factors: MARKET, SIZE, VALUE, MOMENTUM, QUALITY
    with realistic correlations and volatilities.

    Parameters
    ----------
    n_days : int
        Number of trading days.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Factor returns matrix (n_days, 5).
    """
    rng = np.random.default_rng(seed)

    # Factor parameters: (annual_mean, annual_vol)
    factor_params = [
        (0.08, 0.16),   # MARKET
        (0.02, 0.10),   # SIZE
        (0.03, 0.12),   # VALUE
        (0.04, 0.14),   # MOMENTUM
        (0.02, 0.08),   # QUALITY
    ]

    n_factors = len(factor_params)
    factor_returns = np.zeros((n_days, n_factors))

    for i, (ann_mean, ann_vol) in enumerate(factor_params):
        daily_mean = ann_mean / 252
        daily_vol = ann_vol / np.sqrt(252)
        factor_returns[:, i] = rng.normal(daily_mean, daily_vol, n_days)

    # Add some correlation between factors (market factor drives others partially)
    market = factor_returns[:, 0]
    for i in range(1, n_factors):
        corr_with_market = rng.uniform(0.1, 0.4)
        factor_returns[:, i] = (
            corr_with_market * market
            + np.sqrt(1 - corr_with_market ** 2) * factor_returns[:, i]
        )

    return factor_returns


# ═══════════════════════════════════════════════════════════════════════════
# Module self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 78)
    print("  STAT ARB ENGINE — SELF TEST")
    print("=" * 78)

    # 1. Generate synthetic data
    print("\n[1] Generating synthetic cointegrated pairs...")
    price_data = generate_synthetic_prices(n_days=300, n_pairs=10, seed=123)
    print(f"    Generated {len(price_data)} price series")

    # 2. Initialize engine
    print("\n[2] Initializing StatArbEngine...")
    engine = StatArbEngine()
    print(f"    Loaded {engine.n_pairs} pairs from universe")

    # 3. Scan pairs
    print("\n[3] Scanning pairs...")
    stats = engine.scan_pairs(price_data, lookback=60)
    print(f"    Scanned {len(stats)} pairs")

    for s in stats:
        print(f"    {s.ticker_a}/{s.ticker_b}: "
              f"corr={s.correlation:.3f} p={s.cointegration_pvalue:.4f} "
              f"HL={s.half_life:.1f}d z={s.spread_zscore:+.3f} "
              f"[{s.status}]")

    # 4. Get trading signals
    print("\n[4] Generating signals...")
    signals = engine.get_trading_signals()
    for sig in signals:
        print(f"    {sig['pair']}: {sig['direction']} "
              f"z={sig['zscore']:+.3f} size={sig['position_size']:.3f}")

    if not signals:
        print("    No signals generated (all pairs within neutral zone)")

    # 5. Portfolio beta
    beta = engine.compute_portfolio_beta()
    print(f"\n[5] Portfolio beta: {beta:+.4f}")

    # 6. Factor residual model
    print("\n[6] Factor Residual Model...")
    factor_returns = generate_synthetic_factor_returns(n_days=300, seed=456)

    # Create universe returns from price data
    tickers = list(price_data.keys())
    n_days_min = min(len(v) for v in price_data.values())
    universe_prices = np.column_stack([
        price_data[t][-n_days_min:] for t in tickers
    ])
    universe_returns = np.diff(universe_prices, axis=0) / universe_prices[:-1]

    # Align factor returns
    factor_returns_aligned = factor_returns[-len(universe_returns):]

    fm = FactorResidualModel()
    residual_signals = fm.get_residual_signals(
        universe_returns, factor_returns_aligned, tickers
    )

    print(f"    Top alpha signals:")
    for sig in residual_signals[:5]:
        print(f"    {sig['ticker']:<8} alpha={sig['alpha_annual']:>+.4f}  "
              f"IR={sig['information_ratio']:>+.3f}  "
              f"beta={sig['market_beta']:>+.3f}  "
              f"R2={sig['r_squared']:.3f}")

    # 7. Long-short portfolio
    ls_portfolio = fm.get_long_short_portfolio(residual_signals, top_n=3)
    print(f"\n[7] Long-Short Portfolio (beta={ls_portfolio['portfolio_beta']:+.4f}):")
    for pos in ls_portfolio["positions"]:
        print(f"    {pos['side']:<6} {pos['ticker']:<8} "
              f"weight={pos['weight']:>+.4f}  beta={pos['beta']:>+.3f}")

    # 8. Full report
    print("\n[8] Full Report:")
    print(engine.format_stat_arb_report())

    # 9. Summary
    print("\n[9] Engine Summary:")
    summary = engine.summary()
    for k, v in summary.items():
        print(f"    {k:<22}: {v}")

    print("\n" + "=" * 78)
    print("  SELF TEST COMPLETE")
    print("=" * 78)
