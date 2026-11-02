"""MacroEngine — Regime classification & GMTF money velocity analysis.

Incorporates Dataset 3: Global Monetary Tension Framework (GMTF).
Regimes: BULL / BEAR / TRANSITION / STRESS
Money velocity: Fisher equation V = GDP/M2, sigmoid triggers, carry-to-volatility.
Ranks sector universe by macro regime.
Data: OpenBB (market proxies + FRED/FRB direct series).

Extended modules:
    MoneyVelocityModule   — Fisher V=GDP/M2, credit impulse, TED/SOFR, Liquidity Score
    SectorRanker          — Enhanced sector ranking with macro-adjusted momentum
    CarryToVolatility     — Full CtV signal generation for forex pairs
    RegimeTransitionDetector — Detect regime transitions with confidence scoring
    YieldCurveAnalyzer    — Full yield curve analysis (2s10s, 3m10y, term premium)
    CreditPulseMonitor    — Credit spread monitoring, HY/IG differential
    MacroFeatureBuilder   — Build 50+ macro features for ML models

GMTF Enhancement modules:
    MonetaryTensionIndex  — SDR-weighted monetary tension across G5 currencies
    SectorRotationEngine  — GICS sector rotation based on macro regime cycle
    MoneyVelocityEngine   — V=GDP/M2 proxy, velocity regime, inflation signal
    FedReserveIntegration — Fed balance sheet, net liquidity, liquidity impulse
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List

from ..data.openbb_data import get_adj_close, get_macro_data, get_returns

logger = logging.getLogger("metadron.signals.macro_engine")

# ---------------------------------------------------------------------------
# Intelligence Platform: ML-Macro-Market integration
# Provides: HiddenMarkovRegimeModel, FactorModel, NowcastingEngine,
# MacroMLEngine (regime detection, factor exposure, nowcasting, trade ideas).
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    _macro_ml_spec = _ilu.spec_from_file_location(
        "macro_ml_engine",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
            / "intelligence_platform" / "ML-Macro-Market" / "macro_ml_engine.py"),
    )
    _macro_ml_mod = _ilu.module_from_spec(_macro_ml_spec)
    _macro_ml_spec.loader.exec_module(_macro_ml_mod)
    HiddenMarkovRegimeModel = _macro_ml_mod.HiddenMarkovRegimeModel
    MacroMLFactorModel = _macro_ml_mod.FactorModel
    NowcastingEngine = _macro_ml_mod.NowcastingEngine
    MacroMLEngine = _macro_ml_mod.MacroMLEngine
    MACRO_ML_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    HiddenMarkovRegimeModel = None
    MacroMLFactorModel = None
    NowcastingEngine = None
    MacroMLEngine = None
    MACRO_ML_AVAILABLE = False
    __import__("logging").getLogger(__name__).info(
        "ML-Macro-Market integration unavailable — production macro engine only"
    )


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------
class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    TRANSITION = "TRANSITION"
    STRESS = "STRESS"
    CRASH = "CRASH"


class CubeRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGE = "RANGE"
    STRESS = "STRESS"
    CRASH = "CRASH"


# ---------------------------------------------------------------------------
# SDR basket for GMTF
# ---------------------------------------------------------------------------
SDR_WEIGHTS = pd.Series({
    "USD": 0.4338, "EUR": 0.2931, "CNY": 0.1228,
    "JPY": 0.0759, "GBP": 0.0744,
})

# Sigmoid sensitivity for non-linear triggers
SIGMOID_SENSITIVITY = 15.0
ROLLING_WINDOW = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MacroSnapshot:
    """Current macro state."""
    regime: MarketRegime = MarketRegime.TRANSITION
    vix: float = 20.0
    spy_return_1m: float = 0.0
    spy_return_3m: float = 0.0
    yield_10y: float = 4.0
    yield_2y: float = 4.5
    yield_spread: float = -0.5
    credit_spread: float = 3.0
    gold_momentum: float = 0.0
    sector_rankings: dict = field(default_factory=dict)
    gmtf_score: float = 0.0
    money_velocity_signal: float = 0.0
    cube_regime: CubeRegime = CubeRegime.RANGE
    # GMTF provenance: "gmtf" = real 4-gamma tension percentile (0..1);
    # "liquidity_proxy" = fallback liquidity score; "none" = not computed
    gmtf_source: str = "none"
    gmtf_raw: float = 0.0
    gmtf_gammas: dict = field(default_factory=dict)
    gmtf_currencies: list = field(default_factory=list)
    # Gaussian-HMM regime (MarkovRegimeBridge), fitted on SPY/VIX/credit
    hmm_regime: Optional[str] = None
    hmm_confidence: float = 0.0
    hmm_probs: dict = field(default_factory=dict)


@dataclass
class GMTFOutput:
    """Global Monetary Tension Framework output."""
    gmtf_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gmtf_smoothed: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gamma_liquidity: float = 1.0
    gamma_fx: float = 1.0
    gamma_wage: float = 1.0
    gamma_reserve: float = 1.0
    rv_signals: Optional[pd.DataFrame] = None
    ctv_ratios: Optional[pd.DataFrame] = None
    ctv_gate: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Core GMTF functions (from Dataset 3)
# ---------------------------------------------------------------------------
def sigmoid_trigger(x, threshold: float, sensitivity: float = SIGMOID_SENSITIVITY):
    """Smooth non-linear transition (0 to 1)."""
    return 1 / (1 + np.exp(-sensitivity * (x - threshold)))


def compute_gammas(
    m2: pd.DataFrame,
    gdp_g: pd.DataFrame,
    unemp: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
) -> tuple:
    """Compute non-linear gamma multipliers for GMTF.

    Returns (gamma_liquidity, gamma_fx, gamma_wage, gamma_reserve).
    """
    # Gamma_L: Liquidity overload (M2 growth > 2x GDP growth)
    g_liquidity = 1 + sigmoid_trigger(m2 - 2 * gdp_g, 0.01) * 1.5

    # Gamma_FX: Fed shock (USD yield spike contagion)
    usd_delta = rates["USD"].diff().fillna(0) if "USD" in rates.columns else pd.Series(0, index=rates.index)
    g_fx_factor = (1 - sigmoid_trigger(usd_delta, 0.0025) * 0.5).values
    if g_fx_factor.ndim == 1:
        g_fx_factor = g_fx_factor[:, None]

    # Gamma_W: Wage-price spiral (labor tightness)
    g_wage = 1 + sigmoid_trigger(0.042 - unemp, 0) * 2.0

    # Gamma_S: Reserve decay (USD share threshold at 57%)
    usd_share = fx["USD"] / fx.sum(axis=1) if "USD" in fx.columns else pd.Series(0.6, index=fx.index)
    g_reserve = 1 + sigmoid_trigger(0.57 - usd_share, 0) * 1.5

    return g_liquidity, g_fx_factor, g_wage, g_reserve


def compute_gmtf(
    m2: pd.DataFrame,
    gdp: pd.DataFrame,
    unemp: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
) -> GMTFOutput:
    """Full GMTF calculation pipeline.

    Returns GMTFOutput with aggregated tension series and gamma multipliers.
    """
    currencies = SDR_WEIGHTS.index.tolist()
    dates = m2.index

    # Year-over-year growth on the monthly panel (1-period change if the
    # history is too short). Levels must NOT be fed to the liquidity gamma:
    # it compares M2 growth with 2× GDP growth.
    lag = 12 if len(m2) > 24 else 1
    gdp_growth = gdp.pct_change(lag).fillna(0.0001)
    m2_growth = m2.pct_change(lag).fillna(0.0)

    # Base impact: theta
    base_theta = (m2 / gdp) * (1 + unemp) * gdp_growth

    # Non-linear triggers
    gl, gfx, gw, gs = compute_gammas(m2_growth, gdp_growth, unemp, fx, rates)

    # Apply multipliers
    triggered_theta = base_theta * gl * gw
    # Apply FX gamma (broadcasting)
    if hasattr(gfx, "shape") and gfx.ndim == 2:
        n_cols = min(gfx.shape[1], triggered_theta.shape[1])
        for i in range(n_cols):
            triggered_theta.iloc[:, i] *= gfx[:len(triggered_theta), 0]

    # Apply reserve decay to USD
    if "USD" in triggered_theta.columns:
        triggered_theta["USD"] *= gs

    # Aggregate with SDR weights
    available = [c for c in currencies if c in triggered_theta.columns]
    weights = SDR_WEIGHTS[available]
    weights = weights / weights.sum()
    gmtf_series = triggered_theta[available].dot(weights)
    gmtf_smoothed = gmtf_series.rolling(window=ROLLING_WINDOW).mean()

    return GMTFOutput(
        gmtf_series=gmtf_series,
        gmtf_smoothed=gmtf_smoothed,
        gamma_liquidity=float(gl.mean().mean()) if hasattr(gl, "mean") else 1.0,
        gamma_fx=float(np.mean(gfx)),
        gamma_wage=float(gw.mean().mean()) if hasattr(gw, "mean") else 1.0,
        gamma_reserve=float(gs.mean()) if hasattr(gs, "mean") else 1.0,
    )


# FRED series per SDR currency: (M2/M3 money, real GDP, unemployment %, 3M rate %)
GMTF_FRED_SERIES = {
    "USD": ("M2SL", "GDPC1", "UNRATE", "TB3MS"),
    "EUR": ("MABMM301EZM189S", "CLVMNACSCAB1GQEA19", "LRHUTTTTEZM156S", "IR3TIB01EZM156N"),
    "JPY": ("MABMM301JPM189S", "JPNRGDPEXP", "LRUN64TTJPM156S", "IR3TIB01JPM156N"),
    "GBP": ("MABMM301GBM189S", "NGDPRSAXDCGBQ", "LRHUTTTTGBM156S", "IR3TIB01GBM156N"),
    "CNY": ("MYAGM2CNM189N", "CHNGDPNQDSMEI", "LRUN64TTCNQ156S", "IR3TIB01CNM156N"),
}


def build_gmtf_inputs(start: str = "2005-01-01", fetch=None) -> Optional[dict]:
    """Assemble the monthly GMTF input panel from FRED (via OpenBB).

    Money and GDP are indexed to 1.0 at the start of each currency's history
    (so ratios are comparable across currencies); unemployment and rates are
    converted from percent to decimals. A currency is included only if all
    four series are available. Returns None when USD cannot be built.
    """
    if fetch is None:
        try:
            from ..data.openbb_data import get_fred_series as fetch
        except Exception:
            return None
    cols: dict = {"m2": {}, "gdp": {}, "unemp": {}, "rates": {}}
    for ccy, ids in GMTF_FRED_SERIES.items():
        try:
            df = fetch(list(ids), start=start)
        except Exception as exc:
            logger.debug("GMTF fetch %s failed: %s", ccy, exc)
            continue
        if df is None or getattr(df, "empty", True) or not all(i in df.columns for i in ids):
            continue
        df.index = pd.to_datetime(df.index)
        monthly = df[list(ids)].resample("ME").last().ffill().dropna()
        if len(monthly) < 24:
            continue
        money, gdp, unemp, rate = (monthly[i] for i in ids)
        cols["m2"][ccy] = money / money.iloc[0]
        cols["gdp"][ccy] = gdp / gdp.iloc[0]
        cols["unemp"][ccy] = unemp / 100.0
        cols["rates"][ccy] = rate / 100.0
    if "USD" not in cols["m2"]:
        return None
    frames = {k: pd.DataFrame(v).dropna() for k, v in cols.items()}
    idx = frames["m2"].index
    for k in frames:
        idx = idx.intersection(frames[k].index)
    frames = {k: v.loc[idx] for k, v in frames.items()}
    # No reliable monthly reserve-share series → reserve gamma uses its 57% default
    frames["fx"] = pd.DataFrame(index=idx)
    return frames


# ---------------------------------------------------------------------------
# Carry-to-Volatility (CtV) & stop-loss
# ---------------------------------------------------------------------------
def compute_ctv_signals(
    yield_data: pd.DataFrame,
    fx_data: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute carry-to-volatility ratios and entry gates.

    Returns (ctv_ratios, ctv_gate).
    """
    currencies = ["EUR", "JPY", "GBP", "CNY"]
    ctv_ratios = pd.DataFrame(index=yield_data.index)

    for c in currencies:
        y_col = f"yield3m_{c}" if f"yield3m_{c}" in yield_data.columns else None
        y_usd = "yield3m_USD" if "yield3m_USD" in yield_data.columns else None
        fx_col = f"fx_{c}" if f"fx_{c}" in fx_data.columns else None

        if y_col and y_usd and fx_col:
            carry = (yield_data[y_col] - yield_data[y_usd]) / 100
            vol = fx_data[fx_col].pct_change().rolling(window).std() * np.sqrt(252)
            vol = vol.replace(0, np.nan)
            ctv_ratios[c] = carry / vol
        else:
            ctv_ratios[c] = 0.0

    ctv_ratios = ctv_ratios.fillna(0)

    # Stop-loss: trigger if CtV drops > 2 std dev in 1 day
    ctv_rolling_std = ctv_ratios.rolling(window).std().fillna(1)
    stop_signal = (ctv_ratios.diff() < -2 * ctv_rolling_std).astype(int)

    # Gate: CtV > 0.5 to enter, no stop-loss active
    gate = ((ctv_ratios > 0.5) & (stop_signal == 0)).astype(int)

    return ctv_ratios, gate


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 1: MoneyVelocityModule
# ═══════════════════════════════════════════════════════════════════════════
class MoneyVelocityModule:
    """Fisher equation money velocity: V = GDP / M2.

    Tracks credit impulse, TED spread proxy, SOFR tracking,
    and produces a composite Liquidity Score 0-100.
    """

    # Historical baseline ratios — refreshed from FRED on init, fallback to constants
    _V_BASELINE = 1.12          # Long-run US velocity of money
    _M2_GROWTH_MEAN = 0.06      # ~6% annual M2 growth baseline
    _CREDIT_IMPULSE_MEAN = 0.0  # Neutral impulse
    _TED_SPREAD_MEAN = 0.35     # Long-run TED spread ~35bps
    _SOFR_BASELINE = 5.0        # Current-era SOFR baseline (%)

    def __init__(self):
        self.velocity: float = self._V_BASELINE
        self.velocity_change: float = 0.0
        self.credit_impulse: float = 0.0
        self.ted_spread: float = self._TED_SPREAD_MEAN
        self.sofr_rate: float = self._SOFR_BASELINE
        self.liquidity_score: float = 50.0
        self._history: list[dict] = []
        # Refresh baselines from live FRED data
        self._refresh_baselines()

    def _refresh_baselines(self):
        """Pull latest FRED data to update baseline constants.

        Fetches SOFR, M2 velocity, and TED spread from FRED.
        Falls back to hardcoded constants if API unavailable.
        """
        try:
            # SOFR — latest effective rate
            sofr_data = get_macro_data()
            if hasattr(sofr_data, "empty") and not sofr_data.empty:
                if "SOFR" in str(sofr_data.columns):
                    last_sofr = float(sofr_data.iloc[-1].get("SOFR", self._SOFR_BASELINE))
                    if 0 < last_sofr < 20:
                        self.sofr_rate = last_sofr
                        self.__class__._SOFR_BASELINE = last_sofr
                        logger.debug("SOFR baseline refreshed: %.2f%%", last_sofr)
        except Exception:
            pass

        try:
            # M2 velocity from FRED (M2V series)
            from ..data.openbb_data import get_fred_series
            m2v = get_fred_series("M2V", start="2020-01-01")
            if hasattr(m2v, "empty") and not m2v.empty:
                cols = m2v.select_dtypes(include=["number"]).columns
                if len(cols) > 0:
                    last_v = float(m2v[cols[0]].dropna().iloc[-1])
                    if 0.5 < last_v < 3.0:
                        self.velocity = last_v
                        self.__class__._V_BASELINE = last_v
                        logger.debug("M2 velocity baseline refreshed: %.3f", last_v)
        except Exception:
            pass

    def compute_velocity(
        self,
        gdp_proxy: pd.Series,
        m2_proxy: pd.Series,
    ) -> float:
        """Compute Fisher velocity V = GDP / M2.

        Uses nominal GDP proxy (e.g. SPY market cap or GDP ETF) and
        M2 proxy (can be M2 index or aggregate liquidity measure).
        """
        if gdp_proxy is None or m2_proxy is None:
            return self.velocity
        if len(gdp_proxy) < 2 or len(m2_proxy) < 2:
            return self.velocity

        gdp_val = float(gdp_proxy.dropna().iloc[-1])
        m2_val = float(m2_proxy.dropna().iloc[-1])

        if m2_val <= 0:
            return self.velocity

        current_v = gdp_val / m2_val
        # Normalise to baseline range
        self.velocity = current_v
        if len(gdp_proxy) > 21:
            prev_gdp = float(gdp_proxy.dropna().iloc[-22])
            prev_m2 = float(m2_proxy.dropna().iloc[-22])
            if prev_m2 > 0:
                prev_v = prev_gdp / prev_m2
                self.velocity_change = (current_v - prev_v) / max(abs(prev_v), 1e-8)
        return self.velocity

    def compute_credit_impulse(
        self,
        credit_series: pd.Series,
        gdp_proxy: pd.Series,
        lookback: int = 63,
    ) -> float:
        """Credit impulse = d(Credit Growth) / GDP.

        Second derivative of credit relative to GDP — captures acceleration
        in credit creation which leads real economic activity by 3-6 months.
        """
        if credit_series is None or gdp_proxy is None:
            return self.credit_impulse
        if len(credit_series) < lookback + 5 or len(gdp_proxy) < lookback + 5:
            return self.credit_impulse

        credit_growth = credit_series.pct_change(lookback).dropna()
        if len(credit_growth) < 2:
            return self.credit_impulse

        # Second derivative — acceleration of credit
        credit_accel = credit_growth.diff().dropna()
        if len(credit_accel) == 0:
            return self.credit_impulse

        gdp_val = float(gdp_proxy.dropna().iloc[-1])
        if gdp_val <= 0:
            return self.credit_impulse

        self.credit_impulse = float(credit_accel.iloc[-1]) / max(abs(gdp_val), 1e-8)
        # Scale to interpretable range
        self.credit_impulse = np.clip(self.credit_impulse * 1e6, -10.0, 10.0)
        return self.credit_impulse

    def compute_ted_spread(
        self,
        tbill_3m: pd.Series,
        libor_proxy: pd.Series,
    ) -> float:
        """TED spread = 3M LIBOR - 3M T-Bill.

        Uses proxy (e.g. short-term credit ETF yield vs treasury ETF yield).
        Higher TED spread indicates banking system stress.
        """
        if tbill_3m is None or libor_proxy is None:
            return self.ted_spread
        if len(tbill_3m) < 1 or len(libor_proxy) < 1:
            return self.ted_spread

        t_val = float(tbill_3m.dropna().iloc[-1])
        l_val = float(libor_proxy.dropna().iloc[-1])
        self.ted_spread = max(l_val - t_val, 0.0)
        return self.ted_spread

    def compute_sofr_tracking(
        self,
        sofr_proxy: pd.Series,
    ) -> float:
        """Track SOFR rate from proxy series.

        SOFR replaced LIBOR as the primary US overnight rate.
        Spikes in SOFR indicate repo market stress.
        """
        if sofr_proxy is None or len(sofr_proxy) < 1:
            return self.sofr_rate
        self.sofr_rate = float(sofr_proxy.dropna().iloc[-1])
        return self.sofr_rate

    def compute_liquidity_score(
        self,
        vix: float = 20.0,
        credit_spread: float = 3.0,
        yield_spread: float = 0.0,
    ) -> float:
        """Composite Liquidity Score 0-100.

        Aggregates:
          - Velocity signal (25%): Falling velocity → tightening
          - Credit impulse (20%): Negative impulse → tightening
          - TED spread (20%): Wide TED → stress
          - VIX component (20%): High VIX → illiquidity
          - Yield curve (15%): Inverted curve → tightening

        Returns float in [0, 100] where 100 = maximum liquidity.
        """
        # Velocity component: higher velocity_change is expansionary
        v_signal = sigmoid_trigger(self.velocity_change, 0.0, sensitivity=10.0)
        v_score = float(v_signal) * 100.0

        # Credit impulse: positive impulse is expansionary
        ci_signal = sigmoid_trigger(self.credit_impulse, 0.0, sensitivity=2.0)
        ci_score = float(ci_signal) * 100.0

        # TED spread: low is better (inverted — high TED = low liquidity)
        ted_score = max(0.0, 100.0 - self.ted_spread * 100.0)

        # VIX: low is better for liquidity
        vix_score = max(0.0, min(100.0, 100.0 - (vix - 12.0) * 2.5))

        # Yield curve: positive spread = healthy
        yc_score = max(0.0, min(100.0, 50.0 + yield_spread * 20.0))

        self.liquidity_score = np.clip(
            v_score * 0.25
            + ci_score * 0.20
            + ted_score * 0.20
            + vix_score * 0.20
            + yc_score * 0.15,
            0.0,
            100.0,
        )

        self._history.append({
            "velocity": self.velocity,
            "velocity_change": self.velocity_change,
            "credit_impulse": self.credit_impulse,
            "ted_spread": self.ted_spread,
            "sofr_rate": self.sofr_rate,
            "liquidity_score": self.liquidity_score,
        })

        return self.liquidity_score

    def get_state(self) -> dict:
        """Return current money velocity state."""
        return {
            "velocity": round(self.velocity, 6),
            "velocity_change": round(self.velocity_change, 6),
            "credit_impulse": round(self.credit_impulse, 4),
            "ted_spread": round(self.ted_spread, 4),
            "sofr_rate": round(self.sofr_rate, 4),
            "liquidity_score": round(self.liquidity_score, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 2: SectorRanker
# ═══════════════════════════════════════════════════════════════════════════
class SectorRanker:
    """Enhanced sector ranking with macro-adjusted momentum,
    relative strength, and factor rotation signals.
    """

    # Defensive sectors for stress regimes
    DEFENSIVE_SECTORS = {"Utilities", "Consumer Staples", "Health Care", "Real Estate"}
    # Cyclical sectors for bull regimes
    CYCLICAL_SECTORS = {
        "Information Technology", "Consumer Discretionary",
        "Communication Services", "Financials", "Industrials",
    }
    # Rate-sensitive sectors
    RATE_SENSITIVE = {"Utilities", "Real Estate", "Financials"}

    def __init__(self):
        self._rankings: dict[str, float] = {}
        self._relative_strength: dict[str, float] = {}
        self._factor_signals: dict[str, str] = {}

    def rank_sectors(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        regime: MarketRegime,
        vix: float = 20.0,
        yield_spread: float = 0.0,
        credit_spread: float = 3.0,
    ) -> dict[str, float]:
        """Rank sectors by composite score with macro adjustments.

        Composite = momentum_score * regime_multiplier * relative_strength_adj

        Args:
            returns: DataFrame of ETF daily returns
            sector_map: {etf_ticker: sector_name}
            regime: Current market regime
            vix: Current VIX level
            yield_spread: 10Y - 2Y spread
            credit_spread: HY-IG spread proxy

        Returns:
            Dict of {sector: score} sorted descending.
        """
        rankings = {}
        inv = {v: k for k, v in sector_map.items()} if sector_map else {}

        for etf in returns.columns:
            sector = inv.get(etf, etf)
            r = returns[etf].dropna()
            if len(r) < 21:
                continue

            # --- Momentum components ---
            mom_1m = float(r.iloc[-21:].sum())
            mom_3m = float(r.iloc[-63:].sum()) if len(r) >= 63 else mom_1m
            mom_6m = float(r.iloc[-126:].sum()) if len(r) >= 126 else mom_3m

            # Blended momentum: 50% 3M + 30% 1M + 20% 6M
            blended_mom = mom_3m * 0.50 + mom_1m * 0.30 + mom_6m * 0.20

            # --- Risk-adjusted ---
            vol = float(r.std() * np.sqrt(252))
            sharpe = (blended_mom * 4) / vol if vol > 0 else 0.0

            # --- Relative strength vs SPY proxy ---
            # Use mean of all sectors as benchmark
            mkt_mean = float(returns.mean(axis=1).iloc[-63:].sum()) if len(returns) >= 63 else 0.0
            rs_score = mom_3m - mkt_mean
            self._relative_strength[sector] = round(rs_score, 4)

            # --- Regime multiplier ---
            regime_mult = self._regime_multiplier(
                sector, regime, vix, yield_spread, credit_spread,
            )

            # --- Factor rotation signal ---
            factor_signal = self._factor_rotation_signal(
                sector, regime, vix, yield_spread,
            )
            self._factor_signals[sector] = factor_signal

            # Composite score
            composite = sharpe * regime_mult * (1.0 + rs_score * 0.5)
            rankings[sector] = round(composite, 4)

        self._rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
        return self._rankings

    def _regime_multiplier(
        self,
        sector: str,
        regime: MarketRegime,
        vix: float,
        yield_spread: float,
        credit_spread: float,
    ) -> float:
        """Compute regime-based multiplier for a sector."""
        mult = 1.0

        if regime == MarketRegime.STRESS or regime == MarketRegime.CRASH:
            if sector in self.DEFENSIVE_SECTORS:
                mult *= 1.4
            elif sector in self.CYCLICAL_SECTORS:
                mult *= 0.6
        elif regime == MarketRegime.BULL:
            if sector in self.CYCLICAL_SECTORS:
                mult *= 1.25
            elif sector in self.DEFENSIVE_SECTORS:
                mult *= 0.85

        # VIX adjustment
        if vix > 30 and sector in self.CYCLICAL_SECTORS:
            mult *= 0.8
        elif vix < 15 and sector in self.CYCLICAL_SECTORS:
            mult *= 1.1

        # Rate sensitivity
        if sector in self.RATE_SENSITIVE:
            if yield_spread < -0.5:
                mult *= 0.85  # Inverted curve hurts rate-sensitive
            elif yield_spread > 1.0:
                mult *= 1.1   # Steep curve benefits financials

        # Credit stress penalty
        if credit_spread > 5.0 and sector == "Financials":
            mult *= 0.7

        return mult

    def _factor_rotation_signal(
        self,
        sector: str,
        regime: MarketRegime,
        vix: float,
        yield_spread: float,
    ) -> str:
        """Determine factor rotation signal for a sector.

        Returns one of: MOMENTUM, VALUE, QUALITY, DEFENSIVE, NEUTRAL
        """
        if regime == MarketRegime.CRASH or vix > 35:
            return "DEFENSIVE"
        if regime == MarketRegime.STRESS:
            if sector in self.DEFENSIVE_SECTORS:
                return "QUALITY"
            return "DEFENSIVE"
        if regime == MarketRegime.BULL:
            if sector in self.CYCLICAL_SECTORS:
                return "MOMENTUM"
            return "NEUTRAL"
        if regime == MarketRegime.BEAR:
            if yield_spread < -0.3:
                return "VALUE"
            return "QUALITY"
        return "NEUTRAL"

    def get_relative_strength(self) -> dict[str, float]:
        """Return latest relative strength scores."""
        return dict(self._relative_strength)

    def get_factor_signals(self) -> dict[str, str]:
        """Return latest factor rotation signals."""
        return dict(self._factor_signals)


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 3: CarryToVolatility
# ═══════════════════════════════════════════════════════════════════════════
class CarryToVolatility:
    """Full Carry-to-Volatility (CtV) signal generation for forex pairs.

    CtV = (yield_foreign - yield_domestic) / realised_vol(FX pair)

    Includes SDR basket tracking and dynamic stop-loss thresholds.
    """

    # G10 FX pairs vs USD
    FX_PAIRS = {
        "EURUSD": {"carry_ticker": "EUR", "fx_ticker": "EURUSD=X"},
        "GBPUSD": {"carry_ticker": "GBP", "fx_ticker": "GBPUSD=X"},
        "USDJPY": {"carry_ticker": "JPY", "fx_ticker": "JPY=X"},
        "USDCNY": {"carry_ticker": "CNY", "fx_ticker": "CNY=X"},
        "AUDUSD": {"carry_ticker": "AUD", "fx_ticker": "AUDUSD=X"},
        "USDCAD": {"carry_ticker": "CAD", "fx_ticker": "CAD=X"},
        "USDCHF": {"carry_ticker": "CHF", "fx_ticker": "CHF=X"},
        "NZDUSD": {"carry_ticker": "NZD", "fx_ticker": "NZDUSD=X"},
    }

    # SDR basket weights (IMF current allocation)
    SDR_BASKET = SDR_WEIGHTS.to_dict()

    def __init__(self, vol_window: int = 20, stop_loss_sigma: float = 2.0):
        self.vol_window = vol_window
        self.stop_loss_sigma = stop_loss_sigma
        self._ctv_ratios: dict[str, float] = {}
        self._gates: dict[str, bool] = {}
        self._sdr_ctv: float = 0.0

    def compute_pair_ctv(
        self,
        carry_spread: float,
        fx_returns: pd.Series,
    ) -> tuple[float, bool]:
        """Compute CtV for a single pair.

        Args:
            carry_spread: yield_foreign - yield_usd (annualised, decimal)
            fx_returns: Daily FX return series

        Returns:
            (ctv_ratio, gate_open) where gate_open = True if entry is valid.
        """
        if fx_returns is None or len(fx_returns) < self.vol_window:
            return 0.0, False

        realised_vol = float(fx_returns.rolling(self.vol_window).std().iloc[-1]) * np.sqrt(252)
        if realised_vol <= 1e-8:
            return 0.0, False

        ctv = carry_spread / realised_vol

        # Stop-loss: check for sudden CtV deterioration
        if len(fx_returns) >= self.vol_window + 5:
            rolling_std = float(fx_returns.rolling(self.vol_window).std().std()) * np.sqrt(252)
            vol_change = float(fx_returns.iloc[-5:].std()) * np.sqrt(252) - realised_vol
            stop_triggered = vol_change > self.stop_loss_sigma * max(rolling_std, 1e-8)
        else:
            stop_triggered = False

        gate_open = (ctv > 0.5) and (not stop_triggered)
        return ctv, gate_open

    def compute_all_pairs(
        self,
        yield_spreads: dict[str, float],
        fx_returns: dict[str, pd.Series],
    ) -> dict[str, dict]:
        """Compute CtV for all configured FX pairs.

        Args:
            yield_spreads: {pair_name: carry_spread}
            fx_returns: {pair_name: daily_return_series}

        Returns:
            Dict of {pair: {"ctv": float, "gate": bool}}
        """
        results = {}
        for pair in self.FX_PAIRS:
            spread = yield_spreads.get(pair, 0.0)
            rets = fx_returns.get(pair, None)
            ctv, gate = self.compute_pair_ctv(spread, rets)
            self._ctv_ratios[pair] = ctv
            self._gates[pair] = gate
            results[pair] = {"ctv": round(ctv, 4), "gate": gate}
        return results

    def compute_sdr_ctv(self, pair_ctvs: dict[str, float]) -> float:
        """Compute SDR-weighted aggregate CtV.

        Weights individual pair CtVs by SDR basket allocation.
        """
        sdr_mapping = {
            "EURUSD": "EUR", "GBPUSD": "GBP",
            "USDJPY": "JPY", "USDCNY": "CNY",
        }
        total = 0.0
        weight_sum = 0.0
        for pair, ctv in pair_ctvs.items():
            currency = sdr_mapping.get(pair)
            if currency and currency in self.SDR_BASKET:
                w = self.SDR_BASKET[currency]
                total += ctv * w
                weight_sum += w
        if weight_sum > 0:
            self._sdr_ctv = total / weight_sum
        else:
            self._sdr_ctv = 0.0
        return self._sdr_ctv

    def get_top_carry_trades(self, n: int = 3) -> list[tuple[str, float]]:
        """Return top N carry trades by CtV ratio where gate is open."""
        eligible = [
            (pair, ctv) for pair, ctv in self._ctv_ratios.items()
            if self._gates.get(pair, False)
        ]
        eligible.sort(key=lambda x: x[1], reverse=True)
        return eligible[:n]


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 4: RegimeTransitionDetector
# ═══════════════════════════════════════════════════════════════════════════
class RegimeTransitionDetector:
    """Detect regime transitions with confidence scoring and lookback validation.

    Uses a multi-factor scoring approach with hysteresis to avoid whipsaws.
    Tracks the last N regime observations to compute transition probabilities.
    """

    # Transition probability matrix (from → to)
    _BASE_TRANSITION = {
        MarketRegime.BULL: {
            MarketRegime.BULL: 0.80, MarketRegime.TRANSITION: 0.15,
            MarketRegime.BEAR: 0.03, MarketRegime.STRESS: 0.015,
            MarketRegime.CRASH: 0.005,
        },
        MarketRegime.TRANSITION: {
            MarketRegime.BULL: 0.30, MarketRegime.TRANSITION: 0.40,
            MarketRegime.BEAR: 0.20, MarketRegime.STRESS: 0.08,
            MarketRegime.CRASH: 0.02,
        },
        MarketRegime.BEAR: {
            MarketRegime.BULL: 0.05, MarketRegime.TRANSITION: 0.20,
            MarketRegime.BEAR: 0.55, MarketRegime.STRESS: 0.15,
            MarketRegime.CRASH: 0.05,
        },
        MarketRegime.STRESS: {
            MarketRegime.BULL: 0.02, MarketRegime.TRANSITION: 0.10,
            MarketRegime.BEAR: 0.20, MarketRegime.STRESS: 0.53,
            MarketRegime.CRASH: 0.15,
        },
        MarketRegime.CRASH: {
            MarketRegime.BULL: 0.01, MarketRegime.TRANSITION: 0.05,
            MarketRegime.BEAR: 0.15, MarketRegime.STRESS: 0.39,
            MarketRegime.CRASH: 0.40,
        },
    }

    def __init__(self, lookback: int = 20, hysteresis_days: int = 3):
        self.lookback = lookback
        self.hysteresis_days = hysteresis_days
        self._regime_history: list[MarketRegime] = []
        self._transition_confidence: float = 0.0
        self._days_in_regime: int = 0
        self._current_regime: MarketRegime = MarketRegime.TRANSITION

    def update(self, new_regime: MarketRegime) -> dict:
        """Update regime history and detect transitions.

        Returns dict with:
            - regime: confirmed regime
            - transition_detected: bool
            - confidence: float 0-1
            - days_in_regime: int
            - transition_from: previous regime if transition detected
        """
        prev_regime = self._current_regime
        self._regime_history.append(new_regime)

        # Keep lookback window
        if len(self._regime_history) > self.lookback:
            self._regime_history = self._regime_history[-self.lookback:]

        # Hysteresis check: need N consecutive days in new regime
        if new_regime == prev_regime:
            self._days_in_regime += 1
            transition_detected = False
        else:
            # Count consecutive observations of new regime
            consecutive = 0
            for r in reversed(self._regime_history):
                if r == new_regime:
                    consecutive += 1
                else:
                    break

            if consecutive >= self.hysteresis_days:
                # Confirmed transition
                self._current_regime = new_regime
                self._days_in_regime = consecutive
                transition_detected = True
            else:
                # Not yet confirmed — stay in current regime
                self._days_in_regime += 1
                transition_detected = False

        # Compute confidence
        self._transition_confidence = self._compute_confidence(
            prev_regime, new_regime,
        )

        return {
            "regime": self._current_regime,
            "transition_detected": transition_detected,
            "confidence": round(self._transition_confidence, 4),
            "days_in_regime": self._days_in_regime,
            "transition_from": prev_regime if transition_detected else None,
        }

    def _compute_confidence(
        self, from_regime: MarketRegime, to_regime: MarketRegime,
    ) -> float:
        """Compute transition confidence based on historical frequency
        and base transition probabilities.
        """
        if len(self._regime_history) < 3:
            return 0.5

        # Frequency of target regime in recent history
        recent = self._regime_history[-min(self.lookback, len(self._regime_history)):]
        freq = sum(1 for r in recent if r == to_regime) / len(recent)

        # Base probability from transition matrix
        base_prob = self._BASE_TRANSITION.get(from_regime, {}).get(to_regime, 0.1)

        # Combine: 60% frequency + 40% base probability
        confidence = freq * 0.60 + base_prob * 0.40
        return np.clip(confidence, 0.0, 1.0)

    def get_regime_distribution(self) -> dict[str, float]:
        """Return frequency distribution of regimes in lookback window."""
        if not self._regime_history:
            return {}
        total = len(self._regime_history)
        dist = {}
        for regime in MarketRegime:
            count = sum(1 for r in self._regime_history if r == regime)
            dist[regime.value] = round(count / total, 4)
        return dist

    def get_transition_matrix(self) -> dict[str, dict[str, int]]:
        """Compute observed transition matrix from regime history."""
        matrix = {r.value: {r2.value: 0 for r2 in MarketRegime} for r in MarketRegime}
        for i in range(1, len(self._regime_history)):
            fr = self._regime_history[i - 1].value
            to = self._regime_history[i].value
            matrix[fr][to] += 1
        return matrix


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 5: YieldCurveAnalyzer
# ═══════════════════════════════════════════════════════════════════════════
class YieldCurveAnalyzer:
    """Full yield curve analysis.

    Computes:
        - 2s10s spread (classic recession indicator)
        - 3m10y spread (Fed's preferred measure)
        - Term premium estimate
        - Real rate estimate (nominal - breakeven inflation proxy)
        - Curve shape classification (STEEP, FLAT, INVERTED, BEAR_FLAT, BULL_STEEP)
    """

    SHAPE_STEEP = "STEEP"
    SHAPE_FLAT = "FLAT"
    SHAPE_INVERTED = "INVERTED"
    SHAPE_BEAR_FLAT = "BEAR_FLAT"      # Rising rates + flattening
    SHAPE_BULL_STEEP = "BULL_STEEP"     # Falling rates + steepening

    def __init__(self):
        self.spread_2s10s: float = 0.0
        self.spread_3m10y: float = 0.0
        self.term_premium: float = 0.0
        self.real_rate_10y: float = 0.0
        self.curve_shape: str = self.SHAPE_FLAT
        self._history: list[dict] = []

    def analyze(
        self,
        yield_10y: float,
        yield_2y: float,
        yield_3m: Optional[float] = None,
        breakeven_inflation: Optional[float] = None,
        fed_funds: Optional[float] = None,
        prev_10y: Optional[float] = None,
    ) -> dict:
        """Run full yield curve analysis.

        Args:
            yield_10y: 10-year treasury yield (%)
            yield_2y: 2-year treasury yield (%)
            yield_3m: 3-month treasury yield (%), optional
            breakeven_inflation: 10Y breakeven inflation rate (%), optional
            fed_funds: Federal funds rate (%), optional
            prev_10y: Previous period 10Y yield for term premium calc

        Returns:
            Dict with all yield curve metrics.
        """
        # 2s10s spread
        self.spread_2s10s = yield_10y - yield_2y

        # 3m10y spread (use 2y as proxy if 3m not available)
        if yield_3m is not None:
            self.spread_3m10y = yield_10y - yield_3m
        else:
            self.spread_3m10y = self.spread_2s10s * 1.2  # Approximate

        # Term premium: 10Y yield - expected short rate path
        # Simplified: use fed funds as anchor for expected path
        if fed_funds is not None:
            expected_avg_rate = (fed_funds * 0.6 + yield_2y * 0.4)
            self.term_premium = yield_10y - expected_avg_rate
        else:
            self.term_premium = self.spread_2s10s * 0.5  # Rough estimate

        # Real rate
        if breakeven_inflation is not None:
            self.real_rate_10y = yield_10y - breakeven_inflation
        else:
            self.real_rate_10y = yield_10y - 2.5  # Assume ~2.5% inflation

        # Classify curve shape
        self.curve_shape = self._classify_shape(
            yield_10y, yield_2y, prev_10y,
        )

        result = {
            "spread_2s10s": round(self.spread_2s10s, 4),
            "spread_3m10y": round(self.spread_3m10y, 4),
            "term_premium": round(self.term_premium, 4),
            "real_rate_10y": round(self.real_rate_10y, 4),
            "curve_shape": self.curve_shape,
            "inversion_signal": self.spread_2s10s < 0,
            "deep_inversion": self.spread_2s10s < -0.5,
        }
        self._history.append(result)
        return result

    def _classify_shape(
        self,
        yield_10y: float,
        yield_2y: float,
        prev_10y: Optional[float] = None,
    ) -> str:
        """Classify yield curve shape."""
        spread = yield_10y - yield_2y

        if prev_10y is not None:
            rate_direction = yield_10y - prev_10y
        else:
            rate_direction = 0.0

        if spread < -0.25:
            return self.SHAPE_INVERTED
        elif spread < 0.25:
            if rate_direction > 0.1:
                return self.SHAPE_BEAR_FLAT
            return self.SHAPE_FLAT
        elif spread > 1.0:
            if rate_direction < -0.1:
                return self.SHAPE_BULL_STEEP
            return self.SHAPE_STEEP
        else:
            return self.SHAPE_FLAT

    def recession_probability(self) -> float:
        """Estimate recession probability from yield curve signals.

        Based on NY Fed model: uses 3m10y spread.
        """
        # NY Fed probit model approximation
        # P(recession) = Phi(-0.5333 - 0.6330 * spread_3m10y)
        # Using sigmoid as approximation to normal CDF
        z = -0.5333 - 0.6330 * self.spread_3m10y
        prob = 1.0 / (1.0 + np.exp(-1.7 * z))  # Logistic approx to normal CDF
        return float(np.clip(prob, 0.0, 1.0))

    def get_history(self) -> list[dict]:
        """Return analysis history."""
        return list(self._history)


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 6: CreditPulseMonitor
# ═══════════════════════════════════════════════════════════════════════════
class CreditPulseMonitor:
    """Credit spread monitoring, HY/IG differential, and credit impulse.

    Tracks:
        - HY-IG spread differential
        - Credit spread z-score
        - Credit impulse (rate of change of spreads)
        - Stress indicator (combined signal)
    """

    # Thresholds for credit stress levels
    NORMAL_SPREAD = 3.0       # HY-IG spread in normal conditions
    ELEVATED_SPREAD = 4.5     # Elevated credit risk
    STRESS_SPREAD = 6.0       # Credit stress
    CRISIS_SPREAD = 8.0       # Credit crisis

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.hy_ig_spread: float = self.NORMAL_SPREAD
        self.spread_zscore: float = 0.0
        self.credit_impulse: float = 0.0
        self.stress_level: str = "NORMAL"
        self._spread_history: list[float] = []

    def update(
        self,
        hy_spread: float,
        ig_spread: float,
    ) -> dict:
        """Update credit pulse with new HY and IG spread observations.

        Args:
            hy_spread: High-yield credit spread (bps or %)
            ig_spread: Investment-grade credit spread (bps or %)

        Returns:
            Dict with credit pulse metrics.
        """
        self.hy_ig_spread = hy_spread - ig_spread
        self._spread_history.append(self.hy_ig_spread)

        # Keep history bounded
        if len(self._spread_history) > self.lookback:
            self._spread_history = self._spread_history[-self.lookback:]

        # Z-score
        if len(self._spread_history) >= 20:
            arr = np.array(self._spread_history)
            mean = arr.mean()
            std = arr.std()
            self.spread_zscore = (self.hy_ig_spread - mean) / max(std, 1e-8)
        else:
            self.spread_zscore = 0.0

        # Credit impulse: rate of change
        if len(self._spread_history) >= 5:
            recent = np.array(self._spread_history[-5:])
            self.credit_impulse = float(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        else:
            self.credit_impulse = 0.0

        # Stress classification
        self.stress_level = self._classify_stress()

        return {
            "hy_ig_spread": round(self.hy_ig_spread, 4),
            "spread_zscore": round(self.spread_zscore, 4),
            "credit_impulse": round(self.credit_impulse, 4),
            "stress_level": self.stress_level,
            "widening": self.credit_impulse > 0,
        }

    def _classify_stress(self) -> str:
        """Classify credit stress level."""
        spread = abs(self.hy_ig_spread)
        if spread >= self.CRISIS_SPREAD:
            return "CRISIS"
        elif spread >= self.STRESS_SPREAD:
            return "STRESS"
        elif spread >= self.ELEVATED_SPREAD:
            return "ELEVATED"
        else:
            return "NORMAL"

    def compute_from_etfs(
        self,
        hy_prices: pd.Series,
        ig_prices: pd.Series,
        window: int = 20,
    ) -> dict:
        """Compute credit spread proxy from ETF volatility differential.

        HY ETF (e.g. HYG) vs IG ETF (e.g. LQD) realised vol spread
        serves as credit spread proxy.
        """
        if hy_prices is None or ig_prices is None:
            return self.get_state()
        if len(hy_prices) < window + 5 or len(ig_prices) < window + 5:
            return self.get_state()

        hy_vol = float(hy_prices.pct_change().rolling(window).std().dropna().iloc[-1]) * np.sqrt(252) * 100
        ig_vol = float(ig_prices.pct_change().rolling(window).std().dropna().iloc[-1]) * np.sqrt(252) * 100

        return self.update(hy_vol, ig_vol)

    def is_stress(self) -> bool:
        """Return True if credit conditions are stressed."""
        return self.stress_level in ("STRESS", "CRISIS")

    def get_state(self) -> dict:
        """Return current credit pulse state."""
        return {
            "hy_ig_spread": round(self.hy_ig_spread, 4),
            "spread_zscore": round(self.spread_zscore, 4),
            "credit_impulse": round(self.credit_impulse, 4),
            "stress_level": self.stress_level,
            "widening": self.credit_impulse > 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODULE 7: MacroFeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════
class MacroFeatureBuilder:
    """Build macro feature matrix for ML models.

    Generates 50+ features from macro data for regime classification,
    return prediction, and risk models.
    """

    def __init__(self):
        self._feature_names: list[str] = []

    def build_features(
        self,
        macro: pd.DataFrame,
        snapshot: Optional[MacroSnapshot] = None,
    ) -> dict[str, float]:
        """Build complete macro feature dictionary.

        Args:
            macro: DataFrame from get_macro_data() with columns like
                   VIX, S&P 500, 10Y Yield, etc.
            snapshot: Optional MacroSnapshot for derived features.

        Returns:
            Dict of {feature_name: value} with 50+ features.
        """
        features: dict[str, float] = {}

        # --- VIX features (5) ---
        features.update(self._vix_features(macro))

        # --- Equity features (8) ---
        features.update(self._equity_features(macro))

        # --- Yield features (8) ---
        features.update(self._yield_features(macro))

        # --- Credit features (5) ---
        features.update(self._credit_features(macro))

        # --- Commodity features (6) ---
        features.update(self._commodity_features(macro))

        # --- Momentum / trend features (8) ---
        features.update(self._momentum_features(macro))

        # --- Volatility features (5) ---
        features.update(self._volatility_features(macro))

        # --- Snapshot-derived features (7+) ---
        if snapshot is not None:
            features.update(self._snapshot_features(snapshot))

        self._feature_names = list(features.keys())
        return features

    def _safe_last(self, series: pd.Series) -> float:
        """Safely get last non-NaN value from a series."""
        if series is None or len(series) == 0:
            return 0.0
        clean = series.dropna()
        if len(clean) == 0:
            return 0.0
        return float(clean.iloc[-1])

    def _safe_pct_change(self, series: pd.Series, periods: int) -> float:
        """Safely compute percent change over N periods."""
        if series is None or len(series) < periods + 1:
            return 0.0
        clean = series.dropna()
        if len(clean) < periods + 1:
            return 0.0
        return float(clean.iloc[-1] / clean.iloc[-periods - 1] - 1)

    def _safe_rolling_stat(
        self, series: pd.Series, window: int, stat: str = "mean",
    ) -> float:
        """Safely compute rolling statistic."""
        if series is None or len(series) < window:
            return 0.0
        clean = series.dropna()
        if len(clean) < window:
            return 0.0
        rolled = clean.rolling(window)
        if stat == "mean":
            return float(rolled.mean().iloc[-1])
        elif stat == "std":
            return float(rolled.std().iloc[-1])
        elif stat == "min":
            return float(rolled.min().iloc[-1])
        elif stat == "max":
            return float(rolled.max().iloc[-1])
        return 0.0

    def _vix_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """VIX-derived features."""
        vix = macro.get("VIX") if "VIX" in macro.columns else None
        return {
            "vix_level": self._safe_last(vix),
            "vix_5d_ma": self._safe_rolling_stat(vix, 5, "mean"),
            "vix_20d_ma": self._safe_rolling_stat(vix, 20, "mean"),
            "vix_5d_change": self._safe_pct_change(vix, 5),
            "vix_zscore_20d": (
                (self._safe_last(vix) - self._safe_rolling_stat(vix, 20, "mean"))
                / max(self._safe_rolling_stat(vix, 20, "std"), 1e-8)
            ),
        }

    def _equity_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """S&P 500 / equity features."""
        spy = macro.get("S&P 500") if "S&P 500" in macro.columns else None
        return {
            "spy_return_1d": self._safe_pct_change(spy, 1),
            "spy_return_5d": self._safe_pct_change(spy, 5),
            "spy_return_21d": self._safe_pct_change(spy, 21),
            "spy_return_63d": self._safe_pct_change(spy, 63),
            "spy_return_126d": self._safe_pct_change(spy, 126),
            "spy_return_252d": self._safe_pct_change(spy, 252),
            "spy_vol_21d": self._safe_rolling_stat(
                spy.pct_change() if spy is not None else None, 21, "std",
            ) * np.sqrt(252) if spy is not None else 0.0,
            "spy_vol_63d": self._safe_rolling_stat(
                spy.pct_change() if spy is not None else None, 63, "std",
            ) * np.sqrt(252) if spy is not None else 0.0,
        }

    def _yield_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """Interest rate / yield features."""
        y10 = macro.get("10Y Yield") if "10Y Yield" in macro.columns else None
        y5 = macro.get("5Y Yield") if "5Y Yield" in macro.columns else None

        y10_last = self._safe_last(y10)
        y5_last = self._safe_last(y5)

        return {
            "yield_10y": y10_last,
            "yield_2y_proxy": y5_last,
            "yield_spread": y10_last - y5_last,
            "yield_10y_change_5d": self._safe_pct_change(y10, 5),
            "yield_10y_change_21d": self._safe_pct_change(y10, 21),
            "yield_2y_change_5d": self._safe_pct_change(y5, 5),
            "yield_curve_slope": (y10_last - y5_last) / max(abs(y5_last), 1e-8),
            "yield_curve_inverted": 1.0 if (y10_last - y5_last) < 0 else 0.0,
        }

    def _credit_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """Credit market features."""
        hy = macro.get("HY Corporate") if "HY Corporate" in macro.columns else None
        ig = macro.get("IG Corporate") if "IG Corporate" in macro.columns else None

        hy_vol = 0.0
        ig_vol = 0.0
        if hy is not None and len(hy.dropna()) > 21:
            hy_vol = self._safe_rolling_stat(hy.pct_change(), 21, "std") * np.sqrt(252)
        if ig is not None and len(ig.dropna()) > 21:
            ig_vol = self._safe_rolling_stat(ig.pct_change(), 21, "std") * np.sqrt(252)

        return {
            "credit_spread_proxy": (hy_vol - ig_vol) * 100,
            "hy_vol_21d": hy_vol,
            "ig_vol_21d": ig_vol,
            "hy_return_21d": self._safe_pct_change(hy, 21),
            "ig_return_21d": self._safe_pct_change(ig, 21),
        }

    def _commodity_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """Commodity / gold features."""
        gold = macro.get("Gold") if "Gold" in macro.columns else None
        return {
            "gold_level": self._safe_last(gold),
            "gold_return_5d": self._safe_pct_change(gold, 5),
            "gold_return_21d": self._safe_pct_change(gold, 21),
            "gold_return_63d": self._safe_pct_change(gold, 63),
            "gold_vol_21d": self._safe_rolling_stat(
                gold.pct_change() if gold is not None else None, 21, "std",
            ) * np.sqrt(252) if gold is not None else 0.0,
            "gold_spy_corr_63d": self._gold_spy_correlation(macro, 63),
        }

    def _gold_spy_correlation(self, macro: pd.DataFrame, window: int) -> float:
        """Compute rolling gold-SPY correlation."""
        if "Gold" not in macro.columns or "S&P 500" not in macro.columns:
            return 0.0
        gold_ret = macro["Gold"].pct_change()
        spy_ret = macro["S&P 500"].pct_change()
        corr = gold_ret.rolling(window).corr(spy_ret)
        if len(corr.dropna()) == 0:
            return 0.0
        return float(corr.dropna().iloc[-1])

    def _momentum_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """Cross-asset momentum features."""
        spy = macro.get("S&P 500") if "S&P 500" in macro.columns else None
        gold = macro.get("Gold") if "Gold" in macro.columns else None

        spy_above_50 = 0.0
        spy_above_200 = 0.0
        if spy is not None and len(spy.dropna()) > 200:
            last = self._safe_last(spy)
            ma50 = self._safe_rolling_stat(spy, 50, "mean")
            ma200 = self._safe_rolling_stat(spy, 200, "mean")
            spy_above_50 = 1.0 if last > ma50 else 0.0
            spy_above_200 = 1.0 if last > ma200 else 0.0

        gold_trend = 0.0
        if gold is not None and len(gold.dropna()) > 50:
            gold_last = self._safe_last(gold)
            gold_ma50 = self._safe_rolling_stat(gold, 50, "mean")
            gold_trend = 1.0 if gold_last > gold_ma50 else -1.0

        # Trend strength: distance from 200d MA
        trend_strength = 0.0
        if spy is not None and len(spy.dropna()) > 200:
            last = self._safe_last(spy)
            ma200 = self._safe_rolling_stat(spy, 200, "mean")
            if ma200 != 0:
                trend_strength = (last - ma200) / ma200

        return {
            "spy_above_50dma": spy_above_50,
            "spy_above_200dma": spy_above_200,
            "spy_trend_strength": trend_strength,
            "gold_trend": gold_trend,
            "cross_asset_momentum": (
                self._safe_pct_change(spy, 63) if spy is not None else 0.0
            ) + (
                self._safe_pct_change(gold, 63) if gold is not None else 0.0
            ),
            "spy_drawdown": self._compute_drawdown(spy),
            "spy_skew_21d": self._compute_skew(spy, 21),
            "spy_kurt_21d": self._compute_kurtosis(spy, 21),
        }

    def _compute_drawdown(self, series: pd.Series) -> float:
        """Compute current drawdown from peak."""
        if series is None or len(series) < 2:
            return 0.0
        clean = series.dropna()
        if len(clean) < 2:
            return 0.0
        peak = float(clean.cummax().iloc[-1])
        current = float(clean.iloc[-1])
        if peak <= 0:
            return 0.0
        return (current - peak) / peak

    def _compute_skew(self, series: pd.Series, window: int) -> float:
        """Compute rolling skewness of returns."""
        if series is None or len(series) < window + 1:
            return 0.0
        rets = series.pct_change().dropna()
        if len(rets) < window:
            return 0.0
        return float(rets.iloc[-window:].skew())

    def _compute_kurtosis(self, series: pd.Series, window: int) -> float:
        """Compute rolling kurtosis of returns."""
        if series is None or len(series) < window + 1:
            return 0.0
        rets = series.pct_change().dropna()
        if len(rets) < window:
            return 0.0
        return float(rets.iloc[-window:].kurtosis())

    def _volatility_features(self, macro: pd.DataFrame) -> dict[str, float]:
        """Volatility regime features."""
        vix = macro.get("VIX") if "VIX" in macro.columns else None
        spy = macro.get("S&P 500") if "S&P 500" in macro.columns else None

        # VIX term structure proxy: VIX vs realised vol
        realised_vol = 0.0
        if spy is not None and len(spy.dropna()) > 21:
            realised_vol = self._safe_rolling_stat(
                spy.pct_change(), 21, "std",
            ) * np.sqrt(252) * 100  # annualised, in %

        vix_level = self._safe_last(vix)
        vrp = vix_level - realised_vol  # Variance risk premium

        # Vol of vol
        vov = 0.0
        if vix is not None and len(vix.dropna()) > 21:
            vov = self._safe_rolling_stat(vix.pct_change(), 21, "std") * np.sqrt(252)

        return {
            "realised_vol_21d": realised_vol,
            "variance_risk_premium": vrp,
            "vol_of_vol": vov,
            "vix_realised_ratio": vix_level / max(realised_vol, 1e-8),
            "vol_regime": 1.0 if vix_level > 25 else (0.0 if vix_level < 15 else 0.5),
        }

    def _snapshot_features(self, snapshot: MacroSnapshot) -> dict[str, float]:
        """Features derived from MacroSnapshot."""
        # Regime encoding (one-hot)
        regime_features = {
            f"regime_{r.value}": 1.0 if snapshot.regime == r else 0.0
            for r in MarketRegime
        }
        # Cube regime encoding
        cube_features = {
            f"cube_{r.value}": 1.0 if snapshot.cube_regime == r else 0.0
            for r in CubeRegime
        }
        # Continuous features
        derived = {
            "gmtf_score": snapshot.gmtf_score,
            "money_velocity_signal": snapshot.money_velocity_signal,
        }
        features = {}
        features.update(regime_features)
        features.update(cube_features)
        features.update(derived)
        return features

    def get_feature_names(self) -> list[str]:
        """Return list of feature names from last build."""
        return list(self._feature_names)

    def feature_count(self) -> int:
        """Return number of features generated."""
        return len(self._feature_names)


# ═══════════════════════════════════════════════════════════════════════════
# GMTF MODULE 1: MonetaryTensionIndex
# ═══════════════════════════════════════════════════════════════════════════
class MonetaryTensionIndex:
    """SDR-weighted monetary tension index across G5 currencies.

    Tracks monetary conditions across USD, EUR, JPY, GBP, CNY using
    IMF SDR basket weights. Produces a composite tension score from
    rate differentials and FX moves.
    """

    # IMF SDR basket weights (2022 review)
    SDR_BASKET = {
        "USD": 0.4338,
        "EUR": 0.2931,
        "JPY": 0.0759,
        "GBP": 0.0744,
        "CNY": 0.1228,
    }

    # Tension thresholds
    EASING_THRESHOLD = -0.15
    TIGHTENING_THRESHOLD = 0.15

    def __init__(self):
        self._tension_score: float = 0.0
        self._currency_tensions: dict[str, float] = {}
        self._stance: str = "NEUTRAL"
        self._history: list[dict] = []

    def compute_tension(
        self,
        rate_differentials: dict[str, float],
        fx_moves: dict[str, float],
    ) -> float:
        """Compute SDR-weighted monetary tension score.

        Args:
            rate_differentials: {currency: rate_change} — positive means
                tightening (rate hikes or hawkish shift) for each currency.
            fx_moves: {currency: pct_change} — positive means appreciation
                vs trade-weighted basket.

        Returns:
            Weighted tension score in [-1, +1] range.
            Positive = global tightening, Negative = global easing.
        """
        tension = 0.0
        weight_sum = 0.0

        for ccy, weight in self.SDR_BASKET.items():
            rate_d = rate_differentials.get(ccy, 0.0)
            fx_d = fx_moves.get(ccy, 0.0)

            # Rate differential drives 70% of tension, FX moves 30%
            # Positive rate_d = tightening; appreciation (+ fx_d) = tightening
            ccy_tension = rate_d * 0.70 + fx_d * 0.30

            self._currency_tensions[ccy] = round(ccy_tension, 6)
            tension += ccy_tension * weight
            weight_sum += weight

        if weight_sum > 0:
            tension /= weight_sum

        # Clip to [-1, 1]
        self._tension_score = float(np.clip(tension, -1.0, 1.0))

        # Determine stance
        if self._tension_score <= self.EASING_THRESHOLD:
            self._stance = "EASING"
        elif self._tension_score >= self.TIGHTENING_THRESHOLD:
            self._stance = "TIGHTENING"
        else:
            self._stance = "NEUTRAL"

        self._history.append({
            "tension_score": self._tension_score,
            "stance": self._stance,
            "currency_tensions": dict(self._currency_tensions),
        })

        return self._tension_score

    def get_global_easing_tightening(self) -> str:
        """Return global monetary stance: EASING / NEUTRAL / TIGHTENING."""
        return self._stance

    def get_currency_tensions(self) -> dict[str, float]:
        """Return per-currency tension breakdown."""
        return dict(self._currency_tensions)

    def get_state(self) -> dict:
        """Return full tension index state."""
        return {
            "tension_score": round(self._tension_score, 6),
            "stance": self._stance,
            "currency_tensions": {k: round(v, 6) for k, v in self._currency_tensions.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# GMTF MODULE 2: SectorRotationEngine
# ═══════════════════════════════════════════════════════════════════════════
class SectorRotationEngine:
    """GICS sector rotation engine based on macro regime cycle.

    Models sector favourability across four macro cycle phases:
        EARLY_CYCLE  — recovery from recession
        MID_CYCLE    — expansion / steady growth
        LATE_CYCLE   — overheating / peak
        RECESSION    — contraction

    Pre-built rotation matrix covers all 11 GICS sectors.
    """

    # The 11 GICS sectors
    GICS_SECTORS = [
        "Energy",
        "Materials",
        "Industrials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Health Care",
        "Financials",
        "Information Technology",
        "Communication Services",
        "Utilities",
        "Real Estate",
    ]

    # Macro cycle phases
    CYCLE_PHASES = ["EARLY_CYCLE", "MID_CYCLE", "LATE_CYCLE", "RECESSION"]

    # Rotation matrix: {sector: {phase: favourability_score}}
    # Scores in [-1.0, +1.0] where +1.0 = strong overweight, -1.0 = strong underweight
    ROTATION_MATRIX = {
        "Energy": {
            "EARLY_CYCLE": 0.3, "MID_CYCLE": 0.5, "LATE_CYCLE": 0.8, "RECESSION": -0.4,
        },
        "Materials": {
            "EARLY_CYCLE": 0.6, "MID_CYCLE": 0.4, "LATE_CYCLE": 0.5, "RECESSION": -0.5,
        },
        "Industrials": {
            "EARLY_CYCLE": 0.8, "MID_CYCLE": 0.6, "LATE_CYCLE": 0.2, "RECESSION": -0.3,
        },
        "Consumer Discretionary": {
            "EARLY_CYCLE": 0.9, "MID_CYCLE": 0.5, "LATE_CYCLE": -0.2, "RECESSION": -0.6,
        },
        "Consumer Staples": {
            "EARLY_CYCLE": -0.2, "MID_CYCLE": 0.0, "LATE_CYCLE": 0.3, "RECESSION": 0.8,
        },
        "Health Care": {
            "EARLY_CYCLE": 0.1, "MID_CYCLE": 0.3, "LATE_CYCLE": 0.5, "RECESSION": 0.7,
        },
        "Financials": {
            "EARLY_CYCLE": 0.7, "MID_CYCLE": 0.6, "LATE_CYCLE": 0.1, "RECESSION": -0.5,
        },
        "Information Technology": {
            "EARLY_CYCLE": 0.7, "MID_CYCLE": 0.8, "LATE_CYCLE": 0.0, "RECESSION": -0.3,
        },
        "Communication Services": {
            "EARLY_CYCLE": 0.5, "MID_CYCLE": 0.6, "LATE_CYCLE": 0.1, "RECESSION": -0.1,
        },
        "Utilities": {
            "EARLY_CYCLE": -0.4, "MID_CYCLE": -0.2, "LATE_CYCLE": 0.2, "RECESSION": 0.9,
        },
        "Real Estate": {
            "EARLY_CYCLE": 0.6, "MID_CYCLE": 0.3, "LATE_CYCLE": -0.3, "RECESSION": 0.2,
        },
    }

    # Overweight threshold
    OVERWEIGHT_THRESHOLD = 0.3
    UNDERWEIGHT_THRESHOLD = -0.2

    def __init__(self):
        self._current_phase: str = "MID_CYCLE"

    def _resolve_phase(self, regime: str) -> str:
        """Map a MarketRegime or free-form string to a cycle phase.

        Accepts MarketRegime values or direct cycle phase names.
        """
        regime_upper = regime.upper() if isinstance(regime, str) else str(regime).upper()

        # Direct phase names
        if regime_upper in self.CYCLE_PHASES:
            return regime_upper

        # Map MarketRegime values to cycle phases
        mapping = {
            "BULL": "MID_CYCLE",
            "BEAR": "LATE_CYCLE",
            "TRANSITION": "EARLY_CYCLE",
            "STRESS": "RECESSION",
            "CRASH": "RECESSION",
            "TRENDING": "MID_CYCLE",
            "RANGE": "LATE_CYCLE",
        }
        return mapping.get(regime_upper, "MID_CYCLE")

    def get_rotation_signal(self, regime: str, sector: str) -> float:
        """Get favourability score for a sector in a given regime/phase.

        Args:
            regime: Cycle phase name or MarketRegime value.
            sector: GICS sector name.

        Returns:
            Favourability score in [-1.0, +1.0].
        """
        phase = self._resolve_phase(regime)
        self._current_phase = phase

        sector_scores = self.ROTATION_MATRIX.get(sector)
        if sector_scores is None:
            return 0.0
        return sector_scores.get(phase, 0.0)

    def get_recommended_overweights(self, regime: str) -> List[str]:
        """Return list of sectors to overweight for the given regime/phase.

        Sectors with favourability >= OVERWEIGHT_THRESHOLD.
        """
        phase = self._resolve_phase(regime)
        self._current_phase = phase

        overweights = []
        for sector in self.GICS_SECTORS:
            score = self.ROTATION_MATRIX[sector].get(phase, 0.0)
            if score >= self.OVERWEIGHT_THRESHOLD:
                overweights.append(sector)
        return overweights

    def get_recommended_underweights(self, regime: str) -> List[str]:
        """Return list of sectors to underweight for the given regime/phase.

        Sectors with favourability <= UNDERWEIGHT_THRESHOLD.
        """
        phase = self._resolve_phase(regime)
        self._current_phase = phase

        underweights = []
        for sector in self.GICS_SECTORS:
            score = self.ROTATION_MATRIX[sector].get(phase, 0.0)
            if score <= self.UNDERWEIGHT_THRESHOLD:
                underweights.append(sector)
        return underweights

    def get_full_rotation(self, regime: str) -> dict[str, float]:
        """Return full rotation scores for all sectors in a regime/phase.

        Returns dict sorted by favourability descending.
        """
        phase = self._resolve_phase(regime)
        self._current_phase = phase

        scores = {}
        for sector in self.GICS_SECTORS:
            scores[sector] = self.ROTATION_MATRIX[sector].get(phase, 0.0)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ═══════════════════════════════════════════════════════════════════════════
# GMTF MODULE 3: MoneyVelocityEngine
# ═══════════════════════════════════════════════════════════════════════════
class MoneyVelocityEngine:
    """Money velocity engine: V = GDP / M2 proxy computation.

    Computes velocity proxy, classifies velocity regime, and estimates
    velocity-to-inflation transmission signal.
    """

    # Velocity regime thresholds (rate of change)
    ACCEL_THRESHOLD = 0.02    # >2% velocity growth = accelerating
    DECEL_THRESHOLD = -0.02   # <-2% velocity growth = decelerating

    # Inflation transmission parameters
    _INFLATION_SENSITIVITY = 0.6   # elasticity of inflation to velocity
    _INFLATION_LAG_QUARTERS = 2    # velocity leads inflation by ~2 quarters

    def __init__(
        self,
        gdp_proxy: Optional[float] = None,
        m2_proxy: Optional[float] = None,
    ):
        self._gdp: float = gdp_proxy if gdp_proxy is not None else 25_000.0
        self._m2: float = m2_proxy if m2_proxy is not None else 21_000.0
        self._velocity: float = self._gdp / max(self._m2, 1e-8)
        self._prev_velocity: float = self._velocity
        self._velocity_change: float = 0.0
        self._regime: str = "STABLE"
        self._inflation_signal: float = 0.0
        self._history: list[dict] = []

    def compute_velocity_proxy(
        self,
        gdp_proxy: Optional[float] = None,
        m2_proxy: Optional[float] = None,
    ) -> float:
        """Compute V = GDP / M2 proxy.

        Args:
            gdp_proxy: Nominal GDP proxy value. If None, uses stored value.
            m2_proxy: M2 money supply proxy value. If None, uses stored value.

        Returns:
            Current velocity proxy.
        """
        if gdp_proxy is not None:
            self._gdp = gdp_proxy
        if m2_proxy is not None:
            self._m2 = m2_proxy

        if self._m2 <= 0:
            return self._velocity

        self._prev_velocity = self._velocity
        self._velocity = self._gdp / self._m2

        # Compute rate of change
        if self._prev_velocity > 0:
            self._velocity_change = (
                (self._velocity - self._prev_velocity) / self._prev_velocity
            )
        else:
            self._velocity_change = 0.0

        # Update regime
        self._regime = self._classify_regime()

        # Update inflation signal
        self._inflation_signal = self._compute_inflation_signal()

        self._history.append({
            "velocity": self._velocity,
            "velocity_change": self._velocity_change,
            "regime": self._regime,
            "inflation_signal": self._inflation_signal,
        })

        return self._velocity

    def get_velocity_regime(self) -> str:
        """Return velocity regime: ACCELERATING / STABLE / DECELERATING."""
        return self._regime

    def get_inflation_signal(self) -> float:
        """Return velocity-to-inflation transmission signal.

        Positive = inflationary pressure from rising velocity.
        Negative = disinflationary pressure from falling velocity.
        Range approximately [-1, +1].
        """
        return self._inflation_signal

    def _classify_regime(self) -> str:
        """Classify velocity regime based on rate of change."""
        if self._velocity_change >= self.ACCEL_THRESHOLD:
            return "ACCELERATING"
        elif self._velocity_change <= self.DECEL_THRESHOLD:
            return "DECELERATING"
        else:
            return "STABLE"

    def _compute_inflation_signal(self) -> float:
        """Compute inflation transmission from velocity change.

        Uses sigmoid-shaped transmission function to model the
        non-linear relationship between velocity and inflation.
        """
        # Sigmoid transformation of velocity change
        raw_signal = self._velocity_change * self._INFLATION_SENSITIVITY * 10.0
        # Apply sigmoid to bound in [-1, +1]
        signal = 2.0 / (1.0 + np.exp(-raw_signal)) - 1.0
        return float(np.clip(signal, -1.0, 1.0))

    def get_state(self) -> dict:
        """Return full velocity engine state."""
        return {
            "velocity": round(self._velocity, 6),
            "velocity_change": round(self._velocity_change, 6),
            "regime": self._regime,
            "inflation_signal": round(self._inflation_signal, 6),
            "gdp_proxy": round(self._gdp, 2),
            "m2_proxy": round(self._m2, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# GMTF MODULE 4: FedReserveIntegration
# ═══════════════════════════════════════════════════════════════════════════
class FedReserveIntegration:
    """Federal Reserve balance sheet integration.

    Tracks Fed balance sheet proxies:
        - Total assets (WALCL proxy)
        - Reverse repo (RRP / ON RRP)
        - Treasury General Account (TGA)

    Computes net liquidity = Fed Assets - TGA - RRP
    and liquidity impulse (delta net liquidity).
    """

    # Default baseline values — refreshed from FRED on init, fallback to 2024 estimates
    _DEFAULT_ASSETS = 7_700.0
    _DEFAULT_RRP = 500.0
    _DEFAULT_TGA = 750.0

    def __init__(
        self,
        fed_assets: Optional[float] = None,
        rrp: Optional[float] = None,
        tga: Optional[float] = None,
    ):
        # Try live FRED data first, then passed values, then defaults
        live = self._fetch_live_fed_data()
        self._fed_assets: float = fed_assets or live.get("assets", self._DEFAULT_ASSETS)
        self._rrp: float = rrp or live.get("rrp", self._DEFAULT_RRP)
        self._tga: float = tga or live.get("tga", self._DEFAULT_TGA)
        self._prev_net_liquidity: float = self._compute_raw_net_liquidity()
        self._net_liquidity: float = self._prev_net_liquidity
        self._liquidity_impulse: float = 0.0
        self._history: list[dict] = []

    @staticmethod
    def _fetch_live_fed_data() -> dict:
        """Fetch latest Fed balance sheet data from FRED.

        WALCL = Fed total assets
        RRPONTSYD = ON-RRP
        WTREGEN = Treasury General Account
        """
        result = {}
        try:
            from ..data.openbb_data import get_fred_series
            # Fed total assets (WALCL, weekly, in millions)
            walcl = get_fred_series("WALCL", start="2024-01-01")
            if hasattr(walcl, "empty") and not walcl.empty:
                cols = walcl.select_dtypes(include=["number"]).columns
                if len(cols) > 0:
                    val = float(walcl[cols[0]].dropna().iloc[-1])
                    if val > 0:
                        result["assets"] = val / 1000  # millions → billions
                        logger.debug("Fed assets from FRED: $%.0fB", result["assets"])

            # ON-RRP (RRPONTSYD, daily, in billions)
            rrp = get_fred_series("RRPONTSYD", start="2024-01-01")
            if hasattr(rrp, "empty") and not rrp.empty:
                cols = rrp.select_dtypes(include=["number"]).columns
                if len(cols) > 0:
                    val = float(rrp[cols[0]].dropna().iloc[-1])
                    if val >= 0:
                        result["rrp"] = val / 1000 if val > 10000 else val  # normalize
                        logger.debug("ON-RRP from FRED: $%.0fB", result["rrp"])

            # TGA (WTREGEN, weekly, in millions)
            tga = get_fred_series("WTREGEN", start="2024-01-01")
            if hasattr(tga, "empty") and not tga.empty:
                cols = tga.select_dtypes(include=["number"]).columns
                if len(cols) > 0:
                    val = float(tga[cols[0]].dropna().iloc[-1])
                    if val > 0:
                        result["tga"] = val / 1000  # millions → billions
                        logger.debug("TGA from FRED: $%.0fB", result["tga"])
        except Exception as e:
            logger.debug("Fed live data fetch failed: %s — using defaults", e)
        return result

    def _compute_raw_net_liquidity(self) -> float:
        """Internal: compute net liquidity from current state."""
        return self._fed_assets - self._tga - self._rrp

    def update(
        self,
        fed_assets: Optional[float] = None,
        rrp: Optional[float] = None,
        tga: Optional[float] = None,
    ) -> None:
        """Update Fed balance sheet components.

        Args:
            fed_assets: Total Fed assets ($ billions).
            rrp: Reverse repo facility balance ($ billions).
            tga: Treasury General Account balance ($ billions).
        """
        if fed_assets is not None:
            self._fed_assets = fed_assets
        if rrp is not None:
            self._rrp = rrp
        if tga is not None:
            self._tga = tga

        self._prev_net_liquidity = self._net_liquidity
        self._net_liquidity = self._compute_raw_net_liquidity()
        self._liquidity_impulse = self._net_liquidity - self._prev_net_liquidity

        self._history.append({
            "fed_assets": self._fed_assets,
            "rrp": self._rrp,
            "tga": self._tga,
            "net_liquidity": self._net_liquidity,
            "liquidity_impulse": self._liquidity_impulse,
        })

    def compute_net_liquidity(self) -> float:
        """Compute net liquidity = Fed Assets - TGA - RRP.

        Returns:
            Net liquidity in $ billions.
        """
        self._net_liquidity = self._compute_raw_net_liquidity()
        return self._net_liquidity

    def get_liquidity_impulse(self) -> float:
        """Get delta of net liquidity (current - previous).

        Positive impulse = liquidity injection (bullish for risk assets).
        Negative impulse = liquidity drain (bearish for risk assets).

        Returns:
            Liquidity impulse in $ billions.
        """
        return self._liquidity_impulse

    def format_fed_dashboard(self) -> str:
        """Generate ASCII dashboard of Fed balance sheet state.

        Returns:
            Multi-line ASCII string with Fed balance sheet summary.
        """
        net_liq = self.compute_net_liquidity()
        impulse = self._liquidity_impulse
        impulse_dir = "+" if impulse >= 0 else ""

        # Determine liquidity regime
        if impulse > 50:
            regime = "INJECTING"
        elif impulse > 0:
            regime = "MILDLY INJECTING"
        elif impulse > -50:
            regime = "MILDLY DRAINING"
        else:
            regime = "DRAINING"

        lines = [
            "=" * 60,
            "         FEDERAL RESERVE LIQUIDITY DASHBOARD",
            "=" * 60,
            "",
            f"  Fed Total Assets:       ${self._fed_assets:>10,.1f}B",
            f"  Reverse Repo (RRP):     ${self._rrp:>10,.1f}B",
            f"  Treasury Gen Acct (TGA):${self._tga:>10,.1f}B",
            "-" * 60,
            f"  NET LIQUIDITY:          ${net_liq:>10,.1f}B",
            f"  Liquidity Impulse:      {impulse_dir}{impulse:>9,.1f}B",
            f"  Liquidity Regime:        {regime}",
            "",
            "  Formula: Net Liq = Fed Assets - TGA - RRP",
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_state(self) -> dict:
        """Return full Fed integration state."""
        return {
            "fed_assets": round(self._fed_assets, 2),
            "rrp": round(self._rrp, 2),
            "tga": round(self._tga, 2),
            "net_liquidity": round(self._net_liquidity, 2),
            "liquidity_impulse": round(self._liquidity_impulse, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# MacroEngine class (enhanced)
# ═══════════════════════════════════════════════════════════════════════════
class MacroEngine:
    """Macro regime classification and sector ranking engine.

    Enhanced with money velocity, credit pulse, yield curve analysis,
    liquidity scoring, and ML feature generation.
    """

    def __init__(self):
        self._snapshot: Optional[MacroSnapshot] = None
        self._money_velocity = MoneyVelocityModule()
        self._sector_ranker = SectorRanker()
        self._ctv = CarryToVolatility()
        self._regime_detector = RegimeTransitionDetector()
        self._yield_analyzer = YieldCurveAnalyzer()
        self._credit_monitor = CreditPulseMonitor()
        self._feature_builder = MacroFeatureBuilder()
        self._macro_data: Optional[pd.DataFrame] = None
        # GMTF (monthly FRED data → recompute at most once per day)
        self._gmtf_output: Optional[GMTFOutput] = None
        self._gmtf_date = None
        # Gaussian HMM regime model (refit once per day)
        self._hmm = None
        self._hmm_fit_date = None

    def analyze(self, lookback_days: int = 252) -> MacroSnapshot:
        """Run full macro analysis using OpenBB data."""
        snapshot = MacroSnapshot()

        try:
            start = pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)
            start_str = start.strftime("%Y-%m-%d")

            # Fetch core data
            macro = get_macro_data(start=start_str)
            if macro.empty:
                self._snapshot = snapshot
                return snapshot
            self._macro_data = macro

            # VIX
            if "VIX" in macro.columns:
                snapshot.vix = float(macro["VIX"].dropna().iloc[-1])

            # SPY returns
            if "S&P 500" in macro.columns:
                spy = macro["S&P 500"].dropna()
                if len(spy) > 21:
                    snapshot.spy_return_1m = float(spy.iloc[-1] / spy.iloc[-22] - 1)
                if len(spy) > 63:
                    snapshot.spy_return_3m = float(spy.iloc[-1] / spy.iloc[-64] - 1)

            # Yields
            if "10Y Yield" in macro.columns:
                snapshot.yield_10y = float(macro["10Y Yield"].dropna().iloc[-1])
            if "5Y Yield" in macro.columns:
                snapshot.yield_2y = float(macro["5Y Yield"].dropna().iloc[-1])
            snapshot.yield_spread = snapshot.yield_10y - snapshot.yield_2y

            # Credit spread (HY - IG proxy)
            if "HY Corporate" in macro.columns and "IG Corporate" in macro.columns:
                hy_ret = macro["HY Corporate"].pct_change().rolling(20).std() * np.sqrt(252)
                ig_ret = macro["IG Corporate"].pct_change().rolling(20).std() * np.sqrt(252)
                spread = (hy_ret - ig_ret).dropna()
                if len(spread) > 0:
                    snapshot.credit_spread = float(spread.iloc[-1]) * 100

            # Gold momentum
            if "Gold" in macro.columns:
                gold = macro["Gold"].dropna()
                if len(gold) > 63:
                    snapshot.gold_momentum = float(gold.iloc[-1] / gold.iloc[-64] - 1)

            # Classify regime
            snapshot.regime = self._classify_regime(snapshot)
            snapshot.cube_regime = self._classify_cube_regime(snapshot)

            # Rank sectors
            snapshot.sector_rankings = self._rank_sectors(snapshot, start_str)

            # --- Enhanced analysis ---
            # Money velocity
            self._compute_money_velocity(macro, snapshot)

            # GMTF — Global Monetary Tension Framework (4 gammas, FRED data)
            self._compute_gmtf(snapshot)

            # HMM regime (fitted Gaussian HMM on SPY / VIX / credit)
            self._compute_hmm_regime(macro, snapshot)

            # Credit pulse
            self._compute_credit_pulse(macro)

            # Yield curve analysis
            self._compute_yield_curve(snapshot)

            # Regime transition detection
            self._regime_detector.update(snapshot.regime)

        except Exception as e:
            logger.warning(f"MacroEngine analyze failed: {e}")
            snapshot.regime = MarketRegime.TRANSITION

        self._snapshot = snapshot
        return snapshot

    def get_snapshot(self) -> MacroSnapshot:
        if self._snapshot is None:
            return self.analyze()
        return self._snapshot

    # --- Regime classification -----------------------------------------------

    def _classify_regime(self, snap: MacroSnapshot) -> MarketRegime:
        """Rule-based regime classification."""
        score = 0

        # VIX component
        if snap.vix > 35:
            score -= 3
        elif snap.vix > 25:
            score -= 1
        elif snap.vix < 15:
            score += 2
        else:
            score += 1

        # SPY momentum
        if snap.spy_return_3m > 0.10:
            score += 3
        elif snap.spy_return_3m > 0.03:
            score += 1
        elif snap.spy_return_3m < -0.10:
            score -= 3
        elif snap.spy_return_3m < -0.03:
            score -= 1

        # Yield curve
        if snap.yield_spread > 0.5:
            score += 1
        elif snap.yield_spread < -0.5:
            score -= 1

        # Credit
        if snap.credit_spread > 5:
            score -= 2

        if score >= 4:
            return MarketRegime.BULL
        elif score >= 1:
            return MarketRegime.TRANSITION
        elif score >= -2:
            return MarketRegime.BEAR
        else:
            return MarketRegime.STRESS

    def _classify_cube_regime(self, snap: MacroSnapshot) -> CubeRegime:
        """Map market regime to cube regime for MetadronCube."""
        if snap.vix > 40:
            return CubeRegime.CRASH
        elif snap.regime == MarketRegime.STRESS:
            return CubeRegime.STRESS
        elif snap.regime == MarketRegime.BULL:
            return CubeRegime.TRENDING
        else:
            return CubeRegime.RANGE

    # --- Sector ranking ------------------------------------------------------

    def _rank_sectors(self, snap: MacroSnapshot, start_str: str) -> dict[str, float]:
        """Rank sectors by macro favourability."""
        from ..data.universe_engine import SECTOR_ETFS
        etfs = list(SECTOR_ETFS.values())
        inv = {v: k for k, v in SECTOR_ETFS.items()}

        try:
            returns = get_returns(etfs, start=start_str)
            if returns.empty:
                return {}
        except Exception:
            return {}

        rankings = {}
        for etf in returns.columns:
            sector = inv.get(etf, etf)
            r = returns[etf].dropna()
            if len(r) < 21:
                continue

            # Composite score: momentum + risk-adjusted
            mom_1m = float(r.iloc[-21:].sum())
            mom_3m = float(r.iloc[-63:].sum()) if len(r) >= 63 else mom_1m
            vol = float(r.std() * np.sqrt(252))
            sharpe = (mom_3m * 4) / vol if vol > 0 else 0

            # Regime adjustment
            if snap.regime == MarketRegime.STRESS:
                # Favour defensives
                if sector in ["Utilities", "Consumer Staples", "Health Care"]:
                    sharpe *= 1.3
                elif sector in ["Consumer Discretionary", "Information Technology"]:
                    sharpe *= 0.7
            elif snap.regime == MarketRegime.BULL:
                if sector in ["Information Technology", "Consumer Discretionary", "Communication Services"]:
                    sharpe *= 1.2

            rankings[sector] = round(sharpe, 4)

        # Sort descending
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))

    # --- Enhanced analysis methods -------------------------------------------

    def _compute_money_velocity(
        self, macro: pd.DataFrame, snapshot: MacroSnapshot,
    ) -> None:
        """Compute money velocity signals and update snapshot."""
        try:
            # Use S&P 500 as GDP proxy and Gold as M2/monetary base proxy
            gdp_proxy = macro.get("S&P 500") if "S&P 500" in macro.columns else None
            m2_proxy = macro.get("Gold") if "Gold" in macro.columns else None

            if gdp_proxy is not None and m2_proxy is not None:
                self._money_velocity.compute_velocity(gdp_proxy, m2_proxy)

            # Credit impulse from IG corporate bond proxy
            credit_proxy = macro.get("IG Corporate") if "IG Corporate" in macro.columns else None
            if credit_proxy is not None and gdp_proxy is not None:
                self._money_velocity.compute_credit_impulse(credit_proxy, gdp_proxy)

            # Compute liquidity score
            liq_score = self._money_velocity.compute_liquidity_score(
                vix=snapshot.vix,
                credit_spread=snapshot.credit_spread,
                yield_spread=snapshot.yield_spread,
            )
            snapshot.money_velocity_signal = self._money_velocity.velocity_change
            # Fallback only — replaced by the real GMTF in _compute_gmtf()
            snapshot.gmtf_score = liq_score / 100.0  # Normalise to 0-1
            snapshot.gmtf_source = "liquidity_proxy"

        except Exception:
            pass

    def _compute_gmtf(self, snapshot: MacroSnapshot, inputs: Optional[dict] = None) -> None:
        """Run the real GMTF and put its tension percentile on the snapshot.

        gmtf_score = percentile rank (0..1) of today's smoothed SDR-weighted
        tension within its own history: 0 = loosest, 1 = tightest on record.
        Leaves the liquidity-proxy value in place (gmtf_source stays
        "liquidity_proxy") when FRED data is unavailable.
        """
        try:
            today = pd.Timestamp.now().normalize()
            out = self._gmtf_output if (inputs is None and self._gmtf_date == today) else None
            if out is None:
                inputs = inputs if inputs is not None else build_gmtf_inputs()
                if not inputs:
                    return
                out = compute_gmtf(inputs["m2"], inputs["gdp"], inputs["unemp"],
                                   inputs["fx"], inputs["rates"])
                out.ctv_ratios = None
                self._gmtf_output, self._gmtf_date = out, today
                self._gmtf_currencies = list(inputs["m2"].columns)
            series = out.gmtf_smoothed.replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) < 12:
                return
            latest = float(series.iloc[-1])
            snapshot.gmtf_raw = latest
            snapshot.gmtf_score = float((series <= latest).mean())
            snapshot.gmtf_source = "gmtf"
            snapshot.gmtf_currencies = list(getattr(self, "_gmtf_currencies", []))
            snapshot.gmtf_gammas = {
                "liquidity": round(out.gamma_liquidity, 4),
                "fx_shock": round(out.gamma_fx, 4),
                "wage": round(out.gamma_wage, 4),
                "reserve": round(out.gamma_reserve, 4),
            }
        except Exception as exc:
            logger.warning("GMTF computation failed (liquidity proxy kept): %s", exc)

    def _compute_hmm_regime(self, macro: pd.DataFrame, snapshot: MacroSnapshot) -> None:
        """Fit (daily) and apply the Gaussian HMM regime model."""
        try:
            if "S&P 500" not in macro.columns:
                return
            from ..ml.bridges.markov_regime_bridge import MarkovRegimeBridge, REGIME_NAMES
            X = self._hmm_features(macro)
            if X is None or len(X) < 60:
                return
            today = pd.Timestamp.now().normalize()
            if self._hmm is None or self._hmm_fit_date != today or not self._hmm.fitted:
                bridge = MarkovRegimeBridge(n_regimes=4)
                cols = [X[:, i] for i in range(X.shape[1])]
                fit = bridge.fit(cols[0], cols[1], vix=cols[2] if X.shape[1] > 2 else None)
                if "error" in fit:
                    logger.debug("HMM fit skipped: %s", fit["error"])
                self._hmm, self._hmm_fit_date = bridge, today
            pred = self._hmm.predict_regime_sequence(X[-60:])
            snapshot.hmm_regime = pred.regime
            snapshot.hmm_confidence = float(pred.confidence)
            snapshot.hmm_probs = {r: round(float(p), 4)
                                  for r, p in zip(REGIME_NAMES, pred.state_probabilities)}
        except Exception as exc:
            logger.warning("HMM regime failed: %s", exc)

    @staticmethod
    def _hmm_features(macro: pd.DataFrame) -> Optional[np.ndarray]:
        """[daily return, 20d realised vol (daily), VIX/100] rows, NaNs dropped."""
        spy = macro["S&P 500"].astype(float)
        ret = spy.pct_change()
        vol = ret.rolling(20).std()
        feats = [ret, vol]
        if "VIX" in macro.columns:
            feats.append(macro["VIX"].astype(float) / 100.0)
        df = pd.concat(feats, axis=1).dropna()
        return df.values if len(df) else None

    def _compute_credit_pulse(self, macro: pd.DataFrame) -> None:
        """Run credit pulse monitor on latest data."""
        try:
            hy = macro.get("HY Corporate") if "HY Corporate" in macro.columns else None
            ig = macro.get("IG Corporate") if "IG Corporate" in macro.columns else None
            if hy is not None and ig is not None:
                self._credit_monitor.compute_from_etfs(hy, ig)
        except Exception:
            pass

    def _compute_yield_curve(self, snapshot: MacroSnapshot) -> None:
        """Run yield curve analysis."""
        try:
            self._yield_analyzer.analyze(
                yield_10y=snapshot.yield_10y,
                yield_2y=snapshot.yield_2y,
            )
        except Exception:
            pass

    def compute_liquidity_score(self) -> float:
        """Compute and return liquidity score 0-100.

        Uses the MoneyVelocityModule to produce a composite score
        aggregating velocity, credit impulse, TED spread, VIX, and yield curve.
        """
        snap = self._snapshot or MacroSnapshot()
        return self._money_velocity.compute_liquidity_score(
            vix=snap.vix,
            credit_spread=snap.credit_spread,
            yield_spread=snap.yield_spread,
        )

    def get_macro_features(self) -> dict[str, float]:
        """Return macro feature dict for ML models.

        Builds 50+ features from macro data and current snapshot.
        """
        macro = self._macro_data
        if macro is None:
            # Return minimal features from snapshot
            snap = self._snapshot or MacroSnapshot()
            return self._feature_builder._snapshot_features(snap)

        return self._feature_builder.build_features(
            macro, self._snapshot,
        )

    def get_yield_curve_analysis(self) -> dict:
        """Return latest yield curve analysis."""
        snap = self._snapshot or MacroSnapshot()
        return self._yield_analyzer.analyze(
            yield_10y=snap.yield_10y,
            yield_2y=snap.yield_2y,
        )

    def get_credit_pulse(self) -> dict:
        """Return latest credit pulse state."""
        return self._credit_monitor.get_state()

    def get_money_velocity_state(self) -> dict:
        """Return current money velocity module state."""
        return self._money_velocity.get_state()

    def get_regime_transition(self) -> dict:
        """Return latest regime transition detection result."""
        snap = self._snapshot or MacroSnapshot()
        return self._regime_detector.update(snap.regime)

    def get_regime_distribution(self) -> dict[str, float]:
        """Return regime frequency distribution from recent history."""
        return self._regime_detector.get_regime_distribution()

    # --- GMTF Enhancement methods -------------------------------------------

    def get_monetary_tension(self) -> dict:
        """Compute and return SDR-weighted monetary tension index.

        Uses the MonetaryTensionIndex to produce a composite tension
        score from rate differentials and FX moves. When live data is
        unavailable, returns proxy values from the current snapshot.

        Returns:
            Dict with tension_score, stance, and currency_tensions.
        """
        mti = MonetaryTensionIndex()
        snap = self._snapshot or MacroSnapshot()

        # Build rate differentials from available data
        # Use yield spread as a proxy for USD tightening
        rate_diffs = {
            "USD": snap.yield_spread * 0.1,   # normalise spread
            "EUR": -snap.yield_spread * 0.05,  # inverse proxy
            "JPY": -0.02,                      # BOJ perpetual easing proxy
            "GBP": snap.yield_spread * 0.03,   # correlated to USD
            "CNY": -0.01,                      # managed regime proxy
        }

        # FX moves proxy from gold momentum (dollar weakness indicator)
        gold_signal = snap.gold_momentum
        fx_moves = {
            "USD": -gold_signal * 0.5,    # gold up → dollar down
            "EUR": gold_signal * 0.3,
            "JPY": gold_signal * 0.1,
            "GBP": gold_signal * 0.2,
            "CNY": gold_signal * 0.05,
        }

        mti.compute_tension(rate_diffs, fx_moves)
        return mti.get_state()

    def get_rotation_signals(self) -> dict:
        """Get GICS sector rotation signals for current regime.

        Uses the SectorRotationEngine to produce favourability scores
        and over/underweight recommendations based on the current
        macro regime.

        Returns:
            Dict with full_rotation, overweights, and underweights.
        """
        sre = SectorRotationEngine()
        snap = self._snapshot or MacroSnapshot()
        regime_str = snap.regime.value

        full_rotation = sre.get_full_rotation(regime_str)
        overweights = sre.get_recommended_overweights(regime_str)
        underweights = sre.get_recommended_underweights(regime_str)

        return {
            "regime": regime_str,
            "cycle_phase": sre._current_phase,
            "full_rotation": full_rotation,
            "overweights": overweights,
            "underweights": underweights,
        }

    def get_velocity_regime(self) -> dict:
        """Get money velocity engine state.

        Uses the MoneyVelocityEngine to compute V = GDP/M2 proxy,
        velocity regime, and inflation signal.

        Returns:
            Dict with velocity, regime, and inflation_signal.
        """
        mve = MoneyVelocityEngine()
        snap = self._snapshot or MacroSnapshot()

        # Use SPY level and gold as GDP/M2 proxies
        # If macro_data is available, use last values
        if self._macro_data is not None:
            if "S&P 500" in self._macro_data.columns:
                spy_vals = self._macro_data["S&P 500"].dropna()
                if len(spy_vals) > 0:
                    mve._gdp = float(spy_vals.iloc[-1])
                    if len(spy_vals) > 21:
                        # Also set previous velocity from 21 days ago
                        prev_gdp = float(spy_vals.iloc[-22])
                        if "Gold" in self._macro_data.columns:
                            gold_vals = self._macro_data["Gold"].dropna()
                            if len(gold_vals) > 21:
                                prev_m2 = float(gold_vals.iloc[-22])
                                if prev_m2 > 0:
                                    mve._prev_velocity = prev_gdp / prev_m2

            if "Gold" in self._macro_data.columns:
                gold_vals = self._macro_data["Gold"].dropna()
                if len(gold_vals) > 0:
                    mve._m2 = float(gold_vals.iloc[-1])

        mve.compute_velocity_proxy()
        return mve.get_state()

    def get_fed_liquidity(self) -> dict:
        """Get Federal Reserve liquidity state.

        Uses the FedReserveIntegration to compute net liquidity
        and liquidity impulse from Fed balance sheet proxies.

        Returns:
            Dict with fed_assets, rrp, tga, net_liquidity, liquidity_impulse.
        """
        fri = FedReserveIntegration()
        return fri.get_state()

    def generate_macro_intelligence(self) -> str:
        """Generate comprehensive ASCII macro intelligence report.

        Combines all GMTF modules into a single formatted report:
            - Regime classification
            - Monetary tension index
            - Sector rotation signals
            - Money velocity
            - Fed liquidity dashboard
            - Yield curve analysis
            - Credit pulse

        Returns:
            Multi-line ASCII string with full macro intelligence.
        """
        snap = self._snapshot or MacroSnapshot()

        # Gather all components
        tension = self.get_monetary_tension()
        rotation = self.get_rotation_signals()
        velocity = self.get_velocity_regime()
        fed_liq = self.get_fed_liquidity()
        yc = self.get_yield_curve_analysis()
        credit = self.get_credit_pulse()
        mv_state = self.get_money_velocity_state()

        # Fed dashboard
        fri = FedReserveIntegration()
        fed_dashboard = fri.format_fed_dashboard()

        # Build overweight / underweight strings
        ow_str = ", ".join(rotation.get("overweights", [])) or "None"
        uw_str = ", ".join(rotation.get("underweights", [])) or "None"

        # Rotation table
        rotation_lines = []
        full_rot = rotation.get("full_rotation", {})
        for sector, score in full_rot.items():
            bar_len = int(abs(score) * 20)
            bar_char = "+" if score >= 0 else "-"
            bar = bar_char * bar_len
            rotation_lines.append(f"    {sector:<28s} {score:>+5.2f}  {bar}")

        rotation_table = "\n".join(rotation_lines)

        # Currency tension table
        ccy_tensions = tension.get("currency_tensions", {})
        ccy_lines = []
        for ccy, val in ccy_tensions.items():
            weight = MonetaryTensionIndex.SDR_BASKET.get(ccy, 0.0)
            ccy_lines.append(f"    {ccy}  (SDR {weight*100:5.2f}%)  tension: {val:>+8.6f}")
        ccy_table = "\n".join(ccy_lines)

        report = f"""
{'=' * 70}
          METADRON CAPITAL — MACRO INTELLIGENCE REPORT
{'=' * 70}

  REGIME CLASSIFICATION
  ---------------------
    Market Regime:    {snap.regime.value}
    Cube Regime:      {snap.cube_regime.value}
    VIX:              {snap.vix:.1f}
    SPY 3M Return:    {snap.spy_return_3m:+.2%}
    Yield Spread:     {snap.yield_spread:+.4f}
    Credit Spread:    {snap.credit_spread:.4f}

  MONETARY TENSION INDEX (SDR-Weighted)
  -------------------------------------
    Tension Score:    {tension.get('tension_score', 0.0):+.6f}
    Global Stance:    {tension.get('stance', 'N/A')}

    Per-Currency Breakdown:
{ccy_table}

  SECTOR ROTATION (GICS)
  ----------------------
    Cycle Phase:      {rotation.get('cycle_phase', 'N/A')}
    Overweights:      {ow_str}
    Underweights:     {uw_str}

    Favourability Scores:
{rotation_table}

  MONEY VELOCITY (V = GDP/M2)
  ---------------------------
    Velocity:         {velocity.get('velocity', 0.0):.6f}
    Velocity Change:  {velocity.get('velocity_change', 0.0):+.6f}
    Velocity Regime:  {velocity.get('regime', 'N/A')}
    Inflation Signal: {velocity.get('inflation_signal', 0.0):+.6f}

  LIQUIDITY SCORE
  ---------------
    Composite Score:  {mv_state.get('liquidity_score', 0.0):.2f} / 100
    Credit Impulse:   {mv_state.get('credit_impulse', 0.0):+.4f}
    TED Spread:       {mv_state.get('ted_spread', 0.0):.4f}

  YIELD CURVE ANALYSIS
  --------------------
    2s10s Spread:     {yc.get('spread_2s10s', 0.0):+.4f}
    3m10y Spread:     {yc.get('spread_3m10y', 0.0):+.4f}
    Term Premium:     {yc.get('term_premium', 0.0):+.4f}
    Real Rate (10Y):  {yc.get('real_rate_10y', 0.0):+.4f}
    Curve Shape:      {yc.get('curve_shape', 'N/A')}
    Inversion:        {'YES' if yc.get('inversion_signal') else 'NO'}

  CREDIT PULSE
  ------------
    HY-IG Spread:     {credit.get('hy_ig_spread', 0.0):.4f}
    Spread Z-Score:   {credit.get('spread_zscore', 0.0):+.4f}
    Credit Impulse:   {credit.get('credit_impulse', 0.0):+.4f}
    Stress Level:     {credit.get('stress_level', 'N/A')}

{fed_dashboard}

{'=' * 70}
"""
        return report
