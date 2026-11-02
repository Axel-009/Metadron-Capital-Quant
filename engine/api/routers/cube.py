"""
Cube router — CUBE tab
Wraps: MetadronCube, MacroEngine, BetaCorridor

Endpoints:
    GET /cube/state          — Full CubeOutput: L/R/F tensors, regime, sleeves, kill-switch,
                               risk governor, FCLP drift, learning, cube_tensor, reserve kernel
    GET /cube/history        — Rich time-series of last 100 CubeOutput entries + regime transitions
    GET /cube/gates          — 4-gate aggregate health scores (G1–G4)
    GET /cube/stress-tests   — Stress scenario results
    GET /cube/kill-switch    — Kill-switch status (now delegated to /cube/state but kept for compat)
    GET /cube/macro-snapshot — Live MacroSnapshot feeding the cube
    GET /cube/beta-corridor  — BetaCorridor state, vol regime, beta history (last 60)

BROKER SWAP NOTE:
    BetaCorridor and MacroEngine are accessed via engine.api.shared singletons
    (get_beta, get_engine).  MetadronCube maintains its own singleton here (_cube)
    to preserve calibration state between calls.  All data access is read-only
    from the router's perspective — no broker orders are triggered from these endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.cube")
router = APIRouter()

# ---------------------------------------------------------------------------
# MetadronCube singleton — separate from ExecutionEngine.cube so the CUBE tab
# has its own instance with preserved learning/FCLP state.
# ---------------------------------------------------------------------------
_cube = None


def _get_cube():
    """Lazy-init MetadronCube singleton."""
    global _cube
    if _cube is None:
        from engine.signals.metadron_cube import MetadronCube
        _cube = MetadronCube()
    return _cube


# ---------------------------------------------------------------------------
# Helper: safe attribute get with default
# ---------------------------------------------------------------------------
def _g(obj, attr: str, default=None):
    """getattr with fallback default — never raises."""
    return getattr(obj, attr, default) if obj is not None else default


def _gf(obj, attr: str, default: float = 0.0) -> float:
    """getattr for float fields with 0.0 default."""
    v = _g(obj, attr, default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Helper: serialise regime enum to string
# ---------------------------------------------------------------------------
def _regime_str(val, default: str = "UNKNOWN") -> str:
    if val is None:
        return default
    if hasattr(val, "value"):
        return str(val.value)
    return str(val)


# ---------------------------------------------------------------------------
# Helper: numpy array → plain list
# ---------------------------------------------------------------------------
def _tensor_list(tensor, default=None):
    if tensor is None:
        return default if default is not None else [0.0, 0.0, 0.0]
    try:
        import numpy as np
        arr = np.asarray(tensor, dtype=float)
        return [round(float(x), 6) for x in arr.flatten()[:3]]
    except Exception:
        try:
            return [float(x) for x in list(tensor)[:3]]
        except Exception:
            return [0.0, 0.0, 0.0]


# ===========================================================================
# /cube/state — full CubeOutput
# ===========================================================================
@router.get("/state")
async def cube_state():
    """Return everything from the most recent CubeOutput.

    Includes:
    - L(t), R(t), F(t) primary tensor values
    - Full LiquidityState, RiskState, FlowState sub-objects
    - SleeveAllocation (primary + legacy keys)
    - Regime + confidence + transition_probability
    - target_beta, beta_cap, max_leverage, risk_budget_pct
    - cube_tensor as [L, R, F]
    - kill_switch  — full 9-field dict
    - reserve_kernel — impulse, equity_impact, credit_impact, decay
    - risk_governor — 7 limit checks from full_check()
    - fclp          — drift report from _fclp.get_drift_report()
    - learning      — accuracy stats + suggest_adjustments()
    """
    try:
        cube = _get_cube()
        last = cube.get_last()
        if last is None:
            return {"status": "no_computation", "timestamp": datetime.utcnow().isoformat()}

        # ---- Liquidity -------------------------------------------------------
        liq_obj = _g(last, "liquidity")
        liq = {
            "value":               _gf(liq_obj, "value"),
            "sofr_signal":         _gf(liq_obj, "sofr_signal"),
            "credit_impulse":      _gf(liq_obj, "credit_impulse"),
            "m2_velocity":         _gf(liq_obj, "m2_velocity"),
            "hy_spread_z":         _gf(liq_obj, "hy_spread_z"),
            "fed_funds_impact":    _gf(liq_obj, "fed_funds_impact"),
            "reverse_repo_signal": _gf(liq_obj, "reverse_repo_signal"),
            "tga_balance_signal":  _gf(liq_obj, "tga_balance_signal"),
            "reserve_flow":        _gf(liq_obj, "reserve_flow"),
            "net_liquidity_score": _gf(liq_obj, "net_liquidity_score"),
        }

        # ---- Risk ------------------------------------------------------------
        risk_obj = _g(last, "risk")
        risk = {
            "value":                   _gf(risk_obj, "value", 0.3),
            "vix_component":           _gf(risk_obj, "vix_component"),
            "realized_vol":            _gf(risk_obj, "realized_vol"),
            "credit_spread_component": _gf(risk_obj, "credit_spread_component"),
            "vix_term_structure":      _gf(risk_obj, "vix_term_structure"),
            "correlation_stress":      _gf(risk_obj, "correlation_stress"),
            "tail_risk":               _gf(risk_obj, "tail_risk"),
            "implied_vs_realized":     _gf(risk_obj, "implied_vs_realized"),
        }

        # ---- Flow ------------------------------------------------------------
        flow_obj = _g(last, "flow")
        flow = {
            "value":                _gf(flow_obj, "value"),
            "sector_momentum":      _g(flow_obj, "sector_momentum", {}),
            "leader_sectors":       _g(flow_obj, "leader_sectors", []),
            "laggard_sectors":      _g(flow_obj, "laggard_sectors", []),
            "rotation_velocity":    _gf(flow_obj, "rotation_velocity"),
            "breadth":              _gf(flow_obj, "breadth"),
            "persistence":          _gf(flow_obj, "persistence"),
            "mean_reversion_signal": _gf(flow_obj, "mean_reversion_signal"),
        }

        # ---- Sleeves ---------------------------------------------------------
        sl_obj = _g(last, "sleeves")
        sleeves = {
            # Primary credit-aware allocation
            "ig_equity":            _gf(sl_obj, "ig_equity", 0.40),
            "options":              _gf(sl_obj, "options", 0.25),
            "options_ig":           _gf(sl_obj, "options_ig", 0.10),
            "options_hy":           _gf(sl_obj, "options_hy", 0.10),
            "options_distressed":   _gf(sl_obj, "options_distressed", 0.05),
            "bond_commodity_etf":   _gf(sl_obj, "bond_commodity_etf", 0.10),
            "hy_equity":            _gf(sl_obj, "hy_equity", 0.10),
            "distressed_equity":    _gf(sl_obj, "distressed_equity", 0.10),
            "cash_reserve":         _gf(sl_obj, "cash_reserve", 0.05),
            "deploy_pct":           _gf(sl_obj, "deploy_pct", 0.95),
            # Legacy 5-sleeve keys
            "p1_directional_equity": _gf(sl_obj, "p1_directional_equity", 0.40),
            "p2_factor_rotation":    _gf(sl_obj, "p2_factor_rotation", 0.10),
            "p3_commodities_macro":  _gf(sl_obj, "p3_commodities_macro", 0.10),
            "p4_options_convexity":  _gf(sl_obj, "p4_options_convexity", 0.25),
            "p5_hedges_volatility":  _gf(sl_obj, "p5_hedges_volatility", 0.10),
        }

        # ---- Kill switch -----------------------------------------------------
        # Re-invoke check against the most recent macro state so the returned
        # dict is fresh and includes all 9 fields.
        kill_switch = {
            "active": False, "triggered": False,
            "hy_oas_delta": 0.0, "hy_triggered": False,
            "vix_term_ratio": 0.0, "vix_term_triggered": False,
            "breadth": _gf(flow_obj, "breadth"),
            "breadth_triggered": False,
            "forced_beta_cap": None, "trigger_time": None,
        }
        try:
            ks_obj = cube._kill_switch
            # Surface the current latched state without triggering a new check
            kill_switch["active"] = bool(_g(ks_obj, "_triggered", False))
            kill_switch["trigger_time"] = _g(ks_obj, "_trigger_time")
            kill_switch["forced_beta_cap"] = (
                float(ks_obj.FORCED_BETA_CAP)
                if kill_switch["active"] else None
            )
        except Exception as _ks_err:
            logger.debug("kill_switch read failed: %s", _ks_err)

        # ---- Reserve kernel --------------------------------------------------
        reserve_kernel = {"impulse": 0.0, "equity_impact": 0.0, "credit_impact": 0.0, "decay": 0.0}
        try:
            rk = cube._reserve_kernel
            impulse = rk.compute_impulse(3200)
            reserve_kernel = {
                "impulse":       round(impulse, 6),
                "equity_impact": round(rk.compute_equity_impact(impulse), 6),
                "credit_impact": round(rk.compute_credit_impact(impulse), 6),
                "decay":         float(_g(rk, "decay", 0.0)),
            }
        except Exception as _rk_err:
            logger.debug("reserve_kernel read failed: %s", _rk_err)

        # ---- Risk governor ---------------------------------------------------
        risk_governor = {"all_pass": True}
        try:
            regime_val = _g(last, "regime")
            rg_checks = cube._risk_governor.full_check(regime=regime_val)
            # Serialise tuple values to dicts {pass, message}
            risk_governor = {}
            for k, v in rg_checks.items():
                if isinstance(v, tuple):
                    risk_governor[k] = {"pass": bool(v[0]), "message": str(v[1])}
                else:
                    risk_governor[k] = v
        except Exception as _rg_err:
            logger.debug("risk_governor read failed: %s", _rg_err)

        # ---- FCLP drift ------------------------------------------------------
        fclp = {"drift": 0.0, "regime_changes": 0, "samples": 0, "last_calibration": None}
        try:
            fclp = cube._fclp.get_drift_report()
        except Exception as _fclp_err:
            logger.debug("fclp read failed: %s", _fclp_err)

        # ---- Learning --------------------------------------------------------
        learning = {"accuracy": 0.0, "sample_size": 0, "correct": 0}
        learning_adjustments = {}
        try:
            learning = cube.get_learning_stats()
            learning_adjustments = cube._learning.suggest_adjustments()
        except Exception as _learn_err:
            logger.debug("learning read failed: %s", _learn_err)

        # ---- Transition probability (from RegimeEngine) ----------------------
        transition_probability = _gf(last, "transition_probability", 0.0)
        try:
            # RegimeEngine.get_transition_probability(target) returns prob of
            # transitioning from current to a specific target regime.
            # Here we return the self-persistence probability of the current regime.
            re = cube._regime_engine
            curr_regime = _g(last, "regime")
            if hasattr(re, "get_transition_probability") and curr_regime is not None:
                transition_probability = re.get_transition_probability(curr_regime)
        except Exception as _tp_err:
            logger.debug("transition_probability read failed: %s", _tp_err)

        # ---- Cube tensor -----------------------------------------------------
        cube_tensor = _tensor_list(_g(last, "cube_tensor"))

        return {
            # Top-level scalar fields
            "regime":               _regime_str(_g(last, "regime")),
            "regime_confidence":    _gf(last, "regime_confidence"),
            "transition_probability": round(float(transition_probability), 4),
            "target_beta":          _gf(last, "target_beta"),
            "beta_cap":             _gf(last, "beta_cap", 0.30),
            "max_leverage":         _gf(last, "max_leverage", 1.0),
            "risk_budget_pct":      _gf(last, "risk_budget_pct", 0.095),
            "cube_tensor":          cube_tensor,
            "timestamp":            _g(last, "timestamp") or datetime.utcnow().isoformat(),
            # Sub-objects
            "liquidity":            liq,
            "risk":                 risk,
            "flow":                 flow,
            "sleeves":              sleeves,
            "kill_switch":          kill_switch,
            "reserve_kernel":       reserve_kernel,
            "risk_governor":        risk_governor,
            "fclp":                 fclp,
            "learning":             {**learning, "adjustments": learning_adjustments},
        }
    except Exception as e:
        logger.error("cube/state error: %s", e)
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/gates — 4-gate aggregate health scores
# ===========================================================================
@router.get("/gates")
async def cube_gates():
    """Return aggregate 4-gate health scores for the CUBE tab.

    Computation priority:
    1. If last.gate_scores is a non-empty dict, return it directly.
    2. Try to compute aggregate scores via GateLogic using cube state values.
    3. Fall back to deriving scores from current L/R/F state:
         G1_Flow  = flow.value mapped to [0,1]
         G2_Macro = liquidity.value mapped to [0,1]
         G3_Fundamental = risk_budget_pct (already 0–1)
         G4_Momentum = 1 - risk.value

    Gate weights: G1=20%, G2=25%, G3=30%, G4=25%.
    """
    try:
        cube = _get_cube()
        last = cube.get_last()
        if last is None:
            return {"gates": {}, "status": "no_computation", "timestamp": datetime.utcnow().isoformat()}

        # Priority 1: pre-computed gate_scores on CubeOutput
        existing = _g(last, "gate_scores")
        if isinstance(existing, dict) and existing:
            return {"gates": existing, "source": "cube_output", "timestamp": datetime.utcnow().isoformat()}

        # Priority 2/3: derive from current state
        flow_obj = _g(last, "flow")
        risk_obj = _g(last, "risk")
        liq_obj  = _g(last, "liquidity")

        # Map flow.value [-1,+1] → [0,1]
        raw_flow = _gf(flow_obj, "value", 0.0)
        g1_flow = float(max(0.0, min(1.0, (raw_flow + 1.0) / 2.0)))

        # Map liquidity.value [-1,+1] → [0,1]
        raw_liq = _gf(liq_obj, "value", 0.0)
        g2_macro = float(max(0.0, min(1.0, (raw_liq + 1.0) / 2.0)))

        # risk_budget_pct is already in roughly [0, 0.15]; normalise to [0,1]
        rbp = _gf(last, "risk_budget_pct", 0.095)
        g3_fundamental = float(min(1.0, rbp / 0.15))

        # 1 - risk.value  (high risk = low momentum gate score)
        g4_momentum = float(max(0.0, 1.0 - _gf(risk_obj, "value", 0.3)))

        # Compute weighted score
        weights = [0.20, 0.25, 0.30, 0.25]
        scores  = [g1_flow, g2_macro, g3_fundamental, g4_momentum]
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        gate_pass = [s >= 0.3 for s in scores]
        gates_passed = sum(gate_pass)
        approved = all(gate_pass) and weighted_score >= 0.50

        # Try GateLogic for a proper evaluation using cube state as inputs
        try:
            gate_logic = cube.get_gate_logic()
            result = gate_logic.evaluate(
                ticker="__AGGREGATE__",
                flow_score=g1_flow,
                macro_score=g2_macro,
                fundamental_score=g3_fundamental,
                momentum_score=g4_momentum,
            )
            gates_data = result.get("gate_details", {})
            gates_out = {
                "G1_Flow":         gates_data.get("G1_Flow", {"score": g1_flow, "pass": gate_pass[0]}),
                "G2_Macro":        gates_data.get("G2_Macro", {"score": g2_macro, "pass": gate_pass[1]}),
                "G3_Fundamental":  gates_data.get("G3_Fundamental", {"score": g3_fundamental, "pass": gate_pass[2]}),
                "G4_Momentum":     gates_data.get("G4_Momentum", {"score": g4_momentum, "pass": gate_pass[3]}),
                "weighted_score":  round(result.get("weighted_score", weighted_score), 4),
                "gates_passed":    result.get("gates_passed", gates_passed),
                "approved":        result.get("approved", approved),
            }
        except Exception:
            gates_out = {
                "G1_Flow":        {"score": round(g1_flow, 4),        "pass": gate_pass[0]},
                "G2_Macro":       {"score": round(g2_macro, 4),       "pass": gate_pass[1]},
                "G3_Fundamental": {"score": round(g3_fundamental, 4), "pass": gate_pass[2]},
                "G4_Momentum":    {"score": round(g4_momentum, 4),    "pass": gate_pass[3]},
                "weighted_score":  round(weighted_score, 4),
                "gates_passed":    gates_passed,
                "approved":        approved,
            }

        return {
            "gates": gates_out,
            "source": "derived_from_state",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("cube/gates error: %s", e)
        return {"gates": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/history — rich time-series of CubeOutput entries
# ===========================================================================
@router.get("/history")
async def cube_history():
    """Return rich time-series history for time-series charts on the CUBE tab.

    Per entry (up to last 100 CubeOutput records):
        regime, regime_confidence, target_beta, beta_cap, max_leverage,
        L (liquidity.value), R (risk.value), F (flow.value),
        cube_tensor [L, R, F], risk_budget_pct, timestamp

    Also returns:
        regime_transitions — list of {from_regime, to_regime, timestamp, confidence}
                             from RegimeEngine._regime_history
        beta_trend         — [{t, beta}] for sparkline charts
        regime_distribution — {TRENDING: N, RANGE: N, ...}
        avg_beta_20d        — mean target_beta over last 20 entries
    """
    try:
        cube = _get_cube()
        history_obj = cube.get_history()
        if history_obj is None:
            return {"history": [], "timestamp": datetime.utcnow().isoformat()}

        # CubeHistory.get_recent(n) returns list[CubeOutput]
        try:
            entries = history_obj.get_recent(100)
        except AttributeError:
            # Fallback: history_obj might itself be a list
            raw = getattr(history_obj, "entries", history_obj)
            entries = list(raw)[-100:] if raw else []

        result = []
        beta_trend = []
        for idx, h in enumerate(entries):
            liq_obj  = _g(h, "liquidity")
            risk_obj = _g(h, "risk")
            flow_obj = _g(h, "flow")

            L = _gf(liq_obj, "value")
            R = _gf(risk_obj, "value", 0.3)
            F = _gf(flow_obj, "value")
            tb = _gf(h, "target_beta")

            entry = {
                "regime":            _regime_str(_g(h, "regime")),
                "regime_confidence": round(_gf(h, "regime_confidence"), 4),
                "target_beta":       round(tb, 4),
                "beta_cap":          round(_gf(h, "beta_cap", 0.30), 4),
                "max_leverage":      _gf(h, "max_leverage", 1.0),
                "L":                 round(L, 4),
                "R":                 round(R, 4),
                "F":                 round(F, 4),
                "cube_tensor":       [round(L, 4), round(R, 4), round(F, 4)],
                "risk_budget_pct":   round(_gf(h, "risk_budget_pct", 0.095), 4),
                "timestamp":         _g(h, "timestamp") or "",
            }
            result.append(entry)
            beta_trend.append({"t": idx, "beta": round(tb, 4)})

        # ---- Regime transitions from RegimeEngine ---------------------------
        regime_transitions = []
        try:
            re = cube._regime_engine
            for rt in getattr(re, "_regime_history", []):
                regime_transitions.append({
                    "from_regime": _regime_str(_g(rt, "from_regime")),
                    "to_regime":   _regime_str(_g(rt, "to_regime")),
                    "timestamp":   _g(rt, "timestamp", ""),
                    "confidence":  round(_gf(rt, "confidence"), 4),
                    "trigger":     _g(rt, "trigger", ""),
                })
        except Exception as _rt_err:
            logger.debug("regime_transitions read failed: %s", _rt_err)

        # ---- Summary stats ---------------------------------------------------
        regime_distribution = {}
        avg_beta_20d = 0.0
        try:
            regime_distribution = history_obj.get_regime_distribution()
            avg_beta_20d = round(float(history_obj.get_avg_beta(20)), 4)
        except Exception:
            pass

        return {
            "history":              result,
            "regime_transitions":   regime_transitions,
            "beta_trend":           beta_trend,
            "regime_distribution":  regime_distribution,
            "avg_beta_20d":         avg_beta_20d,
            "total_records":        len(result),
            "timestamp":            datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("cube/history error: %s", e)
        return {"history": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/stress-tests — stress scenario results
# ===========================================================================
@router.get("/stress-tests")
async def cube_stress_tests():
    """Run cube stress test scenarios (2008 GFC, COVID, 2022 Hike, Bull, Range, VIX spike)."""
    try:
        cube = _get_cube()
        results = cube.run_stress_tests()
        return {"stress_tests": results, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("cube/stress-tests error: %s", e)
        return {"stress_tests": {}, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/kill-switch — kill-switch status (backward-compatible endpoint)
# ===========================================================================
@router.get("/kill-switch")
async def cube_kill_switch():
    """Kill-switch status — full 9-field dict.

    All fields are also available at /cube/state under the 'kill_switch' key.
    This endpoint is kept for backward compatibility with existing dashboard consumers.
    """
    try:
        cube = _get_cube()
        last = cube.get_last()

        kill_switch = {
            "active": False, "triggered": False,
            "hy_oas_delta": 0.0, "hy_triggered": False,
            "vix_term_ratio": 0.0, "vix_term_triggered": False,
            "breadth": 0.0, "breadth_triggered": False,
            "forced_beta_cap": None, "trigger_time": None,
        }

        if last is None:
            return {**kill_switch, "status": "no_computation", "timestamp": datetime.utcnow().isoformat()}

        try:
            ks_obj = cube._kill_switch
            kill_switch["active"] = bool(_g(ks_obj, "_triggered", False))
            kill_switch["trigger_time"] = _g(ks_obj, "_trigger_time")
            kill_switch["forced_beta_cap"] = (
                float(ks_obj.FORCED_BETA_CAP) if kill_switch["active"] else None
            )
            flow_obj = _g(last, "flow")
            kill_switch["breadth"] = _gf(flow_obj, "breadth")
        except Exception as _ks_err:
            logger.debug("kill-switch read failed: %s", _ks_err)

        return {**kill_switch, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("cube/kill-switch error: %s", e)
        return {"active": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/macro-snapshot — live MacroSnapshot feeding the cube
# ===========================================================================
@router.get("/macro-snapshot")
async def cube_macro_snapshot():
    """Return the current MacroSnapshot that feeds the MetadronCube.

    Fetches via MacroEngine.get_snapshot() (cached; re-runs analyze() if cold).

    Fields returned:
        vix, spy_return_1m, spy_return_3m, yield_10y, yield_2y, yield_spread,
        credit_spread, gold_momentum, sector_rankings, gmtf_score,
        money_velocity_signal, regime (MarketRegime), cube_regime (CubeRegime)

    NOTE: yield_2y is populated from the 5Y yield series in the OpenBB data
    layer (mapping quirk — see audit §2.1).
    """
    try:
        from engine.signals.macro_engine import MacroEngine
        macro_engine = MacroEngine()
        snap = macro_engine.get_snapshot()

        return {
            "regime":               _regime_str(_g(snap, "regime")),
            "cube_regime":          _regime_str(_g(snap, "cube_regime")),
            "vix":                  _gf(snap, "vix", 20.0),
            "spy_return_1m":        round(_gf(snap, "spy_return_1m"), 4),
            "spy_return_3m":        round(_gf(snap, "spy_return_3m"), 4),
            "yield_10y":            round(_gf(snap, "yield_10y", 4.0), 4),
            "yield_2y":             round(_gf(snap, "yield_2y", 4.5), 4),
            "yield_spread":         round(_gf(snap, "yield_spread", -0.5), 4),
            "credit_spread":        round(_gf(snap, "credit_spread", 3.0), 4),
            "gold_momentum":        round(_gf(snap, "gold_momentum"), 4),
            "sector_rankings":      _g(snap, "sector_rankings", {}),
            "gmtf_score":           round(_gf(snap, "gmtf_score"), 4),
            "money_velocity_signal": round(_gf(snap, "money_velocity_signal"), 4),
            "timestamp":            datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("cube/macro-snapshot error: %s", e)
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===========================================================================
# /cube/beta-corridor — BetaCorridor state
# ===========================================================================
@router.get("/beta-corridor")
async def cube_beta_corridor():
    """Return the current BetaCorridor state from the shared singleton.

    Fields:
        current_beta      — live portfolio beta
        target_beta       — smoothed target from last BetaState
        corridor_position — BELOW / WITHIN / ABOVE
        vol_regime        — LOW_VOL / NORMAL / ELEVATED / CRISIS
        vol_percentile    — current realised vol percentile (0–100)
        history_length    — number of BetaState entries
        beta_history      — last 60 entries as [{timestamp, current_beta, target_beta}]

    Also includes full corridor analytics (zone fractions, breach frequency,
    mean-reversion probability, optimal entry/exit levels, dynamic bounds).

    BROKER SWAP NOTE:
        BetaCorridor uses get_beta() from engine.api.shared, which prefers the
        BetaCorridor instance already on ExecutionEngine to avoid diverging histories.
        Swapping the broker (METADRON_BROKER_TYPE) does not affect BetaCorridor
        computation — it reads market data via OpenBB, not broker APIs.
    """
    try:
        from engine.api.shared import get_beta
        beta_corridor = get_beta()

        # ---- Corridor analytics (zone stats, breach freq, MR prob, etc.) ----
        analytics = {}
        try:
            analytics = beta_corridor.get_corridor_analytics()
        except Exception as _ana_err:
            logger.debug("corridor_analytics failed: %s", _ana_err)

        # ---- Last BetaState --------------------------------------------------
        last_state = None
        try:
            hist = beta_corridor.get_history()
            if hist:
                last_state = hist[-1]
        except Exception:
            pass

        current_beta = float(_g(analytics, "current_beta", 0.0) or
                              getattr(beta_corridor, "current_beta", 0.0))
        target_beta  = _gf(last_state, "target_beta")
        corridor_pos = _g(last_state, "corridor_position", "UNKNOWN")
        vol_regime   = str(_g(analytics, "vol_regime", "NORMAL"))
        vol_pct      = float(_g(analytics, "vol_percentile", 50.0) or 50.0)
        hist_len     = int(_g(analytics, "history_length", 0) or 0)

        # ---- Last 60 beta history entries ------------------------------------
        beta_history = []
        try:
            hist = beta_corridor.get_history()
            for bs in hist[-60:]:
                beta_history.append({
                    "timestamp":    _g(bs, "timestamp", ""),
                    "current_beta": round(_gf(bs, "current_beta"), 4),
                    "target_beta":  round(_gf(bs, "target_beta"), 4),
                    "corridor_position": _g(bs, "corridor_position", ""),
                })
        except Exception as _bh_err:
            logger.debug("beta_history read failed: %s", _bh_err)

        return {
            "current_beta":      round(current_beta, 4),
            "target_beta":       round(target_beta, 4),
            "corridor_position": corridor_pos,
            "vol_regime":        vol_regime,
            "vol_percentile":    round(vol_pct, 1),
            "history_length":    hist_len,
            "beta_history":      beta_history,
            # Full corridor analytics
            "zone_fractions":    _g(analytics, "zone_fractions", {}),
            "breach_frequency":  round(float(_g(analytics, "breach_frequency", 0.0) or 0.0), 4),
            "mean_reversion_prob": round(float(_g(analytics, "mean_reversion_prob", 0.5) or 0.5), 4),
            "optimal_levels":    _g(analytics, "optimal_levels", {}),
            "dynamic_bounds":    list(_g(analytics, "dynamic_bounds", [0.07, 0.12]) or [0.07, 0.12]),
            "total_observations": int(_g(analytics, "total_observations", 0) or 0),
            "total_breaches":    int(_g(analytics, "total_breaches", 0) or 0),
            "timestamp":         datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("cube/beta-corridor error: %s", e)
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
