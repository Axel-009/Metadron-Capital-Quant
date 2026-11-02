import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import options as op  # noqa: E402
import decision as dm  # noqa: E402
import alpha as al  # noqa: E402
import protection as pr  # noqa: E402
import schedule as sc  # noqa: E402
import sectors as sec  # noqa: E402


def gbm(n, mu, sig, seed=1, start=100.0):
    rnd = random.Random(seed)
    px = [start]
    for _ in range(n):
        px.append(px[-1] * math.exp(mu / 252 - 0.5 * sig ** 2 / 252 + sig / math.sqrt(252) * rnd.gauss(0, 1)))
    return px


# ------------------------------------------------------------------ options
def test_black_scholes_parity_greeks_and_iv():
    S, K, T, s, r = 100, 105, 0.5, 0.25, 0.04
    c, p = op.bs_price(S, K, T, s, r), op.bs_price(S, K, T, s, r, call=False)
    assert c - p == pytest.approx(S - K * math.exp(-r * T), abs=1e-9)          # put-call parity
    g = op.greeks(S, K, T, s, r)
    assert 0 < g.delta < 1 and g.gamma > 0 and g.vega > 0 and g.theta < 0
    assert op.implied_vol(c, S, K, T, r) == pytest.approx(s, abs=1e-6)
    assert op.implied_vol(0.0, S, K, T) is None


def test_real_world_profit_gate():
    S, K, T, s = 100, 110, 0.75, 0.30
    prem = op.bs_price(S, K, T, s, 0.04)
    p_bull, pnl_bull = op.real_world(S, K, T, s, prem, mu=0.25)
    p_bear, pnl_bear = op.real_world(S, K, T, s, prem, mu=-0.10)
    assert p_bull > p_bear and pnl_bull > 0 > pnl_bear
    p_put, _ = op.real_world(S, 95, T, s, op.bs_price(S, 95, T, s, call=False), mu=-0.10, call=False)
    assert 0 < p_put < 1


def test_selection_respects_spread_delta_and_dte():
    S = 100.0
    qs = []
    for k in (90, 95, 100, 105, 110):
        mid = op.bs_price(S, k, 60 / 365, 0.3)
        qs.append(op.Quote(f"C{k}", k, 60, True, mid * 0.97, mid * 1.03, 500))
    qs.append(op.Quote("WIDE", 100, 60, True, 1.0, 9.0, 500))                    # spread far > 10%
    qs.append(op.Quote("SHORT", 100, 5, True, 2.0, 2.05, 500))                   # outside the DTE window
    pick = op.select(qs, S, True, 0.5, 30, 90)
    assert pick.quote.symbol == "C100" and abs(pick.delta - 0.5) < 0.1 and pick.iv == pytest.approx(0.3, abs=0.02)


def test_option_sizing_one_contract_exception():
    nav = 100_000
    assert op.contracts(0.02, nav, 5.0, 0.4, 1.0) == 4                           # $2,000 / $500
    assert op.contracts(0.001, nav, 15.0, 0.4, 1.0) == 1                         # Kelly $100 < $1,500 → 1 contract
    assert op.contracts(0.001, nav, 25.0, 0.4, 1.0) == 0                         # $2,500 > 2% NAV → none
    assert op.contracts(0.05, nav, 5.0, 0.20, 1.0) == 0                          # fails P(profit) ≥ 25%
    assert op.contracts(0.05, nav, 5.0, 0.40, -0.1) == 0                         # negative expected P&L
    assert 0 < op.kelly_option_fraction(0.4, 2.0, 5.0) < 1


# ------------------------------------------------------------------ decision matrix
def test_decision_full_kelly_and_vetoes():
    spy = gbm(300, 0.10, 0.15, seed=2)
    good = gbm(300, 0.45, 0.25, seed=3)
    d = dm.decide("GOOD", "IG", True, 0.80, good, [1e6] * 301, spy, 0.5, "SECULAR", "TRENDING")
    assert d.approved and d.mu_alpha > 0 and d.p_profit >= dm.P_PROFIT_MIN
    assert d.kelly == pytest.approx(d.mu_alpha / d.idio_vol ** 2)
    assert d.target == pytest.approx(min(d.kelly, dm.MAX_POSITION))
    bad = dm.decide("BAD", "IG", False, 0.80, good, [1e6] * 301, spy, 0.5, "SECULAR", "TRENDING")
    assert not bad.approved and bad.target == 0 and "fundamentals gate failed" in bad.reasons
    veto = dm.decide("V", "IG", True, 0.9, good, [1e6] * 301, spy, -1.0, "CYCLICAL", "CRASH")
    assert any("macro veto" in r for r in veto.reasons)
    wide = dm.decide("W", "IG", True, 0.9, good, [1e6] * 301, spy, 0.5, "SECULAR", "TRENDING", spread_bps=80)
    assert any("spread" in r for r in wide.reasons)
    assert dm.spread_bps(99.9, 100.1) == pytest.approx(20.0)


def _d(t, sleeve, sector, target):
    d = dm.Decision(t, sleeve, {}, 0.7, target=target)
    d.sector = sector
    return d


def test_allocation_matches_repo_buckets():
    assert dm.SLEEVES == {"IG": 0.40, "HY": 0.10, "DISTRESSED": 0.10, "EVENT": 0.10, "CASHFLOW": 0.15,
                          "FI_MACRO": 0.05, "OPTIONS": 0.10}                     # no money market, no margin bucket
    assert sum(dm.SLEEVES.values()) == pytest.approx(1.0)
    assert sum(dm.OPTION_PREMIUM[k] for k in ("IG", "HY", "DISTRESSED")) == pytest.approx(dm.OPTIONS_TOTAL) == 0.10
    assert dm.sleeve_caps()["IG"] == 0.40                                  # no IG cut for futures margin
    assert dm.futures_margin(4.5) == pytest.approx(4.5 * dm.MES_MARGIN_RATE)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from engine.allocation.allocation_engine import AllocationRules
    r = AllocationRules()
    assert (r.options_notional_pct, r.options_ig_pct, r.options_hy_pct, r.options_distressed_pct) == (0.10, 0.04, 0.04, 0.02)
    assert r.money_market_pct == 0.0 and r.ig_reduction_on_margin_breach == 0.0
    import risk_budget as rbx
    assert rbx.MAX_FUTURES_LEVERAGE == 2.0 and r.futures_beta_pct == 2.0                # owner: futures ≤ 2.0x NAV


def test_sleeves_fill_by_gics_sector_budget_with_spill():
    budget = {"Health Care": 0.2, "Utilities": 0.3, "Energy": 0.3, "Materials": 0.2}   # as sector_budgets makes them
    ds = [_d(f"H{i}", "IG", "Health Care", 0.10) for i in range(4)]      # wants 40%; room 8%, spill up to 12%
    ds += [_d("U1", "IG", "Utilities", 0.05)]                           # wants 5%, room 12%
    ds += [_d("T1", "IG", "Information Technology", 0.10)]              # sector with zero budget
    ds += [_d(f"Y{i}", "HY", "Energy", 0.05) for i in range(4)]
    dm.fit_sleeves(ds, dm.SLEEVES, budget)
    hc = sum(d.target for d in ds if d.sector == "Health Care")
    assert hc == pytest.approx(0.40 * dm.SECTOR_MAX)                    # unused room spilled, stopped at 30% of the sleeve
    assert next(d for d in ds if d.ticker == "U1").target == pytest.approx(0.05)
    assert next(d for d in ds if d.ticker == "T1").target == 0.0        # unfavoured sector gets nothing
    assert sum(d.target for d in ds if d.sleeve == "HY") <= 0.10 + 1e-9
    beta, idio = dm.beta_and_idio(gbm(300, 0.1, 0.2, 5), gbm(300, 0.1, 0.15, 6))
    assert idio > 0.1


# ------------------------------------------------------------------ alpha
def test_patterns_and_features():
    up = [100 * 1.002 ** i for i in range(260)]
    s, tags = al.patterns(up, [1e6] * 255 + [3e6] * 5)
    assert s > 0 and "breakout 52w high on volume" in tags
    f = al.price_features(up)
    assert f["mom_12_1"] > 0 and f["vol_63"] >= 0
    assert al.rsi([100 - i for i in range(20)]) == pytest.approx(0.0)


@pytest.mark.skipif(al.np is None, reason="numpy missing")
def test_walk_forward_learns_a_planted_signal():
    rnd = random.Random(0)
    wf = al.WalkForward(features=("x",), horizon_steps=1)
    prices = {f"T{i}": 100.0 for i in range(80)}
    sig = {t: 0.0 for t in prices}
    for step in range(12):
        rows = {}
        for t in prices:
            prices[t] *= 1 + 0.05 * sig[t] + rnd.gauss(0, 0.01)       # last step's x predicts this move
            sig[t] = rnd.uniform(-1, 1)
            rows[t] = ({"x": sig[t]}, prices[t])
        wf.observe(rows)
    assert wf.fit() and wf.predict({"x": 1.0}) > wf.predict({"x": -1.0})


def test_kronos_and_scorecard():
    k = al.kronos_alpha(json.dumps({"horizon_days": 21, "forecasts": {"SPY": 0.01}}))
    assert k["SPY"] == pytest.approx(0.12)
    assert al.kronos_alpha("not json") == {}
    sc_ = al.Scorecard()
    assert sc_.weights()["score"] == 1.0 and sc_.weights()["ml"] == 0.0         # unproven models start at 0
    real = {f"T{i}": i / 100 for i in range(10)}
    for _ in range(4):
        for t, r in real.items():
            sc_.record("ml", t, r)                                              # perfect forecasts
            sc_.record("kronos", t, -r)                                         # always wrong
        sc_.settle(real)
    w = sc_.weights()
    assert w["ml"] == pytest.approx(0.5) and w["kronos"] == 0.0 and sc_.bad_streak["kronos"] == 4


# ------------------------------------------------------------------ protection + schedule
def test_position_protection_trim_then_close():
    p = pr.Protector()
    assert p.update("A", 100, 10) is None
    assert p.update("A", 110, 10) is None
    a = p.update("A", 103, 10)
    assert a.kind == "TRIM" and a.fraction == 0.5
    assert p.update("A", 102, 5) is None                                        # trims once per peak
    assert p.update("A", 98, 5).kind == "CLOSE"


def test_option_exits_hedges_breaker_cvar():
    assert pr.option_exit(2, 60, 1.0, 1.0) == "expiry in 2d"
    assert pr.option_exit(80, 300, 0.2, 1.0) == "LEAPS roll window (80d left)"
    assert pr.option_exit(200, 300, 0.2, 1.0) is None                          # LEAPS: no premium stop
    assert pr.option_exit(30, 60, 0.4, 1.0).startswith("option stop")
    assert pr.option_exit(20, 30, 1.0, 1.0, "HEDGE", days_to_event=-1) == "event passed"
    assert pr.needs_hedge(0.05, 3, False) == "HEDGE" and pr.needs_hedge(0.02, 3, False) is None
    assert pr.needs_hedge(0.05, -1, True) == "UNHEDGE"
    assert pr.breaker(96.9, 100) and not pr.breaker(97.5, 100)
    f = pr.cvar_scale(4.5, 0.30, 0.02, 0.0225)
    assert 0 < f < 1 and pr.book_cvar(4.5 * f, 0.30, 0.02) == pytest.approx(1.5 * 0.0225, rel=1e-6)
    assert pr.cvar_scale(1.0, 0.15, 0.01, 0.07) == 1.0


def test_heartbeat_windows_are_liquid():
    assert sc.CORE >= 30 and sc.CLOSE <= 380                                    # after the open noise, before MOC cut-off
    for m in sc.OPTIONS_SELECT + sc.OPTIONS_EXECUTE:
        assert 45 <= m <= 360                                                   # never in the first 45 or last 30 minutes
    assert sc.risk_minutes()[0] == 30 and sc.risk_minutes()[-1] == 360
    assert len(sec.SECTORS) == 11
