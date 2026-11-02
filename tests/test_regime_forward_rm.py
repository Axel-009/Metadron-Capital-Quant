import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
import regime as rg  # noqa: E402
import forward_rm as fr  # noqa: E402
import risk_budget as rb  # noqa: E402
import corridor as cr  # noqa: E402
import macro_agent as ma  # noqa: E402


def path(segments, seed=1, start=100.0):
    """segments: [(days, annual drift, annual vol)] → close series."""
    rnd = random.Random(seed)
    px = [start]
    for n, mu, sig in segments:
        for _ in range(n):
            px.append(px[-1] * math.exp(mu / 252 - 0.5 * sig ** 2 / 252 + sig / math.sqrt(252) * rnd.gauss(0, 1)))
    return px


CALM_THEN_CRASH = [(400, 0.15, 0.12), (200, 0.05, 0.18), (60, -0.9, 0.55)]
CALM = [(660, 0.15, 0.11)]


def test_rules_fallback():
    assert rg.rules_regime(path(CALM_THEN_CRASH)).regime in ("CRASH", "STRESS")
    assert rg.rules_regime(path(CALM)).regime in ("TRENDING", "RANGE")
    assert rg.rules_regime([100.0] * 10).regime == "RANGE"


@pytest.mark.skipif(not rg.HMM_AVAILABLE, reason="hmmlearn not installed")
def test_hmm_flags_crash_and_calm():
    m = rg.RegimeModel()
    crash = m.classify(path(CALM_THEN_CRASH))
    assert crash.method == "hmm"
    assert crash.probs["STRESS"] + crash.probs["CRASH"] > 0.5
    assert sum(crash.probs.values()) == pytest.approx(1.0)
    calm = rg.RegimeModel().classify(path(CALM))
    assert calm.probs["TRENDING"] + calm.probs["RANGE"] > 0.5


@pytest.mark.skipif(not rg.HMM_AVAILABLE, reason="hmmlearn not installed")
def test_hmm_refits_on_schedule():
    m = rg.RegimeModel()
    px = path(CALM_THEN_CRASH)
    m.classify(px)
    first = m.model
    for _ in range(rg.REFIT_DAYS - 1):     # 20 more days on the same fit
        m.classify(px)
    assert m.model is first
    m.classify(px)
    assert m.model is not first


def test_forward_rm_is_shrunk_and_bounded():
    neutral = fr.estimate(0.10)
    assert neutral.rm == pytest.approx(0.10)
    hot = fr.estimate(0.60)                         # trailing +60% → nowhere near +60%
    assert 0.10 < hot.rm < 0.25
    cold = fr.estimate(-0.33)
    assert -0.15 < cold.rm < 0.10
    assert fr.estimate(5.0).rm == fr.RM_MAX


def test_forward_rm_components():
    crash = fr.estimate(0.10, regime_probs={"CRASH": 1.0})
    assert crash.rm == pytest.approx(0.0)
    assert crash.components["regime_adj"] == pytest.approx(-0.10)
    tight = fr.estimate(0.10, gmtf_score=1.0)       # max monetary tension → −4%
    assert tight.rm == pytest.approx(0.06)
    anchored = fr.estimate(0.10, earnings_yield=0.045, inflation=0.03)
    assert anchored.components["anchor"] == pytest.approx(0.075)
    kron = fr.estimate(0.10, kronos=0.20)
    assert kron.rm == pytest.approx(0.75 * 0.10 + 0.25 * 0.20)


def test_trailing_noise_rarely_flips_zone():
    """Forward Rm moves far less than the raw drift: a ±20% drift swing moves Rm by ±5%."""
    lo, hi = fr.estimate(-0.10).rm, fr.estimate(0.30).rm
    assert hi - lo == pytest.approx(0.25 * 0.40)


def test_beta_burst():
    px_up = [100 * 1.001 ** i for i in range(260)]
    assert rb.momentum_confirms(px_up)
    assert not rb.momentum_confirms(list(reversed(px_up)))
    assert not rb.momentum_confirms(px_up[:150])
    base = rb.beta_cap("RANGE", 0.20)
    burst = rb.beta_cap("RANGE", 0.20, burst=True)
    assert burst == pytest.approx(min(base * 0.55 / 0.45, rb.MAX_GROSS_LEVERAGE))
    assert rb.beta_cap("CRASH", 0.40, burst=True) == pytest.approx(rb.beta_cap("CRASH", 0.40))
    assert rb.beta_cap("TRENDING", 0.10, burst=True) <= rb.MAX_GROSS_LEVERAGE


def test_end_to_end_crash_goes_flat():
    px = path(CALM_THEN_CRASH)
    reg = rg.RegimeModel().classify(px)
    rm = fr.estimate(cr.annualised_drift(px[-253:]), regime_probs=reg.probs).rm
    sigma = cr.realised_vol(px[-64:])
    cap = rb.beta_cap(reg.regime, sigma, rm=rm)
    assert cr.target_leverage(rm, sigma, cap=cap).leverage == pytest.approx(0.0)


# ---------------------------------------------------------------- macro agent (offline)
def monthly(vals, start=(2015, 1)):
    y, m = start
    out = []
    for v in vals:
        out.append((f"{y}-{m:02d}-01", v))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fake_fred(sid):
    n = 120
    if sid == "CPIAUCSL":
        return monthly([250 * 1.025 ** (i / 12) for i in range(n)])
    if sid == "UNRATE":
        return monthly([3.7] * (n - 3) + [3.9, 4.0, 4.1])
    if sid == "AHETPI":
        return monthly([25 * 1.03 ** (i / 12) for i in range(n)])
    if sid == "OPHNFB":
        return monthly([110 * 1.02 ** (i / 4) for i in range(40)])
    if sid == "DGS10":
        return [(f"d{i:04d}", 4.0 + 0.002 * i) for i in range(300)]
    if sid == "BAMLH0A0HYM2":
        return [(f"d{i:04d}", 3.0 + (0.5 if i > 280 else 0)) for i in range(300)]
    if sid == "WALCL":
        return [(f"w{i:03d}", 7_000_000 - 5_000 * i) for i in range(60)]
    if sid in ("WTREGEN", "RRPONTSYD"):
        return [(f"w{i:03d}", 500.0) for i in range(60)]
    if sid == "INDPRO":
        return monthly([100 * 1.002 ** i for i in range(n)])
    return []


def test_macro_agent_readings():
    r = ma.compute_readings(fake_fred)
    assert r["inflation_yoy"] == pytest.approx(0.025, abs=1e-6)
    assert r["unemployment"] == pytest.approx(0.04)
    assert r["unemployment_12m_low"] == pytest.approx(0.037)
    assert r["real_wage_yoy"] == pytest.approx(0.005, abs=1e-6)
    assert r["productivity_yoy"] == pytest.approx(0.02, abs=1e-6)
    assert r["rates_change_3m"] == pytest.approx(0.063 * 0.02, rel=1e-6)
    assert r["hy_oas_change_1m"] == pytest.approx(0.005)
    assert r["liquidity_change_3m"] < 0
    assert -3 <= r["growth_surprise"] <= 3
    assert ma.compute_readings(lambda sid: []) == {}


def test_macro_agent_scenario_validation():
    text = "Here you go:\n" + json.dumps({"scenarios": [
        {"name": "base", "probability": 0.6, "shocks": {"CREDIT": -0.1}},
        {"name": "down", "probability": 0.4, "shocks": {"CREDIT": -0.7}}]}) + "\nThanks"
    out = json.loads(ma.validate_scenarios(text))
    assert out["source"] == "macro-agent" and len(out["scenarios"]) == 2
    assert ma.validate_scenarios("no json here") is None
    assert "Question tree" in ma.build_prompt({"inflation_yoy": 0.03})


def test_macro_agent_run_offline(tmp_path, monkeypatch):
    for k in ("FRED_API_KEY", "ANTHROPIC_API_KEY", "QC_USER_ID", "QC_API_TOKEN", "QC_ORG_ID"):
        monkeypatch.delenv(k, raising=False)
    res = ma.run(str(tmp_path), use_llm=True, upload=True, fetch=fake_fred)
    f = res["files"]["metadron/macro_inputs.json"]
    assert not f["uploaded"] and "metadron/scenarios.json" not in res["files"]
    data = json.loads(open(f["file"]).read())
    assert data["inflation_yoy"] == pytest.approx(0.025, abs=1e-6)
