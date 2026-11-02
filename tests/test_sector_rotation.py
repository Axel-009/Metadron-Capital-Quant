import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import market_map as mm  # noqa: E402
import macro_regime as mr  # noqa: E402
import sector_rotation as sr  # noqa: E402
import rotation as rot  # noqa: E402

# Owner's example: hot resilient economy, inflation in mandate, tight labor, productivity up,
# real wages stagnant, spreads widening, escalation, AI capex strong.
PARADIGM = dict(growth_surprise=0.8, inflation_yoy=0.026, unemployment=0.039, unemployment_12m_low=0.037,
                productivity_yoy=0.028, real_wage_yoy=0.002, liquidity_change_3m=0.01,
                hy_oas_change_1m=0.0045, escalation=0.7, vix=19.0, ai_capex_growth=0.35, commodities_21d=0.05)


def test_market_map_integrity():
    for k, v in mm.SUBSECTORS.items():
        assert v["nature"] in (mm.CYCLICAL, mm.SECULAR, mm.DEFENSIVE), k
        assert all(-1 <= s <= 1 for s in v["sens"].values()), k
    ai = mm.chain("AI")
    for role in ("AI_PLATFORMS", "AI_CHIP_DESIGN", "SEMI_EQUIPMENT", "FOUNDRY_REGIONAL", "DATA_CENTERS",
                 "POWER_GRID", "AI_NEOCLOUD_CREDIT", "LOW_END_SAAS"):
        assert role in ai
    assert set(mm.by_nature(mm.DEFENSIVE)) >= {"STAPLES", "UTILITIES", "ENERGY_INTEGRATED", "DEFENSE"}


def test_factor_states_bounded_and_signed():
    f = mr.factor_states(PARADIGM)
    assert all(-1 <= v <= 1 for v in f.values())
    assert f["GROWTH"] > 0 and f["LABOR"] > 0 and f["CREDIT"] < 0 and f["GEOPOLITICS"] > 0
    assert f["WAGES"] < 0.5 and f["AI_CAPEX"] > 0


def test_question_tree_reaches_ai_substitution_and_credit_stress():
    findings = mr.diagnose(PARADIGM)
    qa = {f.question: f.answer for f in findings}
    assert qa["Is inflation within mandate?"] == "YES"
    assert qa["Is the labor market tight?"] == "YES"
    assert qa["Is labor gapping (deteriorating fast)?"] == "NO"
    assert qa["Is there room for stimulus?"] == "NO"
    assert qa["Are real wages rising?"].startswith("NO")
    tags = {t for f in findings for t in f.tags}
    assert {"AI_SUBSTITUTION", "CREDIT_STRESS", "ESCALATION"} <= tags
    assert "LABOR_GAPPING" not in tags


def test_labor_gap_triggers_stimulus_and_defensive_tilt():
    m = dict(PARADIGM, unemployment=0.046, unemployment_12m_low=0.039)
    tags = {t for f in mr.diagnose(m) for t in f.tags}
    assert "LABOR_GAPPING" in tags and "STIMULUS_ROOM" in tags


def test_paradigm_scoring_ranks_as_a_trader_would():
    findings = mr.diagnose(PARADIGM)
    f = mr.factor_states(PARADIGM)
    views = {v.subsector: v for v in sr.evaluate(f, findings=findings, bias="BULLISH",
                                                  theme_states={"AI_NEOCLOUD_CREDIT": "CRASH"})}
    assert views["DEFENSE"].edge > views["STAPLES"].edge
    assert views["AI_CHIP_DESIGN"].edge > views["LOW_END_SAAS"].edge
    assert views["AI_PLATFORMS"].edge > views["AI_NEOCLOUD_CREDIT"].edge       # financing matters
    assert views["AI_NEOCLOUD_CREDIT"].weight == 0.0                          # crashed + credit stress
    assert views["LOW_END_SAAS"].weight == 0.0                                # substitution risk
    total = sum(v.weight for v in views.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_scenarios_give_edge_and_downside():
    text = json.dumps({"source": "macro-analyst", "scenarios": [
        {"name": "base", "probability": 0.55, "shocks": {"AI_CAPEX": 0.2}},
        {"name": "credit squeeze", "probability": 0.30, "shocks": {"CREDIT": -0.7, "LIQUIDITY": -0.3}},
        {"name": "escalation", "probability": 0.15, "shocks": {"GEOPOLITICS": 0.8, "BOGUS": 1}},
    ]})
    scen = sr.scenarios_from_json(text)
    assert sum(s.probability for s in scen) == pytest.approx(1.0)
    assert "BOGUS" not in scen[2].shocks
    views = {v.subsector: v for v in sr.evaluate({}, scenarios=scen)}
    neo = views["AI_NEOCLOUD_CREDIT"]
    assert neo.downside == pytest.approx(neo.by_scenario["credit squeeze"])
    assert neo.downside < neo.edge
    assert views["DEFENSE"].by_scenario["escalation"] > views["DEFENSE"].by_scenario["base"]


def test_bias_shifts_nature_mix():
    f = mr.factor_states(dict(growth_surprise=-1.5, hy_oas_change_1m=0.008, vix=32,
                              liquidity_change_3m=-0.04))
    bias = mr.market_bias(f, corridor_zone="DEFENSIVE")
    assert bias["bias"] == "BEARISH"
    views = sr.evaluate(f, findings=mr.diagnose(dict(vix=32, hy_oas_change_1m=0.008)), bias=bias["bias"])
    by_nature = {}
    for v in views:
        by_nature[v.nature] = by_nature.get(v.nature, 0) + v.weight
    assert by_nature.get(mm.DEFENSIVE, 0) >= by_nature.get(mm.CYCLICAL, 0)


def test_stock_bond_regime():
    up = [100 * (1.001 ** i) * (1.01 if i % 2 else 0.99) for i in range(80)]
    same = [50 * (1.0005 ** i) * (1.01 if i % 2 else 0.99) for i in range(80)]
    opposite = [50 * (1.0005 ** i) * (0.99 if i % 2 else 1.01) for i in range(80)]
    assert mr.stock_bond_regime(up, same)["hedge"] == "COMMODITIES/GOLD"
    assert mr.stock_bond_regime(up, opposite)["hedge"] == "TREASURIES"


def test_leaps_threshold_is_18pct():
    assert rot.LEAPS_MIN_DISCOUNT == pytest.approx(0.18)


def test_calm_geopolitics_is_mildly_negative_not_peace_dividend():
    assert mr.factor_states({"escalation": 0.25})["GEOPOLITICS"] == pytest.approx(0.0)
    assert mr.factor_states({"escalation": 0.0})["GEOPOLITICS"] == pytest.approx(-0.5)
    assert mr.factor_states({"escalation": 0.75})["GEOPOLITICS"] == pytest.approx(1.0)
