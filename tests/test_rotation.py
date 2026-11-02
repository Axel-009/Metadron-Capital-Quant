import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import rotation as rot
import sectors as sec  # noqa: E402
import risk_budget as rb  # noqa: E402
import corridor as c  # noqa: E402


def path(start, daily, n=260):
    out, p = [], start
    for _ in range(n):
        out.append(p)
        p *= 1 + daily
    return out


def crash_path(start=100.0, n=260, crash_days=40, crash_total=-0.55, noise=0.04):
    """Up-trend, then a crash over the last `crash_days`, with ±noise daily swings (realised vol)."""
    up = path(start, 0.002, n - crash_days)
    d = (1 + crash_total) ** (1 / crash_days) - 1
    out, p = list(up), up[-1]
    for i in range(crash_days):
        p *= (1 + d) * (1 + (noise if i % 2 else -noise))
        out.append(p)
    return out


# Theme members as the profiles would detect them (illustrative names; membership comes from fundamentals live)
TH = {"AI_CREDIT_HUBS": ["CRWV", "NBIS"], "AI_OVERSPEND": ["IREN", "HIVE"]}


def plan_(*a, **k):
    k.setdefault("themes", TH)
    return rot.build_plan(*a, **k)


@pytest.fixture
def paradigm():
    spy = path(100, 0.0006)                                   # ~16%/yr — ROTATION zone
    prices = {
        "CRWV": crash_path(), "NBIS": crash_path(crash_total=-0.50),
        "IREN": crash_path(crash_total=-0.45), "HIVE": crash_path(crash_total=-0.60),
        "LMT": path(100, 0.001), "NOC": path(100, 0.001), "RTX": path(100, 0.001), "GD": path(100, 0.001),
        "LLY": path(100, 0.0008), "JNJ": path(100, 0.0004), "MRK": path(100, 0.0004), "ABBV": path(100, 0.0004),
        "SPY": spy,
    }
    return spy, prices


def test_double_sizing_removed_above_12pct():
    # market strong (Rm 16%) but regime reads CRASH → only Kelly + 4.5x hard cap
    assert rb.beta_cap("CRASH", 0.20, rm=0.16) == pytest.approx(4.5)
    lev = c.target_leverage(0.16, 0.20, cap=rb.beta_cap("CRASH", 0.20, rm=0.16)).leverage
    assert lev == pytest.approx(3.0, abs=0.01)                # Kelly at σ 20%, not 0.3x
    # below 12% the regime ceiling still applies
    assert rb.beta_cap("CRASH", 0.20, rm=0.10) < 0.5
    assert rb.beta_cap("CRASH", 0.20) < 0.5                   # no Rm given → conservative


def test_theme_crash_detected_while_market_strong(paradigm):
    spy, prices = paradigm
    st = rot.theme_state("AI_CREDIT_HUBS", ["CRWV", "NBIS"], prices, spy)
    assert st.state == "CRASH" and st.drawdown > 0.30 and st.rel_spy_21d < -0.15
    calm = rot.theme_state("DEF", ["LMT", "NOC"], prices, spy)
    assert calm.state == "NORMAL"


def test_plan_cuts_and_rotates_into_gics_sectors(paradigm):
    spy, prices = paradigm
    holdings = {"CRWV": 0.06, "NBIS": 0.04, "IREN": 0.03}
    oil = path(70, 0.006, 60)                                  # oil +~43% in 60d
    esc = rot.escalation_score(oil=oil, vix=27.0)
    assert esc >= rot.ESCALATION_THRESHOLD
    scores = sec.sector_scores({"HEALTHCARE_PHARMA": 0.4, "DEFENSE": 0.5, "AI_NEOCLOUD_CREDIT": -0.6},
                               regime="RANGE", escalation=esc, crashing_subsectors=["AI_NEOCLOUD_CREDIT"])
    plan = plan_("ROTATION", prices, spy, holdings, escalation=esc, sector_scores=scores)
    cuts = {a.ticker: a.target_weight for a in plan.actions if a.kind == "CUT"}
    assert cuts == {"CRWV": 0.0, "NBIS": 0.0, "IREN": 0.0}
    dest = {a.ticker: a.target_weight for a in plan.actions if a.kind == "ROTATE"}
    assert set(dest) <= set(sec.SECTORS)                       # GICS sectors (filled with single names), never ETFs
    assert set(dest) == {"Industrials", "Health Care", "Energy"}
    assert "Information Technology" not in dest                # the crashing theme's sector is marked down
    budgets = sec.sector_budgets(scores)
    assert dest["Industrials"] == pytest.approx(budgets["Industrials"])
    esc_calls = [a for a in plan.actions if a.kind == "ESCALATION_CALLS"]
    assert esc_calls and esc_calls[0].ticker == "Energy"       # calls on the best Energy single names
    assert not any(a.kind == "ESCALATION_EQUITY" for a in plan.actions)


def test_sector_budgets_follow_bullish_bearish_reasoning():
    bear = sec.sector_budgets(sec.sector_scores({}, regime="STRESS", bias="BEARISH"))
    bull = sec.sector_budgets(sec.sector_scores({}, regime="TRENDING", bias="BULLISH"))
    defensive = ("Health Care", "Consumer Staples", "Utilities", "Energy")
    assert sum(bear[s] for s in defensive) > 0.6 and sum(bull[s] for s in defensive) < 0.3
    for b in (bear, bull):
        assert sum(b.values()) == pytest.approx(1.0)
        assert max(b.values()) <= sec.MAX_SECTOR + 1e-9
        assert all(v == 0 or v >= sec.MIN_SECTOR - 1e-9 for v in b.values())
    none = sec.sector_budgets({s: -1.0 for s in sec.SECTORS})
    assert {s for s, v in none.items() if v} == set(sec.DEFENSIVE_FALLBACK)
    plan = plan_("ROTATION", {"CRWV": crash_path(), "NBIS": crash_path()}, path(100, 0.0006),
                          {"CRWV": 0.10}, escalation=0.0)        # no scores → defensive fallback sectors
    assert {a.ticker for a in plan.actions if a.kind == "ROTATE"} == set(sec.DEFENSIVE_FALLBACK)


def test_leaps_only_on_solvent_crashed_names(paradigm):
    spy, prices = paradigm
    cap = {"CRWV": {"merton_pd": 0.02, "interest_coverage": 3.5},      # solvent
           "NBIS": {"merton_pd": 0.12, "interest_coverage": 1.1}}      # not solvent
    plan = plan_("ROTATION", prices, spy, {}, cap_structure=cap,
                          implied_vol={"CRWV": 0.70})
    leaps = {a.ticker: a.kind for a in plan.actions if a.kind.startswith("LEAPS")}
    assert "CRWV" in leaps and "NBIS" not in leaps and "IREN" not in leaps   # unknown → no LEAPS
    rich = plan_("ROTATION", prices, spy, {}, cap_structure=cap,
                          implied_vol={"CRWV": 3.0})
    assert any(a.kind == "LEAPS_SPREAD" and a.ticker == "CRWV" for a in rich.actions)


def test_straddles_when_vol_cheap_or_event_near(paradigm):
    spy, prices = paradigm
    plan = plan_("ROTATION", prices, spy, {},
                          implied_vol={"HIVE": 0.10, "CRWV": 5.0}, days_to_event={"NBIS": 5})
    s = {a.ticker for a in plan.actions if a.kind == "STRADDLE"}
    assert "HIVE" in s and "NBIS" in s and "CRWV" not in s
    assert len(s) <= rot.STRADDLE_MAX_NAMES


def test_no_actions_when_no_theme_stress():
    spy = path(100, 0.0006)
    prices = {t: path(100, 0.0007) for t in ("CRWV", "NBIS", "IREN", "HIVE", "LMT", "LLY")}
    plan = plan_("ROTATION", prices, spy, {"CRWV": 0.05})
    assert not plan.active and plan.actions == []


def test_agent_flag_overrides_escalation():
    assert rot.escalation_score(agent_flag=True) == 1.0
    assert rot.escalation_score(agent_flag=False) == 0.0


def test_plan_is_idempotent(paradigm):
    """Re-running the plan on the post-trade book must not compound (owner bug report)."""
    spy, prices = paradigm
    holdings = {"CRWV": 0.06, "NBIS": 0.04}
    plan1 = plan_("ROTATION", prices, spy, holdings, escalation=1.0)
    after = dict(holdings)
    for a in plan1.actions:
        if a.kind == "CUT":
            after[a.ticker] = a.target_weight
    plan2 = plan_("ROTATION", prices, spy, after, escalation=1.0)
    assert not [a for a in plan2.actions if a.kind == "CUT"]          # nothing left to cut
    r1 = {a.ticker: a.target_weight for a in plan1.actions if a.kind == "ROTATE"}
    r2 = {a.ticker: a.target_weight for a in plan2.actions if a.kind == "ROTATE"}
    assert r1 == r2                                                     # budgets come from state, not flows


def test_stress_caps_instead_of_halving():
    spy = path(100, 0.0006)
    stressed = path(100, 0.002, 240) + path(100 * 1.002 ** 240 * 0.99, -0.01, 20)   # ~18% drop
    prices = {"CRWV": stressed, "NBIS": stressed}
    plan = plan_("ROTATION", prices, spy, {"CRWV": 0.08, "NBIS": 0.01})
    cuts = {a.ticker: a.target_weight for a in plan.actions if a.kind == "CUT"}
    assert plan.themes[0].state == "STRESS"
    assert cuts == {"CRWV": rot.STRESS_NAME_CAP}           # NBIS already under the cap


def test_credit_stress_exposes_themes_earlier():
    spy = path(100, 0.0006)
    mild = crash_path(crash_total=-0.12)                       # 12% slide: NORMAL on price alone
    prices = {"CRWV": mild, "NBIS": mild}
    assert rot.theme_state("AI_CREDIT_HUBS", ["CRWV", "NBIS"], prices, spy).state in ("NORMAL", "STRESS")
    st = rot.theme_state("AI_CREDIT_HUBS", ["CRWV", "NBIS"], prices, spy, credit_stress=True)
    assert st.state == "STRESS"
    assert rot.THEMES == {}                                    # no hard-coded ticker watchlist
