"""Pairs (cointegration), playbook (learned regime → sector), theme leadership, option Monte Carlo, event dates."""
import datetime as dt
import math
import os
import random
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import options as op  # noqa: E402
import playbook as pb  # noqa: E402
import statarb as sa  # noqa: E402
import universe as un  # noqa: E402


def walk(n, seed, vol=0.015, start=100.0):
    rnd, x = random.Random(seed), [start]
    for _ in range(n):
        x.append(x[-1] * math.exp(rnd.gauss(0, vol)))
    return x


def cointegrated(x, seed, phi=0.85, hedge=1.1):
    rnd, e = random.Random(1000 + seed), [0.0]
    for _ in range(len(x) - 1):
        e.append(phi * e[-1] + rnd.gauss(0, 0.01))
    return [math.exp(0.2 + hedge * math.log(a) + b) for a, b in zip(x, e)]


def test_cointegrated_pairs_are_found_and_random_walks_are_not():
    hits = false = 0
    for seed in range(30):
        x = walk(300, seed)
        p = sa.test_pair("Y", "X", cointegrated(x, seed), x)
        hits += p is not None
        if p is not None:
            assert 0.9 < p.b < 1.3 and sa.HALF_LIFE[0] <= p.half_life <= sa.HALF_LIFE[1] and p.adf <= sa.ADF_ENTER
        false += sa.test_pair("W", "X", walk(300, 500 + seed), x) is not None
    assert hits >= 22 and false <= 1


def test_pair_opens_at_two_sigma_closes_at_the_mean_and_stops_when_broken():
    p = sa.Pair("Y", "X", a=0.0, b=1.0, mean=0.0, std=0.02, adf=-4.0, half_life=6.0)
    assert sa.update(p, 100.0, 100.0) == "FLAT"
    assert sa.update(p, 100.0 * math.exp(0.05), 100.0) == "OPEN" and p.side == -1     # y rich: short y, long x
    legs = sa.targets([p])
    assert legs == {"Y": -sa.PAIR_LEG, "X": sa.PAIR_LEG}
    assert sa.update(p, 100.0 * math.exp(0.03), 100.0) == "HOLD"
    assert sa.update(p, 100.0 * math.exp(0.005), 100.0) == "CLOSE" and p.side == 0
    sa.update(p, 100.0 * math.exp(-0.05), 100.0)
    assert p.side == 1 and sa.targets([p])["Y"] > 0                                   # y cheap: long y
    assert sa.update(p, 100.0 * math.exp(-0.09), 100.0) == "STOP" and sa.targets([p]) == {}


def test_find_pairs_stays_inside_a_sector_and_uses_a_name_once():
    x = walk(300, 1)
    px = {"A": x, "B": cointegrated(x, 1), "C": cointegrated(x, 2), "Z": cointegrated(x, 3)}
    sec = {"A": "Energy", "B": "Energy", "C": "Energy", "Z": "Utilities"}
    pairs = sa.find_pairs(px, sec)
    names = [n for p in pairs for n in (p.y, p.x)]
    assert pairs and "Z" not in names and len(names) == len(set(names))


def _cycle(n, amp, period=600, drift=0.0003, phase=0.0):
    return [100 * math.exp(drift * i + amp * math.sin(i / period * 2 * math.pi + phase)) for i in range(n)]


def test_playbook_learns_from_the_past_only_and_tilts_the_current_state():
    n = 4000
    spy = _cycle(n, 0.25)
    leader = [s * math.exp(0.0004 * i) for i, s in enumerate(spy)]            # beats SPY in every state
    laggard = [s * math.exp(-0.0004 * i) for i, s in enumerate(spy)]
    table = pb.learn(spy, {"Energy": leader, "Utilities": laggard[-2500:]})    # shorter history: fewer samples
    states = [s for s in pb.STATES if table[s]]
    assert len(states) >= 2
    for st in states:
        assert table[st]["Energy"][0] > 0 > table[st]["Utilities"][0]
        assert table[st]["Utilities"][1] <= table[st]["Energy"][1]
    st = pb.market_state(spy)
    t = pb.tilts(table, st)
    assert 0 < t["Energy"] <= pb.TILT_MAX and -pb.TILT_MAX <= t["Utilities"] < 0
    # no look-ahead: the table built at day 3000 is unchanged by anything that happens later
    early = pb.learn(spy[:3000], {"Energy": leader[:3000]})
    later = pb.learn(spy[:3000] + [1.0] * 0, {"Energy": leader[:3000]})
    assert early == later
    assert pb.market_state([100.0] * 300 + [75.0]) == "CRASH"
    assert pb.market_state([100.0 + 0.06 * i for i in range(300)]) == "BULL"


def test_leadership_finds_the_running_and_the_breaking_group():
    rets = {"AI_CHIP_DESIGN": [0.40, 0.35, 0.50, 0.45], "ENERGY": [0.05, 0.02, 0.04, 0.06],
            "STAPLES": [0.01, 0.03, 0.02, 0.00], "SOFTWARE": [-0.30, -0.25, -0.35, -0.28], "TINY": [0.9]}
    lead = pb.leadership(rets)
    assert lead["AI_CHIP_DESIGN"] > 0.5 and lead["SOFTWARE"] < -0.3 and "TINY" not in lead


def test_monte_carlo_agrees_with_the_closed_form_and_reports_the_tail():
    for call, K, mu in ((True, 105, 0.10), (False, 95, -0.20), (True, 100, 0.0)):
        mc = op.monte_carlo(100, K, 0.25, 0.30, 3.0, mu, call, n=20000)
        p, e = op.real_world(100, K, 0.25, 0.30, 3.0, mu, call)
        assert abs(mc["p_profit"] - p) < 0.02 and abs(mc["exp_pnl"] - e) < 0.25
        assert mc["p5"] == pytest.approx(-3.0) and mc["p95"] > 0 and 0 < mc["p_total_loss"] < 1
    again = op.monte_carlo(100, 105, 0.25, 0.30, 3.0, 0.10, True)
    assert again == op.monte_carlo(100, 105, 0.25, 0.30, 3.0, 0.10, True)       # fixed seed: reproducible


def test_earnings_filing_date_is_read_from_fundamentals():
    f = types.SimpleNamespace(earning_reports=types.SimpleNamespace(
        file_date=types.SimpleNamespace(three_months=dt.datetime(2018, 2, 1))))
    assert un._file_date(f) == "2018-02-01"
    assert un._file_date(types.SimpleNamespace()) is None
