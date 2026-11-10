import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
import feed_guard as fg  # noqa: E402
import risk_officer as ro  # noqa: E402


def warm(g, n=10, spy=500.0, mes=5000.0, t0=0.0):
    for i in range(n):
        g.check(t0 + i, {"SPY": t0 + i, "MES": t0 + i}, spy * (1 + 0.0001 * i), mes * (1 + 0.0001 * i))
    return t0 + n


def test_stale_feed_halts_and_clears_after_clean_checks():
    g = fg.FeedGuard()
    t = warm(g)
    st = g.check(t + 30, {"SPY": t + 30, "MES": t - 1}, 500.5, 5005.0)      # MES feed delayed 31 min
    assert not st.ok and any("MES: stale" in r for r in st.reasons)
    assert not g.check(t + 31, {"SPY": t + 31, "MES": None}, 500.5, 5005.0).ok   # cut feed
    assert not g.check(t + 32, {"SPY": t + 32, "MES": t + 32}, 500.5, 5005.0).ok  # 1 clean check is not enough
    assert g.check(t + 33, {"SPY": t + 33, "MES": t + 33}, 500.5, 5005.0).ok


def test_cross_feed_disagreement_halts():
    g = fg.FeedGuard()
    t = warm(g)
    st = g.check(t, {"SPY": t, "MES": t}, 500.0, 5000.0 * 1.01)              # fresh but 1% apart
    assert not st.ok and any("basis" in r for r in st.reasons)


def test_one_sided_jump_halts():
    g = fg.FeedGuard()
    t = warm(g)
    st = g.check(t, {"SPY": t, "MES": t}, 500 * 0.95, 5000.0)
    assert not st.ok and any("jumped" in r for r in st.reasons)


def test_collars_and_resting_stops():
    assert fg.collar(100.0, True) == 100.3 and fg.collar(100.0, False) == 99.7
    assert fg.collar(5000.0, True, "FUTURE") == 5010.0
    assert fg.position_stop(110.0) == 99.0                                    # peak −10%
    assert fg.futures_stop(5000.0, 2.0) == 4800.0                             # 4% move × 2x = 8% NAV
    assert fg.futures_stop(5000.0, -0.5) == 5500.0                            # short hedge, capped at 10%
    assert fg.futures_stop(5000.0, 8.0) == 4900.0                             # floor 2%
    assert fg.futures_stop(5000.0, 0.0) is None


def test_risk_officer_out_of_band_audit():
    state = {"feed": {"ok": True, "spy_last": 500.0, "spy_time": "2026-11-03 15:45:00"}}
    assert ro.feed_findings(state, dt.date(2026, 11, 3), reference=lambda t, d: 501.0) == []
    bad = ro.feed_findings(state, dt.date(2026, 11, 3), reference=lambda t, d: 520.0)
    assert bad and "independent close" in bad[0]
    halted = ro.feed_findings({"feed": {"ok": False, "reasons": ["MES: stale 12 min"]}}, dt.date(2026, 11, 3),
                              reference=lambda t, d: None)
    assert halted and "HALTED" in halted[0]


def test_phase1_reference_is_the_spx_index():
    g = fg.FeedGuard()
    for i in range(10):
        g.check(i, {"SPY": i, "SPX": i}, 500.0, 5000.0, "SPX")
    st = g.check(10, {"SPY": 10, "SPX": 10}, 500.0, 5000.0 * 1.01, "SPX")
    assert not st.ok and any("SPY/SPX basis" in r for r in st.reasons)
    st = fg.FeedGuard().check(10, {"SPY": 10, "SPX": 2}, 500.0, 5000.0, "SPX")
    assert any("SPX: stale" in r for r in st.reasons)                       # reference held to the 5-min rule


def test_algorithm_files_fit_quantconnect_limit():
    """QuantConnect rejects project files over 64,000 characters (found on the first deploy)."""
    import glob
    algo = os.path.join(os.path.dirname(__file__), "..", "algorithm")
    for path in glob.glob(os.path.join(algo, "*.py")):
        n = len(open(path, encoding="utf-8").read())
        assert n < 60_000, f"{os.path.basename(path)} has {n} characters — split it before QuantConnect rejects it"


def test_algorithm_class_has_only_the_quantconnect_base():
    """QuantConnect forbids combining QCAlgorithm (a C# class) with other Python base classes (found on deploy)."""
    import ast
    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "algorithm", "main.py")).read())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetadronCorridor"][0]
    assert [b.id for b in cls.bases if isinstance(b, ast.Name)] == ["QCAlgorithm"]


def test_persistent_basis_shift_reanchors_instead_of_halting_forever():
    """A structural SPY/reference shift (dividend, index change) must not halt trading permanently."""
    g = fg.FeedGuard()
    t = warm(g)
    shifted = 5000.0 * 1.01                                   # reference now permanently 1% higher vs SPY
    states = [g.check(t + i, {"SPY": t + i, "MES": t + i}, 500.0, shifted).ok for i in range(1, 12)]
    assert states[0] is False                                 # a sudden disagreement still halts at once
    assert g.rebased == 1 and states[-1] is True              # re-anchored after 4 fresh checks, then clears
    assert not g.check(t + 20, {"SPY": t + 20, "MES": t + 20}, 500.0, shifted * 1.02).ok   # new jump halts again


# ---------------------------------------------------------------- protection cooldown (trim / stop is not undone)
def _guard_mixin():
    import sys, types
    sys.modules.setdefault("AlgorithmImports", types.ModuleType("AlgorithmImports"))
    import desk_guard
    return desk_guard.GuardMixin


def _fake_algo(qty=100, price=50.0, nav=100_000.0):
    import datetime as dt, types
    G = _guard_mixin()
    sym = types.SimpleNamespace(value="ABC")
    hold = types.SimpleNamespace(quantity=qty)
    sec = types.SimpleNamespace(price=price)
    a = types.SimpleNamespace(protect={}, targets={"ABC": 0.05}, time=dt.datetime(2024, 3, 4, 10, 30), orders=[])
    a.securities = types.SimpleNamespace(contains_key=lambda s: s is sym, __getitem__=None)
    a.securities = type("S", (), {"contains_key": lambda self, s: s is sym, "__getitem__": lambda self, s: sec})()
    a._symbol = lambda t: sym if t == "ABC" else None
    a._nav = lambda: nav
    a._holding = lambda s: hold if s is sym else None
    a._limit = lambda s, q, kind, tag: a.orders.append((s.value, q, tag))
    for name in ("_protect_note", "_apply_protect", "_reprice_protect", "_ten_year"):
        setattr(a, name, getattr(G, name).__get__(a))
    return a, hold, sec


def test_trim_holds_through_next_research_and_expires():
    import datetime as dt
    a, hold, sec = _fake_algo()
    a._protect_note("ABC", 50, "TRIM drawdown 6%")                 # −6% trim: keep 50 of 100 shares
    assert abs(a.targets["ABC"] - 0.025) < 1e-9                    # today's target capped at the kept shares
    a._reprice_protect()
    assert a.orders[-1][:2] == ("ABC", -50)                        # the protective sell
    a._reprice_protect()
    assert len(a.orders) == 2                                      # unfilled → re-issued at the current price
    hold.quantity = 50
    a._reprice_protect()
    assert len(a.orders) == 2                                      # filled → nothing more
    a.time += dt.timedelta(days=1)                                 # next morning: research wants 5% again
    assert abs(a._apply_protect({"ABC": 0.05})["ABC"] - 0.025) < 1e-9
    a.time += dt.timedelta(days=20)                                # cooldown over → free again
    assert a._apply_protect({"ABC": 0.05})["ABC"] == 0.05 and "ABC" not in a.protect


def test_stop_blocks_rebuy_for_the_cooldown():
    a, hold, sec = _fake_algo()
    a._protect_note("ABC", 0, "CLOSE drawdown 10%")
    assert a.targets["ABC"] == 0.0
    a._reprice_protect()
    assert a.orders[-1][:2] == ("ABC", -100)
    assert a._apply_protect({"ABC": 0.08})["ABC"] == 0.0
    assert a._apply_protect({})["ABC"] == 0.0


def test_ten_year_default_and_macro_value():
    a, _, _ = _fake_algo()
    assert a._ten_year({}) == 0.042
    assert a._ten_year({"treasury_10y": 0.029}) == 0.029
    assert a._ten_year({"treasury_10y": 5.0}) == 0.042            # nonsense reading → default


def test_last_bar_age_uses_the_exchange_clock_not_the_algorithm_clock():
    """SPX bars are stamped in Chicago time; the algorithm runs on New York time. A fresh bar must read as fresh."""
    import datetime as dt, types
    G = _guard_mixin()
    ny = dt.datetime(2018, 3, 5, 10, 0)                       # algorithm clock (New York)
    chicago_now, bar_end = dt.datetime(2018, 3, 5, 9, 0), dt.datetime(2018, 3, 5, 8, 59)
    sec = types.SimpleNamespace(local_time=chicago_now, get_last_data=lambda: types.SimpleNamespace(end_time=bar_end))
    a = types.SimpleNamespace(time=ny, securities={"SPX": sec})
    a._minutes = G._minutes.__get__(a)
    last = G._last_bar.__get__(a)("SPX")
    assert abs((a._minutes(ny) - last) - 1.0) < 1e-6          # 1 minute old, not 61


def test_stale_held_name_blocks_that_name_not_the_book():
    g = fg.FeedGuard()
    now = 10_000.0
    st = g.check(now, {"SPY": now - 1, "SPX": now - 1, "DIVO": now - 177}, 400.0, 4000.0, "SPX")
    assert st.ok and st.stale == ["DIVO"]                          # thin ETF skipped minutes: no halt
    st = g.check(now + 30, {"SPY": now + 29, "SPX": now + 29, "DIVO": now + 29}, 400.0, 4000.0, "SPX")
    assert st.ok and st.stale == []
    st = g.check(now + 60, {"SPY": now + 20, "SPX": now + 59}, 400.0, 4000.0, "SPX")
    assert not st.ok                                               # SPY itself stale → halt
