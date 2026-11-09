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
