import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import risk_budget as r  # noqa: E402
from risk_budget import AlphaPosition as P  # noqa: E402


def test_daily_var_formula():
    assert r.daily_var(1.0, 0.15) == pytest.approx(1.645 * 0.15 / 252 ** 0.5)


def test_regime_leverage_ceilings():
    # TRENDING 7.0% at σ 15% → 4.50x (the owner's max is reachable)
    assert r.regime_leverage_ceiling("TRENDING", 0.15) == pytest.approx(4.5, abs=0.01)
    assert r.regime_leverage_ceiling("TRENDING", 0.20) == pytest.approx(0.07 / r.daily_var(1, 0.20))
    assert r.regime_leverage_ceiling("RANGE", 0.15) == pytest.approx(0.045 / r.daily_var(1, 0.15))
    assert r.regime_leverage_ceiling("STRESS", 0.30) < r.regime_leverage_ceiling("STRESS", 0.15)
    assert r.regime_leverage_ceiling("CRASH", 0.40) < 0.5
    assert r.beta_cap("TRENDING", 0.15, regime_beta_cap=2.0) == pytest.approx(2.0)


def test_diversified_small_mid_cap_book_fits_budget():
    book = [P(f"S{i}", 0.025, 0.45) for i in range(40)]
    assert r.alpha_var(book) == pytest.approx(0.0074, abs=0.0002)
    book20 = [P(f"S{i}", 0.05, 0.45) for i in range(20)]
    assert r.alpha_var(book20) < r.ALPHA_VAR_BUDGET


def test_sector_correlation_raises_var():
    uncorr = [P(f"S{i}", 0.05, 0.45) for i in range(10)]
    same = [P(f"S{i}", 0.05, 0.45, sector="Tech") for i in range(10)]
    assert r.alpha_var(same) > r.alpha_var(uncorr)


def test_long_shorts_offset_within_sector():
    pair = [P("A", 0.05, 0.4, "Banks"), P("B", -0.05, 0.4, "Banks")]
    assert r.alpha_var(pair) < r.alpha_var([P("A", 0.05, 0.4, "Banks"), P("B", 0.05, 0.4, "Banks")])


def test_alpha_trade_scaled_to_fit():
    book = [P(f"S{i}", 0.08, 0.5) for i in range(10)]          # ~1.3% already
    chk = r.check_alpha_trade(book, P("NEW", 0.25, 0.6))
    assert not chk.ok
    assert 0 < chk.max_weight_allowed < 0.25
    fitted = r.check_alpha_trade(book, P("NEW", chk.max_weight_allowed, 0.6))
    assert fitted.ok


def test_per_name_cap_allows_small_cap_at_risk_size():
    # 0.5% name VaR at 50% vol ≈ 9.7% of NAV
    chk = r.check_alpha_trade([], P("SMALL", 0.30, 0.50))
    assert chk.max_weight_allowed == pytest.approx(0.005 / r.daily_var(1, 0.5), rel=1e-3)


def test_defined_risk_option_counts_premium_at_most():
    opt = P("CALL", 0.004, 12.0, defined_risk=True)            # 0.4% NAV premium, extreme vol
    assert r.name_var(opt) == pytest.approx(0.004)


def test_hard_stops():
    assert r.hard_stops(100, 100, 3.0).action == "OK"
    assert r.hard_stops(95, 100, 3.0).action == "TRIM"
    assert r.hard_stops(91, 100, 3.0).action == "CLOSE"
    assert r.hard_stops(100, 100, 4.8).action == "TRIM"
    assert r.hard_stops(100, 100, 2.0, book_cvar=0.06, regime="RANGE").action == "OK"     # limit 6.75%
    assert r.hard_stops(100, 100, 2.0, book_cvar=0.08, regime="RANGE").action == "TRIM"
