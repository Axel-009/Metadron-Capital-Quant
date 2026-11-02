import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import corridor as c  # noqa: E402


def test_zones_and_participation():
    assert c.participation(0.05) == 0.0 and c.zone(0.05) == c.ZONE_DEFENSIVE
    assert c.participation(0.07) == 0.0
    assert c.participation(0.095) == pytest.approx(0.5) and c.zone(0.095) == c.ZONE_DYNAMIC
    assert c.participation(0.12) == pytest.approx(1.0)
    assert c.participation(0.20) == 1.0 and c.zone(0.20) == c.ZONE_ROTATION


def test_zero_long_beta_below_7pct_even_in_low_vol():
    for rm in (-0.30, 0.0, 0.05, 0.0699):
        assert c.target_leverage(rm, 0.10).leverage == 0.0
    # bear market: negative Kelly never becomes a long position
    assert c.target_leverage(-0.20, 0.35).leverage == 0.0


def test_dynamic_zone_matches_design_table():
    # full Kelly, σ = 15%: 8.5% → 0.43, 9.5% → 1.22, 11% → 2.79, 12% → 3.56
    for rm, exp in ((0.085, 0.43), (0.095, 1.22), (0.11, 2.79), (0.12, 3.56)):
        assert c.target_leverage(rm, 0.15).leverage == pytest.approx(exp, abs=0.01)
    # half Kelly halves it
    assert c.target_leverage(0.12, 0.15, k=0.5).leverage == pytest.approx(1.78, abs=0.01)


def test_rotation_zone_vol_adjusted_and_capped_at_4_5():
    s15 = c.target_leverage(0.14, 0.15)
    assert s15.leverage == pytest.approx(4.44, abs=0.01)
    assert c.target_leverage(0.16, 0.15).leverage == pytest.approx(4.5)       # hits the cap
    assert c.target_leverage(0.16, 0.20).leverage == pytest.approx(3.00, abs=0.01)
    assert c.target_leverage(0.16, 0.25).leverage == pytest.approx(1.92, abs=0.01)
    assert c.target_leverage(0.30, 0.10, cap=10).leverage == pytest.approx(4.5)  # hard cap wins


def test_caller_cap_binds():
    assert c.target_leverage(0.12, 0.15, cap=1.0).leverage == pytest.approx(1.0)


def test_corridor_gamma_bell():
    assert c.corridor_gamma(0.07) == 0.0 and c.corridor_gamma(0.12) == 0.0
    assert c.corridor_gamma(0.095) == pytest.approx(30.0)          # 0.30 per 1%
    assert c.corridor_gamma(0.08) == pytest.approx(c.corridor_gamma(0.11))
    n, h = 20000, 0.05 / 20000
    area = sum(c.corridor_gamma(0.07 + (i + 0.5) * h) * h for i in range(n))
    assert area == pytest.approx(1.0, rel=1e-6)
    x, e = 0.0912, 1e-7
    assert (c.participation(x + e) - c.participation(x - e)) / (2 * e) == pytest.approx(
        c.corridor_gamma(x), rel=1e-5)


def test_leverage_slope_is_derivative():
    x, e, sig = 0.105, 1e-7, 0.18
    num = (c.target_leverage(x + e, sig).leverage - c.target_leverage(x - e, sig).leverage) / (2 * e)
    assert num == pytest.approx(c.target_leverage(x, sig).leverage_slope, rel=1e-4)


def test_legacy_identifiers_preserved():
    assert c.legacy_beta(0.07) == pytest.approx(-0.029)
    assert c.legacy_beta(0.12) == pytest.approx(0.425)
    assert c.LEGACY_LAMBDA_SIGMA2 == pytest.approx(0.1101, abs=1e-4)
    assert c.LEGACY_HURDLE == pytest.approx(0.0732, abs=1e-4)


def test_throttle_and_estimators():
    assert not c.should_rebalance(1.00, 1.04)
    assert c.should_rebalance(1.00, 1.06)
    closes = [100 * math.exp(0.10 / 252 * i) for i in range(253)]
    assert c.annualised_drift(closes) == pytest.approx(math.exp(0.10) - 1, rel=1e-6)
    assert c.realised_vol(closes) == pytest.approx(0.0, abs=1e-9)
