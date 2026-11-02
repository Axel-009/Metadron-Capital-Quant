import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import fundamentals as fu  # noqa: E402
import universe as un  # noqa: E402
import rotation as rot  # noqa: E402

TD = fu.TopDown(treasury_10y=0.042, max_pe=16.0)

QUALITY = dict(price=50.0, shares_out=100e6, market_cap=5e9, eps_ttm=5.0, bvps=30.0, tangible_bvps=28.0,
               total_assets=6e9, total_liabilities=2e9, current_assets=3e9, current_liabilities=1e9,
               total_debt=0.8e9, cash=0.6e9, retained_earnings=2.5e9, revenue=5e9, ebit=0.9e9,
               net_income=0.5e9, interest_expense=0.04e9, depreciation=0.2e9, fcf=0.55e9, capex=0.25e9)
JUNK = dict(price=4.0, shares_out=200e6, market_cap=0.8e9, eps_ttm=-0.8, bvps=-1.0, tangible_bvps=-3.0,
            total_assets=5e9, total_liabilities=5.4e9, current_assets=0.6e9, current_liabilities=1.2e9,
            total_debt=4e9, cash=0.1e9, retained_earnings=-2e9, revenue=1.5e9, ebit=0.1e9,
            net_income=-0.16e9, interest_expense=0.3e9, depreciation=0.4e9, fcf=-0.5e9, capex=0.8e9)
NETNET = dict(price=8.0, shares_out=50e6, market_cap=0.4e9, eps_ttm=-0.2, bvps=14.0, tangible_bvps=13.0,
              total_assets=1.2e9, total_liabilities=0.5e9, current_assets=0.95e9, current_liabilities=0.45e9,
              total_debt=0.45e9, cash=0.3e9, retained_earnings=0.1e9, revenue=0.9e9, ebit=0.01e9,
              net_income=-0.01e9, interest_expense=0.03e9, depreciation=0.05e9, fcf=0.02e9, capex=0.03e9)


def test_quality_name_passes_gate():
    s = fu.analyze("QUAL", QUALITY, TD, equity_vol=0.25, eps_history=[3.5, 4.0, 4.4, 4.8, 5.0])
    assert s.graham_number == pytest.approx((22.5 * 4.34 * 28.0) ** 0.5, rel=1e-3)
    assert s.roic_wacc > 0 and s.capital.credit_tier in ("A", "B") and s.capital.sleeve == "IG"
    assert s.capital.merton_pd < 0.01 and s.capital.altman_z > 3
    assert len(s.iv_methods) >= 4 and s.margin_of_safety > 0
    assert s.gate_pass and s.composite >= fu.GATE_MIN and s.quality_tier in "ABCD"


def test_levered_loss_maker_blocked():
    s = fu.analyze("JUNK", JUNK, TD, equity_vol=0.9)
    assert s.capital.credit_tier == "F" and s.capital.sleeve == "DISTRESSED"
    assert s.capital.interest_coverage < 1.5 and s.capital.merton_pd > 0.10
    assert not s.gate_pass and s.gate_score == 0.0
    assert any("negative equity" in r or "no asset cover" in r for r in s.reasons)


def test_net_net_passes_as_distressed_recovery():
    s = fu.analyze("NETN", NETNET, TD, equity_vol=0.45)
    assert s.ncav_per_share == pytest.approx((0.95e9 - 0.5e9) / 50e6)
    assert s.ncav_discount > 0
    assert s.capital.sleeve == "DISTRESSED"                   # coverage 0.3x, debt/EBITDA 7.5x
    assert s.gate_pass and "distressed recovery" in s.reasons[-1]   # but trades below net current assets


def test_missing_price_fails_cleanly():
    s = fu.analyze("NONE", {}, TD)
    assert not s.gate_pass and s.reasons == ["missing price / shares"]


def test_top_down_and_rate_table():
    td = fu.top_down([dict(market_cap=100.0, net_income=4.0), dict(market_cap=100.0, net_income=6.0),
                      dict(market_cap=0, net_income=99)], treasury_10y=0.045)
    assert td.earnings_yield == pytest.approx(0.05) and td.market_pe == pytest.approx(20)
    assert td.max_pe == 16.0 and td.speculative_pct == pytest.approx(0.2)
    assert td.erp == pytest.approx(0.005) and td.regime in ("FAIR_VALUE", "OVERVALUED") and td.n_names == 2
    assert fu.max_pe_for_rate(0.01) == 25 and fu.max_pe_for_rate(0.2) == 8


def test_merton_pd_monotone():
    lo = fu.merton_pd(10e9, 2e9, 0.25)
    hi = fu.merton_pd(1e9, 4e9, 0.8)
    assert 0 <= lo < 0.001 < hi < 1
    assert fu.merton_pd(1e9, 0, 0.5) == 0.0 and fu.merton_pd(0, 1, 0.5) is None


def test_solvency_map_feeds_rotation():
    good = fu.analyze("QUAL", QUALITY, TD, equity_vol=0.25)
    bad = fu.analyze("JUNK", JUNK, TD, equity_vol=0.9)
    m = fu.cap_structure_map({"QUAL": good, "JUNK": bad})
    assert rot.is_solvent(m["QUAL"]) and not rot.is_solvent(m["JUNK"])


# ------------------------------------------------------------------ universe
def mp(v12=None, v=None):
    return NS(twelve_months=v12, value=v, three_months=None)


def fake_fundamental(ticker="ABC", sector=311, group=31130, industry=31130010):
    bs = NS(total_assets=mp(v=6e9), total_liabilities_net_minority_interest=mp(v=2e9), current_assets=mp(v=3e9),
            current_liabilities=mp(v=1e9), total_debt=mp(v=0.8e9), cash_and_cash_equivalents=mp(v=0.6e9),
            retained_earnings=mp(v=2.5e9))
    inc = NS(total_revenue=mp(5e9), ebit=mp(0.9e9), net_income=mp(0.5e9), interest_expense=mp(0.04e9),
             depreciation_and_amortization=mp(0.2e9))
    cf = NS(free_cash_flow=mp(0.55e9), capital_expenditure=mp(-0.25e9))
    return NS(symbol=NS(value=ticker), price=50.0, market_cap=5e9, dollar_volume=50e6,
              company_profile=NS(shares_outstanding=100e6),
              earning_reports=NS(basic_eps=mp(5.0)),
              valuation_ratios=NS(book_value_per_share=30.0),
              operation_ratios=NS(roic=mp(v=0.18)),
              financial_statements=NS(balance_sheet=bs, income_statement=inc, cash_flow_statement=cf),
              asset_classification=NS(morningstar_sector_code=sector, morningstar_industry_group_code=group,
                                      morningstar_industry_code=industry))


def test_adapter_reads_morningstar_paths():
    d, missing = un.extract(fake_fundamental())
    assert missing == []
    assert d["eps_ttm"] == 5.0 and d["total_liabilities"] == 2e9 and d["roic"] == 0.18
    assert d["capex"] == -0.25e9 and d["sector"] == "Information Technology"
    s = fu.analyze("ABC", d, TD, equity_vol=0.3)
    assert s.roic == 0.18 and s.gate_pass
    d2, missing2 = un.extract(NS(symbol=NS(value="X"), price=10.0))
    assert "eps_ttm" in missing2 and "market_cap" in missing2


def _p(sector, **kw):
    d = dict(sector_code=sector, revenue=1e9, capex=0.05e9, ebit=0.2e9, depreciation=0.05e9, total_debt=0.2e9,
             interest_expense=0.01e9, total_assets=3e9, total_liabilities=1e9, fcf=0.1e9, market_cap=5e9)
    d.update(kw)
    return d


def test_subsectors_come_from_fundamental_profiles_not_tickers():
    import profiles as pf
    # debt-funded AI infrastructure: capex 60% of revenue, debt/EBITDA 8x  → whatever the ticker
    neo = _p(311, capex=0.6e9, total_debt=2.0e9, ebit=0.15e9, depreciation=0.1e9, interest_expense=0.12e9)
    assert un.subsector_for("ANYNAME", neo) == "AI_NEOCLOUD_CREDIT"
    assert un.subsector_for("CRWV", _p(311)) != "AI_NEOCLOUD_CREDIT"   # a ticker alone means nothing
    burn = _p(103, capex=0.9e9, total_debt=0.05e9, interest_expense=0.001e9, fcf=-0.4e9)   # miner pivoting to AI hosting
    assert un.subsector_for("X", burn) == "AI_HOSTING_PIVOT"
    assert un.subsector_for("X", _p(311, group_code=31110, market_cap=3e9, ebit=0.05e9)) == "LOW_END_SAAS"
    assert un.subsector_for("X", _p(311, group_code=31110, market_cap=50e9, ebit=0.3e9)) == "SOFTWARE"
    assert un.subsector_for("X", _p(310, capex=0.01e9, ebit=0.05e9)) == "LABOR_INTENSIVE_SERVICES"
    assert un.subsector_for("X", _p(309, industry_code=30910020)) == "COMMODITY_PRODUCERS"
    assert un.subsector_for("X", _p(308)) == "MEDIA_TELECOM"
    assert un.subsector_for("X", _p(308, market_cap=900e9)) == "AI_PLATFORMS"
    assert un.subsector_for("X", _p(311, group_code=31130, industry_code=31130010)) == "SEMI_EQUIPMENT"
    assert un.subsector_for("X", _p(205)) == "STAPLES"
    assert un.subsector_for("X", _p(205), overrides={"X": "DEFENSE"}) == "DEFENSE"
    members = pf.theme_members({"A": "AI_NEOCLOUD_CREDIT", "B": "AI_NEOCLOUD_CREDIT", "C": "STAPLES",
                                "D": "AI_HOSTING_PIVOT"}, {"A": 1e9, "B": 5e9, "D": 2e9})
    assert members == {"AI_CREDIT_HUBS": ["B", "A"], "AI_OVERSPEND": ["D"]}
    assert pf.watch_candidate(311, 3e9, 5e7) and not pf.watch_candidate(205, 3e9, 5e7)
    from market_map import SUBSECTORS
    for table in (pf.SECTOR_DEFAULT, pf.GROUP_CODES, pf.INDUSTRY_CODES):
        assert set(table.values()) <= set(SUBSECTORS)


def test_eps_history_and_price_buffer():
    h = un.EpsHistory()
    for e in [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]:      # duplicate ignored → 9 filings
        h.update("A", e)
    assert h.annual("A") == [1, 5, 9]
    b = un.PriceBuffer()
    for i in range(30):
        b.update("A", 100 * (1.01 if i % 2 else 0.99))
    assert b.vol("A") > 0.1 and b.vol("B") is None


def _score(t, gate, sleeve, sub):
    s = fu.Score(ticker=t, subsector=sub, gate_pass=gate, gate_score=0.7)
    s.capital.sleeve = sleeve
    return s


def test_shortlist_sleeves_tilt_and_veto():
    scores = {f"IG{i}": _score(f"IG{i}", True, "IG", "DEFENSE") for i in range(10)}
    scores.update({f"HY{i}": _score(f"HY{i}", True, "HY", "AI_NEOCLOUD_CREDIT") for i in range(5)})
    scores.update({"D1": _score("D1", True, "DISTRESSED", "MATERIALS"), "F1": _score("F1", False, "IG", "DEFENSE")})
    scores["IG3"].gate_score = 0.6
    views = {"DEFENSE": {"edge": 0.4, "weight": 0.2}, "AI_NEOCLOUD_CREDIT": {"edge": -0.5, "weight": 0.0},
             "MATERIALS": {"edge": -0.2, "weight": 0.05}}
    sl = un.shortlist(scores, views, n=12, held=["HELD"])
    assert len(sl.by_sleeve["IG"]) == 8 and "IG3" not in sl.by_sleeve["IG"]     # 40:10:10 of 12 → 8 IG
    assert sl.by_sleeve["HY"] == [] and len(sl.vetoed) == 5         # macro veto: zero weight + negative edge
    assert sl.by_sleeve["DISTRESSED"] == ["D1"]                        # negative edge but weight > 0 → tilted, kept
    assert "F1" not in sl.names and "HELD" in sl.names
    assert sl.rank["IG0"] == pytest.approx(0.7 + un.MACRO_TILT * 0.4)
