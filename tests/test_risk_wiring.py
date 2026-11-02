"""Tests for the risk wiring: MC gate, decision-matrix hard checks, Kelly,
1-contract exception, 10-tier ML agreement, position protection + expiries,
event tail hedges, L7 G11 / reducing-order exemption, GMTF + HMM."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.risk.portfolio_mc_gate import (
    PortfolioMCGate, Holding, parse_option_symbol, MC_VAR95_MAX,
)
import engine.execution.decision_matrix as dmmod
from engine.execution.decision_matrix import DecisionMatrix, KellySizer, build_trade_proposal
from engine.execution.conviction_override import (
    EligibilityGates, SignalDirection, ML_TIER_COUNT, MIN_ML_AGREEMENT,
)
from engine.execution.position_protection import PositionProtection
from engine.execution.event_tail_hedge import EventTailHedger
from engine.execution.l7_unified_execution_surface import L7RiskEngine, L7Order, ProductType


def _pos(qty, cost, px):
    return SimpleNamespace(quantity=qty, avg_cost=cost, current_price=px, market_value=qty * px)


def _occ(und, exp: date, cp, strike):
    return f"{und:<6}{exp:%y%m%d}{cp}{int(strike * 1000):08d}"


STRONG = dict(sharpe=2.5, quality_tier="A", alpha_signal=0.06, edge_bps=35, rsi=55,
              macd_signal=0.5, breakout=True, momentum_5d=0.03, volatility=0.30)


# ---------------------------------------------------------------- MC gate
class TestMCGate:
    def test_var_limit_and_scaling(self):
        g = PortfolioMCGate()
        trade = {"ticker": "TSLA", "side": "long", "price": 250, "volatility": 0.6, "alpha_signal": 0.05}
        units, res = g.size_to_fit(trade, 4000, 1_000_000)   # $1M position on $1M NAV
        assert 0 < units < 4000
        assert res.var95_post <= MC_VAR95_MAX

    def test_no_edge_equity_rejected(self):
        g = PortfolioMCGate()
        res = g.evaluate_trade({"ticker": "X", "price": 100, "volatility": 0.25, "alpha_signal": 0.0}, 100, 1e6)
        assert not res.passed
        assert any("P(profit)" in r for r in res.reasons)

    def test_hedge_reduces_tail_and_passes(self):
        g = PortfolioMCGate()
        g.set_holdings([Holding("AAPL", exposure=200_000, vol=0.28)])
        hedge = {"ticker": "AAPL", "instrument_type": "OPTION", "side": "long", "option_type": "put",
                 "strike": 190, "underlying_price": 200, "dte": 10, "iv": 0.35, "volatility": 0.28,
                 "is_hedge": True}
        res = g.evaluate_trade(hedge, 10, 1_000_000)
        assert res.passed
        assert res.cvar95_post < res.cvar95_pre

    def test_alpha_drift_scaled_to_hold(self):
        assert PortfolioMCGate.horizon_return({"alpha_signal": 0.06}, 21) == pytest.approx(0.06)
        assert PortfolioMCGate.horizon_return({"alpha_signal": 0.06}, 7) == pytest.approx(0.02)

    def test_occ_parse(self):
        o = parse_option_symbol("AAPL  301220P00150000")
        assert o["underlying"] == "AAPL" and not o["is_call"] and o["strike"] == 150.0


# ---------------------------------------------------------------- Decision matrix
class TestDecisionMatrix:
    def test_adv_constant_removed_and_mc_limits_real(self):
        assert not hasattr(dmmod, "ADV_MIN_RATIO")
        assert dmmod.MC_VAR95_MAX == pytest.approx(0.015)
        assert dmmod.MC_PROB_PROFIT_MIN["EQUITY"] == pytest.approx(0.52)

    def test_full_kelly(self):
        ks = KellySizer()
        assert ks.default_multiplier == 1.0
        assert ks.full_kelly(0.05) == pytest.approx(0.05)
        assert not hasattr(ks, "half_kelly")

    def test_one_contract_exception(self):
        ks = KellySizer()
        sz = ks.compute_size(nav=10_000, conviction=0.8, volatility=0.4, price=30.0, contract_multiplier=100)
        assert sz["shares"] == 1 and sz["cap_exception"] is True
        # equities never get the exception
        sz2 = ks.compute_size(nav=1_000, conviction=0.8, volatility=0.4, price=900.0, contract_multiplier=1)
        assert sz2["shares"] == 0 and sz2["cap_exception"] is False

    def test_spread_rules(self):
        dm = DecisionMatrix(nav=100_000, regime="TRENDING", mc_gate=PortfolioMCGate())
        ok, info = dm.check_spread({"spread_bps": 25})
        assert ok
        ok, info = dm.check_spread({"spread_bps": 45})
        assert not ok
        ok, info = dm.check_spread({"instrument_type": "OPTION", "bid": 1.00, "ask": 1.08})
        assert ok                                  # 7.7% of mid
        ok, info = dm.check_spread({"instrument_type": "OPTION", "bid": 1.00, "ask": 1.30})
        assert not ok
        ok, info = dm.check_spread({"instrument_type": "OPTION"})
        assert not ok and "no option quote" in info["detail"]
        ok, info = dm.check_spread({})
        assert ok and info["verified"] is False

    def test_approval_includes_sizing_and_mc(self):
        dm = DecisionMatrix(nav=100_000, regime="TRENDING", mc_gate=PortfolioMCGate())
        r = dm.evaluate_trade(build_trade_proposal("NVDA", price=120, spread_bps=3, **STRONG))
        assert r["approved"]
        assert r["sizing"]["adjusted_shares"] > 0
        assert r["mc_risk"]["var95_post"] <= MC_VAR95_MAX

    def test_mc_rejects_oversized_option(self):
        dm = DecisionMatrix(nav=10_000, regime="TRENDING", mc_gate=PortfolioMCGate())
        opt = build_trade_proposal("NVDA", instrument_type="OPTION", option_type="call", strike=130,
                                   underlying_price=128, dte=6, iv=0.5, premium=25.0, bid=24.6,
                                   ask=25.4, **STRONG)
        r = dm.evaluate_trade(opt)
        assert r["sizing"]["cap_exception"] is True
        assert not r["approved"] and any("VaR95" in x for x in r["rejection_reasons"])

    def test_tier_aliases_map_to_canonical(self):
        votes = DecisionMatrix._tier_votes({"ml_tier_votes": {"T6_miro": 1, "T10_credit": -1}})
        assert votes == {"T6_social": 1, "T10_credit_quality": -1}

    def test_gmtf_and_hmm_move_macro_gate(self):
        dm = DecisionMatrix(nav=100_000, regime="TRENDING", mc_gate=PortfolioMCGate())
        p = build_trade_proposal("XYZ", price=50, spread_bps=5, **STRONG)
        base = dm._score_macro_regime(p).score
        dm.update_macro_context(gmtf_score=0.95, hmm_regime="CRASH",
                                hmm_probs={"CRASH": 0.9, "STRESS": 0.1})
        stressed = dm._score_macro_regime(p).score
        assert stressed < base
        snap = SimpleNamespace(gmtf_source="liquidity_proxy", gmtf_score=0.1,
                               hmm_regime=None, hmm_probs={}, hmm_confidence=0.0)
        dm2 = DecisionMatrix(nav=100_000, mc_gate=PortfolioMCGate())
        dm2.update_macro_context(snap)
        assert dm2._gmtf_score is None      # proxy values are NOT treated as GMTF


# ---------------------------------------------------------------- Conviction override
class TestMLAgreement:
    def _gate(self, votes):
        return EligibilityGates()._gate_ml_consensus(votes, SignalDirection.LONG, "t")

    def test_ten_tiers(self):
        assert ML_TIER_COUNT == 10 and MIN_ML_AGREEMENT == 8

    def test_seven_of_ten_fails_eight_passes(self):
        keys = [f"T{i}" for i in range(1, 11)]
        assert self._gate({k: (1 if i < 7 else -1) for i, k in enumerate(keys)}).result.value == "FAIL"
        assert self._gate({k: (1 if i < 8 else -1) for i, k in enumerate(keys)}).result.value == "PASS"

    def test_partial_votes_do_not_inflate(self):
        g = self._gate({"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1})   # 5/5 present = 5/10
        assert g.result.value == "FAIL" and g.value == pytest.approx(0.5)


# ---------------------------------------------------------------- Protection + expiries
class TestPositionProtection:
    def test_equity_reduce_once_then_close(self):
        pp = PositionProtection()
        acts = pp.evaluate({"AAA": _pos(100, 100, 93)}, nav=100_000)
        assert [(a.action, a.quantity) for a in acts] == [("REDUCE", 50)]
        acts = pp.evaluate({"AAA": _pos(50, 100, 93)}, nav=100_000)
        assert acts == []                                  # no repeated halving
        acts = pp.evaluate({"AAA": _pos(50, 100, 89)}, nav=100_000)
        assert acts[0].action == "CLOSE" and acts[0].side == "SELL"

    def test_option_expiry_close_and_roll_flag(self):
        today = date(2026, 11, 2)
        exp1 = _occ("MSFT", today + timedelta(days=1), "C", 400)
        exp3 = _occ("MSFT", today + timedelta(days=3), "C", 410)
        acts = PositionProtection().evaluate({exp1: _pos(2, 5.0, 5.0), exp3: _pos(1, 4.0, 4.0)},
                                             nav=100_000, today=today)
        by = {a.ticker: a for a in acts}
        assert by[exp1].action == "CLOSE" and by[exp1].rule == "EXPIRY_DTE1"
        assert by[exp3].action == "FLAG"

    def test_portfolio_hard_stop(self):
        pp = PositionProtection()
        pp.evaluate({"A": _pos(10, 100, 100)}, nav=100_000)
        acts = pp.evaluate({"A": _pos(10, 100, 100), "B": _pos(5, 50, 52)}, nav=91_000)
        assert {a.ticker for a in acts if a.action == "CLOSE"} == {"A", "B"}


# ---------------------------------------------------------------- Event tail hedge
class TestEventTailHedge:
    def test_puts_for_long_with_earnings_in_window(self):
        today = date(2026, 11, 2)
        pos = {"NFLX": _pos(300, 600, 620)}
        ev = [{"ticker": "NFLX", "date": (today + timedelta(days=3)).isoformat(), "type": "EARNINGS"}]
        orders = EventTailHedger().build(pos, nav=1_000_000, events=ev, vols={"NFLX": 0.40}, today=today)
        assert len(orders) == 1
        o = orders[0]
        assert o.option_type == "PUT" and o.contracts >= 1 and o.strike < 620
        assert date.fromisoformat(o.expiry) > today + timedelta(days=3)
        assert o.est_cost <= 0.01 * 1_000_000

    def test_no_hedge_outside_window_or_small(self):
        today = date(2026, 11, 2)
        far = [{"ticker": "NFLX", "date": (today + timedelta(days=30)).isoformat(), "type": "EARNINGS"}]
        assert EventTailHedger().build({"NFLX": _pos(300, 600, 620)}, 1e6, far, today=today) == []
        near = [{"ticker": "NFLX", "date": (today + timedelta(days=2)).isoformat(), "type": "EARNINGS"}]
        assert EventTailHedger().build({"NFLX": _pos(50, 600, 620)}, 1e6, near, today=today) == []


# ---------------------------------------------------------------- L7
class TestL7Gates:
    def test_reducing_sell_not_blocked_by_daily_loss(self):
        eng = L7RiskEngine(initial_nav=100_000)
        order = L7Order(ticker="AAA", side="SELL", quantity=10, arrival_price=100)
        ok, v = eng.pre_trade_check(order, 100_000, 50_000, {"AAA": _pos(10, 120, 100)},
                                    daily_pnl=-10_000, gross_exposure=50_000, net_exposure=50_000)
        assert ok, v

    def test_g11_blocks_oversized_buy(self):
        eng = L7RiskEngine(initial_nav=100_000)
        eng.LIMITS = dict(eng.LIMITS, G1_POSITION=10.0, G4_GROSS_LEVERAGE=10.0, G5_NET_LEVERAGE=10.0)
        order = L7Order(ticker="ZZZ", side="BUY", quantity=2_000, arrival_price=100)
        ok, v = eng.pre_trade_check(order, 100_000, 1e9, {}, 0.0, 0.0, 0.0)
        assert not ok and any(x.startswith("G11_MC") for x in v)

    def test_one_contract_exception_skips_g1_only(self):
        eng = L7RiskEngine(initial_nav=10_000)
        order = L7Order(ticker="NVDA", side="BUY", quantity=1, limit_price=1_500,
                        product_type=ProductType.OPTION, min_size_exception=True,
                        option_type="CALL", strike=130, expiry=(date.today() + timedelta(days=6)).isoformat())
        ok, v = eng.pre_trade_check(order, 10_000, 1e9, {}, 0.0, 0.0, 0.0)
        assert not any(x.startswith("G1_POSITION") for x in v)


# ---------------------------------------------------------------- GMTF + HMM + Fed
class TestMacro:
    def test_fed_reserve_integration_no_crash(self):
        from engine.signals.macro_engine import FedReserveIntegration
        f = FedReserveIntegration()
        f.update(fed_assets=7.0e12)
        assert "liquidity_impulse" in f.get_state()

    def test_gmtf_real_score(self):
        from engine.signals.macro_engine import MacroEngine, MacroSnapshot, build_gmtf_inputs
        idx = pd.date_range("2008-01-01", "2026-09-30", freq="ME")
        rng = np.random.default_rng(1)

        def fake(ids, start=None):
            n = len(idx)
            out = {}
            for i in ids:
                if i.startswith(("M2", "MAB", "MYA")):
                    out[i] = np.cumprod(1 + 0.005 + 0.003 * rng.standard_normal(n))
                elif "GDP" in i or i.startswith(("CLV", "NGDP", "JPNR")):
                    out[i] = np.cumprod(1 + 0.002 + 0.002 * rng.standard_normal(n))
                elif i.startswith(("UN", "LR")):
                    out[i] = 5 + np.cumsum(0.05 * rng.standard_normal(n))
                else:
                    out[i] = 2 + np.cumsum(0.05 * rng.standard_normal(n))
            return pd.DataFrame(out, index=idx)

        inputs = build_gmtf_inputs(fetch=fake)
        snap = MacroSnapshot()
        MacroEngine()._compute_gmtf(snap, inputs=inputs)
        assert snap.gmtf_source == "gmtf" and 0.0 <= snap.gmtf_score <= 1.0
        assert set(snap.gmtf_gammas) == {"liquidity", "fx_shock", "wage", "reserve"}

    def test_hmm_regime_on_snapshot(self):
        pytest.importorskip("hmmlearn")
        from engine.signals.macro_engine import MacroEngine, MacroSnapshot
        rng = np.random.default_rng(0)
        d = pd.bdate_range(end="2026-10-30", periods=300)
        r = np.r_[0.0008 + 0.008 * rng.standard_normal(220), -0.004 + 0.03 * rng.standard_normal(80)]
        macro = pd.DataFrame({"S&P 500": 400 * np.cumprod(1 + r),
                              "VIX": np.r_[np.full(220, 14.0), np.full(80, 38.0)]}, index=d)
        snap = MacroSnapshot()
        MacroEngine()._compute_hmm_regime(macro, snap)
        assert snap.hmm_regime in ("STRESS", "CRASH")
        assert abs(sum(snap.hmm_probs.values()) - 1.0) < 1e-6
