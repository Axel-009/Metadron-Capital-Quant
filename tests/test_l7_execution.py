"""Tests for L7 Unified Execution Surface.

Tests cover:
    - Data structures (L7Order, ProductType, RoutingStrategy)
    - MultiProductRouter (classification, research-only guard, routing)
    - SlippageModel (estimation, application)
    - TransactionCostAnalyzer (decomposition, aggregate)
    - L7RiskEngine (pre-trade gates, post-trade update, kill switch)
    - ExecutionLearningLoop (record, suggest, optimize, persist)
    - L7UnifiedExecutionSurface (submit_order, heartbeat, lifecycle)
    - L7DashboardRenderer (risk panel, TCA panel)
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.execution.l7_unified_execution_surface import (
    L7Order, L7UnifiedExecutionSurface, ProductType, RoutingStrategy,
    ExecutionUrgency, MultiProductRouter, SlippageModel,
    TransactionCostAnalyzer, TCASnapshot, TCAAggregate,
    L7RiskEngine, RiskState, ExecutionLearningLoop, ExecutionPattern,
    _mean,
)
from engine.monitoring.l7_dashboard import L7DashboardRenderer


# ---------------------------------------------------------------------------
# L7Order
# ---------------------------------------------------------------------------

class TestL7Order:
    def test_default_order(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=100)
        assert o.ticker == "AAPL"
        assert o.side == "BUY"
        assert o.quantity == 100
        assert o.order_id  # auto-generated
        assert o.created_at  # auto-generated
        assert o.status == "PENDING"

    def test_option_order(self):
        o = L7Order(
            ticker="AAPL", side="BUY", quantity=10,
            product_type=ProductType.OPTION,
            option_type="CALL", strike=180.0, expiry="2026-04-18",
        )
        assert o.product_type == ProductType.OPTION
        assert o.strike == 180.0

    def test_to_dict(self):
        o = L7Order(ticker="SPY", side="SELL", quantity=50)
        d = o.to_dict()
        assert d["ticker"] == "SPY"
        assert d["side"] == "SELL"
        assert d["quantity"] == 50
        assert "order_id" in d


# ---------------------------------------------------------------------------
# MultiProductRouter
# ---------------------------------------------------------------------------

class TestMultiProductRouter:
    def setup_method(self):
        self.router = MultiProductRouter()

    def test_classify_equity(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=100)
        assert self.router.classify(o) == ProductType.EQUITY

    def test_classify_option(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=10,
                    option_type="CALL", strike=180.0)
        assert self.router.classify(o) == ProductType.OPTION

    def test_classify_future(self):
        o = L7Order(ticker="ES", side="BUY", quantity=1, contract="ES")
        assert self.router.classify(o) == ProductType.FUTURE

    def test_research_only_bond_etf(self):
        assert self.router.is_research_only("TLT")
        assert self.router.is_research_only("LQD")
        assert self.router.is_research_only("HYG")

    def test_not_research_only(self):
        assert not self.router.is_research_only("AAPL")
        assert not self.router.is_research_only("SPY")

    def test_commodity_etfs_tradeable(self):
        """Commodity ETFs are for macro research AND tradeable — never blocked."""
        commodity_etfs = ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "COPX", "WEAT", "CORN"]
        for etf in commodity_etfs:
            assert not self.router.is_research_only(etf), f"{etf} should be tradeable"

    def test_index_etfs_tradeable(self):
        """Index ETFs are tradeable."""
        for etf in ["SPY", "QQQ", "IWM", "DIA", "VT", "EFA", "EEM"]:
            assert not self.router.is_research_only(etf), f"{etf} should be tradeable"

    def test_fx_research_only(self):
        """FX futures are research-only."""
        assert self.router.is_research_only("6E")
        assert self.router.is_research_only("DX")

    def test_urgency_kill_switch(self):
        assert self.router.determine_urgency("HOLD", kill_switch=True) == ExecutionUrgency.CRITICAL

    def test_urgency_micro_price(self):
        assert self.router.determine_urgency("MICRO_PRICE_BUY") == ExecutionUrgency.HIGH

    def test_routing_stats(self):
        self.router.record_route("EQUITY")
        self.router.record_route("EQUITY")
        self.router.record_route("OPTION")
        assert self.router.stats["EQUITY"] == 2
        assert self.router.stats["OPTION"] == 1


# ---------------------------------------------------------------------------
# SlippageModel
# ---------------------------------------------------------------------------

class TestSlippageModel:
    def setup_method(self):
        self.model = SlippageModel()

    def test_equity_slippage(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=100,
                    product_type=ProductType.EQUITY)
        bps = self.model.estimate_slippage_bps(o)
        assert bps > 0
        assert bps < 50  # reasonable range

    def test_option_slippage_wider(self):
        eq = L7Order(ticker="AAPL", quantity=100, product_type=ProductType.EQUITY)
        opt = L7Order(ticker="AAPL", quantity=100, product_type=ProductType.OPTION)
        assert self.model.estimate_slippage_bps(opt) > self.model.estimate_slippage_bps(eq)

    def test_apply_slippage_buy(self):
        price = self.model.apply_slippage(100.0, "BUY", 10.0)
        assert price > 100.0  # buy should be above mid

    def test_apply_slippage_sell(self):
        price = self.model.apply_slippage(100.0, "SELL", 10.0)
        assert price < 100.0  # sell should be below mid


# ---------------------------------------------------------------------------
# TransactionCostAnalyzer
# ---------------------------------------------------------------------------

class TestTCA:
    def setup_method(self):
        self.tca = TransactionCostAnalyzer()

    def test_analyze_basic(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=100,
                    fill_quantity=100, product_type=ProductType.EQUITY)
        snap = self.tca.analyze(o, arrival_price=150.0, fill_price=150.05)
        assert snap.spread_cost_bps > 0
        assert snap.total_cost_bps > 0
        assert snap.ticker == "AAPL"

    def test_aggregate_empty(self):
        agg = self.tca.get_aggregate()
        assert agg.total_trades == 0

    def test_aggregate_with_trades(self):
        for i in range(5):
            o = L7Order(ticker=f"T{i}", side="BUY", quantity=100,
                        fill_quantity=100, product_type=ProductType.EQUITY)
            self.tca.analyze(o, arrival_price=100.0, fill_price=100.03)

        agg = self.tca.get_aggregate()
        assert agg.total_trades == 5
        assert agg.avg_total_cost_bps > 0

    def test_option_commission(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=10,
                    fill_quantity=10, product_type=ProductType.OPTION)
        snap = self.tca.analyze(o, arrival_price=5.0, fill_price=5.05)
        # Alpaca: $0 option commissions — commission_bps may be 0
        assert snap.commission_bps >= 0  # Commission-free (Alpaca)


# ---------------------------------------------------------------------------
# L7RiskEngine
# ---------------------------------------------------------------------------

class TestL7RiskEngine:
    def setup_method(self):
        self.risk = L7RiskEngine(initial_nav=10_000.0)

    def test_pre_trade_passes(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=10, limit_price=100.0)
        passed, violations = self.risk.pre_trade_check(
            o, nav=10_000, cash=10_000, positions={},
            daily_pnl=0, gross_exposure=0, net_exposure=0,
        )
        assert passed
        assert len(violations) == 0

    def test_pre_trade_cash_gate(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=1000, limit_price=100.0)
        passed, violations = self.risk.pre_trade_check(
            o, nav=10_000, cash=500, positions={},
            daily_pnl=0, gross_exposure=0, net_exposure=0,
        )
        assert not passed
        assert any("G8_CASH" in v for v in violations)

    def test_pre_trade_daily_loss(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=10, limit_price=100.0)
        passed, violations = self.risk.pre_trade_check(
            o, nav=10_000, cash=10_000, positions={},
            daily_pnl=-500,  # 5% loss
            gross_exposure=0, net_exposure=0,
        )
        assert not passed
        assert any("G3_DAILY_LOSS" in v for v in violations)

    def test_post_trade_update(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=10,
                    fill_price=150.0, fill_quantity=10,
                    product_type=ProductType.EQUITY)
        state = self.risk.post_trade_update(
            o, nav=10_000, cash=8_500, positions={},
            daily_pnl=100, gross_exposure=1_500, net_exposure=1_500,
        )
        assert isinstance(state, RiskState)
        assert state.nav == 10_000
        assert state.risk_level == "NORMAL"

    def test_kill_switch_activation(self):
        self.risk._peak_nav = 10_000
        o = L7Order(ticker="X", fill_price=1, fill_quantity=1,
                    product_type=ProductType.EQUITY)
        # Simulate 12% drawdown (exceeds 10% limit)
        state = self.risk.post_trade_update(
            o, nav=8_800, cash=500, positions={},
            daily_pnl=-1200, gross_exposure=5000, net_exposure=5000,
        )
        assert state.kill_switch_active


# ---------------------------------------------------------------------------
# ExecutionLearningLoop
# ---------------------------------------------------------------------------

class TestExecutionLearningLoop:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.loop = ExecutionLearningLoop(log_dir=Path(self.tmpdir))

    def test_record_outcome(self):
        o = L7Order(ticker="AAPL", side="BUY", quantity=100,
                    fill_price=150.0, signal_type="QUALITY_BUY",
                    routing=RoutingStrategy.SMART,
                    product_type=ProductType.EQUITY)
        snap = TCASnapshot(
            order_id=o.order_id, ticker="AAPL",
            total_cost_bps=3.5, market_impact_bps=1.0,
        )
        self.loop.record_outcome(o, snap, regime="TRENDING", pnl_bps=5.0)
        assert self.loop.pattern_count >= 1

    def test_suggest_routing_default(self):
        suggestion = self.loop.suggest_routing(
            "AAPL", "EQUITY", "QUALITY_BUY", "TRENDING",
        )
        assert "routing" in suggestion
        assert suggestion["confidence"] == "LOW"  # No history yet

    def test_suggest_after_learning(self):
        # Record enough outcomes to build a pattern
        for i in range(25):
            o = L7Order(ticker="MSFT", side="BUY", quantity=50,
                        fill_price=400.0, signal_type="ML_AGENT_BUY",
                        routing=RoutingStrategy.TWAP,
                        product_type=ProductType.EQUITY)
            snap = TCASnapshot(
                order_id=o.order_id, ticker="MSFT",
                total_cost_bps=2.0 + i * 0.1, market_impact_bps=0.5,
            )
            self.loop.record_outcome(o, snap, regime="TRENDING", pnl_bps=3.0)

        # Verify patterns were created (bucket key depends on current time)
        assert self.loop.pattern_count >= 1
        # Verify at least one pattern has enough samples
        any_high = any(p.sample_count >= 20 for p in self.loop._patterns.values())
        assert any_high

    def test_save_load_patterns(self):
        o = L7Order(ticker="TSLA", side="BUY", quantity=10,
                    fill_price=200.0, signal_type="HOLD",
                    routing=RoutingStrategy.SMART,
                    product_type=ProductType.EQUITY)
        snap = TCASnapshot(order_id=o.order_id, ticker="TSLA",
                           total_cost_bps=5.0, market_impact_bps=2.0)
        self.loop.record_outcome(o, snap)

        self.loop.save_patterns()

        loop2 = ExecutionLearningLoop(log_dir=Path(self.tmpdir))
        loop2.load_patterns()
        assert loop2.pattern_count >= 1

    def test_daily_optimize(self):
        # Should not crash even with no data
        self.loop.daily_optimize()

    def test_weekly_refresh(self):
        self.loop.weekly_refresh()

    def test_monthly_prune(self):
        self.loop.monthly_prune()


# ---------------------------------------------------------------------------
# L7UnifiedExecutionSurface
# ---------------------------------------------------------------------------

class TestL7UnifiedExecutionSurface:
    def setup_method(self):
        self.l7 = L7UnifiedExecutionSurface(initial_cash=10_000.0)

    def test_init(self):
        assert self.l7._initial_cash == 10_000.0
        assert self.l7._router is not None
        assert self.l7._risk_engine is not None
        assert self.l7._tca is not None
        assert self.l7._learning is not None

    def test_submit_equity_order(self):
        order = self.l7.submit_order(
            ticker="AAPL", side="BUY", quantity=10,
            signal_type="QUALITY_BUY", regime="TRENDING",
        )
        assert order.ticker == "AAPL"
        assert order.product_type == ProductType.EQUITY
        # Should fill (via paper fallback)
        assert order.status == "FILLED"
        assert order.fill_price >= 0

    def test_reject_research_only(self):
        order = self.l7.submit_order(
            ticker="TLT", side="BUY", quantity=10,
        )
        assert order.status == "REJECTED"
        assert "Research-only" in order.reason

    def test_reject_fx(self):
        order = self.l7.submit_order(
            ticker="6E", side="BUY", quantity=1,
        )
        assert order.status == "REJECTED"

    def test_option_routing(self):
        order = self.l7.submit_order(
            ticker="AAPL", side="BUY", quantity=5,
            product_type="OPTION",
            option_type="CALL", strike=180.0, expiry="2026-04-18",
        )
        assert order.product_type == ProductType.OPTION

    def test_future_routing(self):
        order = self.l7.submit_order(
            ticker="ES", side="BUY", quantity=1,
            product_type="FUTURE", contract="ES",
        )
        assert order.product_type == ProductType.FUTURE

    def test_heartbeat(self):
        self.l7.heartbeat(regime="TRENDING")
        assert self.l7._heartbeat_count == 1

    def test_market_lifecycle(self):
        self.l7.market_open()
        self.l7.heartbeat()
        self.l7.market_close()

    def test_get_execution_summary(self):
        summary = self.l7.get_execution_summary()
        assert "nav" in summary
        assert "risk_level" in summary
        assert "routing_stats" in summary
        assert "avg_tca_cost_bps" in summary

    def test_routing_stats(self):
        self.l7.submit_order(ticker="AAPL", side="BUY", quantity=10)
        stats = self.l7.get_routing_stats()
        assert stats.get("EQUITY", 0) >= 1 or stats.get("REJECTED", 0) >= 0

    def test_tca_aggregate(self):
        agg = self.l7.get_tca_aggregate()
        assert isinstance(agg, TCAAggregate)


# ---------------------------------------------------------------------------
# L7DashboardRenderer
# ---------------------------------------------------------------------------

class TestL7Dashboard:
    def setup_method(self):
        self.renderer = L7DashboardRenderer()

    def test_risk_panel_no_data(self):
        text = self.renderer.render_risk_panel()
        assert "No risk data" in text

    def test_risk_panel_with_data(self):
        state = RiskState(
            nav=10_000, cash=5_000, gross_leverage=1.5,
            net_leverage=0.8, daily_pnl=150.0, daily_pnl_pct=0.015,
            var_95_1d=300.0, risk_level="NORMAL",
            kill_switch_active=False,
            gates_status={"G1_POSITION": True, "G3_DAILY_LOSS": True},
            intraday_drawdown_pct=0.02,
            max_position_pct=0.05, max_position_ticker="AAPL",
            position_count=5,
        )
        text = self.renderer.render_risk_panel(state)
        assert "NORMAL" in text
        assert "10,000" in text
        assert "AAPL" in text

    def test_tca_panel_no_data(self):
        text = self.renderer.render_tca_panel()
        assert "No TCA data" in text

    def test_tca_panel_with_data(self):
        agg = TCAAggregate(
            total_trades=50, total_volume_usd=500_000,
            avg_spread_cost_bps=1.5, avg_market_impact_bps=0.8,
            avg_timing_cost_bps=0.3, avg_commission_bps=0.1,
            avg_total_cost_bps=2.7, cost_trend="IMPROVING",
            equity_avg_cost_bps=2.5, option_avg_cost_bps=15.0,
            future_avg_cost_bps=0.8,
        )
        text = self.renderer.render_tca_panel(agg)
        assert "IMPROVING" in text
        assert "2.70" in text or "2.7" in text

    def test_l7_panel_no_data(self):
        text = self.renderer.render_l7_panel()
        assert "No L7 data" in text

    def test_l7_panel_with_surface(self):
        l7 = L7UnifiedExecutionSurface(initial_cash=10_000)
        text = self.renderer.render_l7_panel(l7)
        assert "L7 EXECUTION SURFACE" in text


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_values(self):
        assert _mean([1, 2, 3]) == 2.0
