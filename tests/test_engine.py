"""Tests for Metadron Capital investment engine.

Tests all 6 layers + learning loop + asset class routing:
    L1 Data (UniverseEngine, OpenBB data, asset class filtering)
    L2 Signals (MacroEngine, MetadronCube, GMTF)
    L3 ML (AlphaOptimizer)
    L4 Portfolio (BetaCorridor)
    L5 Execution (PaperBroker, TradierBroker, ExecutionEngine)
    L6 Agents (SectorBots, Scorecard)
    Learning Loop (signal accuracy, regime feedback, tier weights)
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data.universe_engine import (
    UniverseEngine, Security, GICS_SECTORS, SECTOR_ETFS, RV_PAIRS, ALL_ETFS,
)
from engine.signals.macro_engine import (
    MacroEngine, MacroSnapshot, MarketRegime, CubeRegime,
    sigmoid_trigger, compute_gammas,
)
from engine.signals.metadron_cube import (
    MetadronCube, CubeOutput, SleeveAllocation, REGIME_PARAMS,
    R_LOW, R_HIGH, BETA_MAX, BETA_INV,
)
from engine.ml.alpha_optimizer import (
    AlphaOptimizer, AlphaOutput, AlphaSignal,
    build_features, ewma_cov, classify_quality,
)
from engine.portfolio.beta_corridor import (
    BetaCorridor, BetaState, BetaAction,
    ALPHA, EXECUTION_MULTIPLIER, VOL_STANDARD,
)
from engine.execution.paper_broker import (
    PaperBroker, OrderSide, OrderStatus, SignalType, Position,
)
from engine.agents.sector_bots import (
    SectorBot, SectorBotManager, AgentScorecard, AgentTier,
)


# ===========================================================================
# L1: Universe Engine
# ===========================================================================
class TestUniverseEngine:
    def test_gics_sectors(self):
        assert len(GICS_SECTORS) == 11
        assert 45 in GICS_SECTORS  # Information Technology

    def test_sector_etfs(self):
        assert len(SECTOR_ETFS) == 11
        assert SECTOR_ETFS["Energy"] == "XLE"

    def test_rv_pairs(self):
        assert len(RV_PAIRS) == 26
        assert ("AAPL", "MSFT") in RV_PAIRS

    def test_all_etfs(self):
        assert len(ALL_ETFS) >= 30

    def test_universe_engine_init(self):
        ue = UniverseEngine()
        assert ue.size() >= 0  # May be 0 if no network

    def test_security_dataclass(self):
        sec = Security(ticker="AAPL", name="Apple", sector="Information Technology")
        assert sec.ticker == "AAPL"
        assert sec.quality_tier == "D"  # Default

    def test_screen(self):
        ue = UniverseEngine()
        ue._equities = [
            Security(ticker="A", sector="Energy", market_cap=100e9),
            Security(ticker="B", sector="Energy", market_cap=5e9),
            Security(ticker="C", sector="Financials", market_cap=50e9),
        ]
        ue._loaded = True
        result = ue.screen(sectors=["Energy"])
        assert len(result) == 2
        result = ue.screen(min_market_cap=10e9)
        assert len(result) == 2


# ===========================================================================
# L2: Signals (MacroEngine, GMTF, MetadronCube)
# ===========================================================================
class TestMacroEngine:
    def test_sigmoid_trigger(self):
        # At threshold → ~0.5
        val = sigmoid_trigger(0.01, 0.01, sensitivity=15)
        assert 0.4 < val < 0.6

        # Well above → ~1.0
        val = sigmoid_trigger(1.0, 0.01, sensitivity=15)
        assert val > 0.99

        # Well below → ~0.0
        val = sigmoid_trigger(-1.0, 0.01, sensitivity=15)
        assert val < 0.01

    def test_regime_classification(self):
        engine = MacroEngine()
        # Bullish snapshot
        snap = MacroSnapshot(vix=12, spy_return_3m=0.15, yield_spread=1.0, credit_spread=2.0)
        regime = engine._classify_regime(snap)
        assert regime == MarketRegime.BULL

        # Stress snapshot
        snap = MacroSnapshot(vix=40, spy_return_3m=-0.15, yield_spread=-1.0, credit_spread=6.0)
        regime = engine._classify_regime(snap)
        assert regime == MarketRegime.STRESS

    def test_cube_regime_mapping(self):
        engine = MacroEngine()
        snap = MacroSnapshot(vix=50)
        assert engine._classify_cube_regime(snap) == CubeRegime.CRASH

        snap = MacroSnapshot(vix=12, spy_return_3m=0.15, yield_spread=1.0, credit_spread=2.0)
        snap.regime = MarketRegime.BULL
        assert engine._classify_cube_regime(snap) == CubeRegime.TRENDING


class TestMetadronCube:
    def test_init(self):
        cube = MetadronCube()
        assert cube.get_last() is None

    def test_compute_basic(self):
        cube = MetadronCube()
        macro = MacroSnapshot(
            regime=MarketRegime.BULL,
            cube_regime=CubeRegime.TRENDING,
            vix=15, spy_return_3m=0.10,
            yield_10y=4.0, yield_2y=3.5,
            credit_spread=2.0,
            sector_rankings={"Information Technology": 0.5, "Financials": 0.3},
        )
        out = cube.compute(macro)
        assert isinstance(out, CubeOutput)
        assert out.regime == CubeRegime.TRENDING
        assert out.max_leverage == 3.0
        assert out.beta_cap == 0.65

    def test_sleeve_allocation_sums_to_one(self):
        cube = MetadronCube()
        for regime in CubeRegime:
            macro = MacroSnapshot(
                regime=MarketRegime.TRANSITION,
                cube_regime=regime,
                vix=20, spy_return_3m=0.05,
                sector_rankings={},
            )
            out = cube.compute(macro)
            total = out.sleeves.total()
            assert abs(total - 1.0) < 0.02, f"Sleeves don't sum to 1 for {regime}: {total}"

    def test_regime_params(self):
        assert REGIME_PARAMS[CubeRegime.TRENDING]["max_leverage"] == 3.0
        assert REGIME_PARAMS[CubeRegime.CRASH]["beta_cap"] == -0.20

    def test_target_beta_corridor(self):
        cube = MetadronCube()
        # At R_LOW → should be near BETA_INV scaled
        beta_low = cube._compute_target_beta(R_LOW, 0.15)
        # At R_HIGH → should be positive
        beta_high = cube._compute_target_beta(R_HIGH, 0.15)
        assert beta_high > beta_low
        # Caps
        assert beta_high <= BETA_MAX
        assert beta_low >= BETA_INV

    def test_risk_state(self):
        cube = MetadronCube()
        macro = MacroSnapshot(vix=40, credit_spread=5.0)
        risk = cube._compute_risk(macro)
        assert 0 <= risk.value <= 1
        assert risk.vix_component > 0.5  # High VIX


# ===========================================================================
# L3: ML (AlphaOptimizer)
# ===========================================================================
class TestAlphaOptimizer:
    def test_classify_quality(self):
        assert classify_quality(2.5, 0.20) == "A"
        assert classify_quality(0.6, 0.02) == "D"
        assert classify_quality(-1.0, -0.20) == "G"

    def test_build_features(self):
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        returns = pd.DataFrame(
            np.random.randn(n, 3) * 0.01,
            index=dates, columns=["A", "B", "C"],
        )
        prices = (1 + returns).cumprod() * 100
        features = build_features(returns, prices)
        assert "mkt_mean" in features.columns
        assert "momentum_3m" in features.columns
        assert len(features) == n

    def test_ewma_cov(self):
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 3) * 0.01,
            columns=["A", "B", "C"],
        )
        cov = ewma_cov(returns)
        assert cov.shape == (3, 3)
        assert np.all(np.diag(cov) > 0)  # Positive variance
        # Symmetric
        assert np.allclose(cov, cov.T, atol=1e-10)


# ===========================================================================
# L4: Portfolio (BetaCorridor)
# ===========================================================================
class TestBetaCorridor:
    def test_init(self):
        bc = BetaCorridor(nav=1_000_000)
        assert bc.nav == 1_000_000
        assert bc.current_beta == 0.0

    def test_target_beta_at_boundaries(self):
        bc = BetaCorridor()
        # Below corridor
        state = bc.calculate_target_beta(Rm=0.05, sigma_m=0.15)
        assert state.corridor_position == "BELOW"

        # Within corridor
        state = bc.calculate_target_beta(Rm=0.09, sigma_m=0.15)
        assert state.corridor_position == "WITHIN"

        # Above corridor
        state = bc.calculate_target_beta(Rm=0.15, sigma_m=0.15)
        assert state.corridor_position == "ABOVE"

    def test_vol_adjustment(self):
        bc = BetaCorridor()
        # Normal vol (15%)
        state_normal = bc.calculate_target_beta(Rm=0.10, sigma_m=0.15)
        # High vol (30%) → beta should contract
        state_high = bc.calculate_target_beta(Rm=0.10, sigma_m=0.30)
        assert abs(state_high.target_beta) < abs(state_normal.target_beta)

    def test_rebalance_hold(self):
        bc = BetaCorridor()
        state = BetaState(target_beta=0.01, current_beta=0.0)
        bc.current_beta = 0.0
        action = bc.compute_rebalance(state, instrument_price=500)
        assert action.action == "HOLD"  # Delta < threshold

    def test_rebalance_buy(self):
        bc = BetaCorridor(nav=1_000_000)
        state = BetaState(target_beta=0.5)
        bc.current_beta = 0.0
        action = bc.compute_rebalance(state, instrument_price=500)
        assert action.action == "BUY"
        assert action.quantity > 0

    def test_alpha_headstart(self):
        assert ALPHA == 0.02  # 2% secular alpha


# ===========================================================================
# L5: Execution (PaperBroker)
# ===========================================================================
class TestPaperBroker:
    def test_init(self):
        broker = PaperBroker(initial_cash=1_000_000)
        assert broker.state.cash == 1_000_000
        assert broker.state.nav == 1_000_000

    def test_place_order_buy(self):
        broker = PaperBroker(initial_cash=100_000)
        # Mock price by directly updating
        order = broker.place_order("TEST", OrderSide.BUY, 100, reason="test")
        # If no price data, should be rejected
        if order.status == OrderStatus.REJECTED:
            assert "No price" in order.reason
        # If filled, check state
        elif order.status == OrderStatus.FILLED:
            assert broker.state.cash < 100_000
            assert "TEST" in broker.state.positions

    def test_signal_types(self):
        assert len(SignalType) == 29  # 15 original + 4 social + 3 distress + 2 CVR + 4 event + HOLD
        assert SignalType.MICRO_PRICE_BUY.value == "MICRO_PRICE_BUY"

    def test_portfolio_summary(self):
        broker = PaperBroker(initial_cash=500_000)
        summary = broker.get_portfolio_summary()
        assert summary["cash"] == 500_000
        assert summary["positions"] == 0
        assert summary["nav"] == 500_000


# ===========================================================================
# L6: Agents (SectorBots, Scorecard)
# ===========================================================================
class TestSectorBots:
    def test_sector_bot_init(self):
        bot = SectorBot(sector="Energy", gics_code=10, etf="XLE")
        assert bot.sector == "Energy"
        assert bot.is_active

    def test_sector_bot_manager_init(self):
        mgr = SectorBotManager()
        assert len(mgr.bots) == 11  # 11 GICS sectors
        assert "Energy" in mgr.bots
        assert "Information Technology" in mgr.bots

    def test_agent_scorecard(self):
        sc = AgentScorecard()
        sc.update_score("EnergyBot", accuracy=0.85, sharpe=2.5, hit_rate=0.80, is_top_this_week=True)
        lb = sc.get_leaderboard()
        assert len(lb) == 1
        assert lb[0][0] == "EnergyBot"
        assert lb[0][1].accuracy == 0.85

    def test_tier_promotion(self):
        sc = AgentScorecard()
        # 4 consecutive top weeks → should promote to General
        for _ in range(4):
            sc.update_score("TopBot", accuracy=0.85, sharpe=2.5, hit_rate=0.80, is_top_this_week=True)
        assert sc.scores["TopBot"].tier == AgentTier.GENERAL

    def test_tier_demotion(self):
        sc = AgentScorecard()
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55)
        assert sc.scores["WeakBot"].tier == AgentTier.CAPTAIN
        # 2 consecutive bottom weeks
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55, is_bottom_this_week=True)
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55, is_bottom_this_week=True)
        assert sc.scores["WeakBot"].tier == AgentTier.LIEUTENANT  # Demoted

    def test_leaderboard_format(self):
        sc = AgentScorecard()
        sc.update_score("Bot_A", accuracy=0.90, sharpe=3.0, hit_rate=0.85)
        sc.update_score("Bot_B", accuracy=0.50, sharpe=0.5, hit_rate=0.40)
        text = sc.print_leaderboard()
        assert "Bot_A" in text
        assert "Bot_B" in text
        assert "Rank" in text


# ===========================================================================
# Integration
# ===========================================================================
class TestIntegration:
    def test_full_cube_pipeline(self):
        """Test Macro → Cube → Beta target."""
        macro = MacroEngine()
        snap = MacroSnapshot(
            regime=MarketRegime.BULL,
            cube_regime=CubeRegime.TRENDING,
            vix=14, spy_return_3m=0.12,
            yield_10y=4.2, yield_2y=3.8,
            credit_spread=2.5,
            sector_rankings={"Information Technology": 0.6, "Financials": 0.4},
        )
        cube = MetadronCube()
        out = cube.compute(snap)

        assert out.regime == CubeRegime.TRENDING
        assert out.target_beta > 0  # Bull regime → positive beta
        assert out.sleeves.p1_directional_equity > 0.10  # Carry allocation in trending

        # Beta corridor check
        bc = BetaCorridor()
        state = bc.calculate_target_beta(
            Rm=snap.spy_return_3m * 4,  # annualise
            sigma_m=snap.vix / 100,
            regime_beta_cap=out.beta_cap,
        )
        assert state.target_beta <= out.beta_cap

    def test_signal_type_coverage(self):
        """Ensure all signal types are defined."""
        expected = [
            "MICRO_PRICE_BUY", "MICRO_PRICE_SELL",
            "RV_LONG", "RV_SHORT", "FALLEN_ANGEL_BUY",
            "ML_AGENT_BUY", "ML_AGENT_SELL",
            "DRL_AGENT_BUY", "DRL_AGENT_SELL",
            "TFT_BUY", "TFT_SELL",
            "MC_BUY", "MC_SELL",
            "QUALITY_BUY", "QUALITY_SELL",
            "HOLD",
            # Distress, CVR, and Event signals
            "DISTRESS_FALLEN_ANGEL", "DISTRESS_RECOVERY", "DISTRESS_AVOID",
            "CVR_BUY", "CVR_SELL",
            "EVENT_MERGER_ARB", "EVENT_PEAD_LONG", "EVENT_PEAD_SHORT", "EVENT_CATALYST",
        ]
        for sig in expected:
            assert hasattr(SignalType, sig)


# ===========================================================================
# Distressed Asset Engine
# ===========================================================================
class TestDistressedAssetEngine:
    def test_init(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        assert len(dae.universe) >= 11  # Dynamic: credit classification E/F tiers + static fallback

    def test_analyze(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine, DistressLevel
        dae = DistressedAssetEngine()
        results = dae.analyze()
        assert len(results) >= 11  # Dynamic universe from credit classification
        for ticker, score in results.items():
            assert score.ticker == ticker
            assert isinstance(score.level, DistressLevel)
            assert 0 <= score.ensemble_prob <= 1
            assert score.feature_count == 40

    def test_altman_z_zones(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        # Low leverage → should be SAFE or GREY
        safe_result = dae._altman_z("TEST", {"debt_to_equity": 0.5, "market_cap_B": 50, "interest_coverage": 8})
        assert safe_result.zone in ("SAFE", "GREY")
        # High leverage → should be GREY or DISTRESS
        risky_result = dae._altman_z("TEST", {"debt_to_equity": 8.0, "market_cap_B": 2, "interest_coverage": 0.5})
        assert risky_result.z_score < safe_result.z_score

    def test_merton_kvv_convergence(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        result = dae._merton_kmv("TEST", {"debt_to_equity": 2.0, "market_cap_B": 10, "interest_coverage": 3})
        assert result.iterations < 100  # Should converge
        assert result.distance_to_default > 0
        assert 0 <= result.default_probability <= 1

    def test_fallen_angels(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        dae.analyze()
        angels = dae.get_fallen_angels()
        assert isinstance(angels, list)

    def test_distress_signals(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        dae.analyze()
        signals = dae.get_distress_signals()
        assert isinstance(signals, dict)
        for ticker, sig in signals.items():
            assert "ensemble_prob" in sig
            assert "level" in sig
            assert "merton_dd" in sig

    def test_report_format(self):
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        dae.analyze()
        report = dae.format_distress_report()
        assert "DISTRESSED ASSET ENGINE" in report
        assert "5-Model Ensemble" in report


# ===========================================================================
# CVR Engine
# ===========================================================================
class TestCVREngine:
    def test_init(self):
        from engine.signals.cvr_engine import CVREngine
        cvr = CVREngine()
        assert len(cvr.catalog) == 4

    def test_analyze(self):
        from engine.signals.cvr_engine import CVREngine
        cvr = CVREngine()
        results = cvr.analyze()
        assert len(results) == 4
        for ticker, val in results.items():
            assert val.fair_value >= 0
            assert val.market_price >= 0

    def test_binary_option(self):
        from engine.signals.cvr_engine import CVREngine, CVRInstrument, CVRType
        cvr = CVREngine()
        inst = CVRInstrument(
            ticker="TEST", name="Test CVR", cvr_type=CVRType.PHARMA_MILESTONE,
            payment_usd=5.0, market_price=2.0, expiry_months=24,
            acquirer_rating="A",
        )
        val = cvr._binary_option(inst, trigger_prob=0.6)
        assert val > 0
        assert val < 5.0  # Less than full payment

    def test_barrier_option(self):
        from engine.signals.cvr_engine import CVREngine, CVRInstrument, CVRType
        cvr = CVREngine()
        inst = CVRInstrument(
            ticker="TEST", name="Test", cvr_type=CVRType.REVENUE_EARNOUT,
            payment_usd=10.0, market_price=4.0, trigger_price=100,
            underlying_price=80, underlying_vol=0.30, expiry_months=36,
        )
        val = cvr._barrier_option(inst)
        assert val > 0

    def test_milestone_tree(self):
        from engine.signals.cvr_engine import CVREngine, CVRInstrument, CVRType, MilestoneStage
        cvr = CVREngine()
        inst = CVRInstrument(
            ticker="TEST", name="Test", cvr_type=CVRType.PHARMA_MILESTONE,
            payment_usd=5.0, market_price=2.0, expiry_months=36,
            milestones=[
                MilestoneStage("Phase II", 0.60, 12),
                MilestoneStage("Phase III", 0.50, 18),
            ],
        )
        val, prob = cvr._milestone_tree(inst)
        assert prob == pytest.approx(0.30, abs=0.01)  # 0.6 * 0.5
        assert val > 0

    def test_trading_signals(self):
        from engine.signals.cvr_engine import CVREngine
        cvr = CVREngine()
        cvr.analyze()
        signals = cvr.get_trading_signals()
        assert isinstance(signals, dict)
        for ticker, sig in signals.items():
            assert "signal" in sig
            assert "fair_value" in sig
            assert "mispricing_pct" in sig

    def test_report_format(self):
        from engine.signals.cvr_engine import CVREngine
        cvr = CVREngine()
        cvr.analyze()
        report = cvr.format_cvr_report()
        assert "CVR ENGINE" in report
        assert "Contingent Value Rights" in report


# ===========================================================================
# Event-Driven Engine
# ===========================================================================
class TestEventDrivenEngine:
    def test_init(self):
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        assert len(ede.events) == 10

    def test_analyze(self):
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        result = ede.analyze()
        assert result.total_events == 10
        assert len(result.positions) == 10
        assert result.weighted_expected_alpha_bps != 0

    def test_merger_arb(self):
        from engine.signals.event_driven_engine import EventDrivenEngine, EventCategory, DealStatus
        ede = EventDrivenEngine()
        event = {
            "type": EventCategory.MERGER_ARB,
            "ticker": "TEST",
            "acquirer": "BIG_CO",
            "deal_price": 100.0,
            "current_price": 95.0,
            "days_to_close": 60,
            "status": DealStatus.REGULATORY_REVIEW,
        }
        arb, pos = ede._merger_arb(event)
        assert arb.gross_spread_pct > 0
        assert arb.deal_break_prob < 0.20
        assert arb.expected_return > 0  # Positive expected return for typical spread

    def test_pead(self):
        from engine.signals.event_driven_engine import EventDrivenEngine, EventCategory
        ede = EventDrivenEngine()
        event = {
            "type": EventCategory.PEAD,
            "ticker": "TEST",
            "sue_zscore": 2.5,
            "surprise_pct": 0.15,
            "revision_momentum": 0.10,
            "days_since": 5,
        }
        pead, pos = ede._pead(event)
        assert pead.drift_alpha_bps > 0  # Positive SUE → positive drift
        assert pead.decay_factor > 0.5   # Recent → low decay

    def test_trading_signals(self):
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        ede.analyze()
        signals = ede.get_trading_signals()
        assert isinstance(signals, dict)
        assert len(signals) > 0
        for ticker, sig in signals.items():
            assert "signal" in sig
            assert "expected_alpha_bps" in sig
            assert "conviction" in sig
            assert "kelly_fraction" in sig

    def test_top_ideas(self):
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        ede.analyze()
        ideas = ede.get_top_ideas(min_alpha_bps=50)
        assert isinstance(ideas, list)
        # Should have several ideas above 50bps
        assert len(ideas) >= 3

    def test_report_format(self):
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        ede.analyze()
        report = ede.format_event_report()
        assert "EVENT-DRIVEN ENGINE" in report
        assert "MERGER ARBITRAGE DETAIL" in report
        assert "PEAD SIGNALS" in report


# ===========================================================================
# Asset Class Routing
# ===========================================================================
class TestAssetClassRouting:
    """Test asset class classification and tradeable vs macro-only filtering."""

    def test_asset_class_enum(self):
        from engine.data.universe_engine import AssetClass
        assert AssetClass.TRADEABLE.value == "TRADEABLE"
        assert AssetClass.MACRO_ONLY.value == "MACRO_ONLY"

    def test_macro_only_tickers(self):
        from engine.data.universe_engine import MACRO_ONLY_TICKERS
        # Fixed income, commodity, volatility ETFs are macro-only
        assert "TLT" in MACRO_ONLY_TICKERS   # Treasury 20Y+
        assert "GLD" in MACRO_ONLY_TICKERS   # Gold
        assert "VXX" in MACRO_ONLY_TICKERS   # VIX
        assert "HYG" in MACRO_ONLY_TICKERS   # High Yield
        assert "USO" in MACRO_ONLY_TICKERS   # Oil
        # Sector ETFs are NOT macro-only
        assert "XLK" not in MACRO_ONLY_TICKERS
        assert "SPY" not in MACRO_ONLY_TICKERS

    def test_tradeable_etfs(self):
        from engine.data.universe_engine import TRADEABLE_ETFS
        # Sector, index, factor, thematic ETFs are tradeable
        assert "XLK" in TRADEABLE_ETFS
        assert "SPY" in TRADEABLE_ETFS
        assert "QQQ" in TRADEABLE_ETFS
        assert "MTUM" in TRADEABLE_ETFS
        assert "SMH" in TRADEABLE_ETFS
        # Commodity/bond ETFs are NOT tradeable
        assert "TLT" not in TRADEABLE_ETFS
        assert "GLD" not in TRADEABLE_ETFS
        assert "VXX" not in TRADEABLE_ETFS

    def test_security_is_tradeable(self):
        from engine.data.universe_engine import Security, SecurityType
        # Equities are tradeable
        eq = Security(ticker="AAPL", security_type=SecurityType.EQUITY.value)
        assert eq.is_tradeable
        assert not eq.is_macro_only

    def test_security_macro_only(self):
        from engine.data.universe_engine import Security
        # TLT is a known macro-only ticker
        bond = Security(ticker="TLT", security_type="FIXED_INCOME_ETF")
        assert bond.is_macro_only
        assert not bond.is_tradeable

    def test_universe_tradeable_filter(self):
        from engine.data.universe_engine import UniverseEngine
        ue = UniverseEngine(load=True)
        all_secs = ue.get_all()
        tradeable = ue.get_tradeable()
        # All equities should be tradeable
        assert len(tradeable) == len(all_secs)  # Universe is all equities
        # Specific ticker checks
        assert ue.is_ticker_tradeable("AAPL")
        assert ue.is_ticker_tradeable("SPY")
        assert ue.is_ticker_tradeable("XLK")
        assert not ue.is_ticker_tradeable("TLT")
        assert not ue.is_ticker_tradeable("GLD")
        assert not ue.is_ticker_tradeable("VXX")

    def test_macro_only_list(self):
        from engine.data.universe_engine import UniverseEngine
        ue = UniverseEngine(load=True)
        macro = ue.get_macro_only()
        assert "TLT" in macro
        assert "GLD" in macro
        assert "VXX" in macro
        assert "AAPL" not in macro

    def test_security_type_mapping(self):
        from engine.data.universe_engine import (
            SecurityType, AssetClass, SECURITY_TYPE_ASSET_CLASS,
        )
        assert SECURITY_TYPE_ASSET_CLASS[SecurityType.EQUITY] == AssetClass.TRADEABLE
        assert SECURITY_TYPE_ASSET_CLASS[SecurityType.SECTOR_ETF] == AssetClass.TRADEABLE
        assert SECURITY_TYPE_ASSET_CLASS[SecurityType.FIXED_INCOME_ETF] == AssetClass.MACRO_ONLY
        assert SECURITY_TYPE_ASSET_CLASS[SecurityType.COMMODITY_ETF] == AssetClass.MACRO_ONLY
        assert SECURITY_TYPE_ASSET_CLASS[SecurityType.VOLATILITY_ETF] == AssetClass.MACRO_ONLY


# ===========================================================================
# Learning Loop
# ===========================================================================
class TestLearningLoop:
    """Test the closed-loop feedback system."""

    def test_import(self):
        from engine.monitoring.learning_loop import (
            LearningLoop, SignalOutcome, RegimeFeedback,
            EngineAccuracy, LearningSnapshot,
        )
        assert LearningLoop is not None

    def test_init(self):
        from engine.monitoring.learning_loop import LearningLoop
        ll = LearningLoop()
        assert ll._total_events == 0
        assert len(ll.SIGNAL_ENGINES) >= 10

    def test_record_signal_outcome(self):
        from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
        ll = LearningLoop()
        outcome = SignalOutcome(
            ticker="AAPL", signal_engine="ml_ensemble",
            signal_type="ML_AGENT_BUY", side="BUY",
            entry_price=150.0, realized_pnl=500.0,
            was_correct=True, vote_score=3.5,
        )
        ll.record_signal_outcome(outcome)
        assert ll._total_events == 1
        stats = ll.get_engine_stats("ml_ensemble")
        assert stats.total_signals == 1
        assert stats.accuracy == 1.0
        assert stats.total_pnl == 500.0

    def test_multiple_outcomes(self):
        from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
        ll = LearningLoop()
        for i in range(10):
            ll.record_signal_outcome(SignalOutcome(
                ticker=f"T{i}", signal_engine="macro",
                was_correct=(i % 3 != 0),
                realized_pnl=100 if i % 3 != 0 else -50,
            ))
        stats = ll.get_engine_stats("macro")
        assert stats.total_signals == 10
        assert 0.5 < stats.accuracy < 0.8

    def test_tier_weight_adjustments(self):
        from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
        ll = LearningLoop()
        # Feed lots of accurate signals to ml_ensemble
        for i in range(20):
            ll.record_signal_outcome(SignalOutcome(
                ticker=f"T{i}", signal_engine="ml_ensemble",
                was_correct=True, realized_pnl=100,
            ))
        adj = ll.compute_tier_weight_adjustments()
        # T1_neural should get a weight boost
        assert adj["T1_neural"] >= ll.DEFAULT_TIER_WEIGHTS["T1_neural"]

    def test_regime_feedback(self):
        from engine.monitoring.learning_loop import LearningLoop, RegimeFeedback
        ll = LearningLoop()
        ll.record_regime_feedback(RegimeFeedback(
            predicted_regime="TRENDING",
            actual_market_behavior="BULL",
            regime_correct=True,
        ))
        ll.record_regime_feedback(RegimeFeedback(
            predicted_regime="TRENDING",
            actual_market_behavior="BEAR",
            regime_correct=False,
        ))
        assert ll.compute_regime_accuracy() == 0.5

    def test_sector_feedback(self):
        from engine.monitoring.learning_loop import LearningLoop
        ll = LearningLoop()
        ll.record_sector_feedback("Information Technology", "OVERWEIGHT", 0.05)
        ll.record_sector_feedback("Energy", "OVERWEIGHT", -0.03)
        acc = ll.get_sector_allocation_accuracy()
        assert acc["Information Technology"] == 1.0
        assert acc["Energy"] == 0.0

    def test_snapshot(self):
        from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
        ll = LearningLoop()
        ll.record_signal_outcome(SignalOutcome(
            ticker="AAPL", signal_engine="social",
            was_correct=True, realized_pnl=200,
        ))
        snap = ll.get_snapshot()
        assert snap.total_learning_events == 1
        assert "social" in snap.engine_accuracies

    def test_learning_report(self):
        from engine.monitoring.learning_loop import LearningLoop, SignalOutcome
        ll = LearningLoop()
        ll.record_signal_outcome(SignalOutcome(
            ticker="MSFT", signal_engine="event_driven",
            was_correct=True, realized_pnl=300,
        ))
        report = ll.format_learning_report()
        assert "LEARNING LOOP" in report
        assert "event_driven" in report

    def test_execution_engine_has_learning(self):
        """ExecutionEngine should have learning loop."""
        from engine.execution.execution_engine import ExecutionEngine
        import inspect
        # Verify the learning attribute exists in __init__
        source = inspect.getsource(ExecutionEngine.__init__)
        assert "self.learning" in source
        assert "LearningLoop" in source

    def test_execution_engine_has_asset_gate(self):
        """ExecutionEngine should have asset class filtering."""
        from engine.execution.execution_engine import ExecutionEngine
        assert hasattr(ExecutionEngine, "_filter_tradeable")


# ===========================================================================
# Tradier Broker
# ===========================================================================
class TestAlpacaBrokerModule:
    """Unit tests for AlpacaBroker — no live API calls (imports + interface)."""

    def test_import(self):
        from engine.execution.alpaca_broker import (
            AlpacaBroker, _ALPACA_BASE_URLS, _ALPACA_STATUS_MAP,
        )
        assert True in _ALPACA_BASE_URLS
        assert False in _ALPACA_BASE_URLS
        assert _ALPACA_BASE_URLS[True] == "https://paper-api.alpaca.markets"
        assert _ALPACA_BASE_URLS[False] == "https://api.alpaca.markets"

    def test_status_mapping(self):
        from engine.execution.alpaca_broker import _ALPACA_STATUS_MAP
        assert _ALPACA_STATUS_MAP.get("filled") == OrderStatus.FILLED
        assert _ALPACA_STATUS_MAP.get("new") == OrderStatus.PENDING
        assert _ALPACA_STATUS_MAP.get("canceled") == OrderStatus.CANCELLED
        assert _ALPACA_STATUS_MAP.get("rejected") == OrderStatus.REJECTED

    def test_broker_interface_matches_paper(self):
        """Verify AlpacaBroker has the same public methods as PaperBroker."""
        from engine.execution.alpaca_broker import AlpacaBroker
        required_methods = [
            "place_order", "compute_nav", "compute_exposures",
            "get_position", "get_all_positions", "get_portfolio_summary",
            "refresh_prices", "reconcile", "get_risk_profile",
            "get_daily_target_state", "reset_daily_target",
            "get_leverage_multiplier", "emit_dashboard_state",
            "get_dashboard_snapshot", "get_dashboard_history",
            "register_dashboard_callback", "get_trade_history",
            "get_daily_pnl", "get_drawdown", "get_performance_metrics",
            "export_positions_csv",
            # Alpaca-specific
            "get_orders", "cancel_order", "get_gainloss",
            "preview_order", "get_asset", "get_clock", "get_portfolio_history",
            "close_position", "close_all_positions",
        ]
        for method in required_methods:
            assert hasattr(AlpacaBroker, method), f"AlpacaBroker missing method: {method}"

    def test_alpaca_broker_init_signatures(self):
        """Verify __init__ accepts same params as TradierBroker."""
        from engine.execution.alpaca_broker import AlpacaBroker
        import inspect
        sig = inspect.signature(AlpacaBroker.__init__)
        params = list(sig.parameters.keys())
        assert "initial_cash" in params
        assert "log_dir" in params
        assert "api_key" in params
        assert "secret_key" in params
        assert "paper" in params

    def test_execution_engine_accepts_alpaca(self):
        """ExecutionEngine should accept broker_type='alpaca'."""
        from engine.execution.execution_engine import ExecutionEngine
        import inspect
        sig = inspect.signature(ExecutionEngine.__init__)
        assert "broker_type" in sig.parameters
        # Check AlpacaBroker import exists in source
        source = inspect.getsource(ExecutionEngine)
        assert "AlpacaBroker" in source
        assert "alpaca" in source

    def test_alpaca_broker_constants(self):
        """Verify commission mapping (Alpaca is $0)."""
        from engine.execution.alpaca_broker import AlpacaBroker
        # Commission is hardcoded as 0.0 in _log_trade and reconcile
        import inspect
        source = inspect.getsource(AlpacaBroker._log_trade)
        assert "commission" in source
        assert "0.0" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
