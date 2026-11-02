"""
Metadron Capital — Prometheus Metrics Endpoint

A FastAPI router that exports Prometheus-format metrics for Grafana,
Datadog, or any Prometheus-compatible monitoring stack.

Mount this router on the engine API server:
    from engine.bridges.prometheus_metrics import create_metrics_router
    app.include_router(create_metrics_router(app))

The /metrics endpoint returns metrics in Prometheus text exposition format.

Requires: prometheus_client (pip install prometheus-client)
"""

import os
import sys
import time
import logging
from typing import Optional

logger = logging.getLogger("prometheus-metrics")

# ─── Prometheus Client Availability ────────────────────────────────

_prometheus_available = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _prometheus_available = True
except ImportError:
    logger.warning("prometheus_client not installed — metrics endpoint will return 503")


# ─── Metric Definitions ───────────────────────────────────────────

def _create_metrics(registry: "CollectorRegistry"):
    """Create all Prometheus metric objects in the given registry."""
    metrics = {}

    # Engine health
    metrics["engine_up"] = Gauge(
        "metadron_engine_up",
        "Whether the Metadron engine API is running (1=up, 0=down)",
        registry=registry,
    )

    # API request counters and latency
    metrics["api_requests_total"] = Counter(
        "metadron_api_requests_total",
        "Total API requests by endpoint, method, and status",
        ["endpoint", "method", "status"],
        registry=registry,
    )
    metrics["api_duration_seconds"] = Histogram(
        "metadron_api_duration_seconds",
        "API request duration in seconds by endpoint",
        ["endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=registry,
    )

    # Portfolio metrics
    metrics["portfolio_nav"] = Gauge(
        "metadron_portfolio_nav",
        "Current portfolio net asset value",
        registry=registry,
    )
    metrics["portfolio_pnl_daily"] = Gauge(
        "metadron_portfolio_pnl_daily",
        "Daily portfolio profit and loss",
        registry=registry,
    )
    metrics["positions_count"] = Gauge(
        "metadron_positions_count",
        "Number of open positions",
        registry=registry,
    )

    # Cube/regime metrics
    metrics["cube_signal_score"] = Gauge(
        "metadron_cube_signal_score",
        "MetadronCube composite signal score",
        registry=registry,
    )
    metrics["cube_regime"] = Gauge(
        "metadron_cube_regime",
        "MetadronCube regime state (1=active for that regime)",
        ["regime_name"],
        registry=registry,
    )

    # Trade metrics
    metrics["trades_total"] = Counter(
        "metadron_trades_total",
        "Total trades executed by side",
        ["side"],
        registry=registry,
    )

    # OpenBB data metrics
    metrics["openbb_requests_total"] = Counter(
        "metadron_openbb_requests_total",
        "Total OpenBB data requests by endpoint",
        ["endpoint"],
        registry=registry,
    )
    metrics["openbb_errors_total"] = Counter(
        "metadron_openbb_errors_total",
        "Total OpenBB data errors by endpoint",
        ["endpoint"],
        registry=registry,
    )

    # LLM metrics
    metrics["llm_requests_total"] = Counter(
        "metadron_llm_requests_total",
        "Total LLM inference requests by backend",
        ["backend"],
        registry=registry,
    )
    metrics["llm_duration_seconds"] = Histogram(
        "metadron_llm_duration_seconds",
        "LLM inference duration in seconds by backend",
        ["backend"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=registry,
    )

    # STRAT engine health gauges (1=healthy, 0=degraded/offline)
    metrics["strat_engine_health"] = Gauge(
        "metadron_strat_engine_health",
        "STRAT engine health (1=healthy, 0=degraded)",
        ["engine"],
        registry=registry,
    )

    # VolatilitySurface metrics
    metrics["vol_surface_iv"] = Gauge(
        "metadron_vol_surface_iv",
        "VolatilitySurface current implied volatility",
        registry=registry,
    )
    metrics["vol_surface_skew"] = Gauge(
        "metadron_vol_surface_skew",
        "VolatilitySurface skew value",
        registry=registry,
    )
    metrics["vol_surface_term_structure"] = Gauge(
        "metadron_vol_surface_term_structure",
        "VolatilitySurface term-structure slope",
        registry=registry,
    )

    # StatArbEngine metrics
    metrics["stat_arb_pairs_count"] = Gauge(
        "metadron_stat_arb_pairs_count",
        "Number of cointegrated pairs tracked by StatArbEngine",
        registry=registry,
    )
    metrics["stat_arb_active_trades"] = Gauge(
        "metadron_stat_arb_active_trades",
        "Number of active stat-arb trades",
        registry=registry,
    )
    metrics["stat_arb_portfolio_beta"] = Gauge(
        "metadron_stat_arb_portfolio_beta",
        "StatArbEngine portfolio beta",
        registry=registry,
    )
    metrics["stat_arb_mean_zscore"] = Gauge(
        "metadron_stat_arb_mean_zscore",
        "Mean z-score across all stat-arb pairs",
        registry=registry,
    )

    # MLVoteEnsemble metrics
    metrics["ml_ensemble_vote_bullish"] = Gauge(
        "metadron_ml_ensemble_vote_bullish",
        "MLVoteEnsemble bullish tier count",
        registry=registry,
    )
    metrics["ml_ensemble_vote_bearish"] = Gauge(
        "metadron_ml_ensemble_vote_bearish",
        "MLVoteEnsemble bearish tier count",
        registry=registry,
    )
    metrics["ml_ensemble_confidence"] = Gauge(
        "metadron_ml_ensemble_confidence",
        "MLVoteEnsemble aggregate confidence score",
        registry=registry,
    )

    # DecisionMatrix metrics
    metrics["decision_matrix_gates_passed"] = Gauge(
        "metadron_decision_matrix_gates_passed",
        "DecisionMatrix gates currently passing",
        registry=registry,
    )
    metrics["decision_matrix_gates_total"] = Gauge(
        "metadron_decision_matrix_gates_total",
        "DecisionMatrix total configured gates",
        registry=registry,
    )
    metrics["decision_matrix_approval_rate"] = Gauge(
        "metadron_decision_matrix_approval_rate",
        "DecisionMatrix trade approval rate (0-1)",
        registry=registry,
    )
    metrics["decision_matrix_evaluations_total"] = Counter(
        "metadron_decision_matrix_evaluations_total",
        "Total DecisionMatrix evaluations by result",
        ["result"],
        registry=registry,
    )

    # MetadronCube confidence & sleeve allocation
    metrics["cube_regime_confidence"] = Gauge(
        "metadron_cube_regime_confidence",
        "MetadronCube regime confidence score (0-1)",
        registry=registry,
    )
    metrics["cube_sleeve_weight"] = Gauge(
        "metadron_cube_sleeve_weight",
        "MetadronCube sleeve allocation weight",
        ["sleeve"],
        registry=registry,
    )

    # PatternRecognitionEngine metrics
    metrics["pattern_recognition_patterns_detected"] = Gauge(
        "metadron_pattern_recognition_patterns_detected",
        "Number of active patterns detected",
        registry=registry,
    )
    metrics["pattern_recognition_confidence"] = Gauge(
        "metadron_pattern_recognition_confidence",
        "Mean pattern recognition confidence",
        registry=registry,
    )

    # PM2 process metrics
    metrics["pm2_process_memory_bytes"] = Gauge(
        "metadron_pm2_process_memory_bytes",
        "PM2 process memory usage in bytes",
        ["process"],
        registry=registry,
    )
    metrics["pm2_process_restarts"] = Gauge(
        "metadron_pm2_process_restarts",
        "PM2 process restart count",
        ["process"],
        registry=registry,
    )

    # Futures engine metrics
    metrics["futures_positions_count"] = Gauge(
        "metadron_futures_positions_count",
        "Number of active futures positions",
        registry=registry,
    )
    metrics["futures_total_pnl"] = Gauge(
        "metadron_futures_total_pnl",
        "Total unrealized P&L on futures positions",
        registry=registry,
    )
    metrics["futures_margin_used"] = Gauge(
        "metadron_futures_margin_used",
        "Total margin used by futures positions",
        registry=registry,
    )
    metrics["futures_margin_utilization"] = Gauge(
        "metadron_futures_margin_utilization",
        "Futures margin utilization percentage (0-100)",
        registry=registry,
    )
    metrics["futures_notional_exposure"] = Gauge(
        "metadron_futures_notional_exposure",
        "Total notional exposure of futures positions",
        registry=registry,
    )
    metrics["futures_beta_current"] = Gauge(
        "metadron_futures_beta_current",
        "Current portfolio beta from BetaCorridor",
        registry=registry,
    )
    metrics["futures_beta_target"] = Gauge(
        "metadron_futures_beta_target",
        "Target portfolio beta from BetaCorridor",
        registry=registry,
    )
    metrics["futures_contracts_tracked"] = Gauge(
        "metadron_futures_contracts_tracked",
        "Number of futures contracts in universe",
        registry=registry,
    )

    # TCA engine metrics
    metrics["tca_avg_total_cost_bps"] = Gauge(
        "metadron_tca_avg_total_cost_bps",
        "TCA average total execution cost in basis points",
        registry=registry,
    )
    metrics["tca_avg_spread_bps"] = Gauge(
        "metadron_tca_avg_spread_bps",
        "TCA average spread cost in basis points",
        registry=registry,
    )
    metrics["tca_avg_impact_bps"] = Gauge(
        "metadron_tca_avg_impact_bps",
        "TCA average market impact in basis points",
        registry=registry,
    )
    metrics["tca_avg_timing_bps"] = Gauge(
        "metadron_tca_avg_timing_bps",
        "TCA average timing cost in basis points",
        registry=registry,
    )
    metrics["tca_total_is_usd"] = Gauge(
        "metadron_tca_total_is_usd",
        "TCA total implementation shortfall in USD",
        registry=registry,
    )
    metrics["tca_execution_quality"] = Gauge(
        "metadron_tca_execution_quality",
        "TCA execution quality score (0-100)",
        registry=registry,
    )
    metrics["tca_trades_analyzed"] = Gauge(
        "metadron_tca_trades_analyzed",
        "Number of trades analyzed by TCA engine",
        registry=registry,
    )
    metrics["tca_outliers_count"] = Gauge(
        "metadron_tca_outliers_count",
        "Number of execution outliers detected",
        registry=registry,
    )
    metrics["tca_total_volume_usd"] = Gauge(
        "metadron_tca_total_volume_usd",
        "Total notional volume analyzed by TCA",
        registry=registry,
    )
    metrics["tca_cost_trend"] = Gauge(
        "metadron_tca_cost_trend",
        "TCA cost trend direction (1=improving, 0=stable, -1=degrading)",
        registry=registry,
    )

    # TXLOG / Trade execution metrics
    metrics["txlog_orders_total"] = Gauge(
        "metadron_txlog_orders_total",
        "Total orders in current session",
        registry=registry,
    )
    metrics["txlog_fill_rate"] = Gauge(
        "metadron_txlog_fill_rate",
        "Order fill rate (0-1)",
        registry=registry,
    )
    metrics["txlog_reject_rate"] = Gauge(
        "metadron_txlog_reject_rate",
        "Order rejection rate (0-1)",
        registry=registry,
    )
    metrics["txlog_avg_latency_ms"] = Gauge(
        "metadron_txlog_avg_latency_ms",
        "Average order fill latency in milliseconds",
        registry=registry,
    )
    metrics["txlog_avg_slippage_bps"] = Gauge(
        "metadron_txlog_avg_slippage_bps",
        "Average order slippage in basis points",
        registry=registry,
    )
    metrics["txlog_notional_volume"] = Gauge(
        "metadron_txlog_notional_volume",
        "Total notional volume of executed orders",
        registry=registry,
    )
    metrics["txlog_orders_by_side"] = Gauge(
        "metadron_txlog_orders_by_side",
        "Order count by side (BUY/SELL/SHORT/COVER)",
        ["side"],
        registry=registry,
    )

    # ─── Agent System Metrics ─────────────────────────────────────
    metrics["agents_total"] = Gauge(
        "metadron_agents_total",
        "Total number of agents in the scorecard",
        registry=registry,
    )
    metrics["agents_active"] = Gauge(
        "metadron_agents_active",
        "Number of active agents",
        registry=registry,
    )
    metrics["agents_by_tier"] = Gauge(
        "metadron_agents_by_tier",
        "Agent count per tier",
        ["tier"],
        registry=registry,
    )
    metrics["agents_avg_accuracy"] = Gauge(
        "metadron_agents_avg_accuracy",
        "Average accuracy across all agents",
        registry=registry,
    )
    metrics["agents_avg_sharpe"] = Gauge(
        "metadron_agents_avg_sharpe",
        "Average Sharpe ratio across all agents",
        registry=registry,
    )
    metrics["agents_avg_composite"] = Gauge(
        "metadron_agents_avg_composite",
        "Average composite score across all agents",
        registry=registry,
    )
    metrics["agents_total_signals"] = Counter(
        "metadron_agents_total_signals",
        "Total signals generated across all agents",
        registry=registry,
    )
    metrics["agents_consensus_bull_pct"] = Gauge(
        "metadron_agents_consensus_bull_pct",
        "Bull consensus percentage across agent fleet",
        registry=registry,
    )
    metrics["agents_herding_risk"] = Gauge(
        "metadron_agents_herding_risk",
        "Collective herding risk from enforcement engine",
        registry=registry,
    )
    metrics["agents_concentration_risk"] = Gauge(
        "metadron_agents_concentration_risk",
        "Concentration risk from enforcement engine",
        registry=registry,
    )
    metrics["agents_gradient_alignment"] = Gauge(
        "metadron_agents_gradient_alignment",
        "Cross-engine gradient alignment score",
        registry=registry,
    )
    metrics["agents_enforcement_events"] = Counter(
        "metadron_agents_enforcement_events",
        "Total enforcement events by severity",
        ["severity"],
        registry=registry,
    )

    # ─── Reconciliation Metrics ───────────────────────────────────
    metrics["recon_positions_matched"] = Gauge(
        "metadron_recon_positions_matched",
        "Number of matched positions in broker reconciliation",
        registry=registry,
    )
    metrics["recon_positions_mismatched"] = Gauge(
        "metadron_recon_positions_mismatched",
        "Number of mismatched positions in broker reconciliation",
        registry=registry,
    )
    metrics["recon_nav_delta"] = Gauge(
        "metadron_recon_nav_delta",
        "NAV delta between Paper and IBKR brokers",
        registry=registry,
    )
    metrics["recon_paper_nav"] = Gauge(
        "metadron_recon_paper_nav",
        "trade log NAV",
        registry=registry,
    )
    metrics["recon_ibkr_nav"] = Gauge(
        "metadron_recon_ibkr_nav",
        "IBKR broker NAV",
        registry=registry,
    )
    metrics["recon_total_positions"] = Gauge(
        "metadron_recon_total_positions",
        "Total positions in reconciliation",
        registry=registry,
    )

    # ─── ETF Dashboard Metrics ─────────────────────────────────────
    metrics["etf_positions_count"] = Gauge(
        "metadron_etf_positions_count",
        "Number of ETF positions held",
        registry=registry,
    )
    metrics["etf_total_market_value"] = Gauge(
        "metadron_etf_total_market_value",
        "Total market value of ETF holdings",
        registry=registry,
    )
    metrics["etf_unrealized_pnl"] = Gauge(
        "metadron_etf_unrealized_pnl",
        "Total unrealized P&L on ETF positions",
        registry=registry,
    )
    metrics["etf_categories_active"] = Gauge(
        "metadron_etf_categories_active",
        "Number of ETF categories with positions",
        registry=registry,
    )
    metrics["etf_portfolio_weight"] = Gauge(
        "metadron_etf_portfolio_weight",
        "ETF allocation as percentage of total portfolio NAV",
        registry=registry,
    )
    metrics["etf_tracked_total"] = Gauge(
        "metadron_etf_tracked_total",
        "Total number of ETFs in tracking universe",
        registry=registry,
    )

    # ─── Fixed Income Metrics ─────────────────────────────────────
    metrics["fi_yield_2y"] = Gauge("metadron_fi_yield_2y", "US Treasury 2Y yield", registry=registry)
    metrics["fi_yield_10y"] = Gauge("metadron_fi_yield_10y", "US Treasury 10Y yield", registry=registry)
    metrics["fi_yield_30y"] = Gauge("metadron_fi_yield_30y", "US Treasury 30Y yield", registry=registry)
    metrics["fi_spread_2s10s"] = Gauge("metadron_fi_spread_2s10s", "2s10s yield spread", registry=registry)
    metrics["fi_ig_oas"] = Gauge("metadron_fi_ig_oas", "IG OAS credit spread (bps)", registry=registry)
    metrics["fi_hy_oas"] = Gauge("metadron_fi_hy_oas", "HY OAS credit spread (bps)", registry=registry)
    metrics["fi_positions_count"] = Gauge("metadron_fi_positions_count", "Number of FI positions", registry=registry)
    metrics["fi_total_exposure"] = Gauge("metadron_fi_total_exposure", "Total FI exposure USD", registry=registry)
    metrics["fi_avg_duration"] = Gauge("metadron_fi_avg_duration", "Average portfolio duration (years)", registry=registry)
    metrics["fi_dv01"] = Gauge("metadron_fi_dv01", "Portfolio DV01", registry=registry)

    # ─── Macro Historical Metrics ─────────────────────────────────
    metrics["macro_vix_current"] = Gauge("metadron_macro_vix_current", "Current VIX level from FRED", registry=registry)
    metrics["macro_dxy_current"] = Gauge("metadron_macro_dxy_current", "Current DXY level from FRED", registry=registry)
    metrics["macro_spread_2s10s_current"] = Gauge("metadron_macro_spread_2s10s_current", "Current 2s10s spread from FRED", registry=registry)

    # ─── Monte Carlo & Simulation Metrics ─────────────────────────
    metrics["mc_var95"] = Gauge("metadron_mc_var95", "Monte Carlo VaR 95%", registry=registry)
    metrics["mc_var99"] = Gauge("metadron_mc_var99", "Monte Carlo VaR 99%", registry=registry)
    metrics["mc_expected_return"] = Gauge("metadron_mc_expected_return", "MC expected portfolio return %", registry=registry)
    metrics["mc_prob_profit"] = Gauge("metadron_mc_prob_profit", "MC probability of profit %", registry=registry)
    metrics["mc_max_drawdown"] = Gauge("metadron_mc_max_drawdown", "MC average max drawdown %", registry=registry)
    metrics["sim_regime_bull_prob"] = Gauge("metadron_sim_regime_bull_prob", "Simulation bull regime probability", registry=registry)
    metrics["sim_regime_bear_prob"] = Gauge("metadron_sim_regime_bear_prob", "Simulation bear regime probability", registry=registry)

    # ─── ML Models Status Metrics ─────────────────────────────────
    metrics["ml_models_online"] = Gauge("metadron_ml_models_online", "Number of ML models online", registry=registry)
    metrics["ml_models_total"] = Gauge("metadron_ml_models_total", "Total ML models registered", registry=registry)
    metrics["ml_models_by_type"] = Gauge("metadron_ml_models_by_type", "Model count by type", ["model_type"], registry=registry)

    # ─── Quant Strategy Engine Metrics ─────────────────────────────
    metrics["quant_strategies_active"] = Gauge(
        "metadron_quant_strategies_active",
        "Number of active HFT strategies firing signals",
        registry=registry,
    )
    metrics["quant_consensus_signal"] = Gauge(
        "metadron_quant_consensus_signal",
        "Weighted consensus signal from all quant strategies (-1 to 1)",
        registry=registry,
    )
    metrics["quant_strategy_agreement"] = Gauge(
        "metadron_quant_strategy_agreement",
        "Agreement ratio across active quant strategies (0-1)",
        registry=registry,
    )
    metrics["quant_vix_regime"] = Gauge(
        "metadron_quant_vix_regime",
        "Current VIX level used for regime gating",
        registry=registry,
    )
    metrics["quant_kill_switch"] = Gauge(
        "metadron_quant_kill_switch",
        "Quant kill switch status (1=active, 0=normal)",
        registry=registry,
    )
    metrics["quant_size_multiplier"] = Gauge(
        "metadron_quant_size_multiplier",
        "Position size multiplier from quant consensus",
        registry=registry,
    )
    metrics["quant_executions_total"] = Gauge(
        "metadron_quant_executions_total",
        "Total quant strategy executions logged",
        registry=registry,
    )
    metrics["quant_patterns_detected"] = Gauge(
        "metadron_quant_patterns_detected",
        "High conviction pattern signals from PatternRecognitionEngine",
        registry=registry,
    )
    metrics["quant_factor_oos_sharpe"] = Gauge(
        "metadron_quant_factor_oos_sharpe",
        "AlphaOptimizer out-of-sample Sharpe ratio",
        registry=registry,
    )
    metrics["quant_learning_consistency"] = Gauge(
        "metadron_quant_learning_consistency",
        "Strategy consistency score from learning loop (0-1)",
        registry=registry,
    )

    # ─── Archive Metrics ─────────────────────────────────────
    metrics["archive_files_today"] = Gauge(
        "metadron_archive_files_today",
        "Files archived today",
        registry=registry,
    )
    metrics["archive_total_files"] = Gauge(
        "metadron_archive_total_files",
        "Total archived files",
        registry=registry,
    )

    # ─── Backtest Metrics ────────────────────────────────────
    metrics["backtest_opportunities"] = Gauge(
        "metadron_backtest_opportunities",
        "Backtest opportunities found",
        registry=registry,
    )
    metrics["backtest_high_conviction"] = Gauge(
        "metadron_backtest_high_conviction",
        "High conviction signals",
        registry=registry,
    )
    metrics["backtest_last_run"] = Gauge(
        "metadron_backtest_last_run",
        "Last backtest run timestamp",
        registry=registry,
    )

    # ─── Velocity Metrics ─────────────────────────────────────────
    metrics["order_flow_velocity"] = Gauge(
        "metadron_order_flow_velocity",
        "Order flow velocity — trades per second",
        registry=registry,
    )
    metrics["volume_per_second"] = Gauge(
        "metadron_volume_per_second",
        "Volume per second in USD",
        registry=registry,
    )
    metrics["signal_velocity"] = Gauge(
        "metadron_signal_velocity",
        "Signal velocity — signals per second across universe scan",
        registry=registry,
    )
    metrics["execution_latency_ms"] = Histogram(
        "metadron_execution_latency_ms",
        "Execution latency in milliseconds",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        registry=registry,
    )
    metrics["capital_deployment_rate"] = Gauge(
        "metadron_capital_deployment_rate",
        "Capital deployment rate in USD per minute",
        registry=registry,
    )
    metrics["momentum_score"] = Gauge(
        "metadron_momentum_score",
        "Rolling portfolio momentum score",
        registry=registry,
    )
    metrics["fill_rate_pct"] = Gauge(
        "metadron_fill_rate_pct",
        "Order fill rate percentage (0-100)",
        registry=registry,
    )
    metrics["momentum_breadth_pct"] = Gauge(
        "metadron_momentum_breadth_pct",
        "Momentum breadth — pct of positions with positive momentum",
        registry=registry,
    )

    # ─── Thinking / Universe Scan Metrics ─────────────────────────
    metrics["universe_scan_progress_pct"] = Gauge(
        "metadron_universe_scan_progress_pct",
        "Universe scan progress percentage (0-100)",
        registry=registry,
    )
    metrics["signals_generated_total"] = Counter(
        "metadron_signals_generated_total",
        "Total signals generated across all scan cycles",
        registry=registry,
    )
    metrics["scan_cycle_duration_seconds"] = Histogram(
        "metadron_scan_cycle_duration_seconds",
        "Scan cycle duration in seconds",
        buckets=(30, 60, 120, 180, 300, 600, 900, 1200),
        registry=registry,
    )
    metrics["active_universe_size"] = Gauge(
        "metadron_active_universe_size",
        "Number of securities in active scan universe",
        registry=registry,
    )
    metrics["scan_run_number"] = Gauge(
        "metadron_scan_run_number",
        "Current scan run number",
        registry=registry,
    )

    # ─── Margin / Collateral Metrics ──────────────────────────────
    metrics["margin_utilization_pct"] = Gauge(
        "metadron_margin_utilization_pct",
        "Margin utilization percentage (0-100)",
        registry=registry,
    )
    metrics["beta_corridor_value"] = Gauge(
        "metadron_beta_corridor_value",
        "Current beta corridor value",
        registry=registry,
    )
    metrics["leverage_multiplier"] = Gauge(
        "metadron_leverage_multiplier",
        "Current leverage multiplier",
        registry=registry,
    )
    metrics["kill_switch_triggered_total"] = Counter(
        "metadron_kill_switch_triggered_total",
        "Total kill switch trigger events",
        registry=registry,
    )
    metrics["kill_switch_pending_reset"] = Gauge(
        "metadron_kill_switch_pending_reset",
        "Kill switch awaiting manual operator reset (1=pending, 0=clear)",
        registry=registry,
    )
    metrics["kill_switch_reset_total"] = Counter(
        "metadron_kill_switch_reset_total",
        "Total operator-confirmed kill switch reset events",
        registry=registry,
    )
    metrics["allocation_cycle_reset_total"] = Counter(
        "metadron_allocation_cycle_reset_total",
        "Total allocation cycle utilization resets",
        registry=registry,
    )
    metrics["drawdown_pct"] = Gauge(
        "metadron_drawdown_pct",
        "Current drawdown from high water mark percentage",
        registry=registry,
    )
    metrics["collateral_available_usd"] = Gauge(
        "metadron_collateral_available_usd",
        "Available collateral in USD",
        registry=registry,
    )

    # ─── Chat / NanoClaw Metrics ──────────────────────────────────
    metrics["nanoclaw_messages_total"] = Counter(
        "metadron_nanoclaw_messages_total",
        "Total NanoClaw chat messages processed",
        registry=registry,
    )
    metrics["openclaw_recommendations_total"] = Counter(
        "metadron_openclaw_recommendations_total",
        "Total OpenClaw recommendation cards generated",
        registry=registry,
    )
    metrics["permission_guard_blocks_total"] = Counter(
        "metadron_permission_guard_blocks_total",
        "Total permission guard block events",
        registry=registry,
    )
    metrics["chat_response_latency_ms"] = Histogram(
        "metadron_chat_response_latency_ms",
        "Chat response latency in milliseconds",
        buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
        registry=registry,
    )
    metrics["active_chat_sessions"] = Gauge(
        "metadron_active_chat_sessions",
        "Number of active chat sessions",
        registry=registry,
    )


    # ─── Model Ensemble Metrics ──────────────────────────────────────

    # Per-model online status (1=online, 0=offline)
    metrics["model_online"] = Gauge(
        "metadron_model_online",
        "Model online status",
        ["model_name", "port"],
        registry=registry,
    )

    # Per-model inference latency
    metrics["model_inference_latency"] = Histogram(
        "metadron_model_inference_latency_seconds",
        "Inference latency per model",
        ["model_name"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        registry=registry,
    )

    # Per-model inference counter
    metrics["model_inference_counter"] = Counter(
        "metadron_model_inference_total",
        "Total inference requests per model",
        ["model_name", "status"],
        registry=registry,
    )

    # Ensemble synthesis latency (end-to-end)
    metrics["ensemble_synthesis_latency"] = Histogram(
        "metadron_ensemble_synthesis_latency_seconds",
        "End-to-end ensemble synthesis latency",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        registry=registry,
    )

    # Brain Power orchestration active
    metrics["brain_power_orchestrating"] = Gauge(
        "metadron_brain_power_orchestrating",
        "Brain Power currently orchestrating (1=active, 0=stub/inactive)",
        registry=registry,
    )

    # Pre-initialize model online gauges
    for model_name, port in [
        ("brain_power", "api"),
        ("air_llm", "8002"),
        ("qwen_2_5_7b", "7860"),
        ("ai_newton", "in-process"),
        ("alpha_optimizer", "in-process"),
        ("deep_learning_engine", "in-process"),
        ("metadron_cube", "in-process"),
    ]:
        metrics["model_online"].labels(model_name=model_name, port=port).set(0)

    return metrics


# ─── Collector: pull live data into gauges ─────────────────────────

def _collect_live_metrics(metrics: dict):
    """Populate gauge metrics from live engine state.

    Called on each /metrics scrape to ensure values are current.
    """
    # Engine up
    metrics["engine_up"].set(1)

    # Portfolio — try to read from engine singletons
    try:
        from engine.execution.execution_engine import ExecutionEngine
        engine = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if engine and hasattr(engine, "broker"):
            broker = engine.broker
            summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
            metrics["portfolio_nav"].set(summary.get("nav", 0))
            metrics["portfolio_pnl_daily"].set(summary.get("total_pnl", 0))
            metrics["positions_count"].set(summary.get("positions_count", 0))
    except Exception:
        pass

    # Cube regime
    try:
        from engine.signals.metadron_cube import MetadronCube
        # Look for a cached cube state file
        import json
        from pathlib import Path
        cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "cube_state_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                state = json.load(f)
            regime = state.get("regime", "RANGE")
            for r in ["TRENDING", "RANGE", "STRESS", "CRASH"]:
                metrics["cube_regime"].labels(regime_name=r).set(1 if r == regime else 0)
            # Composite score from liquidity
            metrics["cube_signal_score"].set(state.get("liquidity", 0))
    except Exception:
        pass

    # ── STRAT Engine Health Collectors ─────────────────────────────

    # VolatilitySurface
    try:
        from engine.execution.options_engine import VolatilitySurface
        vs = VolatilitySurface()
        metrics["strat_engine_health"].labels(engine="VolatilitySurface").set(1)
        # Pull live surface data if available
        if hasattr(vs, "current_iv"):
            metrics["vol_surface_iv"].set(vs.current_iv or 0)
        if hasattr(vs, "skew"):
            metrics["vol_surface_skew"].set(vs.skew or 0)
        if hasattr(vs, "term_structure_slope"):
            metrics["vol_surface_term_structure"].set(vs.term_structure_slope or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="VolatilitySurface").set(0)

    # StatArbEngine
    try:
        from engine.signals.stat_arb_engine import StatArbEngine
        sa = StatArbEngine()
        metrics["strat_engine_health"].labels(engine="StatArbEngine").set(1)
        if hasattr(sa, "pairs") and sa.pairs:
            metrics["stat_arb_pairs_count"].set(len(sa.pairs))
            zscores = [p.z_score for p in sa.pairs if hasattr(p, "z_score") and p.z_score is not None]
            if zscores:
                metrics["stat_arb_mean_zscore"].set(sum(zscores) / len(zscores))
        if hasattr(sa, "active_trades"):
            metrics["stat_arb_active_trades"].set(len(sa.active_trades) if sa.active_trades else 0)
        if hasattr(sa, "portfolio_beta"):
            metrics["stat_arb_portfolio_beta"].set(sa.portfolio_beta or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="StatArbEngine").set(0)

    # MLVoteEnsemble
    try:
        from engine.execution.execution_engine import MLVoteEnsemble
        ens = MLVoteEnsemble()
        metrics["strat_engine_health"].labels(engine="MLVoteEnsemble").set(1)
        if hasattr(ens, "tiers") and ens.tiers:
            bullish = sum(1 for t in ens.tiers if getattr(t, "vote", None) == "BUY")
            bearish = sum(1 for t in ens.tiers if getattr(t, "vote", None) == "SELL")
            metrics["ml_ensemble_vote_bullish"].set(bullish)
            metrics["ml_ensemble_vote_bearish"].set(bearish)
        if hasattr(ens, "confidence"):
            metrics["ml_ensemble_confidence"].set(ens.confidence or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MLVoteEnsemble").set(0)

    # DecisionMatrix
    try:
        from engine.execution.decision_matrix import DecisionMatrix, GATE_CONFIGS
        dm = DecisionMatrix()
        metrics["strat_engine_health"].labels(engine="DecisionMatrix").set(1)
        total_gates = len(GATE_CONFIGS) if GATE_CONFIGS else 6
        metrics["decision_matrix_gates_total"].set(total_gates)
        if hasattr(dm, "gates") and dm.gates:
            passing = sum(1 for g in dm.gates if getattr(g, "passing", False))
            metrics["decision_matrix_gates_passed"].set(passing)
            metrics["decision_matrix_approval_rate"].set(passing / total_gates if total_gates else 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="DecisionMatrix").set(0)

    # MetadronCube extended — confidence + sleeve weights
    try:
        cache_path2 = Path(__file__).resolve().parent.parent.parent / "data" / "cube_state_cache.json"
        if cache_path2.exists():
            with open(cache_path2) as f2:
                cs = json.load(f2)
            metrics["cube_regime_confidence"].set(cs.get("confidence", cs.get("liquidity", 0)))
            sleeves = cs.get("sleeves", cs.get("sleeve_allocation", {}))
            if isinstance(sleeves, dict):
                for name, weight in sleeves.items():
                    metrics["cube_sleeve_weight"].labels(sleeve=name).set(weight or 0)
        metrics["strat_engine_health"].labels(engine="MetadronCube").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MetadronCube").set(0)

    # PatternRecognitionEngine
    try:
        from engine.signals.pattern_recognition import PatternRecognitionEngine
        pre = PatternRecognitionEngine()
        metrics["strat_engine_health"].labels(engine="PatternRecognition").set(1)
        if hasattr(pre, "detected_patterns"):
            patterns = pre.detected_patterns or []
            metrics["pattern_recognition_patterns_detected"].set(len(patterns))
            if patterns:
                confs = [p.get("confidence", 0) for p in patterns if isinstance(p, dict)]
                metrics["pattern_recognition_confidence"].set(sum(confs) / len(confs) if confs else 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="PatternRecognition").set(0)

    # PM2 process metrics via /proc or psutil
    try:
        import subprocess
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            processes = json.loads(result.stdout)
            for proc in processes:
                name = proc.get("name", "unknown")
                monit = proc.get("monit", {})
                metrics["pm2_process_memory_bytes"].labels(process=name).set(
                    monit.get("memory", 0)
                )
                pm2_env = proc.get("pm2_env", {})
                metrics["pm2_process_restarts"].labels(process=name).set(
                    pm2_env.get("restart_time", 0)
                )
    except Exception:
        pass

    # ── Futures Engine Metrics ────────────────────────────────
    try:
        from engine.api.routers.futures import FUTURES_UNIVERSE, FUTURES_PREFIXES
        metrics["futures_contracts_tracked"].set(len(FUTURES_UNIVERSE))

        from engine.execution.execution_engine import ExecutionEngine
        eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if eng is None:
            from engine.api.routers.execution import _get_exec
            eng = _get_exec()
        if eng is not None:
            broker = eng.broker
            positions = broker.get_all_positions()
            proxy_map = {s["proxy"]: s for s in FUTURES_UNIVERSE.values()}
            f_count = 0
            f_pnl = 0
            f_margin = 0
            f_notional = 0
            for ticker, pos in positions.items():
                spec = None
                if any(ticker.startswith(p) for p in FUTURES_PREFIXES) or ticker.endswith("=F"):
                    for ft, s in FUTURES_UNIVERSE.items():
                        if ft.startswith(ticker[:2]):
                            spec = s
                            break
                elif ticker in proxy_map:
                    spec = proxy_map[ticker]
                if spec:
                    qty = abs(getattr(pos, "quantity", 0))
                    f_count += 1
                    f_pnl += getattr(pos, "unrealized_pnl", 0)
                    f_margin += qty * spec["margin_init"]
                    f_notional += qty * getattr(pos, "current_price", 0) * spec["multiplier"]
            metrics["futures_positions_count"].set(f_count)
            metrics["futures_total_pnl"].set(f_pnl)
            metrics["futures_margin_used"].set(f_margin)
            metrics["futures_notional_exposure"].set(f_notional)
            nav = broker.get_portfolio_summary().get("nav", 0) if hasattr(broker, "get_portfolio_summary") else 0
            metrics["futures_margin_utilization"].set((f_margin / nav * 100) if nav > 0 else 0)

        # BetaCorridor
        try:
            from engine.portfolio.beta_corridor import BetaCorridor
            beta = eng.beta if eng and hasattr(eng, "beta") else None
            if beta:
                analytics = beta.get_corridor_analytics()
                metrics["futures_beta_current"].set(analytics.get("current_beta", 0))
                metrics["futures_beta_target"].set(analytics.get("target_beta", 0))
        except Exception:
            pass
    except Exception:
        pass

    # ── TCA Engine Metrics ─────────────────────────────────────────
    try:
        from engine.execution.tca_engine import TCAEngine
        from engine.execution.execution_engine import ExecutionEngine as EE_TCA
        eng_tca = EE_TCA._instance if hasattr(EE_TCA, "_instance") else None
        if eng_tca is None:
            from engine.api.routers.execution import _get_exec as _ge_tca
            eng_tca = _ge_tca()
        if eng_tca is not None:
            tca = TCAEngine()
            broker_tca = eng_tca.broker
            raw_trades = broker_tca.get_trade_history()[-500:]
            td = []
            for t in raw_trades:
                if isinstance(t, dict):
                    td.append(t)
                else:
                    d = {}
                    for attr in ("ticker", "side", "quantity", "fill_price", "arrival_price",
                                 "slippage", "signal_type", "venue", "latency_ms", "product_type",
                                 "fill_timestamp", "status", "vwap_price", "order_id"):
                        val = getattr(t, attr, None)
                        if val is not None:
                            d[attr] = val
                    td.append(d)
            tca.rebuild(td)
            tca_summary = tca.get_summary()
            metrics["tca_avg_total_cost_bps"].set(tca_summary.get("avg_total_cost_bps", 0))
            metrics["tca_avg_spread_bps"].set(tca_summary.get("avg_spread_bps", 0))
            metrics["tca_avg_impact_bps"].set(tca_summary.get("avg_impact_bps", 0))
            metrics["tca_avg_timing_bps"].set(tca_summary.get("avg_timing_bps", 0))
            metrics["tca_total_is_usd"].set(tca_summary.get("total_is_usd", 0))
            metrics["tca_execution_quality"].set(tca_summary.get("execution_quality_score", 0))
            metrics["tca_trades_analyzed"].set(tca_summary.get("total_trades", 0))
            metrics["tca_total_volume_usd"].set(tca_summary.get("total_volume_usd", 0))
            tca_outliers = tca.get_outliers()
            metrics["tca_outliers_count"].set(len(tca_outliers))
            trend_v = {"IMPROVING": 1, "STABLE": 0, "DEGRADING": -1}.get(
                tca_summary.get("cost_trend", "STABLE"), 0
            )
            metrics["tca_cost_trend"].set(trend_v)
            metrics["strat_engine_health"].labels(engine="TCAEngine").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="TCAEngine").set(0)

    # ── ETF Dashboard Metrics ──────────────────────────────────
    try:
        from engine.data.universe_engine import ALL_ETFS
        metrics["etf_tracked_total"].set(len(ALL_ETFS))

        from engine.execution.execution_engine import ExecutionEngine
        eng_etf = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if eng_etf is None:
            from engine.api.routers.execution import _get_exec
            eng_etf = _get_exec()
        if eng_etf is not None:
            broker_etf = eng_etf.broker
            positions_etf = broker_etf.get_all_positions()
            etf_set = set(ALL_ETFS)
            etf_count = 0
            etf_mv = 0
            etf_pnl = 0
            cats = set()
            from engine.data.universe_engine import (
                SECTOR_ETFS, FACTOR_ETFS, COMMODITY_ETFS, FIXED_INCOME_ETFS,
                VOLATILITY_ETFS, INTERNATIONAL_ETFS, INDEX_ETFS, THEMATIC_ETFS,
            )
            cat_map = {}
            for n, t in SECTOR_ETFS.items(): cat_map[t] = "Sector"
            for n, t in FACTOR_ETFS.items(): cat_map[t] = "Factor"
            for n, t in COMMODITY_ETFS.items(): cat_map[t] = "Commodity"
            for n, t in FIXED_INCOME_ETFS.items(): cat_map[t] = "Bond"
            for n, t in VOLATILITY_ETFS.items(): cat_map[t] = "Volatility"
            for n, t in INTERNATIONAL_ETFS.items(): cat_map[t] = "International"
            for n, t in INDEX_ETFS.items(): cat_map[t] = "Equity"
            for n, t in THEMATIC_ETFS.items(): cat_map[t] = "Thematic"

            for ticker, pos in (positions_etf.items() if isinstance(positions_etf, dict) else []):
                if ticker not in etf_set:
                    continue
                qty = getattr(pos, "quantity", 0)
                price = getattr(pos, "current_price", 0)
                etf_count += 1
                etf_mv += qty * price
                etf_pnl += getattr(pos, "unrealized_pnl", 0)
                cats.add(cat_map.get(ticker, "Other"))

            metrics["etf_positions_count"].set(etf_count)
            metrics["etf_total_market_value"].set(etf_mv)
            metrics["etf_unrealized_pnl"].set(etf_pnl)
            metrics["etf_categories_active"].set(len(cats))
            nav_etf = broker_etf.get_portfolio_summary().get("nav", 0) if hasattr(broker_etf, "get_portfolio_summary") else 0
            metrics["etf_portfolio_weight"].set((etf_mv / nav_etf * 100) if nav_etf > 0 else 0)
        metrics["strat_engine_health"].labels(engine="ETFDashboard").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="ETFDashboard").set(0)

    # ── Reconciliation Metrics ─────────────────────────────────
    try:
        from engine.execution.paper_broker import PaperBroker
        pb = PaperBroker()
        paper_pos = pb.get_all_positions()
        paper_nav = pb.compute_nav()
        metrics["recon_paper_nav"].set(paper_nav)
        metrics["recon_total_positions"].set(len(paper_pos))

        ibkr_nav = 0
        broker_pos = {}
        try:
            from engine.execution.alpaca_broker import AlpacaBroker
            ab = AlpacaBroker(initial_cash=0, paper=True)
            broker_pos = ab.get_positions()
            ibkr_nav = ab.compute_nav()
        except Exception:
            pass
        metrics["recon_ibkr_nav"].set(ibkr_nav)
        metrics["recon_nav_delta"].set(paper_nav - ibkr_nav)

        all_t = set(list(paper_pos.keys()) + list(broker_pos.keys()))
        m_count = 0
        mm_count = 0
        for t in all_t:
            in_p = t in paper_pos
            in_a = t in broker_pos
            if in_p and in_a:
                pq = getattr(paper_pos[t], "quantity", 0) if hasattr(paper_pos[t], "quantity") else 0
                aq = broker_pos[t].get("quantity", 0) if isinstance(broker_pos[t], dict) else getattr(broker_pos[t], "quantity", 0)
                if pq == aq:
                    m_count += 1
                else:
                    mm_count += 1
            else:
                mm_count += 1
        metrics["recon_positions_matched"].set(m_count)
        metrics["recon_positions_mismatched"].set(mm_count)
    except Exception:
        pass

    # ── Quant Strategy Engine Metrics ──────────────────────────────
    try:
        from engine.execution.quant_strategy_executor import QuantStrategyExecutor
        qse = QuantStrategyExecutor()
        metrics["strat_engine_health"].labels(engine="QuantStrategyExecutor").set(1)

        log = qse.get_execution_log() if hasattr(qse, "get_execution_log") else []
        metrics["quant_executions_total"].set(len(log))
        if log:
            latest = log[-1] if log else {}
            metrics["quant_strategies_active"].set(latest.get("active_count", 0))
            metrics["quant_consensus_signal"].set(latest.get("consensus_signal", 0))
            metrics["quant_strategy_agreement"].set(latest.get("agreement", 0))
            metrics["quant_kill_switch"].set(1 if latest.get("kill_switch", False) else 0)
            metrics["quant_size_multiplier"].set(latest.get("size_multiplier", 0))
            # Learning consistency
            kill_count = sum(1 for e in log if e.get("kill_switch", False))
            metrics["quant_learning_consistency"].set(
                round(1.0 - (kill_count / len(log)), 3) if len(log) > 0 else 1.0
            )
    except Exception:
        metrics["strat_engine_health"].labels(engine="QuantStrategyExecutor").set(0)

    # PatternRecognitionEngine (quant-specific deep scan)
    try:
        from engine.ml.pattern_recognition import PatternRecognitionEngine
        pre_q = PatternRecognitionEngine()
        if hasattr(pre_q, "get_high_conviction_signals"):
            hc = pre_q.get_high_conviction_signals()
            metrics["quant_patterns_detected"].set(len(hc) if hc else 0)
    except Exception:
        pass

    # AlphaOptimizer OOS Sharpe
    try:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        ao = AlphaOptimizer()
        if hasattr(ao, "get_oos_sharpe"):
            metrics["quant_factor_oos_sharpe"].set(ao.get_oos_sharpe())
    except Exception:
        pass

    # VIX regime
    try:
        from engine.data.openbb_data import get_prices
        from datetime import datetime as dt, timedelta as td
        vix_df = get_prices("^VIX", start=(dt.utcnow() - td(days=5)).strftime("%Y-%m-%d"),
                            end=dt.utcnow().strftime("%Y-%m-%d"))
        if vix_df is not None and not vix_df.empty:
            close_col = vix_df["Close"].iloc[:, 0] if hasattr(vix_df.columns, "levels") and "Close" in vix_df.columns.get_level_values(0) else vix_df.get("Close", vix_df.get("close", vix_df.iloc[:, 0]))
            metrics["quant_vix_regime"].set(float(close_col.iloc[-1]))
    except Exception:
        pass

    # ── Fixed Income Metrics ─────────────────────────────────────────
    try:
        from engine.signals.fixed_income_engine import FixedIncomeEngine
        fi = FixedIncomeEngine()
        fi_summary = fi.get_summary() if hasattr(fi, "get_summary") else {}
        metrics["fi_positions_count"].set(fi_summary.get("positions_count", 0))
        metrics["fi_total_exposure"].set(fi_summary.get("total_exposure", 0))
        metrics["fi_avg_duration"].set(fi_summary.get("avg_duration", 0))
        metrics["fi_dv01"].set(fi_summary.get("dv01", 0))
        metrics["strat_engine_health"].labels(engine="FixedIncomeEngine").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="FixedIncomeEngine").set(0)

    # FI yields and spreads from FRED
    try:
        from engine.data.openbb_data import get_fred_series
        from datetime import datetime as dt_fi, timedelta as td_fi
        end_fi = dt_fi.utcnow().strftime("%Y-%m-%d")
        start_fi = (dt_fi.utcnow() - td_fi(days=10)).strftime("%Y-%m-%d")
        fi_yields = get_fred_series(["DGS2", "DGS10", "DGS30", "BAMLH0A0HYM2", "BAMLC0A4CBBB"], start=start_fi, end=end_fi)
        if fi_yields is not None and not fi_yields.empty:
            if "DGS2" in fi_yields.columns:
                val = fi_yields["DGS2"].dropna().iloc[-1]
                metrics["fi_yield_2y"].set(float(val))
            if "DGS10" in fi_yields.columns:
                val = fi_yields["DGS10"].dropna().iloc[-1]
                metrics["fi_yield_10y"].set(float(val))
            if "DGS30" in fi_yields.columns:
                val = fi_yields["DGS30"].dropna().iloc[-1]
                metrics["fi_yield_30y"].set(float(val))
            if "DGS2" in fi_yields.columns and "DGS10" in fi_yields.columns:
                y2 = fi_yields["DGS2"].dropna().iloc[-1]
                y10 = fi_yields["DGS10"].dropna().iloc[-1]
                metrics["fi_spread_2s10s"].set(float(y10 - y2))
            if "BAMLC0A4CBBB" in fi_yields.columns:
                metrics["fi_ig_oas"].set(float(fi_yields["BAMLC0A4CBBB"].dropna().iloc[-1]))
            if "BAMLH0A0HYM2" in fi_yields.columns:
                metrics["fi_hy_oas"].set(float(fi_yields["BAMLH0A0HYM2"].dropna().iloc[-1]))
    except Exception:
        pass

    # ── Macro Historical Metrics (VIX, DXY, 2s10s) ───────────────────
    try:
        from engine.data.openbb_data import get_fred_series as gfs_macro
        from datetime import datetime as dt_m, timedelta as td_m
        end_m = dt_m.utcnow().strftime("%Y-%m-%d")
        start_m = (dt_m.utcnow() - td_m(days=10)).strftime("%Y-%m-%d")
        macro_series = gfs_macro(["VIXCLS", "DTWEXBGS", "T10Y2Y"], start=start_m, end=end_m)
        if macro_series is not None and not macro_series.empty:
            if "VIXCLS" in macro_series.columns:
                metrics["macro_vix_current"].set(float(macro_series["VIXCLS"].dropna().iloc[-1]))
            if "DTWEXBGS" in macro_series.columns:
                metrics["macro_dxy_current"].set(float(macro_series["DTWEXBGS"].dropna().iloc[-1]))
            if "T10Y2Y" in macro_series.columns:
                metrics["macro_spread_2s10s_current"].set(float(macro_series["T10Y2Y"].dropna().iloc[-1]))
    except Exception:
        pass

    # ── Monte Carlo & Simulation Metrics ──────────────────────────────
    try:
        from engine.ml.bridges.monte_carlo_bridge import MonteCarloBridge
        mc = MonteCarloBridge()
        mc_result = mc.compute_portfolio_risk() if hasattr(mc, "compute_portfolio_risk") else {}
        metrics["mc_var95"].set(mc_result.get("var_95", 0))
        metrics["mc_var99"].set(mc_result.get("var_99", 0))
        metrics["mc_expected_return"].set(mc_result.get("expected_return", 0))
        metrics["mc_prob_profit"].set(mc_result.get("prob_profit", 0))
        metrics["mc_max_drawdown"].set(mc_result.get("max_drawdown", 0))
        metrics["strat_engine_health"].labels(engine="MonteCarloRiskEngine").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MonteCarloRiskEngine").set(0)

    # Regime simulation probabilities
    try:
        from engine.ml.bridges.markov_regime_bridge import MarkovRegimeBridge
        mrb = MarkovRegimeBridge()
        regime_probs = mrb.get_regime_probabilities() if hasattr(mrb, "get_regime_probabilities") else {}
        metrics["sim_regime_bull_prob"].set(regime_probs.get("bull", regime_probs.get("BULL", 0)))
        metrics["sim_regime_bear_prob"].set(regime_probs.get("bear", regime_probs.get("BEAR", 0)))
        metrics["strat_engine_health"].labels(engine="MarkovRegimeBridge").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MarkovRegimeBridge").set(0)

    # ── ML Models Status Metrics ──────────────────────────────────────
    try:
        import importlib
        engine_modules = [
            "engine.data.universe_engine", "engine.data.openbb_data",
            "engine.signals.macro_engine", "engine.signals.metadron_cube",
            "engine.signals.stat_arb_engine", "engine.signals.contagion_engine",
            "engine.signals.fixed_income_engine", "engine.ml.alpha_optimizer",
            "engine.ml.backtester", "engine.ml.pattern_recognition",
            "engine.ml.universe_classifier", "engine.ml.deep_learning_engine",
            "engine.execution.execution_engine", "engine.execution.decision_matrix",
            "engine.execution.options_engine", "engine.agents.investor_personas",
        ]
        online_count = 0
        total_count = len(engine_modules)
        type_counts = {"ML": 0, "Statistical": 0, "Neural Net": 0, "LLM": 0,
                       "Rule-Based": 0, "Ensemble": 0, "Framework": 0}
        type_map = {
            "alpha_optimizer": "ML", "universe_classifier": "Ensemble",
            "deep_learning_engine": "Neural Net", "backtester": "Statistical",
            "pattern_recognition": "Rule-Based", "investor_personas": "LLM",
            "options_engine": "Statistical", "decision_matrix": "Rule-Based",
            "execution_engine": "Neural Net", "stat_arb_engine": "Statistical",
            "contagion_engine": "Statistical", "macro_engine": "Rule-Based",
            "metadron_cube": "Ensemble", "fixed_income_engine": "Rule-Based",
            "universe_engine": "Rule-Based", "openbb_data": "Framework",
        }
        for mod_path in engine_modules:
            try:
                importlib.import_module(mod_path)
                online_count += 1
                mod_name = mod_path.split(".")[-1]
                mtype = type_map.get(mod_name, "Rule-Based")
                type_counts[mtype] = type_counts.get(mtype, 0) + 1
            except Exception:
                pass
        metrics["ml_models_online"].set(online_count)
        metrics["ml_models_total"].set(total_count)
        for mtype, count in type_counts.items():
            metrics["ml_models_by_type"].labels(model_type=mtype).set(count)
    except Exception:
        pass

    # ── TXLOG: Trade Execution Metrics ─────────────────────────────
    try:
        from engine.execution.execution_engine import ExecutionEngine
        eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if eng is None:
            from engine.api.routers.execution import _get_exec
            eng = _get_exec()
        if eng is not None:
            broker = eng.broker
            trades = broker.get_trade_history()
            total = len(trades)
            metrics["txlog_orders_total"].set(total)
            if total > 0:
                filled = [t for t in trades if t.get("fill_price", 0) > 0]
                rejected = total - len(filled)
                metrics["txlog_fill_rate"].set(len(filled) / total if total else 0)
                metrics["txlog_reject_rate"].set(rejected / total if total else 0)
                # Notional volume
                notional = sum(t.get("fill_price", 0) * t.get("quantity", 0) for t in filled)
                metrics["txlog_notional_volume"].set(notional)
                # Slippage (if tracked)
                slippages = [t.get("slippage", t.get("slippage_bps", 0)) or 0 for t in filled]
                if slippages:
                    metrics["txlog_avg_slippage_bps"].set(sum(slippages) / len(slippages))
                # Orders by side
                from collections import Counter
                side_counts = Counter(str(t.get("side", "UNKNOWN")).upper() for t in trades)
                for side_name in ["BUY", "SELL", "SHORT", "COVER"]:
                    metrics["txlog_orders_by_side"].labels(side=side_name).set(side_counts.get(side_name, 0))
    except Exception:
        pass

    # ── Archive Metrics ───────────────────────────────────────
    try:
        from engine.ops.archive_engine import ArchiveEngine, ARCHIVE_DIR
        from datetime import date as _date
        today = _date.today()
        today_dir = ARCHIVE_DIR / str(today.year) / f"{today.month:02d}" / f"{today.day:02d}"
        if today_dir.exists():
            metrics["archive_files_today"].set(len(list(today_dir.iterdir())))
        total = sum(1 for _ in ARCHIVE_DIR.rglob("*.json")) if ARCHIVE_DIR.exists() else 0
        metrics["archive_total_files"].set(total)
    except Exception:
        pass

    # ── Backtest Metrics ──────────────────────────────────────
    try:
        from engine.ml.evening_backtester import EveningBacktester, BACKTEST_DIR
        import json as _json
        files = sorted(BACKTEST_DIR.glob("*_evening.json"), reverse=True) if BACKTEST_DIR.exists() else []
        if files:
            data = _json.loads(files[0].read_text())
            summary = data.get("summary", {})
            metrics["backtest_opportunities"].set(summary.get("total_opportunities", 0))
            metrics["backtest_high_conviction"].set(summary.get("high_conviction", 0))
            gen_at = data.get("generated_at", "")
            if gen_at:
                try:
                    from datetime import datetime as _dt
                    ts = _dt.fromisoformat(gen_at).timestamp()
                    metrics["backtest_last_run"].set(ts)
                except Exception:
                    pass
    except Exception:
        pass

    # ── Velocity Metrics ──────────────────────────────────────────
    try:
        from engine.velocity.velocity_engine import VelocityEngine
        _vel = VelocityEngine()
        snap = _vel.get_snapshot()
        metrics["order_flow_velocity"].set(snap.get("trades_per_second", 0))
        metrics["volume_per_second"].set(snap.get("volume_per_second", 0))
        metrics["signal_velocity"].set(snap.get("signals_per_second", 0))
        metrics["capital_deployment_rate"].set(snap.get("capital_deployed_per_min", 0))
        metrics["momentum_score"].set(snap.get("momentum_score", 0))
        metrics["fill_rate_pct"].set(snap.get("fill_rate_pct", 0))
        metrics["momentum_breadth_pct"].set(snap.get("momentum_breadth", 0))
        avg_lat = snap.get("avg_latency_ms", 0)
        if avg_lat > 0:
            metrics["execution_latency_ms"].observe(avg_lat)
        metrics["strat_engine_health"].labels(engine="VelocityEngine").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="VelocityEngine").set(0)

    # ── Thinking / Universe Scan Metrics ──────────────────────────
    try:
        from engine.allocation.universe_scan import UniverseScanOrchestrator
        orch = UniverseScanOrchestrator._instance if hasattr(UniverseScanOrchestrator, "_instance") else None
        if orch is not None and hasattr(orch, "status"):
            status = orch.status
            # Progress: compute from current run vs total runs
            total_runs = len(getattr(status, "runs", [])) or 4
            completed_runs = sum(1 for r in getattr(status, "runs", []) if getattr(r, "completed", False))
            progress = (completed_runs / total_runs) * 100 if total_runs > 0 else 0
            metrics["universe_scan_progress_pct"].set(round(progress, 1))
            metrics["scan_run_number"].set(getattr(status, "current_run", 0))
            metrics["active_universe_size"].set(1044)  # SP1500 universe
            elapsed = getattr(status, "elapsed_seconds", 0)
            if elapsed > 0 and getattr(status, "completed", False):
                metrics["scan_cycle_duration_seconds"].observe(elapsed)
        else:
            # Try scan status cache
            from pathlib import Path as _ScanPath
            scan_cache = _ScanPath(__file__).resolve().parent.parent.parent / "data" / "scan_status_cache.json"
            if scan_cache.exists():
                scan_data = json.load(open(scan_cache))
                total_sigs = scan_data.get("total_signals", 0)
                cycle = scan_data.get("cycle_number", 0)
                metrics["scan_run_number"].set(scan_data.get("current_run", 0))
                metrics["active_universe_size"].set(1044)
                elapsed = scan_data.get("elapsed_seconds", 0)
                if elapsed > 0:
                    runs_list = scan_data.get("runs", [])
                    completed = sum(1 for r in runs_list if r.get("completed", False))
                    total_r = len(runs_list) or 4
                    metrics["universe_scan_progress_pct"].set(round((completed / total_r) * 100, 1))
    except Exception:
        pass

    # ── Margin / Collateral Metrics ───────────────────────────────
    try:
        from engine.allocation.allocation_engine import AllocationEngine
        alloc = AllocationEngine._instance if hasattr(AllocationEngine, "_instance") else None
        if alloc is not None:
            status = alloc.get_collateral_status() if hasattr(alloc, "get_collateral_status") else {}
            if isinstance(status, dict):
                bc = status.get("beta_corridor", {})
                metrics["beta_corridor_value"].set(bc.get("beta", 0))
                metrics["leverage_multiplier"].set(bc.get("leverage_multiplier", 0))

                mb = status.get("margin_bucket", {})
                metrics["margin_utilization_pct"].set(mb.get("real_capital_deployed_pct", 0) * 100)

                ks = status.get("kill_switch", {})
                metrics["drawdown_pct"].set(ks.get("current_drawdown", 0) * 100)
                metrics["collateral_available_usd"].set(status.get("nav", 0) - mb.get("real_capital_deployed_usd", 0))
        else:
            # Try collateral cache
            from pathlib import Path as _ColPath
            col_cache = _ColPath(__file__).resolve().parent.parent.parent / "data" / "collateral_cache.json"
            if col_cache.exists():
                col_data = json.load(open(col_cache))
                bc = col_data.get("beta_corridor", {})
                metrics["beta_corridor_value"].set(bc.get("beta", 0))
                metrics["leverage_multiplier"].set(bc.get("leverage_multiplier", 0))
                mb = col_data.get("margin_bucket", {})
                metrics["margin_utilization_pct"].set(mb.get("real_capital_deployed_pct", 0) * 100)
                ks = col_data.get("kill_switch", {})
                metrics["drawdown_pct"].set(ks.get("current_drawdown", 0) * 100)
                metrics["collateral_available_usd"].set(col_data.get("nav", 0) - mb.get("real_capital_deployed_usd", 0))
    except Exception:
        pass

    # ── Chat / NanoClaw Metrics ───────────────────────────────────
    try:
        from engine.agents.nanoclaw.nanoclaw_agent import NanoClawAgent
        agent = NanoClawAgent._instance if hasattr(NanoClawAgent, "_instance") else None
        if agent is not None:
            history = agent.get_history() if hasattr(agent, "get_history") else []
            metrics["active_chat_sessions"].set(1 if history else 0)
    except Exception:
        pass


# ─── Middleware for automatic request tracking ─────────────────────

def create_metrics_middleware(app, metrics: dict):
    """Add middleware to automatically track API request metrics."""
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
    except ImportError:
        logger.warning("starlette not available — skipping request tracking middleware")
        return

    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start

            endpoint = request.url.path
            method = request.method
            status = str(response.status_code)

            metrics["api_requests_total"].labels(
                endpoint=endpoint, method=method, status=status,
            ).inc()
            metrics["api_duration_seconds"].labels(endpoint=endpoint).observe(duration)

            return response

    app.add_middleware(MetricsMiddleware)


# ─── Router Factory ────────────────────────────────────────────────

def create_metrics_router(app=None):
    """Create a FastAPI router that serves /metrics in Prometheus format.

    Args:
        app: Optional FastAPI app instance. If provided, request-tracking
             middleware is automatically installed.

    Returns:
        APIRouter with GET /metrics endpoint.
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import Response
    except ImportError:
        logger.error("FastAPI not installed — cannot create metrics router")
        return None

    if not _prometheus_available:
        router = APIRouter()

        @router.get("/metrics")
        async def metrics_unavailable():
            return Response(
                content="# prometheus_client not installed\n",
                media_type="text/plain",
                status_code=503,
            )
        return router

    # Create a dedicated registry (avoids default process collector noise)
    registry = CollectorRegistry()
    metrics = _create_metrics(registry)

    # Install middleware if app provided
    if app is not None:
        create_metrics_middleware(app, metrics)

    router = APIRouter()

    @router.get("/metrics")
    async def prometheus_metrics():
        _collect_live_metrics(metrics)
        body = generate_latest(registry)
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    return router


# ─── Standalone helper functions for instrumenting other modules ───

_global_metrics = None
_global_registry = None


def get_metrics():
    """Get or create the global metrics dict for use by other modules.

    Example usage in engine code:
        from engine.bridges.prometheus_metrics import get_metrics
        metrics = get_metrics()
        if metrics:
            metrics["openbb_requests_total"].labels(endpoint="get_prices").inc()
    """
    global _global_metrics, _global_registry

    if not _prometheus_available:
        return None

    if _global_metrics is None:
        _global_registry = CollectorRegistry()
        _global_metrics = _create_metrics(_global_registry)

    return _global_metrics
