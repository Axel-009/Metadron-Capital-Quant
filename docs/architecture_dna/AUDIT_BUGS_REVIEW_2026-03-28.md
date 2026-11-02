# Metadron Capital — Audit & Bugs Review
# Date: 2026-03-28
# Branch: claude/create-audit-bugs-review-QO1sE
# Import Session & Bobby Agent Audit

---

## Platform Status Summary

| Metric | Value |
|--------|-------|
| **Tests** | 158 PASSED / 1 FAILED (99.4%) |
| **Engine Files** | 77 files (~70,902 lines) |
| **Plugins** | 8 files (3,945 lines) |
| **Test Files** | 3 files (159 tests, 1,661 lines) |
| **Claude-Mem** | INSTALLED (v10.4.1, 6 lifecycle hooks) |
| **PIP Dependencies** | 39 declared (Python >= 3.11) |

---

## HARD CRASH BUGS (Will crash at runtime)

| # | File:Line | Bug | Severity |
|---|-----------|-----|----------|
| 1 | `engine/monitoring/heatmap.py:23-24` | Imports `FI_ETFS`, `INTL_ETFS`, `VOL_ETFS` — names don't exist in `universe_engine.py`. Correct names are `FIXED_INCOME_ETFS`, `INTERNATIONAL_ETFS`, `VOLATILITY_ETFS`. Module fails to import entirely. | CRITICAL |
| 2 | `engine/risk/monte_carlo_risk.py:143` | `field(default_factory=...)` used inside a regular `__init__` method (not a dataclass). Crashes on instantiation. | CRITICAL |
| 3 | `engine/risk/monte_carlo_risk.py:145` | Constructor raises `ImportError` if MiroFish unavailable, defeating its own import guard. Engine is completely unusable without MiroFish. | CRITICAL |
| 4 | `engine/execution/execution_engine.py:660` | `SocialFeatureBuilder()` called when it could be `None` (import failed). Raises `TypeError: 'NoneType' object is not callable`. | CRITICAL |
| 5 | `engine/execution/execution_engine.py:1221` | `LearningLoop()` called when it could be `None`. Same crash pattern as #4. | CRITICAL |
| 6 | `engine/execution/execution_engine.py:1149` | `TradierBroker(initial_cash=initial_nav)` called without null check. Alpaca path has a guard but Tradier path doesn't. | CRITICAL |
| 7 | `engine/agents/paul_orchestrator.py:569` | `tier_engine_map` is defined inside `if self._gsd` but referenced inside `if self._paul`. If GSD is `None` but Paul is active, `get_ensemble_adjustments()` crashes with `NameError`. | CRITICAL |
| 8 | `engine/platform_orchestrator.py:1724` | References `self._execution.execute()` but the attribute is `self._execution_internal`. Raises `AttributeError` on HFT trade execution. | CRITICAL |

---

## LOGIC BUGS (Wrong results, not crashes)

| # | File:Line | Bug | Severity |
|---|-----------|-----|----------|
| 9 | `engine/ml/alpha_optimizer.py:666` | `save_model()` saves a freshly instantiated untrained model, not the last trained one. Saved models have random weights and are useless. Model persistence is broken. | HIGH |
| 10 | `engine/data/openbb_data.py` / `engine/portfolio/beta_corridor.py:304` | `period=` parameter passed to `get_adj_close()` which doesn't support it. Kwarg silently ignored, defaulting to `start="2020-01-01"` (5+ years instead of intended 1-2 years). Vol regime classification uses wrong date range. | HIGH |
| 11 | `engine/data/universal_pooling.py:700` | `_pool_econometrics()` calls `get_macro_data(series_id)` but `get_macro_data()` doesn't accept individual series IDs — it fetches ETF proxies. Should call `get_fred_series()` instead. | HIGH |
| 12 | `run_close.py:33-38` | Hardcoded `total_pnl: 0`, `positions: 0` — close report always shows zero P&L regardless of actual trading. | HIGH |
| 13 | `engine/agents/agent_monitor.py:333` | Composite score double-counts accuracy (60% instead of documented 40%). `hit_rate` is set equal to `accuracy_norm`, so both contribute. | MODERATE |
| 14 | `engine/agents/research_bots.py:808` | "Accuracy" metric uses `confidence > 0.5` as proxy for "correct" instead of actual outcomes. Weekly promotion/demotion decisions based on meaningless number. | MODERATE |
| 15 | `intelligence_platform/Kserve/investment_platform_integration.py:219,267` | `ab_test_model()` and `batch_predict()` return random numbers (`np.random.normal`), not actual predictions. Trading decisions partially random if called. | HIGH |

---

## SECURITY ISSUES

| # | File:Line | Bug | Severity |
|---|-----------|-----|----------|
| 16 | `engine/ml/model_store.py:37` | HMAC signing key hardcoded as `"metadron-dev-key-change-me"`. MUST set `MODEL_SIGNING_KEY` env var before live trading or tampered models pass verification. | CRITICAL |
| 17 | `app/backend/main.py:62` | `allow_origins=["*"]` with `allow_credentials=True` and zero authentication middleware. API keys stored but never validated. | HIGH |

---

## UNGUARDED IMPORTS (Violate Design Rule #8)

### Critical (Blocks pipeline if dependency missing)

| # | File:Line | Unguarded Import |
|---|-----------|-----------------|
| 18 | `engine/ml/alpha_optimizer.py:34-38` | `scipy`, `sklearn`, `xgboost` ALL DIRECT — no try/except, no fallback |
| 19 | `engine/ml/model_evaluator.py:28` | `sklearn.metrics` DIRECT |
| 20 | `engine/ml/bridges/markov_regime_bridge.py:44` | `hmmlearn.hmm` DIRECT |
| 21 | `engine/signals/macro_engine.py:31` → `metadron_cube.py:44` | Chain failure: `yahoo_data` fails → `macro_engine` fails → `metadron_cube` fails → ENTIRE PIPELINE DOWN |
| 22 | `engine/agents/sector_bots.py:20-25` | 5 unguarded imports (worst offender in agents layer) |
| 23 | `engine/agents/gics_sector_agents.py:23-24`, `research_bots.py:34-35` | `numpy`/`pandas` DIRECT |

### Moderate (Blocks specific features)

| # | File | Unguarded Import |
|---|------|-----------------|
| 24 | `engine/execution/paper_broker.py:46` | `yahoo_data` DIRECT |
| 25 | `engine/execution/tradier_broker.py:36` | `httpx` DIRECT |
| 26 | `engine/execution/decision_matrix.py:32` | `beta_corridor` DIRECT |
| 27 | `engine/portfolio/beta_corridor.py:15` | `yahoo_data` DIRECT |
| 28 | `engine/monitoring/daily_report.py:27-29` | `yahoo_data`, `universe_engine`, `macro_engine` DIRECT |
| 29 | `engine/monitoring/heatmap.py:23-26` | `yahoo_data`, `universe_engine` DIRECT |
| 30 | `engine/monitoring/hourly_recap.py:22-23` | `yahoo_data`, `universe_engine` DIRECT |

---

## TEST FAILURE

| Test | File | Issue | Fix |
|------|------|-------|-----|
| `test_status_mapping` | `tests/test_engine.py` (TestAlpacaBrokerModule) | `_ALPACA_STATUS_MAP` is empty when `alpaca-py` SDK not installed. Test asserts `map["filled"] == OrderStatus.FILLED` but gets `None`. | Add fallback status map when `_ALPACA_AVAILABLE is False`, OR `pytest.skipif` when alpaca-py not installed. |

---

## ADDITIONAL FINDINGS

### State Persistence Gap
- `run_open.py:35`, `run_hourly.py:29`, `run_close.py:33-38` all create fresh `ExecutionEngine` instances with hardcoded $1M NAV.
- Positions, P&L, and portfolio state are lost between invocations.
- **Recommendation**: Use `LiveLoopOrchestrator` exclusively for live trading — it has proper state persistence, circuit breakers, and thread safety.

### Silent Paper-Fallback in Live Mode
- `l7_unified_execution_surface.py:1642-1647` — When both Alpaca and Tradier connections fail, orders silently "fill" in paper-fallback mode with `[paper-fallback]` appended to the reason. No alert is raised. Orders appear filled but never reach the market.

### Backend API is a Shell (0% wired to engine)
- `/api/hedge-fund/pipeline` returns hardcoded empty stubs
- `/api/hedge-fund/macro` returns `None` for all fields
- `/api/hedge-fund/signals` returns empty list always
- `AgentService` imports from non-existent `engine.agents.registry`
- `GraphService` imports from non-existent `engine.pipeline.PipelineGraph`
- Does NOT block live trading (CLI-based), but API is non-functional.

### Stale Data
- `event_driven_engine.py` — `LIVE_EVENTS` catalog contains stale 2023 deals (HZNP/AMGN, ATVI/MSFT). Needs refreshing.
- `social_prediction_engine.py` — Crypto ticker mappings (BITO) conflict with no-crypto policy.

### Dead Code
- `agent_sim_engine.py` — `AgentType`, `MarketAgent` imported but never used; `median_return` computed but unused.
- `worldmonitor_bridge.py:276` — `_fetch_category_events()` is a no-op returning empty lists. All WorldMonitor signals will be zero.
- `engine/ml/bridges/__init__.py` — `MarkovRegimeBridge` and `WorldMonitorBridge` NOT re-exported.

### Model Store Bug
- `model_store.py:265` — `cleanup()` timestamp parsing broken for same-date model versions (splits on `_` and takes only date portion).

### NAV Inconsistency
- `platform_orchestrator.py` defaults to $100M NAV
- `run_*.py` scripts use $1M NAV

### Test Contradictions
- `test_l7_execution.py:97-100` says commodity ETFs (GLD, SLV, USO) ARE tradeable
- `test_engine.py:664` says GLD not in `TRADEABLE_ETFS` (macro-only)

### Agent Scorecard Risk
- `agent_scorecard.py:306` — Module-level `assert len(AGENT_REGISTRY) == 25` will crash entire module at import if someone adds/removes an agent. Should be a warning.

---

## IMPROVEMENTS SINCE LAST AUDIT (2026-03-26)

| # | Change | Impact |
|---|--------|--------|
| 1 | `execution_engine.py` — ALL 12 optional imports now GUARDED | MAJOR fix |
| 2 | `alpaca_broker.py` — NEW file, proper alpaca SDK guard | NEW |
| 3 | `agent_sim_engine.py` — NEW signal source, MiroFish simulation bridge | NEW |
| 4 | `model_store.py` — NEW secure model persistence with HMAC-SHA256 | NEW |
| 5 | `monte_carlo_risk.py` — NEW risk engine with VaR/CVaR/Stress | NEW |
| 6 | `live_earnings_graph.py` — NEW ASCII P&L chart with np=None guard | NEW |
| 7 | `platinum_report_v2.py` — NEW enhanced report with proper guards | NEW |
| 8 | Test coverage: 64 → 159 tests (+148%) | MAJOR |

---

## EXEMPLARY GRACEFUL DEGRADATION (Best Practice)

1. `live_loop_orchestrator.py` — ALL 20+ imports guarded (best in codebase)
2. `l7_unified_execution_surface.py` — ALL 9 sub-engines guarded
3. `execution_engine.py` — ALL 12 optional imports now guarded (IMPROVED)
4. `universe_classifier.py` — `HAS_SKLEARN` + `HAS_XGB` flags + rule-based fallback
5. `research_bots.py` — Full stub functions for every import
6. `fed_liquidity_plumbing.py` — Stub functions returning empty DataFrames
7. `ingestion_orchestrator.py` — 3 availability flags + comprehensive stubs
8. `universal_pooling.py` — 7+ guarded imports with full stubs
9. `run_open.py` / `run_close.py` — Every engine wrapped in individual try/except

---

## PRE-LIVE TRADING CHECKLIST

### MUST-FIX

- [ ] Fix 8 hard crash bugs listed above (especially #1-8)
- [ ] Set `MODEL_SIGNING_KEY` environment variable (security issue #16)
- [ ] Ensure `scipy`, `sklearn`, `xgboost` installed: `pip install scipy scikit-learn xgboost`
- [ ] Ensure `alpaca-py` installed if using Alpaca broker: `pip install alpaca-py`
- [ ] Ensure `httpx` installed if using Tradier broker: `pip install httpx`
- [ ] Set `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` in `.env` if going live with Alpaca

### SHOULD-FIX

- [ ] Wrap `alpha_optimizer.py:34-38` in try/except with rule-based fallback
- [ ] Wrap `model_evaluator.py:28` sklearn.metrics in try/except
- [ ] Wrap `markov_regime_bridge.py:44` hmmlearn.hmm in try/except
- [ ] Fix `alpha_optimizer.py:666` save_model() to save trained model
- [ ] Fix alpaca_broker test (add fallback status map or `pytest.skipif`)
- [ ] Guard `macro_engine.py:31` yahoo_data import to prevent cascade failure

### NICE-TO-HAVE

- [ ] Guard `daily_report.py` / `heatmap.py` / `hourly_recap.py` data imports
- [ ] Guard `sector_bots.py` core imports (follow `research_bots.py` pattern)
- [ ] Guard `beta_corridor.py` and `paper_broker.py` yahoo_data imports
- [ ] Update stale LIVE_EVENTS in event_driven_engine.py
- [ ] Wire backend API to actual engine (currently returns stubs)
- [ ] Add authentication middleware to FastAPI backend

---

## COMPONENT STATUS MATRIX

### Signals (13 files, 13,632 lines)

| Component | Lines | Guards | Status |
|-----------|-------|--------|--------|
| macro_engine.py | 2,503 | DIRECT | FULL |
| security_analysis_engine.py | 1,432 | GUARD | FULL |
| social_prediction_engine.py | 670 | DIRECT | FULL |
| metadron_cube.py | 1,302 | DIRECT | FULL |
| event_driven_engine.py | 663 | DIRECT | FULL |
| distressed_asset_engine.py | 1,105 | DIRECT | FULL |
| cvr_engine.py | 702 | DIRECT | FULL |
| stat_arb_engine.py | 1,689 | GUARD | FULL |
| contagion_engine.py | 1,563 | DIRECT | FULL |
| fed_liquidity_plumbing.py | 1,319 | GUARD | FULL |
| pattern_discovery_engine.py | 281 | GUARD | FULL |
| agent_sim_engine.py | 394 | GUARD | FULL (NEW) |

### ML (17 files, 10,770 lines)

| Component | Lines | Guards | Status |
|-----------|-------|--------|--------|
| alpha_optimizer.py | 1,515 | UNGUARDED | FULL (3 deps) |
| backtester.py | 1,429 | GUARD | FULL |
| model_evaluator.py | 279 | UNGUARDED | sklearn req'd |
| model_store.py | 276 | GUARD | FULL (NEW) |
| deep_learning_engine.py | 943 | DIRECT | FULL |
| pattern_recognition.py | 1,591 | GUARD | FULL |
| universe_classifier.py | 780 | GUARD | FULL (BEST) |
| qstrader_backtest_bridge.py | 2,033 | GUARD | FULL |
| social_features.py | 193 | DIRECT | FULL |
| 8 bridges | ~1,695 | GUARD (mostly) | FULL |

### Agents (10 files, 8,049 lines)

| Component | Lines | Guards | Status |
|-----------|-------|--------|--------|
| paul_orchestrator.py | 617 | GUARD | BUG (#7) |
| sector_bots.py | 897 | UNGUARDED | 5 imports |
| research_bots.py | 967 | GUARD | BEST PRACTICE |
| investor_personas.py | 992 | GUARD | FULL |
| gics_sector_agents.py | 1,224 | GUARD | np/pd unguarded |
| dynamic_agent_factory.py | 870 | GUARD | FULL |
| enforcement_engine.py | 617 | GUARD | FULL |
| agent_scorecard.py | 1,337 | GUARD | assert risk |
| agent_monitor.py | 513 | DIRECT | accuracy bug |

### Execution (13 files, 16,181 lines)

| Component | Lines | Guards | Status |
|-----------|-------|--------|--------|
| execution_engine.py | 2,364 | GUARD | 3 null bugs |
| l7_unified_execution_surface.py | 1,832 | GUARD | BEST PRACTICE |
| alpaca_broker.py | 1,146 | GUARD | FULL (NEW) |
| exchange_core_engine.py | 1,354 | GUARD | FULL |
| wondertrader_engine.py | 868 | DIRECT | FULL |
| paper_broker.py | 1,486 | UNGUARDED | yahoo_data |
| tradier_broker.py | 1,018 | UNGUARDED | httpx |
| options_engine.py | 1,591 | GUARD | FULL |
| decision_matrix.py | 1,344 | UNGUARDED | beta_corridor |
| conviction_override.py | 1,260 | GUARD | FULL |
| quant_strategy_executor.py | 683 | DIRECT | FULL |
| missed_opportunities.py | 1,234 | GUARD | FULL |

---

*Audit conducted by Bobby Agent & Import Session — 2026-03-28*
*Platform: Metadron Capital Intelligence Platform*
*Branch: claude/create-audit-bugs-review-QO1sE*
