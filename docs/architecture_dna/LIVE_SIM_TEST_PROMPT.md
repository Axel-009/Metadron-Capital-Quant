# METADRON CAPITAL — LIVE SIMULATION TEST PROMPT

> **Copy this entire file as the opening prompt when testing with any AI model
> (Perplexity, Claude CLI, ChatGPT, etc.).** It instructs the model to build,
> wire, and execute the full signal-to-trade pipeline without frontend,
> monitoring, or infrastructure components.

---

## Prompt

You are testing a quantitative investment platform called Metadron Capital. The
repository is at the GitHub link provided. Your job is to wire and execute the
**full signal-to-trade pipeline** — from data ingestion through to live orders
on IBKR — following the architecture exactly as documented. You are NOT building
the frontend, monitoring stack, or server infrastructure. You are testing the
**engine flow only**: data in, signals processed, intelligence scored, decisions
made, trades executed on the broker.

### What you MUST do before writing any code

1. Read `Architecture_DNA/DATA_FLOW_CHART.md` — this is the authoritative
   single-run data flow. Every engine, every connection, every stage.

2. Read `engine/wiring_manifest.py` — this is the machine-readable component
   registry. Run `python3 -m engine.wiring_manifest` to validate all 43
   components import correctly and the execution chain is connected.

3. Read `Architecture_DNA/ARCHITECTURE_DNA.md` — full architecture specification
   including broker config, allocation rules, gate weights, and risk limits.

4. **Do NOT skip components.** Every engine listed in the wiring manifest must
   be instantiated. Do not cherry-pick. Do not use raw HTTP requests to any
   broker API. Do not bypass L7.

### What you ARE testing (in scope)

**Stage 1 — SCAN:** Instantiate UniverseEngine (4-run scan: SP500, SP400,
SP600, ETF/FI + 26 RV pairs), DataIngestionOrchestrator, DataQualityGate,
UniversalDataPool (cross-asset data dispatcher), and NewsEngine (newsfilter.io
WebSocket + FMP fallback). Pull live market data via OpenBB and IBKR quotes.

**Stage 2 — LAYERS (two independent parallel tracks):**

- **Track A** (MetadronCube path): FedLiquidityPlumbing → MacroEngine (GMTF
  regime + sector ranking) → MetadronCube C(t)=f(L,R,F) with 10-layer tensor
  + KillSwitch → then fans out to parallel signal engines: SecurityAnalysis
  (Graham-Dodd-Klarman), ContagionEngine (21 nodes, 7 shocks), StatArbEngine
  (Medallion mean-reversion), FixedIncomeEngine (yield curve), DistressedAsset
  (5-model ensemble), PatternDiscovery (MiroFish + AI-Newton),
  AdaptiveThresholdCalibrator.

- **Track B** (NewsEngine path, independent from Cube): NewsEngine →
  MiroMomentumEngine (40% news sentiment + 60% agent simulation per
  news-flagged ticker) → enriches EventDrivenEngine (12 event categories) +
  CVREngine (5-model valuation). Track B feeds into MLVoteEnsemble Tier 6.

Track A and Track B run independently with zero cross-dependencies. Their
signals converge at Stage 3.

**Stage 3 — INTELLIGENCE:** AlphaOptimizer receives ALL signals from both
tracks and performs: (1) universe merge from 4 runs, (2) aggregate + dedup by
ticker keeping highest confidence, (3) cap enforcement per allocation bucket
(IG 40%, HY 10%, Distressed 10%, TLTW 15%, FI 5%, CVR 10%, Options 25%
notional, Futures 15%), (4) dual alpha scoring pipeline (Standard: XGBoost 60%
+ LinReg 40% + CAPM 20% blend; Enhanced: WalkForward + FactorLibrary + sector
MVO). Then concurrent ML models feed MLVoteEnsemble (10 tiers, all must vote):
T1 Neural, T2 Momentum, T3 VolRegime, T4 MonteCarlo, T5 Quality, T6
MiroMomentum, T7 Distressed, T8 EventDriven, T9 CVR, T10 CreditQuality.
Output: FullUniverseEngine Scan Slate.

**Stage 4 — DECISION + EXECUTION:**

The scan slate passes through DecisionMatrix — 4 cross-asset gates (all
asset-agnostic, applied uniformly to equities/options/futures/ETFs):

| Gate | Weight | ML Tiers | Function |
|------|--------|----------|----------|
| FUNDAMENTALS | 40% | T1,T5,T7,T9,T10 | Quality, ROIC, credit + regime quality modifier |
| FLOW_HEADLINES | 20% | T6,T8 | News sentiment, ETF flow, sector rotation |
| MACRO_REGIME | 20% | T3,T4 | Direction alignment, VaR headroom, drawdown |
| MOMENTUM | 20% | T1,T2 | RSI, MACD, breakout, cross-asset momentum |

Composite ≥ 0.55 to approve. FUNDAMENTALS is the critical gate.

Approved trades → KellySizer (1.5x multiplier) → AllocationEngine bucket
enforcement → BetaCorridor (7-12% return corridor).

Options MUST go through OptionsSizer (Black-Scholes + Monte Carlo 10K paths +
Kelly): no market price → REJECTED, edge < 200bps → REJECTED, < 5 contracts →
REJECTED. Only mispriced options are traded.

ALL orders route through L7UnifiedExecutionSurface.submit_order() — this is
mandatory. Never call IBKRBroker directly. The L7 surface performs:
1. Research-only guard (reject FI/FX/credit)
2. MultiProductRouter classifies EQUITY / OPTION / FUTURE
3. WonderTrader micro-price adjustment (equities)
4. L7RiskEngine 10-gate pre-trade check (all must pass)
5. SlippageModel pre-trade cost estimate
6. IBKR algo routing: >$50K notional → TWAP, else VWAP/Adaptive/Market
7. IBKRBroker executes (sole broker, via ib_insync)
8. Post-trade: risk state update, TCA decomposition, trade log, Prometheus
9. ExecutionLearningLoop records outcome

Trade log records every generated order for reconciliation (generated vs
executed on broker).

### What you are NOT testing (out of scope)

- Frontend (Express, React, all 30 tabs) — skip entirely
- Nginx, SSL, Cloudflare domain — skip entirely
- Hetzner/Contabo server setup — skip entirely
- Docker, PM2 ecosystem — skip entirely
- Prometheus, Grafana dashboards — skip entirely (metrics are registered but
  scraping infrastructure is not needed for this test)
- Overnight backtesting, QSTrader — skip entirely
- LearningLoop feedback, GSD/Paul plugins, agent scorecard — skip entirely
  (these run post-execution and are not needed to verify the trade flow)

### Step 1: Install ALL dependencies

```bash
pip install -r requirements.txt
```

This installs every package the platform needs. If any package fails, install
it individually. The critical ones are:
```
ib_insync          — IBKR broker connection (sole execution broker)
openbb             — market data (34+ providers, sole data source)
numpy pandas       — core data processing
scikit-learn       — ML models (AlphaOptimizer, UniverseClassifier)
xgboost            — gradient boosted models (alpha scoring, distress)
scipy              — optimization (SLSQP mean-variance)
hmmlearn           — Hidden Markov Models (RegimeEngine)
httpx aiohttp      — async HTTP for data ingestion
prometheus-client  — metrics export (registered but scraping not needed for test)
python-dotenv      — environment variable loading
rich               — terminal output formatting
```

### Step 2: Configure environment

IBKR TWS or IB Gateway must be running on the machine:
```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=7497          # 7497 = paper trading, 7496 = live
export IBKR_CLIENT_ID=1
export IBKR_PAPER_TRADE=True
```

OpenBB for market data:
```bash
export OPENBB_TOKEN=your_token
export FMP_API_KEY=your_fmp_key
```

### Step 3: Validate wiring (MANDATORY before any trade)

```bash
python3 -m engine.wiring_manifest
```

This validates all 43 components import correctly, all 59 wiring edges are
valid, and all 10 routing rules are displayed. **Verdict must be ALL CHECKS
PASSED.** If any component fails to import, fix the dependency before
proceeding. Do NOT skip this step.

### Step 4: Bootstrap ALL components

Use the wiring manifest's bootstrap function to instantiate every component
in the correct dependency order:

```python
from engine.wiring_manifest import bootstrap_full_system

system = bootstrap_full_system(nav=100_000, ibkr_paper=True)
```

This instantiates all 43 components across 7 phases in the correct order:

**Phase 1 DATA (5 components):**
- `system.get("universe_engine")` — UniverseEngine: 4-run scan, ~1,600 securities
- `system.get("data_quality_gate")` — DataQualityGate: stale/completeness/outlier checks
- `system.get("data_ingestion")` — DataIngestionOrchestrator: multi-asset scheduler
- `system.get("data_pool")` — UniversalDataPool: cross-asset dispatcher to engine layers
- `system.get("news_engine")` — NewsEngine: newsfilter.io WebSocket + FMP fallback

**Phase 2 SIGNALS — Track A (8 components, fed by MetadronCube):**
- `system.get("fed_liquidity")` — FedLiquidityPlumbing: SOFR, reserves, TGA, ON-RRP, M2V
- `system.get("macro_engine")` — MacroEngine: GMTF regime + sector ranking (7 sub-engines)
- `system.get("metadron_cube")` — MetadronCube: C(t)=f(L,R,F), 10-layer tensor + KillSwitch
- `system.get("security_analysis")` — SecurityAnalysisEngine: Graham-Dodd-Klarman
- `system.get("contagion_engine")` — ContagionEngine: 21 nodes, 7 shock scenarios
- `system.get("stat_arb_engine")` — StatArbEngine: Medallion mean-reversion + cointegration
- `system.get("distressed_assets")` — DistressedAssetEngine: 5-model ensemble
- `system.get("pattern_discovery")` — PatternDiscoveryEngine: MiroFish + AI-Newton
- `system.get("adaptive_thresholds")` — AdaptiveThresholdCalibrator: rolling percentile
- `system.get("fixed_income_engine")` — FixedIncomeEngine: yield curve, credit spreads, FI signals

**Phase 2 SIGNALS — Track B (3 components, fed by NewsEngine, independent):**
- `system.get("social_prediction")` — MiroMomentumEngine: agent sim on news-flagged tickers
- `system.get("event_driven")` — EventDrivenEngine: 12 categories, enriched by Track B
- `system.get("cvr_engine")` — CVREngine: 5-model valuation, enriched by Track B

**Phase 3 INTELLIGENCE (6 components):**
- `system.get("alpha_optimizer")` — AlphaOptimizer: universe merge + dedup + cap enforce + dual scoring
- `system.get("pattern_recognition")` — PatternRecognitionEngine: candlestick, chart, breakout
- `system.get("universe_classifier")` — UniverseClassifier: XGBoost 4-model, tiers A-G
- `system.get("deep_learning_engine")` — DeepLearningEngine: PPO agent, 50-feature state
- `system.get("social_features")` — SocialFeatureBuilder: sentiment features for ML
- `system.get("model_evaluator")` — ModelEvaluator: P/R/F1 evaluation

**Phase 4 DECISION (5 components):**
- `system.get("decision_matrix")` — DecisionMatrix: 4-gate quality filter (FUNDAMENTALS 40%, FLOW 20%, MACRO 20%, MOMENTUM 20%)
- `system.get("allocation_engine")` — AllocationEngine: bucket sizing + KillSwitch monitor
- `system.get("beta_corridor")` — BetaCorridor: 7-12% return corridor
- `system.get("options_engine")` — OptionsEngine: Black-Scholes Greeks, vol surface
- `system.get("options_sizer")` — OptionsSizer: BS + MC + Kelly, edge ≥ 200bps gate

**Phase 5 EXECUTION (6 components):**
- `system.get("wondertrader_engine")` — WonderTraderEngine: micro-price + CTA + TWAP/VWAP
- `system.get("exchange_core_engine")` — ExchangeCoreEngine: order matching simulation
- `system.get("ibkr_broker")` — IBKRBroker: sole broker, native TWAP/VWAP/Adaptive algos
- `system.get("conviction_override")` — ConvictionOverride: 3-tier override system
- `system.get("l7_execution_surface")` — L7UnifiedExecutionSurface: MANDATORY routing point
- `system.get("tca_engine")` — TCAEngine: spread/impact/timing/commission decomposition

**Phase 6 LEARNING (5 components — instantiated but not active during test):**
- `system.get("learning_loop")` — LearningLoop: 7-channel feedback (records outcomes)
- `system.get("agent_scorecard")` — FullAgentScorecard: agent ranking
- `system.get("sector_bots")` — SectorBotManager: 11 GICS sector bots
- `system.get("research_bots")` — ResearchBotManager: 11 research bots
- `system.get("investor_personas")` — InvestorPersonaManager: 12 personas

**Phase 7 MONITORING (3 components — instantiated but not active during test):**
- `system.get("anomaly_detector")` — AnomalyDetector: statistical anomaly scanner
- `system.get("portfolio_analytics")` — PortfolioAnalytics: scenario engine
- `system.get("memory_monitor")` — MemoryMonitor: session tracking

Every component must instantiate successfully. If IBKRBroker fails (ib_insync
not installed or TWS not running), the system falls back to trade-log-only
mode — orders are recorded but not sent to a broker.

### Step 5: Run the pipeline

After bootstrap, execute the pipeline stages in order:

```python
# Access key components via shortcuts
l7 = system.l7          # L7UnifiedExecutionSurface
cube = system.cube      # MetadronCube
broker = system.broker  # IBKRBroker

# Or run the full live loop orchestrator
from engine.live_loop_orchestrator import LiveLoopOrchestrator
llo = LiveLoopOrchestrator(initial_nav=100_000)

# Execute each phase
data_result = llo.run_data_phase()          # Stage 1: Scan
signals_result = llo.run_signals_phase()    # Stage 2: Track A + Track B parallel
intel_result = llo.run_intelligence_phase() # Stage 3: AlphaOptimizer + ML ensemble
decision_result = llo.run_decision_phase()  # Stage 4: 4-gate filter + allocation
exec_result = llo.run_execution_phase()     # Stage 4: L7 → IBKR
```

Each phase returns a PhaseResult with timing, signal counts, and error details.
Check `signals_result.data["track_a"]` and `signals_result.data["track_b"]`
for per-track telemetry.

### Step 6: Verify trades reached IBKR

Check the trade log for reconciliation:
```python
trade_log = l7.get_trade_log()
recon = l7.get_recon_summary()
print(f"Generated: {recon['total_generated']}")
print(f"Executed on IBKR: {recon['executed_on_broker']}")
print(f"Not executed: {recon['not_executed']}")
```

Check IBKR TWS/Gateway dashboard — orders should be visible if the broker
connection was established.

### Expected output

After running the pipeline, you should see:
1. Universe scan: ~1,600 securities ingested across 4 runs
2. Track A signals: macro regime, cube output, signal engine results
3. Track B signals: news-flagged tickers with MiroMomentum scores
4. AlphaOptimizer: merged, deduped, cap-enforced, scored slate
5. DecisionMatrix: 4-gate composite scores, approved/rejected per ticker
6. L7 execution: orders submitted to IBKR with algo routing
7. Trade log: JSONL entries in logs/l7_execution/trade_log/
8. IBKR dashboard: orders visible on TWS/Gateway

If IBKR is not connected, all orders will appear in the trade log as
NOT_EXECUTED — this is correct behavior. The trade log is the reconciliation
audit trail showing what the platform wanted to do vs what actually executed.

### Critical rules (do not violate)

1. ALL orders through L7UnifiedExecutionSurface — never direct broker calls
2. ALL options through OptionsSizer — rejected without mispricing ≥ 200bps
3. MetadronCube KillSwitch halts ALL execution when triggered
4. DecisionMatrix composite ≥ 0.55 — no trade below this threshold
5. IBKR is the sole broker — no Alpaca, no Tradier, no raw API requests
6. Track A and Track B are independent — no cross-dependencies
7. Every engine in the wiring manifest must be instantiated — no cherry-picking
