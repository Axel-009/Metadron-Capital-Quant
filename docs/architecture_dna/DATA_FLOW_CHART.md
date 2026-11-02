# METADRON CAPITAL — SINGLE RUN DATA FLOW CHART

Every engine in the system mapped to its position in a single execution cycle.
This is the authoritative reference for how data flows from market scan to
trade execution to learning feedback.

```
═══════════════════════════════════════════════════════════════════════════════
 STAGE 1: SCAN — Data Ingestion (every tick)
═══════════════════════════════════════════════════════════════════════════════

  EXTERNAL SOURCES                    INTERNAL PIPELINE
  ─────────────────                   ──────────────────

  OpenBB (34+ providers)─┐
  FRED (40+ series)──────┤
  FMP (fundamentals)─────┤     ┌──────────────────────────────┐
  IBKR (real-time quotes)┤────→│  DataIngestionOrchestrator   │
  newsfilter.io (10K+)───┤     │  (multi-asset scheduler)     │
  SEC filings────────────┘     └──────────────┬───────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  DataQualityGate   │
                                    │  stale │ complete │ │
                                    │  outlier checks    │
                                    └─────────┬──────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                │ UniverseEngine  │ │UniversalDataPool│ │   NewsEngine    │
                │                 │ │ cross-asset      │ │ newsfilter.io   │
                │ 4-RUN SCAN:     │ │ routing to       │ │ + FMP fallback  │
                │ Run 1: SP500    │ │ engine layers    │ │ sentiment score │
                │  (~500 tickers) │ │                  │ │ urgency rating  │
                │ Run 2: SP400    │ │                  │ │                 │
                │  (~400 tickers) │ │                  │ │                 │
                │ Run 3: SP600    │ │                  │ │                 │
                │  (~600 tickers) │ │                  │ │                 │
                │ Run 4: ETF + FI │ │                  │ │                 │
                │  (~70 tickers)  │ │                  │ │                 │
                │ + 26 RV pairs   │ │                  │ │                 │
                │                 │ │                  │ │                 │
                │ Total: ~1,600   │ │                  │ │                 │
                │ securities      │ │                  │ │                 │
                └────────┬────────┘ └────────┬─────────┘ └────────┬────────┘
                         │                   │                    │
                         └───────────┬───────┘                    │
                                     │                            │
                                     │                            │
═════════════════════════════════════╪════════════════════════════╪════════
 STAGE 2: LAYERS — Signal Processing (1-min cadence)             │
═════════════════════════════════════╪════════════════════════════╪════════
                                     │                            │
   TRACK A: SIGNAL ENGINES           │       TRACK B: NEWS+MIRO  │
   (MetadronCube → signals)          │       (NewsEngine → Miro) │
                                     │                            │
          ┌──────────────────────────▼──────────────────┐         │
          │            FedLiquidityPlumbing              │         │
          │  SOFR │ reserves │ TGA │ ON-RRP │ M2V       │         │
          │  → CubeLiquidityTensor │ MoneyVelocityTracker│        │
          └──────────────────────────┬──────────────────┘         │
                                     │                            │
          ┌──────────────────────────▼──────────────────┐         │
          │              MacroEngine (GMTF)              │         │
          │  ┌────────────────────────────────────────┐  │         │
          │  │ MoneyVelocityModule │ SectorRanker     │  │         │
          │  │ CarryToVolatility   │ RegimeTransition │  │         │
          │  │ YieldCurveAnalyzer  │ CreditPulseMonitor│ │         │
          │  │ MacroFeatureBuilder (50+ features)     │  │         │
          │  └────────────────────────────────────────┘  │         │
          │  → regime │ rm_adjustment │ sector_weights   │         │
          └──────────────────────────┬──────────────────┘         │
                                     │                            │
          ┌──────────────────────────▼──────────────────┐         │
          │         MetadronCube  C(t) = f(L,R,F)       │         │
          │  ┌────────────────────────────────────────┐  │         │
          │  │ L0: FedPlumbingLayer                   │  │         │
          │  │ L1: LiquidityTensor    L(t) ∈ [-1,+1] │  │         │
          │  │ L2: ReserveFlowKernel  ΔRes → ΔSector  │  │         │
          │  │ LR: RiskStateModel     R(t) ∈ [0,1]   │  │         │
          │  │ LF: CapitalFlowModel   F(t) rotation   │  │         │
          │  │ L4: RegimeEngine       HMM+RL 4-regime │  │         │
          │  │ LG: GateZAllocator     5-sleeve alloc  │  │         │
          │  │ LE: GateLogic          4-gate entry     │  │         │
          │  │ LK: KillSwitch ════════════════════════╪══╪════► HALT
          │  │ LC: FCLPLoop           daily recal      │  │         │
          │  └────────────────────────────────────────┘  │         │
          │  → regime │ leverage │ beta_cap │ sleeves    │         │
          └──────────────────────────┬──────────────────┘         │
                                     │                            │
          ┌──────────────────────────┴──────────────┐             │
          │                                         │             │
          │   TRACK A: PARALLEL SIGNAL ENGINES      │             │
          │   (fed by MetadronCube regime output)    │             │
          │                                         │             │
          │  ┌───────────────────┐  ┌──────────────┐│  ┌─────────▼───────────┐
          │  │SecurityAnalysis   │  │ContagionEngin││  │ News+MiroMomentum   │
          │  │Graham-Dodd-Klarman│  │21 nodes,     ││  │ Pipeline            │
          │  │top-down + bottom  │  │7 shock scen, ││  │                     │
          │  │up, MoS ≥33%      │  │multi-step    ││  │ run_miro_on_news_   │
          │  └────────┬──────────┘  └──────┬───────┘│  │  tickers():         │
          │           │                    │        │  │ 1. Flag tickers     │
          │  ┌────────▼──────────┐  ┌──────▼───────┐│  │    with breaking    │
          │  │  StatArbEngine    │  │FixedIncome   ││  │    news             │
          │  │  Medallion mean-  │  │Engine, yield ││  │ 2. MiroMomentum     │
          │  │  reversion + co-  │  │curve, credit ││  │    per ticker       │
          │  │  integration pairs│  │spreads       ││  │ 3. Combined score:  │
          │  └────────┬──────────┘  └──────┬───────┘│  │    40% sentiment    │
          │           │                    │        │  │    60% agent sim    │
          │  ┌────────▼──────────┐  ┌──────▼───────┐│  │                     │
          │  │DistressedAsset    │  │PatternDisc   ││  │ → EventDrivenEngine │
          │  │5-model ensemble:  │  │MiroFish +    ││  │ → CVREngine         │
          │  │Altman Z, Merton   │  │AI-Newton PySR││  │ → MLVoteEnsemble T6 │
          │  │KMV, Ohlson,       │  │→ PatternBus  ││  │ → MLVoteEnsemble T6│
          │  │Zmijewski, ML GBM  │  │              ││  └─────────┬───────────┘
          │  └────────┬──────────┘  └──────┬───────┘│            │
          │           │                    │        │  ┌─────────▼───────────┐
          │  ┌────────▼──────────┐  ┌──────▼───────┐│  │ EventDrivenEngine   │
          │  │AdaptiveThreshold  │  │VelocityEngin ││  │ 12 categories:      │
          │  │252-day rolling    │  │money velocity││  │ M&A arb, PEAD,      │
          │  │percentile calib   │  │tracking      ││  │ spinoff, activist,  │
          │  └────────┬──────────┘  └──────┬───────┘│  │ buyback, catalyst   │
          │           │                    │        │  └─────────┬───────────┘
          │           └─────────┬──────────┘        │            │
          │                     │                   │  ┌─────────▼───────────┐
          │        TRACK A signals                  │  │     CVREngine       │
          │                     │                   │  │ 5-model valuation:  │
          └─────────────────────┼───────────────────┘  │ binary opt, barrier,│
                                │                      │ milestone, MC, real │
                                │                      │ liquidity/credit adj│
                                │                      └─────────┬───────────┘
                                │                                │
                                │                   TRACK B signals
                                │                                │
                                └────────────────┬───────────────┘
                                                 │
                                                 ▼
                                        ALL SIGNALS MERGE
                                       (Track A + Track B)
                                                 │
═══════════════════════════╪══════════════════════════════════════════════
 STAGE 3: INTELLIGENCE — ML/AI Decision Layer (5-min cadence)
═══════════════════════════╪══════════════════════════════════════════════
                        │
     ┌──────────────────▼──────────────────────────────────────┐
     │      AlphaOptimizer (dual pipeline + universe merge)    │
     │                                                         │
     │  STEP 1 — Universe Merge:                               │
     │    Receive all ~1,600 tickers from UniverseEngine       │
     │    (4 runs: SP500 + SP400 + SP600 + ETF/FI + RV pairs) │
     │    + all signal outputs from Track A + Track B          │
     │                                                         │
     │  STEP 2 — Aggregate + Dedup:                            │
     │    aggregate_runs(): merge all 4 universe runs          │
     │    dedup by ticker: keep highest-confidence signal      │
     │    when same ticker appears across multiple runs        │
     │                                                         │
     │  STEP 3 — Cap Enforcement:                              │
     │    Enforce allocation bucket limits per ticker:          │
     │    IG ≤ 40% │ HY ≤ 10% │ Distressed ≤ 10%             │
     │    TLTW ≤ 15% │ FI ≤ 5% │ CVR ≤ 10%                   │
     │    Options ≤ 25% notional │ Futures ≤ 15%              │
     │    Reject tickers that would breach bucket caps         │
     │                                                         │
     │  STEP 4 — Alpha Scoring (dual pipeline):                │
     │    Standard: XGBoost(60%) + LinReg(40%) + CAPM(20%)    │
     │              + QualityRanker + 22 features + EWMA cov   │
     │              → SLSQP mean-variance optimization         │
     │    Enhanced: WalkForward + FactorLibrary(50+ factors)   │
     │              + sector MVO                               │
     │                                                         │
     │  → FullUniverseEngine Scan Slate                        │
     │    (scored, deduped, cap-enforced, allocation-ready)     │
     └──────────────────┬──────────────────────────────────────┘
                        │
     CONCURRENT ML MODELS (all feed into MLVoteEnsemble):
     ┌──────────────────┼──────────────────────────────────────┐
     │                  │                                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  PatternRecognitionEngine     │                      │
     │  │  candlestick, chart, breakout │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  UniverseClassifier           │                      │
     │  │  XGBoost 4-model, tiers A-G  │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  DeepLearningEngine           │                      │
     │  │  PPO agent, 50-feature state  │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  ML Bridges (6):              │                      │
     │  │  FinRL │ MonteCarlo │ StockPred│                     │
     │  │  Markov │ TFT │ DeepTrading   │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  SectorBotManager (11 GICS)   │                      │
     │  │  + SocialFeatureBuilder       │                      │
     │  │  + ModelEvaluator             │                      │
     │  └───────────────┬───────────────┘                      │
     │                  │                                      │
     └──────────────────┼──────────────────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────────────────┐
     │          MLVoteEnsemble (10-tier weighted vote)          │
     │                                                         │
     │  T1:  Neural Net (FinRL/DRL)              weight 1.0   │
     │  T2:  Momentum/Mean-Reversion             weight 1.2   │
     │  T3:  Volatility Regime                   weight 0.8   │
     │  T4:  Monte Carlo                         weight 0.9   │
     │  T5:  Quality (SecurityAnalysis grade)    weight 1.1   │
     │  T6:  MiroMomentum (News+AgentSim) ◄──── cross-stage   │
     │  T7:  Distressed                          weight 0.9   │
     │  T8:  EventDriven                         weight 1.0   │
     │  T9:  CVR                                 weight 0.7   │
     │  T10: CreditQuality                       weight 0.9   │
     │                                                         │
     │  → weighted vote score [-10, +10]                       │
     │  → minimum edge = 2.0 + max(0, -vote_score) bps        │
     └──────────────────┬──────────────────────────────────────┘
                        │
═══════════════════════════╪══════════════════════════════════════════════
 STAGE 4: DECISION + EXECUTION
═══════════════════════════╪══════════════════════════════════════════════
                        │
          ┌─────────────▼─────────────────────────────────────┐
          │    DECISION (single path — no bypass)             │
          │                                                    │
          │  FullUniverseEngine Scan Slate (from Stage 3)     │
          │    → DecisionMatrix 4-gate quality filter          │
          │      (cross-asset, asset-agnostic, pass/fail)     │
          │                                                    │
          │    ┌──────────────────────────────────────────┐    │
          │    │ G1 FUNDAMENTALS  40%  ── quality, ROIC,  │    │
          │    │    FCF, Graham-Dodd, credit, earnings    │    │
          │    │    ML tiers: T1,T5,T7,T9,T10             │    │
          │    │    + regime quality modifier              │    │
          │    │                                           │    │
          │    │ G2 FLOW_HEADLINES 20% ── ETF flow, news, │    │
          │    │    sector rotation                        │    │
          │    │    ML tiers: T6 (MiroMomentum), T8 (Event)│   │
          │    │                                           │    │
          │    │ G3 MACRO_REGIME  20%  ── direction align, │    │
          │    │    VaR headroom, drawdown check           │    │
          │    │    ML tiers: T3 (VolRegime), T4 (MC)     │    │
          │    │                                           │    │
          │    │ G4 MOMENTUM      20%  ── RSI, MACD,      │    │
          │    │    breakout, cross-asset momentum         │    │
          │    │    ML tiers: T1 (Neural), T2 (Momentum)  │    │
          │    │                                           │    │
          │    │ composite ≥ 0.55 → APPROVED               │    │
          │    │ FUNDAMENTALS must pass (critical gate)    │    │
          │    └──────────────────────────────────────────┘    │
          │                                                    │
          │    → KellySizer (1.5x multiplier)                  │
          │    → AllocationEngine.apply_rules()                │
          │    → BetaCorridor (7-12% return corridor)          │
          │                                                    │
          │  DIRECT ROUTES (high conviction, still via L7):    │
          │    EVENT_DIRECT      (conviction ≥ 0.7)            │
          │    CVR_DIRECT        (STRONG_BUY)                  │
          └─────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────────────────────────────┐
          │  OPTIONS SIZING (edge-gated, runs before L7)      │
          │                                                    │
          │  OptionsSizer:                                     │
          │    1. Black-Scholes theoretical price               │
          │    2. Monte Carlo (10K paths) → win prob            │
          │    3. Fair value = avg(BS, MC) vs market price      │
          │    4. Edge < 200bps → REJECTED (no allocation)     │
          │    5. Kelly criterion sizes on detected edge        │
          │    6. Minimum 5 contracts                           │
          │    → contracts + greeks + edge_bps                  │
          └─────────────┬─────────────────────────────────────┘
                        │
   ╔════════════════════▼═════════════════════════════════════╗
   ║        L7 UNIFIED EXECUTION SURFACE                      ║
   ║        (SOLE entry point — ALL orders route here)        ║
   ║                                                          ║
   ║  ┌───────────────────────────────────────────────────┐   ║
   ║  │ Step 1: Research-only guard (reject FI/FX/credit) │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 2: MultiProductRouter.classify()             │   ║
   ║  │         EQUITY │ OPTION │ FUTURE                  │   ║
   ║  └────┬──────────────┬──────────────┬────────────────┘   ║
   ║       │              │              │                    ║
   ║  ┌────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐            ║
   ║  │ EQUITY   │  │  OPTION   │  │  FUTURE   │            ║
   ║  │WonderTrdr│  │OptionsSzr │  │BetaCorrdr │            ║
   ║  │micro-    │  │BS+MC+Kelly│  │hedge val  │            ║
   ║  │price adj │  │edge gate  │  │           │            ║
   ║  └────┬─────┘  └─────┬─────┘  └────┬──────┘            ║
   ║       │              │              │                    ║
   ║  ┌────▼──────────────▼──────────────▼────────────────┐   ║
   ║  │ Step 4: L7RiskEngine — 10 gates (ALL must pass)   │   ║
   ║  │ G1 pos 10% │ G2 sector 30% │ G3 loss 3%          │   ║
   ║  │ G4 gross 250% │ G5 net 150% │ G6 throttle 100    │   ║
   ║  │ G7 dd 10% │ G8 cash │ G9 opt Δ 20% │ G10 fut 50% │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 5: SlippageModel (pre-trade cost estimate)   │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 6: IBKR ALGO ROUTING                         │   ║
   ║  │                                                    │   ║
   ║  │  Notional > $50K ──→ TWAP (server-side splitting) │   ║
   ║  │  VWAP routing ─────→ VWAP (volume participation)  │   ║
   ║  │  Medium urgency ───→ Adaptive (IBKR auto-selects) │   ║
   ║  │  HIGH/CRITICAL ────→ Market order (immediate fill) │   ║
   ║  │                                                    │   ║
   ║  │  ┌─────────────────────────────────────────────┐   │   ║
   ║  │  │           IBKRBroker                        │   │   ║
   ║  │  │  TWS/Gateway via ib_insync                  │   │   ║
   ║  │  │  Equities + Options + Futures               │   │   ║
   ║  │  │  $0.005/share │ $0.65/option │ $1.25/future │   │   ║
   ║  │  └─────────────────────────────────────────────┘   │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 7: Post-trade                                │   ║
   ║  │  → L7RiskEngine.post_trade_update()               │   ║
   ║  │  → TransactionCostAnalyzer.analyze()              │   ║
   ║  │  │   spread │ impact │ timing │ commission        │   ║
   ║  │  → ExecutionLearningLoop.record_outcome()         │   ║
   ║  │  → Trade Log (JSONL recon audit)                  │   ║
   ║  │  → Prometheus metrics (17 gauges/counters)        │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ╚══════════════════════╪═══════════════════════════════════╝
                          │
              EXECUTION OUTCOMES
              (fills, costs, slippage, P&L)
                          │
═════════════════════════╪════════════════════════════════════════════
 STAGE 5: LEARNING + MONITORING (continuous, feeds back to Stages 2-4)
═════════════════════════╪════════════════════════════════════════════
                          │
     ┌────────────────────▼────────────────────────────────────┐
     │              LearningLoop (7 channels)                  │
     │                                                         │
     │  CH1: SIGNAL_ACCURACY   → retune tier weights ─────────┼──→ Stage 3
     │  CH2: EXECUTION_QUALITY → retune slippage model ────────┼──→ Stage 4
     │  CH3: REGIME_FEEDBACK   → retune HMM priors ───────────┼──→ Stage 2
     │  CH4: ALPHA_DECAY       → retune half-life ─────────────┼──→ Stage 3
     │  CH5: RISK_CALIBRATION  → retune gate thresholds ───────┼──→ Stage 4
     │  CH6: AGENT_PERFORMANCE → promote/demote agents ────────┼──→ Stage 3
     │  CH7: CROSS_ASSET       → retune sector allocation ─────┼──→ Stage 2
     │                                                         │
     │  Damping: MAX_WEIGHT_CHANGE = 0.05/cycle                │
     │  Oscillation detection: 3-reversal threshold             │
     └────────────────────┬────────────────────────────────────┘
                          │
     CONCURRENT LEARNING SYSTEMS:
     ┌────────────────────┼────────────────────────────────────┐
     │                    │                                    │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  GSDPlugin (gradient dynamics)    │                  │
     │  │  signal momentum │ convergence    │                  │
     │  │  divergence │ gradient decay      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  PaulPlugin (pattern memory)      │                  │
     │  │  pattern matching │ evolution     │                  │
     │  │  context-aware replay │ library   │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  PaulOrchestrator                 │                  │
     │  │  agent create/promote/demote      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  Agent Hierarchy:                 │                  │
     │  │  FullAgentScorecard               │                  │
     │  │  ├─ DIRECTOR (Sharpe>2.5, acc>85%)│                  │
     │  │  ├─ GENERAL  (Sharpe>2.0, acc>80%)│                  │
     │  │  ├─ CAPTAIN  (Sharpe>1.5, acc>55%)│                  │
     │  │  ├─ LIEUTENANT (Sharpe>1.0)       │                  │
     │  │  └─ RECRUIT  (below thresholds)   │                  │
     │  │                                   │                  │
     │  │  SectorBotManager (11 GICS bots)  │                  │
     │  │  ResearchBotManager (11 research) │                  │
     │  │  GICSSectorAgentManager            │                  │
     │  │  AgentMonitor (4-tier hierarchy)  │                  │
     │  │  InvestorPersonaManager (12)      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │                    │                                    │
     └────────────────────┼────────────────────────────────────┘
                          │
     MONITORING:
     ┌────────────────────┼────────────────────────────────────┐
     │                    │                                    │
     │  AnomalyDetector ──┤── statistical anomaly scan         │
     │  PortfolioAnalytics┤── scenario engine + risk metrics   │
     │  MemoryMonitor ────┤── session tracking + EOD summary   │
     │  20% position DD ──┤── auto-liquidation via L7          │
     │  Circuit breakers ─┘── halt on breach                   │
     │                                                         │
     └─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 CROSS-STAGE PIPELINES (span multiple stages)
═══════════════════════════════════════════════════════════════════════════════

  1. UNIVERSE → INTELLIGENCE PIPELINE:
     Stage 1 (UniverseEngine 4-run scan: SP500+SP400+SP600+ETF/FI+RV pairs)
     → Stage 2 (Track A signals enrich each ticker)
     → Stage 3 (AlphaOptimizer: aggregate + dedup + cap enforce + score)
     → Stage 4 (FullUniverseEngine Scan Slate → AllocationEngine)

  2. NEWS+MIRO PIPELINE:
     Stage 1 (NewsEngine) → Stage 2 Track B (run_miro_on_news_tickers:
     40% sentiment + 60% agent sim) → enrich EventDriven + CVR
     → Stage 3 (MLVoteEnsemble T6 + AlphaOptimizer convergence)

  3. KILL SWITCH CHAIN:
     Stage 2 (MetadronCube.KillSwitch) → Stage 4 (AllocationEngine.cube_kill
     _switch_override) → Stage 4 (L7RiskEngine.G7) → HALT all execution

  4. LEARNING FEEDBACK LOOP:
     Stage 5 (LearningLoop) → Stage 2 (regime priors, sector weights)
                             → Stage 3 (tier weights, ensemble tuning)
                             → Stage 4 (gate thresholds, slippage model)

  5. OPTIONS EDGE PIPELINE:
     Stage 3 (signal with market_price) → Stage 4 (OptionsSizer: BS + MC →
     fair value vs market → edge ≥ 200bps? → Kelly sizing → 5+ contracts)
     → Stage 4 (L7 IBKR execution)


═══════════════════════════════════════════════════════════════════════════════
 LEGEND
═══════════════════════════════════════════════════════════════════════════════

  ──→    Data flow direction
  ◄──    Feedback / enrichment
  ═══    Stage boundary
  ╔══╗   L7 Execution Surface (mandatory routing point)
  │  │   Engine / component box
  ┌──┐   Sub-component or group
  HALT   Kill switch termination point

  TIMING:
    Stage 1: Every tick (continuous)
    Stage 2: 1-minute cadence
    Stage 3: 5-minute cadence
    Stage 4: On signal delta (event-driven)
    Stage 5: Continuous (per-fill for learning, 5-min for monitoring)

  TOTAL ENGINES: 70+
  TOTAL ML MODELS: 15+ (10-tier ensemble + bridges + PPO + XGBoost)
  TOTAL SIGNAL TYPES: 29
  SINGLE RUN CYCLE: ~20 minutes (4 universe runs × 150s + 5min aggregate + 5min risk)
```
