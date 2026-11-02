# Intelligence Platform — Complete Repository Collection

All 28 reference repositories fully consolidated into Metadron Capital as the unified
intelligence platform. **Every file from every source repo is included** — Python, Go,
Rust, C++, CUDA, TypeScript, JavaScript, Vue, Java, proto, configs, data, and assets.

## Repository Index

### Layer 1: Data Ingestion & Market Intelligence

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 1 | **Financial-Data** | Data pipelines | Financial data ingestion, cleaning |
| 2 | **open-bb** | OpenBB terminal | Financial data terminal, research platform |
| 3 | **hedgefund-tracker** | Fund tracking | Hedge fund performance, 13F filings |
| 4 | **FRB** | Federal Reserve data | FRED API, M1/M2, rates, GDP, employment |
| 5 | **EquityLinkedGICPooling** | GIC pooling | Fair value tolerance, notional minimums |
| 6 | **Quant-Developers-Resources** | Quant reference | AI/ML, statistics, financial engineering |

### Layer 2: Signal Processing & Feature Engineering

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 7 | **Mav-Analysis** | Technical analysis | Maverick MCP, indicators, backtesting |
| 8 | **quant-trading** | Quant strategies | Bollinger, London Breakout, Dual Thrust |
| 9 | **stock-chain** | Chain analysis | Price-volume relationships, order flow |
| 10 | **CTA-code** | CTA strategies | Trend following, momentum, mean reversion |
| 11 | **TradeTheEvent** | Event-driven ML | BERT event detection, news trading |
| 12 | **wondertrader** | HFT execution | C++ quant framework, CTA/HFT strategies |

### Layer 3: ML/AI Models & Prediction

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 13 | **QLIB** | Quant ML framework | Microsoft QLIB, alpha factor mining |
| 14 | **Stock-techincal-prediction-model** | Price prediction | LSTM, CNN, ensemble forecasting |
| 15 | **Stock-prediction** | Technical prediction | Additional prediction models |
| 16 | **ML-Macro-Market** | Macro ML | Regime classification, business cycle |
| 17 | **AI-Newton** | Physics-informed ML | Neural ODEs, differential equations |

### Layer 4: Portfolio Construction & Risk Management

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 18 | **ai-hedgefund** | AI hedge fund | Multi-agent portfolio, risk management |
| 19 | **financial-distressed-repo** | Credit risk | Altman Z, Ohlson O, ML distress |
| 20 | **sophisticated-distress-analysis** | Advanced distress | Multi-factor deep health scoring |
| 21 | **FinancialDistressPrediction** | Distress ML | GBM bankruptcy prediction |

### Layer 5: Execution & Serving Infrastructure

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 22 | **Kserve** | Model serving | KServe ML inference, auto-scaling |
| 23 | **nividia-repo** | GPU deep learning | NVIDIA optimized training/inference |
| 24 | **Air-LLM** | LLM inference | Memory-efficient large model serving |
| 25 | **exchange-core** | Exchange engine | Ultra-low latency Java order matching |

### Layer 6: Agent Orchestration & Decision Intelligence

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 26 | **Ruflo-agents** | Agent orchestration | Multi-agent workflows, swarm intelligence |
| 27 | **MiroFish** | Social prediction | Agent-based social sentiment simulation |

### Layer 7: Global Monitoring & Regime Models

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 28 | **worldmonitor** | Global event monitoring | 30+ categories: market, economic, conflict, supply-chain, trade, news |
| 29 | **markov-model** | HMM regime detection | hmmlearn library: GaussianHMM, GMMHMM for market regime classification |

## Integration into Metadron Capital

These repositories serve as reference implementations. The production engines
in `engine/` are institutional-grade implementations that synthesize and elevate
the reference code:

- **SecurityAnalysisEngine** (`engine/signals/security_analysis_engine.py`)
  - References: Security Analysis 7th Edition (Graham, Dodd, Klarman)
  - Stage 3.1: Top-down macro (CAPE, ERP, speculative component) + bottom-up (Graham Number, NCAV, MoS, ROIC-WACC, DuPont, 8-test grading, 5-method IV). Feeds Tier-5 MLVoteEnsemble

- **DistressedAssetEngine** (`engine/signals/distressed_asset_engine.py`)
  - References: `FinancialDistressPrediction/`, `financial-distressed-repo/`, `sophisticated-distress-analysis/`, Security Analysis 7th Ed. (Mielle, Marks)
  - Elevation: Single GBM → 5-model ensemble + Graham-Mielle (fulcrum security, Ch.42 liquidation rates, Marks 8-factor credit)

- **CVREngine** (`engine/signals/cvr_engine.py`)
  - References: `ai-hedgefund/`, `Financial-Data/`
  - Elevation: Basic valuation → 5-model CVR (binary option, barrier, milestone tree, Monte Carlo, real options)

- **EventDrivenEngine** (`engine/signals/event_driven_engine.py`)
  - References: `TradeTheEvent/`, `CTA-code/`, `quant-trading/`
  - Elevation: BERT classification → 12-category quantitative models (Mitchell-Pulvino, SUE PEAD, Kelly)

- **SocialPredictionEngine** (`engine/signals/social_prediction_engine.py`)
  - References: `MiroFish/`, `Ruflo-agents/`
  - Elevation: Raw social data → agent-based sentiment simulation with topic modeling

- **MacroEngine** (`engine/signals/macro_engine.py`)
  - References: `FRB/`, `ML-Macro-Market/`, `QLIB/`
  - Elevation: FRED data → GMTF with SDR tension, rotation, velocity

- **StatArbEngine** (`engine/signals/stat_arb_engine.py`)
  - References: `quant-trading/`, `CTA-code/`, `stock-chain/`
  - Elevation: Simple pairs → Medallion mean reversion + cointegration + factor residuals

- **AlphaOptimizer** (`engine/ml/alpha_optimizer.py`)
  - References: `QLIB/`, `AI-Newton/`, `Stock-techincal-prediction-model/`
  - Elevation: Single model → Walk-forward ML alpha + mean-variance + credit quality scoring

- **UniverseClassifier** (`engine/ml/universe_classifier.py`)
  - References: `FinancialDistressPrediction/`, `sophisticated-distress-analysis/`
  - Elevation: Binary distress → XGBoost 4-model ensemble quality tiers (A-G) + credit ratings (AAA-D)

- **DeepLearningEngine** (`engine/ml/deep_learning_engine.py`)
  - References: `QLIB/`, `Stock-techincal-prediction-model/`, `nividia-repo/`
  - Elevation: Framework-specific → Pure-numpy PPO agent with 50-feature state vector

- **ML Model Bridges** (`engine/ml/bridges/`)
  - References: `QLIB/`, `Kserve/`, `nividia-repo/`, `Stock-techincal-prediction-model/`
  - Elevation: Siloed models → Unified bridge adapters producing SignalType for MLVoteEnsemble

- **InvestorPersonaManager** (`engine/agents/investor_personas.py`)
  - References: `ai-hedgefund/`, `Ruflo-agents/`
  - Elevation: Single agent → 12 investor personas + 8 core analysis agents

- **GICSSectorAgentManager** (`engine/agents/gics_sector_agents.py`)
  - References: `Mav-Analysis/`, `CTA-code/`
  - Elevation: Basic sector tracking → 11 sector agents with 8 scoring dimensions each

- **AgentMonitor** (`engine/agents/agent_monitor.py`)
  - References: `Ruflo-agents/`
  - Elevation: Basic scoring → 4-tier performance hierarchy with memory profiling

- **LiveDashboard** (`engine/monitoring/live_dashboard.py`)
  - References: `hedgefund-tracker/`, `open-bb/`
  - Elevation: Static reports → Rich-based real-time terminal dashboard (550+ symbols)

- **FastAPI Web App** (`app/backend/`)
  - References: `ai-hedgefund/`, `open-bb/`
  - Elevation: CLI-only → REST API + SSE streaming + SQLAlchemy persistence

## Multi-Language Ecosystem

The intelligence platform contains code across multiple languages, integrated
through the `plugins/` bridge layer:

| Language | Files | Key Repos |
|----------|-------|-----------|
| Python | ~12,000+ | All repos |
| C++ / CUDA | ~4,000+ | Kserve (TensorFlow), nividia-repo, wondertrader |
| Go | ~4,000+ | Kserve (KFServing controller) |
| Java | ~300+ | Kserve, exchange-core |
| TypeScript/JS | ~160+ | ai-hedgefund, open-bb |
| Rust | ~40+ | AI-Newton |
| Vue | ~15+ | MiroFish |

### Plugin Bridges (`plugins/`)

- `unified_bridge.py` — Central multi-language integration
- `cuda_plugin.py` — GPU compute bridge (nividia-repo, Kserve)
- `rust_plugin.py` — Rust bridge (AI-Newton)
- `go_plugin.py` — Go bridge (Kserve KFServing)
- `typescript_plugin.py` — TypeScript bridge (ai-hedgefund, open-bb)
