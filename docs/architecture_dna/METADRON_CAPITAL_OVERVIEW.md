# METADRON CAPITAL

## Systematic Multi-Asset Investment Platform

---

## Executive Summary

Metadron Capital is a fully automated quantitative investment platform that
combines macro regime analysis, fundamental security selection, and
machine learning intelligence to generate alpha across equities, options,
and futures. The system continuously scans approximately 1,600 securities,
processes signals through multiple independent analytical tracks, and
executes via institutional-grade algorithmic order routing.

The platform is designed for autonomous operation during market hours with
continuous learning feedback that improves signal accuracy and execution
quality over time.

---

## Investment Philosophy

- **Beta managed within a 7–12% return corridor** aligned with S&P 500
  historical earnings yield
- **Alpha extracted through top-down macro regime classification**
  flowing into bottom-up security selection
- **Fundamentals-first approach**: 40% of the decision weight is
  allocated to fundamental quality (ROIC, FCF, Graham-Dodd valuation,
  credit quality)
- **Risk-adjusted position sizing** via Kelly criterion with
  volatility normalisation
- **Options allocated only on detected mispricing** — fair value
  computed via Black-Scholes and Monte Carlo simulation, requiring
  a minimum edge before any position is taken

---

## Universe Coverage

| Asset Class | Scope | Purpose |
|-------------|-------|---------|
| US Equities | S&P 500 + S&P 400 + S&P 600 (~1,500) | Core alpha generation |
| UK Equities | FTSE 100 representatives | International diversification |
| Options | Selected securities (edge-gated) | Convexity + mispricing capture |
| Futures | ES, NQ, YM, RTY, VX, ZN, ZB | Beta corridor management |
| Fixed Income | G10 + India sovereign, US IG/HY corporate | Macro signal + hedging |
| Commodities | GLD, SLV, USO, UNG, DBA, DBC + 3 more | Cyclical pattern signals |
| Currencies | G10 + INR + JPY | Macro regime context |
| Econometrics | 40+ FRED series (GDP, CPI, M2, SOFR) | Regime classification |

**Total coverage**: ~1,600 securities scanned continuously during market hours.

---

## Signal Generation Architecture

The platform operates two independent parallel analytical tracks that
converge before any investment decision is made:

### Track A — Macro-Driven Fundamental Analysis

A sequential pipeline that begins with Federal Reserve balance sheet
analysis, flows through macro regime classification, and fans out into
specialised signal engines:

- **Fed Liquidity Analysis**: SOFR, reserves, TGA, ON-RRP, money
  velocity tracking
- **Macro Regime Engine**: Global Monetary Tension Framework with
  4 non-linear gamma multipliers, sector ranking, yield curve analysis,
  credit pulse monitoring
- **10-Layer Intelligence Tensor**: Combines liquidity state, risk
  state, and capital flow into a unified regime signal
  (TRENDING / RANGE / STRESS / CRASH)
- **Parallel Signal Engines**: Graham-Dodd-Klarman security analysis,
  cross-asset contagion modelling (21-node graph), statistical
  arbitrage (mean-reversion + cointegration), distressed asset
  screening (5-model ensemble), fixed income analysis, and
  pattern discovery (symbolic regression)

### Track B — News-Driven Event Intelligence

An independent pipeline that processes real-time news from 10,000+
sources and runs agent-based market simulation on flagged securities:

- **Real-Time News Processing**: WebSocket feed with sentiment
  scoring and urgency categorisation
- **Agent-Based Market Simulation**: Kyle Lambda and Heterogeneous
  Agent Model simulations per news-flagged ticker
- **Combined Score**: 40% news sentiment + 60% agent simulation
  direction, feeding into event-driven and contingent value rights
  analysis

Both tracks run independently with zero cross-dependencies and converge
at the intelligence layer.

---

## Intelligence & Decision Layer

### Machine Learning Ensemble

A 10-tier weighted voting ensemble where each tier contributes an
independent directional vote:

| Tier | Model | Focus |
|------|-------|-------|
| 1 | Neural Network | Price direction prediction |
| 2 | Momentum / Mean-Reversion | Technical momentum signals |
| 3 | Volatility Regime | Vol compression / expansion |
| 4 | Monte Carlo | Probabilistic risk assessment |
| 5 | Fundamental Quality | Graham-Dodd investment grading |
| 6 | News + Agent Simulation | Real-time sentiment + microstructure |
| 7 | Distressed Asset | Credit distress + recovery detection |
| 8 | Event-Driven | M&A arbitrage, PEAD, catalysts |
| 9 | Contingent Value Rights | 5-model CVR valuation |
| 10 | Credit Quality | Credit scoring (AAA–D) |

### Alpha Scoring Pipeline

All signals from both tracks converge into a dual alpha scoring engine:

- **Standard Pipeline**: Gradient boosted models + linear regression +
  CAPM alpha extraction with 22 engineered features
- **Enhanced Pipeline**: Walk-forward validation + 50+ factor library +
  sector-level mean-variance optimisation

Output: a scored, deduplicated, allocation-ready slate of opportunities.

### 4-Gate Quality Filter

Every potential trade must pass a cross-asset quality filter before
execution:

| Gate | Weight | Function |
|------|--------|----------|
| Fundamentals | 40% | Quality, ROIC, FCF, credit, earnings |
| Flow & Headlines | 20% | ETF flow, news sentiment, sector rotation |
| Macro Regime | 20% | Direction alignment, VaR headroom |
| Momentum | 20% | RSI, MACD, breakout confirmation |

Minimum composite score of 0.55 required. The Fundamentals gate must
pass independently — no trade proceeds without fundamental conviction
regardless of momentum or sentiment.

---

## Portfolio Construction

### Allocation Structure

| Sleeve | Target | Description |
|--------|--------|-------------|
| IG Equities | 40% | Investment grade, all market caps |
| HY Equities | 10% | BB-B rated, leveraged opportunities |
| Distressed Equity | 10% | Fallen angels, recovery plays |
| Cashflow ETFs | 15% | Monthly distribution vehicles (DRIP) |
| Fixed Income / Macro | 5% | FI signals + macro relative value |
| Event-Driven / CVR | 10% | M&A arb, PEAD, contingent value |
| Options (notional) | 25% | Edge-gated only (≥200bps mispricing) |
| Futures (notional) | 15% | Beta corridor hedging (ES/NQ/VX) |
| Margin | 8% | Initial margin for derivatives overlay |
| Cash Reserve | 2% | Dry powder (hard floor, never breached) |

### Options Discipline

Options are treated as a precision instrument, not a speculative tool:

1. **Black-Scholes** theoretical pricing establishes fair value
2. **Monte Carlo simulation** (10,000 paths) estimates win probability
3. **Fair value vs market price** must show ≥200 basis points of edge
4. **Kelly criterion** sizes the position based on detected mispricing
5. **Minimum 5 contracts** — no token positions
6. **No edge detected → no allocation** regardless of available budget

### Risk Management

- **Kill Switch**: 5% portfolio drawdown triggers automatic halt
  (requires manual operator reset)
- **10-Gate Pre-Trade Risk Check**: Position size, sector concentration,
  daily loss, gross/net leverage, trade throttle, drawdown, cash
  sufficiency, options delta, futures notional — all must pass
- **Beta Corridor**: Portfolio beta managed within 7–12% return
  corridor with volatility normalisation and Kalman-filtered smoothing
- **Per-Position Drawdown**: 20% individual position loss triggers
  automatic liquidation

---

## Execution

### Algorithmic Order Routing

All orders route through a unified execution surface with native
institutional-grade algorithms:

| Order Size | Algorithm | Method |
|-----------|-----------|--------|
| > $50,000 notional | TWAP | Time-weighted server-side splitting |
| Standard | VWAP | Volume-weighted market participation |
| Medium urgency | Adaptive | Broker selects optimal strategy |
| High urgency | Market | Immediate fill |

### Execution Quality

- **Micro-price estimation** adjusts limit prices based on order flow
  imbalance before submission
- **Transaction cost analysis** decomposes every fill into spread,
  market impact, timing, and commission components
- **Execution learning loop** continuously optimises routing strategy,
  slice count, and timing per context bucket (ticker × product ×
  signal × regime × time-of-day × volatility × order size)
- **Trade log** records every generated order for reconciliation —
  full audit trail of what the system intended vs what executed

---

## Continuous Learning

The platform operates a closed-loop feedback system with 7 independent
learning channels:

| Channel | What It Learns | What It Adjusts |
|---------|---------------|-----------------|
| Signal Accuracy | Which signals were profitable | ML ensemble tier weights |
| Execution Quality | Slippage patterns | Routing strategy selection |
| Regime Feedback | Regime prediction accuracy | HMM transition priors |
| Alpha Decay | How quickly alpha erodes | Position holding period |
| Risk Calibration | Risk event frequency | Gate thresholds |
| Agent Performance | Individual agent accuracy | Agent promotion / demotion |
| Cross-Asset | Sector allocation effectiveness | Macro sector weights |

Weight changes are damped at ±5% per cycle with oscillation detection
to prevent the system from chasing noise.

### Agent Hierarchy

Autonomous sector-specialist agents are ranked and promoted based on
sustained performance:

| Rank | Requirements | Autonomy |
|------|-------------|----------|
| Director | Sharpe > 2.5, accuracy > 85% | Full autonomy |
| General | Sharpe > 2.0, accuracy > 80% | High autonomy |
| Captain | Sharpe > 1.5, accuracy > 55% | Standard |
| Lieutenant | Sharpe > 1.0, accuracy > 50% | Restricted sizing |
| Recruit | Below thresholds | Under review |

---

## Monitoring & Observability

- **17 real-time metrics** exported for dashboard monitoring (NAV,
  leverage, risk level, kill switch status, slippage, fill latency,
  order counts by algo type)
- **Anomaly detection** across 8 dimensions (z-score, volume,
  correlation, VIX, credit, breadth)
- **Hourly CSV snapshots** of portfolio state for audit trail
- **Reconciliation engine** compares generated orders vs broker fills
  to detect any execution drift

---

## Technology

| Component | Implementation |
|-----------|---------------|
| Core Engine | Python (pure-numpy fallbacks, no single-framework dependency) |
| ML Models | XGBoost, scikit-learn, HMM (hmmlearn), pure-numpy PPO |
| Data Source | Interactive Brokers (real-time) + OpenBB (historical, 34+ providers) |
| Execution | Interactive Brokers (TWS/Gateway via ib_insync) |
| Options Pricing | Black-Scholes + Monte Carlo (10K path simulation) |
| Monitoring | Prometheus + Grafana |
| Process Management | PM2 (24/7 continuous operation) |
| Deployment | Hetzner dedicated server + Cloudflare SSL |

The system is designed with graceful degradation — every external
dependency is wrapped in try/except. If any component fails, the
platform continues operating with reduced capability rather than
halting entirely.

---

## Key Differentiators

1. **Fundamentals-first, not momentum-driven**: 40% gate weight on
   fundamental quality prevents the system from chasing price action
   without underlying value

2. **Edge-gated options**: Unlike systems that deploy options budget
   mechanically, positions are only taken when quantitative mispricing
   is detected and sized by Kelly criterion

3. **Independent parallel tracks**: News-driven and macro-driven
   analysis run independently, preventing one track's noise from
   contaminating the other

4. **Institutional execution**: Native TWAP/VWAP algorithms with
   micro-price estimation and continuous execution learning

5. **Closed-loop learning**: Every execution outcome feeds back into
   the signal weights, ensuring the system adapts to changing market
   conditions without manual recalibration

6. **Full auditability**: Every signal, decision, order, and fill is
   logged with timestamps. The reconciliation engine provides
   complete transparency into what the system intended vs what executed.

---

*Metadron Capital — Systematic Alpha Through Intelligent Automation*
