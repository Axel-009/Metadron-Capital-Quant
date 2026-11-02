# Metadron Fund — Frontend Build Instructions

> **UI Name:** Metadron Fund
> **Architecture:** Tabbed UI shell with each tab connected to a functional layer of the intelligence platform
> **Build Target:** Local development, integrated system

---

## EXISTING FRONTEND MODULES (Audit)

### Active Modules (Build-System Backed)

| # | Module | Path | Tech Stack | Purpose |
|---|--------|------|------------|---------|
| 1 | AI Hedgefund | `intelligence_platform/ai-hedgefund/app/frontend/` | React 18 + TypeScript + Vite 5 + Tailwind + React Flow + Radix UI (shadcn/ui) | Visual strategy builder, node-based flow editor, backtest dashboard, real-time execution monitoring. Most complete standalone frontend — becomes Tab 9. |
| 2 | Ruvocal Chat UI | `intelligence_platform/Ruflo-agents/ruflo/src/ruvocal/` | SvelteKit 2 + Svelte 5 + Vite 6 + Tailwind + MongoDB | Chat interface with voice input, markdown rendering, file upload for LLM agent interaction. Receives orders from Openclaw (CEO assistant). |
| 3 | OpenBB Plotly Components | `intelligence_platform/open-bb/frontend-components/plotly/` | React 18 + Vite 4 + Plotly.js + Tailwind + TypeScript | Shared Plotly financial charting component library. Reusable — shells into all tabs. Exports as single-file bundle via rollup. |
| 4 | OpenBB Table Components | `intelligence_platform/open-bb/frontend-components/tables/` | React 17 + Vite 4 + React Table 8 + React Virtual + Tailwind + TypeScript | Shared data table component with virtual scrolling. Reusable — shells into all tabs. Exports as single-file bundle via rollup. |

### Non-Build Frontends (No npm build pipeline)

| # | Module | Path | Tech Stack | Status |
|---|--------|------|------------|--------|
| 5 | MiroFish Frontend | `mirofish/frontend/` | Vue 3 + Vite + D3.js (documented target) | **Stub only** — to be deleted (see Pre-Build Task 3). |
| 6 | Claude Flow Browser Dashboard | `intelligence_platform/Ruflo-agents/v2/examples/browser-dashboard/` | Vanilla JS + WebSocket + Node.js server | Proof-of-concept swarm monitoring dashboard. |
| 7 | Stock Forecasting JS | `intelligence_platform/Stock-prediction/stock-forecasting-js/` | Vanilla HTML/CSS/JS (D3, Plotly, TensorFlow.js, Materialize CSS) | In-browser LSTM stock prediction. No build tool — needs Vite integration. Duplicate exists (see Pre-Build Task 2). |

### Technology Summary

- **Build tool:** Vite (all 4 primary modules)
- **Primary framework:** React (modules 1, 3, 4)
- **Secondary frameworks:** SvelteKit (module 2), Vue 3 (module 5 — stub, to be deleted)
- **Styling:** Tailwind CSS (all modules)
- **TypeScript:** Used in all build-backed modules

---

## PRE-BUILD TASKS

### 1. Build the Stock Prediction Model System

Initialize and build the stock forecasting JS module located at:
- `intelligence_platform/Stock-techincal-prediction-model/stock-forecasting-js/`

This module uses Vanilla JS + Materialize CSS + D3/Plotly/TensorFlow.js for in-browser LSTM stock price prediction. It needs a proper build pipeline (package.json, Vite) to integrate into the consolidated frontend.

### 2. Deduplicate Stock Prediction Modules

Two duplicate stock prediction directories exist:
- `intelligence_platform/Stock-techincal-prediction-model/stock-forecasting-js/`
- `intelligence_platform/Stock-prediction/stock-forecasting-js/`

**Action:** Merge these into a single module under one path. Remove all duplicate files. The consolidated module should contain no redundant code — one source of truth for stock forecasting JS.

### 3. Remove MiroFish Frontend Stub

The MiroFish frontend at `mirofish/frontend/` is an empty stub (no package.json, just an empty `src/assets/` directory). It will not be part of the Metadron Fund UI.

**Action:** Delete the `mirofish/frontend/` directory entirely. MiroFish agent-based market simulation data will be consumed via backend APIs through the integrated system, not through a standalone frontend.

### 4. Ruvocal Chat — Openclaw Integration

The Ruvocal Chat UI (`intelligence_platform/Ruflo-agents/ruflo/src/ruvocal/`) will receive its orders from **Openclaw**, the CEO assistant agent. Ruvocal is the chat interface through which Openclaw communicates directives, reviews, and instructions to the system. Configure Ruvocal to accept and display Openclaw command streams.

### 5. AI Hedgefund Frontend — Separate Tab (Tab 9)

The AI Hedgefund frontend (`intelligence_platform/ai-hedgefund/app/frontend/`) will be integrated as its own dedicated tab (Tab 9) within the Metadron Fund UI. It connects to the entire intelligence platform and all its systems — all engine layers (L1-L7), agents (L6), and monitoring systems. See Tab 9 specification below.

### 6. OpenBB Components — Shell Integration

The OpenBB Plotly and Tables components:
- `intelligence_platform/open-bb/frontend-components/plotly/`
- `intelligence_platform/open-bb/frontend-components/tables/`

These will be installed and shelled into the overall Metadron Fund frontend build. They provide the reusable charting (Plotly) and data table (React Table + Virtual Scrolling) components used across all tabs.

---

## METADRON FUND — UI TAB LAYOUT

The UI is structured as a tabbed interface. Each tab connects to a different functional layer of the integrated intelligence platform.

---

### Tab 1 — OpenBB (Platform Shell)

The entire OpenBB package in its entirety, serving as the foundational shell for the Metadron Fund UI. This is the base layer that all other tabs sit within.

**Source:** `intelligence_platform/open-bb/`

---

### Tab 2 — Market Wrap & News

| Component | Source | Description |
|-----------|--------|-------------|
| Market Direction | MarketWrap engine | General market direction narrative |
| Scenario Thesis | MacroEngine + MetadronCube | Daily base, up, and down scenarios — short, medium, and long term |
| Sector Heatmap | `engine/monitoring/heatmap_engine.py` | GICS sector heatmap visualization (HeatmapEngine) |
| CVR Dates & News | `engine/signals/cvr_engine.py` | CVR event dates, acquisition news, catalyst calendar |
| Fed Calendar | MacroEngine / FRED | Fed meeting dates, rate decisions, balance sheet updates |
| Incoming Events | EventDrivenEngine + earnings feeds | Earnings releases, dividend dates, ex-dates |

---

### Tab 3 — Live Dashboard

| Component | Source | Description |
|-----------|--------|-------------|
| **Data Source Indicator** | `openbb_data.py` + `alpaca_broker.py` | **Status badge showing active data source: `[LIVE] Alpaca` or `[EOD] OpenBB`. Must be visible at all times in the dashboard header. Clickable to toggle data source on command (switches provider in `openbb_data.py`).** |
| Live Execution P&L | ExecutionEngine + PaperBroker/AlpacaBroker | Real-time profit & loss display |
| TCA | L7 Unified Execution Surface | Transaction Cost Analysis |
| Slippage | L7 Execution | Slippage tracking per fill |
| Execution Efficiency | L7 Dashboard | Fill rate, latency, routing quality |
| Technical Analysis | `intelligence_platform/quant-trading/` | Chart patterns, indicators, similar to quant-trading module visualizations |

---

### Tab 4 — Asset Allocation

| Component | Source | Description |
|-----------|--------|-------------|
| Live NAV | PaperBroker / AlpacaBroker | Real-time net asset value display |
| Basket Formation | DecisionMatrix + AlphaOptimizer | Pre-execution basket construction view |
| Dynamic Movers Engine | UniverseEngine + signals | Real-time movers, momentum shifts |
| Allocation CSV | Portfolio export | Downloadable CSV of current allocation |
| VIX | OpenBB data / CBOE | Live VIX display |
| Major Indices | SPY, QQQ, IWM, DIA, VT, EFA, EEM | Live index pricing |
| GICS Filter | UniverseEngine GICS 4-tier | Filter portfolio by GICS sector, industry group, industry, sub-industry |
| Instrument Classification | Cross-asset universe | Filter by instrument type (equity, option, future, FI, FX) |

---

### Tab 5 — Risk & Portfolio Monitor

| Component | Source | Description |
|-----------|--------|-------------|
| Options Risk & Greeks | `engine/execution/options_engine.py` | Delta, Gamma, Theta, Vega, Rho — live |
| Portfolio Metrics | PortfolioAnalytics | Sharpe ratio, Sortino, max drawdown, Calmar — all dynamic |
| Filled / No Fills | ExecutionEngine | Order fill status, partial fills, rejects |
| Live Return | PaperBroker / AlpacaBroker | Real-time return calculation |
| Portfolio Risk | `engine/risk/monte_carlo_risk.py` | VaR, CVaR, stress VaR |
| Total Initial Margin | AlpacaBroker account | Margin utilization |
| Greeks Risk | OptionsEngine aggregate | Portfolio-level Greeks exposure |
| Liquidity Risk | ContagionEngine + universe data | Liquidity scoring, bid-ask spread analysis |

---

### Tab 6 — Machine Learning UI

| Component | Source | Description |
|-----------|--------|-------------|
| Stress Testing | `engine/ml/backtester.py` + MonteCarloRiskEngine | Scenario stress test results |
| Backtesting Results | QSTrader backtest bridge | Walk-forward, regime-specific, strategy comparison |
| Trend & Pattern Recognition | `engine/ml/pattern_recognition.py` + PatternDiscoveryEngine | Candlestick, chart patterns, symbolic regression discoveries |
| Statistical Anomalies | `engine/monitoring/anomaly_detector.py` | Z-score anomalies, distribution breaks |
| Relative Value Report | `engine/signals/stat_arb_engine.py` | RV pair spreads, z-scores, cointegration status for 26 pairs |

---

### Tab 7 — Technical Dashboard

| Component | Source | Description |
|-----------|--------|-------------|
| Active Engine Visualization | All engine modules | Visual status of each active engine (L1-L7) with health indicators |
| Memory Consumption | System monitoring | RAM, CPU, process memory per engine |
| VPS Health | System monitoring | Disk, network, uptime, load average |
| Learning Engine | `engine/monitoring/learning_loop.py` | GSD + Paul plugin status, gradient dynamics, pattern evolution |
| Learning Files Usage | ModelStore + learning data | Model versions, training data freshness, storage utilization |
| Latency | L7 Dashboard + heartbeat config | Per-engine latency, heartbeat cadence, API response times |
| Error Log (Live) | Structured logging stream | Real-time error message feed from all engines |
| Fallback Updates | ExecutionEngine | Any orders routed to PaperBroker due to live broker failures |

---

### Tab 8 — Monitoring & Reporting

All reports available in **PDF and CSV** export formats.

| Report | Source | Description |
|--------|--------|-------------|
| Platinum Report | `engine/monitoring/platinum_report.py` | 30-section executive macro state (9 parts) |
| Platinum Report V2 | `engine/monitoring/platinum_report_v2.py` | Enhanced platinum report generator |
| Portfolio Report | `engine/monitoring/portfolio_report.py` | Scenario engine + performance deep-dive (3 parts) |
| Portfolio Analytics | `engine/monitoring/portfolio_analytics.py` | Deep portfolio analytics + scenario engine |
| Daily Report | `engine/monitoring/daily_report.py` | Open/close reports + sector heatmap |
| Sector Tracker | `engine/monitoring/sector_tracker.py` | Sector performance + missed opportunities (>20% movers) — Daily |
| Anomaly Detector | `engine/monitoring/anomaly_detector.py` | Statistical anomaly scanner |
| Market Wrap | `engine/monitoring/market_wrap.py` | Narrative market summary |
| Memory Monitor | `engine/monitoring/memory_monitor.py` | Session tracking + EOD summary |

---

### Tab 9 — AI Hedgefund (Strategy Builder)

The AI Hedgefund frontend is a standalone React application integrated as its own tab. It provides the visual strategy building, node-based flow editing, backtesting dashboard, and real-time execution monitoring capabilities.

**Source:** `intelligence_platform/ai-hedgefund/app/frontend/`
**Tech:** React 18 + TypeScript + Vite 5 + Tailwind + React Flow + Radix UI (shadcn/ui)

| Component | Source | Description |
|-----------|--------|-------------|
| Visual Strategy Builder | React Flow | Node-based drag-and-drop strategy construction |
| Flow Editor | React Flow + custom nodes | Connect signal engines, ML models, and execution targets visually |
| Backtest Dashboard | QSTrader bridge + backtester | Run and visualize walk-forward backtests from the UI |
| Execution Monitor | ExecutionEngine + broker feeds | Real-time order flow, fill status, P&L per strategy |
| Agent Interaction | L6 Agent layer | View agent scores, hierarchy, persona recommendations |
| Signal Pipeline View | All L2 signal engines | Visual representation of the full signal pipeline (L1→L7) |
| ML Model Status | ModelStore + AlphaOptimizer | Model versions, training status, feature importance |

This tab connects to the full intelligence platform backend. All engine outputs, agent decisions, and execution results are routed here for visual strategy management.

---

## BUILD & INTEGRATION RECOMMENDATIONS

> These recommendations are designed so that AI assistants (Claude Code, Cursor, Openclaw agents) can build the entire UI from these instructions, or generate the scaffolding files for local Cursor development.

### Recommended Tech Stack (Unified)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | **React 18 + TypeScript** | 3 of 4 existing modules are React. AI Hedgefund (Tab 9) is React. Minimizes framework fragmentation. |
| Build Tool | **Vite 5** | Already used by all 4 frontend modules. Fast HMR, native ESM. |
| Styling | **Tailwind CSS 3** | Already used across all modules. Utility-first, consistent theming. |
| Component Library | **Radix UI + shadcn/ui** | Already in AI Hedgefund. Accessible, composable, unstyled primitives. |
| Charts | **Plotly.js** (via OpenBB component) | Already built. Financial charting, candlesticks, heatmaps. |
| Tables | **React Table 8 + React Virtual** (via OpenBB component) | Already built. Virtual scrolling for 1,044+ securities. |
| State Management | **Zustand** or **React Context** | Lightweight. No Redux overhead. Each tab manages its own state slice. |
| Data Fetching | **TanStack Query (React Query)** | Cache management, background refetching, SSE/WebSocket support for live data. |
| Routing | **React Router 6** | Tab-based routing: `/openbb`, `/market-wrap`, `/live`, `/allocation`, etc. |
| Real-time | **Server-Sent Events (SSE)** + **WebSocket** | SSE for live P&L/prices (one-way). WebSocket for Ruvocal chat (bidirectional). |
| PDF/CSV Export | **jsPDF + Papa Parse** | Tab 8 report generation client-side. |

### Monorepo Structure

```
metadron-fund/                              ← New frontend root
├── package.json                            ← Workspace root (pnpm workspaces)
├── pnpm-workspace.yaml
├── vite.config.ts                          ← Shared Vite config
├── tailwind.config.ts                      ← Shared Tailwind theme (Metadron branding)
├── tsconfig.json                           ← Shared TypeScript config
├── apps/
│   └── metadron-fund/                      ← Main application shell
│       ├── package.json
│       ├── src/
│       │   ├── main.tsx                    ← Entry point
│       │   ├── App.tsx                     ← Tab router + layout shell
│       │   ├── tabs/
│       │   │   ├── Tab1_OpenBB.tsx         ← OpenBB platform shell
│       │   │   ├── Tab2_MarketWrap.tsx     ← Market wrap & news
│       │   │   ├── Tab3_LiveDashboard.tsx  ← Live execution P&L
│       │   │   ├── Tab4_AssetAllocation.tsx ← NAV + basket + movers
│       │   │   ├── Tab5_RiskPortfolio.tsx  ← Risk & portfolio monitor
│       │   │   ├── Tab6_MachineLearning.tsx ← ML UI
│       │   │   ├── Tab7_TechDashboard.tsx  ← Technical dashboard
│       │   │   ├── Tab8_Reporting.tsx      ← Monitoring & reporting
│       │   │   └── Tab9_AIHedgefund.tsx    ← AI hedgefund strategy builder
│       │   ├── components/                 ← Shared UI components
│       │   │   ├── TabShell.tsx            ← Tab navigation bar
│       │   │   ├── LiveTicker.tsx          ← Price ticker strip
│       │   │   └── GICSFilter.tsx          ← GICS sector/industry filter
│       │   ├── hooks/
│       │   │   ├── useSSE.ts              ← SSE hook for live data streams
│       │   │   ├── useWebSocket.ts        ← WebSocket hook for Ruvocal
│       │   │   └── useEngineStatus.ts     ← Engine health polling
│       │   ├── api/
│       │   │   └── client.ts              ← FastAPI client (TanStack Query)
│       │   └── stores/
│       │       ├── portfolioStore.ts       ← NAV, positions, P&L state
│       │       ├── signalStore.ts          ← Live signal feed state
│       │       └── engineStore.ts          ← Engine health + latency state
│       └── index.html
├── packages/
│   ├── charts/                             ← OpenBB Plotly wrapper (from open-bb/frontend-components/plotly/)
│   │   ├── package.json
│   │   └── src/
│   ├── tables/                             ← OpenBB Tables wrapper (from open-bb/frontend-components/tables/)
│   │   ├── package.json
│   │   └── src/
│   ├── stock-prediction/                   ← Consolidated stock forecasting (deduplicated)
│   │   ├── package.json
│   │   └── src/
│   └── strategy-builder/                   ← AI Hedgefund React Flow (from ai-hedgefund/app/frontend/)
│       ├── package.json
│       └── src/
└── .env.local                              ← API endpoints (VITE_API_URL=http://localhost:8000)
```

### Backend API Contract

The frontend connects to the existing FastAPI backend (`app/backend/main.py`). The following SSE and REST endpoints are needed:

```
GET  /api/v1/portfolio/live          → SSE stream: NAV, positions, P&L (Tab 3, 4)
GET  /api/v1/signals/stream          → SSE stream: live signal feed (Tab 3, 6)
GET  /api/v1/engines/status          → JSON: all engine health + latency (Tab 7)
GET  /api/v1/market/wrap             → JSON: market wrap narrative (Tab 2)
GET  /api/v1/market/heatmap          → JSON: GICS sector heatmap data (Tab 2)
GET  /api/v1/risk/portfolio          → JSON: VaR, CVaR, Greeks, margin (Tab 5)
GET  /api/v1/ml/backtest/{id}        → JSON: backtest results (Tab 6)
GET  /api/v1/ml/anomalies            → JSON: statistical anomalies (Tab 6)
GET  /api/v1/reports/{type}          → PDF/CSV: report download (Tab 8)
GET  /api/v1/allocation/basket       → JSON: current basket + GICS filter (Tab 4)
GET  /api/v1/system/health           → JSON: memory, CPU, disk, VPS health (Tab 7)
GET  /api/v1/system/errors           → SSE stream: live error log (Tab 7)
WS   /ws/ruvocal                     → WebSocket: Openclaw ↔ Ruvocal chat (Ruvocal panel)
POST /api/v1/strategy/backtest       → JSON: trigger backtest from strategy builder (Tab 9)
GET  /api/v1/strategy/flow           → JSON: saved strategy flows (Tab 9)
POST /api/v1/strategy/flow           → JSON: save strategy flow (Tab 9)
```

### Build Order (Recommended Sequence)

| Phase | Task | Deliverable | Est. |
|-------|------|-------------|------|
| **Phase 1** | Scaffold monorepo + tab shell + routing | Empty tabs with navigation, Tailwind theme, Vite builds | 2-3 hrs |
| **Phase 2** | Migrate OpenBB Plotly + Tables into `packages/` | Reusable `@metadron/charts` and `@metadron/tables` packages | 1-2 hrs |
| **Phase 3** | Build Tab 7 (Technical Dashboard) | Engine status, memory, latency — validates API contract | 2-3 hrs |
| **Phase 4** | Build Tab 3 (Live Dashboard) + Tab 4 (Asset Allocation) | SSE live data, P&L display, GICS filter | 3-4 hrs |
| **Phase 5** | Build Tab 2 (Market Wrap) + Tab 5 (Risk) | Heatmap, CVR calendar, Greeks, portfolio risk | 3-4 hrs |
| **Phase 6** | Build Tab 6 (ML UI) + Tab 8 (Reporting) | Backtest viz, anomalies, PDF/CSV export | 3-4 hrs |
| **Phase 7** | Migrate AI Hedgefund into Tab 9 | Strategy builder, React Flow, backtest trigger | 2-3 hrs |
| **Phase 8** | Integrate Ruvocal as slide-out panel | WebSocket chat, Openclaw command stream | 2-3 hrs |
| **Phase 9** | Tab 1 (OpenBB shell) + stock prediction integration | Full OpenBB embed, LSTM prediction widget | 2-3 hrs |
| **Phase 10** | Polish, theming, responsive layout, error boundaries | Production-ready UI | 2-3 hrs |

### Key Files for Cursor Local Build

To enable an AI assistant or Cursor to build this locally, generate these files first:

```
1. metadron-fund/package.json              ← pnpm workspace root with scripts
2. metadron-fund/pnpm-workspace.yaml       ← workspace: ["apps/*", "packages/*"]
3. metadron-fund/vite.config.ts            ← shared config with proxy to FastAPI :8000
4. metadron-fund/tailwind.config.ts        ← Metadron brand colors + dark mode
5. metadron-fund/tsconfig.json             ← paths aliases (@metadron/charts, etc.)
6. metadron-fund/apps/metadron-fund/src/App.tsx  ← Tab router shell
7. metadron-fund/apps/metadron-fund/src/main.tsx ← Entry point
8. metadron-fund/apps/metadron-fund/src/tabs/    ← One file per tab (9 files)
9. metadron-fund/apps/metadron-fund/src/api/client.ts ← API client with TanStack Query
10. metadron-fund/.env.local               ← VITE_API_URL + VITE_WS_URL
```

These 10 scaffolding files are sufficient for any AI assistant to begin building each tab incrementally. Each tab is self-contained — they can be built and tested independently against the FastAPI backend.
