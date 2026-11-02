# METADRON CAPITAL — AI SESSION INSTRUCTIONS

> **Read this file first.** It explains how this monorepo is structured
> and how every change must flow through the system. These rules apply to
> all AI agents, all sessions, and all contributors.

---

## 1. Two-Zone Architecture

The repository has **two zones** that mirror each other but serve different purposes:

```
Metadron-Capital/
├── repos/                      ← ZONE 1: Systematic Archive (independent repos)
│   ├── layer1_data/
│   ├── layer2_signals/
│   ├── layer3_ml/
│   ├── layer4_portfolio/
│   ├── layer5_infra/
│   └── layer6_agents/
│
├── intelligence_platform/      ← ZONE 2: Unified Investment Platform (meshed system)
│   └── <all repos merged into one orchestrated flow>
│
├── Architecture_DNA/           ← System documentation (this folder)
│   ├── ARCHITECTURE_DNA.md     ← Full architecture specification
│   └── AI_SESSION_INSTRUCTIONS.md  ← This file
│
├── platform_orchestrator.py    ← Master orchestrator (ties everything together)
└── openbb_universe.py          ← Root-level universe engine
```

### Zone 1: `repos/` — The Systematic Archive

Each sub-directory under `repos/layerN_*/` is a **complete copy** of an
independent GitHub repository. These repos **maintain their independence** —
they keep their own directory structures, their own READMEs, their own tests.

**Purpose:** Preserve the original repo context so you can always refer back
to the upstream source. Think of it as a versioned archive of every component.

### Zone 2: `intelligence_platform/` — The Unified System

This is where all repos are **meshed into one investment system**. Files here
are wired together through shared imports, the platform orchestrator, and
OpenBB data bridges. All development work and new features happen here first.

---

## 2. Change Flow Rules

```
                    ┌─────────────────────────┐
                    │  intelligence_platform/  │ ← ALL changes start here
                    │   (unified system)       │
                    └───────────┬──────────────┘
                                │
                     changes propagate down
                                │
                    ┌───────────▼──────────────┐
                    │       repos/layerN/      │ ← Archive updated to match
                    │   (independent repos)    │
                    └──────────────────────────┘
```

1. **All changes are applied to `intelligence_platform/`** — this is the
   single source of truth for the running investment system.

2. **Archive repos (`repos/`) are updated to reflect those changes** — each
   independent repo in the archive gets the same updates so it stays in sync.

3. **Architecture DNA is updated** whenever structural changes are made to the
   investment platform (new engines, new signal flows, new layers, etc.).

---

## 3. Adding a New Repository

When a new repo is introduced to the system, follow this exact procedure:

### Step 1: Determine the Layer

| Layer | Purpose | Examples |
|-------|---------|----------|
| `layer1_data` | Data ingestion, universe definitions, market feeds | Financial-Data, open-bb, hedgefund-tracker |
| `layer2_signals` | Signal generation, technical analysis, event detection | quant-trading, CTA-code, stock-chain, Mav-Analysis, TradeTheEvent |
| `layer3_ml` | Machine learning models, predictions, alpha generation | QLIB, ML-Macro-Market, AI-Newton, Stock-prediction |
| `layer4_portfolio` | Portfolio construction, risk management, distress analysis | ai-hedgefund, sophisticated-distress-analysis, FinancialDistressPrediction |
| `layer5_infra` | Infrastructure, serving, GPU acceleration, execution | Kserve, nividia-repo, Air-LLM, exchange-core |
| `layer6_agents` | Autonomous agents, orchestration, multi-agent systems | Ruflo-agents, MiroFish |

### Step 2: Copy into Both Zones

```bash
# Archive copy (maintains independence)
cp -r /path/to/new-repo repos/layerN_purpose/new-repo/

# Platform copy (will be integrated)
cp -r /path/to/new-repo intelligence_platform/new-repo/
```

### Step 3: Add GitHub Source Reference

In the repo's main entry point or a dedicated file, add a header comment:

```python
# ============================================================
# SOURCE: https://github.com/Axel-009/<repo-name>
# LAYER:  layerN_purpose
# ROLE:   <brief description of what this repo does in the platform>
# ============================================================
```

### Step 4: Create Integration Files

Create the appropriate bridge files in `intelligence_platform/`:
- `openbb_universe.py` or `openbb_data.py` — for data/signal repos
- `investment_platform_integration.py` — for infra/agent repos
- Custom engine files as needed

### Step 5: Update Architecture DNA

Add the new repo to `Architecture_DNA/ARCHITECTURE_DNA.md` under its layer,
including its role in the signal pipeline and any new connections it creates.

---

## 4. Current Repo Registry

| Repo | Layer | GitHub Source | Role |
|------|-------|--------------|------|
| Financial-Data | layer1_data | `github.com/Axel-009/Financial-Data` | Core financial data collection and OpenBB universe |
| hedgefund-tracker | layer1_data | `github.com/Axel-009/hedgefund-tracker` | Hedge fund 13F tracking, daily PnL, thesis generation |
| open-bb | layer1_data | `github.com/Axel-009/open-bb` | OpenBB Platform — primary data provider SDK |
| quant-trading | layer2_signals | `github.com/Axel-009/quant-trading` | Quantitative trading signals, universe scanning, arbitrage |
| CTA-code | layer2_signals | `github.com/Axel-009/CTA-code` | CTA/managed futures strategies, Metadron Cube rotation |
| stock-chain | layer2_signals | `github.com/Axel-009/stock-chain` | Blockchain-based stock analysis, multi-asset class signals |
| Mav-Analysis | layer2_signals | `github.com/Axel-009/Mav-Analysis` | Multi-asset volatility analysis across equities/FX/crypto |
| TradeTheEvent | layer2_signals | `github.com/Zhihan1996/TradeTheEvent` | Event-driven trading from news/earnings using NLP |
| QLIB | layer3_ml | `github.com/Axel-009/QLIB` | Microsoft Qlib ML framework for quant research |
| ML-Macro-Market | layer3_ml | `github.com/Axel-009/ML-Macro-Market` | Macro-level ML predictions, regime detection |
| AI-Newton | layer3_ml | `github.com/Axel-009/AI-Newton` | AI-driven fundamental analysis and factor models |
| Stock-prediction | layer3_ml | `github.com/Axel-009/Stock-techincal-prediction-model` | Technical prediction models, multi-asset prediction |
| ai-hedgefund | layer4_portfolio | `github.com/Axel-009/ai-hedgefund` | Core hedge fund engine — portfolio, execution, HFT, reporting |
| sophisticated-distress-analysis | layer4_portfolio | `github.com/Axel-009/sophisticated-distress-analysis` | Advanced distress scanning and credit analysis |
| financial-distressed-repo | layer4_portfolio | `github.com/Axel-009/financial-distressed-repo` | Financial distress prediction and credit risk engines |
| FinancialDistressPrediction | layer4_portfolio | `github.com/keigito/FinancialDistressPrediction` | Academic distress prediction models |
| Kserve | layer5_infra | `github.com/Axel-009/Kserve` | Model serving infrastructure (KServe/KNative) |
| nividia-repo | layer5_infra | `github.com/Axel-009/nividia-repo` | NVIDIA GPU-accelerated computing, CUDA plugins |
| Air-LLM | layer5_infra | `github.com/Axel-009/Air-LLM` | Lightweight LLM inference for research agents |
| Ruflo-agents | layer6_agents | `github.com/Axel-009/Ruflo-agents` | Multi-agent orchestration, financial risk plugins |
| MiroFish | layer6_agents | `github.com/666ghj/MiroFish` | Multi-agent research and investment coordination |

---

## 5. Key Files to Know

| File | Purpose |
|------|---------|
| `platform_orchestrator.py` | Master orchestrator — runs the full daily pipeline |
| `openbb_universe.py` | Root universe engine — defines the 150+ security universe |
| `Architecture_DNA/ARCHITECTURE_DNA.md` | Full architecture spec, signal pipeline, layer definitions |
| `.gitignore` | Has negation overrides for ML model artifacts (.pkl) and CUDA binaries (.so/.o) |

---

## 6. Rules for AI Agents

1. **Always read this file first** when starting a new session on Metadron-Capital.
2. **Never modify only one zone** — changes to `intelligence_platform/` must be
   reflected in `repos/` and vice versa.
3. **Always include GitHub source references** when adding new files or repos.
4. **Update Architecture DNA** when making structural changes.
5. **All development targets `intelligence_platform/`** — the archive follows.
6. **Respect the layer hierarchy** — don't put a data repo in layer4 or a
   portfolio repo in layer1.
7. **The `.gitignore` has intentional negation overrides** — don't remove them.
   ML model files (.pkl) and CUDA binaries (.so/.o) from specific sub-repos
   are deliberately tracked.
8. **Branch convention:** Development happens on `claude/import-session-*` branches.

---

## 7. Quick Reference Commands

```bash
# See what layer a repo is in
ls repos/layer*/

# Check all tracked files for a specific repo
git ls-files repos/layer3_ml/QLIB/ | wc -l

# Verify no files are being silently ignored
git status --short

# See the full signal pipeline
cat Architecture_DNA/ARCHITECTURE_DNA.md
```

---

*Last updated: 2026-03-16*
*This file lives in: `Architecture_DNA/AI_SESSION_INSTRUCTIONS.md`*
