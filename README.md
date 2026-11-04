# Metadron Capital — self-contained build

The Metadron intelligence engine extracted into one standalone, runnable repository:
QuantConnect LEAN algorithm + IBKR execution, the full engine, research integrations, vendored
Kronos / LEAN / OpenJarvis, Jarvis, docs and tests. Nothing here depends on the old monorepo.

**Start with `CLAUDE.md`** — layout, commands, design rules, and the canonical maths
(smart-beta corridor v2, risk budget, allocation, target architecture).

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest tests -q          # 285 tests
```

## What is where

| Folder | Contents |
|--------|----------|
| `algorithm/` | QuantConnect algorithm: `main.py` (intraday heartbeat + wiring), corridor, risk budget, HMM regime, forward Rm, universe + fundamentals, decision matrix, alpha models, options, protection, GICS sectors, theme rotation, trader framework |
| `agents/` | Propose-only agents run by GitHub Actions: macro, events, Kronos, risk officer, vault journalist, QuantConnect deploy (`roster.yaml`) |
| `vault/` | Obsidian control room: Dashboard + generated Daily / Positions / Risk / Universe / Market notes |
| `engine/` | 168 engine modules: macro (GMTF, HMM), MetadronCube, security analysis, distress, events, CVR, patterns, ML, decision matrix, options, protection, tail hedges, allocation, risk, API, agents, monitoring |
| `core/` | Original platform core: cyclical vs secular decomposition, money velocity |
| `intelligence_platform/` | Research integrations the engine loads (MiroFish, QLIB, distress models, ai-hedgefund, quant-trading, qstrader, plugins, Air-LLM, …) |
| `vendor/` | Kronos, LEAN, OpenJarvis — full source, pinned in `vendor/VENDOR.md`, update with `scripts/update_vendor.sh` |
| `jarvis/` | Jarvis setup guide (`jarvis/README.md`) + dashboard page |
| `docs/` | `architecture_dna/` (reference design), `charts/` (corridor derivation charts + scripts) |
| `governance/`, `config/` | Credit classification, repo / heartbeat config |
| `tests/` | Engine suite + algorithm tests |

## Build roadmap

| Step | Scope | Status |
|------|-------|--------|
| 1–2b | Corridor v2, risk budget, MES delivery, no double sizing, theme rotation, trader framework, HMM regime, forward Rm, beta burst, macro agent | done |
| 3 | S&P 1500 universe, Morningstar adapter, Graham-Dodd analysis, capital structure, Fundamentals gate, shortlist | done |
| 3+ | Rotation across the 11 GICS sectors into single stocks + options (research, regime, bullish/bearish budgets); repo allocation buckets; intraday liquidity heartbeat | done |
| 4 | Alpha: walk-forward ML, quality ensemble, chart patterns, Kronos (CPU), model scorecard | done |
| 5 | Decision matrix: 4 gates, spread, P(profit), full Kelly, sleeves, alpha VaR, MES overlay = L − sleeve beta | done |
| 6 | Options engine (BS, IV, selection, 1-contract exception), protection, event tail hedges, expiries | done |
| 7 | Agents (events, risk officer, journalist, forecaster), reconciliation, Obsidian vault, GitHub Actions hosting | done |
| next | First real QuantConnect backtest + paper live (owner setup, `SETUP.md`); Kronos fine-tuning on GPU (later) | owner |

## Run it

Follow **`SETUP.md`** (one afternoon): GitHub repo + secrets → the deploy workflow pushes `algorithm/` to
QuantConnect and backtests it → one click *Deploy Live* on the IBKR paper account → Obsidian opens `vault/`.

## Licenses

In-house code: proprietary (Metadron Capital). Vendored: Kronos (MIT), LEAN (Apache-2.0),
OpenJarvis (Apache-2.0) — each keeps its own LICENSE. freqtrade is a design reference only (GPL-3.0,
not included).
