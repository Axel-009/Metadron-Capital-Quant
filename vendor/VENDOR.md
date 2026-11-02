# Vendored repositories

Full source copies (not submodules), so this repo builds and runs on its own. Each keeps its
upstream LICENSE. Update with `scripts/update_vendor.sh` (re-clones, re-applies the exclusions,
rewrites the pins below).

| Repo | Upstream | Pinned commit | Commit date | License | Excluded |
|------|----------|---------------|-------------|---------|----------|
| Kronos | https://github.com/shiyu-coder/Kronos | 67b630e67f6a18c9e9be918d9b4337c960db1e9a | 2026-04-13 | MIT | model weights (download from Hugging Face `NeoQuasar/Kronos-*`) |
| Lean | https://github.com/QuantConnect/Lean | 1383825685849e32e0074f34c3be85b627e0f50f | 2026-10-30 | Apache-2.0 | `Data/` (227 MB sample data — QuantConnect cloud / `lean data download` supply data) |
| OpenJarvis | https://github.com/open-jarvis/OpenJarvis | 570d75425d50592d931bac7a17ce6b2621561c06 | 2026-10-31 | Apache-2.0 | `desktop/src-tauri/binaries/` (prebuilt desktop binaries, >50 MB) |

Reference only, NOT vendored: freqtrade (GPL-3.0 — copying it would put this repo under GPL; its
design ideas are documented in CLAUDE.md instead).

`intelligence_platform/` holds the in-house integration copies of the research repos the engine
loads at runtime (MiroFish, QLIB, ai-hedgefund, quant-trading, distress models, qstrader, plugins,
Air-LLM, …). For Kserve, nividia-repo and AI-Newton only the integration module the engine loads
is kept (their full trees are 140–530 MB).
