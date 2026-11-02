# Metadron Capital — Infrastructure & Hosting Plan

> **CONFIRMED:** Hetzner GEX44 purchased (April 2026).
> Production config: Qwen2.5-7B FP16 + Qwen2.5-Math-7B Q4 via llama.cpp.
> See [GEX44 Setup Guide](#gex44-setup-guide) for deployment steps.

## Codebase Profile (as of 2026-04)

| Metric | Value |
|--------|-------|
| Total repo size | 3.6 GB |
| Total files | 42,253 |
| Python files | 8,963 |
| intelligence_platform (reference repos) | 2.4 GB |
| Engine (core trading logic) | 3.1 MB |
| ML dependencies | PyTorch, XGBoost, hmmlearn, PySR, FinRL (bridges) |
| Web stack | FastAPI (backend) + Express.js (proxy) + React (client) + Vue 3 (MiroFish) |
| Live loop | 3-min heartbeat intraday, 30-min after-hours, 1-hr overnight |

---

## Actual Runtime Profile

| Layer | Cadence | Latency | Nature |
|-------|---------|---------|--------|
| L1 Data Ingestion | 3 min | 100-500ms | I/O bound (OpenBB API) |
| L2 Signal Generation | 3 min | 500ms-2s | CPU (numerical) |
| L3 Intelligence/ML | 5 min | 2-5s | CPU (XGBoost, SLSQP, HMM) |
| L4 Decision | On signal delta | 100-200ms | CPU (vote ensemble) |
| L5 Execution | On approval | 50-200ms | I/O (Alpaca API) |
| L6 Learning | Every heartbeat | 100-500ms | CPU (gradient, feedback) |
| L7 Monitoring | 5 min | 100-200ms | CPU + I/O |
| FastAPI SSE | 5-10s push | <100ms | I/O |
| Express Proxy | On-demand | <50ms overhead | I/O |
| React Client | 5-30s polling | <100ms | Browser |

**Key finding: This is a 3-minute batch system, not sub-second HFT.**
No tick-by-tick streams. No WebSocket data feeds. WonderTrader/exchange-core
concepts are present but operate on OHLCV bars, not live ticks.

---

## Storage Requirements

| Component | Estimated Size |
|-----------|---------------|
| Codebase + intelligence_platform | 5 GB |
| Python virtualenv + ML deps (PyTorch, OpenBB, XGBoost, PySR) | 15-20 GB |
| Node.js (Express proxy, React client, MiroFish Vue) | 2 GB |
| Market data cache (1,044+ securities, 1-min bars, FRED, options) | 50-100 GB |
| Mimo v2 model weights (local LLM, if self-hosted) | 20-80 GB |
| ML model artifacts (HMM, XGBoost ensemble, PPO, walk-forward) | 10-20 GB |
| QSTrader backtesting output + Monte Carlo | 20-30 GB |
| Database (positions, signals, learning loop history) | 10-20 GB |
| OS + system overhead | 10 GB |
| Growth headroom | 50 GB |
| **Total recommended** | **500 GB - 1 TB NVMe SSD** |

Note: Storage jumps to 500GB+ if hosting Mimo v2 locally. If Mimo v2 is
API-based (like Anthropic), 250-500 GB is sufficient.

---

## Self-Hosted Model Strategy — GPU Infrastructure

### LLM Touchpoints in Metadron (35+ locations)

The codebase has **35+ locations** where LLM inference is used or could be used.
These break down into distinct task categories:

| Task Category | Locations | Current Provider | Call Frequency |
|--------------|-----------|-----------------|----------------|
| **Investor Persona Reasoning** (12 agents) | `ai-hedgefund/src/agents/{persona}.py` | Claude/GPT-4 | Per-ticker per heartbeat |
| **News Sentiment Classification** | `ai-hedgefund/src/agents/news_sentiment.py` | Claude/GPT-4 | 5 articles/ticker |
| **MiroFish Report Generation** | `mirofish/backend/app/services/report_agent.py` | OpenAI-compatible | On-demand (multi-turn) |
| **MiroFish Profile/Ontology** | `mirofish/backend/app/services/oasis_profile_generator.py` | OpenAI-compatible | Per-simulation |
| **Investor Persona Orchestration** | `engine/agents/investor_personas.py` | Configurable | Per-decision cycle |
| **News Engine** | `engine/signals/news_engine.py` | newsfilter.io (no LLM) | Continuous |
| **Research/Sector Bots** | `engine/agents/research_bots.py`, `sector_bots.py` | Quantitative (no LLM) | Per-heartbeat |
| **Signal Engines** | `engine/signals/*.py` | Quantitative (no LLM) | Per-heartbeat |

### Task-to-Model Routing — The Multi-Model Strategy

Instead of one model for everything, route tasks to specialized models:

```
TASK ROUTING ARCHITECTURE:

Financial Sentiment / Classification + Math (HIGH VOLUME, LOW-MED COMPLEXITY)
  └── Qwen2.5-7B-Instruct + Qwen2.5-Math-7B-Instruct (FP16 or Q4)
      ├── News headline sentiment → BULLISH/BEARISH/NEUTRAL
      ├── SEC filing classification → MATERIAL/ROUTINE
      ├── Earnings surprise direction
      ├── Numerical calculations, factor math
      └── ~50-100 calls per heartbeat, short input, classification output
      └── Replaces FinMA-7B (which is outdated LLaMA-1 based)

Quantitative Analysis / Code / Structured Data (MEDIUM VOLUME, HIGH COMPLEXITY)
  └── Qwen2.5-14B-Instruct (Q4_K_M on GEX44) or Qwen2.5-32B (Q4 on GEX131)
      ├── Financial data structuring + table parsing
      ├── Walk-forward strategy code generation (overnight)
      ├── Backtest result interpretation
      ├── Factor engineering suggestions
      └── ~10-20 calls per intelligence phase, medium input, structured output

Investment Reasoning / Narrative (LOW VOLUME, HIGHEST COMPLEXITY)
  └── Mimo v2 (API) OR InvestLM-65B (GEX131 only, ~40 GB Q4)
      ├── 12 investor persona agents (Buffett, Munger, etc.)
      ├── MiroFish report generation (multi-turn)
      ├── Trade thesis narrative
      ├── Risk commentary
      └── ~12-24 calls per decision cycle, long input, reasoning output
```

### Model Specifications & VRAM Requirements

#### FinMA-7B (PIXIU Project) — CAUTION
- **Base:** Original LLaMA-7B (NOT LLaMA-2/3) fine-tuned on 136K financial instructions
- **Variants:** `finma-7b-nlp` (sentiment/NER) and `finma-7b-full` (NLP + generation)
- **VRAM:** ~14 GB (FP16) / ~4-5 GB (INT4 GGUF via llama.cpp conversion)
- **Speed:** ~30-50 tokens/sec on RTX 4000 (INT4)
- **PROBLEM:** Based on original LLaMA (2023), now significantly behind Qwen2.5-7B
  on most financial benchmarks due to 2+ years of architectural advances.
- **Verdict:** Skip FinMA. Use **Qwen2.5-7B** instead — it outperforms FinMA on
  financial sentiment while also handling code, math, and structured output.
  Or use **Qwen2.5-Math-7B-Instruct** which surpasses Qwen2-Math-72B on math.

#### Qwen2.5 (Alibaba) — PRIMARY RECOMMENDATION
- **Sizes available:** 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- **Strength:** Best open-source model for quantitative tasks, code, structured output, math
- **Specialized variants:** Qwen2.5-Math (surpasses 72B on math at 7B), Qwen2.5-Coder
- **Quantization:** Official GGUF, AWQ, GPTQ from Qwen team on HuggingFace
- **Quality retention:** AWQ ~95% quality, GGUF Q4_K_M ~92% (loss is in creative, not numerical)
- **VRAM by size:**

| Size | FP16 VRAM | INT8 VRAM | INT4 (Q4_K_M) VRAM | Tokens/sec (RTX 4000 Ada, INT4) |
|------|-----------|-----------|---------------------|---------------------------------|
| 7B | ~14 GB | ~8 GB | ~4-6 GB | ~40-60 t/s |
| 14B | ~28 GB | ~14 GB | ~10-12 GB | ~20-35 t/s |
| 32B | ~64 GB | ~32 GB | ~18-22 GB | ~10-20 t/s |
| 72B | ~144 GB | ~72 GB | ~36-48 GB | Needs GEX131 (96 GB) |

- **GEX44 sweet spot:** Qwen2.5-7B-FP16 (14 GB) + Qwen2.5-Math-7B-Q4 (5 GB) = 19 GB
  OR Qwen2.5-14B-Q4 (10 GB) + Qwen2.5-7B-Q4 (5 GB) = 15 GB
- **GEX131 sweet spot:** Qwen2.5-72B-Q4 (48 GB) + Qwen2.5-Math-7B-FP16 (14 GB) = 62 GB

#### InvestLM-65B
- **Base:** LLaMA-65B fine-tuned on investment texts
- **Strength:** Investment-specific reasoning, financial QA, portfolio commentary
- **VRAM:** ~130 GB (FP16) / ~35-40 GB (INT4 GGUF via TheBloke)
- **Problem:** Even quantized, needs ~40 GB VRAM — won't fit alongside other models
  on a single 20 GB GPU. Requires 48+ GB VRAM (A6000, L40, A100).
- **Alternative:** Use via API, or skip it — Qwen2.5-32B with financial prompting
  achieves ~85% of InvestLM quality for investment reasoning.

#### Air-LLM (Already in your stack)
- **What it is:** Inference framework, not a model — enables layer-by-layer loading
  to run models larger than VRAM (e.g., 70B on 4 GB VRAM)
- **Tradeoff:** 10-50x slower inference (seconds per token, not tokens per second)
- **Use case:** Overnight batch jobs only (not real-time). Good for:
  - Nightly report generation with InvestLM-65B
  - Weekly deep analysis runs
  - NOT for intraday decision loop (too slow)

### Recommended Self-Hosted Configuration

#### Option A: GEX44 — Dual-Model Serving (~EUR 184/month)

```
Hetzner GEX44
├── GPU: NVIDIA RTX 4000 SFF Ada — 20 GB GDDR6 ECC
├── CPU: Intel Core i5-13500
├── RAM: 64 GB DDR4
├── Storage: 2x 512 GB NVMe
├── Price: EUR 184/month + EUR 79 setup
│
├── VRAM Allocation — Option 1 (dual 7B, both always loaded):
│   ├── Qwen2.5-7B-Instruct FP16    → ~14 GB (general + sentiment)
│   ├── Qwen2.5-Math-7B-Instruct Q4 → ~5 GB  (numerical tasks)
│   ├── Overhead/KV cache            → ~1 GB
│   └── TOTAL: 20 GB ✓ fits, both always loaded
│
├── VRAM Allocation — Option 2 (14B + 7B, hot-swap via llama.cpp):
│   ├── Qwen2.5-14B-Instruct Q4     → ~10 GB (primary, always loaded)
│   ├── Qwen2.5-7B Q4               → ~5 GB  (secondary, swappable)
│   ├── Overhead/KV cache            → ~5 GB
│   └── TOTAL: 20 GB ✓ fits via llama.cpp LRU model management
│
├── Serving: llama.cpp server (RECOMMENDED for GEX44)
│   ├── Native multi-model with LRU eviction
│   ├── OpenAI-compatible API (drop-in replacement for Anthropic/OpenAI calls)
│   ├── Single server, multiple models, auto-swap
│   └── Best VRAM efficiency with GGUF quantization
│
├── Mimo v2: via API (not local — save GPU for specialist models)
│
└── Overnight (Air-LLM):
    ├── Unload daytime models
    ├── Load InvestLM-65B layer-by-layer via Air-LLM (~30-120s per response)
    ├── Generate deep analysis reports
    └── Reload daytime models before market open
```

**Monthly cost: EUR 184 fixed** — no per-token API charges for classification
and quant tasks. Only pay API for Mimo v2 reasoning calls.

#### Option B: GEX131 — Full Self-Hosted (~EUR 889/month)

```
Hetzner GEX131
├── GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q — 96 GB GDDR7 ECC
│   └── 5th-gen Tensor Cores: ~50-100+ tok/s on 72B Q4
├── CPU: Intel Xeon Gold 5412U (24 cores / 48 threads)
├── RAM: 256 GB DDR5 ECC
├── Storage: 2x 960 GB Gen4 NVMe
├── Price: EUR 889/month (EUR 1.42/hr)
│
├── VRAM Allocation (96 GB total):
│   ├── Qwen2.5-72B-Instruct Q4  → ~48 GB  (primary: all complex tasks)
│   ├── Qwen2.5-Math-7B FP16     → ~14 GB  (numerical precision)
│   ├── InvestLM-65B-AWQ          → ~40 GB  (swap with 72B overnight)
│   ├── OR: run 72B + Math-7B daytime (62 GB), swap to InvestLM overnight
│   ├── Overhead/KV cache         → ~34 GB (massive context windows)
│   └── TOTAL: 96 GB — fits 72B + 7B simultaneously
│
├── Serving: vLLM (RECOMMENDED for GEX131)
│   ├── PagedAttention for efficient KV cache
│   ├── High throughput with continuous batching
│   ├── OpenAI-compatible API
│   └── 50-100+ tok/s on Blackwell Tensor Cores
│
└── Advantage: ZERO external API dependency
    ├── Complete data privacy (no financial data leaves your server)
    ├── Fixed EUR 889/month regardless of volume
    ├── No rate limits, no API outages
    └── 72B quality rivals GPT-4 class on financial tasks
```

### Cost Comparison: Self-Hosted vs API

| Scenario | API Cost/month | Self-Hosted Cost/month | Break-Even |
|----------|---------------|----------------------|------------|
| Light (100 calls/day) | ~EUR 30-50 | EUR 184 (GEX44) | Never — API cheaper |
| Medium (1,000 calls/day) | ~EUR 150-300 | EUR 184 (GEX44) | ~1 month |
| Heavy (5,000 calls/day) | ~EUR 500-1,500 | EUR 184 (GEX44) | Immediately |
| Full self-hosted | ~EUR 0 | EUR 889 (GEX131) | vs ~EUR 2,000+ API |

**Your usage estimate:** 12 personas × every decision cycle + sentiment on
100+ headlines + quant analysis = ~2,000-5,000 LLM calls/day at production.
**Self-hosted wins at your volume.**

### Serving Infrastructure: llama.cpp vs vLLM vs TGI

| Feature | llama.cpp | vLLM | TGI |
|---------|-----------|------|-----|
| **Status** | Active | Active, recommended | **Maintenance mode** (Dec 2025) |
| **Multi-model** | LRU eviction (hot-swap) | No native support | No |
| **Quantization** | GGUF (Q2-Q8, imatrix) | AWQ/GPTQ/FP8/Marlin | AWQ/GPTQ |
| **Throughput** | Good for 1-5 users | Best (120-160 req/s on A100) | Good |
| **TTFT** | Varies | 50-80ms | 60-90ms |
| **Memory efficiency** | Best (GGUF) | Good (PagedAttention) | Good |
| **OpenAI-compatible API** | Yes (built-in) | Yes (built-in) | Yes |
| **Multi-GPU** | Layer splitting (suboptimal) | Tensor + pipeline parallelism | Tensor parallelism |
| **LoRA adapters** | Limited | Via Lorax (thousands on 1 base) | Limited |
| **Best for GEX44 (20 GB)** | **Yes** | Tight | Skip |
| **Best for GEX131 (96 GB)** | Works | **Yes** | Skip |

**GEX44 recommendation:** **llama.cpp server** — only option with native
multi-model LRU management. Load Qwen2.5-7B + Math-7B simultaneously, or
hot-swap between 14B and 7B models transparently.

**GEX131 recommendation:** **vLLM** — PagedAttention + continuous batching
for high-throughput single-model serving (Qwen2.5-72B). If you need
multiple models, run separate vLLM instances with `--gpu-memory-utilization`.

**LoRA alternative:** If you fine-tune multiple adapters on Qwen2.5 base
(e.g., sentiment adapter, quant adapter, reasoning adapter), Lorax via
vLLM serves thousands of adapters from one base with near-zero VRAM overhead.
This is the most efficient multi-"model" approach if applicable.

### Integration Architecture

```
Engine Heartbeat (3-min cycle)
│
├── Phase 1: DATA → OpenBB API (unchanged)
│
├── Phase 2: SIGNALS
│   ├── news_engine.py → newsfilter.io headlines
│   └── Headlines → LOCAL Qwen2.5-7B (llama.cpp, port 8100)
│       └── Sentiment: BULLISH/BEARISH/NEUTRAL + confidence
│       └── Latency: ~50-100ms per headline (batched)
│       └── Replaces Claude/GPT-4 API calls in news_sentiment_agent
│
├── Phase 3: INTELLIGENCE
│   ├── alpha_optimizer.py → LOCAL Qwen2.5-14B or Math-7B (port 8100)
│   │   └── Feature importance interpretation
│   │   └── Numerical calculations (Math-7B surpasses 72B on math)
│   └── investor_personas.py → Mimo v2 API (complex reasoning)
│       └── 12 persona signals (Buffett, Munger, etc.)
│
├── Phase 4: DECISION
│   └── decision_matrix.py → LOCAL Qwen2.5-14B (port 8100)
│       └── Trade thesis structuring
│       └── Risk narrative
│
├── Phase 5: EXECUTION → Alpaca API (unchanged)
│
├── Phase 6: LEARNING
│   └── learning_loop.py → LOCAL Qwen2.5-14B
│       └── Performance attribution analysis
│
└── Phase 7: MONITORING
    └── market_wrap.py → LOCAL Qwen2.5-14B
        └── Daily narrative generation

OVERNIGHT (Air-LLM on GEX44, or vLLM model swap on GEX131):
├── InvestLM-65B → Deep investment analysis reports
├── Platinum Report generation → narrative sections
└── Weekly strategy review → multi-turn reasoning
```

### Why Qwen2.5 Replaces FinMA

FinMA-7B was fine-tuned on financial data but is based on **original LLaMA (2023)**.
Qwen2.5-7B (2025) has 2+ years of architectural advances and outperforms FinMA
on most financial benchmarks while ALSO handling code, math, and structured output.

Additionally, Qwen2.5-Math-7B-Instruct specifically surpasses Qwen2-Math-72B
on mathematical benchmarks — meaning a 7B specialized model beats a 72B general
model on numerical tasks. This is directly relevant for:
- Alpha optimizer calculations
- Risk metric computation  
- Factor engineering
- Backtest result interpretation

Using the Qwen2.5 family (7B general + 7B math, or 14B general) gives you
better performance than FinMA + separate math model, with fewer moving parts.

---

## Language Performance — Task Allocation Strategy

### Current Stack

```
Python   → Engine (live_loop_orchestrator, all signals, ML, execution)
Python   → FastAPI backend (port 8001)
TypeScript → Express.js proxy server (port 5000)
TypeScript → React client (Vite, browser)
Python   → Flask (MiroFish backend, port 5001)
TypeScript → Vue 3 (MiroFish frontend, port 5174)
```

### Where Language Performance Actually Matters

Your engine runs on a **3-minute heartbeat**. Each phase completes in
milliseconds to low seconds. Python is not the bottleneck — OpenBB API
calls and network I/O are. Rewriting to Rust/Go would save microseconds
on a 180-second cycle. Not worth it.

**However**, there are specific components where language choice matters
for the NEXT evolution of the platform:

| Component | Current | Bottleneck | Recommendation |
|-----------|---------|------------|----------------|
| **Live loop orchestrator** | Python | Not bottlenecked (3-min cycle) | Keep Python |
| **Signal engines** | Python (numpy/scipy) | CPU numerical — already fast via numpy C extensions | Keep Python |
| **ML models** | Python (XGBoost, hmmlearn) | Already C/C++ under the hood | Keep Python |
| **Alpha optimizer** | Python (scipy SLSQP) | Already Fortran under the hood | Keep Python |
| **FastAPI backend** | Python | <100ms response, fine | Keep Python |
| **Express proxy** | TypeScript | <50ms overhead, fine | Keep TypeScript |
| **React client** | TypeScript | Browser, irrelevant | Keep TypeScript |
| **Order matching** | Python (exchange_core) | Simulated, not real matching | Keep Python (or Rust if going real HFT) |
| **Data ingestion** | Python | I/O bound (API calls) | Keep Python |
| **WebSocket/SSE server** | Python SSE (stub) | Could be bottleneck at scale | Consider Go/Rust for high-fan-out |

### What Would Actually Benefit from Rust/Go

Only if you evolve toward real-time:

| Future Component | Why Faster Language Helps | Language |
|-----------------|--------------------------|----------|
| **Tick-by-tick data ingestion** | Millions of messages/sec | Rust |
| **Real order matching engine** | Microsecond latency matters | Rust (or keep Java exchange-core) |
| **WebSocket fan-out** (1000+ clients) | Concurrent connections | Go |
| **Time-series DB writes** | High-throughput append | Rust |

### Practical Recommendation

```
KEEP PYTHON for:
  ├── Engine (live_loop_orchestrator.py) — 3-min batch, not bottlenecked
  ├── All signal engines — numpy/scipy already call C/Fortran
  ├── ML pipeline — XGBoost/hmmlearn are C++ underneath
  ├── FastAPI backend — async, <100ms, adequate
  └── MiroFish backend — Flask, low traffic

KEEP TYPESCRIPT for:
  ├── Express proxy — lightweight, adequate
  ├── React client — browser rendering
  └── MiroFish Vue frontend — browser rendering

CONSIDER ADDING (future, only if needed):
  ├── Go — High-concurrency API gateway / WebSocket server
  │         (if you need 1000+ simultaneous dashboard connections)
  └── Rust — Real tick-by-tick processing / order matching
              (only if moving to true sub-second HFT)
```

**Bottom line: Python's "slowness" is a myth for your use case.**
Your hot path is 3-minute batches where 95% of wall-clock time is
waiting for API responses. numpy/scipy/XGBoost are C/Fortran underneath.
Rewriting to Rust saves you 50ms on a 180,000ms cycle.

---

## PM2 Process Management — 24/7 Architecture

### Why PM2 (Not Docker for Process Management)

PM2 is purpose-built for keeping Node.js + Python processes alive.
It handles restart-on-crash, log rotation, clustering, and monitoring
out of the box. Docker handles isolation. Use both.

### PM2 Ecosystem File

```javascript
// ecosystem.config.js
module.exports = {
  apps: [
    // === CORE ENGINE (Python, 24/7) ===
    {
      name: "metadron-engine",
      script: "python3",
      args: "-u engine/live_loop_orchestrator.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,        // 5s backoff
      max_memory_restart: "8G",   // restart if memory leak
      env: {
        PYTHONUNBUFFERED: "1",
        BROKER_TYPE: "paper",
        OBB_TOKEN: "...",
      },
      // Self-healing: restart on crash, PM2 watches process
    },

    // === FASTAPI BACKEND (Python, port 8001) ===
    {
      name: "metadron-api",
      script: "uvicorn",
      args: "app.backend.main:app --host 0.0.0.0 --port 8001 --workers 2",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_memory_restart: "2G",
    },

    // === EXPRESS PROXY (Node.js, port 5000) ===
    {
      name: "metadron-proxy",
      script: "server/index.ts",
      cwd: "/opt/metadron",
      interpreter: "npx",
      interpreter_args: "tsx",
      autorestart: true,
      env: {
        PORT: 5000,
        ENGINE_API_PORT: 8001,
      },
    },

    // === MIROFISH BACKEND (Python/Flask, port 5001) ===
    {
      name: "mirofish-backend",
      script: "python3",
      args: "mirofish/backend/run.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
    },

    // === MIROFISH FRONTEND (Vue/Vite, port 5174) ===
    // In production: serve built static files via Nginx instead
    // Only use PM2 for dev mode:
    {
      name: "mirofish-frontend",
      script: "npm",
      args: "run dev",
      cwd: "/opt/metadron/mirofish/frontend",
      autorestart: true,
    },

    // === LEARNING LOOP (overnight ML retraining) ===
    {
      name: "metadron-learner",
      script: "python3",
      args: "-u engine/monitoring/learning_loop.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      cron_restart: "0 20 * * 1-5",  // Restart nightly at 20:00 ET weekdays
      autorestart: false,             // One-shot, cron triggers next run
    },
  ],
};
```

### PM2 Commands

```bash
# Start all services
pm2 start ecosystem.config.js

# Monitor real-time
pm2 monit

# View logs (all or specific)
pm2 logs
pm2 logs metadron-engine --lines 100

# Restart on deploy
pm2 reload all --update-env

# Save process list (survives reboot)
pm2 save
pm2 startup  # generates systemd service

# Health check
pm2 status
```

### Self-Healing Architecture

```
PM2 Watchdog
├── metadron-engine     → autorestart: true, max_restarts: 50
│   ├── Crash → restart in 5s (restart_delay)
│   ├── Memory leak → restart at 8 GB (max_memory_restart)
│   └── Heartbeat continues from last state (deque persists in Redis)
│
├── metadron-api        → autorestart: true, 2 uvicorn workers
│   └── Worker crash → uvicorn respawns, PM2 catches full exit
│
├── metadron-proxy      → autorestart: true
│   └── Express crash → PM2 restarts, Nginx buffers requests
│
└── metadron-learner    → cron_restart nightly
    └── Walk-forward retrain → save models → engine hot-reloads
```

---

## Latency Analysis — Hetzner Location vs Mimo v2

### API Call Latency Map

| Route | Germany (Falkenstein) | Finland (Helsinki) | US East (Ashburn) |
|-------|----------------------|--------------------|--------------------|
| → Alpaca API (US) | 80-120ms | 90-130ms | 5-15ms |
| → OpenBB/FRED (US) | 80-120ms | 90-130ms | 10-30ms |
| → Anthropic API (US) | 50-80ms | 60-90ms | 5-15ms |
| → Mimo v2 (if US-hosted) | 50-80ms | 60-90ms | 5-15ms |
| → Mimo v2 (if local) | <1ms | <1ms | <1ms |
| User browser (EU) | 5-20ms | 10-30ms | 80-120ms |
| User browser (US) | 80-120ms | 90-130ms | 5-20ms |

### Latency Impact on Your System

Your engine heartbeat is **180 seconds (3 minutes)**. Even from Germany:
- Worst-case API round-trip: ~120ms
- Number of API calls per heartbeat: ~5-20
- Total API latency per cycle: ~600ms-2.4s
- **As % of cycle: 0.3% - 1.3%** — completely irrelevant

### When Latency WOULD Matter

Only if you evolve to:
- Real tick-by-tick execution (sub-second) → need US East colo
- Local Mimo v2 inference in the decision loop → local is always fastest
- WebSocket streaming to 100+ clients → location near users matters

### Recommendation

| Your Users | Mimo v2 | Best Location |
|------------|---------|---------------|
| EU-based | API (remote) | **Hetzner Germany** (cheapest, closest to you) |
| EU-based | Local (self-hosted) | **Hetzner Germany** (GPU server) |
| US-based | API (remote) | **Hetzner Ashburn** or US provider |
| Global | Either | Germany (cheapest) — 120ms to US is fine for 3-min cycles |

---

## Hetzner Product Comparison

### Web Hosting vs VPS vs Dedicated

| Feature | Web Hosting | VPS (Cloud) | Dedicated Server |
|---------|-------------|-------------|------------------|
| Hardware | Shared | Virtual (shared physical) | Physical machine, all yours |
| CPU | Shared, throttled | vCPUs (burstable or dedicated) | Full dedicated cores |
| RAM | 1-4 GB | 4-64 GB | 64-256+ GB |
| Root access | No | Yes | Yes |
| Run Python/Docker/PM2 | No | Yes | Yes |
| 24/7 sustained load | No | Throttled on shared | Yes, full speed |
| Noisy neighbors | Yes | Yes (shared) / No (dedicated vCPU) | No |
| Use case | WordPress, HTML | Dev, small apps | Production trading |
| **For Metadron?** | **Not suitable** | **Dev/paper only** | **Production** |

### Recommended Products

#### CPU Only: AX102 (~EUR 80-90/month)

- AMD Ryzen 9 7950X3D — 16 cores / 32 threads
- 128 GB DDR5 ECC RAM
- 2x 1 TB NVMe SSD (RAID-1)
- **Best for:** All LLM via API, full engine + web stack + PM2

#### Entry GPU: GEX44 (EUR 184/month + EUR 79 setup)

- Intel Core i5-13500
- 64 GB DDR5 RAM
- NVIDIA RTX 4000 SFF Ada — **20 GB GDDR6 ECC VRAM**
- 2x 512 GB NVMe
- Location: Falkenstein (FSN1)
- **Best for:** Qwen2.5-7B FP16 + Math-7B Q4 dual serving, or 14B Q4 single
- **Fits:** 2 Qwen2.5-7B models simultaneously, overnight InvestLM via Air-LLM

#### Full GPU: GEX131 (EUR 889/month)

- Intel Xeon Gold 5412U — 24 cores
- 256 GB DDR5 ECC RAM
- NVIDIA RTX PRO 6000 Blackwell — **96 GB GDDR7 ECC VRAM**
- 2x 960 GB NVMe
- Location: Nuremberg (NBG1) or Falkenstein (FSN1)
- **Best for:** ALL models loaded concurrently, zero API dependency, complete privacy

---

## Deployment Architecture (PM2 + Nginx)

```
Hetzner AX102 / GEX44
│
├── PM2 (process manager, 24/7)
│   ├── metadron-engine         Python: live_loop_orchestrator (24/7 daemon)
│   │   └── Self-learning: LearningLoop + GSD + Paul plugins
│   │       └── Walk-forward retrain, agent promotion/demotion
│   ├── metadron-api            FastAPI (port 8001, 2 workers)
│   ├── metadron-proxy          Express.js (port 5000, proxies to 8001)
│   ├── mirofish-backend        Flask (port 5001)
│   ├── metadron-learner        Overnight cron: ML retrain + backtest
│   └── mimo-v2-server          (optional) Local LLM inference server
│
├── Nginx (reverse proxy + SSL)
│   ├── metadron.capital → React client (static build)
│   ├── api.metadron.capital → Express proxy → FastAPI
│   ├── mirofish.metadron.capital → Vue static build
│   └── Let's Encrypt auto-renewal
│
├── PostgreSQL (market data, positions, signals, learning history)
├── Redis (signal cache, pub/sub, session state, heartbeat deque)
│
└── systemd
    └── pm2-startup (auto-start PM2 on reboot)
```

### Self-Learning / Self-Managed Flow

```
24/7 CONTINUOUS OPERATION:

INTRADAY (09:30-16:00 ET) — 3-min heartbeat:
  Engine → Data → Signals → ML → Decision → Execute → Learn → Monitor
  └── LearningLoop: signal accuracy tracking, agent scoring
  └── GSDPlugin: gradient signal dynamics, convergence detection
  └── PaulPlugin: pattern memory, context-aware replay

AFTER-HOURS (16:00-20:00 ET) — 30-min heartbeat:
  Engine → Reduced signals → Earnings scan → Agent scoring
  └── AgentMonitor: promote/demote based on daily performance

OVERNIGHT (20:00-08:00 ET) — 1-hr heartbeat:
  Engine → Minimal monitoring
  PM2 cron → metadron-learner:
    ├── QSTrader walk-forward backtesting
    ├── XGBoost/HMM/PPO model retraining on new data
    ├── Pattern evolution (PaulPlugin memory update)
    ├── Agent scorecard recalculation
    └── Model artifacts saved → engine hot-reloads on next heartbeat

PRE-MARKET (08:00-09:30 ET) — 3-min heartbeat:
  Engine → Full data refresh → Overnight signal integration → SEC scan
  └── New models loaded from overnight training
```

---

## Scaling Path

```
Phase 1 — GEX44 (Recommended Start)                  EUR 184/mo FIXED
  ├── Qwen2.5-7B FP16 + Qwen2.5-Math-7B Q4 on GPU (20 GB VRAM)
  ├── llama.cpp server with LRU model management
  ├── Mimo v2 via API (complex reasoning only)
  ├── Overnight: InvestLM-65B via Air-LLM
  ├── PM2 managing all services
  ├── ~2,000-5,000 local LLM calls/day at zero marginal cost
  └── Only API cost: Mimo v2 reasoning (~EUR 50-100/mo estimated)
      TOTAL: ~EUR 234-284/mo

Phase 2 — GEX131 (Full Self-Hosted)                  EUR 889/mo FIXED
  ├── Qwen2.5-72B Q4 (48 GB) + Math-7B FP16 (14 GB) daytime
  ├── InvestLM-65B AWQ (40 GB) overnight swap
  ├── Zero API costs, zero external dependency
  ├── Complete data privacy (no data leaves server)
  └── 24-core Xeon + 256 GB RAM for engine + ML
      TOTAL: EUR 889/mo (zero variable costs)

Phase 3 — Split Architecture                         EUR 1,100+/mo
  ├── GEX131 for LLM inference + engine
  ├── AX102 for backtesting + overnight ML training
  └── Separate compute prevents backtest from competing with live engine

Phase 4 — US East Node (HFT Evolution)               EUR 1,500+/mo
  ├── Add US East dedicated for sub-second execution
  ├── Rust order matching engine
  └── EU GPU server for LLM + intelligence, US for execution
```

## Security Advantage of Self-Hosted

```
API-based:
  └── Every LLM call sends financial signals, positions, trade reasoning
      to third-party servers (Anthropic, OpenAI)
      └── Risk: data retention policies, potential exposure

Self-hosted (GEX44/GEX131):
  └── ALL inference happens on YOUR server
      ├── No financial data leaves the machine
      ├── No API keys to manage (except Mimo v2 on GEX44)
      ├── No rate limits during market hours
      ├── No API outages during critical trading decisions
      └── Fixed monthly cost regardless of call volume
```
