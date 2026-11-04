# Metadron Capital — self-contained build (LEAN / QuantConnect + IBKR)

This repository is the clean, standalone extraction of the Metadron intelligence engine. It runs on
its own: every engine module, the research integrations it loads, the vendored upstream repos
(Kronos, LEAN, OpenJarvis), Jarvis, the docs and the tests are inside it.

Reference architecture: `docs/architecture_dna/ARCHITECTURE_DNA.md` (the original design — where it
disagrees with this file, THIS file wins; e.g. its corridor section was linear and is superseded).

## Layout

```
algorithm/              QuantConnect LEAN algorithm: main (heartbeat + wiring), corridor, risk_budget, regime, forward_rm,
                        fundamentals, universe, profiles, decision, alpha, options, protection, feed_guard, schedule, sectors, rotation,
                        market_map, macro_regime, sector_rotation
agents/                 market_reasoning.md, roster.yaml, llm.py, qc_api.py, macro_agent, events_agent, kronos_agent,
                        risk_officer, vault_sync, qc_deploy (all run by GitHub Actions; propose-only)
vault/                  Obsidian control room (Dashboard + generated Daily / Positions / Risk notes)
.github/workflows/      tests, deploy-to-QuantConnect, agents pre-market / post-close, Kronos weekly
SETUP.md                one-afternoon setup + hosting (QuantConnect cloud + GitHub Actions + Obsidian)
engine/                 Intelligence engine (extracted from the legacy monorepo, package paths unchanged)
  data/ signals/ ml/    universe, macro (GMTF, HMM), cube, security analysis, distress, events, CVR, patterns, ML
  allocation/           AllocationEngine + universe scan — THE allocation reference
  execution/            decision matrix, options engine, protection, event tail hedges, conviction override, L7
  risk/                 Monte Carlo risk + pre-trade MC gate
  portfolio/            legacy beta corridor (superseded by algorithm/corridor.py)
  agents/ monitoring/   sector/research bots, scorecard, NanoClaw permission guard, learning loop, reports
  api/ db/ security/    FastAPI engine API (port 8001), SQLite, auth
  bridges/              Jarvis, Llama / Qwen / Air-LLM servers, LLM inference bridge, Prometheus
core/                   original platform core (cyclical vs secular decomposition, money velocity)
intelligence_platform/  research integrations the engine loads at runtime (MiroFish, QLIB, distress, qstrader, …)
vendor/                 Kronos, LEAN, OpenJarvis — full source copies, pinned in vendor/VENDOR.md
jarvis/                 Jarvis setup guide + dashboard page
governance/ config/     credit classification, repo/heartbeat config
docs/                   architecture DNA, corridor charts
tests/                  285 tests (legacy engine suite + new algorithm tests)
scripts/update_vendor.sh  re-sync vendored repos to upstream
```

## Commands

```bash
pip install -r requirements.txt
python3 -m pytest tests -q                    # full suite
python3 docs/charts/corridor_v2.py            # regenerate the corridor chart
pm2 start ecosystem.config.cjs                # engine API + Jarvis + model servers
scripts/update_vendor.sh                      # update Kronos / LEAN / OpenJarvis
```
QuantConnect: see SETUP.md (GitHub Actions pushes `algorithm/` and starts backtests; going live is one owner click).

## Design rules

1. QuantConnect (LEAN) is the execution and backtest engine; IBKR is the broker (paper → live).
   QuantConnect data for signals, IBKR quotes only for cross-checks.
2. Every module stays pure Python where possible; all optional imports are guarded (`try/except`)
   and degrade gracefully — never crash on a missing library.
3. Hard caps live in code constants (`algorithm/corridor.py`, `algorithm/risk_budget.py`) and change
   only with the owner's approval; agents may propose, never apply.
4. Agents (and Jarvis) are read-only on the trading system: they research, explain and recommend.
5. Tests must pass before pushing. Vendored code is updated only through `scripts/update_vendor.sh`.
6. Agents use the Anthropic API (current default model: claude-opus-5-5).

### Smart Beta v2 — CURRENT definition (owner-approved; code: `algorithm/corridor.py`)

Chart: `docs/charts/corridor_v2.png` (script: `docs/charts/corridor_v2.py`).

```
L(Rm, σm) = k · s(Rm) · (Rm − rf) / σm²          long market-beta leverage, capped at 4.5x

Rm     annualised S&P 500 return (expected)          rf  risk-free rate = 4%
σm     current annualised market volatility           k   share of full Kelly = 1.0 (parameter)
t      = clip((Rm − 7%) / 5%, 0, 1)
s(Rm)  = 3t² − 2t³                                   corridor participation (smooth 0 → 1)
γ(Rm)  = ds/dRm = 6t(1 − t) / 5%                     corridor gamma — flattened bell,
                                                     peak 0.30 per 1% at 9.5%, area 1
(Rm − rf)/σm²                                        "gamma market": vol-adjusted full Kelly
```

| Zone | Rm (annualised S&P) | Long beta |
|------|---------------------|-----------|
| DEFENSIVE | < 7% (bear / weak market) | **0** — only shorts / hedges |
| DYNAMIC | 7% – 12% | smooth ramp, k · s · (Rm − 4%)/σ² |
| ROTATION | > 12% | full vol-adjusted Kelly + active sector rotation, up to the 4.5x cap |

Leverage (k = 1): σ 15% → 0.43 / 1.22 / 2.79 / 3.56 at Rm 8.5 / 9.5 / 11 / 12%, 4.44 at 14%, 4.5 cap
from ≈14.1%; σ 20% → 2.00 at 12%, 3.00 at 16%; σ 25% → 1.28 at 12%. Rising volatility shrinks
leverage by 1/σ², so the book de-risks itself in volatile tapes without exiting.
Rebalance only when |ΔL| > 0.05. Delivery: Micro E-mini S&P futures (MES) — 4.5x is not reachable with
Reg-T stock margin (≈2x); SPY is the ≤2x alternative. Extra leverage above 12% may come from long calls
when implied vol ≤ realised (delta efficiency Ω = Δ·S/premium), else futures / call spreads.

### Risk budget (owner-approved; code: `algorithm/risk_budget.py`) — replaces "VaR ≤ 1.5% NAV"

A single 1.5% 1-day VaR on the whole book caps equity leverage at 0.97x at σ 15% (0.72x at 20%) and
blocks most small/mid caps — superseded. Under vol-adjusted Kelly, beta VaR = 1.645·k·s·Sharpe/√252
(falls as σ rises), so VaR is a regime ceiling, not the sizing tool.

| Layer | Rule |
|-------|------|
| 1 Sizing | corridor k-Kelly leverage above |
| 2 Regime VaR ceiling on market-beta risk (1-day 95%) | TRENDING 7.0% · RANGE 4.5% · STRESS 2.25% · CRASH 1.0% → leverage ceiling = VaR / (1.645·σm/√252). TRENDING 7.0% = the VaR of 4.5x at σ 15%, so the owner's max is reachable |
| 3 Stock-specific (alpha) VaR budget, 1-day 95% | **2.5% of NAV** (parameter) on idiosyncratic risk after removing market beta; per-name ≤ 0.5% NAV (≈ 9.7% weight at 50% vol); sizing ∝ 1/vol; same-sector names correlated 0.3; long options count at premium (defined risk); trades scaled down to fit, not rejected |
| 4 Hard stops | gross ≤ 4.5x; MES futures notional ≤ 2.0x NAV long or short (owner); CVaR95 ≤ 1.5 × regime ceiling; drawdown −4% trim (halve), −8% close all but hedges |
| No double sizing | When Rm > 12% (ROTATION) and the regime reads STRESS/CRASH, the regime VaR ceiling is NOT applied on top of Kelly — Kelly's 1/σ² already sizes for volatility; only the 4.5x cap remains. A crash *inside* a strong market is handled by theme rotation, not by cutting market beta twice. |

Diversified small/mid-cap books fit easily: 40 names × 2.5% at 45% vol → 0.74%; 20 × 5% → 1.04%;
the budget binds only on concentration (10 × 10% at 50% vol → 1.64%; 5 × 20% → 2.32%).

### Four levels of "crash" + theme rotation (owner-approved; code: `algorithm/rotation.py`)

| Level | Detected by | Response |
|-------|-------------|----------|
| MARKET | S&P regime + Rm (corridor) | sizes market beta (MES overlay) |
| THEME | equal-weight theme index ≥ 30% below its 63-day high AND ≥ 15 pts under SPY over 21d = CRASH (≥ 15% / ≥ 8 pts = STRESS) | rotation playbook below — market beta untouched |
| POSITION | one holding's own drawdown | position protection (stops / trims) |
| PORTFOLIO | NAV drawdown | hard stops (−4% trim, −8% close) |

Paradigm (Oct–Nov): SPY > 12% at max leverage while AI names with heavy debt / private-credit exposure
(e.g. CRWV, NBIS) and capex over-spenders (e.g. IREN, HIVE) crash. Playbook — all equity sizes pass the
2.5% stock-specific VaR budget; targets are idempotent (re-running daily never compounds):

1. **CUT** crashed-theme longs to 0 (CRASH) or ≤ 2% NAV per name (STRESS).
2. **ROTATE** through the **11 GICS sectors** (`algorithm/sectors.py`) into **single stocks and their options** —
   never sector ETFs (owner rule). The equity book is split across the sectors every day by the research: sector
   score = mean edge of its sub-sectors + regime tilt (STRESS/CRASH: defensives +0.30, cyclicals −0.20, growth −0.10;
   TRENDING: cyclicals/growth +0.10) + bullish/bearish bias tilt + escalation (Energy +0.30, Industrials +0.15)
   − 0.50 for the crashed theme's own sector → sector budget ∝ max(score + 0.10, 0), each active sector 3–30% of a
   sleeve (none positive → Health Care / Staples / Utilities). The universe shortlist gives each sector candidate
   slots in proportion; the decision matrix fills each (sleeve × sector) bucket with single names; the plan names
   the top 3 destination sectors and buys rotation calls on their 2 best names (IV ≤ 1.25 × RV). Budgets are
   recomputed from state, so rotation is idempotent.
3. **ESCALATION** posture (score ≥ 0.5): Energy sector tilt in the budgets + long calls on the 2 best Energy single
   names (1% NAV premium, ~60 DTE; USO/XLE only if no Energy name qualifies). Score = mean of: oil +10%/21d, gold +5%/21d, defense beats SPY by 5 pts/21d,
   VIX ≥ 25; an events-agent flag overrides.
4. **HARVEST** the drawdown: LEAPS calls (≥ 270 DTE, 0.5% NAV premium) on crashed names ≥ 18% below their
   1-year high that pass the solvency check (Merton PD ≤ 5%, interest coverage ≥ 2; unknown = no LEAPS);
   call spreads instead when IV/RV > 1.3. Short-dated straddles (7–30 DTE, 0.3% NAV, max 3) where realised
   vol ≥ 1.1 × implied or an event is within 14 days.

**Themes are reasoning profiles, never ticker lists** (owner rule; code `algorithm/profiles.py`). The names in the
paradigm are illustrations. Members are detected from fundamentals at every rescore, in the S&P 1500 and in a
liquid off-index watch screen (AI-chain sectors, cap ≥ $2B, $20M/day):
- AI_CREDIT_HUBS = AI-chain sector + capex ≥ 30% of revenue + credit-funded (debt/EBITDA ≥ 4, cover < 3 or D/E ≥ 1.5)
- AI_OVERSPEND = AI-chain sector + capex ≥ 40% of revenue + free-cash-flow burn
Largest 10 per theme form the equal-weight theme index. When corporate spreads widen (question tree CREDIT_STRESS,
HY OAS +35 bp/1m) a theme only 10% off its high is already STRESS — credit-funded themes break first.
Off-index watch names are monitored (theme states, LEAPS solvency, events) but never enter the shortlist.
The escalation defense signal uses profiled aerospace & defense names (ITA only as fallback). `rotation_trading=true` (default) trades the equity legs; option legs go through
the options engine. Targets are idempotent (equities and options); min trade 0.5% NAV (no churn).

### Trader framework → sub-sector rotation (owner-approved; code: `algorithm/market_map.py`, `macro_regime.py`, `sector_rotation.py`; agent guide: `agents/market_reasoning.md`)

The theme-crash playbook above is ONE instance of a general framework that agents must understand:

1. **Market map** (`market_map.py`): 26 sub-sectors tagged CYCLICAL / SECULAR / DEFENSIVE, their role in
   value chains (AI: platforms → chip design → semi equipment → regional foundries → memory → networking →
   data centers → power/grid; debt-funded neoclouds and capex-heavy pivots; at-risk low-end SaaS and
   labor-intensive services), ETF proxies, illustrative names (documentation only), and sensitivities (priors, re-estimated
   weekly) to 12 factors: GROWTH, INFLATION, LABOR, WAGES, PRODUCTIVITY, RATES, LIQUIDITY, CREDIT,
   GEOPOLITICS, SENTIMENT, AI_CAPEX, COMMODITIES.
2. **Macro question tree** (`macro_regime.diagnose`): inflation within mandate? → economy hot (data vs
   expectation)? → labor tight? → gapping? → room for stimulus? → productivity? → real wages? (productivity ↑
   + wages flat + employment retained ⇒ AI_SUBSTITUTION) → liquidity? → corporate bonds (HY OAS +35 bp/1m ⇒
   CREDIT_STRESS)? → geopolitics (ESCALATION)? → sentiment (EUPHORIA / CAPITULATION)? Each tag moves named
   sub-sectors (IMPLICATION_EFFECTS). Also: stock–bond correlation regime (negative ⇒ Treasuries hedge;
   positive ⇒ commodities / gold / cash) and market bias BULLISH / NEUTRAL / BEARISH.
3. **Scenarios** (`sector_rotation`): base / up / down what-ifs with probabilities and factor shocks, written
   by the macro agent or the owner to ObjectStore `metadron/scenarios.json`; macro readings (FRED) to
   `metadron/macro_inputs.json`. Score = sensitivities × factors + implications + 0.5 × fundamentals
   (revisions, revenue, legal fights) + theme state (STRESS −0.3, CRASH −0.8). **Edge** = probability-weighted
   score; **downside** = worst scenario. Equity-sleeve mix by bias (cyclical / secular / defensive):
   BULLISH 40/45/15, NEUTRAL 30/40/30, BEARISH 15/30/55; within each, weight ∝ max(edge − 0.5·max(−downside,0), 0).
4. Output each day: plots (*Market view*, *Equity mix*) and ObjectStore `metadron/sector_views.json`
   (factors, findings, overweight / avoid with drivers, edge and downside) for agents, Jarvis and Obsidian.

### Regime, forward Rm, beta burst, macro agent (build step 2b; code: `algorithm/regime.py`, `forward_rm.py`, `risk_budget.py`, `agents/macro_agent.py`)

**Regime (MARKET level)** — 4-state Gaussian HMM (hmmlearn, refit every 21 days on ≤ 750 days) on SPY 20-day
return, 20-day realised vol, VIX and HY−IG 21-day relative return; today's regime = posterior of the last day.
States are labelled ABSOLUTELY from their own means (never by rank, so a calm market cannot produce a CRASH
state): CRASH vol ≥ 40% or r20 ≤ −15% · STRESS vol ≥ 25% or r20 ≤ −7% · TRENDING r20 ≥ +0.4% and vol < 18% ·
else RANGE. No hmmlearn / short history / fit error → transparent rules on 63-day return and VIX.

**Forward Rm** (replaces the trailing drift, which swings −45% … +63%):

```
anchor  = earnings yield + inflation (macro agent), else 10% long run
trend   = anchor + 0.5 · (EWMA trailing return − anchor)
Rm      = (1 − wK) · [0.5·anchor + 0.5·trend + Σ P(regime)·drift + tension] + wK · Kronos
drift   TRENDING +3% · RANGE 0 · STRESS −4% · CRASH −10%      tension = −8% · (GMTF percentile − 0.5)
wK      = 0.25 when a Kronos index forecast is supplied, else 0        Rm clipped to [−30%, +40%]
```
A 40-pt swing in trailing return moves Rm by 10 pts. Every component is logged on regime changes and plotted.

**Beta burst** — when momentum confirms (price > 50-day MA > 200-day MA and 63-day return > 0) and the regime is
not CRASH, the regime VaR ceiling is multiplied by the legacy burst/cap ratio (TRENDING 0.70/0.65, RANGE
0.55/0.45, STRESS 0.20/0.15), never above 4.5x. It lifts the ceiling only; Kelly still sets the target.

**Macro agent** (`agents/macro_agent.py`, daily before the open, off the trading node) writes ObjectStore
`metadron/macro_inputs.json` (FRED: CPI, unemployment + 12m low, real wages, productivity, 10y change, net
liquidity WALCL−TGA−RRP, HY OAS change, INDPRO growth surprise, GMTF tension percentile) and
`metadron/scenarios.json` (Claude with `market_reasoning.md` as the system prompt; validated before saving).
Secrets: `FRED_API_KEY`, `ANTHROPIC_API_KEY`, `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORG_ID`; each missing piece is
skipped, and files always land in `agents/out/` for manual upload. Owner/agents may add `earnings_yield`,
`escalation`, `kronos_spx_annual` to the readings file.

### Universe + fundamentals → selection (build step 3; code: `algorithm/universe.py`, `fundamentals.py`)

**Universe** — point-in-time S&P 1500 = constituents of IVV + IJH + IJR (QuantConnect ETF-constituent
universes, daily), scored on QuantConnect's Morningstar fundamentals every `fundamental_refresh_days` (5).
Fallback `universe_mode=fundamental`: price ≥ $5, cap ≥ $1B, dollar volume ≥ $5M. Theme members found by the
off-index watch screen are scored too (solvency for LEAPS). Sub-sector per name = fundamentals profile
(`profiles.classify`: capex intensity, leverage, cover, cash burn, margins, size, Morningstar codes): neocloud credit,
hosting pivot, platforms (≥ $200B, FCF+), low-end SaaS (software < $10B, margin < 10%), labor-intensive services
(capex < 3%, margin < 8%), commodity producers / refiners / semis / defense by industry code, else sector default;
ObjectStore `metadron/subsector_map.json` may pin a name (26 sub-sectors).

**Top-down** (Graham Part I) from the universe aggregate: earnings yield Σ earnings / Σ cap, ERP = EY − 10y
(10y from the macro agent, default 4.2%), max investment P/E by rate (25/20/16/14/10/8x), speculative %,
valuation regime. The aggregate EY is the forward-Rm valuation anchor when the macro agent sends none.

**Bottom-up** (per name): normalized EPS (annual samples of trailing EPS, up to 10y), Graham number, P/E×P/B,
NCAV, ROIC (Morningstar) − WACC, ROE + DuPont, owner earnings, accruals (FCF/NI), 5-method intrinsic value
(median) → margin of safety, 8-test investment grade, composite 0–100 (MoS 25, earnings quality 20, balance
sheet 15, ROIC 15, coverage 10, valuation 10, top-down 5), quality tier A ≥ 80 … G < 30.

**Capital structure** (architecture point 9, free-data part): Altman Z, Merton 1y PD (equity vol from the
universe price feed), interest coverage, debt/EBITDA, debt/equity, current ratio, FCF vs capex. Credit tier
A–F = Egan-Jones D/E + current ratio, downgraded for Z < 1.8 (2), PD > 10% (2) / > 2% (1), coverage < 1.5 (2)
/ < 3 (1), debt/EBITDA > 5 (1). Sleeve IG (A/B) · HY (C/D) · DISTRESSED (E/F). Feeds `rotation.is_solvent`.

**Fundamentals gate** (must pass; 40% weight in the step-5 decision matrix): IG/HY pass at composite ≥ 50;
DISTRESSED pass only as recovery plays (NCAV discount ≥ 0 or MoS ≥ 33%, PD ≤ 25%, no cash burn); blocked:
negative equity with cash burn, coverage < 1.5 with PD > 10%.

**Shortlist** (macro ↔ micro): gate passers ranked by gate score + 0.25 × sub-sector edge, split IG 40 : HY 10 :
DISTRESSED 10 of `shortlist_size` (60), held names always kept; a sub-sector with zero weight and negative
edge is vetoed (macro can veto, micro cannot override). Sub-sector mean gate scores flow back into the sector
view as its `fundamentals` input. Outputs: charts *Data health* (members vs 1500, scored, missing fields,
gate passes, shortlist; alert below 90% coverage or > 20% missing) and *Valuation*; ObjectStore
`metadron/universe_scores.json` (top 100 with full analysis + reasons, theme watch list). Shortlisted names
are subscribed for step 5; nothing is traded from the shortlist yet.

### Intraday heartbeat (code: `algorithm/schedule.py`) — trades where liquidity is deepest

| ET | Job |
|----|-----|
| 09:31 | risk-open: hard stops + position protection only (no entries in the opening auction noise) |
| 09:50 | research: regime, forward Rm, corridor L, sector view, GICS scores, theme plan, decision matrix, option plans |
| 10:00 | core: MES overlay; equities trade 40% of each delta |
| 10:15 → 10:45 · 14:15 → 14:45 | options windows: subscribe candidates, then limit orders at mid + 0.3 × half-spread (option spreads are widest at the open and into the close) |
| every 30 min 10:00–15:30 | risk: daily-loss breaker (−3% → no entries, overlay halved), position stops, option exits, event hedges |
| 15:30 | expiry: close options ≤ 2 DTE, beta true-up |
| 15:45 | closing auction: the remaining 60% of equity deltas as market-on-close (deepest print of the day) |

### Decision matrix, sizing, overlay (code: `algorithm/decision.py`, `main.py`) — full Kelly (owner)

Gates: Fundamentals 40% (must pass) · Flow 20% (dollar-volume trend + 21d strength vs SPY) · Macro 20% (sub-sector
edge + regime fit of the name's nature; < 0.3 vetoes) · Momentum 20% (12-1, trend, chart patterns). Trade when
composite ≥ 0.55, spread ≤ 30 bps, P(profit over 21d) ≥ 52% with α > 0. Expected alpha α = composite-implied
0.40·(c − 0.5) blended with the models by their scorecard weights. Size = **full Kelly** α/σε² (k = 1), ≤ 20% NAV,
then the allocation buckets (below), each single-name sleeve split across the GICS sector budgets (unused sector
room spills to favoured sectors up to 30% of the sleeve), then scaled into the 2.5% stock-specific VaR budget.
Names with a catalyst (event ≤ 21 days or an 8-K material agreement) go to the EVENT bucket. ETF buckets:
TLTW / cash-flow 15% (income ETFs TLTW, JEPI, JEPQ, DIVO; GLD/DBC take ½ when stocks and bonds move together, ⅓ in
GOLD/CASH) and FI / macro 5% (TLT/IEF when bonds hedge, else SHY). **MES overlay = corridor L − Σ wᵢβᵢ**, capped at ±2.0× NAV (owner; so below Rm 7% the sleeves are
beta-hedged to zero, and above it total market beta ≈ sleeves' ≈ 0.7 + ≤ 2.0 futures), scaled so book CVaR95 ≤ 1.5 × regime VaR ceiling, halved by the breaker, then portfolio hard stops
(−4% halve, −8% flat; re-armed after 5 days flat).

### Feed guard + live data (code: `algorithm/feed_guard.py`; owner threat model: a delayed, cut or altered feed)

Phase 1: IBKR **paper** account for execution (separate IBKR username for QuantConnect); live prices from Tradier
(token) or QuantConnect's feed. Cross-feed reference = the S&P 500 index (SPX) vs SPY; phase 2 adds MES (CME, IBKR
feed) as the reference.

| Defence | Rule |
|---------|------|
| Staleness | SPY and MES last bar ≤ 5 min, held names ≤ 15 min (daily backtests: multi-day gaps only) |
| Cross-feed | reference/SPY ratio within 0.5% of its rolling median (phase 1 SPX index, phase 2 MES); a one-sided > 3% jump between checks |
| Halt | any failure → no new entries, no leverage increases, no options entries, no market-on-close; reductions only, collared; clears after 2 clean checks |
| Collars | every stock / ETF / futures order is a limit order within 30 bps (futures 20 bps) of the last trusted price |
| Broker-side stops | resting GTC stops at IBKR: long stocks at peak −10%, MES at the adverse move costing ≈ 8% NAV (2–10%); they execute even if the feed or the algorithm is cut. A name's stop is cancelled before any trade on it and re-placed once no order is working (never stop + sell → short); re-placed only when the level moves > 2% |
| Out-of-band audit | risk officer compares the algorithm's SPY print with an independent close (Stooq) and reports halts; failure → e-mail alert |

### Options, protection, tail hedges (code: `algorithm/options.py`, `protection.py`)

Black-Scholes pricing / greeks / implied vol; contract = closest |Δ| to target inside the DTE window with spread
≤ 10% of mid. Gate under the real-world drift: P(profit at expiry) ≥ 25% and expected P&L > 0 (hedges and straddles
exempt). Size = Kelly on premium, capped at 10% of NAV in total premium (hedges included) and per credit sleeve
IG 4 · HY 4 · distressed 2; **1-contract exception**: if one contract costs more
than the Kelly budget it buys exactly 1 contract when it passes the gate and costs ≤ 2% NAV. Uses: escalation oil
calls, LEAPS / LEAPS call spreads on solvent crashed names, straddles, alpha calls on the 5 strongest names only
when implied ≤ realised vol, event tail hedges (≥ 3% positions with an event ≤ 5 days: ~30Δ put, ≤ 0.3% NAV).
One open position per (ticker, kind, side). Protection: position −6% from peak → trim half (once per peak), −10% →
close; long options close at ≤ 2 DTE, −50% premium stop (not LEAPS), LEAPS rolled under 90 DTE; hedges and
straddles closed after the event.

### Alpha models (code: `algorithm/alpha.py`) — free / open source, inside the algorithm

Walk-forward 21-day return model (ridge + gradient boosting, weekly samples, refit every 4 weeks), 4-classifier
quality ensemble (Gaussian NB, gradient boosting, random forest, XGBoost) for P(top tercile), chart patterns
(52-week breakout on volume, golden cross, oversold bounce, 200-day breakdown), Kronos forecasts (CPU, weekly,
`agents/kronos_agent.py`). Scorecard: rank IC per model on matured forecasts; weight ∝ max(IC, 0); unproven
models start at 0; IC ≤ 0 four periods running → 0 (ablation rule).

### Agents, reconciliation, hosting (code: `agents/`, `.github/workflows/`, `vault/`; setup: `SETUP.md`)

Roster (`agents/roster.yaml`, all propose-only, Claude via `agents/llm.py` with server-side refusal fallback):
macro-analyst (FRED + GMTF + scenarios) and events-analyst (SEC 8-K flags → gate penalties, earnings dates →
hedges/straddles, escalation flag) pre-market; forecaster (Kronos) weekly; risk-officer (QuantConnect vs IBKR Flex
reconciliation + risk checks) and journalist (Obsidian notes) post-close. Hosting is GitHub Actions + QuantConnect
cloud: no server; a failed job e-mails the owner; going live with a new version is one owner click.

### Smart Beta v1 — corridor derivation (superseded by v2, kept for reference)

Chart: `docs/charts/beta_gamma_smooth.png` (script: `docs/charts/beta_gamma_smooth.py`).

**Labels (use these names everywhere — code, docs, dashboards):**

| Symbol | Name | Meaning |
|--------|------|---------|
| Rm | expected market return | expected annual S&P 500 return; the corridor is Rm ∈ [7%, 12%] (S&P historical earnings range) |
| β(Rm) | corridor beta | target exposure identifier, a smooth S-curve in Rm |
| β_L, β_H | corridor floor / ceiling | β_L = −0.029 at Rm = 7%, β_H = 0.425 at Rm = 12% |
| Δβ | corridor span | β_H − β_L = 0.454 |
| γ(Rm) | corridor gamma | **γ = dβ/dRm**, the rate at which beta changes with Rm (as option gamma is the slope of delta). A flattened bell. |
| λ | risk aversion | the Merton/CRRA coefficient (NOT called gamma, to avoid the clash with corridor gamma) |
| h | hurdle | Rm at which β = 0 |
| σ_std | standard volatility | 15% thesis standard |
| M | execution multiplier | 4.7 — converts β into portfolio leverage |

**1. Endpoints (Merton, Brownian motion).** Under dS/S = μ dt + σ dW a CRRA investor holds
β = (Rm − h)/(λσ²). The two corridor endpoints pin λσ² and h:

```
λσ² = (12% − 7%) / (β_H − β_L) = 0.05 / 0.454 = 0.1101   → λ ≈ 4.89 at σ = 15%
h   = 7% − β_L · λσ²          = 7% + 0.029 × 0.1101 = 7.32%
check: β(7%) = (0.07 − 0.0732)/0.1101 = −0.029 ; β(12%) = (0.12 − 0.0732)/0.1101 = 0.425
```
Only the product λσ² is identified (λ = 0.1101/σ²: 7.65 at 12%, 4.89 at 15%, 3.40 at 18%,
2.75 at 20%). Equivalently β = 1/λ ≈ 20% of full Kelly at σ = 15%.

**2. Smooth beta curve (the shape).** Inside the corridor β follows a smoothstep S-curve; outside
it is flat:

```
t     = clip((Rm − 7%) / 5%, 0, 1)
β(Rm) = β_L + Δβ · (3t² − 2t³)
```
Values: β(7%) = −0.029, β(9.5%) = 0.198, β(12%) = 0.425; flat beyond both edges.

**3. Corridor gamma (the flattened bell).**

```
γ(Rm) = dβ/dRm = Δβ · 6t(1 − t) / 5%      for 7% < Rm < 12%,  0 outside
```
- zero at and beyond 7% and 12% (exposure stops changing at the corridor edges)
- peak at Rm = 9.5%: γ_max = 1.5 · Δβ / 5% = 13.62 per unit Rm = **0.136 per 1% of Rm**
- area under γ over the corridor = Δβ = **0.454**
- the old linear ramp has a flat-box gamma of 0.091 per 1% with jumps at 7% and 12% —
  superseded by the smooth curve.

**4. From beta to leverage.**

```
leverage(Rm) = β(Rm) × M × vol_scale(σ_m)      M = 4.7
             → −0.136 at Rm ≤ 7% (hedge floor, BETA_INV) … 2.00 at Rm ≥ 12% (BETA_MAX)
vol_scale    = σ_std / σ_m  (current, linear)   or  (σ_std / σ_m)²  (Merton-consistent) — OPEN
```
Then: regime cap (or burst cap when momentum confirms) → hard caps (VaR95 ≤ 1.5% NAV,
gross-leverage gate). M ≈ λ, so leverage ≈ full Kelly at σ = 15%; at higher true volatility a
fixed M = 4.7 is over-Kelly (full-Kelly leverage at 12%: 2.08 at σ 15%, 1.44 at 18%, 1.17 at 20%).

**5. Inputs each cycle.** Rm_adj = realised SPY drift + macro adjustment (GMTF tension, HMM regime
probabilities) + Kronos index forecast → β(Rm_adj), γ(Rm_adj) → leverage budget L* (gross, split
across sleeves by Kelly risk-adjusted alpha within sleeve caps) → SPY / index-option overlay closes
the residual net-beta gap. Rebalance only when |Δβ| > 0.05 (throttle).

**Known defects in the legacy engine (fixed in the LEAN build):** linear ramp instead of the smooth
curve; constants duplicated in 3 places; AlphaBetaUnleashed calls a non-existent method and uses a
fallback that saturates at 2.0 near Rm ≈ 7.4%, and its output is never executed; sleeve deployment
ignores the corridor; TRENDING cap 0.65 makes the top of the corridor inert; beta burst never applied.
ARCHITECTURE_DNA.md's "linear(0 → BETA_MAX)" description is wrong.

## Target Architecture — LEAN / QuantConnect migration (agreed design)

Stack: QuantConnect cloud (LEAN, dashboard, backtests, live node) → IBKR (paper now,
live later; QC data for signals, IBKR for execution + cross-check). Agents: Claude
Managed Agents (Anthropic Console = agent dashboard). Readable control room: Obsidian
vault in this repo (agents write notes; Dataview tables; graph view). Kept from the
engine: beta corridor, options engine, Markov HMM, GMTF, MetadronCube, pattern
recognition, security analysis, ML ensemble, decision gates, MC gate, protection,
event hedges. Added: Kronos (forecasting). Reference only: freqtrade (GPLv3 — copy
design, never code): FreqAI-style retraining, protections, dry-run first, chat control.

1. **Forecasting → reallocation.** Kronos + walk-forward ML give per-name return and
   vol forecasts → (a) alpha model expected return, (b) forecast vol in Kelly, vol
   scaling and the MC gate, (c) index forecast into Rm_adj → β*, L*. Rebalance only
   when a weight or the beta gap moves > 0.05 (gamma throttle).
2. **Fundamentals → selection.** SecurityAnalysisEngine (CAPE/ERP/max P/E top-down;
   Graham number, NCAV, MoS ≥ 33%, ROIC−WACC, DuPont, owner earnings, 8-test grade,
   5-method IV bottom-up) + ML quality tier + credit score → Fundamentals gate (40%,
   must pass). Data: QC Morningstar fundamentals + SEC filings.
3. **Universe data.** QC point-in-time S&P 1500 universe (daily refresh); prices/options
   from QC, cross-checked vs IBKR; macro from FRED. "Data health" chart: names scanned
   vs expected, stale/missing data, failed cross-checks, skipped chains → alerts.
4. **Position monitoring.** QC live dashboard; IBKR app/TWS (independent check); daily
   reconciliation by the risk-officer agent; Obsidian daily note per position (why
   held, gate scores, drawdown, protection, upcoming events/hedges); phone alerts; Jarvis.
5. **Profitability gaps.** (a) live paper P&L vs backtest expectation; (b) per-model
   logged forecasts scored (hit rate, IC) + weekly ablation backtests — zero
   contribution 4 weeks → down-weight then remove; (c) per-agent scorecard (accuracy,
   P&L of approved changes, cost, rejection rate) → DIRECTOR…RECRUIT, demote/pause.
6. **Dynamic tilt under hard caps.** Adaptive layer (gate/tier weights, corridor slope
   within band, Kelly vol adj, thresholds) retuned weekly by walk-forward optimisation
   inside fixed min/max, adopted only if it beats current out of sample; regime shifts it
   too. Hard caps (VaR95 1.5%, CVaR 2.5%, position 20%, regime leverage/beta caps,
   drawdown stops −6/−10% position, −4/−8% portfolio, daily-loss breaker) live in a
   locked config — owner-only changes; agents can only propose.
7. **ML models.** All free/open source, run inside the QC algorithm: walk-forward alpha
   (linear + XGBoost, core), 4-classifier quality tier (NB, GBM, RF, XGBoost), HMM,
   Kronos-small (PyTorch; fine-tuned monthly off-platform, weights in QC Object
   Store). The first ablation decides which survive.
8. **Macro ↔ micro.** Macro (GMTF, HMM, Fed liquidity, curve, credit) sets regime, Rm_adj,
   β*, leverage budget, sector and sleeve weights → micro (fundamentals, ML, Kronos,
   patterns, events) picks names inside that budget → decision matrix needs both
   (Fundamentals 40 / Flow 20 / Macro 20 / Momentum 20) → Kelly size capped by regime
   and VaR → corridor overlay closes the beta gap. Macro can shrink/veto; micro cannot
   override a macro cap.
9. **Capital structure.** New monitor for every held/universe name: HY/IG OAS level +
   change (FRED ICE BofA series), debt/EBITDA, interest coverage, 1–3y maturity wall,
   FCF vs capex burn, leases (filings), new facilities / private credit / secured debt
   (8-K, events agent), Merton equity-implied default probability and credit spread vs
   market spread. Deteriorating structure → Fundamentals-gate penalty or block, tighter
   protection, event tail hedge. Issuer-level bond/CDS data = paid vendor, only if the
   free signals prove out in backtests. (Current "credit_spread" is an HY−IG ETF
   volatility difference, not a spread — replace.)

## Allocation reference

Code: `engine/allocation/allocation_engine.py` (`AllocationRules`) — THE reference; the algorithm uses these buckets
(`algorithm/decision.py` SLEEVES):

| Bucket | % NAV | Instruments |
|--------|-------|-------------|
| IG equities | 40 | single names, credit tier A/B, split across GICS sector budgets |
| HY equities | 10 | single names, C/D |
| Distressed equity | 10 | single names, E/F recovery plays |
| Event-driven / CVR | 10 | single names with catalysts |
| TLTW / cash-flow ETFs | 15 | income ETFs (+ GLD/DBC when bonds stop hedging) |
| FI / macro | 5 | TLT / IEF / SHY |
| Options premium | 10 | long options on the single names (+ hedges); per credit sleeve IG 4 · HY 4 · distressed 2 |
| Futures | **PHASE 2 — off** | MES overlay = corridor L − sleeve beta, capped at ±2.0× NAV notional; code kept (`main._set_beta_leverage`, `risk_budget.MAX_FUTURES_LEVERAGE`, futures stops in `feed_guard`). Enable with `beta_instrument=MES`. Phase 1: corridor L is computed, plotted and reported (`overlay_phase2`) but not traded; market beta comes from the stock sleeves only |

Owner update (Nov 2026): MES futures overlay moved to PHASE 2 (off; phase-2 cap 2.0× NAV); money market removed; options = 10% of NAV in premium (was 25% notional funded by the
8% margin bucket); no IG cut for futures margin. Mirrored in `AllocationRules`.

Older summary table (pre-code, superseded where it differs):


| Sleeve | Allocation | Credit Tier | Notes |
|--------|-----------|-------------|-------|
| IG Equities | 40% | A/B (IG) | All caps — no mega cap restriction |
| Options | 25% | Mixed | 10% IG + 10% HY + 5% Distressed |
| Bond/Commodity ETFs | 10% | — | TLT, GLD, USO, HYG, LQD, etc. |
| HY Equities | 10% | C/D (HY) | BB-B rated, leveraged |
| Distressed Equity | 10% | E/F | Fallen angels, recovery plays |
| Cash (Dry Powder) | 5% | — | Never deployed, buying power reserve |

Deploy 95% of NAV + leverage. Only 5% remains as dry powder.

## MetadronCube

### 10-Layer Intelligence

| Layer | Name | Function |
|-------|------|----------|
| 0 | FedPlumbingLayer | SOFR, reserves, TGA, ON-RRP |
| 1 | LiquidityTensor | L(t) in [-1,+1] |
| 2 | ReserveFlowKernel | ΔReserves → ΔEquity/Credit |
| Risk | RiskStateModel | R(t) in [0,1] |
| Flow | CapitalFlowModel | F(t) sector rotation |
| 4 | RegimeEngine (HMM) | TRENDING/RANGE/STRESS/CRASH |
| Gate | GateZAllocator | 5-sleeve allocation |
| Entry | GateLogic | 4-gate entry scoring |
| Kill | KillSwitch | Auto-derisking triggers |
| Cal | FCLPLoop | Daily recalibration |

### Regime Parameters (95% Alpha Target)

| Regime | Leverage | Beta Cap | Beta Burst | Crash Floor |
|--------|----------|----------|------------|-------------|
| TRENDING | 3.0x | 0.65 | 0.70 | ≥+25% |
| RANGE | 2.5x | 0.45 | 0.55 | ≥+25% |
| STRESS | 1.5x | 0.15 | 0.20 | ≥+25% |
| CRASH | 0.8x | -0.20 | -0.10 | ≥+25% |

### MLVoteEnsemble (10 Tiers)

| Tier | Name | Weight | Source |
|------|------|--------|--------|
| T1 | Neural | 1.0 | DRL/FinRL bridge |
| T2 | Momentum | 1.2 | Technical signals |
| T3 | Vol Regime | 0.8 | Volatility regime |
| T4 | Monte Carlo | 0.9 | MC simulation |
| T5 | Quality | 1.1 | Quality ranker |
| T6 | Social | 1.0 | Agent sim consensus |
| T7 | Distress | 0.9 | Distressed asset engine |
| T8 | Event | 1.0 | Event-driven engine |
| T9 | CVR | 0.7 | CVR engine |
| T10 | Credit Quality | 0.9 | UniverseClassifier/CreditQualityClassifier |
