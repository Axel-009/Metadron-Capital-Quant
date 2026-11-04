# Setup — one afternoon, then hands-off

Everything runs on managed services; there is no server of yours to keep alive.

| Piece | Runs on | Maintenance |
|-------|---------|-------------|
| Trading algorithm (`algorithm/`) | QuantConnect cloud live node → IBKR paper account | none; QC restarts and reconnects the node |
| Agents (macro, events, Kronos, risk officer, vault) | GitHub Actions on a schedule (`.github/workflows/`) | none; a failed job e-mails you |
| Deploys + backtests | GitHub Actions on every push to `main` touching `algorithm/` | none |
| Control room | Obsidian vault in this repo (`vault/`), pulled by the Obsidian Git plugin | none |

## 1. GitHub repo (5 min)
1. Create a **private** repo (e.g. `metadron-capital`) and push the contents of this folder to it as the root
   (the workflows must sit at `.github/workflows/` of the repo).
2. Settings → Actions → General → Workflow permissions: **Read and write** (the post-close job commits the vault).

## 2. QuantConnect + IBKR (20 min)
1. quantconnect.com → create an organization on a plan that allows live trading and ObjectStore
   (check the current pricing page; a live node for one algorithm is the main cost).
2. Create an empty **Python** project named `Metadron` → note its **project id** (in the URL).
3. Account → API: note **User ID** and **API token**; Organization page: note the **organization id**.
4. Data: make sure US equities (minute), US equity options, CME futures (MES) and Morningstar fundamentals are
   included for live and backtests.
5. IBKR: in Client Portal, enable market data for US stocks, US options and CME futures on the **paper** account.
6. IBKR Flex (for daily reconciliation): Reporting → Flex Queries → new **Activity Flex Query** with section
   *Open Positions* (all fields), format XML, period *Last Business Day* → note the **query id**; then
   Flex Web Service → generate a **token**.

## 3. Other keys (5 min)
- **Anthropic** API key (console.anthropic.com) — agents use `claude-opus-5-5` with web search.
- **FRED** API key (free, fredaccount.stlouisfed.org).
- **SEC user agent**: your name and e-mail, e.g. `Jane Doe jane@example.com` (SEC fair-access policy).

## 4. GitHub secrets (5 min)
Repo → Settings → Secrets and variables → Actions → *New repository secret*:

| Secret | Value |
|--------|-------|
| `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORG_ID`, `QC_PROJECT_ID` | step 2 |
| `ANTHROPIC_API_KEY`, `FRED_API_KEY`, `SEC_USER_AGENT` | step 3 |
| `IBKR_FLEX_TOKEN`, `IBKR_FLEX_QUERY_ID` | step 2.6 |

## 5. First deploy and backtest (10 min)
1. Actions → **deploy-to-quantconnect** → *Run workflow*. It runs the tests, pushes `algorithm/` to your
   QuantConnect project, compiles and starts a backtest.
2. Open the project in QuantConnect → the backtest shows the charts: *Rm vs corridor, Regime, Corridor,
   Leverage, Risk, Themes, Market view, Equity mix, Data health, Valuation, Decisions, Model weights*.
3. Actions → **agents-premarket** → *Run workflow* once; check that `macro_inputs.json`, `scenarios.json` and
   `events.json` appear in QuantConnect → Organization → Object Store under `metadron/`.

## 6. Go live on paper (5 min, the only manual trading step)
QuantConnect project → **Deploy Live** → brokerage *Interactive Brokers* → your **paper** account → node.
- Use a **separate IBKR username** for QuantConnect (IBKR account settings → users), so your own TWS / app logins
  never disconnect the algorithm.
- Data providers: **Tradier** (paste a Tradier API token; needs an open Tradier account) for stocks and options,
  **Interactive Brokers** for the MES futures. If QuantConnect does not offer that combination, use QuantConnect's
  own feed for stocks / options instead of Tradier.
- IBKR asks for a 2FA re-login about once a week; approve it in the IBKR mobile app (resting stops stay at IBKR).
After that, a new algorithm version goes live only when you press *Deploy Live* again (design rule: you approve
what trades). Deploys from GitHub only update the code and run a backtest.

## 7. Obsidian (5 min)
Clone the repo on your computer (GitHub Desktop is fine), open the `vault/` folder as a vault, install the
community plugins **Obsidian Git** (set *auto pull interval* to 30 min) and **Dataview**. Open `Dashboard.md`.

## Daily rhythm (automatic)
| UTC (New York summer) | What happens |
|-----------------------|--------------|
| 12:20 (08:20) | macro + events agents write the day's inputs |
| 13:31 → 20:00 (09:31 → 16:00) | the algorithm's heartbeat: risk-open, research 09:50, core 10:00, options windows 10:15–10:45 and 14:15–14:45, risk every 30 min, expiry 15:30, closing auction 15:45 |
| 21:40 (17:40) | vault notes + reconciliation; failed job = e-mail alert |
| Sat 10:00 | Kronos forecasts (CPU) |

## First run checks (once)
The QuantConnect REST calls (ObjectStore download, project files) and Morningstar field paths follow the public
docs but could only be tested against a stand-in here. After the first runs, check:
- the backtest log has no `DATA-HEALTH alert` about missing fields (if it has, field paths in
  `algorithm/universe.py` `FIELDS` need a tweak);
- the post-close job wrote `vault/Daily/<date>.md` (if empty, ObjectStore download in `agents/qc_api.py`);
- `vault/Risk/<date>.md` shows the reconciliation (if "not run", IBKR Flex secrets).

## Costs to expect (estimates — check each provider's current pricing)
- QuantConnect: plan + one live node (the largest item).
- Anthropic: two agent calls per trading day with web search — roughly $10–30 / month at Opus 5.5 prices.
- GitHub Actions: ~10 min per weekday + ~60 min weekly for Kronos — inside the free minutes of most plans.
- FRED, SEC EDGAR, IBKR Flex, Obsidian: free.

## Turning things off
- Pause all agents: Actions → each workflow → *Disable workflow*.
- Stop trading: QuantConnect → live algorithm → *Stop* (positions stay at IBKR; liquidate there if needed).
- Equity or option trading only: set `trade_equities` / `trade_options` to `false` in `algorithm/config.json`.
