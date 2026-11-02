# Market reasoning — how Metadron agents think about the market

This is the owner's trading framework. Every research agent (macro analyst, events analyst, quant
reviewer, risk officer, Jarvis) follows it. Agents reason and **propose**; the quant layer
(`algorithm/macro_regime.py`, `algorithm/sector_rotation.py`) turns proposals into sized tilts, and the
risk budget and hard caps always apply. Agents never place orders.

## 1. Start from a thesis, map the value chain

Example: *"AI has a great future → AI is a great investment."* Then ask:
- **Where is the supply chain?** Demand (platforms / hyperscalers) → compute (chip design) → fab tools
  (semi equipment) → fabs (foundries, incl. **regional players** — Taiwan, Netherlands, Korea, US) →
  memory → networking → data centers → power & grid.
- **What are the major drivers?** Capex budgets, power availability, chip supply, financing.
- **Who is financing it, and how?** Cash-funded platforms vs debt / private-credit funded neoclouds and
  capex-heavy pivots. The financing structure decides who survives a downturn.
- **Who is at risk?** Business models AI substitutes (low-end SaaS, labor-intensive services).
- **Who has pivoted?** Companies re-positioning into the chain (e.g. into network infrastructure).

The map lives in `algorithm/market_map.py` (`chain("AI")`). Propose additions there.

## 2. Walk the macro question tree (in order)

1. Is **inflation within mandate** (2% ± 1)? 2. Is the economy **hot / resilient** (data vs expectation)?
3. Is **labor tight**? 4. Is labor **gapping** (unemployment ≥ 0.5 pt above its 12-month low)?
5. Is there **room for stimulus**? 6. Is **productivity** rising? 7. Are **real wages** rising?
   → productivity ↑ + wages flat + employment retained ⇒ **skills are being substituted** — *which skills,
   which industries?* (AI substitution: pressure on low-end SaaS, labor-intensive services.)
8. How is **system liquidity** (Fed balance sheet − TGA − RRP)? 9. How are **corporate bonds**
   (HY / IG spreads)? Widening spreads expose debt-funded names first.
10. Is **geopolitics** escalating? Which names are the *current* defensives for this paradigm?
11. Where is **investor sentiment** (euphoria / fear are contrarian signals)?

`macro_regime.diagnose()` runs this tree on the data; agents add what data can't see (policy intent,
consensus expectations, legal fights, supply-chain news).

## 3. Know where each sector sits

| Nature | Sectors | Behaviour |
|--------|---------|-----------|
| **Cyclical** | materials, commodities, industrials, financials, consumer discretionary, real estate | lead in recoveries, suffer in slowdowns; rate- and credit-sensitive |
| **Secular** | IT, communication, healthcare | driven by long-run adoption (AI) more than the cycle; rate-sensitive valuations |
| **Defensive** | staples, utilities, energy (+ defense in escalations, pharma) | hold up when growth or sentiment breaks |

Split secular names into **infrastructure** (power, data centers, fabs, networking) vs **AI
beneficiaries / at-risk**. Infrastructure behaves partly cyclically (capex, credit, rates).

## 4. Cyclical vs secular interaction, and stocks vs bonds

- A secular trend (AI) can run *through* a cyclical slowdown — but its **financing** is cyclical:
  when credit tightens, debt-funded secular names fall like cyclicals.
- **Stock–bond correlation** decides the hedge (`macro_regime.stock_bond_regime`):
  *negative* (growth-driven) → Treasuries hedge equities; *positive* (inflation-driven) → bonds fall with
  stocks, hedge with commodities, gold or cash.
- Distinguish **correlation from causation**: state the mechanism ("spreads widened *because* private
  credit marks rose, *which* raises refinancing costs for neoclouds") before acting on a co-movement.

## 5. Write what-if scenarios (base / up / down)

For each live question, give 2–4 scenarios with probabilities that sum to 1 and the **factor shocks**
each implies (changes to factor states in [−1, +1]: GROWTH, INFLATION, LABOR, WAGES, PRODUCTIVITY, RATES,
LIQUIDITY, CREDIT, GEOPOLITICS, SENTIMENT, AI_CAPEX, COMMODITIES). Save as JSON to the QuantConnect
ObjectStore key `metadron/scenarios.json`:

```json
{
  "source": "macro-analyst",
  "as_of": "2026-11-02",
  "scenarios": [
    {"name": "base: AI capex continues, credit contained", "probability": 0.55,
     "shocks": {"AI_CAPEX": 0.2, "CREDIT": -0.1},
     "narrative": "Hyperscaler capex guides up; neocloud spreads stabilise."},
    {"name": "down: private-credit squeeze", "probability": 0.30,
     "shocks": {"CREDIT": -0.7, "LIQUIDITY": -0.3, "AI_CAPEX": -0.3, "SENTIMENT": -0.4},
     "narrative": "Refinancing walls hit debt-funded AI infra; spreads +100 bp."},
    {"name": "up: escalation lifts energy and defense", "probability": 0.15,
     "shocks": {"GEOPOLITICS": 0.8, "COMMODITIES": 0.5, "INFLATION": 0.3},
     "narrative": "Supply shock; oil +20%."}
  ]
}
```

The system scores every sub-sector under each scenario: **edge** = probability-weighted score,
**downside** = worst-scenario score. For names with an edge, *always state the downside*.

## 6. Fundamentals and events

Per sub-sector (or name) report `fundamentals` in [−1, +1]: earnings and revenue revisions breadth,
margin trend, balance-sheet strain, **legal fights** (antitrust, export controls, IP), and upcoming
catalysts (earnings dates feed the event tail hedges and straddles).

## 7. How rotation follows the news

New data or news → update the scenario probabilities / shocks → the system re-scores → tilts move only
when they change by more than the throttle. Agents log *why* the probabilities changed; the quant
reviewer later scores those calls (hit rate, P&L of changes) — that is how agents are measured.

## Rules

- Propose, never execute. Hard caps and the risk budget are owner-only.
- Themes and sub-sectors are reasoning profiles detected from fundamentals (`algorithm/profiles.py`): capex
  intensity, leverage, interest cover, cash burn, margins, size. Names you cite are illustrations of a pattern,
  never a watchlist — propose profile changes (thresholds, new themes with their reasoning), not tickers.
- Every claim gets a mechanism, a probability and a downside.
