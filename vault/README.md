# Metadron vault (Obsidian)

Open this folder as an Obsidian vault. Install two community plugins: **Obsidian Git** (auto-pull every
30 minutes) and **Dataview**. The post-close job writes and commits these notes every trading day:

| Note | What it shows |
|------|---------------|
| `Dashboard.md` | one page: latest day, holdings table, alerts |
| `Daily/YYYY-MM-DD.md` | regime, Rm and its components, corridor leverage, MES overlay, GICS sector scores, model weights |
| `Positions/<TICKER>.md` | why each position is held (gate scores, reasons), weight, protection peak, sub-sector, credit tier |
| `Risk/YYYY-MM-DD.md` | reconciliation QuantConnect vs IBKR, risk checks, alerts |
| `Universe.md` / `Market.md` | data health + shortlist; macro question tree, overweight / avoid with drivers |

Notes are generated — edit `Templates/` or `agents/vault_sync.py`, not the generated files.
