"""Vault sync — turns the algorithm's ObjectStore reports into Obsidian notes (the readable control room).

After the close (GitHub Actions commits the vault; Obsidian Git pulls it on the owner's machine/phone).

    vault/Daily/YYYY-MM-DD.md        regime, Rm, leverage, sector scores, model weights, approvals, alerts
    vault/Positions/<TICKER>.md      one note per holding: why held (gates, reasons), weight, protection peak,
                                     sub-sector, credit tier, option kind — frontmatter for Dataview
    vault/Universe.md                data health, valuation, shortlist by sleeve
    vault/Market.md                  macro findings, overweight / avoid sub-sectors with drivers, GICS scores
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
KEYS = ["metadron/state.json", "metadron/universe_scores.json", "metadron/sector_views.json", "metadron/events.json"]


def _j(raw: Optional[str]) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _fm(d: dict) -> str:
    return "---\n" + "\n".join(f"{k}: {json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in d.items()) + "\n---\n"


def daily(state: dict, today: dt.date) -> str:
    st = state.get("state") or {}
    pos = state.get("positions") or {}
    scores = sorted((state.get("sector_scores") or {}).items(), key=lambda x: -x[1])
    lines = [_fm({"date": str(today), "type": "daily", "regime": st.get("regime"), "zone": st.get("zone"),
                  "rm": round(st.get("rm", 0) or 0, 4), "leverage": round(st.get("L", 0) or 0, 2),
                  "overlay": round(st.get("beta_overlay", 0) or 0, 2), "nav": round(state.get("nav", 0) or 0),
                  "positions": len(pos)}),
             f"# {today}", "",
             f"**Regime** {st.get('regime')} ({st.get('method')}) · **zone** {st.get('zone')} · **Rm** {st.get('rm', 0):.2%} "
             f"· **σ** {st.get('sigma', 0):.1%} · **corridor L** {st.get('L', 0):.2f} (cap {st.get('cap', 0):.2f}"
             f"{', burst' if st.get('burst') else ''}) · **MES overlay** {st.get('beta_overlay', 0):.2f} "
             f"(sleeve beta {st.get('sleeve_beta', 0):.2f}) · **bias** {st.get('bias')} · **escalation** {st.get('escalation', 0):.2f}",
             "", "## Rm components", "", "| component | value |", "|---|---|"]
    lines += [f"| {k} | {v:+.2%} |" for k, v in (st.get("components") or {}).items() if isinstance(v, (int, float)) and v == v]
    lines += ["", "## GICS sector scores", "", "| sector ETF | score |", "|---|---|"]
    lines += [f"| {e} | {s:+.2f} |" for e, s in scores]
    lines += ["", "## Model weights (scorecard)", ""]
    lines += [f"- {m}: {w:.0%}" for m, w in (state.get("model_weights") or {}).items()]
    lines += ["", "## Holdings", ""] + [f"- [[Positions/{t}|{t}]] {p.get('weight', 0):.1%}" for t, p in
                                        sorted(pos.items(), key=lambda x: -abs(x[1].get("weight", 0)))]
    return "\n".join(lines) + "\n"


def position(t: str, p: dict, today: dt.date) -> str:
    fm = {"ticker": t, "type": "position", "updated": str(today), "weight": round(p.get("weight", 0), 4),
          "target": p.get("target"), "subsector": p.get("subsector"), "credit_tier": p.get("credit_tier"),
          "option": p.get("option"), "unrealised_pct": round(p.get("unrealised_pct", 0) or 0, 4)}
    gates = p.get("gates") or {}
    lines = [_fm(fm), f"# {t}", "", "## Why held"]
    lines += [f"- {r}" for r in (p.get("reasons") or ["rotation / ETF sleeve / hedge"])]
    if gates:
        lines += ["", "## Gates", "", "| gate | score |", "|---|---|"] + [f"| {g} | {v:.2f} |" for g, v in gates.items()]
    lines += ["", f"Protection peak: {p.get('peak')} · quantity {p.get('quantity')}"]
    return "\n".join(lines) + "\n"


def universe(us: dict) -> str:
    h, td = us.get("health") or {}, us.get("top_down") or {}
    lines = [_fm({"type": "universe", "updated": us.get("time"), **{k: h.get(k) for k in ("members", "scored", "gate_pass", "shortlist")}}),
             "# Universe", "", f"Data health: {h}", "",
             f"Earnings yield {td.get('earnings_yield', 0):.2%} · ERP {td.get('erp', 0):+.2%} · market P/E "
             f"{td.get('market_pe', 0):.1f} vs max {td.get('max_pe', 0):.0f} · {td.get('regime')}", "", "## Shortlist"]
    for sleeve, names in (us.get("shortlist") or {}).items():
        lines.append(f"- **{sleeve}**: " + ", ".join(names))
    return "\n".join(lines) + "\n"


def market(sv: dict) -> str:
    lines = [_fm({"type": "market", "updated": sv.get("time"), "bias": (sv.get("bias") or {}).get("bias")}),
             "# Market view", "", f"Bias: {sv.get('bias')}", f"Stock–bond: {sv.get('stock_bond')}", "", "## Question tree"]
    lines += [f"- {f.get('question')} → **{f.get('answer')}** {f.get('tags') or ''}" for f in sv.get("findings") or []]
    for k in ("overweight", "avoid"):
        lines += ["", f"## {k.title()}"]
        lines += [f"- {v.get('subsector')}: edge {v.get('edge', 0):+.2f}, downside {v.get('downside', 0):+.2f} — "
                  f"{', '.join(v.get('drivers') or [])}" for v in sv.get(k) or []]
    return "\n".join(lines) + "\n"


def run(vault: str, files: Optional[Dict[str, str]] = None, today: Optional[dt.date] = None) -> int:
    today = today or dt.date.today()
    if files is None:
        import qc_api
        files = qc_api.object_get(KEYS)
    state, us, sv = (_j(files.get(k)) for k in KEYS[:3])
    written = 0
    for sub in ("Daily", "Positions"):
        os.makedirs(os.path.join(vault, sub), exist_ok=True)
    if state:
        with open(os.path.join(vault, "Daily", f"{today}.md"), "w", encoding="utf-8") as fh:
            fh.write(daily(state, today))
        written += 1
        for t, p in (state.get("positions") or {}).items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in t)
            with open(os.path.join(vault, "Positions", f"{safe}.md"), "w", encoding="utf-8") as fh:
                fh.write(position(t, p, today))
            written += 1
    if us:
        with open(os.path.join(vault, "Universe.md"), "w", encoding="utf-8") as fh:
            fh.write(universe(us))
        written += 1
    if sv:
        with open(os.path.join(vault, "Market.md"), "w", encoding="utf-8") as fh:
            fh.write(market(sv))
        written += 1
    print(f"vault: {written} notes")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ObjectStore → Obsidian vault")
    ap.add_argument("--vault", default=os.path.join(ROOT, "vault"))
    run(ap.parse_args().vault)
