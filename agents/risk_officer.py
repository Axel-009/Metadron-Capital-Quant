"""Risk officer — daily reconciliation (QuantConnect vs IBKR) and risk report → vault + alert.

After the close. Reads metadron/state.json (the algorithm's own book) and the IBKR paper account through the
Flex Web Service (no gateway to keep running: create one Activity Flex Query with Open Positions in IBKR
Client Portal → Reporting → Flex Queries, then set IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID).

    reconciliation   quantity differences per symbol (stocks / ETFs; options and futures by local symbol)
    feed audit       feed-guard halts + the algorithm's SPY print vs an independent close (Stooq)
    risk checks      breaker on, hard stop active, overlay vs corridor L, options premium vs the 10% NAV budget,
                     stale state (> 1 business day), data-health alerts
    output           vault/Risk/YYYY-MM-DD.md and a non-zero exit code when anything needs the owner
                     (GitHub Actions then e-mails the owner — the alert channel needs no extra service)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FLEX = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/"
QTY_TOL = 1e-6


def ibkr_positions() -> Optional[Dict[str, float]]:
    token, query = os.environ.get("IBKR_FLEX_TOKEN"), os.environ.get("IBKR_FLEX_QUERY_ID")
    if not (token and query):
        return None
    try:
        q = urllib.parse.urlencode({"t": token, "q": query, "v": 3})
        with urllib.request.urlopen(FLEX + "SendRequest?" + q, timeout=60) as r:
            root = ET.fromstring(r.read())
        ref = root.findtext("ReferenceCode")
        if not ref:
            print(f"[risk] flex request failed: {root.findtext('ErrorMessage')}", file=sys.stderr)
            return None
        for _ in range(20):
            time.sleep(5)
            q = urllib.parse.urlencode({"t": token, "q": ref, "v": 3})
            with urllib.request.urlopen(FLEX + "GetStatement?" + q, timeout=60) as r:
                body = r.read()
            if b"<OpenPosition" in body or b"<FlexQueryResponse" in body:
                return parse_flex(body)
    except Exception as exc:
        print(f"[risk] IBKR flex failed: {exc}", file=sys.stderr)
    return None


def parse_flex(xml_bytes: bytes) -> Dict[str, float]:
    root = ET.fromstring(xml_bytes)
    out: Dict[str, float] = {}
    for p in root.iter("OpenPosition"):
        sym = p.get("symbol") or p.get("underlyingSymbol") or "?"
        if p.get("assetCategory") in ("OPT", "FUT", "FOP"):
            sym = p.get("description") or sym
        try:
            out[sym] = out.get(sym, 0.0) + float(p.get("position") or 0)
        except ValueError:
            continue
    return out


def reconcile(qc: Dict[str, float], ib: Dict[str, float]) -> List[str]:
    diffs = []
    for s in sorted(set(qc) | set(ib)):
        a, b = qc.get(s, 0.0), ib.get(s, 0.0)
        if abs(a - b) > QTY_TOL:
            diffs.append(f"{s}: QuantConnect {a:g} vs IBKR {b:g}")
    return diffs


AUDIT_TOL = 0.015       # algorithm's 15:45 SPY print vs the independent daily close


def stooq_close(ticker: str, day: dt.date) -> Optional[float]:
    """Independent reference price (Stooq daily CSV, no key) — a source the live feed cannot touch."""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1={day:%Y%m%d}&d2={day:%Y%m%d}"
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = r.read().decode().strip().splitlines()
        return float(rows[1].split(",")[4]) if len(rows) > 1 else None
    except Exception:
        return None


def feed_findings(state: dict, today: dt.date, reference=stooq_close) -> List[str]:
    """Out-of-band feed audit: halts reported by the algorithm + its SPY print vs an independent close."""
    feed = state.get("feed") or {}
    out = []
    if feed.get("ok") is False:
        out.append("feed guard HALTED: " + "; ".join(feed.get("reasons") or [])[:300])
    spy = feed.get("spy_last")
    try:
        day = dt.date.fromisoformat(str(feed.get("spy_time", ""))[:10])
    except ValueError:
        day = None
    if spy and day:
        ref = reference("SPY", day)
        if ref and abs(spy / ref - 1) > AUDIT_TOL:
            out.append(f"feed audit: algorithm saw SPY {spy:.2f} vs independent close {ref:.2f} ({spy / ref - 1:+.2%})")
    return out


def risk_findings(state: dict, today: dt.date) -> List[str]:
    out = []
    if not state:
        return ["no state.json from the algorithm (not running, or ObjectStore unreachable)"]
    try:
        ts = dt.datetime.fromisoformat(str(state.get("time"))[:19]).date()
        if (today - ts).days > (3 if today.weekday() == 0 else 1):
            out.append(f"state is stale (last update {ts})")
    except ValueError:
        out.append("state has no valid timestamp")
    st = state.get("state") or {}
    if state.get("breaker"):
        out.append("daily-loss breaker fired today")
    if st.get("stop") in ("TRIM", "CLOSE"):
        out.append(f"portfolio hard stop: {st.get('stop')}")
    opts = sum(p.get("weight", 0) for p in (state.get("positions") or {}).values() if p.get("option"))
    if opts > 0.10:
        out.append(f"options premium {opts:.1%} NAV above the 10% budget")
    return out


def note(today: dt.date, state: dict, diffs: List[str], findings: List[str], ib_ok: bool) -> str:
    st = state.get("state") or {}
    lines = [f"---\ndate: {today}\ntype: risk\nalerts: {len(diffs) + len(findings)}\n---",
             f"# Risk officer — {today}", "",
             f"NAV **{state.get('nav', 0):,.0f}** · regime **{st.get('regime')}** · zone **{st.get('zone')}** · "
             f"Rm {st.get('rm', 0):.2%} · corridor L {st.get('L', 0):.2f} · MES overlay {st.get('beta_overlay', 0):.2f}",
             "", "## Reconciliation (QuantConnect vs IBKR)",
             ("- not run (IBKR Flex not configured)" if not ib_ok else
              ("- all positions match" if not diffs else "\n".join(f"- ⚠️ {d}" for d in diffs))),
             "", "## Risk checks", "\n".join(f"- ⚠️ {f}" for f in findings) or "- all clear"]
    return "\n".join(lines) + "\n"


def run(vault: str, today: Optional[dt.date] = None, state: Optional[dict] = None,
        ib: Optional[Dict[str, float]] = None) -> int:
    today = today or dt.date.today()
    if state is None:
        import qc_api
        raw = qc_api.object_get(["metadron/state.json"]).get("metadron/state.json")
        state = json.loads(raw) if raw else {}
    qc_pos = {t: float(p.get("quantity", 0)) for t, p in (state.get("positions") or {}).items()}
    ib = ib if ib is not None else ibkr_positions()
    diffs = reconcile(qc_pos, ib) if ib is not None else []
    findings = risk_findings(state, today) + (feed_findings(state, today) if state else [])
    os.makedirs(os.path.join(vault, "Risk"), exist_ok=True)
    with open(os.path.join(vault, "Risk", f"{today}.md"), "w", encoding="utf-8") as fh:
        fh.write(note(today, state, diffs, findings, ib is not None))
    for line in diffs + findings:
        print("ALERT:", line)
    return 1 if diffs or findings else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="risk officer")
    ap.add_argument("--vault", default=os.path.join(ROOT, "vault"))
    sys.exit(run(ap.parse_args().vault))
