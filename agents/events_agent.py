"""Events agent — SEC filings + news → metadron/events.json (earnings dates, capital-structure flags, escalation).

Daily before the open. Propose-only: the algorithm reads the file; nothing here trades.

    tickers        held / target names (metadron/state.json) + shortlist (metadron/universe_scores.json)
                   + theme members found by the profiles (universe_scores 'watch'); or --tickers
    SEC EDGAR      free JSON (needs SEC_USER_AGENT = "Name email" per SEC policy):
                   8-K items in the last 30 days → flags and fundamentals-gate penalties
                       1.03 bankruptcy 1.0 · 4.02 non-reliance on financials 0.4 · 2.04 debt acceleration 0.3 ·
                       3.01 delisting notice 0.3 · 2.03 new direct financial obligation (debt / private credit) 0.1 ·
                       2.06 impairment 0.1 · 1.01 material agreement 0.0 (flag only) · 5.02 officer change 0.05
                   next earnings ≈ last 8-K item 2.02 (results) + 91 days (else last 10-Q/10-K + 91 days)
    Claude         (web search) confirms earnings dates within 21 days, adds legal fights / guidance / credit news
                   as penalties (0–1) and sets the global escalation flag (true / false / null = no view)
Output: {"as_of", "tickers": {t: {"days_to_event", "penalty", "flags"}}, "escalation_flag"}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "algorithm"))

ITEM_PENALTY = {"1.03": 1.0, "4.02": 0.4, "2.04": 0.3, "3.01": 0.3, "2.03": 0.1, "2.06": 0.1, "1.01": 0.0, "5.02": 0.05}
ITEM_NAME = {"1.03": "bankruptcy/receivership", "4.02": "non-reliance on prior financials", "2.04": "debt acceleration",
             "3.01": "delisting notice", "2.03": "new direct financial obligation", "2.06": "impairment",
             "1.01": "material definitive agreement", "5.02": "officer/director change", "2.02": "results"}
LOOKBACK_DAYS = 30
EARNINGS_GAP = 91


def _get(url: str) -> Optional[dict]:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as exc:
        print(f"[events] {url}: {exc}", file=sys.stderr)
        return None


def cik_map() -> Dict[str, int]:
    data = _get("https://www.sec.gov/files/company_tickers.json") or {}
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values() if "ticker" in v}


def analyse_filings(recent: dict, today: dt.date) -> dict:
    """recent = submissions['filings']['recent'] (parallel lists). Returns flags, penalty, days_to_event."""
    forms, dates, items = recent.get("form", []), recent.get("filingDate", []), recent.get("items", [])
    flags, penalty, last_results, last_periodic = [], 0.0, None, None
    for f, d, it in zip(forms, dates, items + [""] * (len(forms) - len(items))):
        try:
            day = dt.date.fromisoformat(d)
        except ValueError:
            continue
        codes = [c.strip() for c in (it or "").split(",") if c.strip()]
        if f == "8-K" and "2.02" in codes and (last_results is None or day > last_results):
            last_results = day
        if f in ("10-Q", "10-K") and (last_periodic is None or day > last_periodic):
            last_periodic = day
        if f == "8-K" and (today - day).days <= LOOKBACK_DAYS:
            for c in codes:
                if c in ITEM_PENALTY:
                    flags.append(f"{d} 8-K {c} {ITEM_NAME[c]}")
                    penalty = max(penalty, ITEM_PENALTY[c])
    anchor = last_results or last_periodic
    days = None
    if anchor is not None:
        nxt = anchor + dt.timedelta(days=EARNINGS_GAP)
        while (nxt - today).days < -10:                  # missed cycles → roll forward
            nxt += dt.timedelta(days=EARNINGS_GAP)
        days = (nxt - today).days
    return {"days_to_event": days, "penalty": penalty, "flags": flags, "estimated": True}


def tickers_from_store() -> List[str]:
    import qc_api
    files = qc_api.object_get(["metadron/state.json", "metadron/universe_scores.json"])
    out = set()
    try:
        st = json.loads(files.get("metadron/state.json", "{}"))
        out |= set(st.get("positions", {})) | {t for t, w in (st.get("targets") or {}).items() if w}
    except Exception:
        pass
    try:
        us = json.loads(files.get("metadron/universe_scores.json", "{}"))
        out |= {t for v in (us.get("shortlist") or {}).values() for t in v} | set(us.get("watch", {}))
    except Exception:
        pass
    return sorted(t for t in out if t.isalpha() and len(t) <= 5)


def claude_review(events: dict, today: dt.date) -> Optional[dict]:
    import llm
    near = {t: e for t, e in events.items() if e["days_to_event"] is not None and e["days_to_event"] <= 21}
    system = ("You are the events analyst of an equity fund. You verify facts with web search and answer with one "
              "JSON object only. Never invent dates: use null when unsure.")
    prompt = (f"Today is {today.isoformat()}. Estimated next earnings (days from today) and recent SEC 8-K flags:\n"
              + json.dumps({t: {k: e[k] for k in ('days_to_event', 'flags')} for t, e in events.items()}, indent=1)
              + "\n\n1. For these names with estimated earnings within 21 days, confirm the date: "
              + ", ".join(sorted(near)) +
              "\n2. For every name, note material legal fights, guidance cuts, new debt / private-credit facilities or "
              "covenant issues from the last 30 days, as a penalty 0–1 (0 = nothing material, 1 = existential).\n"
              "3. Is geopolitical escalation (war, sanctions, supply shocks) materially rising this week? true/false/null.\n"
              'Reply: {"tickers": {"TICK": {"days_to_event": int|null, "penalty": float, "flags": [str]}}, '
              '"escalation_flag": true|false|null}')
    raw = llm.extract_json(llm.ask(system, prompt, web_search=True, max_searches=15))
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


def merge(events: dict, review: Optional[dict]) -> dict:
    out = {"tickers": {t: dict(e) for t, e in events.items()}, "escalation_flag": None}
    if not review:
        return out
    for t, r in (review.get("tickers") or {}).items():
        if t not in out["tickers"] or not isinstance(r, dict):
            continue
        e = out["tickers"][t]
        d = r.get("days_to_event")
        if isinstance(d, int) and -5 <= d <= 120:
            e["days_to_event"], e["estimated"] = d, False
        try:
            e["penalty"] = min(1.0, max(e["penalty"], float(r.get("penalty") or 0.0)))
        except (TypeError, ValueError):
            pass
        e["flags"] = e["flags"] + [str(x) for x in (r.get("flags") or [])][:5]
    if isinstance(review.get("escalation_flag"), bool):
        out["escalation_flag"] = review["escalation_flag"]
    return out


def run(tickers: List[str], out_dir: str, use_llm=True, upload=True, today: Optional[dt.date] = None,
        fetch=None, ciks: Optional[Dict[str, int]] = None) -> dict:
    today = today or dt.date.today()
    fetch = fetch or (lambda cik: (_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json") or {}))
    ciks = cik_map() if ciks is None else ciks
    events = {}
    for t in tickers:
        cik = ciks.get(t.upper())
        sub = fetch(cik) if cik else {}
        events[t] = analyse_filings(((sub or {}).get("filings") or {}).get("recent") or {}, today)
    result = merge(events, claude_review(events, today) if use_llm and events else None)
    result["as_of"] = today.isoformat()
    text = json.dumps(result, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "events.json"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if upload:
        import qc_api
        result["uploaded"] = qc_api.object_set("metadron/events.json", text)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="events agent")
    ap.add_argument("--tickers", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    tks = sorted(set([t for t in a.tickers.split(",") if t] or tickers_from_store()))   # incl. profiled theme members
    res = run(tks, a.out, not a.no_llm, not a.no_upload)
    print(json.dumps({k: v for k, v in res.items() if k != "tickers"}, indent=1), f"{len(res['tickers'])} names")
