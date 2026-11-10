"""Backtest status — is the latest QuantConnect backtest running, trading and healthy? (GitHub Actions; read-only)

    python3 agents/qc_status.py            # newest backtest in the project
    python3 agents/qc_status.py <id>       # a specific backtest
Prints: status / progress / error, headline statistics, order count + a sample of orders by kind, key log lines.
Exit code 1 when the backtest has a runtime error; 0 otherwise (also while it is still running).
"""

from __future__ import annotations

import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qc_api  # noqa: E402

KEY_LOG = ("FEED", "UNIVERSE", "REGIME", "VIEW", "BREAKER", "HARD STOP", "DATA-HEALTH", "ROTATION", "SPLIT", "Error",
           "error", "refused", "unavailable")


def _sym(o: dict) -> str:
    s = o.get("symbol")
    return str(s.get("value") or s.get("Value") or s.get("id")) if isinstance(s, dict) else str(s)


def orders(pid: int, bid: str, limit: int = 2000) -> list:
    out, start = [], 0
    while start < limit:
        res = qc_api.post("backtests/orders/read", {"projectId": pid, "backtestId": bid, "start": start,
                                                    "end": start + 100}) or {}
        page = res.get("orders") or []
        out += page
        if len(page) < 100:
            break
        start += 100
    return out


def logs(pid: int, bid: str) -> list:
    out, start = [], 0
    while start < 3000:
        res = qc_api.post("backtests/read/log", {"projectId": pid, "backtestId": bid, "start": start,
                                                 "end": start + 200, "query": " "}) or {}
        page = res.get("logs") or res.get("backtestLogs") or []
        out += [str(x) for x in page]
        if len(page) < 200:
            break
        start += 200
    return out


def report(pid: int, bid: str = "") -> int:
    if not bid:
        bts = (qc_api.post("backtests/list", {"projectId": pid}) or {}).get("backtests") or []
        if not bts:
            print("no backtests in the project")
            return 0
        bid = bts[0].get("backtestId")
    bt = (qc_api.post("backtests/read", {"projectId": pid, "backtestId": bid}) or {}).get("backtest") or {}
    print(f"BACKTEST {bt.get('name')!r} id={bid}")
    print(f"status={bt.get('status')!r} progress={bt.get('progress')} completed={bt.get('completed')}")
    err = bt.get("error") or bt.get("stacktrace")
    if err:
        print("RUNTIME ERROR:", str(bt.get("error"))[:1500])
        print(str(bt.get("stacktrace") or "")[:3000])
    for name in ("runtimeStatistics", "statistics"):
        st = bt.get(name) or {}
        if st:
            print(f"{name}: " + " | ".join(f"{k}={v}" for k, v in st.items()))
    od = orders(pid, bid)
    print(f"ORDERS read: {len(od)}")
    if od:
        kinds = collections.Counter(str(o.get("tag") or "").split(" ")[0] or "(no tag)" for o in od)
        status = collections.Counter(str(o.get("status")) for o in od)
        types = collections.Counter(str(o.get("type")) for o in od)
        print("  by tag:", dict(kinds.most_common(12)))
        print("  by status code:", dict(status), "| by type code:", dict(types))
        for o in od[:8] + od[-8:]:
            print(f"  {str(o.get('createdTime') or o.get('time'))[:19]} {_sym(o):<8} qty={o.get('quantity')} "
                  f"px={o.get('price')} status={o.get('status')} | {str(o.get('tag'))[:110]}")
    lg = logs(pid, bid)
    print(f"LOG lines read: {len(lg)}")
    halts = [x for x in lg if "FEED HALT" in x]
    print(f"  FEED HALT lines: {len(halts)}")
    key = [x for x in lg if any(k in x for k in KEY_LOG)]
    for x in key[:25] + (["  ..."] if len(key) > 50 else []) + key[-25:]:
        print("  " + x[:260])
    return 1 if err else 0


if __name__ == "__main__":
    pid = os.environ.get("QC_PROJECT_ID")
    if not pid or qc_api.creds() is None:
        print("::error::Missing QuantConnect repository secrets")
        sys.exit(1)
    sys.exit(report(int(pid), sys.argv[1] if len(sys.argv) > 1 else ""))
