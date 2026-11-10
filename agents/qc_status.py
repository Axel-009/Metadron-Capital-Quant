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


def orders(pid: int, bid: str, limit: int = 3000) -> list:
    """QuantConnect serves at most 100 orders per call: [start, end) with end − start ≤ 100 (inclusive on some
    versions, so 99-wide pages are requested)."""
    out, start, seen = [], 0, set()
    while start < limit:
        res = qc_api.post("backtests/orders/read", {"projectId": pid, "backtestId": bid, "start": start,
                                                    "end": start + 99}) or {}
        page = res.get("orders") or []
        if isinstance(page, dict):
            page = list(page.values())
        if start == 0 and not page:
            print("  (orders endpoint keys:", sorted(res.keys()), "length:", res.get("length"), "errors:",
                  res.get("errors"), ")")
        new = [o for o in page if o.get("id") not in seen]
        seen |= {o.get("id") for o in new}
        out += new
        if len(page) < 99:
            break
        start += 99
    return out


def logs(pid: int, bid: str) -> list:
    out, start = [], 0
    while start < 3000:
        res = qc_api.post("backtests/read/log", {"projectId": pid, "backtestId": bid, "start": start,
                                                 "end": start + 200, "query": " "}) or {}
        page = res.get("logs") or res.get("backtestLogs") or []
        if start == 0 and not page:
            print("  (log endpoint keys:", sorted(res.keys()), "length:", res.get("length"), ")")
        out += [str(x) for x in page]
        if len(page) < 200:
            break
        start += 200
    return out


def chart_tail(pid: int, bid: str, name: str, n: int = 4) -> None:
    """Last few points of every series of a chart (first, last, min, max, count)."""
    import time
    res = {}
    for _ in range(4):
        res = qc_api.post("backtests/chart/read", {"projectId": pid, "backtestId": bid, "name": name, "count": 500,
                                                   "start": 0, "end": int(time.time())}) or {}
        if res.get("chart"):
            break
        time.sleep(4)
    series = (res.get("chart") or {}).get("series") or {}
    if not series:
        print(f"CHART {name}: no data (keys {sorted(res.keys())}, errors {res.get('errors')})")
        return
    print(f"CHART {name}:")
    for sname, sr in series.items():
        vals, xs = [], []
        for pt in sr.get("values") or []:
            try:
                vals.append(float(pt[1] if isinstance(pt, (list, tuple)) else pt.get("y")))
                xs.append(float(pt[0] if isinstance(pt, (list, tuple)) else pt.get("x")))
            except (TypeError, ValueError, IndexError):
                pass
        if xs:
            d = lambda x: time.strftime("%Y-%m-%d", time.gmtime(x))
            print(f"  {sname}: algorithm dates {d(xs[0])} → {d(xs[-1])}")
        if vals:
            print(f"  {sname}: n={len(vals)} first={vals[0]:.4g} last={vals[-1]:.4g} min={min(vals):.4g} "
                  f"max={max(vals):.4g} tail={[round(v, 4) for v in vals[-n:]]}")


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
    print("backtest keys:", sorted(bt.keys()))
    tp = bt.get("totalPerformance") or {}
    print("totalPerformance keys:", sorted(tp.keys()) if isinstance(tp, dict) else type(tp))
    trades = (tp.get("closedTrades") or []) if isinstance(tp, dict) else []
    print(f"CLOSED TRADES: {len(trades)}")
    def _pl(t):
        try:
            return float(t.get("profitLoss") or 0)
        except (TypeError, ValueError):
            return 0.0
    for t in sorted(trades, key=lambda t: -abs(_pl(t)))[:25]:
        sy = t.get("symbol") or (t.get("symbols") or [{}])[0]
        name = (sy.get("value") or sy.get("id")) if isinstance(sy, dict) else sy
        print(f"  {str(name):<24} qty={t.get('quantity')} {str(t.get('entryTime'))[:16]} @ {t.get('entryPrice')} → "
              f"{str(t.get('exitTime'))[:16]} @ {t.get('exitPrice')}  P&L {_pl(t):,.0f}")
    if trades:
        print(f"  sum P&L {sum(_pl(t) for t in trades):,.0f}; winners {sum(1 for t in trades if _pl(t) > 0)}, "
              f"losers {sum(1 for t in trades if _pl(t) < 0)}")
    od = orders(pid, bid)
    print(f"ORDERS read: {len(od)}")
    if od:
        kinds = collections.Counter(str(o.get("tag") or "").split(" ")[0] or "(no tag)" for o in od)
        status = collections.Counter(str(o.get("status")) for o in od)
        types = collections.Counter(str(o.get("type")) for o in od)
        print("  by tag:", dict(kinds.most_common(12)))
        print("  by status code:", dict(status), "| by type code:", dict(types))
        val = lambda o: abs(float(o.get("value") or 0) or float(o.get("quantity") or 0) * float(o.get("price") or 0))
        print("  order keys:", sorted(od[0].keys()))
        print("  traded value by tag:", {k: round(sum(val(o) for o in od if (str(o.get("tag") or "").split(" ")[0] or "(no tag)") == k))
                                         for k, _ in kinds.most_common(12)})
        print("  LARGEST 25 by value:")
        for o in sorted(od, key=lambda o: -val(o))[:25]:
            print(f"  {str(o.get('createdTime') or o.get('time'))[:19]} {_sym(o):<22} qty={o.get('quantity')} "
                  f"px={o.get('price')} val={val(o):,.0f} st={o.get('status')} | {str(o.get('tag'))[:120]}")
        print("  FIRST 12 / LAST 6:")
        for o in od[:12] + od[-6:]:
            print(f"  {str(o.get('createdTime') or o.get('time'))[:19]} {_sym(o):<22} qty={o.get('quantity')} "
                  f"px={o.get('price')} st={o.get('status')} | {str(o.get('tag'))[:120]}")
    for chart in ("Strategy Equity", "Decisions", "Feed", "Data health", "Leverage", "Regime", "Rm vs corridor", "Risk"):
        chart_tail(pid, bid, chart)
    lg = logs(pid, bid)
    print(f"LOG lines read: {len(lg)}")
    halts = [x for x in lg if "FEED HALT" in x]
    print(f"  FEED HALT lines: {len(halts)}")
    key = [x for x in lg if any(k in x for k in KEY_LOG)]
    for x in lg[:30]:
        print("  | " + x[:260])
    for x in key[:25] + (["  ..."] if len(key) > 50 else []) + key[-25:]:
        print("  " + x[:260])
    return 1 if err else 0


if __name__ == "__main__":
    pid = os.environ.get("QC_PROJECT_ID")
    if not pid or qc_api.creds() is None:
        print("::error::Missing QuantConnect repository secrets")
        sys.exit(1)
    sys.exit(report(int(pid), sys.argv[1] if len(sys.argv) > 1 else ""))
