"""Deploy the algorithm to QuantConnect cloud and (optionally) run a backtest — used by GitHub Actions.

    python3 agents/qc_deploy.py --backtest        # push algorithm/*.py, compile, start a backtest

Needs QC_USER_ID, QC_API_TOKEN and QC_PROJECT_ID (create an empty Python project once in the QuantConnect
web IDE and copy its id from the URL). Live trading is NEVER started from here: switching the live node to a
new version is a one-click owner action in QuantConnect (hard-cap rule — the owner approves what trades).
Parameters from algorithm/config.json are pushed as project parameters.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import qc_api  # noqa: E402


def files_to_push():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "algorithm", "*.py"))):
        with open(p, encoding="utf-8") as fh:
            out[os.path.basename(p)] = fh.read()
    return out


def still_running(b: dict) -> bool:
    """A backtest that is queued or running (never a completed or failed one — those keep their results)."""
    if b.get("completed") or b.get("error") or b.get("stacktrace"):
        return False
    status = str(b.get("status") or "").strip().lower()
    if status.startswith(("completed", "runtime error")):
        return False
    if status.startswith(("in queue", "in progress", "running", "initializing", "launching")):
        return True
    return float(b.get("progress") or 0.0) < 1.0


def free_node(project_id: int) -> int:
    """Owner decision (Nov 2026): a backtest of an older version still running in this project is deleted so the
    new one can start (one backtest node). Completed and failed backtests are never touched."""
    res = qc_api.post("backtests/list", {"projectId": project_id}) or {}
    n = 0
    bts = res.get("backtests") or []
    print(f"backtests in project: {len(bts)}")
    for b in bts[:6]:                                      # what the project holds (newest first), for diagnosis
        print(f"  - {b.get('name')!r} status={b.get('status')!r} progress={b.get('progress')} "
              f"completed={b.get('completed')} error={bool(b.get('error'))}")
    org = os.environ.get("QC_ORG_ID")
    if org:                                                # which node is busy (it may be another project / research)
        nodes = qc_api.post("nodes/read", {"organizationId": org}) or {}
        for kind in ("backtest", "research", "live"):
            for nd in nodes.get(kind) or []:
                print(f"  node[{kind}] {nd.get('name')!r} sku={nd.get('sku')} busy={nd.get('busy')} "
                      f"usedBy={nd.get('usedBy') or nd.get('projectName')}")
    for b in bts:
        bid = b.get("backtestId")
        if bid and still_running(b):
            ok = qc_api.post("backtests/delete", {"projectId": project_id, "backtestId": bid})
            print(f"deleted still-running backtest {b.get('name')!r} ({bid}):", "ok" if ok and ok.get("success") else "FAILED")
            n += 1
    if n:
        time.sleep(15)
    return n


def deploy(project_id: int, backtest: bool = False, name: str = "") -> bool:
    for fname, content in files_to_push().items():
        ok = qc_api.post("files/update", {"projectId": project_id, "name": fname, "content": content})
        if not ok or not ok.get("success"):
            ok = qc_api.post("files/create", {"projectId": project_id, "name": fname, "content": content})
        if not ok or not ok.get("success"):
            print(f"push failed: {fname}")
            return False
        print(f"pushed {fname}")
    with open(os.path.join(ROOT, "algorithm", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    params = cfg.get("parameters", {})
    update = {"projectId": project_id, "parameters": [{"key": k, "value": v} for k, v in params.items()]}
    if cfg.get("project-name"):                            # owner-chosen QuantConnect project name
        update["name"] = cfg["project-name"]
    res = qc_api.post("projects/update", update)
    if cfg.get("project-name"):
        print("project name:", cfg["project-name"], "ok" if res and res.get("success") else "NOT applied")
    comp = qc_api.post("compile/create", {"projectId": project_id}) or {}
    cid = comp.get("compileId")
    state = comp.get("state")
    t0 = time.time()
    while cid and state not in ("BuildSuccess", "BuildError") and time.time() - t0 < 300:
        time.sleep(5)
        state = (qc_api.post("compile/read", {"projectId": project_id, "compileId": cid}) or {}).get("state")
    print("compile:", state)
    if state != "BuildSuccess":
        return False
    if backtest:
        free_node(project_id)
        payload = {"projectId": project_id, "compileId": cid,
                   "backtestName": name or f"ci {time.strftime('%Y-%m-%d %H:%M')}"}
        bt = qc_api.post("backtests/create", payload)
        for _ in range(3):                                 # the node is released a few seconds after a delete
            if (bt or {}).get("backtest", {}).get("backtestId"):
                break
            time.sleep(20)
            bt = qc_api.post("backtests/create", payload)
        bid = (bt or {}).get("backtest", {}).get("backtestId")
        print("backtest:", bid)
        if not bid:
            errs = "; ".join((bt or {}).get("errors") or ["no backtest id returned"])
            print(f"::error::Code pushed and compiled, but the backtest did not start: {errs} "
                  "(stop the running backtest in QuantConnect, then re-run this workflow or press Backtest there).")
            return False
        print(f"link: https://www.quantconnect.com/project/{project_id}/{bid}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="push the algorithm to QuantConnect")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    pid = os.environ.get("QC_PROJECT_ID")
    if not pid or qc_api.creds() is None:
        missing = [k for k in ("QC_PROJECT_ID", "QC_USER_ID", "QC_API_TOKEN") if not os.environ.get(k)]
        print(f"::error::Missing repository secrets: {', '.join(missing)} — add them under Settings → Secrets "
              "and variables → Actions → Repository secrets (Secrets tab, not Variables). Nothing deployed.")
        sys.exit(1)
    sys.exit(0 if deploy(int(pid), a.backtest, a.name) else 1)
