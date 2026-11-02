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
        params = json.load(fh).get("parameters", {})
    qc_api.post("projects/update", {"projectId": project_id,
                                    "parameters": [{"key": k, "value": v} for k, v in params.items()]})
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
        bt = qc_api.post("backtests/create", {"projectId": project_id, "compileId": cid,
                                              "backtestName": name or f"ci {time.strftime('%Y-%m-%d %H:%M')}"})
        print("backtest:", (bt or {}).get("backtest", {}).get("backtestId"))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="push the algorithm to QuantConnect")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    pid = os.environ.get("QC_PROJECT_ID")
    if not pid or qc_api.creds() is None:
        print("QC_PROJECT_ID / QC_USER_ID / QC_API_TOKEN not set — nothing deployed")
        sys.exit(0)
    sys.exit(0 if deploy(int(pid), a.backtest, a.name) else 1)
