import datetime as dt
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import events_agent as ev  # noqa: E402
import risk_officer as ro  # noqa: E402
import vault_sync as vs  # noqa: E402
import llm  # noqa: E402
import qc_api  # noqa: E402
import qc_deploy  # noqa: E402

TODAY = dt.date(2026, 11, 2)
RECENT = {"form": ["8-K", "8-K", "10-Q", "8-K"],
          "filingDate": ["2026-10-28", "2026-08-06", "2026-08-07", "2026-10-15"],
          "items": ["2.03,9.01", "2.02,9.01", "", "5.02"]}


def test_edgar_flags_and_earnings_estimate():
    r = ev.analyse_filings(RECENT, TODAY)
    assert r["penalty"] == pytest.approx(0.1)                     # 2.03 new debt beats 5.02 (0.05)
    assert any("2.03" in f for f in r["flags"]) and any("5.02" in f for f in r["flags"])
    assert r["days_to_event"] == (dt.date(2026, 8, 6) + dt.timedelta(days=91) - TODAY).days   # last 2.02 + 91d
    assert ev.analyse_filings({}, TODAY)["days_to_event"] is None


def test_claude_review_merge_is_validated():
    base = {"ABC": ev.analyse_filings(RECENT, TODAY)}
    review = {"tickers": {"ABC": {"days_to_event": 3, "penalty": 0.4, "flags": ["lawsuit"]},
                          "ZZZ": {"days_to_event": 1}},
              "escalation_flag": True}
    m = ev.merge(base, review)
    assert m["tickers"]["ABC"]["days_to_event"] == 3 and m["tickers"]["ABC"]["penalty"] == 0.4
    assert "ZZZ" not in m["tickers"] and m["escalation_flag"] is True
    bad = ev.merge(base, {"tickers": {"ABC": {"days_to_event": 999, "penalty": "x"}}, "escalation_flag": "maybe"})
    assert bad["tickers"]["ABC"]["days_to_event"] == base["ABC"]["days_to_event"] and bad["escalation_flag"] is None


def test_events_run_offline(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "QC_USER_ID", "QC_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    res = ev.run(["ABC"], str(tmp_path), use_llm=True, upload=True, today=TODAY,
                 fetch=lambda cik: {"filings": {"recent": RECENT}}, ciks={"ABC": 1})
    assert res["tickers"]["ABC"]["penalty"] == pytest.approx(0.1) and res["uploaded"] is False
    assert json.loads((tmp_path / "events.json").read_text())["as_of"] == "2026-11-02"


FLEX = b"""<FlexQueryResponse><FlexStatements><FlexStatement><OpenPositions>
<OpenPosition symbol="AAPL" assetCategory="STK" position="10"/>
<OpenPosition symbol="XLV" assetCategory="STK" position="5"/>
</OpenPositions></FlexStatement></FlexStatements></FlexQueryResponse>"""


def test_reconciliation_and_alert_exit_code(tmp_path):
    ib = ro.parse_flex(FLEX)
    assert ib == {"AAPL": 10.0, "XLV": 5.0}
    state = {"time": "2026-11-02 15:45:00", "nav": 1e6, "breaker": False,
             "state": {"regime": "RANGE", "zone": "DYNAMIC", "rm": 0.1, "L": 1.2, "beta_overlay": 0.8},
             "positions": {"AAPL": {"quantity": 10, "weight": 0.02}, "XLV": {"quantity": 4, "weight": 0.01}}}
    assert ro.reconcile({"AAPL": 10, "XLV": 4}, ib) == ["XLV: QuantConnect 4 vs IBKR 5"]
    assert ro.run(str(tmp_path), TODAY, state, ib) == 1
    text = (tmp_path / "Risk" / "2026-11-02.md").read_text()
    assert "XLV: QuantConnect 4 vs IBKR 5" in text and "alerts: 1" in text
    state["positions"]["XLV"]["quantity"] = 5
    assert ro.run(str(tmp_path), TODAY, state, ib) == 0
    assert ro.risk_findings({}, TODAY) and "stale" in ro.risk_findings(dict(state, time="2026-10-20"), TODAY)[0]


def test_vault_sync_writes_notes(tmp_path):
    state = {"time": "2026-11-02", "nav": 1e6,
             "state": {"regime": "TRENDING", "zone": "ROTATION", "rm": 0.13, "L": 3.1, "sigma": 0.15, "cap": 4.5,
                       "beta_overlay": 2.7, "sleeve_beta": 0.4, "components": {"anchor": 0.08, "kronos": float("nan")}},
             "sector_scores": {"XLV": 0.3, "XLK": 0.1}, "model_weights": {"score": 1.0},
             "positions": {"AAPL": {"weight": 0.03, "gates": {"FUNDAMENTALS": 0.7}, "reasons": ["composite 0.70"]}}}
    files = {"metadron/state.json": json.dumps(state),
             "metadron/universe_scores.json": json.dumps({"health": {"members": 1500}, "shortlist": {"IG": ["AAPL"]}}),
             "metadron/sector_views.json": json.dumps({"bias": {"bias": "BULLISH"}, "findings": [], "overweight": []})}
    assert vs.run(str(tmp_path), files, TODAY) == 4
    daily = (tmp_path / "Daily" / "2026-11-02.md").read_text()
    assert "regime: TRENDING" in daily and "[[Positions/AAPL|AAPL]]" in daily and "| XLV | +0.30 |" in daily
    assert "composite 0.70" in (tmp_path / "Positions" / "AAPL.md").read_text()


def test_llm_and_qc_helpers_degrade_without_credentials(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "QC_USER_ID", "QC_API_TOKEN", "QC_ORG_ID"):
        monkeypatch.delenv(k, raising=False)
    assert llm.ask("sys", "hi") is None
    assert llm.extract_json('noise {"a": 1} tail') == '{"a": 1}' and llm.extract_json("none") is None
    assert qc_api.post("files/read", {}) is None and qc_api.object_set("k", "v") is False and qc_api.object_get(["k"]) == {}
    files = qc_deploy.files_to_push()
    assert "main.py" in files and "corridor.py" in files and all(n.endswith(".py") for n in files)


def test_deploy_frees_only_running_backtests(monkeypatch):
    import qc_deploy
    calls = []

    def post(endpoint, payload):
        calls.append((endpoint, payload))
        if endpoint == "backtests/list":
            return {"success": True, "backtests": [
                {"backtestId": "run", "status": "Running: 40%", "name": "old"},
                {"backtestId": "queued", "status": "In Queue...", "name": "q"},
                {"backtestId": "done", "status": "Completed.", "name": "good"},
                {"backtestId": "err", "status": "Runtime Error", "name": "bad"},
                {"backtestId": "nostatus-done", "completed": True, "progress": 1.0},
                {"backtestId": "nostatus-run", "completed": False, "progress": 0.3}]}
        return {"success": True}

    monkeypatch.setattr(qc_deploy.qc_api, "post", post)
    monkeypatch.setattr(qc_deploy.time, "sleep", lambda s: None)
    assert qc_deploy.free_node(1) == 3
    deleted = [p["backtestId"] for e, p in calls if e == "backtests/delete"]
    assert deleted == ["run", "queued", "nostatus-run"]          # completed / failed runs keep their results
