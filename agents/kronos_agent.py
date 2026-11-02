"""Kronos agent — CPU inference of the Kronos foundation model → metadron/kronos.json.

Runs weekly (GitHub Actions, CPU only; fine-tuning on a GPU is out of scope for now). For SPY and the shortlist it
forecasts the next HORIZON trading days from the last LOOKBACK daily OHLCV bars and writes the expected return:

    {"as_of": "...", "horizon_days": 21, "model": "NeoQuasar/Kronos-small", "forecasts": {"SPY": 0.012, ...}}

The algorithm annualises it (alpha.kronos_alpha), blends it into forward Rm (SPY) and into per-name alpha with
a weight earned on the model scorecard (starts at 0 until its forecasts prove out).
Data: OpenBB when installed (design rule: OpenBB is the data layer), else Stooq daily CSV (free, no key).
Everything is guarded: no torch / no network → the job exits 0 and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import urllib.request
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "vendor", "Kronos"))

MODEL = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
LOOKBACK, HORIZON, SAMPLES = 400, 21, 3


def ohlcv(ticker: str):
    """DataFrame with timestamps, open, high, low, close, volume (oldest first) or None."""
    import pandas as pd
    try:
        from openbb import obb
        df = obb.equity.price.historical(ticker, start_date=(dt.date.today() - dt.timedelta(days=900)).isoformat()).to_df()
        df = df.reset_index().rename(columns={"date": "timestamps"})
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        pass
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = list(csv.DictReader(io.StringIO(r.read().decode())))
        df = pd.DataFrame(rows).rename(columns=str.lower)
        df["timestamps"] = pd.to_datetime(df["date"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        print(f"[kronos] no data for {ticker}: {exc}", file=sys.stderr)
        return None


def forecast(tickers: List[str]) -> Dict[str, float]:
    try:
        import pandas as pd
        from model import Kronos, KronosTokenizer, KronosPredictor
    except Exception as exc:
        print(f"[kronos] unavailable ({exc}) — install torch + vendor/Kronos requirements", file=sys.stderr)
        return {}
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mdl = Kronos.from_pretrained(MODEL)
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    out = {}
    for t in tickers:
        df = ohlcv(t)
        if df is None or len(df) < LOOKBACK:
            continue
        df = df.tail(LOOKBACK).reset_index(drop=True)
        future = pd.bdate_range(df["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZON)
        try:
            p = pred.predict(df[["open", "high", "low", "close", "volume"]], df["timestamps"],
                             pd.Series(future), pred_len=HORIZON, T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
            out[t] = float(p["close"].iloc[-1] / df["close"].iloc[-1] - 1)
        except Exception as exc:
            print(f"[kronos] {t}: {exc}", file=sys.stderr)
    return out


def tickers_from_store() -> List[str]:
    import qc_api
    raw = qc_api.object_get(["metadron/universe_scores.json"]).get("metadron/universe_scores.json")
    try:
        us = json.loads(raw) if raw else {}
        return sorted({t for v in (us.get("shortlist") or {}).values() for t in v})
    except Exception:
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kronos CPU forecasts")
    ap.add_argument("--tickers", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    tks = ["SPY"] + ([t for t in a.tickers.split(",") if t] or tickers_from_store())
    fc = forecast(list(dict.fromkeys(tks)))
    if not fc:
        print("[kronos] no forecasts written")
        sys.exit(0)
    text = json.dumps({"as_of": dt.date.today().isoformat(), "horizon_days": HORIZON, "model": MODEL,
                       "forecasts": fc}, indent=1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "kronos.json"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if not a.no_upload:
        import qc_api
        print("uploaded:", qc_api.object_set("metadron/kronos.json", text))
    print(f"{len(fc)} forecasts")
