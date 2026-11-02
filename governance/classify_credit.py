#!/usr/bin/env python3
"""Governance: Credit rating classification via FMP Rating API.

Uses FMP's /api/v3/rating/{symbol} endpoint which provides a composite
credit rating based on their proprietary model incorporating:
  - Discounted cash flow score
  - Return on equity score
  - Return on assets score
  - Debt/equity score
  - P/E score
  - P/B score

Ratings: AAA, AA, A, BBB (IG) | BB, B, CCC, CC, C, D (HY)

The Egan-Jones D/E + current ratio proxy is NOT used here — it lives in
engine/signals/security_analysis_engine.py as a bottom-up fundamental
analysis measure within the Graham-Dodd framework.

Output: governance/credit_classification.json
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data.openbb_data import _obb, _openbb_available

OUTPUT = Path(__file__).parent / "credit_classification.json"

# FMP API key
_fmp_key = os.environ.get("FMP_API_KEY", "")
if not _fmp_key:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        _fmp_key = os.environ.get("FMP_API_KEY", "")
    except ImportError:
        pass

# FMP rating → standardized IG/HY
_FMP_RATING_MAP = {
    "AAA": "IG", "AA+": "IG", "AA": "IG", "AA-": "IG",
    "A+": "IG", "A": "IG", "A-": "IG",
    "BBB+": "IG", "BBB": "IG", "BBB-": "IG",
    "BB+": "HY", "BB": "HY", "BB-": "HY",
    "B+": "HY", "B": "HY", "B-": "HY",
    "CCC+": "HY", "CCC": "HY", "CCC-": "HY",
    "CC": "HY", "C": "HY", "D": "HY",
}

# FMP also returns a recommendation string — map those too
_FMP_RECOMMENDATION_MAP = {
    "strong buy": "A", "buy": "BBB", "hold": "BB",
    "sell": "B", "strong sell": "CCC",
}


def fetch_fmp_ratings(tickers: list[str]) -> dict:
    """Fetch credit ratings from FMP via OpenBB.

    Tries multiple OpenBB endpoints to get the FMP composite rating.
    """
    ratings = {}
    if not _openbb_available or not _fmp_key:
        print("WARNING: OpenBB or FMP_API_KEY unavailable — cannot fetch ratings")
        return ratings

    for i, ticker in enumerate(tickers):
        try:
            # Primary: FMP equity estimates/consensus
            result = _obb.equity.estimates.consensus(symbol=ticker, provider="fmp")
            df = result.to_dataframe()
            if not df.empty:
                row = df.iloc[0]
                # Look for rating fields
                for field in ("rating", "rating_recommendation"):
                    val = row.get(field, None) if hasattr(row, "get") else getattr(row, field, None)
                    if val and isinstance(val, str):
                        upper = val.upper().strip()
                        if upper in _FMP_RATING_MAP:
                            ratings[ticker] = {
                                "rating": upper,
                                "category": _FMP_RATING_MAP[upper],
                                "source": "fmp_rating",
                            }
                            break
                # Try recommendation as fallback mapping
                if ticker not in ratings:
                    for field in ("recommendation_key", "recommendation_type"):
                        val = row.get(field, None) if hasattr(row, "get") else getattr(row, field, None)
                        if val and isinstance(val, str):
                            lower = val.lower().strip()
                            if lower in _FMP_RECOMMENDATION_MAP:
                                mapped = _FMP_RECOMMENDATION_MAP[lower]
                                ratings[ticker] = {
                                    "rating": mapped,
                                    "category": _FMP_RATING_MAP.get(mapped, "HY"),
                                    "source": "fmp_recommendation",
                                }
                                break
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(tickers) * 100
            print(f"  {pct:.0f}% ({len(ratings)} rated)", flush=True)
            time.sleep(0.2)

    return ratings


def main():
    print("=== GOVERNANCE: FMP CREDIT RATING CLASSIFICATION ===")

    from engine.data.universe_engine import get_engine
    ue = get_engine()
    ue.load()
    tickers = [s.ticker for s in ue.get_all()]
    print(f"Universe: {len(tickers)} tickers")

    print("\nFetching FMP credit ratings...")
    ratings = fetch_fmp_ratings(tickers)
    print(f"\nRated: {len(ratings)} / {len(tickers)} tickers")

    ig = sum(1 for v in ratings.values() if v["category"] == "IG")
    hy = sum(1 for v in ratings.values() if v["category"] == "HY")
    grade_counts = Counter(v["rating"] for v in ratings.values())

    print(f"IG: {ig}  |  HY: {hy}")
    print(f"By grade: {dict(sorted(grade_counts.items()))}")

    output = {
        "source": "FMP credit rating API (/api/v3/rating)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(ratings),
        "summary": {"IG": ig, "HY": hy},
        "grades": dict(grade_counts),
        "ratings": ratings,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
