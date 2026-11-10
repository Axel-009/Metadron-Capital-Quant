"""IBKR Flex connection test (read-only): does the token + query return the account's statement?

Prints the outcome step by step. The account id is masked; no secret is printed. Exit 1 on failure.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import risk_officer as ro  # noqa: E402


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "metadron-flex/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    token, query = os.environ.get("IBKR_FLEX_TOKEN"), os.environ.get("IBKR_FLEX_QUERY_ID")
    missing = [k for k, v in (("IBKR_FLEX_TOKEN", token), ("IBKR_FLEX_QUERY_ID", query)) if not v]
    if missing:
        print(f"::error::Missing repository secrets: {', '.join(missing)}")
        return 1
    print(f"secrets present: token {len(token.strip())} chars, query id {len(query.strip())} chars "
          f"({'digits only' if query.strip().isdigit() else 'NOT digits only — the query id is a number'})")
    token, query = token.strip(), query.strip()
    root = ET.fromstring(_get(ro.FLEX + "SendRequest?" + urllib.parse.urlencode({"t": token, "q": query, "v": 3})))
    status, ref = root.findtext("Status"), root.findtext("ReferenceCode")
    print(f"step 1 request: status={status} reference={'yes' if ref else 'no'}")
    if not ref:
        print(f"::error::IBKR refused the request: code {root.findtext('ErrorCode')} — {root.findtext('ErrorMessage')}")
        return 1
    body = b""
    for i in range(24):
        time.sleep(5)
        body = _get(ro.FLEX + "GetStatement?" + urllib.parse.urlencode({"t": token, "q": ref, "v": 3}))
        if b"<FlexQueryResponse" in body:
            break
        try:
            r2 = ET.fromstring(body)
            print(f"  waiting ({i + 1}): {r2.findtext('ErrorCode')} {r2.findtext('ErrorMessage')}")
        except ET.ParseError:
            print(f"  waiting ({i + 1}): unparsed reply of {len(body)} bytes")
    if b"<FlexQueryResponse" not in body:
        print("::error::IBKR accepted the request but no statement arrived within 2 minutes")
        return 1
    doc = ET.fromstring(body)
    print(f"step 2 statement: query name {doc.get('queryName')!r}, type {doc.get('type')}")
    for st in doc.iter("FlexStatement"):
        acct = st.get("accountId") or ""
        kind = "PAPER" if acct.upper().startswith("DU") else "LIVE (not a paper account id)"
        print(f"  account {acct[:2]}{'*' * max(len(acct) - 2, 0)} → {kind}; period {st.get('fromDate')} → {st.get('toDate')}; "
              f"generated {st.get('whenGenerated')}")
        print("  sections in the statement:", sorted({c.tag for c in st}))
    pos = ro.parse_flex(body)
    print(f"step 3 open positions parsed: {len(pos)}")
    for s, q in list(sorted(pos.items()))[:15]:
        print(f"  {s}: {q:g}")
    print("RESULT: IBKR Flex connection works" + ("" if pos else " (no open positions — expected for an unused paper account)"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"::error::IBKR Flex test failed: {type(exc).__name__}: {exc}")
        sys.exit(1)
