"""QuantConnect REST API v2 — the few calls the agents and the deploy job need.

Auth: HTTP Basic with user id and sha256("{api_token}:{timestamp}"), plus a Timestamp header.
Secrets: QC_USER_ID, QC_API_TOKEN, QC_ORG_ID (ObjectStore), QC_PROJECT_ID (deploy / backtests).
Every call returns None / False on failure (logged to stderr) so agent jobs degrade instead of crashing.
Endpoint shapes follow the public API docs; verify on first run (see SETUP.md, "First run checks").
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, Optional

BASE = "https://www.quantconnect.com/api/v2/"


def creds() -> Optional[tuple]:
    uid, token = os.environ.get("QC_USER_ID"), os.environ.get("QC_API_TOKEN")
    return (uid, token) if uid and token else None


def _headers(extra: Optional[dict] = None) -> Optional[dict]:
    c = creds()
    if not c:
        return None
    ts = str(int(time.time()))
    hashed = hashlib.sha256(f"{c[1]}:{ts}".encode()).hexdigest()
    h = {"Authorization": "Basic " + base64.b64encode(f"{c[0]}:{hashed}".encode()).decode(), "Timestamp": ts}
    h.update(extra or {})
    return h


def post(endpoint: str, payload: dict) -> Optional[dict]:
    h = _headers({"Content-Type": "application/json"})
    if h is None:
        return None
    req = urllib.request.Request(BASE + endpoint, data=json.dumps(payload).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
    except Exception as exc:
        print(f"[qc] {endpoint} failed: {exc}", file=sys.stderr)
        return None
    if not out.get("success", False):
        print(f"[qc] {endpoint} unsuccessful: {out.get('errors')}", file=sys.stderr)
    return out


def object_set(key: str, text: str) -> bool:
    """Upload one ObjectStore file (multipart)."""
    org = os.environ.get("QC_ORG_ID")
    h = _headers()
    if h is None or not org:
        return False
    boundary = "----metadron" + h["Timestamp"]
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"organizationId\"\r\n\r\n{org}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"key\"\r\n\r\n{key}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"objectData\"; filename=\"data.json\"\r\n"
            f"Content-Type: application/json\r\n\r\n{text}\r\n--{boundary}--\r\n").encode()
    h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(BASE + "object/set", data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return bool(json.load(r).get("success"))
    except Exception as exc:
        print(f"[qc] object/set {key} failed: {exc}", file=sys.stderr)
        return False


def object_get(keys, wait_s: int = 90) -> Dict[str, str]:
    """Download ObjectStore files: object/get starts a job, then poll until a download URL is ready."""
    org = os.environ.get("QC_ORG_ID")
    if not org or creds() is None:
        return {}
    job = post("object/get", {"organizationId": org, "keys": list(keys)})
    job_id = (job or {}).get("jobId")
    url = (job or {}).get("url")
    t0 = time.time()
    while job_id and not url and time.time() - t0 < wait_s:
        time.sleep(3)
        url = (post("object/get", {"organizationId": org, "jobId": job_id}) or {}).get("url")
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            blob = r.read()
        out = {}
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                for k in keys:
                    if name.endswith(k) or name.endswith(k.split("/")[-1]):
                        out[k] = z.read(name).decode("utf-8", "replace")
        return out
    except Exception as exc:
        print(f"[qc] object download failed: {exc}", file=sys.stderr)
        return {}
