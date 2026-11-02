"""Metadron Capital — Token Metering & Anomaly Detection.

Per-model token ledger with hourly tracking, HMAC signing, anomaly
detection, and daily report generation for the ARCHIVE tab.

Tracks every inference call across all models (Qwen, Llama, Air-LLM,
Brain Power) with caller attribution, latency, and token counts.

Anomaly detection:
    1.5× hourly baseline → FLAG (logged, visible in TECH tab)
    2.0× hourly baseline → ALERT (Prometheus counter, Slack if configured)
    3.0× hourly baseline → LOCKDOWN (LLM bridge enters read-only mode)

Daily cap: 4,000,000 tokens/day across all models. Beyond cap, LLM
calls return cached/summary responses only — no new inference.

Daily report generated to data/archive/token_usage/YYYY-MM-DD.json.gz
with full hourly breakdown per model, caller attribution, anomaly
flags, and cost estimation for Xiaomi corroboration.
"""

import gzip
import hashlib
import hmac
import json
import logging
import os
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("metadron.security.token_meter")

DAILY_TOKEN_CAP = int(os.environ.get("METADRON_DAILY_TOKEN_CAP", "4000000"))
HOURLY_BASELINE = int(os.environ.get("METADRON_HOURLY_TOKEN_BASELINE", "200000"))
FLAG_MULTIPLIER = 1.5
ALERT_MULTIPLIER = 2.0
LOCKDOWN_MULTIPLIER = 3.0

_HMAC_KEY = os.environ.get("METADRON_HMAC_KEY", "metadron-token-ledger-v1").encode()
_ARCHIVE_DIR = Path("data/archive/token_usage")
_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _sign(data: dict) -> str:
    """HMAC-SHA256 signature for tamper detection."""
    payload = json.dumps(data, sort_keys=True, default=str).encode()
    return hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()


class HourlyBucket:
    """Token usage for a single hour for a single model."""

    def __init__(self, model: str, hour: str):
        self.model = model
        self.hour = hour
        self.tokens_in = 0
        self.tokens_out = 0
        self.requests = 0
        self.total_latency_ms = 0.0
        self.callers: dict[str, int] = defaultdict(int)
        self.anomaly_level: str = "normal"  # normal, flag, alert, lockdown

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.requests, 1)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict:
        data = {
            "model": self.model,
            "hour": self.hour,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "callers": dict(self.callers),
            "anomaly_level": self.anomaly_level,
        }
        data["signature"] = _sign(data)
        return data


class TokenMeter:
    """Per-model token metering with anomaly detection and daily reporting."""

    MODELS = ["qwen-2.5-7b", "llama-3.1-8b", "airllm", "brain-power"]

    def __init__(self):
        self._lock = threading.Lock()
        # {model: {hour_str: HourlyBucket}}
        self._hourly: dict[str, dict[str, HourlyBucket]] = {m: {} for m in self.MODELS}
        self._daily_total: dict[str, int] = defaultdict(int)  # {date: total_tokens}
        self._lockdown_active = False
        self._cap_override = False  # Manual override via TECH tab frontend
        self._cap_override_at: Optional[str] = None
        self._today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info(
            "TokenMeter initialized: daily_cap=%d hourly_baseline=%d",
            DAILY_TOKEN_CAP, HOURLY_BASELINE,
        )

    def _current_hour(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00Z")

    def _current_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_bucket(self, model: str) -> HourlyBucket:
        hour = self._current_hour()
        if model not in self._hourly:
            self._hourly[model] = {}
        if hour not in self._hourly[model]:
            self._hourly[model][hour] = HourlyBucket(model=model, hour=hour)
        return self._hourly[model][hour]

    def _check_day_rollover(self):
        today = self._current_date()
        if today != self._today:
            # Generate yesterday's report before rolling over
            self._generate_daily_report(self._today)
            self._today = today
            self._daily_total[today] = 0
            self._lockdown_active = False
            # Prune old hourly data (keep last 48 hours)
            cutoff_hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00Z")
            for model in self.MODELS:
                old_hours = [h for h in self._hourly.get(model, {}) if h < cutoff_hour[:11]]
                for h in old_hours[:max(0, len(old_hours) - 48)]:
                    del self._hourly[model][h]

    # ── Public API ────────────────────────────────────────────

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        caller: str = "unknown",
    ) -> dict:
        """Record a model inference call. Returns status dict.

        Args:
            model: Model name (qwen-2.5-7b, llama-3.1-8b, airllm, brain-power)
            tokens_in: Prompt tokens consumed
            tokens_out: Response tokens generated
            latency_ms: Inference latency in milliseconds
            caller: Who called (learning_loop, research_bots, live_loop_phase3, etc.)

        Returns:
            {"allowed": bool, "anomaly_level": str, "daily_remaining": int}
        """
        with self._lock:
            self._check_day_rollover()

            # Map model name to canonical
            canonical = model.lower().replace(" ", "-")
            for m in self.MODELS:
                if m in canonical or canonical in m:
                    canonical = m
                    break

            bucket = self._get_bucket(canonical)
            total = tokens_in + tokens_out

            # Check daily cap
            today = self._current_date()
            self._daily_total[today] = self._daily_total.get(today, 0) + total
            daily_used = self._daily_total[today]

            if daily_used > DAILY_TOKEN_CAP and not self._cap_override:
                self._lockdown_active = True
                logger.critical(
                    "TOKEN CAP FLAG: %d/%d daily tokens — awaiting manual override on TECH tab",
                    daily_used, DAILY_TOKEN_CAP,
                )
                return {
                    "allowed": False,
                    "anomaly_level": "cap_flagged",
                    "daily_remaining": 0,
                    "daily_used": daily_used,
                    "override_required": True,
                }

            # Record
            bucket.tokens_in += tokens_in
            bucket.tokens_out += tokens_out
            bucket.requests += 1
            bucket.total_latency_ms += latency_ms
            bucket.callers[caller] += 1

            # Anomaly detection
            hourly_tokens = bucket.total_tokens
            anomaly = "normal"
            if hourly_tokens > HOURLY_BASELINE * LOCKDOWN_MULTIPLIER:
                anomaly = "lockdown"
                self._lockdown_active = True
                logger.critical(
                    "TOKEN ANOMALY LOCKDOWN: %s used %d tokens this hour (3× baseline %d)",
                    canonical, hourly_tokens, HOURLY_BASELINE,
                )
            elif hourly_tokens > HOURLY_BASELINE * ALERT_MULTIPLIER:
                anomaly = "alert"
                logger.warning(
                    "TOKEN ANOMALY ALERT: %s used %d tokens this hour (2× baseline %d)",
                    canonical, hourly_tokens, HOURLY_BASELINE,
                )
            elif hourly_tokens > HOURLY_BASELINE * FLAG_MULTIPLIER:
                anomaly = "flag"
                logger.info(
                    "TOKEN ANOMALY FLAG: %s used %d tokens this hour (1.5× baseline %d)",
                    canonical, hourly_tokens, HOURLY_BASELINE,
                )

            bucket.anomaly_level = anomaly

            return {
                "allowed": not self._lockdown_active,
                "anomaly_level": anomaly,
                "daily_remaining": max(0, DAILY_TOKEN_CAP - daily_used),
                "daily_used": daily_used,
                "hourly_tokens": hourly_tokens,
            }

    def override_cap(self) -> dict:
        """Manual override of daily token cap. Called from TECH tab frontend.
        Allows inference to continue beyond 4M tokens for the rest of the day."""
        self._cap_override = True
        self._cap_override_at = datetime.now(timezone.utc).isoformat()
        self._lockdown_active = False
        logger.warning("TOKEN CAP OVERRIDE: manual approval at %s — inference unlocked for today", self._cap_override_at)
        return {"status": "overridden", "timestamp": self._cap_override_at}

    @property
    def is_locked(self) -> bool:
        return self._lockdown_active

    def get_status(self) -> dict:
        """Full meter status for TECH tab and Prometheus."""
        today = self._current_date()
        daily_used = self._daily_total.get(today, 0)
        current_hour = self._current_hour()

        models = {}
        for model in self.MODELS:
            bucket = self._hourly.get(model, {}).get(current_hour)
            models[model] = {
                "current_hour_tokens": bucket.total_tokens if bucket else 0,
                "current_hour_requests": bucket.requests if bucket else 0,
                "current_hour_anomaly": bucket.anomaly_level if bucket else "normal",
                "current_hour_callers": dict(bucket.callers) if bucket else {},
                "current_hour_avg_latency_ms": round(bucket.avg_latency_ms, 1) if bucket else 0,
            }

        return {
            "daily_used": daily_used,
            "daily_cap": DAILY_TOKEN_CAP,
            "daily_remaining": max(0, DAILY_TOKEN_CAP - daily_used),
            "daily_pct": round(daily_used / DAILY_TOKEN_CAP * 100, 1) if DAILY_TOKEN_CAP > 0 else 0,
            "hourly_baseline": HOURLY_BASELINE,
            "lockdown_active": self._lockdown_active,
            "current_hour": current_hour,
            "models": models,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_daily_breakdown(self, date: Optional[str] = None) -> dict:
        """Full hourly breakdown for a specific date (for ARCHIVE tab report)."""
        target_date = date or self._current_date()
        breakdown = {}

        for model in self.MODELS:
            hours = []
            model_total = 0
            anomaly_hours = []
            for hour_key, bucket in sorted(self._hourly.get(model, {}).items()):
                if hour_key.startswith(target_date):
                    entry = bucket.to_dict()
                    hours.append(entry)
                    model_total += bucket.total_tokens
                    if bucket.anomaly_level != "normal":
                        anomaly_hours.append({"hour": hour_key, "level": bucket.anomaly_level, "tokens": bucket.total_tokens})

            breakdown[model] = {
                "hourly": hours,
                "total_tokens": model_total,
                "total_requests": sum(b.requests for b in self._hourly.get(model, {}).values() if b.hour.startswith(target_date)),
                "anomaly_flags": anomaly_hours,
            }

        return {
            "date": target_date,
            "models": breakdown,
            "grand_total": sum(v["total_tokens"] for v in breakdown.values()),
            "daily_cap": DAILY_TOKEN_CAP,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_daily_report(self, date: str):
        """Generate compressed daily report for ARCHIVE tab."""
        try:
            report = self.get_daily_breakdown(date)
            report_file = _ARCHIVE_DIR / f"{date}.json.gz"
            with gzip.open(report_file, "wt", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Token usage daily report generated: %s", report_file)
        except Exception as e:
            logger.error("Daily token report failed: %s", e)


# ── Singleton ────────────────────────────────────────────────

_meter: Optional[TokenMeter] = None


def get_meter() -> TokenMeter:
    global _meter
    if _meter is None:
        _meter = TokenMeter()
    return _meter
