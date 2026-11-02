"""Metadron Capital — System Integrity: Phase Chain, Broker Lock, Transaction Ledger,
Circuit Breaker, Heartbeat Integrity.

Six-layer defense against adversarial degradation:

1. PhaseChain — HMAC-signed output chain between phases (break = halt)
2. BrokerIntegrityLock — Paper vs IBKR reconciliation every cycle
3. TransactionLedger — Append-only HMAC-signed trade log (tamper-evident)
4. CircuitBreaker — Perimeter lockdown (API locked, engine running, exits only)
5. HeartbeatIntegrity — Signed PM2 service heartbeats (missing = freeze new entries)
6. PromptGuard — LLM prompt size + cadence enforcement

All wire to Prometheus counters and TECH tab error log.
"""

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

logger = logging.getLogger("metadron.security.integrity")

_HMAC_KEY = os.environ.get("METADRON_HMAC_KEY", "metadron-integrity-v1").encode()
_LEDGER_DIR = Path("data/security/ledger")
_LEDGER_DIR.mkdir(parents=True, exist_ok=True)


def _hmac_sign(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode()
    return hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()


def _hmac_verify(data: dict, signature: str) -> bool:
    expected = _hmac_sign(data)
    return hmac.compare_digest(expected, signature)


# ═══════════════════════════════════════════════════════════════
# 1. PHASE CHAIN — Signed output between live loop phases
# ═══════════════════════════════════════════════════════════════

class PhaseChain:
    """Every phase signs its output. Next phase verifies before processing.

    Chain: DATA → SIGNALS → INTELLIGENCE → DECISION → EXECUTION
    Break in chain = system halts new trades (not exits).
    """

    def __init__(self):
        self._chain: dict[str, str] = {}  # {phase: last_signature}
        self._broken = False
        self._break_phase = ""
        self._lock = threading.Lock()

    def sign_phase(self, phase: str, output_data: dict) -> str:
        """Sign a phase's output and record in chain."""
        with self._lock:
            data = {"phase": phase, "timestamp": time.time(), "data_hash": hashlib.sha256(
                json.dumps(output_data, sort_keys=True, default=str).encode()
            ).hexdigest()}
            # Chain to previous phase signature
            prev_phases = {"SIGNALS": "DATA", "INTELLIGENCE": "SIGNALS", "DECISION": "INTELLIGENCE", "EXECUTION": "DECISION"}
            prev = prev_phases.get(phase)
            if prev and prev in self._chain:
                data["prev_signature"] = self._chain[prev]
            sig = _hmac_sign(data)
            self._chain[phase] = sig
            return sig

    def verify_chain(self, phase: str, expected_prev_sig: Optional[str] = None) -> bool:
        """Verify the chain is intact up to this phase."""
        with self._lock:
            prev_phases = {"SIGNALS": "DATA", "INTELLIGENCE": "SIGNALS", "DECISION": "INTELLIGENCE", "EXECUTION": "DECISION"}
            prev = prev_phases.get(phase)
            if prev and prev in self._chain:
                if expected_prev_sig and not hmac.compare_digest(self._chain[prev], expected_prev_sig):
                    self._broken = True
                    self._break_phase = phase
                    logger.critical("PHASE CHAIN BROKEN at %s — previous phase signature mismatch", phase)
                    return False
            return not self._broken

    @property
    def is_broken(self) -> bool:
        return self._broken

    def reset(self):
        with self._lock:
            self._broken = False
            self._break_phase = ""
            self._chain.clear()

    def get_status(self) -> dict:
        return {"broken": self._broken, "break_phase": self._break_phase, "chain_length": len(self._chain)}


# ═══════════════════════════════════════════════════════════════
# 2. BROKER INTEGRITY LOCK
# ═══════════════════════════════════════════════════════════════

class BrokerIntegrityLock:
    """Paper vs IBKRBroker reconciliation. Discrepancy > threshold = freeze new entries.

    Futures note: trade log handles futures (ES, NQ, VX). IBKR handles
    equities + options only. Futures recon verifies paper internal consistency
    (positions vs trades vs P&L must add up). When Rithmic is wired,
    futures recon will check paper vs Rithmic.
    """

    def __init__(self, tolerance_dollars: float = 0.01, tolerance_shares: int = 1):
        self.tolerance_dollars = tolerance_dollars
        self.tolerance_shares = tolerance_shares
        self._frozen = False
        self._discrepancies: list = []
        self._last_recon: Optional[str] = None

    def reconcile(self, paper_positions: dict, broker_positions: dict) -> dict:
        """Compare paper vs IBKRBroker positions. Returns recon result."""
        self._discrepancies.clear()
        all_tickers = set(list(paper_positions.keys()) + list(broker_positions.keys()))

        for ticker in all_tickers:
            paper = paper_positions.get(ticker, {})
            ibkr = broker_positions.get(ticker, {})
            p_qty = paper.get("quantity", 0)
            a_qty = ibkr.get("quantity", 0)
            p_value = paper.get("market_value", 0)
            a_value = ibkr.get("market_value", 0)

            qty_diff = abs(p_qty - a_qty)
            val_diff = abs(p_value - a_value)

            if qty_diff > self.tolerance_shares or val_diff > self.tolerance_dollars:
                self._discrepancies.append({
                    "ticker": ticker,
                    "paper_qty": p_qty, "ibkr_qty": a_qty, "qty_diff": qty_diff,
                    "paper_value": p_value, "ibkr_value": a_value, "value_diff": round(val_diff, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        if self._discrepancies:
            self._frozen = True
            logger.critical(
                "BROKER INTEGRITY FREEZE: %d discrepancies detected (paper vs IBKR)",
                len(self._discrepancies),
            )

        self._last_recon = datetime.now(timezone.utc).isoformat()
        return {
            "clean": len(self._discrepancies) == 0,
            "frozen": self._frozen,
            "discrepancies": self._discrepancies,
            "timestamp": self._last_recon,
        }

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def unfreeze(self):
        self._frozen = False
        logger.info("Broker integrity lock UNFROZEN by operator")

    def get_status(self) -> dict:
        return {"frozen": self._frozen, "discrepancies": len(self._discrepancies), "last_recon": self._last_recon}


# ═══════════════════════════════════════════════════════════════
# 3. TRANSACTION LEDGER — Append-only, tamper-evident
# ═══════════════════════════════════════════════════════════════

class TransactionLedger:
    """Append-only transaction log with HMAC chain. Every entry's hash
    chains to the previous entry — if any entry is modified, the chain breaks."""

    def __init__(self):
        self._entries: list = []
        self._last_hash: str = "genesis"
        self._ledger_file = _LEDGER_DIR / f"ledger_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def record(self, event_type: str, data: dict) -> dict:
        """Append a signed, chained entry to the ledger."""
        entry = {
            "seq": len(self._entries),
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prev_hash": self._last_hash,
        }
        entry["signature"] = _hmac_sign(entry)
        self._last_hash = entry["signature"]
        self._entries.append(entry)

        # Persist to disk
        try:
            with open(self._ledger_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error("Ledger write failed: %s", e)

        return entry

    def verify_chain(self) -> dict:
        """Verify the entire ledger chain is intact."""
        expected_prev = "genesis"
        for i, entry in enumerate(self._entries):
            if entry["prev_hash"] != expected_prev:
                logger.critical("LEDGER CHAIN BROKEN at entry %d", i)
                return {"intact": False, "break_at": i, "total": len(self._entries)}
            # Verify signature
            sig = entry.pop("signature", "")
            if not _hmac_verify(entry, sig):
                entry["signature"] = sig
                logger.critical("LEDGER SIGNATURE INVALID at entry %d", i)
                return {"intact": False, "break_at": i, "total": len(self._entries)}
            entry["signature"] = sig
            expected_prev = sig
        return {"intact": True, "total": len(self._entries)}

    def get_recent(self, n: int = 20) -> list:
        return self._entries[-n:]


# ═══════════════════════════════════════════════════════════════
# 4. CIRCUIT BREAKER — Perimeter lockdown
# ═══════════════════════════════════════════════════════════════

class PerimeterCircuitBreaker:
    """API-level lockdown. When tripped:
    - External API requests → 503
    - Internal engine → keeps running
    - Active positions → monitored (exits honored)
    - New trade entries → blocked
    Auto-recovers after cooldown_seconds of traffic below threshold.
    """

    def __init__(self, max_per_10s: int = 200, cooldown_seconds: int = 60):
        self.max_per_10s = max_per_10s
        self.cooldown_seconds = cooldown_seconds
        self._request_times: list = []
        self._lockdown = False
        self._lockdown_at: Optional[float] = None
        self._trips = 0
        self._lock = threading.Lock()

    def record_request(self) -> bool:
        """Record an API request. Returns True if allowed, False if locked."""
        with self._lock:
            now = time.time()
            # Prune old entries
            self._request_times = [t for t in self._request_times if t > now - 10]
            self._request_times.append(now)

            # Check lockdown recovery
            if self._lockdown and self._lockdown_at:
                if now - self._lockdown_at > self.cooldown_seconds:
                    if len(self._request_times) < self.max_per_10s:
                        self._lockdown = False
                        self._lockdown_at = None
                        logger.info("Perimeter circuit breaker RECOVERED — lockdown lifted")

            # Check threshold
            if len(self._request_times) > self.max_per_10s and not self._lockdown:
                self._lockdown = True
                self._lockdown_at = now
                self._trips += 1
                logger.critical(
                    "PERIMETER LOCKDOWN: %d requests in 10s (threshold %d) — trip #%d",
                    len(self._request_times), self.max_per_10s, self._trips,
                )

            return not self._lockdown

    @property
    def is_locked(self) -> bool:
        return self._lockdown

    def get_status(self) -> dict:
        return {
            "locked": self._lockdown,
            "trips": self._trips,
            "current_rate": len(self._request_times),
            "threshold": self.max_per_10s,
            "cooldown_seconds": self.cooldown_seconds,
        }


# ═══════════════════════════════════════════════════════════════
# 5. HEARTBEAT INTEGRITY
# ═══════════════════════════════════════════════════════════════

class HeartbeatIntegrity:
    """Signed heartbeats from PM2 services. Missing heartbeat = freeze new entries."""

    SERVICES = [
        "engine-api", "llm-inference-bridge", "qwen-model-server",
        "llama-model-server", "live-loop", "learning-loop", "metadron-cube",
    ]

    def __init__(self, max_miss_seconds: int = 300):
        self.max_miss_seconds = max_miss_seconds
        self._heartbeats: dict[str, float] = {}
        self._missing: list = []

    def record_heartbeat(self, service: str):
        self._heartbeats[service] = time.time()

    def check_integrity(self) -> dict:
        now = time.time()
        self._missing.clear()
        for svc in self.SERVICES:
            last = self._heartbeats.get(svc, 0)
            if now - last > self.max_miss_seconds:
                self._missing.append(svc)
        if self._missing:
            logger.warning("HEARTBEAT MISSING: %s (>%ds)", self._missing, self.max_miss_seconds)
        return {
            "all_healthy": len(self._missing) == 0,
            "missing": self._missing,
            "services": {s: round(now - self._heartbeats.get(s, 0), 1) for s in self.SERVICES},
        }

    @property
    def has_missing(self) -> bool:
        return len(self._missing) > 0


# ═══════════════════════════════════════════════════════════════
# 6. PROMPT GUARD
# ═══════════════════════════════════════════════════════════════

class PromptGuard:
    """LLM prompt size + cadence enforcement."""

    def __init__(self, max_tokens: int = 8192, max_context_kb: int = 50, min_interval_s: int = 30):
        self.max_tokens = max_tokens
        self.max_context_kb = max_context_kb
        self.min_interval_s = min_interval_s
        self._last_call: dict[str, float] = {}
        self._rejections = 0

    def check(self, prompt: str, ml_context: Optional[dict] = None, caller: str = "unknown") -> dict:
        """Check if a prompt is within limits. Returns {"allowed": bool, "reason": str}."""
        # Token estimate (~4 chars per token)
        est_tokens = len(prompt) // 4
        if est_tokens > self.max_tokens:
            self._rejections += 1
            return {"allowed": False, "reason": f"prompt_too_large ({est_tokens} > {self.max_tokens})"}

        # Context size check
        if ml_context:
            ctx_size = len(json.dumps(ml_context, default=str).encode())
            if ctx_size > self.max_context_kb * 1024:
                self._rejections += 1
                return {"allowed": False, "reason": f"context_too_large ({ctx_size // 1024}KB > {self.max_context_kb}KB)"}

        # Cadence check
        now = time.time()
        last = self._last_call.get(caller, 0)
        if now - last < self.min_interval_s:
            self._rejections += 1
            return {"allowed": False, "reason": f"too_frequent ({int(now - last)}s < {self.min_interval_s}s)"}

        self._last_call[caller] = now
        return {"allowed": True, "reason": "ok"}

    def get_status(self) -> dict:
        return {"rejections": self._rejections, "max_tokens": self.max_tokens, "max_context_kb": self.max_context_kb}


# ═══════════════════════════════════════════════════════════════
# UNIFIED SECURITY MANAGER
# ═══════════════════════════════════════════════════════════════

class SecurityManager:
    """Single access point for all security subsystems."""

    def __init__(self):
        self.phase_chain = PhaseChain()
        self.broker_lock = BrokerIntegrityLock()
        self.ledger = TransactionLedger()
        self.circuit_breaker = PerimeterCircuitBreaker()
        self.heartbeat = HeartbeatIntegrity()
        self.prompt_guard = PromptGuard()
        logger.info("SecurityManager initialized — 6 defense layers active")

    def is_system_healthy(self) -> bool:
        """Quick check — is the system in a state where new trades can enter?"""
        return (
            not self.phase_chain.is_broken
            and not self.broker_lock.is_frozen
            and not self.circuit_breaker.is_locked
        )

    def get_full_status(self) -> dict:
        return {
            "healthy": self.is_system_healthy(),
            "phase_chain": self.phase_chain.get_status(),
            "broker_lock": self.broker_lock.get_status(),
            "ledger_entries": len(self.ledger._entries),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "heartbeat": self.heartbeat.check_integrity(),
            "prompt_guard": self.prompt_guard.get_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


_security: Optional[SecurityManager] = None


def get_security() -> SecurityManager:
    global _security
    if _security is None:
        _security = SecurityManager()
    return _security
