"""
KServe v2 REST inference client with circuit breaker pattern.
Falls back to simple prediction when circuit is open or KServe is unreachable.
Uses only urllib.request -- no requests dependency.
"""

import json
import logging
import time

import numpy as np

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except Exception:
    URLLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# InvestmentModelServer from Kserve integration — manifest gen, A/B testing,
# IC tracking, batch predict orchestration.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    _kserve_spec = _ilu.spec_from_file_location(
        "kserve_integration",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent
            / "intelligence_platform" / "Kserve"
            / "investment_platform_integration.py"),
    )
    _kserve_mod = _ilu.module_from_spec(_kserve_spec)
    _kserve_spec.loader.exec_module(_kserve_mod)
    InvestmentModelServer = _kserve_mod.InvestmentModelServer
    KSERVE_INTEGRATION_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError, Exception):
    InvestmentModelServer = None
    KSERVE_INTEGRATION_AVAILABLE = False
    logger.info("Kserve InvestmentModelServer unavailable — transport-only mode")


class CircuitState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class KServeAdapter:
    """
    KServe v2 inference protocol client with circuit breaker.

    Circuit breaker transitions:
      CLOSED  -> OPEN       after ``failure_threshold`` consecutive failures
      OPEN    -> HALF_OPEN  after ``recovery_timeout`` seconds
      HALF_OPEN -> CLOSED   on first success
      HALF_OPEN -> OPEN     on first failure
    """

    DEFAULT_ENDPOINT = "http://localhost:8080"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        request_timeout: float = 5.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    # ------------------------------------------------------------------
    # Circuit breaker logic
    # ------------------------------------------------------------------

    def get_circuit_state(self) -> str:
        """Return the current circuit breaker state, applying time-based transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioned to HALF_OPEN.")
        return self._state

    def _record_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            logger.info("Circuit breaker transitioned to CLOSED.")

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker transitioned back to OPEN from HALF_OPEN.")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures.", self._failure_count
            )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _do_request(self, url: str, payload: dict) -> dict:
        """Send a POST request using urllib.request. Raises on failure."""
        if not URLLIB_AVAILABLE:
            raise RuntimeError("urllib not available")

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    # ------------------------------------------------------------------
    # Fallback prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_predict(features: np.ndarray) -> dict:
        """Simple linear prediction as fallback when KServe is unavailable."""
        if len(features) == 0:
            return {"prediction": 0.0, "fallback": True}
        # Weighted sum with decaying weights
        n = len(features)
        weights = np.exp(-0.1 * np.arange(n)[::-1])
        weights /= weights.sum()
        prediction = float(np.dot(features, weights))
        return {"prediction": prediction, "fallback": True}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if the KServe endpoint is reachable."""
        ep = endpoint or self.DEFAULT_ENDPOINT
        url = f"{ep}/v2/health/ready"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def infer(
        self,
        model_name: str,
        features: np.ndarray,
        endpoint: str | None = None,
    ) -> dict:
        """
        Run inference against a KServe v2 model endpoint.

        Parameters
        ----------
        model_name : name of the deployed model
        features : numpy array of input features
        endpoint : KServe base URL (default: http://localhost:8080)

        Returns
        -------
        dict with prediction results, or fallback prediction if circuit is open.
        """
        ep = endpoint or self.DEFAULT_ENDPOINT
        current_state = self.get_circuit_state()

        if current_state == CircuitState.OPEN:
            logger.warning("Circuit OPEN for %s. Returning fallback prediction.", model_name)
            return self._fallback_predict(features)

        # Build KServe v2 inference request
        url = f"{ep}/v2/models/{model_name}/infer"
        payload = {
            "inputs": [
                {
                    "name": "input_0",
                    "shape": list(features.shape),
                    "datatype": "FP64",
                    "data": features.tolist(),
                }
            ],
        }

        try:
            response = self._do_request(url, payload)
            self._record_success()

            # Parse KServe v2 response
            outputs = response.get("outputs", [])
            if outputs:
                prediction = outputs[0].get("data", [0.0])
                if isinstance(prediction, list):
                    prediction = prediction[0] if prediction else 0.0
            else:
                prediction = 0.0

            return {
                "prediction": float(prediction),
                "model_name": model_name,
                "fallback": False,
            }

        except Exception as exc:
            logger.warning("KServe inference failed for %s: %s", model_name, exc)
            self._record_failure()
            return self._fallback_predict(features)

    def batch_predict(
        self,
        model_name: str,
        feature_batch: list[np.ndarray],
        endpoint: str | None = None,
    ) -> list[dict]:
        """
        Batch inference via InvestmentModelServer if available, else sequential.

        Uses intelligence_platform/Kserve InvestmentModelServer.batch_predict()
        for optimized batching when available.
        """
        if KSERVE_INTEGRATION_AVAILABLE and InvestmentModelServer is not None:
            try:
                server = InvestmentModelServer()
                return server.batch_predict(model_name, feature_batch)
            except Exception as exc:
                logger.warning("InvestmentModelServer batch_predict failed: %s", exc)

        # Fallback: sequential inference
        return [self.infer(model_name, f, endpoint) for f in feature_batch]

    def get_model_server(self):
        """
        Get the InvestmentModelServer instance for manifest generation,
        A/B testing, and IC tracking.

        Returns None if Kserve integration is unavailable.
        """
        if KSERVE_INTEGRATION_AVAILABLE and InvestmentModelServer is not None:
            try:
                return InvestmentModelServer()
            except Exception as exc:
                logger.warning("Failed to create InvestmentModelServer: %s", exc)
        return None
