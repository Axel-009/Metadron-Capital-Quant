"""
Metadron Capital — AI-Newton Discovery Service Wrapper

PM2-managed service that wraps the AINewtonDiscoveryWorker as a
long-running process with a lightweight health endpoint.

The worker runs symbolic regression experiments to discover market
microstructure patterns and feeds them into the PatternDiscoveryEngine.
This wrapper adds PM2-compatible lifecycle management and a health
check for monitoring.

Usage:
    python3 engine/bridges/ainewton_service.py
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("ainewton-service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Import the actual worker ──────────────────────────────────────

_worker_available = False
_AINewtonDiscoveryWorker = None

try:
    from engine.bridges.ainewton_discovery_worker import AINewtonDiscoveryWorker
    _AINewtonDiscoveryWorker = AINewtonDiscoveryWorker
    _worker_available = True
    logger.info("AINewtonDiscoveryWorker: AVAILABLE")
except ImportError as e:
    logger.warning(f"AINewtonDiscoveryWorker import failed: {e}")
    logger.warning("Service will run in health-only mode")


# ─── Service Wrapper ───────────────────────────────────────────────

class AINewtonService:
    """Wraps AINewtonDiscoveryWorker with lifecycle management and health reporting."""

    def __init__(self):
        self.worker = None
        self.worker_thread = None
        self.running = False
        self.started_at = None
        self.health_port = int(os.environ.get("AINEWTON_HEALTH_PORT", "0"))
        self._http_server = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def _run_worker(self):
        """Run the discovery worker in a background thread."""
        try:
            self.worker = _AINewtonDiscoveryWorker()
            self.worker.run()
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            self.running = False

    def _get_health(self) -> dict:
        """Build health status dict."""
        status = {
            "status": "healthy" if self.running and self.worker else "degraded",
            "service": "ainewton-service",
            "worker_available": _worker_available,
            "started_at": self.started_at,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
        }

        if self.worker:
            status.update({
                "cycle_count": getattr(self.worker, "cycle_count", 0),
                "current_experiment": getattr(self.worker, "current_experiment", None),
                "discoveries_count": len(getattr(self.worker, "discovered_patterns", [])),
                "worker_running": getattr(self.worker, "running", False),
            })

        return status

    def _start_health_server(self):
        """Start a minimal HTTP health endpoint if port is configured."""
        if self.health_port <= 0:
            return

        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler

            service = self

            class HealthHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        body = json.dumps(service._get_health()).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_response(404)
                        self.end_headers()

                def log_message(self, format, *args):
                    pass  # Suppress default access logs

            self._http_server = HTTPServer(("0.0.0.0", self.health_port), HealthHandler)
            health_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="ainewton-health",
            )
            health_thread.start()
            logger.info(f"Health endpoint listening on port {self.health_port}")

        except Exception as e:
            logger.warning(f"Failed to start health server: {e}")

    def start(self):
        """Start the AI-Newton discovery service."""
        logger.info("AI-Newton Discovery Service starting...")
        self.started_at = time.time()
        self.running = True

        self._start_health_server()

        if not _worker_available:
            logger.error("Worker not available — entering idle loop")
            self._idle_loop()
            return

        # Run worker in a thread so we can handle signals in main thread
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            daemon=False,
            name="ainewton-worker",
        )
        self.worker_thread.start()
        logger.info("Worker thread started")

        # Main thread: wait for worker or signal
        try:
            while self.running and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5.0)
        except KeyboardInterrupt:
            self.stop()

        logger.info("AI-Newton Discovery Service exited")

    def _idle_loop(self):
        """Run an idle loop when the worker is not available (health-only mode)."""
        try:
            while self.running:
                time.sleep(30)
                logger.info(
                    f"Health-only mode — waiting for worker availability "
                    f"(uptime: {round(time.time() - self.started_at)}s)"
                )
        except KeyboardInterrupt:
            pass

    def stop(self):
        """Gracefully stop the service."""
        logger.info("Stopping AI-Newton Discovery Service...")
        self.running = False

        if self.worker:
            self.worker.running = False

        if self._http_server:
            self._http_server.shutdown()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=30)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not exit within timeout")

        logger.info("Service stopped")


# ─── Entry Point ───────────────────────────────────────────────────

def main():
    """Run the AI-Newton discovery service as a PM2-managed process."""
    service = AINewtonService()
    service.start()


if __name__ == "__main__":
    main()
