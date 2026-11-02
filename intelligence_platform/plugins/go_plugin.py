"""
Go Integration Plugin - Kserve ML Model Serving.

Bridges the Go-based Kserve inference serving infrastructure into Python via:
1. gRPC/REST API client (runtime mode)
2. Subprocess execution of Go binaries
3. Python reimplementation of key serving patterns
"""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

GO_SOURCE_DIR = Path(__file__).parent.parent / "Kserve"
KSERVE_CONFIGS = GO_SOURCE_DIR / "config"


@dataclass
class InferenceServiceSpec:
    """Python representation of a Kserve InferenceService."""
    name: str
    predictor_model_format: str  # sklearn, tensorflow, pytorch, xgboost, etc.
    predictor_runtime: Optional[str] = None
    predictor_storage_uri: Optional[str] = None
    transformer_image: Optional[str] = None
    explainer_type: Optional[str] = None
    min_replicas: int = 1
    max_replicas: int = 3
    target_utilization: int = 80
    namespace: str = "default"

    def to_k8s_manifest(self) -> dict:
        """Generate Kubernetes YAML-compatible dict."""
        spec = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": self.predictor_model_format},
                        "runtime": self.predictor_runtime,
                        "storageUri": self.predictor_storage_uri,
                    },
                    "minReplicas": self.min_replicas,
                    "maxReplicas": self.max_replicas,
                },
            },
        }
        if self.transformer_image:
            spec["spec"]["transformer"] = {
                "containers": [{"image": self.transformer_image}]
            }
        if self.explainer_type:
            spec["spec"]["explainer"] = {"type": self.explainer_type}
        return spec


@dataclass
class ModelServingClient:
    """Python client for Kserve inference endpoints."""
    base_url: str = "http://localhost:8080"
    model_name: str = ""
    timeout: float = 30.0

    def predict(self, instances: list[dict | list]) -> dict:
        """Send prediction request to a Kserve model endpoint."""
        import urllib.request
        url = f"{self.base_url}/v1/models/{self.model_name}:predict"
        payload = json.dumps({"instances": instances}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def explain(self, instances: list[dict | list]) -> dict:
        """Send explanation request."""
        import urllib.request
        url = f"{self.base_url}/v1/models/{self.model_name}:explain"
        payload = json.dumps({"instances": instances}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> dict:
        """Check model readiness."""
        import urllib.request
        url = f"{self.base_url}/v1/models/{self.model_name}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"ready": False, "error": str(e)}


class GoIntegration:
    """
    Integration layer for Kserve Go-based ML serving infrastructure.

    Supports:
    1. 'api' - REST/gRPC client to running Kserve instance
    2. 'subprocess' - Direct Go binary execution
    3. 'config' - Parse and generate Kserve YAML configurations
    4. 'fallback' - Python reimplementation of serving patterns
    """

    def __init__(self, mode: str = "auto"):
        self.source_dir = GO_SOURCE_DIR
        self.config_dir = KSERVE_CONFIGS

        if mode == "auto":
            self.mode = self._detect_mode()
        else:
            self.mode = mode
        logger.info(f"GoIntegration initialized in '{self.mode}' mode")

    def _detect_mode(self) -> str:
        """Auto-detect best available execution mode."""
        if shutil.which("go") and (self.source_dir / "go.mod").exists():
            return "subprocess"
        return "config"

    def compile(self, target: str = "controller") -> bool:
        """Compile Go source."""
        if not shutil.which("go"):
            logger.error("Go toolchain not found")
            return False
        try:
            subprocess.run(
                ["go", "build", "-o", f"bin/{target}", f"./cmd/{target}"],
                cwd=str(self.source_dir), check=True,
                capture_output=True, text=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Go compilation failed: {e.stderr}")
            return False

    def create_inference_service(self, **kwargs) -> InferenceServiceSpec:
        """Create an InferenceService specification."""
        return InferenceServiceSpec(**kwargs)

    def create_serving_client(self, base_url: str = "http://localhost:8080",
                              model_name: str = "") -> ModelServingClient:
        """Create a model serving client."""
        return ModelServingClient(base_url=base_url, model_name=model_name)

    def parse_kserve_configs(self) -> list[dict]:
        """Parse all Kserve YAML configurations from source."""
        configs = []
        if not self.config_dir.exists():
            return configs
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed; returning raw file list")
            return [{"file": str(f)} for f in self.config_dir.rglob("*.yaml")]

        for yaml_file in sorted(self.config_dir.rglob("*.yaml")):
            try:
                with open(yaml_file) as f:
                    docs = list(yaml.safe_load_all(f))
                configs.append({"file": str(yaml_file), "documents": docs})
            except Exception as e:
                configs.append({"file": str(yaml_file), "error": str(e)})
        return configs

    def deploy_model(self, spec: InferenceServiceSpec,
                     kubeconfig: str | None = None) -> dict:
        """Deploy a model using kubectl (requires cluster access)."""
        if not shutil.which("kubectl"):
            return {"error": "kubectl not found", "manifest": spec.to_k8s_manifest()}
        try:
            import yaml
            manifest_yaml = yaml.dump(spec.to_k8s_manifest())
        except ImportError:
            manifest_yaml = json.dumps(spec.to_k8s_manifest())

        cmd = ["kubectl", "apply", "-f", "-"]
        if kubeconfig:
            cmd.extend(["--kubeconfig", kubeconfig])
        try:
            result = subprocess.run(
                cmd, input=manifest_yaml, capture_output=True,
                text=True, timeout=30,
            )
            return {"stdout": result.stdout, "stderr": result.stderr,
                    "returncode": result.returncode}
        except Exception as e:
            return {"error": str(e), "manifest": spec.to_k8s_manifest()}

    def get_source_info(self) -> dict:
        """Return metadata about Go source code."""
        go_files = list(self.source_dir.rglob("*.go")) if self.source_dir.exists() else []
        yaml_files = list(self.source_dir.rglob("*.yaml")) if self.source_dir.exists() else []
        return {
            "source_dir": str(self.source_dir),
            "go_files": len(go_files),
            "yaml_configs": len(yaml_files),
            "mode": self.mode,
            "packages": sorted(set(
                str(f.parent.relative_to(self.source_dir))
                for f in go_files if f.parent != self.source_dir
            ))[:20],  # Top 20 packages
        }
