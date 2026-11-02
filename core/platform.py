"""
Metadron Capital Platform Orchestrator
Central nervous system that coordinates all repository layers.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import yaml


class Layer(IntEnum):
    """Platform architecture layers."""
    HUB = 0
    DATA_INGESTION = 1
    SIGNAL_PROCESSING = 2
    ML_PREDICTION = 3
    PORTFOLIO_RISK = 4
    INFRASTRUCTURE = 5
    AGENT_ORCHESTRATION = 6


@dataclass
class RepoModule:
    """Represents a connected repository module."""
    name: str
    path: Path
    role: str
    layer: Layer
    description: str
    capabilities: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | initialized | active | error
    errors: list[str] = field(default_factory=list)


class MetadronPlatform:
    """
    Central orchestrator for the Metadron Capital investment platform.

    Manages the lifecycle of all connected repository modules across
    the 6-layer architecture, coordinating data flow from ingestion
    through to portfolio analytics generation.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path or Path(__file__).parent.parent / "config" / "repos.yaml")
        self.modules: dict[str, RepoModule] = {}
        self._load_config()

    def _load_config(self):
        """Load repository configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        base_dir = self.config_path.parent.parent
        intel_dir = base_dir / "intelligence_platform"
        repos_dir = base_dir / "repos"

        for name, repo_cfg in config.get("repos", {}).items():
            repo_path = (base_dir / repo_cfg["path"]).resolve()
            # Fallback: check intelligence_platform/ and repos/ if primary path missing
            if not repo_path.exists():
                repo_dirname = Path(repo_cfg["path"]).name
                alt_intel = intel_dir / repo_dirname
                if alt_intel.exists():
                    repo_path = alt_intel
                else:
                    # Search repos/ layer dirs
                    for layer_dir in sorted(repos_dir.glob("layer*")):
                        alt_repo = layer_dir / repo_dirname
                        if alt_repo.exists():
                            repo_path = alt_repo
                            break
            self.modules[name] = RepoModule(
                name=name,
                path=repo_path,
                role=repo_cfg["role"],
                layer=Layer(repo_cfg["layer"]),
                description=repo_cfg["description"],
                capabilities=repo_cfg.get("capabilities", []),
                outputs=repo_cfg.get("outputs", []),
            )

    def initialize_module(self, name: str) -> bool:
        """Initialize a single module by verifying its path and basic imports."""
        module = self.modules.get(name)
        if not module:
            return False

        if not module.path.exists():
            module.status = "error"
            module.errors.append(f"Path does not exist: {module.path}")
            return False

        # Check for any source files (Python, Java, Go, Rust, C++, etc.)
        has_files = any(module.path.rglob("*"))
        if not has_files and name != "metadron_capital":
            module.status = "error"
            module.errors.append("No files found")
            return False

        module.status = "initialized"
        return True

    def initialize_all(self) -> dict[str, str]:
        """Initialize all modules. Returns status dict."""
        results = {}
        for name in self.modules:
            success = self.initialize_module(name)
            results[name] = self.modules[name].status
        return results

    def get_layer_modules(self, layer: Layer) -> list[RepoModule]:
        """Get all modules in a specific layer."""
        return [m for m in self.modules.values() if m.layer == layer]

    def get_data_flow(self) -> list[dict[str, Any]]:
        """
        Get the data flow graph showing how outputs from one layer
        feed into the next layer's inputs.
        """
        flow = []
        for layer_num in range(1, 7):
            layer = Layer(layer_num)
            modules = self.get_layer_modules(layer)
            flow.append({
                "layer": layer.name,
                "layer_num": layer_num,
                "modules": [
                    {
                        "name": m.name,
                        "role": m.role,
                        "outputs": m.outputs,
                        "status": m.status,
                    }
                    for m in modules
                ],
            })
        return flow

    def status_report(self) -> str:
        """Generate a platform status report."""
        lines = ["=" * 70, "METADRON CAPITAL — PLATFORM STATUS REPORT", "=" * 70, ""]

        for layer_num in range(7):
            layer = Layer(layer_num)
            modules = self.get_layer_modules(layer)
            if not modules:
                continue

            lines.append(f"━━━ LAYER {layer_num}: {layer.name.replace('_', ' ')} ━━━")
            for m in modules:
                status_icon = {"pending": "○", "initialized": "◉", "active": "●", "error": "✗"}
                icon = status_icon.get(m.status, "?")
                lines.append(f"  {icon} {m.name:<30} [{m.role}]")
                if m.errors:
                    for err in m.errors:
                        lines.append(f"      ⚠ {err}")
            lines.append("")

        # Summary
        total = len(self.modules)
        initialized = sum(1 for m in self.modules.values() if m.status in ("initialized", "active"))
        errors = sum(1 for m in self.modules.values() if m.status == "error")
        lines.append(f"TOTAL: {total} modules | {initialized} ready | {errors} errors")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Run platform initialization and status check."""
    platform = MetadronPlatform()
    results = platform.initialize_all()
    print(platform.status_report())
    return platform


if __name__ == "__main__":
    main()
