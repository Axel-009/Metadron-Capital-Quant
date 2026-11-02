"""
Unified Multi-Language Bridge for Metadron Capital Intelligence Platform.

Provides a single entry point to all language integrations:
- Rust (AI-Newton symbolic physics)
- Go (Kserve ML serving)
- C++/CUDA (NVIDIA deep learning acceleration)
- TypeScript/JavaScript (Frontend dashboards)

Automatically detects available toolchains and selects optimal execution modes.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rust_plugin import RustIntegration
from .go_plugin import GoIntegration
from .cuda_plugin import CUDAIntegration
from .typescript_plugin import TypeScriptIntegration

logger = logging.getLogger(__name__)

PLATFORM_DIR = Path(__file__).parent.parent


@dataclass
class LanguageCapability:
    """Describes a language integration's capabilities."""
    language: str
    integration: Any
    mode: str
    source_files: int
    available: bool
    projects: list[str] = field(default_factory=list)


class UnifiedLanguageBridge:
    """
    Unified entry point for all multi-language integrations.

    Usage:
        bridge = UnifiedLanguageBridge()
        print(bridge.status())

        # Access specific integrations
        rust = bridge.rust          # AI-Newton symbolic physics
        go = bridge.go              # Kserve model serving
        cuda = bridge.cuda          # NVIDIA CUDA acceleration
        ts = bridge.typescript      # Frontend dashboards

        # Compile all available toolchains
        bridge.compile_all()

        # Generate cross-language data pipeline
        bridge.create_data_pipeline(...)
    """

    def __init__(self):
        self.rust = RustIntegration()
        self.go = GoIntegration()
        self.cuda = CUDAIntegration()
        self.typescript = TypeScriptIntegration()
        logger.info("UnifiedLanguageBridge initialized")

    def status(self) -> dict[str, LanguageCapability]:
        """Get status of all language integrations."""
        rust_info = self.rust.get_source_info()
        go_info = self.go.get_source_info()
        cuda_info = self.cuda.get_source_info()
        ts_info = self.typescript.get_source_info()

        return {
            "rust": LanguageCapability(
                language="Rust",
                integration=self.rust,
                mode=self.rust.mode,
                source_files=rust_info["rust_files"] + rust_info["lalrpop_files"],
                available=self.rust.mode != "fallback" or rust_info["rust_files"] > 0,
                projects=rust_info.get("modules", []),
            ),
            "go": LanguageCapability(
                language="Go",
                integration=self.go,
                mode=self.go.mode,
                source_files=go_info["go_files"],
                available=go_info["go_files"] > 0,
                projects=go_info.get("packages", [])[:10],
            ),
            "cuda": LanguageCapability(
                language="C++/CUDA",
                integration=self.cuda,
                mode=self.cuda.mode,
                source_files=(cuda_info["cpp_files"] + cuda_info["cuda_files"]
                              + cuda_info["header_files"]),
                available=cuda_info["cpp_files"] + cuda_info["cuda_files"] > 0,
                projects=cuda_info.get("projects", []),
            ),
            "typescript": LanguageCapability(
                language="TypeScript/JavaScript",
                integration=self.typescript,
                mode="node" if len(self.typescript.projects) > 0 else "none",
                source_files=sum(
                    p.get("ts_files", 0) + p.get("tsx_files", 0)
                    + p.get("js_files", 0) + p.get("vue_files", 0)
                    for p in ts_info.get("projects", {}).values()
                ),
                available=len(self.typescript.projects) > 0,
                projects=list(self.typescript.projects.keys()),
            ),
        }

    def compile_all(self) -> dict[str, bool]:
        """Attempt to compile all language integrations."""
        results = {}
        results["rust"] = self.rust.compile()
        results["go"] = self.go.compile()
        results["cuda"] = self.cuda.compile()
        # TypeScript projects need per-project install + build
        for name, project in self.typescript.projects.items():
            install_result = project.install()
            if install_result.get("success"):
                build_result = project.build()
                results[f"ts:{name}"] = build_result.get("success", False)
            else:
                results[f"ts:{name}"] = False
        return results

    def create_data_pipeline(self, name: str,
                             stages: list[dict]) -> "DataPipeline":
        """
        Create a cross-language data pipeline.

        Stages can reference any language:
        [
            {"lang": "python", "func": "load_data", "args": {...}},
            {"lang": "rust", "func": "symbolic_transform", "args": {...}},
            {"lang": "cuda", "func": "gpu_inference", "args": {...}},
            {"lang": "go", "func": "serve_model", "args": {...}},
            {"lang": "typescript", "func": "render_dashboard", "args": {...}},
        ]
        """
        return DataPipeline(name=name, stages=stages, bridge=self)

    def summary(self) -> str:
        """Human-readable summary of all integrations."""
        status = self.status()
        lines = ["=== Metadron Capital Multi-Language Bridge ===\n"]
        for key, cap in status.items():
            icon = "+" if cap.available else "-"
            lines.append(
                f"  [{icon}] {cap.language}: {cap.source_files} files "
                f"(mode: {cap.mode})"
            )
            if cap.projects:
                lines.append(f"      Projects: {', '.join(cap.projects[:5])}")
        total = sum(c.source_files for c in status.values())
        lines.append(f"\n  Total source files: {total}")
        return "\n".join(lines)


@dataclass
class DataPipeline:
    """Cross-language data pipeline."""
    name: str
    stages: list[dict]
    bridge: UnifiedLanguageBridge
    results: list[dict] = field(default_factory=list)

    def execute(self, initial_data: Any = None) -> list[dict]:
        """Execute pipeline stages sequentially."""
        data = initial_data
        self.results = []

        for i, stage in enumerate(self.stages):
            lang = stage.get("lang", "python")
            func = stage.get("func", "")
            args = stage.get("args", {})

            logger.info(f"Pipeline '{self.name}' stage {i}: {lang}.{func}")

            try:
                if lang == "python":
                    # Direct Python execution
                    result = {"data": data, "stage": i, "status": "pass-through"}
                elif lang == "rust":
                    expr = self.bridge.rust.create_expression(
                        args.get("expression", "0"))
                    result = {
                        "value": expr.evaluate(args.get("bindings", {})),
                        "stage": i,
                    }
                elif lang == "go":
                    client = self.bridge.go.create_serving_client(
                        model_name=args.get("model", ""))
                    result = client.predict(args.get("instances", []))
                elif lang == "cuda":
                    config = self.bridge.cuda.create_faster_transformer(
                        **args.get("config", {}))
                    result = {"config": config.__dict__, "stage": i}
                elif lang == "typescript":
                    project = self.bridge.typescript.get_project(
                        args.get("project", ""))
                    if project:
                        api = self.bridge.typescript.create_api_bridge(
                            args.get("project", ""))
                        result = {"project": project.name,
                                  "scripts": project.get_scripts(), "stage": i}
                    else:
                        result = {"error": "Project not found", "stage": i}
                else:
                    result = {"error": f"Unknown language: {lang}", "stage": i}

                data = result
                self.results.append({"stage": i, "lang": lang, "func": func,
                                     "status": "success", "result": result})
            except Exception as e:
                self.results.append({"stage": i, "lang": lang, "func": func,
                                     "status": "error", "error": str(e)})
                logger.error(f"Pipeline stage {i} failed: {e}")

        return self.results
