"""
TypeScript/JavaScript Integration Plugin - Frontend Dashboards & Apps.

Bridges TypeScript/JavaScript frontends from ai-hedgefund, open-bb, MiroFish,
and Stock-technical-prediction-model into the Python platform via:
1. Node.js subprocess execution
2. REST API bridge to running frontend dev servers
3. Python-based SSR/data generation for frontend consumption
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PLATFORM_DIR = Path(__file__).parent.parent


@dataclass
class FrontendProject:
    """Represents a TypeScript/JavaScript frontend project."""
    name: str
    source_dir: Path
    framework: str  # "react", "vue", "next", "vanilla"
    package_manager: str = "npm"  # npm, yarn, pnpm
    dev_port: int = 3000
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)

    @property
    def package_json(self) -> Path:
        return self.source_dir / "package.json"

    def get_scripts(self) -> dict[str, str]:
        """Read available npm scripts."""
        if not self.package_json.exists():
            return {}
        with open(self.package_json) as f:
            data = json.load(f)
        return data.get("scripts", {})

    def get_dependencies(self) -> dict[str, str]:
        """Read project dependencies."""
        if not self.package_json.exists():
            return {}
        with open(self.package_json) as f:
            data = json.load(f)
        deps = {}
        deps.update(data.get("dependencies", {}))
        deps.update(data.get("devDependencies", {}))
        return deps

    def install(self) -> dict:
        """Install dependencies."""
        cmd = [self.package_manager, "install"]
        try:
            result = subprocess.run(
                cmd, cwd=str(self.source_dir),
                capture_output=True, text=True, timeout=120,
            )
            return {"success": result.returncode == 0,
                    "stdout": result.stdout[-500:],
                    "stderr": result.stderr[-500:]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def build(self) -> dict:
        """Build for production."""
        cmd = [self.package_manager, "run", "build"]
        try:
            result = subprocess.run(
                cmd, cwd=str(self.source_dir),
                capture_output=True, text=True, timeout=300,
            )
            return {"success": result.returncode == 0,
                    "stdout": result.stdout[-500:],
                    "stderr": result.stderr[-500:]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_dev_server(self) -> dict:
        """Start development server (non-blocking)."""
        cmd = [self.package_manager, "run", "dev"]
        scripts = self.get_scripts()
        if "dev" not in scripts and "start" in scripts:
            cmd = [self.package_manager, "run", "start"]
        try:
            self._process = subprocess.Popen(
                cmd, cwd=str(self.source_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, "PORT": str(self.dev_port)},
            )
            return {"started": True, "pid": self._process.pid,
                    "port": self.dev_port}
        except Exception as e:
            return {"started": False, "error": str(e)}

    def stop_dev_server(self) -> bool:
        """Stop development server."""
        if self._process:
            self._process.terminate()
            self._process = None
            return True
        return False


@dataclass
class APIBridge:
    """Python-to-JavaScript API bridge for data exchange."""
    project: FrontendProject
    api_base: str = ""

    def __post_init__(self):
        if not self.api_base:
            self.api_base = f"http://localhost:{self.project.dev_port}"

    def send_data(self, endpoint: str, data: dict) -> dict:
        """Send data to frontend API endpoint."""
        import urllib.request
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def fetch_data(self, endpoint: str) -> dict:
        """Fetch data from frontend API."""
        import urllib.request
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def generate_json_feed(self, data: dict | list,
                           output_path: str | None = None) -> str:
        """Generate JSON data file for frontend consumption."""
        if output_path is None:
            output_path = str(
                self.project.source_dir / "public" / "data" / "feed.json"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return output_path


class TypeScriptIntegration:
    """
    Integration layer for TypeScript/JavaScript frontend projects.

    Manages:
    - ai-hedgefund: React/TypeScript trading dashboard
    - open-bb: TypeScript financial terminal CLI
    - MiroFish: Vue.js social prediction platform
    - Stock-technical-prediction-model: JavaScript stock visualization
    """

    KNOWN_PROJECTS = {
        "ai-hedgefund": {
            "frontend_dir": "app/frontend",
            "framework": "react",
            "dev_port": 3000,
        },
        "open-bb": {
            "frontend_dir": ".",
            "framework": "next",
            "dev_port": 3001,
        },
        "MiroFish": {
            "frontend_dir": "frontend",
            "framework": "vue",
            "dev_port": 3002,
        },
        "Stock-techincal-prediction-model": {
            "frontend_dir": "stock-forecasting-js",
            "framework": "vanilla",
            "dev_port": 3003,
        },
    }

    def __init__(self):
        self.platform_dir = PLATFORM_DIR
        self.projects: dict[str, FrontendProject] = {}
        self._discover_projects()
        logger.info(f"TypeScriptIntegration: found {len(self.projects)} projects")

    def _discover_projects(self):
        """Auto-discover frontend projects."""
        for name, config in self.KNOWN_PROJECTS.items():
            source_dir = self.platform_dir / name / config["frontend_dir"]
            if source_dir.exists():
                self.projects[name] = FrontendProject(
                    name=name,
                    source_dir=source_dir,
                    framework=config["framework"],
                    dev_port=config["dev_port"],
                )
        # Also scan for any package.json in unknown locations
        for pkg_json in self.platform_dir.glob("*/package.json"):
            repo_name = pkg_json.parent.name
            if repo_name not in self.projects:
                with open(pkg_json) as f:
                    data = json.load(f)
                deps = {**data.get("dependencies", {}),
                        **data.get("devDependencies", {})}
                if "react" in deps or "next" in deps:
                    fw = "react"
                elif "vue" in deps:
                    fw = "vue"
                else:
                    fw = "vanilla"
                self.projects[repo_name] = FrontendProject(
                    name=repo_name, source_dir=pkg_json.parent,
                    framework=fw,
                )

    def get_project(self, name: str) -> FrontendProject | None:
        """Get a specific frontend project."""
        return self.projects.get(name)

    def create_api_bridge(self, project_name: str) -> APIBridge | None:
        """Create an API bridge for a project."""
        project = self.projects.get(project_name)
        if project:
            return APIBridge(project=project)
        return None

    def run_node_script(self, script: str, args: list[str] | None = None,
                        cwd: str | None = None) -> dict:
        """Execute a Node.js script."""
        if not shutil.which("node"):
            return {"error": "Node.js not found"}
        cmd = ["node", script] + (args or [])
        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=60,
            )
            return {"stdout": result.stdout, "stderr": result.stderr,
                    "returncode": result.returncode}
        except Exception as e:
            return {"error": str(e)}

    def transpile_to_python(self, ts_file: str) -> str:
        """
        Generate a Python equivalent stub for a TypeScript file.
        Extracts interfaces, types, and function signatures.
        """
        path = Path(ts_file)
        if not path.exists():
            return f"# Source not found: {ts_file}"

        content = path.read_text()
        lines = content.split("\n")
        python_lines = [
            f'"""Auto-generated Python stub from {path.name}"""',
            "from dataclasses import dataclass",
            "from typing import Any, Optional",
            "",
        ]

        in_interface = False
        interface_name = ""
        for line in lines:
            stripped = line.strip()
            # Convert TypeScript interfaces to Python dataclasses
            if stripped.startswith("export interface ") or stripped.startswith("interface "):
                name = stripped.split("interface")[1].split("{")[0].strip()
                python_lines.append(f"\n@dataclass\nclass {name}:")
                interface_name = name
                in_interface = True
            elif in_interface and stripped == "}":
                in_interface = False
            elif in_interface and ":" in stripped:
                field_name = stripped.split(":")[0].strip().rstrip("?")
                ts_type = stripped.split(":")[1].strip().rstrip(";").rstrip(",")
                py_type = self._ts_type_to_python(ts_type)
                if stripped.split(":")[0].strip().endswith("?"):
                    python_lines.append(f"    {field_name}: Optional[{py_type}] = None")
                else:
                    python_lines.append(f"    {field_name}: {py_type} = None")
            # Convert exported functions to Python function stubs
            elif "export function " in stripped or "export const " in stripped:
                if "export function " in stripped:
                    sig = stripped.split("export function ")[1].split("{")[0].strip()
                    fname = sig.split("(")[0]
                    python_lines.append(f"\ndef {fname}(*args, **kwargs) -> Any:")
                    python_lines.append(f'    """Stub for TypeScript function."""')
                    python_lines.append(f"    raise NotImplementedError")

        return "\n".join(python_lines)

    @staticmethod
    def _ts_type_to_python(ts_type: str) -> str:
        """Convert TypeScript type to Python type hint."""
        type_map = {
            "string": "str", "number": "float", "boolean": "bool",
            "any": "Any", "void": "None", "null": "None",
            "undefined": "None", "Date": "str",
            "string[]": "list[str]", "number[]": "list[float]",
        }
        return type_map.get(ts_type.strip(), "Any")

    def get_source_info(self) -> dict:
        """Return metadata about all frontend projects."""
        info = {"projects": {}}
        for name, project in self.projects.items():
            ts_files = list(project.source_dir.rglob("*.ts"))
            tsx_files = list(project.source_dir.rglob("*.tsx"))
            js_files = list(project.source_dir.rglob("*.js"))
            vue_files = list(project.source_dir.rglob("*.vue"))
            info["projects"][name] = {
                "source_dir": str(project.source_dir),
                "framework": project.framework,
                "ts_files": len(ts_files),
                "tsx_files": len(tsx_files),
                "js_files": len(js_files),
                "vue_files": len(vue_files),
                "scripts": project.get_scripts(),
            }
        return info
