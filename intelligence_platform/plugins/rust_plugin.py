"""
Rust Integration Plugin - AI-Newton Symbolic Physics Engine.

Bridges the Rust-based symbolic algebra and physics simulation engine
from AI-Newton into the Python intelligence platform via:
1. PyO3/maturin FFI bindings (compiled mode)
2. Subprocess execution (interpreted mode)
3. Pure Python fallback reimplementation of core algorithms
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

RUST_SOURCE_DIR = Path(__file__).parent.parent / "AI-Newton" / "src"
CARGO_TOML = Path(__file__).parent.parent / "AI-Newton" / "Cargo.toml"


@dataclass
class SymbolicExpression:
    """Python representation of AI-Newton's Rust symbolic expression."""
    expr_type: str  # "variable", "constant", "binary_op", "function"
    value: Any = None
    operator: Optional[str] = None
    children: list = field(default_factory=list)

    def evaluate(self, bindings: dict[str, float] | None = None) -> float:
        """Evaluate expression with variable bindings."""
        bindings = bindings or {}
        if self.expr_type == "constant":
            return float(self.value)
        elif self.expr_type == "variable":
            if self.value not in bindings:
                raise ValueError(f"Unbound variable: {self.value}")
            return bindings[self.value]
        elif self.expr_type == "binary_op":
            left = self.children[0].evaluate(bindings)
            right = self.children[1].evaluate(bindings)
            ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
                   "*": lambda a, b: a * b, "/": lambda a, b: a / b,
                   "**": lambda a, b: a ** b}
            return ops[self.operator](left, right)
        elif self.expr_type == "function":
            import math
            arg = self.children[0].evaluate(bindings)
            funcs = {"sin": math.sin, "cos": math.cos, "exp": math.exp,
                     "log": math.log, "sqrt": math.sqrt, "abs": abs}
            return funcs[self.value](arg)
        raise ValueError(f"Unknown expression type: {self.expr_type}")

    def __repr__(self) -> str:
        if self.expr_type == "constant":
            return str(self.value)
        elif self.expr_type == "variable":
            return self.value
        elif self.expr_type == "binary_op":
            return f"({self.children[0]} {self.operator} {self.children[1]})"
        elif self.expr_type == "function":
            return f"{self.value}({self.children[0]})"
        return f"Expr({self.expr_type})"


@dataclass
class PhysicsSimulation:
    """Python fallback for AI-Newton's Rust physics simulation."""
    objects: list[dict] = field(default_factory=list)
    time_step: float = 0.01
    gravity: float = 9.81

    def add_particle(self, mass: float, position: tuple[float, ...],
                     velocity: tuple[float, ...] = (0.0, 0.0)) -> int:
        obj = {"type": "particle", "mass": mass,
               "position": list(position), "velocity": list(velocity)}
        self.objects.append(obj)
        return len(self.objects) - 1

    def add_spring(self, k: float, rest_length: float,
                   obj1_idx: int, obj2_idx: int) -> None:
        self.objects.append({"type": "spring", "k": k,
                             "rest_length": rest_length,
                             "obj1": obj1_idx, "obj2": obj2_idx})

    def step(self) -> list[dict]:
        """Advance simulation by one time step (Euler integration)."""
        import math
        particles = [o for o in self.objects if o["type"] == "particle"]
        springs = [o for o in self.objects if o["type"] == "spring"]

        # Apply gravity
        for p in particles:
            p["velocity"][1] -= self.gravity * self.time_step

        # Apply spring forces
        for s in springs:
            p1, p2 = particles[s["obj1"]], particles[s["obj2"]]
            dx = [p2["position"][i] - p1["position"][i] for i in range(2)]
            dist = math.sqrt(sum(d * d for d in dx))
            if dist > 0:
                force = s["k"] * (dist - s["rest_length"])
                for i in range(2):
                    f = force * dx[i] / dist
                    p1["velocity"][i] += f / p1["mass"] * self.time_step
                    p2["velocity"][i] -= f / p2["mass"] * self.time_step

        # Update positions
        for p in particles:
            for i in range(len(p["position"])):
                p["position"][i] += p["velocity"][i] * self.time_step

        return particles


class RustIntegration:
    """
    Integration layer for AI-Newton's Rust symbolic physics engine.

    Supports three execution modes:
    1. 'compiled' - Uses PyO3 bindings (requires cargo build)
    2. 'subprocess' - Calls compiled Rust binary
    3. 'fallback' - Pure Python reimplementation
    """

    def __init__(self, mode: str = "auto"):
        self.source_dir = RUST_SOURCE_DIR
        self.cargo_toml = CARGO_TOML
        self._native_module = None

        if mode == "auto":
            self.mode = self._detect_mode()
        else:
            self.mode = mode
        logger.info(f"RustIntegration initialized in '{self.mode}' mode")

    def _detect_mode(self) -> str:
        """Auto-detect best available execution mode."""
        try:
            import ainewton  # noqa: F401
            return "compiled"
        except ImportError:
            pass
        if shutil.which("cargo") and self.cargo_toml.exists():
            return "subprocess"
        return "fallback"

    def compile(self) -> bool:
        """Attempt to compile Rust source via maturin/cargo."""
        if not self.cargo_toml.exists():
            logger.error(f"Cargo.toml not found at {self.cargo_toml}")
            return False
        try:
            if shutil.which("maturin"):
                subprocess.run(
                    ["maturin", "develop", "--release"],
                    cwd=self.cargo_toml.parent, check=True,
                    capture_output=True, text=True,
                )
                self.mode = "compiled"
                return True
            elif shutil.which("cargo"):
                subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=self.cargo_toml.parent, check=True,
                    capture_output=True, text=True,
                )
                self.mode = "subprocess"
                return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Rust compilation failed: {e.stderr}")
        return False

    def create_expression(self, expr_str: str) -> SymbolicExpression:
        """Parse a symbolic expression string."""
        # Simple recursive descent parser for math expressions
        return self._parse_expr(expr_str.strip())

    def _parse_expr(self, s: str) -> SymbolicExpression:
        """Parse expression string into SymbolicExpression tree."""
        s = s.strip()
        # Try to parse as number
        try:
            return SymbolicExpression("constant", value=float(s))
        except ValueError:
            pass
        # Variable (single identifier)
        if s.isidentifier():
            return SymbolicExpression("variable", value=s)
        # Parenthesized expression
        if s.startswith("(") and s.endswith(")"):
            return self._parse_expr(s[1:-1])
        # Binary operators (lowest precedence first)
        for ops in [["+", "-"], ["*", "/"], ["**"]]:
            depth = 0
            for i in range(len(s) - 1, -1, -1):
                if s[i] == ")": depth += 1
                elif s[i] == "(": depth -= 1
                elif depth == 0 and s[i] in ops:
                    if i > 0:
                        left = self._parse_expr(s[:i])
                        right = self._parse_expr(s[i + 1:])
                        return SymbolicExpression(
                            "binary_op", operator=s[i],
                            children=[left, right])
        # Function call
        if "(" in s:
            fname = s[:s.index("(")]
            arg = s[s.index("(") + 1:s.rindex(")")]
            return SymbolicExpression(
                "function", value=fname,
                children=[self._parse_expr(arg)])
        raise ValueError(f"Cannot parse expression: {s}")

    def create_simulation(self, **kwargs) -> PhysicsSimulation:
        """Create a new physics simulation."""
        return PhysicsSimulation(**kwargs)

    def run_rust_command(self, args: list[str],
                        input_data: str | None = None) -> dict:
        """Execute a Rust binary command and return parsed output."""
        if self.mode == "fallback":
            return {"error": "No Rust toolchain available", "mode": "fallback"}
        binary = self.cargo_toml.parent / "target" / "release" / "ainewton"
        if not binary.exists():
            binary = self.cargo_toml.parent / "target" / "debug" / "ainewton"
        if not binary.exists():
            return {"error": f"Binary not found. Run compile() first."}
        try:
            result = subprocess.run(
                [str(binary)] + args,
                input=input_data, capture_output=True, text=True, timeout=30,
            )
            return {"stdout": result.stdout, "stderr": result.stderr,
                    "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "Rust command timed out"}

    def get_source_info(self) -> dict:
        """Return metadata about the Rust source code."""
        rs_files = list(self.source_dir.rglob("*.rs")) if self.source_dir.exists() else []
        lalrpop_files = list(self.source_dir.rglob("*.lalrpop")) if self.source_dir.exists() else []
        return {
            "source_dir": str(self.source_dir),
            "cargo_toml": str(self.cargo_toml),
            "rust_files": len(rs_files),
            "lalrpop_files": len(lalrpop_files),
            "mode": self.mode,
            "modules": sorted(set(
                f.parent.name for f in rs_files if f.parent != self.source_dir
            )),
        }
