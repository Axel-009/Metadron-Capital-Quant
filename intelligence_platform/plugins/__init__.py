"""
Multi-Language Integration Plugins for Metadron Capital Intelligence Platform.

Provides Python bindings and integration layers for non-Python components:
- Rust (AI-Newton symbolic physics engine)
- Go (Kserve ML model serving)
- C++/CUDA (NVIDIA deep learning optimizations)
- TypeScript/JavaScript (Frontend dashboards, React/Vue apps)
- GSD Plugin (Gradient Signal Dynamics — agent learning)
- Paul Plugin (Pattern Awareness & Unified Learning)
- GSD Workflow Bridge (Get Shit Done — meta-prompting orchestration)
"""

from .rust_plugin import RustIntegration
from .go_plugin import GoIntegration
from .cuda_plugin import CUDAIntegration
from .typescript_plugin import TypeScriptIntegration
from .unified_bridge import UnifiedLanguageBridge

try:
    from .gsd_paul_plugin import GSDPlugin, PaulPlugin, AgentLearningWrapper
except ImportError:
    GSDPlugin = None
    PaulPlugin = None
    AgentLearningWrapper = None

try:
    from .gsd_workflow_bridge import GSDWorkflowBridge
except ImportError:
    GSDWorkflowBridge = None

__all__ = [
    "RustIntegration",
    "GoIntegration",
    "CUDAIntegration",
    "TypeScriptIntegration",
    "UnifiedLanguageBridge",
    "GSDPlugin",
    "PaulPlugin",
    "AgentLearningWrapper",
    "GSDWorkflowBridge",
]
