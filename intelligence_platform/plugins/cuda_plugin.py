"""
CUDA/C++ Integration Plugin - NVIDIA Deep Learning Optimizations.

Bridges NVIDIA's C++/CUDA implementations (FasterTransformer, FastSpeech,
Drug Discovery SE3Transformer) into Python via:
1. ctypes/cffi FFI bindings
2. Subprocess execution of compiled binaries
3. PyTorch/TensorFlow Python equivalents
"""

import ctypes
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CUDA_SOURCE_DIR = Path(__file__).parent.parent / "nividia-repo"


@dataclass
class CUDAKernelConfig:
    """Configuration for a CUDA kernel execution."""
    grid_dim: tuple[int, ...] = (1, 1, 1)
    block_dim: tuple[int, ...] = (256, 1, 1)
    shared_mem_bytes: int = 0
    stream: Optional[int] = None


@dataclass
class FasterTransformerConfig:
    """Configuration for FasterTransformer inference."""
    head_num: int = 12
    size_per_head: int = 64
    num_layers: int = 12
    vocab_size: int = 30522
    max_seq_len: int = 512
    tensor_para_size: int = 1
    pipeline_para_size: int = 1
    data_type: str = "fp16"  # fp16, fp32, bf16, int8
    batch_size: int = 1

    @property
    def hidden_size(self) -> int:
        return self.head_num * self.size_per_head

    def to_args(self) -> list[str]:
        """Convert to command-line arguments for FT binary."""
        return [
            f"--head_num={self.head_num}",
            f"--size_per_head={self.size_per_head}",
            f"--num_layers={self.num_layers}",
            f"--vocab_size={self.vocab_size}",
            f"--max_seq_len={self.max_seq_len}",
            f"--tensor_para_size={self.tensor_para_size}",
            f"--data_type={self.data_type}",
            f"--batch_size={self.batch_size}",
        ]


@dataclass
class FastSpeechConfig:
    """Configuration for FastSpeech TTS inference."""
    max_seq_len: int = 2048
    encoder_hidden: int = 256
    encoder_head: int = 2
    encoder_layers: int = 4
    decoder_hidden: int = 256
    decoder_head: int = 2
    decoder_layers: int = 4
    fft_conv_kernel: int = 3
    duration_predictor_filters: int = 256
    num_mels: int = 80
    use_fp16: bool = True


class PythonFasterTransformer:
    """
    Pure Python/PyTorch fallback for FasterTransformer attention.
    Uses standard PyTorch operations when CUDA kernels aren't available.
    """

    def __init__(self, config: FasterTransformerConfig):
        self.config = config
        self._torch = None
        self._model = None

    def _ensure_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                raise RuntimeError("PyTorch required for fallback mode")

    def multi_head_attention(self, query, key, value,
                             mask=None) -> Any:
        """Multi-head attention (Python fallback for CUDA fused kernel)."""
        self._ensure_torch()
        torch = self._torch
        B, L, D = query.shape
        H = self.config.head_num
        d_k = self.config.size_per_head

        q = query.view(B, L, H, d_k).transpose(1, 2)
        k = key.view(B, L, H, d_k).transpose(1, 2)
        v = value.view(B, L, H, d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(B, L, D)

    def layer_norm(self, x, weight, bias, eps=1e-5) -> Any:
        """Layer normalization (Python fallback for CUDA fused kernel)."""
        self._ensure_torch()
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return weight * (x - mean) / (var + eps).sqrt() + bias


class CUDAIntegration:
    """
    Integration layer for NVIDIA CUDA/C++ deep learning optimizations.

    Supports:
    1. 'compiled' - Load shared libraries via ctypes
    2. 'subprocess' - Execute compiled CUDA binaries
    3. 'pytorch' - PyTorch fallback implementations
    4. 'config' - Configuration generation only
    """

    def __init__(self, mode: str = "auto"):
        self.source_dir = CUDA_SOURCE_DIR

        if mode == "auto":
            self.mode = self._detect_mode()
        else:
            self.mode = mode
        logger.info(f"CUDAIntegration initialized in '{self.mode}' mode")

    def _detect_mode(self) -> str:
        """Auto-detect best available execution mode."""
        # Check for pre-compiled shared libs
        so_files = list(self.source_dir.rglob("*.so")) if self.source_dir.exists() else []
        if so_files:
            return "compiled"
        if shutil.which("nvcc"):
            return "subprocess"
        try:
            import torch
            if torch.cuda.is_available():
                return "pytorch"
        except ImportError:
            pass
        return "config"

    def compile(self, project: str = "FasterTransformer") -> bool:
        """Compile CUDA project using CMake."""
        project_dir = self.source_dir / project
        if not project_dir.exists():
            # Search subdirectories
            matches = list(self.source_dir.rglob(project))
            if matches:
                project_dir = matches[0]
            else:
                logger.error(f"Project {project} not found")
                return False

        build_dir = project_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_file = project_dir / "CMakeLists.txt"
        if not cmake_file.exists():
            logger.error(f"No CMakeLists.txt in {project_dir}")
            return False

        try:
            subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=str(build_dir), check=True,
                capture_output=True, text=True,
            )
            subprocess.run(
                ["cmake", "--build", ".", "--parallel"],
                cwd=str(build_dir), check=True,
                capture_output=True, text=True,
            )
            self.mode = "compiled"
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"CUDA compilation failed: {e}")
            return False

    def create_faster_transformer(self, **kwargs) -> FasterTransformerConfig:
        """Create FasterTransformer configuration."""
        return FasterTransformerConfig(**kwargs)

    def create_fastspeech(self, **kwargs) -> FastSpeechConfig:
        """Create FastSpeech TTS configuration."""
        return FastSpeechConfig(**kwargs)

    def get_python_transformer(self,
                               config: FasterTransformerConfig) -> PythonFasterTransformer:
        """Get Python/PyTorch fallback for FasterTransformer."""
        return PythonFasterTransformer(config)

    def load_shared_library(self, lib_path: str) -> ctypes.CDLL:
        """Load a compiled CUDA shared library."""
        path = Path(lib_path)
        if not path.exists():
            # Search in source directory
            matches = list(self.source_dir.rglob(path.name))
            if matches:
                path = matches[0]
            else:
                raise FileNotFoundError(f"Library not found: {lib_path}")
        return ctypes.CDLL(str(path))

    def get_cuda_info(self) -> dict:
        """Get CUDA device and toolkit information."""
        info = {"nvcc_available": bool(shutil.which("nvcc")), "devices": []}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        info["devices"].append({
                            "name": parts[0], "memory": parts[1],
                            "compute_capability": parts[2],
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return info

    def get_source_info(self) -> dict:
        """Return metadata about C++/CUDA source code."""
        cpp_files = list(self.source_dir.rglob("*.cpp")) if self.source_dir.exists() else []
        cu_files = list(self.source_dir.rglob("*.cu")) if self.source_dir.exists() else []
        h_files = list(self.source_dir.rglob("*.h")) if self.source_dir.exists() else []
        cmake_files = list(self.source_dir.rglob("CMakeLists.txt")) if self.source_dir.exists() else []
        return {
            "source_dir": str(self.source_dir),
            "cpp_files": len(cpp_files),
            "cuda_files": len(cu_files),
            "header_files": len(h_files),
            "cmake_files": len(cmake_files),
            "mode": self.mode,
            "projects": sorted(set(
                str(f.parent.relative_to(self.source_dir)).split("/")[0]
                for f in cpp_files + cu_files
                if f.parent != self.source_dir
            ))[:20],
        }
