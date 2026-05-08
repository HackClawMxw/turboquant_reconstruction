"""
TurboQuant framework adapter layer.

Provides abstract interfaces for integrating TurboQuant with different
inference frameworks. Currently supports:
  - vLLM (NVIDIA GPU)
  - vLLM-Ascend (placeholder for future support)

The adapter pattern allows the core TurboQuant runtime to be
framework-agnostic, while framework-specific hooks are implemented
in concrete adapter classes.
"""

from turboquant.adapter.base import FrameworkAdapter
from turboquant.adapter.vllm_adapter import VllmAdapter

__all__ = [
    "FrameworkAdapter",
    "VllmAdapter",
]
