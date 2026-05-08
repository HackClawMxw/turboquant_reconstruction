"""
vLLM-Ascend adapter for TurboQuant (placeholder).

This module provides a skeleton adapter for future vLLM-Ascend support.
The adapter interface is defined in turboquant.adapter.base.FrameworkAdapter.

To implement vLLM-Ascend support:
1. Implement all abstract methods from FrameworkAdapter
2. Handle Ascend-specific tensor layouts and attention metadata
3. Replace Triton kernels with Ascend-compatible kernels (e.g., using Ascend C API)
4. Update the adapter factory to select this adapter when running on Ascend hardware

Key differences to handle:
  - Ascend NPU tensor format (NCHW vs NHWC differences)
  - Ascend-specific attention backend (MindSpore / CANN)
  - Different KV cache layout in vLLM-Ascend
  - No Triton support — need custom C++/CANN kernels
"""

import logging
from typing import Any, Optional

import torch

from turboquant.adapter.base import (
    FrameworkAdapter,
    AttentionLayerInfo,
    SequenceInfo,
)

logger = logging.getLogger("turboquant.adapter.ascend")


class AscendAdapter(FrameworkAdapter):
    """
    Placeholder adapter for vLLM-Ascend.

    TODO: Implement all abstract methods for Ascend NPU support.
    """

    def discover_layers(self, model: Any) -> list[AttentionLayerInfo]:
        raise NotImplementedError(
            "AscendAdapter not yet implemented. "
            "Contributions welcome!"
        )

    def get_sequence_info(self, attn_metadata: Any) -> list[SequenceInfo]:
        raise NotImplementedError("AscendAdapter not yet implemented.")

    def get_request_id(self, seq_info: SequenceInfo) -> str:
        raise NotImplementedError("AscendAdapter not yet implemented.")

    def extract_kv_tensors(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("AscendAdapter not yet implemented.")

    def write_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ):
        raise NotImplementedError("AscendAdapter not yet implemented.")

    def install_hooks(self, model: Any, config: dict = None) -> dict:
        raise NotImplementedError(
            "AscendAdapter.install_hooks not yet implemented. "
            "This adapter requires Ascend-specific kernel implementations "
            "and attention metadata handling."
        )

    def free_kv_cache(self, model: Any) -> int:
        raise NotImplementedError("AscendAdapter not yet implemented.")
