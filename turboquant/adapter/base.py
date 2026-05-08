"""
Abstract base class for framework adapters.

Defines the interface that all framework adapters must implement.
This allows TurboQuant to work with different inference frameworks
(vLLM, vLLM-Ascend, etc.) through a uniform API.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

import torch


@dataclass
class AttentionLayerInfo:
    """Information about an attention layer extracted from the framework."""
    layer_name: str
    layer_idx: int
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    device: torch.device
    dtype: torch.dtype
    is_mla: bool = False  # Multi-head Latent Attention (skip TQ)


@dataclass
class SequenceInfo:
    """Per-request sequence information from the framework scheduler."""
    request_id: str
    seq_len: int           # total context length
    num_new_tokens: int    # tokens in current step
    is_prefill: bool       # prefill or decode
    slot_mapping: Optional[torch.Tensor] = None  # token -> cache slot mapping
    block_table: Optional[torch.Tensor] = None    # (num_blocks,) paged block table


class FrameworkAdapter(ABC):
    """
    Abstract base class for inference framework adapters.

    Subclasses implement framework-specific hooks for:
    1. Discovering attention layers and their configuration
    2. Extracting per-request sequence metadata
    3. Mapping framework attention metadata to TQ-internal format
    4. Hooking into the attention forward pass
    5. Managing KV cache lifecycle (alloc/free)

    The adapter is responsible for translating between the framework's
    data structures and TurboQuant's internal RequestKVState/AttentionEngine.
    """

    @abstractmethod
    def discover_layers(self, model: Any) -> list[AttentionLayerInfo]:
        """
        Discover all attention layers in the model.

        Args:
            model: The framework's model object

        Returns:
            List of AttentionLayerInfo describing each layer
        """
        ...

    @abstractmethod
    def get_sequence_info(self, attn_metadata: Any) -> list[SequenceInfo]:
        """
        Extract per-request sequence information from framework metadata.

        Called each forward pass to map the batch of requests to
        individual SequenceInfo objects.

        Args:
            attn_metadata: Framework-specific attention metadata

        Returns:
            List of SequenceInfo, one per active request
        """
        ...

    @abstractmethod
    def get_request_id(self, seq_info: SequenceInfo) -> str:
        """
        Get a unique request identifier from sequence info.

        This ID is used to look up per-request KV state in the
        RequestSlotManager.

        Args:
            seq_info: Per-request sequence information

        Returns:
            Unique string identifier for the request
        """
        ...

    @abstractmethod
    def extract_kv_tensors(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and reshape KV tensors from framework format.

        Converts framework-specific KV tensor layout to TQ's expected
        format: (H_kv, seq_len, D).

        Args:
            key: Framework-format key tensor
            value: Framework-format value tensor
            layer_info: Layer configuration

        Returns:
            (key, value) in (H_kv, seq_len, D) format
        """
        ...

    @abstractmethod
    def write_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ):
        """
        Write attention output back to framework's expected format.

        Args:
            output: TQ attention output (QH, D)
            target: Framework's output tensor to write into
            layer_info: Layer configuration
        """
        ...

    @abstractmethod
    def install_hooks(
        self,
        model: Any,
        config: dict,
    ) -> dict:
        """
        Install TurboQuant hooks into the model.

        This is the main entry point for framework integration.
        Monkey-patches attention forward methods to route through
        TurboQuant's request-isolated pipeline.

        Args:
            model: The framework's model runner
            config: TQ configuration (key_bits, value_bits, etc.)

        Returns:
            dict of installed hooks and state for later cleanup
        """
        ...

    @abstractmethod
    def free_kv_cache(self, model: Any) -> int:
        """
        Free paged KV cache for TQ-managed layers.

        After prefill, the paged cache is no longer needed since
        TQ maintains its own compressed store. This frees the
        framework's paged cache to reclaim memory.

        Args:
            model: The framework's model runner

        Returns:
            Number of bytes freed
        """
        ...
