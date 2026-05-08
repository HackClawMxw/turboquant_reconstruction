"""
Memory estimation utilities for TurboQuant.

Provides functions to estimate memory usage of compressed KV caches
and plan memory allocation for different configurations.
"""

import math
from dataclasses import dataclass


@dataclass
class MemoryEstimate:
    """Memory estimate for a TQ configuration."""
    per_request_per_layer_bytes: int
    per_layer_overhead_bytes: int
    total_per_request_bytes: int  # across all layers
    max_concurrent_requests: int
    total_tq_memory_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_tq_memory_bytes / 1e9

    def __str__(self) -> str:
        return (
            f"MemoryEstimate: "
            f"{self.total_gb:.2f} GB total "
            f"({self.max_concurrent_requests} reqs x {self.total_per_request_bytes / 1e6:.1f} MB/req) "
            f"layer overhead: {self.per_layer_overhead_bytes / 1e6:.1f} MB"
        )


def estimate_memory(
    num_layers: int,
    head_dim: int,
    num_kv_heads: int,
    num_query_heads: int,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    max_tokens_per_request: int = 32768,
    max_num_seqs: int = 256,
    gpu_memory_bytes: int = 80 * 1024**3,  # 80 GB default
    model_memory_bytes: int = 40 * 1024**3,  # 40 GB default
) -> MemoryEstimate:
    """
    Estimate memory usage for TurboQuant configuration.

    Args:
        num_layers: Number of attention layers
        head_dim: Dimension per head
        num_kv_heads: Number of KV heads
        num_query_heads: Number of query heads
        key_bits: Bits per key quantization
        value_bits: Bits per value quantization
        value_group_size: Value quantization group size
        ring_capacity: Recent token buffer size
        max_tokens_per_request: Max compressed tokens per request
        max_num_seqs: Max concurrent sequences
        gpu_memory_bytes: Total GPU memory
        model_memory_bytes: Memory used by model weights + activations

    Returns:
        MemoryEstimate with detailed breakdown
    """
    N = max_tokens_per_request
    H = num_kv_heads
    D = head_dim

    # Quantization geometry
    mse_bits = key_bits - 1
    if mse_bits == 1:
        packed_d_mse = D // 8
    elif mse_bits == 2:
        packed_d_mse = D // 4
    else:
        packed_d_mse = D // 2

    packed_d_signs = D // 8
    n_groups = D // value_group_size

    # Per-request per-layer: compressed store
    store_bytes = (
        H * N * packed_d_mse +      # MSE indices (uint8)
        H * N * packed_d_signs +     # QJL signs (uint8)
        H * N * 2 +                  # Key norms (float16)
        H * N * 2 +                  # Residual norms (float16)
        H * N * D +                  # Value data (uint8)
        H * N * n_groups * 2 +       # Value scales (float16)
        H * N * n_groups * 2 +       # Value zeros (float16)
        4                            # Device N counter (int32)
    )

    # Per-request per-layer: ring buffer
    ring_bytes = (
        2 * ring_capacity * H * D * 2 +  # Keys + Values (float16)
        3 * 8                             # Device counters (int64)
    )

    per_request_per_layer = store_bytes + ring_bytes

    # Per-layer overhead (quantizer buffers, shared across requests)
    overhead = (
        D * D * 4 +    # Rotation matrix (float32)
        D * D * 4 +    # QJL matrix (float32)
        2**mse_bits * 4 +  # Centroids (float32)
        (2**mse_bits + 1) * 4 +  # Boundaries (float32)
        # Attention engine buffers
        num_query_heads * D * 4 * 7  # 7 pre-allocated buffers (float32)
    )

    # Total per request across all layers
    per_request_total = per_request_per_layer * num_layers

    # Available memory for TQ
    available = gpu_memory_bytes - model_memory_bytes
    max_concurrent = min(
        max_num_seqs,
        max(1, (available - overhead * num_layers) // per_request_total),
    )

    total = (
        per_request_per_layer * num_layers * max_concurrent +
        overhead * num_layers
    )

    return MemoryEstimate(
        per_request_per_layer_bytes=per_request_per_layer,
        per_layer_overhead_bytes=overhead,
        total_per_request_bytes=per_request_total,
        max_concurrent_requests=max_concurrent,
        total_tq_memory_bytes=total,
    )


def estimate_compression_ratio(
    head_dim: int,
    key_bits: int = 3,
    value_bits: int = 2,
) -> dict:
    """
    Estimate compression ratio vs fp16 KV cache.

    Returns dict with bits-per-element and compression ratio.
    """
    # Original: 2 bytes per element (fp16) for both key and value
    original_bits_per_element = 16  # 16 bits for each of K and V

    # TQ key: key_bits per coordinate (MSE bits-1 + 1 QJL bit)
    # Plus norms: ~2 bits amortized (float16 / D)
    key_bits_per_element = key_bits + 32.0 / head_dim  # norms amortized

    # TQ value: value_bits + scale/zero overhead per group
    value_bits_per_element = value_bits + 2 * 16.0 / 32  # scale + zero per group

    total_bits = key_bits_per_element + value_bits_per_element
    ratio = original_bits_per_element / total_bits

    return {
        "key_bits_per_element": key_bits_per_element,
        "value_bits_per_element": value_bits_per_element,
        "total_bits_per_element": total_bits,
        "compression_ratio": ratio,
        "savings_pct": (1 - 1 / ratio) * 100,
    }
