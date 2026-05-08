"""
TurboQuant runtime — request-isolated KV cache management.

This package provides the runtime engine for TurboQuant, with full
request isolation. Each inference request gets its own KV state,
preventing data pollution between concurrent requests.

Key classes:
  - RequestKVState: Per-request per-layer KV state (isolated store + ring buffer)
  - RequestSlotManager: Pool manager for request KV states per layer
  - RingBuffer: Fixed-size circular buffer for recent exact KV tokens
  - CompressedKVStore: Pre-allocated compressed KV storage with CUDA-Graph support
  - AttentionEngine: Hybrid attention computation (compressed + ring buffer)
"""

from turboquant.runtime.context import RequestKVState, RequestSlotManager
from turboquant.runtime.ring_buffer import RingBuffer
from turboquant.runtime.kv_store import CompressedKVStore
from turboquant.runtime.attention import AttentionEngine

__all__ = [
    "RequestKVState",
    "RequestSlotManager",
    "RingBuffer",
    "CompressedKVStore",
    "AttentionEngine",
]
