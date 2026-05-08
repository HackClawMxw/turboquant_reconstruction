"""
Request context and slot management for TurboQuant.

Provides per-request KV state isolation. Each inference request
gets its own CompressedKVStore and RingBuffer, preventing any
data sharing or pollution between concurrent requests.

Design principle: ONE RequestKVState per (request, layer) pair.
The RequestSlotManager maps request IDs to their states.
"""

import logging
import threading
from typing import Optional

import torch

from turboquant.runtime.kv_store import CompressedKVStore
from turboquant.runtime.ring_buffer import RingBuffer
from turboquant.runtime.capture import KVCaptureEngine

logger = logging.getLogger("turboquant.runtime")


class RequestKVState:
    """
    Per-request per-layer KV state.

    Each request gets its own isolated state containing:
    - A CompressedKVStore for quantized historical tokens
    - A RingBuffer for recent exact tokens
    - Independent token counter

    This ensures complete data isolation between concurrent requests.
    """

    def __init__(
        self,
        request_id: str,
        layer_idx: int,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int,
        value_bits: int,
        value_group_size: int,
        ring_capacity: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.request_id = request_id
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.device = device
        self.dtype = dtype

        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.ring_capacity = ring_capacity

        # Per-request compressed KV store (isolated from other requests)
        self.store = CompressedKVStore(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            device=device,
            layer_idx=layer_idx,
            max_tokens=max_tokens,
        )

        # Per-request ring buffer (isolated from other requests)
        self.ring_buffer = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

        # Per-request capture engine (orchestrates ring buffer -> store pipeline)
        self.capture_engine = KVCaptureEngine(
            store=self.store,
            ring_buffer=self.ring_buffer,
        )

        self.num_tokens: int = 0
        self.is_prefill_complete: bool = False

    def reset(self):
        """Reset all state for request completion / recycling."""
        self.capture_engine.reset()
        self.num_tokens = 0
        self.is_prefill_complete = False

    def memory_bytes(self) -> int:
        """Total memory used by this request's KV state."""
        return self.store.memory_bytes() + self.ring_buffer.memory_bytes()


class RequestSlotManager:
    """
    Thread-safe pool of RequestKVState instances per layer.

    Manages the lifecycle of per-request KV states:
    - allocate(request_id): creates a new isolated state for a request
    - get(request_id): retrieves the state for a request
    - release(request_id): resets and returns the state to the pool
    - active_count(): number of currently active requests

    Thread safety: all operations are protected by a lock for
    concurrent access from multiple inference threads.
    """

    def __init__(
        self,
        layer_idx: int,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int,
        value_bits: int,
        value_group_size: int,
        ring_capacity: int,
        max_tokens: int,
        max_num_seqs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.ring_capacity = ring_capacity
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.dtype = dtype

        self._slots: dict[str, RequestKVState] = {}
        self._lock = threading.Lock()

    def allocate(self, request_id: str) -> RequestKVState:
        """Create a new isolated KV state for a request."""
        with self._lock:
            if request_id in self._slots:
                logger.warning(
                    f"Request {request_id} already has a slot at layer {self.layer_idx}, "
                    "reusing existing state."
                )
                return self._slots[request_id]

            state = RequestKVState(
                request_id=request_id,
                layer_idx=self.layer_idx,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                value_group_size=self.value_group_size,
                ring_capacity=self.ring_capacity,
                max_tokens=self.max_tokens,
                device=self.device,
                dtype=self.dtype,
            )
            self._slots[request_id] = state
            return state

    def get(self, request_id: str) -> Optional[RequestKVState]:
        """Get the KV state for a request, or None if not found."""
        with self._lock:
            return self._slots.get(request_id)

    def release(self, request_id: str) -> bool:
        """Release a request's KV state. Returns True if found and released."""
        with self._lock:
            state = self._slots.pop(request_id, None)
            if state is not None:
                state.reset()
                return True
            return False

    def active_request_ids(self) -> list[str]:
        """List all active request IDs."""
        with self._lock:
            return list(self._slots.keys())

    def active_count(self) -> int:
        """Number of currently active requests."""
        with self._lock:
            return len(self._slots)

    def total_memory_bytes(self) -> int:
        """Total memory across all active requests."""
        with self._lock:
            return sum(s.memory_bytes() for s in self._slots.values())

    def reset_all(self):
        """Reset all active request states (e.g., on error)."""
        with self._lock:
            for state in self._slots.values():
                state.reset()
            self._slots.clear()
