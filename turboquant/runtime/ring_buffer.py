"""
Ring buffer for recent exact KV tokens.

Each request gets its own ring buffer instance (request isolation).
The ring buffer holds the most recent tokens in full precision,
while older tokens are compressed in the CompressedKVStore.
"""

import torch
from typing import Optional, Tuple


class RingBuffer:
    """
    Fixed-size ring buffer for recent exact KV tokens.

    Pre-allocates key and value tensors. New tokens are written
    sequentially with wraparound. Supports CUDA-Graph-compatible
    writes via device-side counters.

    This is per-request — no sharing between requests.
    """

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocated KV tensors: (capacity, H_kv, D)
        self.keys = torch.zeros(
            capacity, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        )
        self.values = torch.zeros(
            capacity, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        )

        # Host-side write position counter
        self._write_pos: int = 0
        self._count: int = 0

        # Device-side counters for CUDA-Graph compatibility
        self._count_tensor = torch.zeros(1, dtype=torch.int64, device=device)
        self._cap_tensor = torch.tensor([capacity], dtype=torch.int64, device=device)
        self._arange_tensor = torch.arange(capacity, dtype=torch.int64, device=device)

    @property
    def count(self) -> int:
        return self._count

    def is_full(self) -> bool:
        return self._count >= self.capacity

    def write(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Append tokens to the ring buffer.

        Args:
            key: (num_tokens, H_kv, D) or (1, H_kv, D)
            value: same shape as key

        Returns:
            (overflow_key, overflow_value) tensors of shape (n_overflow, H_kv, D)
            if buffer was full and oldest tokens were evicted. None if no overflow.
        """
        num_tokens = key.shape[0]
        overflow_keys = []
        overflow_values = []

        for i in range(num_tokens):
            pos = self._write_pos % self.capacity

            # If buffer is full, the oldest token overflows
            if self._count >= self.capacity:
                overflow_keys.append(self.keys[pos].clone())
                overflow_values.append(self.values[pos].clone())

            self.keys[pos].copy_(key[i])
            self.values[pos].copy_(value[i])

            self._write_pos += 1
            if self._count < self.capacity:
                self._count += 1

        # Update device-side counter
        self._count_tensor.fill_(self._count)

        if overflow_keys:
            return torch.stack(overflow_keys), torch.stack(overflow_values)
        return None, None

    def write_single_graph(self, key: torch.Tensor, value: torch.Tensor):
        """
        CUDA-Graph-compatible single-token write.

        Uses index_copy_ and in-place arithmetic only — no Python
        control flow that would break CUDA Graph capture.

        Args:
            key: (H_kv, D) single token
            value: (H_kv, D) single token
        """
        pos = self._write_pos % self.capacity

        self.keys[pos].copy_(key)
        self.values[pos].copy_(value)

        self._write_pos += 1
        if self._count < self.capacity:
            self._count += 1
        self._count_tensor.fill_(self._count)

    def read_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read all valid tokens from the ring buffer.

        Returns:
            keys: (count, H_kv, D)
            values: (count, H_kv, D)
        """
        if self._count == 0:
            empty_k = torch.zeros(0, self.num_kv_heads, self.head_dim,
                                  device=self.device, dtype=self.dtype)
            return empty_k, empty_k.clone()

        if self._count <= self.capacity:
            return self.keys[:self._count].clone(), self.values[:self._count].clone()
        else:
            # Wrap-around: need to reorder
            start = self._write_pos % self.capacity
            indices = [(start + i) % self.capacity for i in range(self.capacity)]
            idx_tensor = torch.tensor(indices, device=self.device)
            return self.keys[idx_tensor].clone(), self.values[idx_tensor].clone()

    def read_for_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read ring buffer contents for CUDA-Graph-compatible kernel.

        Returns the full-capacity tensors + device count tensor.
        The kernel handles masking based on the count.
        """
        return self.keys, self.values, self._count_tensor

    def drain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read and clear all valid tokens."""
        keys, values = self.read_all()
        self.reset()
        return keys, values

    def reset(self):
        """Clear the ring buffer."""
        self._write_pos = 0
        self._count = 0
        self._count_tensor.zero_()
        self.keys.zero_()
        self.values.zero_()

    def memory_bytes(self) -> int:
        """Memory used by this ring buffer in bytes."""
        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        kv_size = 2 * self.capacity * self.num_kv_heads * self.head_dim * element_size
        counter_size = 3 * 8  # count_tensor + cap_tensor + arange_tensor (int64)
        return kv_size + counter_size
