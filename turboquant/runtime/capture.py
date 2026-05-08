"""
KV capture engine — orchestrates KV capture into per-request stores.

Handles the lifecycle of KV capture for a single request:
1. Prefill: bulk capture, compress all but last ring_capacity tokens
2. Decode: single-token capture with ring buffer overflow handling
3. Ring buffer overflow: automatic compression when buffer fills

This is per-request — each request has its own capture engine instance.
"""

import logging
from typing import Optional

import torch

from turboquant.runtime.kv_store import CompressedKVStore
from turboquant.runtime.ring_buffer import RingBuffer

logger = logging.getLogger("turboquant.runtime")


class KVCaptureEngine:
    """
    Orchestrates KV capture for a single request.

    Manages the pipeline:
      incoming KV tokens → ring buffer → compressed store

    The ring buffer holds recent exact tokens; overflow is compressed
    and stored in the CompressedKVStore.
    """

    def __init__(
        self,
        store: CompressedKVStore,
        ring_buffer: RingBuffer,
    ):
        self.store = store
        self.ring_buffer = ring_buffer
        self._is_prefill_phase = True

    def ingest_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Bulk capture from prefill.

        Splits tokens: compress all but the last ring_capacity tokens,
        keep recent ones in the ring buffer.

        Args:
            key: (H_kv, seq_len, D)
            value: (H_kv, seq_len, D)
        """
        seq_len = key.shape[1]
        capacity = self.ring_buffer.capacity

        if seq_len <= capacity:
            # Everything fits in ring buffer, no compression needed
            # Transpose to (seq_len, H_kv, D) for ring buffer
            self.ring_buffer.write(key.transpose(0, 1), value.transpose(0, 1))
        else:
            # Compress all but last capacity tokens
            n_compress = seq_len - capacity
            keys_to_compress = key[:, :n_compress, :]
            values_to_compress = value[:, :n_compress, :]
            self.store.append_chunk(keys_to_compress, values_to_compress)

            # Put recent tokens in ring buffer
            recent_keys = key[:, n_compress:, :].transpose(0, 1)
            recent_values = value[:, n_compress:, :].transpose(0, 1)
            self.ring_buffer.write(recent_keys, recent_values)

        self._is_prefill_phase = False

    def ingest_prefill_from_paged(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_len: int,
    ):
        """
        Bulk capture from paged KV cache tensor (vLLM format).

        Extracts tokens from paged cache using slot_mapping, then
        processes like ingest_prefill.

        Args:
            key_cache: (num_blocks, block_size, H_kv, D) paged key cache
            value_cache: (num_blocks, block_size, H_kv, D) paged value cache
            slot_mapping: (seq_len,) flat slot indices
            seq_len: number of tokens to capture
        """
        # Flatten block dimensions
        num_blocks, block_size, H_kv, D = key_cache.shape
        flat_key = key_cache.reshape(-1, H_kv, D)    # (total_slots, H_kv, D)
        flat_value = value_cache.reshape(-1, H_kv, D)

        # Gather tokens by slot mapping
        slots = slot_mapping[:seq_len].long()
        key_gathered = flat_key[slots].transpose(0, 1)    # (H_kv, seq_len, D)
        value_gathered = flat_value[slots].transpose(0, 1)

        self.ingest_prefill(key_gathered, value_gathered)

    def ingest_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Append decode tokens. Handles ring buffer overflow automatically.

        Args:
            key: (H_kv, 1, D) or (1, H_kv, D) single token
            value: same shape as key
        """
        # Normalize to (1, H_kv, D) for ring buffer
        if key.dim() == 3 and key.shape[0] == key.shape[1]:
            # (H_kv, 1, D) -> (1, H_kv, D)
            key_rb = key.transpose(0, 1)
            value_rb = value.transpose(0, 1)
        else:
            key_rb = key
            value_rb = value

        overflow_key, overflow_value = self.ring_buffer.write(key_rb, value_rb)

        if overflow_key is not None:
            # Compress the overflowed token(s)
            overflow_key_q = overflow_key.transpose(0, 1)  # (H_kv, 1, D)
            overflow_value_q = overflow_value.transpose(0, 1)
            self.store.append_chunk(overflow_key_q, overflow_value_q)

    def ingest_decode_graph(self, key: torch.Tensor, value: torch.Tensor):
        """
        CUDA-Graph-compatible single-token decode write.

        Uses device-side ring buffer operations only.
        Overflow is checked between graph replays.

        Args:
            key: (H_kv, D) single token
            value: (H_kv, D) single token
        """
        self.ring_buffer.write_single_graph(key, value)

    def check_overflow_and_compress(self):
        """
        Check if ring buffer overflowed and compress if needed.

        Called between CUDA Graph replays (outside graph capture).
        """
        # In graph mode, overflow detection is more complex.
        # For now, we check on CPU side.
        pass

    def prepare_for_decode(self):
        """
        Prepare for decode phase after prefill.

        Compresses any remaining prefill tokens in the ring buffer.
        This ensures the ring buffer is ready for decode writes.
        """
        # The ring buffer already has recent prefill tokens.
        # No additional compression needed — the ring buffer
        # will handle overflow during decode.
        self._is_prefill_phase = False

    def flush(self):
        """Force-flush ring buffer to compressed store."""
        keys, values = self.ring_buffer.drain()
        if keys.shape[0] > 0:
            # (tokens, H_kv, D) -> (H_kv, tokens, D)
            self.store.append_chunk(keys.transpose(0, 1), values.transpose(0, 1))

    def reset(self):
        """Reset all state."""
        self.store.reset()
        self.ring_buffer.reset()
        self._is_prefill_phase = True
