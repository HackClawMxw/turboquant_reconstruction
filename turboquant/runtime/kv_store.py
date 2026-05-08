"""
Compressed KV store — per-request compressed KV cache.

Each request gets its own CompressedKVStore instance (request isolation).
Stores quantized keys (TurboQuantProd) and quantized values (group quantization)
in pre-allocated fixed-address buffers for CUDA-Graph compatibility.

Key design change from original:
  - The store is per-request, not per-layer-shared
  - Pre-allocated buffers with device-side token counter
  - No page cache after prefill (trade compute for storage)
"""

import math
import logging
from typing import Optional, NamedTuple

import torch
import torch.nn.functional as F

from turboquant.core.quantizer import TurboQuantProd, ProdQuantized

logger = logging.getLogger("turboquant.runtime")


class ValueQuantized(NamedTuple):
    """Quantized value cache."""
    data: torch.Tensor       # (H_kv, N, D) uint8 quantized values (unpacked)
    scales: torch.Tensor     # (H_kv, N, n_groups) scale per group
    zeros: torch.Tensor      # (H_kv, N, n_groups) zero point per group
    bits: int = 2


def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """
    Asymmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)

    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def dequantize_values(
    vq: ValueQuantized,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize value vectors."""
    data = vq.data.float()
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)


def unpack_values(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack bit-packed value data to per-element uint8."""
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 2)
    return packed


class CompressedKVStore:
    """
    Per-request compressed KV store with pre-allocated buffers.

    Stores quantized keys and values in contiguous tensors with
    a device-side counter for CUDA-Graph compatibility.

    Each request has its own store — no shared mutable state.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int,
        value_bits: int,
        value_group_size: int,
        device: torch.device,
        layer_idx: int = 0,
        max_tokens: int = 32768,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.device = device
        self.layer_idx = layer_idx
        self.max_tokens = max_tokens

        # Key quantizer (shared algorithm, but stateless — per-call isolation)
        self.key_quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=device,
            seed=42 + layer_idx * 7,
        )

        # Compute quantization geometry
        mse_bits = key_bits - 1
        if mse_bits == 1:
            self._packed_d_mse = head_dim // 8
        elif mse_bits == 2:
            self._packed_d_mse = head_dim // 4
        else:
            self._packed_d_mse = head_dim // 2  # 3-4 bit -> 2 per byte

        self._packed_d_signs = head_dim // 8  # 1 bit per coord, 8 per byte
        self._n_value_groups = head_dim // value_group_size

        # Pre-allocate buffers (fixed-address for CUDA Graph)
        self._preallocate(max_tokens)

    def _preallocate(self, max_tokens: int):
        """Pre-allocate all compressed KV buffers."""
        H = self.num_kv_heads
        N = max_tokens

        # Key MSE indices: (H, N, packed_d_mse) uint8
        self.mse_indices_buf = torch.zeros(
            H, N, self._packed_d_mse, device=self.device, dtype=torch.uint8)

        # Key QJL signs: (H, N, packed_d_signs) uint8
        self.qjl_signs_buf = torch.zeros(
            H, N, self._packed_d_signs, device=self.device, dtype=torch.uint8)

        # Key norms: (H, N) float16
        self.key_norms_buf = torch.zeros(
            H, N, device=self.device, dtype=torch.float16)

        # Key residual norms: (H, N) float16
        self.res_norms_buf = torch.zeros(
            H, N, device=self.device, dtype=torch.float16)

        # Value data: (H, N, D) uint8 (unpacked per-element)
        self.value_data_buf = torch.zeros(
            H, N, self.head_dim, device=self.device, dtype=torch.uint8)

        # Value scales: (H, N, n_groups) float16
        self.value_scales_buf = torch.zeros(
            H, N, self._n_value_groups, device=self.device, dtype=torch.float16)

        # Value zeros: (H, N, n_groups) float16
        self.value_zeros_buf = torch.zeros(
            H, N, self._n_value_groups, device=self.device, dtype=torch.float16)

        # Device-side token counter
        self._n_tensor = torch.zeros(1, dtype=torch.int32, device=self.device)

        self._num_stored = 0

    @property
    def n_stored(self) -> int:
        return self._num_stored

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """
        Quantize and store a chunk of KV tokens.

        Args:
            key: (H_kv, chunk_len, D)
            value: (H_kv, chunk_len, D)
        """
        H, chunk_len, D = key.shape
        assert D == self.head_dim
        assert H == self.num_kv_heads

        if self._num_stored + chunk_len > self.max_tokens:
            logger.error(
                f"Layer {self.layer_idx}: KV store overflow! "
                f"stored={self._num_stored}, chunk={chunk_len}, max={self.max_tokens}"
            )
            return

        # Quantize keys
        key_q = self.key_quantizer.quantize(key)
        # key_q.mse_indices: (H, chunk_len, packed_d_mse) uint8
        # key_q.qjl_signs:   (H, chunk_len, packed_d_signs) uint8
        # key_q.norms:        (H, chunk_len) float
        # key_q.residual_norms: (H, chunk_len) float

        # Quantize values
        val_q = quantize_values(value, bits=self.value_bits, group_size=self.value_group_size)
        # val_q.data:   (H, chunk_len, D) uint8
        # val_q.scales: (H, chunk_len, n_groups)
        # val_q.zeros:  (H, chunk_len, n_groups)

        # Write into pre-allocated buffers
        start = self._num_stored
        end = start + chunk_len

        self.mse_indices_buf[:, start:end, :].copy_(key_q.mse_indices)
        self.qjl_signs_buf[:, start:end, :].copy_(key_q.qjl_signs)
        self.key_norms_buf[:, start:end].copy_(key_q.norms.to(torch.float16))
        self.res_norms_buf[:, start:end].copy_(key_q.residual_norms.to(torch.float16))

        self.value_data_buf[:, start:end, :].copy_(val_q.data)
        self.value_scales_buf[:, start:end, :].copy_(val_q.scales.to(torch.float16))
        self.value_zeros_buf[:, start:end, :].copy_(val_q.zeros.to(torch.float16))

        self._num_stored = end
        self._n_tensor.fill_(self._num_stored)

    def get_flat_cache(self):
        """
        Get compressed KV as contiguous views for kernel access.

        Returns views into the pre-allocated buffers. The actual valid
        token count is tracked by the device-side _n_tensor.

        For empty stores, still returns valid tensors (needed for CUDA Graph).
        """
        return (
            self.mse_indices_buf,   # (H, max_tokens, packed_d_mse)
            self.qjl_signs_buf,     # (H, max_tokens, packed_d_signs)
            self.key_norms_buf,     # (H, max_tokens)
            self.res_norms_buf,     # (H, max_tokens)
            self.value_data_buf,    # (H, max_tokens, D)
            self.value_scales_buf,  # (H, max_tokens, n_groups)
            self.value_zeros_buf,   # (H, max_tokens, n_groups)
            self._n_tensor,         # (1,) int32 device counter
        )

    def get_quantized_view(self, n: Optional[int] = None):
        """
        Get quantized KV as ProdQuantized + ValueQuantized for score computation.

        Args:
            n: number of tokens to include (default: all stored)
        """
        if n is None:
            n = self._num_stored
        n = min(n, self._num_stored)

        if n == 0:
            return None, None

        key_q = ProdQuantized(
            mse_indices=self.mse_indices_buf[:, :n, :],
            qjl_signs=self.qjl_signs_buf[:, :n, :],
            norms=self.key_norms_buf[:, :n].float(),
            residual_norms=self.res_norms_buf[:, :n].float(),
            mse_bits=self.key_bits - 1,
        )

        val_q = ValueQuantized(
            data=self.value_data_buf[:, :n, :],
            scales=self.value_scales_buf[:, :n, :].float(),
            zeros=self.value_zeros_buf[:, :n, :].float(),
            bits=self.value_bits,
        )

        return key_q, val_q

    def reset(self):
        """Clear all stored tokens."""
        self._num_stored = 0
        self._n_tensor.zero_()
        self.mse_indices_buf.zero_()
        self.qjl_signs_buf.zero_()
        self.key_norms_buf.zero_()
        self.res_norms_buf.zero_()
        self.value_data_buf.zero_()
        self.value_scales_buf.zero_()
        self.value_zeros_buf.zero_()

    def memory_bytes(self) -> int:
        """Total memory used by this store in bytes."""
        total = 0
        total += self.mse_indices_buf.nelement()       # uint8
        total += self.qjl_signs_buf.nelement()          # uint8
        total += self.key_norms_buf.nelement() * 2      # float16
        total += self.res_norms_buf.nelement() * 2      # float16
        total += self.value_data_buf.nelement()         # uint8
        total += self.value_scales_buf.nelement() * 2   # float16
        total += self.value_zeros_buf.nelement() * 2    # float16
        total += self._n_tensor.nelement() * 4          # int32
        return total
