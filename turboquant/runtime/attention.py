"""
Hybrid attention engine for TurboQuant decode.

Computes attention over two segments:
  1. Compressed KV store (quantized historical tokens)
  2. Ring buffer (recent exact tokens)

Uses fused Triton kernels for maximum efficiency:
  - Fused decode kernel for compressed KV (single-pass online softmax)
  - Fused recent buffer kernel for exact KV
  - Hybrid merge kernel to combine both segments

Supports CUDA-Graph-compatible execution with pre-allocated buffers.
"""

import math
import logging
from typing import Optional

import torch

from turboquant.runtime.context import RequestKVState
from turboquant.runtime.kv_store import CompressedKVStore
from turboquant.runtime.ring_buffer import RingBuffer
from turboquant.core.triton_kernels import (
    tq_fused_decode,
    tq_fused_decode_graph,
    tq_recent_buffer_decode,
    tq_hybrid_merge,
    tq_fused_score,
)

logger = logging.getLogger("turboquant.runtime")


class AttentionEngine:
    """
    Hybrid attention computation engine.

    Dispatches to the optimal kernel path:
      1. Graph path: CUDA-Graph-compatible fused kernels with pre-allocated buffers
      2. Fused path: One-pass fused Triton kernels (no graph)
      3. Score-only path: Score computation + PyTorch value aggregation

    All paths support GQA (Grouped Query Attention) natively.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int,
        key_bits: int,
        device: torch.device,
        group_size: int = 32,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.key_bits = key_bits
        self.device = device
        self.group_size = group_size

        self.gqa_ratio = num_query_heads // num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(head_dim)

        # Quantization params for kernel dispatch
        mse_bits = key_bits - 1
        self.qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

    def compute_decode_attention(
        self,
        query: torch.Tensor,
        state: RequestKVState,
        output_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute decode attention for a single request.

        Dispatches to the appropriate kernel path based on data availability:
        - Compressed only: if no ring buffer tokens
        - Ring buffer only: if no compressed tokens
        - Hybrid: both segments (most common during decode)

        Args:
            query: (num_query_heads, D) or (num_query_heads, 1, D) decode query
            state: per-request KV state (store + ring buffer)
            output_buf: pre-allocated output buffer (for graph mode)

        Returns:
            output: (num_query_heads, D) attention output
        """
        if query.dim() == 3:
            query = query.squeeze(1)  # (QH, D)

        QH, D = query.shape
        has_compressed = state.store.n_stored > 0
        has_ring = state.ring_buffer.count > 0

        if not has_compressed and not has_ring:
            # No KV data — return zeros
            if output_buf is not None:
                output_buf.zero_()
                return output_buf
            return torch.zeros(QH, D, device=self.device, dtype=query.dtype)

        if has_compressed and has_ring:
            return self._hybrid_attention(query, state, output_buf)
        elif has_compressed:
            return self._compressed_attention(query, state, output_buf)
        else:
            return self._ring_buffer_attention(query, state, output_buf)

    def _compressed_attention(
        self,
        query: torch.Tensor,
        state: RequestKVState,
        output_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention over compressed KV store only."""
        QH, D = query.shape

        store = state.store
        key_q, val_q = store.get_quantized_view()
        if key_q is None:
            if output_buf is not None:
                output_buf.zero_()
                return output_buf
            return torch.zeros(QH, D, device=self.device, dtype=query.dtype)

        acc, m, l = tq_fused_decode(
            query=query,
            mse_packed=key_q.mse_indices,
            qjl_signs=key_q.qjl_signs,
            norms=key_q.norms,
            res_norms=key_q.residual_norms,
            centroids=store.key_quantizer.mse_quantizer.centroids,
            v_data=val_q.data,
            v_scales=val_q.scales,
            v_zeros=val_q.zeros,
            Pi=store.key_quantizer.mse_quantizer.Pi,
            S=store.key_quantizer.S,
            mse_bits=key_q.mse_bits,
            qjl_scale=self.qjl_scale,
            sm_scale=self.sm_scale,
            gqa_ratio=self.gqa_ratio,
            group_size=self.group_size,
        )

        # Normalize: out = acc / l
        output = acc / l.unsqueeze(-1).clamp(min=1e-10)
        output = output.to(query.dtype)

        if output_buf is not None:
            output_buf.copy_(output)
            return output_buf
        return output

    def _ring_buffer_attention(
        self,
        query: torch.Tensor,
        state: RequestKVState,
        output_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention over ring buffer only."""
        QH, D = query.shape
        ring = state.ring_buffer

        query_f32 = query.float()
        acc, m, l = tq_recent_buffer_decode(
            query=query_f32,
            ring_k=ring.keys,
            ring_v=ring.values,
            count_tensor=ring._count_tensor,
            arange_buf=ring._arange_tensor,
            cap_tensor=ring._cap_tensor,
            sm_scale=self.sm_scale,
            gqa_ratio=self.gqa_ratio,
        )

        output = acc / l.unsqueeze(-1).clamp(min=1e-10)
        output = output.to(query.dtype)

        if output_buf is not None:
            output_buf.copy_(output)
            return output_buf
        return output

    def _hybrid_attention(
        self,
        query: torch.Tensor,
        state: RequestKVState,
        output_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hybrid attention: compressed KV + ring buffer.

        Two-pass online softmax:
        1. Fused decode over compressed KV → (acc_c, m_c, l_c)
        2. Fused decode over ring buffer → (acc_r, m_r, l_r)
        3. Merge: out = (acc_c * alpha_c + acc_r * alpha_r) / l_merged
        """
        QH, D = query.shape
        store = state.store
        ring = state.ring_buffer

        key_q, val_q = store.get_quantized_view()

        # Pass 1: Compressed KV attention
        acc_c, m_c, l_c = tq_fused_decode(
            query=query,
            mse_packed=key_q.mse_indices,
            qjl_signs=key_q.qjl_signs,
            norms=key_q.norms,
            res_norms=key_q.residual_norms,
            centroids=store.key_quantizer.mse_quantizer.centroids,
            v_data=val_q.data,
            v_scales=val_q.scales,
            v_zeros=val_q.zeros,
            Pi=store.key_quantizer.mse_quantizer.Pi,
            S=store.key_quantizer.S,
            mse_bits=key_q.mse_bits,
            qjl_scale=self.qjl_scale,
            sm_scale=self.sm_scale,
            gqa_ratio=self.gqa_ratio,
            group_size=self.group_size,
        )

        # Pass 2: Ring buffer attention
        query_f32 = query.float()
        acc_r, m_r, l_r = tq_recent_buffer_decode(
            query=query_f32,
            ring_k=ring.keys,
            ring_v=ring.values,
            count_tensor=ring._count_tensor,
            arange_buf=ring._arange_tensor,
            cap_tensor=ring._cap_tensor,
            sm_scale=self.sm_scale,
            gqa_ratio=self.gqa_ratio,
        )

        # Pass 3: Merge
        output = tq_hybrid_merge(acc_c, m_c, l_c, acc_r, m_r, l_r)
        output = output.to(query.dtype)

        if output_buf is not None:
            output_buf.copy_(output)
            return output_buf
        return output

    def compute_decode_attention_graph(
        self,
        query: torch.Tensor,
        state: RequestKVState,
        layer_buffers: dict,
    ) -> torch.Tensor:
        """
        CUDA-Graph-compatible decode attention.

        Uses pre-allocated buffers at fixed addresses. All intermediate
        tensors are allocated once and reused across graph replays.

        Args:
            query: (QH, D) query (same tensor address each replay)
            state: per-request KV state
            layer_buffers: dict of pre-allocated buffers for this layer

        Returns:
            output: (QH, D) attention output (into pre-allocated buffer)
        """
        QH, D = query.shape
        store = state.store
        ring = state.ring_buffer
        has_compressed = store.n_stored > 0
        has_ring = ring.count > 0

        if not has_compressed and not has_ring:
            layer_buffers["output"].zero_()
            return layer_buffers["output"]

        # Pre-compute rotated/sketched queries
        Pi = store.key_quantizer.mse_quantizer.Pi
        S = store.key_quantizer.S

        q_f32 = query.float()  # Original query for ring buffer (un-rotated)
        q_rot = torch.matmul(q_f32, Pi.T)
        q_sketch = torch.matmul(q_f32, S.T)

        if has_compressed and has_ring:
            # Compressed pass
            key_q, val_q = store.get_quantized_view()
            flat = store.get_flat_cache()

            tq_fused_decode_graph(
                q_rot=q_rot,
                q_sketch=q_sketch,
                mse_packed=flat[0],  # mse_indices_buf
                qjl_signs=flat[1],   # qjl_signs_buf
                norms=flat[2].float(),
                res_norms=flat[3].float(),
                centroids=store.key_quantizer.mse_quantizer.centroids,
                v_data=flat[4],      # value_data_buf
                v_scales=flat[5].float(),
                v_zeros=flat[6].float(),
                n_tensor=flat[7],
                acc_buf=layer_buffers["comp_acc"],
                m_buf=layer_buffers["comp_m"],
                l_buf=layer_buffers["comp_l"],
                mse_bits=store.key_bits - 1,
                qjl_scale=self.qjl_scale,
                sm_scale=self.sm_scale,
                gqa_ratio=self.gqa_ratio,
                group_size=self.group_size,
            )

            # Ring buffer pass (use un-rotated query — ring buffer keys are exact)
            tq_recent_buffer_decode(
                query=q_f32,
                ring_k=ring.keys,
                ring_v=ring.values,
                count_tensor=ring._count_tensor,
                arange_buf=ring._arange_tensor,
                cap_tensor=ring._cap_tensor,
                sm_scale=self.sm_scale,
                gqa_ratio=self.gqa_ratio,
                acc_buf=layer_buffers["ring_acc"],
                m_buf=layer_buffers["ring_m"],
                l_buf=layer_buffers["ring_l"],
            )

            # Merge
            tq_hybrid_merge(
                acc_c=layer_buffers["comp_acc"],
                m_c=layer_buffers["comp_m"],
                l_c=layer_buffers["comp_l"],
                acc_r=layer_buffers["ring_acc"],
                m_r=layer_buffers["ring_m"],
                l_r=layer_buffers["ring_l"],
                out_buf=layer_buffers["output"],
            )

        elif has_compressed:
            key_q, val_q = store.get_quantized_view()
            flat = store.get_flat_cache()

            tq_fused_decode_graph(
                q_rot=q_rot,
                q_sketch=q_sketch,
                mse_packed=flat[0],
                qjl_signs=flat[1],
                norms=flat[2].float(),
                res_norms=flat[3].float(),
                centroids=store.key_quantizer.mse_quantizer.centroids,
                v_data=flat[4],
                v_scales=flat[5].float(),
                v_zeros=flat[6].float(),
                n_tensor=flat[7],
                acc_buf=layer_buffers["output"],
                m_buf=layer_buffers["comp_m"],
                l_buf=layer_buffers["comp_l"],
                mse_bits=store.key_bits - 1,
                qjl_scale=self.qjl_scale,
                sm_scale=self.sm_scale,
                gqa_ratio=self.gqa_ratio,
                group_size=self.group_size,
            )

            # Normalize
            l = layer_buffers["comp_l"].unsqueeze(-1).clamp(min=1e-10)
            layer_buffers["output"].div_(l)

        else:
            # Ring buffer only (use un-rotated query)
            tq_recent_buffer_decode(
                query=q_f32,
                ring_k=ring.keys,
                ring_v=ring.values,
                count_tensor=ring._count_tensor,
                arange_buf=ring._arange_tensor,
                cap_tensor=ring._cap_tensor,
                sm_scale=self.sm_scale,
                gqa_ratio=self.gqa_ratio,
                acc_buf=layer_buffers["output"],
                m_buf=layer_buffers["ring_m"],
                l_buf=layer_buffers["ring_l"],
            )

            l = layer_buffers["ring_l"].unsqueeze(-1).clamp(min=1e-10)
            layer_buffers["output"].div_(l)

        output = layer_buffers["output"].to(query.dtype)
        return output

    @staticmethod
    def preallocate_layer_buffers(
        num_query_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> dict:
        """Pre-allocate all buffers for CUDA-Graph-compatible decode."""
        QH = num_query_heads
        D = head_dim

        buffers = {
            "output": torch.zeros(QH, D, device=device, dtype=torch.float32),
            "comp_acc": torch.zeros(QH, D, device=device, dtype=torch.float32),
            "comp_m": torch.zeros(QH, device=device, dtype=torch.float32),
            "comp_l": torch.zeros(QH, device=device, dtype=torch.float32),
            "ring_acc": torch.zeros(QH, D, device=device, dtype=torch.float32),
            "ring_m": torch.zeros(QH, device=device, dtype=torch.float32),
            "ring_l": torch.zeros(QH, device=device, dtype=torch.float32),
        }
        return buffers
