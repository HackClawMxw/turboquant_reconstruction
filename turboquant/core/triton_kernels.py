"""
TurboQuant fused Triton kernels for decode attention.

Refactored from the original with the following improvements:
1. Fused MSE+QJL scoring into a single kernel (replaces two separate kernel launches)
2. Cleaner separation between graph-compatible and non-graph paths
3. Removed redundant PyTorch fallback paths (direct computation mode eliminated)
4. Value bit-packing helpers inlined for zero-copy access

Kernel overview:
  - _tq_fused_score_kernel:   Combined MSE+QJL score computation (single kernel)
  - _tq_fused_decode_kernel:  Full fused: TQ scores + online softmax + value dequant
  - _tq_fused_decode_graph_kernel: CUDA-Graph-compatible variant (dynamic N)
  - _tq_recent_buffer_kernel: Fused attention over exact ring buffer KV
  - _tq_hybrid_merge_kernel:  Merge two online softmax states
"""

import math
import torch
import triton
import triton.language as tl


# ─── Value bit-packing helpers (shared across kernels) ─────────────────

def unpack_values_uint8(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack bit-packed uint8 value data to per-element uint8."""
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


def _get_packing_params(bits: int):
    """Get packing parameters matching _pack_indices logic."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


# ─── Kernel 1: Fused MSE+QJL Score ────────────────────────────────────
#
# Combines the original separate MSE and QJL score kernels into one.
# This eliminates one kernel launch and one intermediate tensor write/read.

@triton.jit
def _tq_fused_score_kernel(
    # Query (pre-rotated for MSE, pre-sketched for QJL)
    Q_ROT_ptr,       # (QH, D)
    Q_SKETCH_ptr,    # (QH, D)
    # Quantized keys
    MSE_ptr,         # (H_kv, N, packed_d_mse) uint8
    SIGNS_ptr,       # (H_kv, N, packed_d_signs) uint8
    NORMS_ptr,       # (H_kv, N) float
    RES_NORMS_ptr,   # (H_kv, N) float
    CENTROIDS_ptr,   # (n_clusters,) float32
    # Output
    OUT_ptr,         # (QH, N) float32
    # Strides
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_n_kv, stride_n_n,
    stride_rn_kv, stride_rn_n,
    stride_o_qh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    # Quant params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    GQA_RATIO: tl.constexpr,
    # Block
    BLOCK_N: tl.constexpr,
):
    """Fused MSE + QJL score computation for a single query head."""
    pid_q = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_kv = pid_q // GQA_RATIO

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load full query vectors into registers (reused across all packed bytes)
    d_offs = tl.arange(0, D)
    q_rot = tl.load(Q_ROT_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)
    q_sketch = tl.load(Q_SKETCH_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)

    # ── Part 1: MSE score ──
    mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    for byte_idx in range(PACKED_D_MSE):
        packed = tl.load(
            MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0,
        ).to(tl.int32)
        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                q_val = tl.load(Q_ROT_ptr + pid_q * stride_q_qh + coord_idx * stride_q_d).to(tl.float32)
                mse_scores += q_val * centroid_val

    key_norms = tl.load(
        NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    mse_scores = mse_scores * key_norms

    # ── Part 2: QJL score ──
    qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
    for byte_idx in range(PACKED_D_SIGNS):
        packed = tl.load(
            SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0,
        ).to(tl.int32)
        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                qs_val = tl.load(Q_SKETCH_ptr + pid_q * stride_q_qh + coord_idx * stride_q_d).to(tl.float32)
                qjl_dot += qs_val * sign_val

    res_norms = tl.load(
        RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    qjl_scores = qjl_dot * res_norms * QJL_SCALE

    # ── Combined score ──
    scores = mse_scores + qjl_scores

    tl.store(
        OUT_ptr + pid_q * stride_o_qh + n_offs * stride_o_n,
        scores, mask=n_mask,
    )


# ─── Kernel 2: Fused Decode (TQ scores + softmax + value aggregation) ──

@triton.jit
def _tq_fused_decode_kernel(
    # Query (pre-rotated / pre-sketched)
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    # Quantized keys
    MSE_ptr,
    SIGNS_ptr,
    NORMS_ptr,
    RES_NORMS_ptr,
    CENTROIDS_ptr,
    # Values (group-quantized, unpacked to per-element uint8)
    V_DATA_ptr,
    V_SCALES_ptr,
    V_ZEROS_ptr,
    # Outputs: unnormalised accumulator, running max, running sum
    OUT_ptr,
    M_OUT_ptr,
    L_OUT_ptr,
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_n_kv, stride_n_n,
    stride_rn_kv, stride_rn_n,
    stride_v_kv, stride_v_n, stride_v_d,
    stride_vs_kv, stride_vs_n, stride_vs_g,
    stride_vz_kv, stride_vz_n, stride_vz_g,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- dims ---
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # --- quant params ---
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    GQA_RATIO: tl.constexpr,
    # --- block ---
    BLOCK_N: tl.constexpr,
):
    """Fully fused decode: TQ scores + online softmax + value dequant + weighted sum."""
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Load query vectors once into registers
    d_offs = tl.arange(0, D)
    q_rot = tl.load(Q_ROT_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)
    q_sketch = tl.load(Q_SKETCH_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    for block_idx in range(tl.cdiv(N, BLOCK_N)):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── MSE score ──
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ROT_ptr + pid_q * stride_q_qh + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(
            NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── QJL score ──
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    qs_val = tl.load(Q_SKETCH_ptr + pid_q * stride_q_qh + coord_idx * stride_q_d).to(tl.float32)
                qjl_dot += qs_val * sign_val

        res_norms = tl.load(
            RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # Combined score
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Value dequantize + accumulate ──
        v_quant = tl.load(
            V_DATA_ptr + pid_kv * stride_v_kv
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_kv * stride_vs_kv
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_kv * stride_vz_kv
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(M_OUT_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(L_OUT_ptr + pid_q * stride_l_qh, tl.sum(l_i))


# ─── Kernel 3: CUDA-Graph-compatible Fused Decode ─────────────────────
#
# Reads N from device memory (while loop) so graph replay uses current N.
# All buffers must be pre-allocated at fixed addresses.

@triton.jit
def _tq_fused_decode_graph_kernel(
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    MSE_ptr,
    SIGNS_ptr,
    NORMS_ptr,
    RES_NORMS_ptr,
    CENTROIDS_ptr,
    V_DATA_ptr,
    V_SCALES_ptr,
    V_ZEROS_ptr,
    OUT_ptr,
    M_OUT_ptr,
    L_OUT_ptr,
    N_PTR,
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_n_kv, stride_n_n,
    stride_rn_kv, stride_rn_n,
    stride_v_kv, stride_v_n, stride_v_d,
    stride_vs_kv, stride_vs_n, stride_vs_g,
    stride_vz_kv, stride_vz_n, stride_vz_g,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- constexpr dims ---
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Read N dynamically from device memory
    N = tl.load(N_PTR).to(tl.int32)

    # Store base pointers for element-wise access (Triton 3.x workaround)
    q_rot_base = Q_ROT_ptr + pid_q * stride_q_qh
    q_sketch_base = Q_SKETCH_ptr + pid_q * stride_q_qh
    d_offs = tl.arange(0, D)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    block_idx = 0
    while block_idx * BLOCK_N < N:
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── MSE score ──
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(q_rot_base + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(
            NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── QJL score ──
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    s_val = tl.load(q_sketch_base + coord_idx * stride_q_d).to(tl.float32)
                    qjl_dot += s_val * sign_val

        res_norms = tl.load(
            RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Value dequantize + accumulate ──
        v_quant = tl.load(
            V_DATA_ptr + pid_kv * stride_v_kv
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_kv * stride_vs_kv
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_kv * stride_vz_kv
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new
        block_idx += 1

    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(M_OUT_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(L_OUT_ptr + pid_q * stride_l_qh, tl.sum(l_i))


# ─── Kernel 4: Recent Buffer Attention ─────────────────────────────────

@triton.jit
def _tq_recent_buffer_kernel(
    Q_ptr,
    RING_K_ptr,
    RING_V_ptr,
    COUNT_ptr,
    ARANGE_ptr,
    CAP_ptr,
    OUT_ACC_ptr,
    OUT_M_ptr,
    OUT_L_ptr,
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_k_cap, stride_k_h, stride_k_d,
    stride_v_cap, stride_v_h, stride_v_d,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- constexpr dims ---
    D: tl.constexpr,
    CAP: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE,
):
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO

    valid_count = tl.load(COUNT_ptr).to(tl.int32)
    cap_val = tl.load(CAP_ptr).to(tl.int32)
    actual_count = tl.minimum(valid_count, cap_val)

    d_offs = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    for block_start in range(0, CAP, BLOCK_N):
        n_offs = block_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < actual_count

        k_bf16 = tl.load(
            RING_K_ptr + n_offs[:, None] * stride_k_cap + pid_kv * stride_k_h + d_offs[None, :] * stride_k_d,
            mask=n_mask[:, None], other=0,
        )
        k_f32 = k_bf16.to(tl.float32)

        scores = tl.sum(q[None, :] * k_f32, axis=1)
        scores = scores * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        v_bf16 = tl.load(
            RING_V_ptr + n_offs[:, None] * stride_v_cap + pid_kv * stride_v_h + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        )
        v_f32 = v_bf16.to(tl.float32)

        acc += tl.sum(p[:, None] * v_f32, 0)
        m_i = m_new

    tl.store(OUT_ACC_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(OUT_M_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(OUT_L_ptr + pid_q * stride_l_qh, tl.sum(l_i))


# ─── Kernel 5: Hybrid Merge ───────────────────────────────────────────

@triton.jit
def _tq_hybrid_merge_kernel(
    ACC_C_ptr,
    M_C_ptr,
    L_C_ptr,
    ACC_R_ptr,
    M_R_ptr,
    L_R_ptr,
    OUT_ptr,
    stride_a_qh, stride_a_d,
    stride_m_qh,
    stride_l_qh,
    stride_o_qh, stride_o_d,
    D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    d_offs = tl.arange(0, D)

    m_c = tl.load(M_C_ptr + pid_q * stride_m_qh)
    l_c = tl.load(L_C_ptr + pid_q * stride_l_qh)
    m_r = tl.load(M_R_ptr + pid_q * stride_m_qh)
    l_r = tl.load(L_R_ptr + pid_q * stride_l_qh)

    m_merged = tl.maximum(m_c, m_r)
    alpha_c = tl.exp(m_c - m_merged)
    alpha_r = tl.exp(m_r - m_merged)
    l_merged = l_c * alpha_c + l_r * alpha_r

    acc_c = tl.load(ACC_C_ptr + pid_q * stride_a_qh + d_offs * stride_a_d)
    acc_r = tl.load(ACC_R_ptr + pid_q * stride_a_qh + d_offs * stride_a_d)

    acc_merged = acc_c * alpha_c + acc_r * alpha_r
    out = acc_merged / l_merged

    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, out)


# ─── Python Wrappers ──────────────────────────────────────────────────

def tq_fused_score(
    query: torch.Tensor,       # (QH, D) raw query
    mse_packed: torch.Tensor,  # (H_kv, N, packed_d) uint8
    qjl_signs: torch.Tensor,   # (H_kv, N, packed_d_signs) uint8
    norms: torch.Tensor,       # (H_kv, N) float
    res_norms: torch.Tensor,   # (H_kv, N) float
    centroids: torch.Tensor,   # (n_clusters,) float32
    Pi: torch.Tensor,          # (D, D) float32
    S: torch.Tensor,           # (D, D) float32
    mse_bits: int,
    qjl_scale: float,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute TQ attention scores (fused MSE+QJL in a single kernel).

    Args:
        query:       (QH, D) raw query vectors
        mse_packed:  (H_kv, N, packed_d) uint8 bit-packed MSE indices
        qjl_signs:   (H_kv, N, packed_d_signs) uint8 packed sign bits
        norms:       (H_kv, N) key L2 norms
        res_norms:   (H_kv, N) residual L2 norms
        centroids:   (n_clusters,) codebook
        Pi:          (D, D) rotation matrix
        S:           (D, D) QJL projection matrix
        mse_bits:    bits per MSE index
        qjl_scale:   sqrt(pi/2) / D
        gqa_ratio:   num_query_heads / num_kv_heads

    Returns:
        scores: (QH, N) raw logits (before 1/sqrt(d) scaling).
    """
    if query.dim() == 3:
        query = query.squeeze(1)

    QH, D = query.shape
    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    scores = torch.zeros(QH, N, device=query.device, dtype=torch.float32)
    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (QH, triton.cdiv(N, BLOCK_N))

    _tq_fused_score_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        scores,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        scores.stride(0), scores.stride(1),
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
    )

    return scores


def tq_fused_decode(
    query: torch.Tensor,
    mse_packed: torch.Tensor,
    qjl_signs: torch.Tensor,
    norms: torch.Tensor,
    res_norms: torch.Tensor,
    centroids: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    v_zeros: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    gqa_ratio: int,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully fused decode: TQ scores + online softmax + value aggregation.

    Returns:
        acc: (QH, D) unnormalised weighted-sum
        m:   (QH,)   running max of scaled scores
        l:   (QH,)   running sum of exponentials
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    QH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    if v_data.dim() > 3:
        H_kv = mse_packed.shape[0]
        v_data = v_data.reshape(H_kv, N, -1)
        v_scales = v_scales.reshape(H_kv, N, -1)
        v_zeros = v_zeros.reshape(H_kv, N, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    acc = torch.zeros(QH, D, device=query.device, dtype=torch.float32)
    m_out = torch.zeros(QH, device=query.device, dtype=torch.float32)
    l_out = torch.zeros(QH, device=query.device, dtype=torch.float32)

    BLOCK_N = min(64, triton.next_power_of_2(N))
    grid = (QH,)

    _tq_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        acc, m_out, l_out,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        acc.stride(0), acc.stride(1),
        m_out.stride(0),
        l_out.stride(0),
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return acc, m_out, l_out


def tq_fused_decode_graph(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    mse_packed: torch.Tensor,
    qjl_signs: torch.Tensor,
    norms: torch.Tensor,
    res_norms: torch.Tensor,
    centroids: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    v_zeros: torch.Tensor,
    n_tensor: torch.Tensor,
    acc_buf: torch.Tensor,
    m_buf: torch.Tensor,
    l_buf: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    gqa_ratio: int,
    group_size: int = 32,
):
    """CUDA-Graph-compatible fused decode with GQA.

    Reads N from n_tensor device memory at runtime (while loop).
    Writes into pre-allocated acc_buf, m_buf, l_buf.
    All tensor addresses remain stable across calls.

    Returns (acc, m, l) — the SAME tensors passed in.
    """
    QH, D = q_rot.shape

    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]
    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    acc_buf.zero_()
    m_buf.zero_()
    l_buf.zero_()

    BLOCK_N = 64
    grid = (QH,)

    _tq_fused_decode_graph_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        acc_buf, m_buf, l_buf,
        n_tensor,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        acc_buf.stride(0), acc_buf.stride(1),
        m_buf.stride(0),
        l_buf.stride(0),
        D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return acc_buf, m_buf, l_buf


def tq_recent_buffer_decode(
    query: torch.Tensor,
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    count_tensor: torch.Tensor,
    arange_buf: torch.Tensor,
    cap_tensor: torch.Tensor,
    sm_scale: float,
    gqa_ratio: int,
    acc_buf: torch.Tensor = None,
    m_buf: torch.Tensor = None,
    l_buf: torch.Tensor = None,
):
    """Fused recent-buffer attention using Triton.

    Returns (acc, m, l) into pre-allocated buffers.
    """
    QH, D = query.shape
    cap = ring_k.shape[0]

    if acc_buf is None:
        acc_buf = torch.zeros(QH, D, device=query.device, dtype=torch.float32)
    if m_buf is None:
        m_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)
    if l_buf is None:
        l_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)

    acc_buf.zero_()
    m_buf.fill_(float("-inf"))
    l_buf.zero_()

    BLOCK_N = 32
    grid = (QH,)

    _tq_recent_buffer_kernel[grid](
        query, ring_k, ring_v,
        count_tensor, arange_buf, cap_tensor,
        acc_buf, m_buf, l_buf,
        query.stride(0), query.stride(1),
        ring_k.stride(0), ring_k.stride(1), ring_k.stride(2),
        ring_v.stride(0), ring_v.stride(1), ring_v.stride(2),
        acc_buf.stride(0), acc_buf.stride(1),
        m_buf.stride(0),
        l_buf.stride(0),
        D=D, CAP=cap, GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N, SM_SCALE=sm_scale,
    )

    return acc_buf, m_buf, l_buf


def tq_hybrid_merge(
    acc_c: torch.Tensor,
    m_c: torch.Tensor,
    l_c: torch.Tensor,
    acc_r: torch.Tensor,
    m_r: torch.Tensor,
    l_r: torch.Tensor,
    out_buf: torch.Tensor = None,
) -> torch.Tensor:
    """Merge online softmax states from compressed and recent attention."""
    QH, D = acc_c.shape

    if out_buf is None:
        out_buf = torch.zeros(QH, D, device=acc_c.device, dtype=torch.float32)

    grid = (QH,)
    _tq_hybrid_merge_kernel[grid](
        acc_c, m_c, l_c,
        acc_r, m_r, l_r,
        out_buf,
        acc_c.stride(0), acc_c.stride(1),
        m_c.stride(0),
        l_c.stride(0),
        out_buf.stride(0), out_buf.stride(1),
        D=D,
    )

    return out_buf
