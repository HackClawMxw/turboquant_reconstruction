"""
vLLM adapter for TurboQuant.

Implements the FrameworkAdapter interface for vLLM (NVIDIA GPU).
Uses monkey-patching (closure) to intercept attention forward calls
and route decode attention through TurboQuant's request-isolated pipeline.

Key design changes from original:
  - Per-request RequestSlotManager instead of shared LayerSlotPool
  - Request IDs from vLLM scheduler instead of block table heuristics
  - Cleaner separation: capture in forward, compute in AttentionEngine
  - No direct computation mode (removed pseudo-quantization path)
  - Abstract adapter interface enables future vLLM-Ascend support

Supports vLLM >= 0.18.0.
"""

from __future__ import annotations

import math
import logging
import time
import types
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
import torch.nn.functional as F

from turboquant.adapter.base import (
    FrameworkAdapter,
    AttentionLayerInfo,
    SequenceInfo,
)
from turboquant.runtime.context import RequestSlotManager, RequestKVState
from turboquant.runtime.capture import KVCaptureEngine
from turboquant.runtime.attention import AttentionEngine

logger = logging.getLogger("turboquant.adapter.vllm")


# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class TQConfig:
    """TurboQuant configuration."""
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128
    max_tokens_per_request: int = 4096
    max_num_seqs: int = 256
    no_alloc: bool = True  # Free paged KV cache after prefill


# ── Request Tracking ────────────────────────────────────────────────────

class RequestTracker:
    """
    Tracks request lifecycle for per-request KV state management.

    Maps vLLM sequence indices to TurboQuant request IDs across
    prefill and decode steps.

    Assumptions (true for vLLM v1 typical scheduling):
    - At most one prefill at a time
    - Prefill and decode are in separate batches
    - Requests complete in roughly FIFO order
    - Layer 0 is always called first in each forward pass
    """

    def __init__(self):
        self._req_counter: int = 0
        self._pending_prefill_id: Optional[str] = None
        self._active_decode_ids: list[str] = []

    def on_prefill(self) -> str:
        """Get/create request ID for current prefill.

        Returns the same ID for chunked prefill (multiple chunks
        for one request).
        """
        if self._pending_prefill_id is None:
            self._pending_prefill_id = f"tq_req_{self._req_counter}"
            self._req_counter += 1
        return self._pending_prefill_id

    def on_decode_start(self, num_seqs: int) -> list[str]:
        """Called at the start of a decode step (layer 0 only).

        Moves pending prefill to active decode list.
        Removes finished requests (FIFO assumption).

        Returns list of finished request IDs for cleanup.
        """
        finished = []

        if self._pending_prefill_id is not None:
            self._active_decode_ids.append(self._pending_prefill_id)
            self._pending_prefill_id = None

        # Trim finished requests (FIFO: oldest first)
        if len(self._active_decode_ids) > num_seqs:
            n_finished = len(self._active_decode_ids) - num_seqs
            finished = self._active_decode_ids[:n_finished]
            self._active_decode_ids = self._active_decode_ids[n_finished:]

        return finished

    def get_decode_id(self, seq_idx: int) -> Optional[str]:
        """Get request ID for a decode sequence index."""
        if 0 <= seq_idx < len(self._active_decode_ids):
            return self._active_decode_ids[seq_idx]
        return None

    def active_count(self) -> int:
        return len(self._active_decode_ids)

    def reset(self):
        """Reset all tracking state."""
        self._pending_prefill_id = None
        self._active_decode_ids.clear()


# ── Layer Registration ─────────────────────────────────────────────────

@dataclass
class VllmLayerState:
    """Per-layer state: slot manager + attention engine + config."""
    layer_info: AttentionLayerInfo
    config: TQConfig
    slot_manager: RequestSlotManager
    attention_engine: AttentionEngine
    layer_buffers: Optional[dict] = None  # Pre-allocated CUDA Graph buffers
    # Shared across all layers (set by adapter after all layers are created)
    tracker: Optional[RequestTracker] = None
    all_layer_states: Optional[dict] = None

    def get_or_create_state(self, request_id: str) -> RequestKVState:
        return self.slot_manager.allocate(request_id)

    def get_state(self, request_id: str) -> Optional[RequestKVState]:
        return self.slot_manager.get(request_id)

    def release_state(self, request_id: str):
        self.slot_manager.release(request_id)


# ── vLLM Adapter ───────────────────────────────────────────────────────

class VllmAdapter(FrameworkAdapter):
    """
    TurboQuant adapter for vLLM on NVIDIA GPUs.

    Monkey-patches vLLM attention layers to intercept KV writes
    and route decode attention through TurboQuant's request-isolated
    compressed KV pipeline.
    """

    def __init__(self, config: Optional[TQConfig] = None):
        self.config = config or TQConfig()
        self._layer_states: dict[str, VllmLayerState] = {}
        self._installed = False

    # ── FrameworkAdapter interface ──────────────────────────────────────

    @staticmethod
    def _get_static_ctx(model) -> dict:
        """Resolve static_forward_context across vLLM versions.

        v0.18.0+: model.compilation_config.static_forward_context
        Older:    model.static_forward_context
        """
        cfg = getattr(model, 'compilation_config', None)
        if cfg is not None:
            return getattr(cfg, 'static_forward_context', {})
        return getattr(model, 'static_forward_context', {})

    def discover_layers(self, model: Any) -> list[AttentionLayerInfo]:
        """Discover attention layers from vLLM model runner."""
        layers = []
        static_ctx = self._get_static_ctx(model)

        for idx, (name, attn_module) in enumerate(static_ctx.items()):
            impl = getattr(attn_module, 'impl', None)
            if impl is None:
                continue

            # Skip MLA layers (TQ doesn't compress those)
            if _is_mla_impl(impl):
                logger.info(f"[TQ] Skipping MLA layer: {name}")
                continue

            # head_size is on impl, not on attn_module (vLLM convention)
            if hasattr(impl, "head_size"):
                head_dim = int(impl.head_size)
            elif hasattr(impl, "kv_lora_rank"):
                head_dim = int(impl.kv_lora_rank)
            else:
                continue

            num_kv_heads = getattr(impl, "num_kv_heads", None)
            if num_kv_heads is None:
                continue
            num_kv_heads = int(num_kv_heads)

            num_query_heads = _infer_num_query_heads(attn_module, impl)

            layers.append(AttentionLayerInfo(
                layer_name=name,
                layer_idx=idx,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                num_query_heads=num_query_heads,
                device=model.device,
                dtype=getattr(
                    getattr(model, 'model_config', None),
                    'dtype', torch.bfloat16,
                ),
                is_mla=False,
            ))

        return layers

    def get_sequence_info(self, attn_metadata: Any) -> list[SequenceInfo]:
        """Extract per-request sequence info from vLLM metadata."""
        infos = []
        num_reqs = getattr(attn_metadata, 'num_reqs', None)
        if num_reqs is None:
            # Single request or older vLLM version
            return [SequenceInfo(
                request_id="default",
                seq_len=getattr(attn_metadata, 'num_actual_tokens', 0),
                num_new_tokens=getattr(attn_metadata, 'num_actual_tokens', 0),
                is_prefill=getattr(attn_metadata, 'max_query_len', 0) > 1,
            )]

        seq_lens = getattr(attn_metadata, 'seq_lens', None)
        query_start_loc = getattr(attn_metadata, 'query_start_loc', None)
        is_prefilling = getattr(attn_metadata, 'is_prefilling', None)

        for i in range(num_reqs):
            sl = int(seq_lens[i]) if seq_lens is not None else 0
            is_pf = bool(is_prefilling[i]) if is_prefilling is not None else (sl > 1)
            n_new = 1 if not is_pf else sl

            infos.append(SequenceInfo(
                request_id=f"req_{i}",  # Will be overridden by block table
                seq_len=sl,
                num_new_tokens=n_new,
                is_prefill=is_pf,
            ))

        return infos

    def get_request_id(self, seq_info: SequenceInfo) -> str:
        return seq_info.request_id

    def extract_kv_tensors(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape vLLM KV to (H_kv, seq_len, D)."""
        # vLLM format: (num_tokens, H_kv, D) or (num_tokens, D)
        if key.dim() == 2:
            key = key.view(-1, layer_info.num_kv_heads, layer_info.head_dim)
            value = value.view(-1, layer_info.num_kv_heads, layer_info.head_dim)

        # Transpose to (H_kv, num_tokens, D)
        return key.transpose(0, 1), value.transpose(0, 1)

    def write_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        layer_info: AttentionLayerInfo,
    ):
        """Write TQ output into vLLM's expected format."""
        # TQ output: (QH, D) or (num_tokens, QH, D)
        if target.dim() == 2:
            target.copy_(output.reshape(-1, layer_info.num_query_heads * layer_info.head_dim))
        else:
            target.copy_(output)

    def install_hooks(self, model: Any, config: dict = None) -> dict:
        """
        Install TurboQuant hooks into vLLM model runner.

        This is the main entry point. Creates per-layer state,
        monkey-patches attention forward methods, and pre-allocates
        CUDA Graph buffers.

        Args:
            model: vLLM GPUModelRunner
            config: Override TQ config parameters

        Returns:
            dict of installed hooks and state
        """
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        layers = self.discover_layers(model)
        print(f"[TurboQuant] Discovered {len(layers)} attention layers", flush=True)

        hooks = {}
        static_ctx = self._get_static_ctx(model)

        for layer_info in layers:
            # Create per-layer state
            slot_mgr = RequestSlotManager(
                layer_idx=layer_info.layer_idx,
                head_dim=layer_info.head_dim,
                num_kv_heads=layer_info.num_kv_heads,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                value_group_size=self.config.value_group_size,
                ring_capacity=self.config.ring_capacity,
                max_tokens=self.config.max_tokens_per_request,
                max_num_seqs=self.config.max_num_seqs,
                device=layer_info.device,
                dtype=layer_info.dtype,
            )

            attn_engine = AttentionEngine(
                head_dim=layer_info.head_dim,
                num_kv_heads=layer_info.num_kv_heads,
                num_query_heads=layer_info.num_query_heads,
                key_bits=self.config.key_bits,
                device=layer_info.device,
                group_size=self.config.value_group_size,
            )

            layer_buffers = AttentionEngine.preallocate_layer_buffers(
                num_query_heads=layer_info.num_query_heads,
                head_dim=layer_info.head_dim,
                device=layer_info.device,
            )

            vllm_state = VllmLayerState(
                layer_info=layer_info,
                config=self.config,
                slot_manager=slot_mgr,
                attention_engine=attn_engine,
                layer_buffers=layer_buffers,
            )

            self._layer_states[layer_info.layer_name] = vllm_state

            # Monkey-patch the attention layer
            attn_module = static_ctx[layer_info.layer_name]
            impl = attn_module.impl

            orig_forward = impl.forward
            patched_forward = _make_patched_forward(
                orig_forward, vllm_state, self.config.no_alloc
            )
            impl.forward = patched_forward

            # Patch KV cache update if no_alloc mode
            if self.config.no_alloc and hasattr(impl, 'do_kv_cache_update'):
                orig_kv_update = impl.do_kv_cache_update
                impl.do_kv_cache_update = _make_patched_kv_update(
                    orig_kv_update, self.config.no_alloc
                )

            hooks[layer_info.layer_name] = {
                "orig_forward": orig_forward,
                "impl": impl,
                "vllm_state": vllm_state,
            }

        self._installed = True
        self._model = model

        # Wire up shared request tracker across all layers
        shared_tracker = RequestTracker()
        for name, ls in self._layer_states.items():
            ls.tracker = shared_tracker
            ls.all_layer_states = self._layer_states

        print(f"[TurboQuant] Installed hooks for {len(hooks)} layers", flush=True)

        # Store layer states on model runner for external access
        model._tq_layer_states = self._layer_states
        model._tq_config = self.config
        model._tq_adapter = self

        return hooks

    def free_kv_cache(self, model: Any) -> int:
        """Free paged KV cache for TQ-managed layers."""
        freed = 0
        kv_caches = getattr(model, 'kv_caches', None) or []

        for layer_name, state in self._layer_states.items():
            layer_idx = state.layer_info.layer_idx
            if layer_idx < len(kv_caches):
                cache_tensor = kv_caches[layer_idx]
                if cache_tensor is not None:
                    freed += cache_tensor.nelement() * cache_tensor.element_size()
                    # Replace with tiny placeholder
                    kv_caches[layer_idx] = torch.zeros(
                        1, 1, 1, 1, device=cache_tensor.device, dtype=cache_tensor.dtype
                    )

        if freed > 0:
            torch.cuda.empty_cache()
            logger.info(f"[TQ] Freed {freed / 1e9:.2f} GB of paged KV cache")

        return freed

    # ── Public API ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get statistics across all layers."""
        stats = {}
        for name, state in self._layer_states.items():
            active = state.slot_manager.active_count()
            total_mem = state.slot_manager.total_memory_bytes()
            stats[name] = {
                "active_requests": active,
                "memory_bytes": total_mem,
            }
        return stats

    def release_request(self, request_id: str):
        """Release all layer states for a completed request."""
        for state in self._layer_states.values():
            state.release_state(request_id)

    def reset_all(self):
        """Emergency reset of all request states."""
        for state in self._layer_states.values():
            state.slot_manager.reset_all()


# ── Patched forward ────────────────────────────────────────────────────

def _make_patched_forward(orig_fn, vllm_state: VllmLayerState, no_alloc: bool):
    """Create patched attention forward that uses TurboQuant for decode.

    NOTE: The patched function is assigned as ``impl.forward = patched``,
    which makes it a plain function (NOT a bound method).  Therefore the
    signature must match the way vLLM *calls* impl.forward — without an
    explicit ``self_impl`` parameter.  ``orig_fn`` IS a bound method, so
    calling it does not require passing the impl object either.
    """

    # Extract impl reference from the bound method for _prefill_attention_sdpa
    impl_ref = getattr(orig_fn, '__self__', None)

    def patched(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        is_decode = (
            attn_metadata is not None
            and getattr(attn_metadata, 'max_query_len', 0) <= 1
        )
        is_prefill = (
            attn_metadata is not None
            and getattr(attn_metadata, 'max_query_len', 0) > 1
        )

        # ── Prefill: capture KV, use original attention for output ──
        if is_prefill:
            _handle_prefill(vllm_state, key, value, attn_metadata, no_alloc)
            if no_alloc:
                result = _prefill_attention_sdpa(
                    vllm_state, impl_ref, query, key, value, attn_metadata
                )
                _write_result(result, output, vllm_state.layer_info, attn_metadata)
                return output
            return orig_fn(
                layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # ── Decode: capture KV, compute TQ attention ──
        if is_decode:
            _handle_decode_capture(vllm_state, key, value, attn_metadata)

            num_actual = getattr(attn_metadata, 'num_actual_tokens', query.shape[0])

            # In vLLM decode mode, each token is a separate request.
            # num_actual > 1 means multiple concurrent requests.
            if num_actual > 1:
                return _multi_seq_decode(
                    vllm_state, query, num_actual, num_actual,
                    attn_metadata, output, orig_fn,
                    layer, key, value, kv_cache,
                    output_scale, output_block_scale,
                )
            else:
                # Single-sequence decode
                return _single_seq_decode(
                    vllm_state, query, num_actual, attn_metadata,
                    output, orig_fn,
                    layer, key, value, kv_cache,
                    output_scale, output_block_scale,
                )

        # ── Fallback: no metadata or profiling ──
        return orig_fn(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    return patched


def _handle_prefill(
    vllm_state: VllmLayerState,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
    no_alloc: bool,
):
    """Capture KV from prefill into per-request store via capture engine.

    Uses RequestTracker to assign a unique per-request ID.
    Each request gets its own CompressedKVStore + RingBuffer, ensuring
    complete data isolation between concurrent requests.

    For chunked prefill, the same request ID is reused across chunks
    (tracker returns the same pending ID until decode starts).
    """
    num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
    tracker = vllm_state.tracker

    # Get or create per-request state
    if tracker is not None:
        request_id = tracker.on_prefill()
    else:
        request_id = "prefill_fallback"

    state = vllm_state.get_state(request_id)
    if state is None:
        state = vllm_state.get_or_create_state(request_id)
    else:
        # Chunked prefill continuation — state already exists.
        # Do NOT reset; just keep appending.
        pass

    # Reshape from vLLM (T, H, D) to TQ (H, T, D)
    k, v = _reshape_kv(key[:num_tokens], value[:num_tokens], vllm_state.layer_info)

    # Use capture engine for proper prefill splitting (ring buffer + store)
    state.capture_engine.ingest_prefill(k, v)

    state.num_tokens = num_tokens
    state.is_prefill_complete = True


def _handle_decode_capture(
    vllm_state: VllmLayerState,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
):
    """Capture decode KV tokens into per-request stores via capture engine.

    Uses RequestTracker to route each token to its own per-request state.
    On layer 0, triggers request lifecycle management (adding new requests,
    cleaning up finished ones).
    """
    num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
    tracker = vllm_state.tracker
    is_first_layer = vllm_state.layer_info.layer_idx == 0

    # Layer 0: update tracker and clean up finished requests
    if is_first_layer and tracker is not None:
        finished = tracker.on_decode_start(num_tokens)
        for rid in finished:
            if vllm_state.all_layer_states is not None:
                for ls in vllm_state.all_layer_states.values():
                    ls.release_state(rid)

    # Capture each token into its per-request state
    if tracker is not None:
        for si in range(num_tokens):
            req_id = tracker.get_decode_id(si)
            if req_id is None:
                continue
            state = vllm_state.get_state(req_id)
            if state is None:
                continue
            # Single token: (1, H_kv, D) → (H_kv, 1, D)
            k, v = _reshape_kv(
                key[si:si+1], value[si:si+1], vllm_state.layer_info
            )
            state.capture_engine.ingest_decode(k, v)
            state.num_tokens += 1
    else:
        # Fallback: no tracker, use legacy shared state
        request_id = "prefill_fallback"
        state = vllm_state.get_state(request_id)
        if state is None:
            state = vllm_state.get_or_create_state("decode_fallback")
        k, v = _reshape_kv(key[:num_tokens], value[:num_tokens], vllm_state.layer_info)
        state.capture_engine.ingest_decode(k, v)
        state.num_tokens += num_tokens


def _single_seq_decode(
    vllm_state: VllmLayerState,
    query: torch.Tensor,
    num_actual: int,
    attn_metadata,
    output,
    orig_fn, layer, key, value, kv_cache,
    output_scale, output_block_scale,
):
    """Single-sequence TQ decode attention using per-request state."""
    tracker = vllm_state.tracker
    req_id = tracker.get_decode_id(0) if tracker else None
    state = vllm_state.get_state(req_id) if req_id else None

    # Only fall back to orig_fn when there is NO TQ data at all.
    # With no_alloc mode, orig_fn reads from an empty paged KV cache,
    # so we must use TQ attention whenever ring buffer or store has data.
    if state is None or (state.store.n_stored == 0 and state.ring_buffer.count == 0):
        # No compressed data yet — fallback to original
        return orig_fn(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    # Reshape query
    q = query[:num_actual]
    if q.dim() == 2:
        q = q.view(1, vllm_state.layer_info.num_query_heads, vllm_state.layer_info.head_dim)
    q = q.squeeze(0)  # (QH, D)

    # Compute TQ attention
    result = vllm_state.attention_engine.compute_decode_attention(q, state)
    _write_result(result, output, vllm_state.layer_info, attn_metadata, num_actual)
    return output


def _multi_seq_decode(
    vllm_state: VllmLayerState,
    query: torch.Tensor,
    num_actual: int,
    num_reqs: int,
    attn_metadata,
    output,
    orig_fn, layer, key, value, kv_cache,
    output_scale, output_block_scale,
):
    """Multi-sequence TQ decode attention using per-request states.

    Each request uses its own per-request KV state for attention,
    ensuring complete data isolation between concurrent requests.
    """
    info = vllm_state.layer_info
    tracker = vllm_state.tracker

    # Check if any request has TQ data
    has_any_tq = False
    req_states = []
    for si in range(num_reqs):
        req_id = tracker.get_decode_id(si) if tracker else None
        state = vllm_state.get_state(req_id) if req_id else None
        req_states.append(state)
        if state is not None and (
            state.store.n_stored > 0 or state.ring_buffer.count > 0
        ):
            has_any_tq = True

    if not has_any_tq:
        # No TQ data — pure orig_fn for all
        return orig_fn(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    # Run orig_fn as baseline (handles paged cache for non-TQ layers)
    orig_fn(
        layer, query, key, value, kv_cache,
        attn_metadata, output, output_scale, output_block_scale,
    )

    # Overwrite each sequence with TQ attention from its own state
    for si in range(num_reqs):
        state = req_states[si]
        if state is None or (
            state.store.n_stored == 0 and state.ring_buffer.count == 0
        ):
            continue  # Keep orig_fn result for this sequence

        q = query[si:si+1]
        if q.dim() == 2:
            q = q.view(1, info.num_query_heads, info.head_dim)
        q = q.squeeze(0)  # (QH, D)

        result = vllm_state.attention_engine.compute_decode_attention(q, state)

        result_flat = result.reshape(1, info.num_query_heads * info.head_dim)
        if output is not None:
            out_slice = output[si:si+1]
            if out_slice.dim() == 3:
                out_slice.copy_(result.unsqueeze(0))
            else:
                out_slice.copy_(result_flat)

    return output


def _prefill_attention_sdpa(
    vllm_state: VllmLayerState,
    self_impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
):
    """Compute prefill attention using SDPA (no paged cache needed)."""
    num_actual = getattr(attn_metadata, 'num_actual_tokens', query.shape[0])
    info = vllm_state.layer_info

    q = query[:num_actual]
    k = key[:num_actual]
    v = value[:num_actual]

    if q.dim() == 2:
        q = q.view(num_actual, info.num_query_heads, info.head_dim)
        k = k.view(num_actual, info.num_kv_heads, info.head_dim)
        v = v.view(num_actual, info.num_kv_heads, info.head_dim)

    if info.num_query_heads != info.num_kv_heads:
        repeats = info.num_query_heads // info.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    q_t = q.unsqueeze(0).transpose(1, 2)  # (1, QH, T, D)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)

    scale = getattr(self_impl, "scale", 1.0 / math.sqrt(info.head_dim))
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1)  # (T, QH, D)


def _write_result(
    result: torch.Tensor,
    output: Optional[torch.Tensor],
    layer_info: AttentionLayerInfo,
    attn_metadata,
    num_actual: int = None,
):
    """Write TQ result into vLLM's output tensor."""
    if output is None:
        return

    if num_actual is None:
        num_actual = getattr(attn_metadata, 'num_actual_tokens', result.shape[0])

    result_flat = result.reshape(num_actual, layer_info.num_query_heads * layer_info.head_dim)
    out_slice = output[:num_actual]
    if out_slice.dim() == 3:
        out_slice.copy_(result.to(out_slice.dtype))
    else:
        out_slice.copy_(result_flat.to(out_slice.dtype))


def _reshape_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_info: AttentionLayerInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape KV from vLLM format (T, H, D) to TQ format (H, T, D)."""
    if key.dim() == 2:
        key = key.view(-1, layer_info.num_kv_heads, layer_info.head_dim)
        value = value.view(-1, layer_info.num_kv_heads, layer_info.head_dim)
    return key.transpose(0, 1), value.transpose(0, 1)


def _make_patched_kv_update(orig_fn, no_alloc: bool):
    """Patch KV cache update — skip when no_alloc.

    Same binding issue as _make_patched_forward: assigned as plain function.
    """

    def patched(layer, key, value, kv_cache, slot_mapping):
        if not no_alloc:
            orig_fn(layer, key, value, kv_cache, slot_mapping)

    return patched


# ── Helper functions ───────────────────────────────────────────────────

def _is_mla_impl(impl) -> bool:
    """Detect Multi-head Latent Attention (MLA/GDN) backends."""
    return (
        hasattr(impl, "forward_mqa")
        and hasattr(impl, "do_kv_cache_update")
        and not hasattr(impl, "forward")
    )


def _infer_num_query_heads(attn_module, impl) -> int:
    """Infer number of query heads from attention module."""
    for candidate in (
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(impl, "num_heads", None),
    ):
        if candidate:
            return int(candidate)
    return int(impl.num_kv_heads)


# ── Public convenience functions ───────────────────────────────────────

def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    max_tokens: int = 4096,
    max_num_seqs: int = 256,
    no_alloc: bool = True,
) -> VllmAdapter:
    """
    Install TurboQuant hooks into a vLLM model runner.

    Convenience function that creates a VllmAdapter and installs hooks.

    Args:
        model_runner: vLLM GPUModelRunner
        key_bits: Bits per key quantization (2-4)
        value_bits: Bits per value quantization (2 or 4)
        value_group_size: Elements per value quantization group
        ring_capacity: Recent exact token buffer size
        max_tokens: Max tokens per request in compressed store
        max_num_seqs: Max concurrent sequences
        no_alloc: Free paged KV cache after prefill

    Returns:
        VllmAdapter instance for further control
    """
    config = TQConfig(
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=ring_capacity,
        max_tokens_per_request=max_tokens,
        max_num_seqs=max_num_seqs,
        no_alloc=no_alloc,
    )

    adapter = VllmAdapter(config)
    adapter.install_hooks(model_runner)

    # NOTE: Do NOT call free_kv_cache here — at this point (during
    # get_kv_cache_specs) the KV cache has not been allocated yet.
    # The paged cache for the single target layer will remain allocated
    # but unused; its memory cost is negligible compared to the N-1
    # layers whose specs were removed by enable_no_alloc.

    return adapter


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache for all TQ-managed layers."""
    adapter = getattr(model_runner, '_tq_adapter', None)
    if adapter is not None:
        return adapter.free_kv_cache(model_runner)
    return 0


def get_stats(model_runner) -> dict:
    """Get TurboQuant statistics."""
    adapter = getattr(model_runner, '_tq_adapter', None)
    if adapter is not None:
        return adapter.get_stats()
    return {}


# ── Pre-import patching (must be called before vLLM engine init) ────────

_TQ_NO_ALLOC_CONFIG = None


def enable_no_alloc(
    key_bits: int = 4,
    value_bits: int = 3,
    buffer_size: int = 128,
    initial_layers_count: int = 32,
    value_group_size: int = 32,
    max_num_seqs: int = 1,
):
    """Call BEFORE creating vLLM engine.

    Patches Executor.get_kv_cache_specs so TQ hooks are installed
    automatically during engine initialization.

    Args:
        key_bits: Bits per key quantization
        value_bits: Bits per value quantization
        buffer_size: Ring buffer capacity (recent exact tokens)
        initial_layers_count: Number of layers (unused, kept for compat)
        value_group_size: Value quantization group size
        max_num_seqs: Max concurrent request slots per layer
    """
    global _TQ_NO_ALLOC_CONFIG
    _TQ_NO_ALLOC_CONFIG = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        value_group_size=value_group_size,
        max_num_seqs=max_num_seqs,
    )

    from vllm.v1.executor.abstract import Executor

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        GPUModelRunner = None

    if hasattr(Executor, "_tq_patched"):
        return

    # Patch layout update for shared KV cache layers (hybrid attention+mamba)
    if (
        GPUModelRunner is not None
        and not hasattr(GPUModelRunner, "_tq_layout_patch")
        and hasattr(GPUModelRunner, "_update_hybrid_attention_mamba_layout")
    ):
        _orig_layout_update = GPUModelRunner._update_hybrid_attention_mamba_layout

        def _patched_layout_update(self_runner, kv_caches):
            for layer_name, target_name in getattr(
                self_runner, "shared_kv_cache_layers", {}
            ).items():
                if layer_name not in kv_caches and target_name in kv_caches:
                    kv_caches[layer_name] = kv_caches[target_name]
            return _orig_layout_update(self_runner, kv_caches)

        GPUModelRunner._update_hybrid_attention_mamba_layout = _patched_layout_update
        GPUModelRunner._tq_layout_patch = True

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _TQ_NO_ALLOC_CONFIG
        if cfg is None:
            return orig_get_specs(self)

        def _worker_install_tq(worker):
            from turboquant.adapter.vllm_adapter import install_hooks
            adapter = install_hooks(
                worker.model_runner,
                key_bits=cfg["key_bits"],
                value_bits=cfg["value_bits"],
                value_group_size=cfg["value_group_size"],
                ring_capacity=cfg["buffer_size"],
                max_num_seqs=cfg.get("max_num_seqs", 1),
                no_alloc=True,
            )

            # ── KV sharing: all TQ-managed layers share one paged cache ──
            # This reduces KV cache allocation from N layers to 1.
            # KV capture is done in patched forward, so skipping
            # do_kv_cache_update for shared layers is safe.
            static_ctx = VllmAdapter._get_static_ctx(worker.model_runner)
            tq_layer_names = list(adapter._layer_states.keys())
            shared_layer_names = []

            if len(tq_layer_names) > 1:
                target = tq_layer_names[0]
                target_attn = static_ctx.get(target)
                if target_attn is not None and hasattr(
                    target_attn, "kv_sharing_target_layer_name"
                ):
                    target_attn.kv_sharing_target_layer_name = None
                for name in tq_layer_names[1:]:
                    attn = static_ctx.get(name)
                    if attn is not None and hasattr(
                        attn, "kv_sharing_target_layer_name"
                    ):
                        attn.kv_sharing_target_layer_name = target
                        shared_layer_names.append(name)

            print(
                f"[TurboQuant] TQ layers: {len(tq_layer_names)}, "
                f"shared: {len(shared_layer_names)}",
                flush=True,
            )
            return {"shared_layer_names": shared_layer_names}

        try:
            hooks = self.collective_rpc(_worker_install_tq)
            print(f"[TurboQuant] collective_rpc OK: {hooks}", flush=True)
        except Exception as e:
            print(f"[TurboQuant] collective_rpc FAILED: {e}", flush=True)
            logger.error("[TurboQuant] collective_rpc FAILED: %s", e, exc_info=True)
            return orig_get_specs(self)

        specs = orig_get_specs(self)

        # Remove specs for shared layers so vLLM doesn't allocate KV cache
        # for them — they'll share the target layer's paged cache instead.
        shared = []
        if hooks and isinstance(hooks, list) and len(hooks) > 0:
            shared = hooks[0].get("shared_layer_names", [])
        for worker_specs in specs:
            for name in shared:
                worker_specs.pop(name, None)
        if shared:
            print(
                f"[TurboQuant] Removed {len(shared)} shared layer specs "
                f"(layers share one paged cache)",
                flush=True,
            )

        return specs

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._tq_patched = True
    logger.info("[TurboQuant] Patched Executor for auto TQ hook installation")
