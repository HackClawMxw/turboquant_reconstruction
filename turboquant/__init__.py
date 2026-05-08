"""
TurboQuant — KV Cache Compression for LLM Inference (v2.0)

Refactored architecture with request isolation for multi-tenant
concurrent inference support.

Key improvements over v1:
  1. Request-isolated KV state: each request gets its own CompressedKVStore
     and RingBuffer, eliminating data pollution between concurrent requests
  2. Abstract adapter layer: framework-agnostic integration via FrameworkAdapter
     interface (vLLM, vLLM-Ascend, etc.)
  3. Fused Triton kernels: MSE+QJL scoring combined into single kernel
  4. No page cache after prefill: extreme memory savings (trade compute for storage)
  5. GPU monitoring tool: real-time compute and memory load monitoring
  6. Removed direct computation mode (pseudo-quantization path)

Subpackages:
  - core:       Quantization algorithms (preserved from original)
  - runtime:    Request-isolated KV cache management and attention engine
  - adapter:    Framework integration adapters (vLLM, vLLM-Ascend)
  - monitor:    GPU compute and memory monitoring
  - utils:      Memory estimation utilities
"""

__version__ = "2.0.0"
