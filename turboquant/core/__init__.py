"""
TurboQuant core quantization algorithms.

This package contains the core MSE and inner-product quantization algorithms
from the TurboQuant paper. These modules are preserved from the original
implementation and MUST NOT be modified (project constraint).

- codebook: Lloyd-Max codebook computation for Beta distribution
- rotation: Random rotation and QJL projection matrices
- quantizer: TurboQuantMSE (Algorithm 1) and TurboQuantProd (Algorithm 2)
- triton_kernels: Fused Triton GPU kernels for decode attention
"""

from turboquant.core.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.core.quantizer import MSEQuantized, ProdQuantized

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "MSEQuantized",
    "ProdQuantized",
]
