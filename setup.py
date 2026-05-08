"""
TurboQuant v2.0 — KV Cache Compression for LLM Inference.

Refactored with request isolation for multi-tenant concurrent inference.
"""

from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="2.0.0",
    description="KV Cache Compression for LLM Inference with Request Isolation",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "turboquant": ["core/codebooks/*.json"],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "numpy",
        "scipy",
    ],
    extras_require={
        "vllm": [
            "vllm>=0.18.0",
            "triton>=3.0",
        ],
        "monitor": [
            "pynvml",
        ],
        "dev": [
            "pytest",
            "black",
        ],
    },
)
