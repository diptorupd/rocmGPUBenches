"""ROCm GPU Benchmarks with hipRTC support."""

import warnings

try:
    from .rocmGPUBenches import hello, HipRTCCompiler, GPUCacheBenchmark
    __all__ = ['hello', 'HipRTCCompiler', 'GPUCacheBenchmark']
except ImportError as e:
    warnings.warn(f"C++ extension not available: {e}")
    __all__ = []

__version__ = "0.0.1"
