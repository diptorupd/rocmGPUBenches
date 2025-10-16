"""ROCm GPU Benchmarks with hipRTC support."""

import warnings

# Import C++ extension (compiled module in site-packages)
try:
    import _rocmGPUBenches
    hello = _rocmGPUBenches.hello if hasattr(_rocmGPUBenches, 'hello') else None
    HipRTCCompiler = _rocmGPUBenches.HipRTCCompiler
    GPUCacheBenchmark = _rocmGPUBenches.GPUCacheBenchmark if hasattr(_rocmGPUBenches, 'GPUCacheBenchmark') else None
    BenchmarkRunner = _rocmGPUBenches.BenchmarkRunner
    BenchmarkResult = _rocmGPUBenches.BenchmarkResult
    create_cache_benchmark_runner = _rocmGPUBenches.create_cache_benchmark_runner
    _cpp_extensions_available = True
except ImportError as e:
    warnings.warn(f"C++ extension not available: {e}")
    _cpp_extensions_available = False

# Python modules (always available)
from .storage import BenchmarkDB

# Build __all__ dynamically
__all__ = ['BenchmarkDB']

if _cpp_extensions_available:
    __all__.extend([
        'hello',
        'HipRTCCompiler', 
        'GPUCacheBenchmark',
        'BenchmarkRunner',
        'BenchmarkResult',
        'create_cache_benchmark_runner'
    ])

__version__ = "0.0.1"
