"""ROCm GPU Benchmarks - A tool for running ROCm GPU microbenchmarks."""

__version__ = "0.0.1"

# Import the compiled extension module
try:
    from .rocmGPUBenches import hello, HipRTCCompiler
    __all__ = ['hello', 'HipRTCCompiler']
except ImportError as e:
    # Module not yet built
    import warnings
    warnings.warn(f"C++ extension not available: {e}")
