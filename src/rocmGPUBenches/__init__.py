"""
rocmGPUBenches: GPU Benchmarking Framework for AMD ROCm
"""

import _rocmGPUBenches

# Core benchmark framework
BenchmarkRunner = _rocmGPUBenches.BenchmarkRunner
BenchmarkResult = _rocmGPUBenches.BenchmarkResult

# Benchmark factory functions
create_cache_benchmark_runner = _rocmGPUBenches.create_cache_benchmark_runner
create_latency_benchmark_runner = _rocmGPUBenches.create_latency_benchmark_runner
create_stream_benchmark_runner = _rocmGPUBenches.create_stream_benchmark_runner

# HipRTC compiler utilities
HipRTCCompiler = _rocmGPUBenches.HipRTCCompiler

# Storage module
from .storage.benchmark_db import BenchmarkDB

# Visualization module
from .visualization import (
    plot_sweep,
    plot_comparison,
    plot_heatmap,
    plot_gpu_comparison_sweep,
    format_data_size_axis
)

__all__ = [
    # Core framework
    'BenchmarkRunner',
    'BenchmarkResult',
    'HipRTCCompiler',
    
    # Benchmark factories
    'create_cache_benchmark_runner',
    'create_latency_benchmark_runner',
    'create_stream_benchmark_runner',
    
    # Storage
    'BenchmarkDB',
    
    # Visualization
    'plot_sweep',
    'plot_comparison',
    'plot_heatmap',
    'plot_gpu_comparison_sweep',
    'format_data_size_axis',
]

__version__ = '0.0.1'
