# ROCm GPU Benchmarks

A modern Python package for ROCm GPU microbenchmarking with dynamic kernel compilation, persistent storage, and interactive visualization.

Based on the [gpu-benches](https://github.com/te42kyfo/gpu-benches) project, focusing exclusively on AMD ROCm GPUs.

## Features

- **Dynamic Compilation**: hipRTC-based runtime kernel compilation
- **Python API**: Easy-to-use Python interface with type hints
- **Generic Framework**: Configuration-based BenchmarkRunner for pluggable benchmarks
- **Persistent Storage**: SQLite + pandas for benchmark result management
- **Visualization**: matplotlib-based plotting with KB/MB formatting and multi-GPU comparison
- **Architecture-Aware**: Automatic GPU detection and optimization

## Installation

### Prerequisites
- ROCm 6.0+ (tested with ROCm 6.4.1)
- Python 3.10+
- AMD GPU (tested on MI325X, MI300X)

### Install from source
```bash
git clone https://github.com/diptorupd/rocmGPUBenches.git
cd rocmGPUBenches
pip install -e .
```

## Quick Start

### Running a Single Benchmark
```python
from rocmGPUBenches import create_cache_benchmark_runner

runner = create_cache_benchmark_runner()
result = runner.run('cache', problem_size=256)

print(f"{result.primary_metric:.2f} {result.metric_name}")
# Output: 21695.15 bandwidth_gbs
```

### Complete Workflow: Run → Store → Visualize
```python
from rocmGPUBenches import (
    BenchmarkDB, 
    create_cache_benchmark_runner,
    plot_gpu_comparison_sweep
)

# Setup
db = BenchmarkDB('results.db')
runner = create_cache_benchmark_runner()
gpu_info = {'name': runner.get_device_name(), 'arch': 'gfx942'}

# Run parameter sweep
for size in [128, 256, 512, 1024, 2048]:
    result = runner.run('cache', problem_size=size)
    db.save_result('cache', result, 
                   params={'problem_size': size, 'block_size': 256},
                   gpu_info=gpu_info)
    print(f"size={size:4d}: {result.primary_metric:8.2f} GB/s")

# Query results
df = db.query(benchmark='cache')
df_sweep = db.get_sweep_data('cache', 'problem_size')

# Visualize with KB/MB formatting
plot_gpu_comparison_sweep(df, xscale='log2', 
                         title='Cache Hierarchy Analysis')
```

### Multi-GPU Comparison
```python
# Run on GPU 1
runner1 = create_cache_benchmark_runner()
for size in sizes:
    result = runner1.run('cache', problem_size=size)
    db.save_result('cache', result, {'problem_size': size}, 
                   {'name': 'MI325X', 'arch': 'gfx942'})

# Run on GPU 2 (different system)
runner2 = create_cache_benchmark_runner()
for size in sizes:
    result = runner2.run('cache', problem_size=size)
    db.save_result('cache', result, {'problem_size': size}, 
                   {'name': 'MI300X', 'arch': 'gfx940'})

# Compare both GPUs
df_all = db.query(benchmark='cache')
plot_gpu_comparison_sweep(df_all, title='MI325X vs MI300X')
```

## Available Benchmarks

| Benchmark | Status | Description |
|-----------|--------|-------------|
| cache     | L1/L2/L3 cache bandwidth characterization |     | 
| latency   | ⏳     | Memory latency profiling |
| stream    | ⏳     | Memory bandwidth (STREAM benchmark) |
| roofline  | ⏳     | Roofline model data collection |
| *more*    | ⏳     | 7 more benchmarks planned |

## Documentation

- [Architecture & Adding Benchmarks](docs/architecture.md)
- [Project Plan](PROJECT-PLAN.md) - Detailed development roadmap
- Jupyter Notebooks: `examples/` directory (coming soon)

## Project Structure

```
rocmGPUBenches/
 src/rocmGPUBenches/
   ├── framework/          # BenchmarkRunner infrastructure
 benchmarks/         # Benchmark configurations   ├─
   ├── kernels/            # HIP kernel implementations
   ├── hiprtc_utils/       # Runtime compilation
   ├── storage/            # Database persistence (pandas + SQLite)
   ├── visualization/      # Plotting functions
   └── utils/              # Measurement utilities
 tests/                  # Test suite
 examples/               # Jupyter notebooks
 docs/                   # Documentation
```

## License

GPLv3 - Matching the original [gpu-benches](https://github.com/te42kyfo/gpu-benches) project.

## Citation

If you use this tool in your research, please cite the original gpu-benches project:
```
@misc{gpu-benches,
  author = {Huthmann, Jens},
  title = {GPU Microbenchmarks},
  url = {https://github.com/te42kyfo/gpu-benches}
}
```

## Contributing

See [docs/architecture.md](docs/architecture.md) for information on adding new benchmarks.

## Development Status

**Current**: Phase 7 complete (Storage + Visualization) ✅  
**Next**: Jupyter notebooks, additional benchmarks  
**Progress**: 6/9 success criteria (67%), 1/11 benchmarks (9%)

See [PROJECT-PLAN.md](PROJECT-PLAN.md) for detailed roadmap.
