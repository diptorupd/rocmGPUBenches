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

### System Requirements

**Hardware**:
- AMD GPU with ROCm support (tested on MI325X, MI300X, MI210)
- Recommended: 16GB+ system RAM for compilation

**Software**:
- ROCm 6.0+ (tested with ROCm 6.4.1)
- Python 3.10+ (tested with Python 3.12)
- C++ compiler with C++17 support (g++ 9.0+)
- CMake 3.18+

### ROCm Installation

If ROCm is not installed, follow AMD's official guide:
```bash
# Ubuntu/Debian
# See: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
```

### Python Dependencies

The package requires the following Python packages (automatically installed):

**Build Dependencies** (required for installation):
- `scikit-build-core[pyproject]` - Build system
- `pybind11` - Python/C++ bindings

**Runtime Dependencies** (required to run benchmarks):
- `pandas >= 2.0.0` - Data manipulation and storage
- `matplotlib >= 3.7.0` - Visualization
- `numpy >= 1.24.0` - Numerical operations

**Optional Dependencies** (recommended):
- `jupyter` - For interactive notebooks
- `seaborn` - Enhanced plotting styles
- `ipywidgets` - Interactive notebook widgets

### Installation Methods

#### Method 1: Install from Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/diptorupd/rocmGPUBenches.git
cd rocmGPUBenches

# Create micromamba environment (recommended)
micromamba create -n rocm-bench python=3.12
micromamba activate rocm-bench

# Install dependencies
micromamba install pandas matplotlib numpy
# Or using pip:
# pip install pandas matplotlib numpy

# Install package in editable mode
pip install -e . --no-build-isolation

# Verify installation
python -c "from rocmGPUBenches import create_cache_benchmark_runner; print('✓ Installation successful')"
```

**Note**: The `--no-build-isolation` flag is recommended to use your environment's dependencies instead of building in an isolated environment.

#### Method 2: Install with All Optional Dependencies

```bash
# Install with Jupyter and visualization tools
pip install -e ".[dev]"

# This includes: jupyter, ipywidgets, seaborn, pytest
```

#### Method 3: Quick Install (Runtime Only)

```bash
# Minimal installation without dev tools
pip install pandas matplotlib numpy
pip install -e . --no-build-isolation
```

### Troubleshooting Installation

**Issue**: `hipRTC.h not found`
```bash
# Ensure ROCm is in your path
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

**Issue**: `ImportError: cannot import name 'create_cache_benchmark_runner'`
```bash
# Rebuild the C++ extension
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
```bash
# Install missing runtime dependencies
pip install pandas matplotlib numpy
```

**Issue**: Build fails with "hip/hip_runtime.h: No such file or directory"
```bash
# Install ROCm development packages
sudo apt-get install rocm-dev rocm-libs
```

### Verifying Installation

Run the following to verify everything works:

```python
from rocmGPUBenches import create_cache_benchmark_runner

# Create runner
runner = create_cache_benchmark_runner()
print(f"GPU: {runner.get_device_name()}")

# Run single benchmark
result = runner.run('cache', problem_size=256)
print(f"Bandwidth: {result.primary_metric:.2f} GB/s")

# Test storage
from rocmGPUBenches import BenchmarkDB
db = BenchmarkDB(':memory:')  # In-memory database for testing
db.save_result('cache', result, {'problem_size': 256},
               {'name': runner.get_device_name(), 'arch': 'gfx942'})
print(f"✓ Storage working")

# Test visualization
from rocmGPUBenches import plot_sweep
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
df = db.query(benchmark='cache')
plot_sweep(df, x='problem_size', y='primary_metric', show=False)
print(f"✓ Visualization working")

print("\n✅ All systems operational!")
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
| cache     | ✅     | L1/L2/L3 cache bandwidth characterization |
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
   ├── benchmarks/         # Benchmark configurations
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
