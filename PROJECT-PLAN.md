# rocmGPUBenches Project Plan

## Project Overview

**rocmGPUBenches** is a modern Python package that consolidates GPU microbenchmarks from the original gpu-benches repository into a unified tool with dynamic kernel compilation and interactive visualization capabilities.

### Key Features
- **Dynamic Compilation**: Use hipRTC to compile HIP kernels on-the-fly
- **Python API**: Easy-to-use Python interface for running benchmarks
- **Generic Framework**: Configuration-based BenchmarkRunner for pluggable benchmarks
- **Storage & Visualization**: SQLite database + matplotlib plotting suite
- **Interactive Notebooks**: Jupyter integration for exploration and analysis
- **Architecture-Aware**: Automatically detects and optimizes for GPU architecture

### Original Source
Based on the [gpu-benches](https://github.com/te42kyfo/gpu-benches) repository, which contains microbenchmarks for NVIDIA and AMD GPUs. We are focusing exclusively on ROCm/AMD GPUs.

## Current Status

### ✅ COMPLETED (Merged to main - Oct 16, 2025)

#### Phase 1-4: Foundation & First Benchmark
- [x] Created project structure with `scikit-build-core`
- [x] Set up Python package layout
- [x] Configured CMake build system (C++ only, no device code at build time)
- [x] Added GPLv3 license (matching original project)
- [x] Initialized git repository at `github.com/diptorupd/rocmGPUBenches`
- [x] Hipified all 11 CUDA microbenchmarks to HIP format
- [x] Created `HipRTCCompiler` C++ class wrapper
- [x] Implemented pybind11 bindings for Python access
- [x] Added GPU architecture auto-detection
- [x] **Architecture Decision**: Runtime compilation only - no device code at build time
- [x] Created first working benchmark (GPU Cache): 414 TB/s bandwidth!
- [x] Copied shared utilities from gpu-benches (MeasurementSeries, dtime, gpu-error, rocm-metrics)

#### Phase 5: Generic Benchmark Framework ✅
- [x] **TODO 1**: Implemented generic BenchmarkRunner infrastructure
  - Configuration-based design with BenchmarkConfig struct
  - Flexible ParamMap using `std::variant<int, double, string>`
  - `run()` and `sweep()` methods with parameter validation
  - Memory allocation and kernel execution pipeline
  - Python bindings with kwargs support
- [x] **TODO 2**: Optimized kernel compilation strategy
  - Runtime parameters instead of compile-time templates
  - Single kernel compilation for all parameter combinations
  - Added hipRTC optimization flags: -O3, -ffast-math, --gpu-max-threads-per-block=1024
  - Result: 0.27s compilation, ~1.00x performance (already optimal)
- [x] Repository organization and cleanup
  - Moved HIP RTC files to `hiprtc_utils/` subdirectory
  - Renamed `benchmarks/` → `framework/` (infrastructure)
  - Created new `benchmarks/` for actual benchmark configs
  - Deleted deprecated cache implementation (527 lines)
  - Clear separation: framework/ (HOW), benchmarks/ (WHAT), kernels/ (CODE)

**Current Structure**:
```
src/rocmGPUBenches/
  framework/          # BenchmarkRunner infrastructure
    ├── benchmark_runner.hpp
    ├── benchmark_runner.cpp
    └── benchmark_runner_bindings.cpp
  benchmarks/         # Benchmark configurations
    └── cache_benchmark_config.hpp
  kernels/            # Kernel implementations
    ├── cache_kernels.hpp (active)
 *.hip (11 reference files to be converted)    └─
  hiprtc_utils/       # HIP runtime compilation
    ├── hip_rtc_compiler.hpp
    └── hip_rtc_bindings.cpp
  utils/              # Measurement utilities
    ├── MeasurementSeries.hpp
    ├── dtime.hpp
 gpu-error.h    ├
    └── rocm-metrics.hpp
  storage/            # Database persistence
    ├── __init__.py
    └── benchmark_db.py (300+ lines)
  visualization/      # Plotting functions
    ├── __init__.py
    └── plotter.py (450+ lines)
```

**Test Environment**:
- **GPU**: AMD Instinct MI325X (gfx942:sramecc+:xnack-)
- **Compute Units**: 304
- **ROCm**: 6.4.1
- **Python**: 3.12 in conda environment `rocm-gpubench-env`
- **Build**: scikit-build-core + CMake, g++ for host, hipRTC for kernels


#### Phase 6: Storage Module ✅ (Oct 16, 2025)
- [x] **Implemented pandas + SQLite storage backend**
  - Created `BenchmarkDB` class for persistent result storage
  - Schema: 15 columns including timestamp, gpu_info, parameters (JSON), metrics
  - Methods: `save_result()`, `query()`, `get_sweep_data()`, `export_csv/json()`, `stats()`
  - Flexible querying with filters (benchmark, gpu_name, problem_size, etc.)
  - Aggregation support for parameter sweeps (mean, std, count)
  - **Architecture Decision**: Pandas + SQLite (lightweight) vs SQLAlchemy+Alembic (enterprise)
  - Result: 0.5-1 session effort, perfect for single-user/small-team workflow
  - **Status**: MERGED to main (PR #4)

#### Phase 7: Visualization Module ✅ (Oct 16, 2025)
- [x] **Implemented comprehensive visualization suite**
  - **Core Functions**:
    * `plot_sweep()` - Parameter sweep line plots with error bars, grouping, log scales
    * `plot_comparison()` - Bar charts for metric comparison across configurations
    * `plot_heatmap()` - 2D heatmap for dual-parameter sweeps
  - **KB/MB Formatting**:
    * `format_data_size_axis()` - Smart formatter: <1024→KB, ≥1024→MB
    * Matches gpu-benches style: "data volume per SM/CU"
    * Supports custom element sizes (float32=4B, float64=8B)
  - **Multi-GPU Comparison**:
    * `plot_gpu_comparison_sweep()` - Convenience function for cross-GPU plots
    * Database schema supports multi-GPU (gpu_name, gpu_arch columns)
    * `plot_sweep(group_by='gpu_name')` for custom grouping
  - **Styling**:
    * Ported GPU color palette from gpu-benches (9 colors)
    * BMH matplotlib style, export: PNG/SVG/PDF (300 DPI)
  - **Status**: MERGED to main (PR #5)

#### Phase 8: Jupyter Notebook Demo ✅ (Oct 16, 2025)
- [x] **Created comprehensive demo notebook**
  - **Location**: `demos/demo_cache_benchmark.ipynb`
  - **Content**: 10 sections covering complete workflow
  - **Features**:
    * Imports and setup with GPU detection
    * Single benchmark run with detailed output
    * Parameter sweep (8 problem sizes)
    * Database query and aggregation examples
    * Sweep plot with KB/MB formatting + bytes on secondary axis
    * Bar chart comparison (ordered by problem size with bytes)
    * Cache hierarchy analysis: L1/L2/L3/Infinity Cache/HBM identification
    * Performance cliff detection (⚠️ marker for >10% drops)
    * Export to CSV/JSON with use case explanations
    * Multi-GPU comparison support (placeholder)
  - **Educational Content**:
    * Hardcoded benchmark description (TODO: move to framework)
    * Cache architecture explanations
    * Interpretation guide for results
    * Tips for cross-GPU comparison
  - **Status**: MERGED to main (PR #9)

**Working Example**:
```python
# Complete workflow: benchmark → storage → visualization
from rocmGPUBenches import (
    BenchmarkDB, 
    create_cache_benchmark_runner,
    plot_sweep,
    format_data_size_axis
)

# Run and store
db = BenchmarkDB('results.db')
runner = create_cache_benchmark_runner()
for size in [128, 256, 512, 1024]:
    result = runner.run('cache', problem_size=size)
    db.save_result('cache', result, {'problem_size': size}, 
                   {'name': runner.get_device_name(), 'arch': 'gfx942'})

# Query and visualize
df_sweep = db.get_sweep_data('cache', 'problem_size')
fig = plot_sweep(df_sweep, x='problem_size', y='bandwidth_gbs_mean', xscale='log2')
format_data_size_axis(fig.axes[0], base_unit_bytes=4)  # KB/MB formatting
```

### 🔄 Current Focus: Adding More Benchmarks

**Phase 6-8 COMPLETE!** ✅ Storage + Visualization + Jupyter Demo all merged to main!

**Foundation Complete**: We now have a solid base with:
- ✅ Generic BenchmarkRunner framework
- ✅ Storage (BenchmarkDB with SQLite)
- ✅ Visualization (5 plotting functions)
- ✅ Demo notebook showing complete workflow
- ✅ 1 working benchmark (GPU Cache)

**Next Priorities**:
1. **TODO 5**: Add GPU Latency benchmark (validate framework with different metric type)
2. **TODO 6**: Add GPU Stream benchmark (validate with multiple kernels)
3. Clean up minor issues (debug prints, benchmark descriptions)
4. **TODO 7**: Add testing infrastructure once we have 2-3 benchmarks

---

## 📋 Active TODOs

### TODO 5: Add GPU Latency Benchmark (NEXT UP)
**Priority**: HIGH  
**Effort**: 1-1.5 sessions  
**Status**: 🔄 In Progress

**Goal**: Validate framework with a different metric type (latency vs bandwidth)

**Why GPU Latency?**
- Simplest benchmark after cache
- Different metric type: latency (ns) instead of bandwidth (GB/s)
- Single kernel with pointer chasing
- Tests framework's flexibility with different result types
- Useful for understanding memory subsystem characteristics

**Implementation Plan**:
1. **Study original**: `/devel/gpu-benches/gpu-latency/`
2. **Create kernel**: `src/rocmGPUBenches/kernels/latency_kernels.hpp`
   - Port pointer-chasing kernel from CUDA to HIP
   - Configurable: problem_size (array size for pointer chain)
3. **Create config**: `src/rocmGPUBenches/benchmarks/latency_benchmark_config.hpp`
   - BenchmarkConfig with name="latency"
   - metric_name="latency_ns" (different from bandwidth_gbs)
   - Default params: problem_size, iterations
4. **Add binding**: `framework/benchmark_runner_bindings.cpp`
   - `create_latency_benchmark_runner()` function
5. **Python export**: Update `src/rocmGPUBenches/__init__.py`
6. **Test**: Run single + sweep, verify results make sense

**Deliverable**: Working latency benchmark demonstrating framework's metric flexibility

**Expected Validation**:
- Framework handles different metric names correctly
- BenchmarkDB stores latency results properly
- Visualization works with latency data (different units/scales)

---

### TODO 6: Add GPU Stream Benchmark
**Priority**: MEDIUM  
**Effort**: 1.5 sessions  
**Status**: Not started

**Goal**: Validate framework with multiple kernel variants (copy, scale, add, triad)

**Why GPU Stream?**
- Classic memory bandwidth benchmark (ported from STREAM)
- Tests framework with multiple related kernels
- Different access patterns: copy (1 read + 1 write), scale (1 read + 1 write with multiply), add (2 reads + 1 write), triad (2 reads + 1 write with multiply)
- Widely recognized benchmark for comparison
- Metric: sustained bandwidth (GB/s) for different operations

**Implementation Plan**:
1. **Study original**: `/devel/gpu-benches/gpu-stream/`
2. **Create kernels**: `src/rocmGPUBenches/kernels/stream_kernels.hpp`
   - Four kernel variants: copy, scale, add, triad
   - Configurable array size
3. **Create config**: `src/rocmGPUBenches/benchmarks/stream_benchmark_config.hpp`
   - BenchmarkConfig for each variant or unified config
   - Parameters: array_size, operation_type
4. **Add binding**: Create runner factory function
5. **Test**: Verify all 4 operations produce reasonable bandwidth numbers

**Deliverable**: Working stream benchmark with 4 operation types

---

### TODO 7: Testing Infrastructure
**Priority**: MEDIUM  
**Effort**: 2 sessions  
**Status**: Not started

- [ ] Add pytest framework
- [ ] Unit tests for BenchmarkRunner
- [ ] Unit tests for HipRTCCompiler
- [ ] Unit tests for BenchmarkDB (storage)
- [ ] Unit tests for visualization functions
- [ ] Integration tests for each benchmark
- [ ] CI/CD with GitHub Actions
- [ ] Test on multiple ROCm versions
- [ ] Performance regression tests

**Deliverable**: Automated testing with >80% coverage

---

### TODO 8: Complete Remaining Benchmarks
**Priority**: LOW  
**Effort**: 4-6 sessions  
**Status**: Not started (1/11 complete)

**Remaining Benchmarks**:
- [x] gpu-cache (DONE)
- [ ] gpu-latency (in progress - TODO 5)
- [ ] gpu-stream (planned - TODO 6)
- [ ] gpu-l2-cache (medium)
- [ ] gpu-l2-stream (medium)
- [ ] gpu-roofline (complex)
- [ ] gpu-small-kernels (medium)
- [ ] gpu-strides (medium)
- [ ] cuda-incore (simple)
- [ ] cuda-memcpy (simple)
- [ ] um-stream (medium)

**Strategy**: Follow established pattern from TODO 5 & 6

---

## 📋 Future TODOs

### TODO 9: Minor Framework Improvements
**Priority**: LOW  
**Effort**: 0.5 sessions  
**Status**: Not started

- [ ] Remove "Registered benchmark" debug prints from `benchmark_runner.cpp`
- [ ] Add `description` field to `BenchmarkConfig`
- [ ] Update cache benchmark to use description field
- [ ] Remove hardcoded description from Jupyter notebook

**Deliverable**: Cleaner framework with self-documenting benchmarks

---

### TODO 10: YAML Configuration (Optional)
**Priority**: LOW  
**Effort**: 2-3 sessions  
**Status**: Deferred

**Current State**: Benchmark configs are C++ code with lambdas - works well!

**YAML Proposal**: Would allow non-programmers to add benchmarks
```yaml
name: cache
kernel: kernels/cache_kernels.hpp
default_params:
  problem_size: 256
  block_size: 256
```

**Analysis**:
- **Pros**: Easier to modify, clean separation
- **Cons**: Loss of type safety, C++ lambdas are very powerful, harder to debug
- **Decision**: Wait until we have 5-10 benchmarks, then reassess if there's repetitive boilerplate

**Alternative**: Python helper to generate C++ configs from templates

---

### TODO 11: Documentation
**Priority**: MEDIUM  
**Effort**: 2-3 sessions  
**Status**: Partially complete

**Completed**:
- [x] README with usage examples and installation
- [x] docs/ARCHITECTURE.md with project structure
- [x] Jupyter notebook as living documentation

**TODO**:
- [ ] API documentation with Sphinx
- [ ] Tutorial notebooks (beyond demos)
- [ ] Performance analysis guide
- [ ] Contributing guidelines

---

### TODO 12: Packaging & Distribution
**Priority**: LOW  
**Effort**: 1-2 sessions  
**Status**: Not started

- [ ] Prepare for PyPI
- [ ] Add version management
- [ ] Create release workflow
- [ ] Docker container with ROCm

---

## 🎯 Recommended Roadmap

### Near-Term (Next 3-4 sessions)
1. **Session 1**: GPU Latency (TODO 5) - Validate framework with different metric
2. **Session 2**: GPU Stream (TODO 6) - Validate multi-kernel configs
3. **Session 3**: Clean up framework (TODO 9) - Remove debug prints, add descriptions
4. **Session 4**: Testing (TODO 7) - Basic test coverage for storage/visualization

**Goal**: 3 working benchmarks with tests, proving framework's generality

### Mid-Term (5-8 sessions)
- Add 3-4 more benchmarks (TODO 8)
- Expand test coverage
- Enhanced documentation

### Long-Term (10+ sessions)
- Complete all 11 benchmarks (TODO 8)
- Advanced features (YAML configs if needed, multi-GPU, etc.)
- Distribution (TODO 12)

---

## 📊 Progress Summary

| Phase | Status | Sessions | Completed |
|-------|--------|----------|-----------|
| **Phase 1-4: Foundation** | ✅ Complete | ~8 | Oct 16, 2025 |
| **Phase 5: Generic Framework** | ✅ Complete | 4 | Oct 16, 2025 |
| **Phase 6: Storage Module** | ✅ Complete | 1 | Oct 16, 2025 |
| **Phase 7: Visualization** | ✅ Complete | 1.5 | Oct 16, 2025 |
| **Phase 8: Jupyter Demo** | ✅ Complete | 1 | Oct 16, 2025 |
| **TODO 5: GPU Latency** | 🔄 In Progress | 1-1.5 | - |
| **TODO 6: GPU Stream** | ⏳ Pending | 1.5 | - |
| **TODO 7: Testing** | ⏳ Pending | 2 | - |
| **TODO 8: More Benchmarks** | ⏳ Pending | 4-6 | 1/11 done |
| **TODO 9: Framework Polish** | ⏳ Pending | 0.5 | - |
| **TODO 10: YAML (optional)** | ⏸️ Deferred | 2-3 | - |
| **TODO 11: Docs** | 🔄 Partial | 1-2 | README done |
| **TODO 12: Packaging** | ⏳ Pending | 1-2 | - |

**Total Completed**: ~15.5 sessions  
**Estimated Remaining**: ~15-20 sessions  
**Current Benchmark Count**: 1/11 (9%) - Adding #2 now!

---

## 🎯 Immediate Next Steps

### Current Status: Foundation Complete! 🎉

**What's Done**:
- ✅ Generic BenchmarkRunner framework (Phase 5)
- ✅ Storage module with SQLite (Phase 6)
- ✅ Visualization suite with 5 plotting functions (Phase 7)
- ✅ Comprehensive Jupyter notebook demo (Phase 8)
- ✅ 1 working benchmark (GPU Cache) with KB/MB formatting

**What's Next**: Validate the framework by adding more benchmarks!

### Recommended Immediate Order:
1. 🔄 **GPU Latency Benchmark (TODO 5)** - IN PROGRESS
   - Different metric type (latency vs bandwidth)
   - Validates framework flexibility
   - Simplest next benchmark
   
2. ⬜ **GPU Stream Benchmark (TODO 6)** - 1.5 sessions
   - Multiple kernel variants (copy, scale, add, triad)
   - Classic bandwidth benchmark
   - Validates multi-kernel configs
   
3. ⬜ **Clean up minor issues (TODO 9)** - 0.5 sessions
   - Remove "Registered benchmark" debug prints
   - Add description field to BenchmarkConfig
   - Update cache benchmark to use description
   
4. ⬜ **Testing Infrastructure (TODO 7)** - 2 sessions
   - Add pytest framework
   - Unit tests for storage, visualization
   - Integration tests for benchmarks
   - Useful once we have 2-3 benchmarks

**Rationale**:
- Latency validates that framework handles different metric types
- Stream validates multi-kernel configurations
- Testing becomes valuable with multiple benchmarks to protect
- Clean foundation enables faster benchmark additions

---

## 💡 Key Insights & Decisions

### Architecture Decisions
1. **Runtime Compilation Only**: No HIP device code at build time, all via hipRTC
2. **Configuration-Based Framework**: BenchmarkConfig with flexible ParamMap
3. **C++ for Performance**: Infrastructure in C++ with Python bindings via pybind11
4. **Separation of Concerns**: framework/ (HOW), benchmarks/ (WHAT), kernels/ (CODE)
5. **Pandas + SQLite**: Lightweight storage, perfect for single-user workflow
6. **matplotlib**: Standard Python plotting, familiar to scientific community

### What's Working Well
- BenchmarkRunner abstraction is clean and extensible
- Runtime parameters avoid template explosion
- Python API is intuitive with kwargs
- Single kernel compilation is fast (0.27s)
- BenchmarkDB makes data persistence trivial
- Visualization functions are flexible and reusable
- Jupyter notebook validates complete workflow

### Lessons Learned
- C++ lambdas in configs are more powerful than YAML would be
- Repository organization matters - clear naming improves intuition
- Generic framework was worth the investment - adding benchmarks is now straightforward
- Storage + visualization should have come earlier (but better late than never!)
- Jupyter notebook is excellent living documentation

### Known Issues
- "Registered benchmark" debug prints appear (minor annoyance)
- Benchmark descriptions hardcoded in notebook (should be in BenchmarkConfig)
- No automated testing yet
- Only 1 of 11 benchmarks implemented

---

## Success Criteria

The project will be considered successful when:

1. ✅ All CUDA benchmarks converted to HIP
2. ✅ hipRTC integration working
3. ✅ Can run at least 1 benchmark end-to-end from Python
4. ✅ Generic framework supports pluggable benchmarks
5. ✅ Visualizations match quality of original gpu-benches
6. ✅ Example Jupyter notebooks demonstrate key features
7. ⬜ At least 5-6 benchmarks working (50%+)
8. ⬜ Documentation covers installation and usage
9. ⬜ CI/CD pipeline ensures builds work

cat > PROJECT-PLAN.md << 'EOF'
# rocmGPUBenches Project Plan

## Project Overview

**rocmGPUBenches** is a modern Python package that consolidates GPU microbenchmarks from the original gpu-benches repository into a unified tool with dynamic kernel compilation and interactive visualization capabilities.

### Key Features
- **Dynamic Compilation**: Use hipRTC to compile HIP kernels on-the-fly
- **Python API**: Easy-to-use Python interface for running benchmarks
- **Generic Framework**: Configuration-based BenchmarkRunner for pluggable benchmarks
- **Storage & Visualization**: SQLite database + matplotlib plotting suite
- **Interactive Notebooks**: Jupyter integration for exploration and analysis
- **Architecture-Aware**: Automatically detects and optimizes for GPU architecture

### Original Source
Based on the [gpu-benches](https://github.com/te42kyfo/gpu-benches) repository, which contains microbenchmarks for NVIDIA and AMD GPUs. We are focusing exclusively on ROCm/AMD GPUs.

## Current Status

### ✅ COMPLETED (Merged to main - Oct 16, 2025)

#### Phase 1-4: Foundation & First Benchmark
- [x] Created project structure with `scikit-build-core`
- [x] Set up Python package layout
- [x] Configured CMake build system (C++ only, no device code at build time)
- [x] Added GPLv3 license (matching original project)
- [x] Initialized git repository at `github.com/diptorupd/rocmGPUBenches`
- [x] Hipified all 11 CUDA microbenchmarks to HIP format
- [x] Created `HipRTCCompiler` C++ class wrapper
- [x] Implemented pybind11 bindings for Python access
- [x] Added GPU architecture auto-detection
- [x] **Architecture Decision**: Runtime compilation only - no device code at build time
- [x] Created first working benchmark (GPU Cache): 414 TB/s bandwidth!
- [x] Copied shared utilities from gpu-benches (MeasurementSeries, dtime, gpu-error, rocm-metrics)

#### Phase 5: Generic Benchmark Framework ✅
- [x] **TODO 1**: Implemented generic BenchmarkRunner infrastructure
  - Configuration-based design with BenchmarkConfig struct
  - Flexible ParamMap using `std::variant<int, double, string>`
  - `run()` and `sweep()` methods with parameter validation
  - Memory allocation and kernel execution pipeline
  - Python bindings with kwargs support
- [x] **TODO 2**: Optimized kernel compilation strategy
  - Runtime parameters instead of compile-time templates
  - Single kernel compilation for all parameter combinations
  - Added hipRTC optimization flags: -O3, -ffast-math, --gpu-max-threads-per-block=1024
  - Result: 0.27s compilation, ~1.00x performance (already optimal)
- [x] Repository organization and cleanup
  - Moved HIP RTC files to `hiprtc_utils/` subdirectory
  - Renamed `benchmarks/` → `framework/` (infrastructure)
  - Created new `benchmarks/` for actual benchmark configs
  - Deleted deprecated cache implementation (527 lines)
  - Clear separation: framework/ (HOW), benchmarks/ (WHAT), kernels/ (CODE)

**Current Structure**:
```
src/rocmGPUBenches/
  framework/          # BenchmarkRunner infrastructure
    ├── benchmark_runner.hpp
    ├── benchmark_runner.cpp
    └── benchmark_runner_bindings.cpp
  benchmarks/         # Benchmark configurations
    └── cache_benchmark_config.hpp
  kernels/            # Kernel implementations
    ├── cache_kernels.hpp (active)
 *.hip (11 reference files to be converted)    └─
  hiprtc_utils/       # HIP runtime compilation
    ├── hip_rtc_compiler.hpp
    └── hip_rtc_bindings.cpp
  utils/              # Measurement utilities
    ├── MeasurementSeries.hpp
    ├── dtime.hpp
 gpu-error.h    ├
    └── rocm-metrics.hpp
  storage/            # Database persistence
    ├── __init__.py
    └── benchmark_db.py (300+ lines)
  visualization/      # Plotting functions
    ├── __init__.py
    └── plotter.py (450+ lines)
```

**Test Environment**:
- **GPU**: AMD Instinct MI325X (gfx942:sramecc+:xnack-)
- **Compute Units**: 304
- **ROCm**: 6.4.1
- **Python**: 3.12 in conda environment `rocm-gpubench-env`
- **Build**: scikit-build-core + CMake, g++ for host, hipRTC for kernels


#### Phase 6: Storage Module ✅ (Oct 16, 2025)
- [x] **Implemented pandas + SQLite storage backend**
  - Created `BenchmarkDB` class for persistent result storage
  - Schema: 15 columns including timestamp, gpu_info, parameters (JSON), metrics
  - Methods: `save_result()`, `query()`, `get_sweep_data()`, `export_csv/json()`, `stats()`
  - Flexible querying with filters (benchmark, gpu_name, problem_size, etc.)
  - Aggregation support for parameter sweeps (mean, std, count)
  - **Architecture Decision**: Pandas + SQLite (lightweight) vs SQLAlchemy+Alembic (enterprise)
  - Result: 0.5-1 session effort, perfect for single-user/small-team workflow
  - **Status**: MERGED to main (PR #4)

#### Phase 7: Visualization Module ✅ (Oct 16, 2025)
- [x] **Implemented comprehensive visualization suite**
  - **Core Functions**:
    * `plot_sweep()` - Parameter sweep line plots with error bars, grouping, log scales
    * `plot_comparison()` - Bar charts for metric comparison across configurations
    * `plot_heatmap()` - 2D heatmap for dual-parameter sweeps
  - **KB/MB Formatting**:
    * `format_data_size_axis()` - Smart formatter: <1024→KB, ≥1024→MB
    * Matches gpu-benches style: "data volume per SM/CU"
    * Supports custom element sizes (float32=4B, float64=8B)
  - **Multi-GPU Comparison**:
    * `plot_gpu_comparison_sweep()` - Convenience function for cross-GPU plots
    * Database schema supports multi-GPU (gpu_name, gpu_arch columns)
    * `plot_sweep(group_by='gpu_name')` for custom grouping
  - **Styling**:
    * Ported GPU color palette from gpu-benches (9 colors)
    * BMH matplotlib style, export: PNG/SVG/PDF (300 DPI)
  - **Status**: MERGED to main (PR #5)

#### Phase 8: Jupyter Notebook Demo ✅ (Oct 16, 2025)
- [x] **Created comprehensive demo notebook**
  - **Location**: `demos/demo_cache_benchmark.ipynb`
  - **Content**: 10 sections covering complete workflow
  - **Features**:
    * Imports and setup with GPU detection
    * Single benchmark run with detailed output
    * Parameter sweep (8 problem sizes)
    * Database query and aggregation examples
    * Sweep plot with KB/MB formatting + bytes on secondary axis
    * Bar chart comparison (ordered by problem size with bytes)
    * Cache hierarchy analysis: L1/L2/L3/Infinity Cache/HBM identification
    * Performance cliff detection (⚠️ marker for >10% drops)
    * Export to CSV/JSON with use case explanations
    * Multi-GPU comparison support (placeholder)
  - **Educational Content**:
    * Hardcoded benchmark description (TODO: move to framework)
    * Cache architecture explanations
    * Interpretation guide for results
    * Tips for cross-GPU comparison
  - **Status**: MERGED to main (PR #9)

**Working Example**:
```python
# Complete workflow: benchmark → storage → visualization
from rocmGPUBenches import (
    BenchmarkDB, 
    create_cache_benchmark_runner,
    plot_sweep,
    format_data_size_axis
)

# Run and store
db = BenchmarkDB('results.db')
runner = create_cache_benchmark_runner()
for size in [128, 256, 512, 1024]:
    result = runner.run('cache', problem_size=size)
    db.save_result('cache', result, {'problem_size': size}, 
                   {'name': runner.get_device_name(), 'arch': 'gfx942'})

# Query and visualize
df_sweep = db.get_sweep_data('cache', 'problem_size')
fig = plot_sweep(df_sweep, x='problem_size', y='bandwidth_gbs_mean', xscale='log2')
format_data_size_axis(fig.axes[0], base_unit_bytes=4)  # KB/MB formatting
```

### 🔄 Current Focus: Adding More Benchmarks

**Phase 6-8 COMPLETE!** ✅ Storage + Visualization + Jupyter Demo all merged to main!

**Foundation Complete**: We now have a solid base with:
- ✅ Generic BenchmarkRunner framework
- ✅ Storage (BenchmarkDB with SQLite)
- ✅ Visualization (5 plotting functions)
- ✅ Demo notebook showing complete workflow
- ✅ 1 working benchmark (GPU Cache)

**Next Priorities**:
1. **TODO 5**: Add GPU Latency benchmark (validate framework with different metric type)
2. **TODO 6**: Add GPU Stream benchmark (validate with multiple kernels)
3. Clean up minor issues (debug prints, benchmark descriptions)
4. **TODO 7**: Add testing infrastructure once we have 2-3 benchmarks

---

## 📋 Active TODOs

### TODO 5: Add GPU Latency Benchmark (NEXT UP)
**Priority**: HIGH  
**Effort**: 1-1.5 sessions  
**Status**: 🔄 In Progress

**Goal**: Validate framework with a different metric type (latency vs bandwidth)

**Why GPU Latency?**
- Simplest benchmark after cache
- Different metric type: latency (ns) instead of bandwidth (GB/s)
- Single kernel with pointer chasing
- Tests framework's flexibility with different result types
- Useful for understanding memory subsystem characteristics

**Implementation Plan**:
1. **Study original**: `/devel/gpu-benches/gpu-latency/`
2. **Create kernel**: `src/rocmGPUBenches/kernels/latency_kernels.hpp`
   - Port pointer-chasing kernel from CUDA to HIP
   - Configurable: problem_size (array size for pointer chain)
3. **Create config**: `src/rocmGPUBenches/benchmarks/latency_benchmark_config.hpp`
   - BenchmarkConfig with name="latency"
   - metric_name="latency_ns" (different from bandwidth_gbs)
   - Default params: problem_size, iterations
4. **Add binding**: `framework/benchmark_runner_bindings.cpp`
   - `create_latency_benchmark_runner()` function
5. **Python export**: Update `src/rocmGPUBenches/__init__.py`
6. **Test**: Run single + sweep, verify results make sense

**Deliverable**: Working latency benchmark demonstrating framework's metric flexibility

**Expected Validation**:
- Framework handles different metric names correctly
- BenchmarkDB stores latency results properly
- Visualization works with latency data (different units/scales)

---

### TODO 6: Add GPU Stream Benchmark
**Priority**: MEDIUM  
**Effort**: 1.5 sessions  
**Status**: Not started

**Goal**: Validate framework with multiple kernel variants (copy, scale, add, triad)

**Why GPU Stream?**
- Classic memory bandwidth benchmark (ported from STREAM)
- Tests framework with multiple related kernels
- Different access patterns: copy (1 read + 1 write), scale (1 read + 1 write with multiply), add (2 reads + 1 write), triad (2 reads + 1 write with multiply)
- Widely recognized benchmark for comparison
- Metric: sustained bandwidth (GB/s) for different operations

**Implementation Plan**:
1. **Study original**: `/devel/gpu-benches/gpu-stream/`
2. **Create kernels**: `src/rocmGPUBenches/kernels/stream_kernels.hpp`
   - Four kernel variants: copy, scale, add, triad
   - Configurable array size
3. **Create config**: `src/rocmGPUBenches/benchmarks/stream_benchmark_config.hpp`
   - BenchmarkConfig for each variant or unified config
   - Parameters: array_size, operation_type
4. **Add binding**: Create runner factory function
5. **Test**: Verify all 4 operations produce reasonable bandwidth numbers

**Deliverable**: Working stream benchmark with 4 operation types

---

### TODO 7: Testing Infrastructure
**Priority**: MEDIUM  
**Effort**: 2 sessions  
**Status**: Not started

- [ ] Add pytest framework
- [ ] Unit tests for BenchmarkRunner
- [ ] Unit tests for HipRTCCompiler
- [ ] Unit tests for BenchmarkDB (storage)
- [ ] Unit tests for visualization functions
- [ ] Integration tests for each benchmark
- [ ] CI/CD with GitHub Actions
- [ ] Test on multiple ROCm versions
- [ ] Performance regression tests

**Deliverable**: Automated testing with >80% coverage

---

### TODO 8: Complete Remaining Benchmarks
**Priority**: LOW  
**Effort**: 4-6 sessions  
**Status**: Not started (1/11 complete)

**Remaining Benchmarks**:
- [x] gpu-cache (DONE)
- [ ] gpu-latency (in progress - TODO 5)
- [ ] gpu-stream (planned - TODO 6)
- [ ] gpu-l2-cache (medium)
- [ ] gpu-l2-stream (medium)
- [ ] gpu-roofline (complex)
- [ ] gpu-small-kernels (medium)
- [ ] gpu-strides (medium)
- [ ] cuda-incore (simple)
- [ ] cuda-memcpy (simple)
- [ ] um-stream (medium)

**Strategy**: Follow established pattern from TODO 5 & 6

---

## 📋 Future TODOs

### TODO 9: Minor Framework Improvements
**Priority**: LOW  
**Effort**: 0.5 sessions  
**Status**: Not started

- [ ] Remove "Registered benchmark" debug prints from `benchmark_runner.cpp`
- [ ] Add `description` field to `BenchmarkConfig`
- [ ] Update cache benchmark to use description field
- [ ] Remove hardcoded description from Jupyter notebook

**Deliverable**: Cleaner framework with self-documenting benchmarks

---

### TODO 10: YAML Configuration (Optional)
**Priority**: LOW  
**Effort**: 2-3 sessions  
**Status**: Deferred

**Current State**: Benchmark configs are C++ code with lambdas - works well!

**YAML Proposal**: Would allow non-programmers to add benchmarks
```yaml
name: cache
kernel: kernels/cache_kernels.hpp
default_params:
  problem_size: 256
  block_size: 256
```

**Analysis**:
- **Pros**: Easier to modify, clean separation
- **Cons**: Loss of type safety, C++ lambdas are very powerful, harder to debug
- **Decision**: Wait until we have 5-10 benchmarks, then reassess if there's repetitive boilerplate

**Alternative**: Python helper to generate C++ configs from templates

---

### TODO 11: Documentation
**Priority**: MEDIUM  
**Effort**: 2-3 sessions  
**Status**: Partially complete

**Completed**:
- [x] README with usage examples and installation
- [x] docs/ARCHITECTURE.md with project structure
- [x] Jupyter notebook as living documentation

**TODO**:
- [ ] API documentation with Sphinx
- [ ] Tutorial notebooks (beyond demos)
- [ ] Performance analysis guide
- [ ] Contributing guidelines

---

### TODO 12: Packaging & Distribution
**Priority**: LOW  
**Effort**: 1-2 sessions  
**Status**: Not started

- [ ] Prepare for PyPI
- [ ] Add version management
- [ ] Create release workflow
- [ ] Docker container with ROCm

---

## 🎯 Recommended Roadmap

### Near-Term (Next 3-4 sessions)
1. **Session 1**: GPU Latency (TODO 5) - Validate framework with different metric
2. **Session 2**: GPU Stream (TODO 6) - Validate multi-kernel configs
3. **Session 3**: Clean up framework (TODO 9) - Remove debug prints, add descriptions
4. **Session 4**: Testing (TODO 7) - Basic test coverage for storage/visualization

**Goal**: 3 working benchmarks with tests, proving framework's generality

### Mid-Term (5-8 sessions)
- Add 3-4 more benchmarks (TODO 8)
- Expand test coverage
- Enhanced documentation

### Long-Term (10+ sessions)
- Complete all 11 benchmarks (TODO 8)
- Advanced features (YAML configs if needed, multi-GPU, etc.)
- Distribution (TODO 12)

---

## 📊 Progress Summary

| Phase | Status | Sessions | Completed |
|-------|--------|----------|-----------|
| **Phase 1-4: Foundation** | ✅ Complete | ~8 | Oct 16, 2025 |
| **Phase 5: Generic Framework** | ✅ Complete | 4 | Oct 16, 2025 |
| **Phase 6: Storage Module** | ✅ Complete | 1 | Oct 16, 2025 |
| **Phase 7: Visualization** | ✅ Complete | 1.5 | Oct 16, 2025 |
| **Phase 8: Jupyter Demo** | ✅ Complete | 1 | Oct 16, 2025 |
| **TODO 5: GPU Latency** | 🔄 In Progress | 1-1.5 | - |
| **TODO 6: GPU Stream** | ⏳ Pending | 1.5 | - |
| **TODO 7: Testing** | ⏳ Pending | 2 | - |
| **TODO 8: More Benchmarks** | ⏳ Pending | 4-6 | 1/11 done |
| **TODO 9: Framework Polish** | ⏳ Pending | 0.5 | - |
| **TODO 10: YAML (optional)** | ⏸️ Deferred | 2-3 | - |
| **TODO 11: Docs** | 🔄 Partial | 1-2 | README done |
| **TODO 12: Packaging** | ⏳ Pending | 1-2 | - |

**Total Completed**: ~15.5 sessions  
**Estimated Remaining**: ~15-20 sessions  
**Current Benchmark Count**: 1/11 (9%) - Adding #2 now!

---

## 🎯 Immediate Next Steps

### Current Status: Foundation Complete! 🎉

**What's Done**:
- ✅ Generic BenchmarkRunner framework (Phase 5)
- ✅ Storage module with SQLite (Phase 6)
- ✅ Visualization suite with 5 plotting functions (Phase 7)
- ✅ Comprehensive Jupyter notebook demo (Phase 8)
- ✅ 1 working benchmark (GPU Cache) with KB/MB formatting

**What's Next**: Validate the framework by adding more benchmarks!

### Recommended Immediate Order:
1. 🔄 **GPU Latency Benchmark (TODO 5)** - IN PROGRESS
   - Different metric type (latency vs bandwidth)
   - Validates framework flexibility
   - Simplest next benchmark
   
2. ⬜ **GPU Stream Benchmark (TODO 6)** - 1.5 sessions
   - Multiple kernel variants (copy, scale, add, triad)
   - Classic bandwidth benchmark
   - Validates multi-kernel configs
   
3. ⬜ **Clean up minor issues (TODO 9)** - 0.5 sessions
   - Remove "Registered benchmark" debug prints
   - Add description field to BenchmarkConfig
   - Update cache benchmark to use description
   
4. ⬜ **Testing Infrastructure (TODO 7)** - 2 sessions
   - Add pytest framework
   - Unit tests for storage, visualization
   - Integration tests for benchmarks
   - Useful once we have 2-3 benchmarks

**Rationale**:
- Latency validates that framework handles different metric types
- Stream validates multi-kernel configurations
- Testing becomes valuable with multiple benchmarks to protect
- Clean foundation enables faster benchmark additions

---

## 💡 Key Insights & Decisions

### Architecture Decisions
1. **Runtime Compilation Only**: No HIP device code at build time, all via hipRTC
2. **Configuration-Based Framework**: BenchmarkConfig with flexible ParamMap
3. **C++ for Performance**: Infrastructure in C++ with Python bindings via pybind11
4. **Separation of Concerns**: framework/ (HOW), benchmarks/ (WHAT), kernels/ (CODE)
5. **Pandas + SQLite**: Lightweight storage, perfect for single-user workflow
6. **matplotlib**: Standard Python plotting, familiar to scientific community

### What's Working Well
- BenchmarkRunner abstraction is clean and extensible
- Runtime parameters avoid template explosion
- Python API is intuitive with kwargs
- Single kernel compilation is fast (0.27s)
- BenchmarkDB makes data persistence trivial
- Visualization functions are flexible and reusable
- Jupyter notebook validates complete workflow

### Lessons Learned
- C++ lambdas in configs are more powerful than YAML would be
- Repository organization matters - clear naming improves intuition
- Generic framework was worth the investment - adding benchmarks is now straightforward
- Storage + visualization should have come earlier (but better late than never!)
- Jupyter notebook is excellent living documentation

### Known Issues
- "Registered benchmark" debug prints appear (minor annoyance)
- Benchmark descriptions hardcoded in notebook (should be in BenchmarkConfig)
- No automated testing yet
- Only 1 of 11 benchmarks implemented

---

## Success Criteria

The project will be considered successful when:

1. ✅ All CUDA benchmarks converted to HIP
2. ✅ hipRTC integration working
3. ✅ Can run at least 1 benchmark end-to-end from Python
4. ✅ Generic framework supports pluggable benchmarks
5. ✅ Visualizations match quality of original gpu-benches
6. ✅ Example Jupyter notebooks demonstrate key features
7. ⬜ At least 5-6 benchmarks working (50%+)
8. ⬜ Documentation covers installation and usage
9. ⬜ CI/CD pipeline ensures builds work


**Milestone Achieved**: Foundation complete! Storage, visualization, and demo notebook all working.

---

**Last Updated**: October 16, 2025  
**Current Phase**: Benchmark Expansion (Foundation Complete!)  
**Next Session Focus**: TODO 5 - Add GPU Latency benchmark  
**Repository**: github.com/diptorupd/rocmGPUBenches  
**Branch**: main (all Phase 6-8 features merged)  
**Maintainer**: @diptorupd

**Recent Achievements** (Oct 16, 2025):
- ✅ Storage module merged (PR #4)
- ✅ Visualization module merged (PR #5)
- ✅ README & documentation updates merged (PR #7, #8)
- ✅ Jupyter demo notebook merged (PR #9)
- 🎯 Ready to add GPU Latency benchmark!
