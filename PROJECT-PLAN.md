# rocmGPUBenches Project Plan

## Project Overview

**rocmGPUBenches** is a modern Python package that consolidates GPU microbenchmarks from the original gpu-benches repository into a unified tool with dynamic kernel compilation and interactive visualization capabilities.

### Key Features
- **Dynamic Compilation**: Use hipRTC to compile HIP kernels on-the-fly
- **Python API**: Easy-to-use Python interface for running benchmarks
- **Generic Framework**: Configuration-based BenchmarkRunner for pluggable benchmarks
- **Interactive Notebooks**: Jupyter integration for exploration and visualization
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
 cache_kernels.hpp (active)   ├
   └── *.hip (11 reference files to be converted)
 hiprtc_utils/       # HIP runtime compilation
   ├── hip_rtc_compiler.hpp
   └── hip_rtc_bindings.cpp
 utils/              # Measurement utilities
    ├── MeasurementSeries.hpp
    ├── dtime.hpp
    ├── gpu-error.h
    └── rocm-metrics.hpp
```

**Current Working Example**:
```python
from rocmGPUBenches import create_cache_benchmark_runner
runner = create_cache_benchmark_runner()

# Single run
result = runner.run('cache', problem_size=256)
# → bandwidth_gbs: 21695.15, spread: 0.3%

# Parameter sweep
results = runner.sweep('cache', 'problem_size', [128, 256, 512, 1024])
```

**Test Environment**:
- **GPU**: AMD Instinct MI325X (gfx942:sramecc+:xnack-)
- **Compute Units**: 304
- **ROCm**: 6.4.1
- **Python**: 3.12 in conda environment `rocm-gpubench-env`
- **Build**: scikit-build-core + CMake, g++ for host, hipRTC for kernels

### 🔄 Current Focus: Visualization & Jupyter Integration

We have a solid foundation with the generic framework. Now we need to:
1. Visualize benchmark results
2. Create interactive Jupyter notebooks
3. Add more benchmarks using the established pattern

---

## 📋 Active TODOs

### TODO 3: Integrate Visualization (NEXT UP)
**Priority**: HIGH  
**Effort**: 1 session  
**Status**: Not started

**Goal**: Port plotting functionality from gpu-benches and adapt for BenchmarkRunner results

**Plan**:
1. Create `src/rocmGPUBenches/visualization/` module
2. Port gpu-benches cache plotter (`/devel/gpu-benches/gpu-cache/plot.py`)
3. Adapt to work with `BenchmarkResult` objects instead of CSV files
4. Create unified plotting API
5. Test with cache benchmark sweep results

**Structure**:
```python
src/rocmGPUBenches/visualization/
 __init__.py
 plot_cache.py       # Cache bandwidth vs size
 plot_latency.py     # (future)
 common.py           # Shared plotting utilities
```

**API Design**:
```python
from rocmGPUBenches import create_cache_benchmark_runner
from rocmGPUBenches.visualization import plot_cache

runner = create_cache_benchmark_runner()
results = runner.sweep('cache', 'problem_size', [64, 128, 256, 512])
plot_cache(results, save='cache_sweep.png', show=True)
```

**Deliverable**: Working `plot_cache()` function that generates bandwidth plots

---

### TODO 4: Create Jupyter Notebook Demo
**Priority**: HIGH  
**Effort**: 1 session  
**Status**: Not started  
**Dependencies**: TODO 3 (visualization)

**Goal**: End-to-end interactive notebook demonstrating cache benchmark

**Content Outline**:
```
1. Introduction
   - What is cache benchmark
   - Why it matters for GPU performance
   
2. Setup & Environment
   - Import rocmGPUBenches
   - Check GPU info (device name, CUs, architecture)
   
3. Single Benchmark Run
   - Run with default parameters
   - Display result (bandwidth, spread)
   
4. Parameter Sweep: Problem Size
   - Sweep problem_size: [64, 128, 256, 512, 1024]
   - Plot bandwidth vs problem size
   - Identify performance characteristics
   
5. Parameter Sweep: Block Size
   - Compare block_size: [256, 512, 1024]
   - Find optimal configuration
   
6. Performance Analysis
   - Compare with theoretical peak bandwidth
   - Analyze cache hierarchy effects
   - Discuss performance implications
```

**Location**: `notebooks/cache_benchmark_demo.ipynb`

**Deliverable**: Runnable Jupyter notebook with rich output and visualizations

---

### TODO 5: Add More Benchmarks (2-3 examples)
**Priority**: MEDIUM  
**Effort**: 2 sessions (1 per benchmark + 1 for polish)  
**Status**: Not started

**Goal**: Validate framework generality by adding different benchmark types

**Candidates** (in order of simplicity):

1. **GPU Latency** (Simplest - good test)
   - Measures memory access latency
   - Single kernel, different metrics than cache
   - Files: `kernels/latency_kernels.hpp`, `benchmarks/latency_benchmark_config.hpp`
   - Metric: latency in nanoseconds
   
2. **GPU Stream** (Different pattern)
   - Tests sustained memory bandwidth
   - Multiple arrays (copy, scale, add, triad)
   - Different memory access pattern than cache
   - Metric: bandwidth GB/s
   
3. **GPU L2 Cache** (Related to cache)
   - Extends cache benchmark
   - Tests L2 cache specifically
   - Similar to cache but different working set sizes

**Per Benchmark Checklist**:
- [ ] Create `kernels/<name>_kernels.hpp` with kernel source
- [ ] Create `benchmarks/<name>_benchmark_config.hpp` with BenchmarkConfig
- [ ] Add `create_<name>_benchmark_runner()` to `framework/benchmark_runner_bindings.cpp`
- [ ] Create `visualization/plot_<name>.py`
- [ ] Create `notebooks/<name>_demo.ipynb`
- [ ] Add to `__init__.py` exports
- [ ] Test run, sweep, and plot

**Deliverable**: 3-4 total working benchmarks (cache + 2-3 new)

---

## 📋 Future TODOs

### TODO 6: Persistent Result Storage
**Priority**: MEDIUM  
**Effort**: 1-2 sessions  
**Status**: Not started

**Current Problem**: Results only exist in memory during benchmark run

**Proposed Solution: SQLite database**

```sql
CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    gpu_name TEXT,
    gpu_arch TEXT,
    benchmark_type TEXT,  -- 'cache', 'latency', etc.
    parameters JSON,      -- {problem_size: 256, block_size: 256, ...}
    results JSON,         -- {bandwidth_gbs: 21695.15, spread_percent: 0.3, ...}
    rocm_version TEXT
);
```

**Implementation**:
- [ ] Create `BenchmarkDB` class
- [ ] Add `save_result()` method to BenchmarkRunner
- [ ] Query API: `db.get_results(gpu='MI325X', benchmark='cache')`
- [ ] CLI: `rocmgpubench history --benchmark cache`
- [ ] Export: `export_csv()`, `export_json()`

**Use Cases**:
- Historical comparison
- Performance regression tracking
- Cross-GPU comparison
- Sharing results

**Deliverable**: Automatic result persistence with query/export functionality

---

### TODO 7: Testing Infrastructure
**Priority**: MEDIUM  
**Effort**: 2 sessions  
**Status**: Not started

- [ ] Add pytest framework
- [ ] Unit tests for BenchmarkRunner
- [ ] Unit tests for HipRTCCompiler
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
- [ ] gpu-latency (simple)
- [ ] gpu-stream (medium)
- [ ] gpu-l2-cache (medium)
- [ ] gpu-l2-stream (medium)
- [ ] gpu-roofline (complex)
- [ ] gpu-small-kernels (medium)
- [ ] gpu-strides (medium)
- [ ] cuda-incore (simple)
- [ ] cuda-memcpy (simple)
- [ ] um-stream (medium)

**Strategy**: Follow established pattern from TODO 5

---

### TODO 9: YAML Configuration (Optional)
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

### TODO 10: Documentation
**Priority**: MEDIUM  
**Effort**: 2-3 sessions  
**Status**: Not started

- [ ] API documentation with Sphinx
- [ ] Installation guide
- [ ] Tutorial notebooks (beyond demos)
- [ ] Performance analysis guide
- [ ] Contributing guidelines
- [ ] Architecture documentation

---

### TODO 11: Packaging & Distribution
**Priority**: LOW  
**Effort**: 1-2 sessions  
**Status**: Not started

- [ ] Prepare for PyPI
- [ ] Add version management
- [ ] Create release workflow
- [ ] Docker container with ROCm

---

## 🎯 Recommended Roadmap

### Near-Term (Next 5-6 sessions)
1. **Session 1**: Visualization (TODO 3) - Port plotter, adapt to BenchmarkResult
2. **Session 2**: Jupyter Notebook (TODO 4) - End-to-end cache demo
3. **Session 3**: Add GPU Latency benchmark (TODO 5) - Validate framework
4. **Session 4**: Add GPU Stream benchmark (TODO 5) - Different pattern
5. **Session 5-6**: Result Storage (TODO 6) - Persistent data & queries

**Goal**: Solid foundation with 3-4 benchmarks, visualization, and notebooks

### Mid-Term (7-10 sessions)
- Testing infrastructure (TODO 7)
- Add 3-4 more benchmarks (TODO 8)
- Documentation (TODO 10)

### Long-Term (10+ sessions)
- Complete all 11 benchmarks (TODO 8)
- Advanced features (YAML configs, multi-GPU, etc.)
- Distribution (TODO 11)

---

## 📊 Progress Summary

| Phase | Status | Sessions | Completed |
|-------|--------|----------|-----------|
| **Phase 1-4: Foundation** | ✅ Complete | ~8 | Oct 16, 2025 |
| **Phase 5: Generic Framework** | ✅ Complete | 4 | Oct 16, 2025 |
| **TODO 3: Visualization** | 🔄 Next | 1 | - |
| **TODO 4: Jupyter** | ⏳ Pending | 1 | - |
| **TODO 5: More Benchmarks** | ⏳ Pending | 2 | - |
| **TODO 6: Storage** | ⏳ Pending | 1-2 | - |
| **TODO 7: Testing** | ⏳ Pending | 2 | - |
| **TODO 8: All Benchmarks** | ⏳ Pending | 4-6 | 1/11 done |
| **TODO 9: YAML (optional)** | ⏸️ Deferred | 2-3 | - |
| **TODO 10: Docs** | ⏳ Pending | 2-3 | - |
| **TODO 11: Packaging** | ⏳ Pending | 1-2 | - |

**Total Completed**: ~12 sessions  
**Estimated Remaining**: ~20-25 sessions  
**Current Benchmark Count**: 1/11 (9%)

---

## 🎯 Immediate Next Steps

### Decision on Next Focus

**Question A**: YAML configs?
- **Answer**: ❌ Not now. C++ configs work well, lambdas are powerful. Revisit after 5-10 benchmarks.

**Question B**: Integrate plotter?
- **Answer**: ✅ Yes! Do this next (TODO 3). 1 session, high value, unblocks Jupyter.

**Question C**: Jupyter notebook?
- **Answer**: ✅ Yes! After plotter (TODO 4). 1 session, validates everything end-to-end.

### Recommended Immediate Order:
1. ✅ **Visualization (TODO 3)** - 1 session - HIGH priority
2. ✅ **Jupyter Notebook (TODO 4)** - 1 session - HIGH priority
3. ⬜ **GPU Latency (TODO 5)** - 1 session - Validates framework
4. ⬜ **GPU Stream (TODO 5)** - 1 session - Different pattern
5. ⬜ **Result Storage (TODO 6)** - 1-2 sessions - Useful with multiple benchmarks

**Rationale**:
- Visualization is the bottleneck - we have working benchmarks but can't see results well
- Jupyter validates the entire stack and serves as living documentation
- 2-3 more benchmarks prove the framework's generality
- Storage becomes valuable once we have multiple benchmarks generating data
- This order gives quick wins and steady progress

---

## 💡 Key Insights & Decisions

### Architecture Decisions
1. **Runtime Compilation Only**: No HIP device code at build time, all via hipRTC
2. **Configuration-Based Framework**: BenchmarkConfig with flexible ParamMap
3. **C++ for Performance**: Infrastructure in C++ with Python bindings via pybind11
4. **Separation of Concerns**: framework/ (HOW), benchmarks/ (WHAT), kernels/ (CODE)

### What's Working Well
- BenchmarkRunner abstraction is clean and extensible
- Runtime parameters avoid template explosion
- Python API is intuitive with kwargs
- Single kernel compilation is fast (0.27s)

### Lessons Learned
- C++ lambdas in configs are more powerful than YAML would be
- Repository organization matters - clear naming improves intuition
- Generic framework was worth the investment - adding benchmarks is now straightforward
- Visualization and Jupyter should have come earlier in the plan

### Known Issues
- Benchmarks run slowly (15 iterations default) - could add quick mode
- No result persistence yet - data lost after run
- Only 1 of 11 benchmarks implemented

---

## Success Criteria

The project will be considered successful when:

1. ✅ All CUDA benchmarks converted to HIP
2. ✅ hipRTC integration working
3. ✅ Can run at least 1 benchmark end-to-end from Python
4. ✅ Generic framework supports pluggable benchmarks
5. ⬜ Visualizations match quality of original gpu-benches
6. ⬜ Example Jupyter notebooks demonstrate key features
7. ⬜ At least 5-6 benchmarks working (50%+)
8. ⬜ Documentation covers installation and usage
9. ⬜ CI/CD pipeline ensures builds work

**Current Score**: 4/9 (44%)

---

**Last Updated**: October 16, 2025  
**Current Phase**: Post-Phase 5 - Visualization & Demos  
**Next Session Focus**: TODO 3 - Integrate visualization from gpu-benches  
**Repository**: github.com/diptorupd/rocmGPUBenches  
**Branch**: main (feature/gpu_benchmark_api merged)  
**Maintainer**: @diptorupd
