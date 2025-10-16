# rocmGPUBenches Project Plan

## Project Overview

**rocmGPUBenches** is a modern Python package that consolidates GPU microbenchmarks from the original gpu-benches repository into a unified tool with dynamic kernel compilation and interactive visualization capabilities.

### Key Features
- **Dynamic Compilation**: Use hipRTC to compile HIP kernels on-the-fly
- **Python API**: Easy-to-use Python interface for running benchmarks
- **Interactive Notebooks**: Jupyter integration for exploration and visualization
- **Architecture-Aware**: Automatically detects and optimizes for GPU architecture

### Original Source
Based on the [gpu-benches](https://github.com/te42kyfo/gpu-benches) repository, which contains microbenchmarks for NVIDIA and AMD GPUs. We are focusing exclusively on ROCm/AMD GPUs.

## Goals

### Primary Goals
1. ✅ **Consolidate Benchmarks**: Convert all CUDA microbenchmarks to HIP
2. ✅ **Dynamic Compilation**: Integrate hipRTC for runtime kernel compilation
3. ✅ **Python API**: Create a clean Python interface (GPUCacheBenchmark working!)
4. ✅ **Execution Framework**: Execute compiled kernels and collect results
5. 🔄 **Visualization**: Generate plots and performance metrics
6. 🔄 **Jupyter Integration**: Interactive notebook experience

### Secondary Goals
- Performance comparison across different GPU architectures
- CI/CD pipeline for automated testing
- Documentation and examples
- Package distribution via PyPI

## Project Status

### ✅ Completed

#### Phase 1: Project Setup (Completed)
- [x] Created project structure with `scikit-build-core`
- [x] Set up Python package layout
- [x] Configured CMake build system (C++ only, no device code at build time)
- [x] Added GPLv3 license (matching original project)
- [x] Initialized git repository
- [x] Pushed to GitHub: `github.com/diptorupd/rocmGPUBenches`

#### Phase 2: Kernel Conversion (Completed)
- [x] Hipified all CUDA microbenchmarks to HIP format
- [x] Stored hipified kernels in `src/rocmGPUBenches/kernels/`
- [x] Available benchmarks:
  - `cuda-incore.hip`
  - `cuda-memcpy.hip`
  - `gpu-cache.hip`
  - `gpu-l2-cache.hip`
  - `gpu-l2-stream.hip`
  - `gpu-latency.hip`
  - `gpu-roofline.hip`
  - `gpu-small-kernels.hip`
  - `gpu-stream.hip`
  - `gpu-strides.hip`
  - `um-stream.hip`

#### Phase 3: hipRTC Integration (Completed)
- [x] Created `HipRTCCompiler` C++ class wrapper (header-only)
- [x] Implemented pybind11 bindings for Python access
- [x] Configured CMake to link HIP runtime and hipRTC libraries
- [x] Successfully tested kernel compilation
- [x] Added GPU architecture auto-detection
- [x] **Architecture Decision**: Runtime compilation only - no device code at build time

**Current GPU:** `gfx942:sramecc+:xnack-` (AMD MI325X)

#### Phase 4: Kernel Execution Framework (Completed - First Benchmark!)
- [x] Copied shared utilities from gpu-benches (MeasurementSeries, dtime, gpu-error, rocm-metrics)
- [x] Verified utilities are HIP-compatible
- [x] Created `GPUCacheBenchmark` C++ class with runtime hipRTC compilation
- [x] Implemented memory allocation and kernel execution
- [x] Added HIP event-based timing
- [x] Created Python bindings for `GPUCacheBenchmark`
- [x] **Successfully tested end-to-end**: Running on MI325X, achieving 414 TB/s bandwidth!
- [x] Kernel compilation happens at runtime via hipRTC
- [x] Kernels are cached to avoid recompilation

**Key Achievement**: First working benchmark using pure runtime compilation approach!

### 🔄 Current Status

**Current Phase**: Phase 5 - Generalizing the Benchmark Framework

The GPU cache benchmark is working, but we need to:
1. Generalize the approach for all 11 benchmarks
2. Optimize compilation strategy (avoid compiling too many kernels)
3. Define standard APIs and project structure

### 📋 TODO: Critical Next Steps

#### TODO 1: Generic GPUBenchmark Base Class API 🎯
**Priority**: HIGH
**Effort**: 2-3 sessions

**Goal**: Make GPUCacheBenchmark into a generic framework

- [ ] Design `GPUBenchmark` base class with virtual methods:
  - `get_kernel_source()` - Returns kernel source as string
  - `configure_kernel(params)` - Sets runtime parameters
  - `run(iterations)` - Executes benchmark and returns results
  - `get_results()` - Returns structured results
- [ ] Refactor `GPUCacheBenchmark` to inherit from base class
- [ ] Document the API for adding new benchmarks
- [ ] Create example template for new benchmarks

**Benefits**: All 11 benchmarks can plug into common framework

#### TODO 2: Optimize Kernel Compilation Strategy 🚀
**Priority**: HIGH
**Effort**: 1-2 sessions

**Current Issue**: We generate unique kernel for each (N, iters, blockSize) combination, causing 15+ compilations per sweep

**Solutions to implement**:
- [ ] **Use runtime parameters instead of compile-time constants**
  - Compile one parameterized kernel
  - Pass N, iters, blockSize as kernel arguments
  - Trade-off: Slightly less optimal but 15x faster compilation
- [ ] **Implement kernel caching**
  - ✅ Already caching in memory
  - [ ] Add disk-based cache (serialize compiled modules)
  - [ ] Cache key: `(kernel_source_hash, gpu_arch)`
- [ ] **Lazy compilation**
  - Only compile when needed
  - First run compiles, subsequent runs use cache

**Expected speedup**: 10-15x faster for sweep tests

#### TODO 3: Pre-compilation for Fixed GPU Architectures 📦
**Priority**: MEDIUM
**Effort**: 2-3 sessions

**Goal**: Optional AOT (Ahead-of-Time) compilation at package build time

- [ ] Add CMake option: `-DPRECOMPILE_KERNELS=ON`
- [ ] Specify target architectures: `gfx90a,gfx942,gfx1100`
- [ ] Use `hipcc --genco` to generate code objects
- [ ] Bundle pre-compiled kernels in Python package
- [ ] Runtime: Load pre-compiled if available, else use hipRTC
- [ ] Document trade-offs:
  - ✅ Faster first-run (no compilation)
  - ✅ Works offline
  - ❌ Larger package size
  - ❌ Limited to pre-selected architectures

**Use case**: Production deployments, benchmark suites

#### TODO 4: Reevaluate Project Source Structure 🗂️
**Priority**: MEDIUM
**Effort**: 1 session

**Current structure needs clarification**:

```
src/rocmGPUBenches/
 kernels/              # What should go here?
   ├── gpu-cache.hip     # Full original benchmark (not used)
   └── ...               # Other hipified benchmarks
 utils/                # ✅ Shared utilities
 gpu_cache_benchmark.{hpp,cpp}  # Individual benchmark classes
 gpu_cache_kernel_source.hpp    # Kernel sources as strings
 hip_rtc_compiler.hpp  # ✅ Runtime compiler
```

#### TODO 5: Development workflow for new benchmarks

- Create new benchmark as standalone .hip file
- Compile/test with amdclang++ for fast iteration
- Once working, convert to string literal for hipRTC
- Optional: Script to auto-convert .hip → .hpp string

**Questions to answer**:
- [ ] Should `kernels/` contain:
  - Option A: Full `.hip` files from gpu-benches (current)
  - Option B: Cleaned-up kernel-only versions
  - Option C: Just kernel source string generators (`.hpp` files)
- [ ] Where should benchmark classes live?
  - Option A: `benchmarks/` subdirectory
  - Option B: Root of `src/rocmGPUBenches/`
- [ ] How to organize for 11 benchmarks?
  - Option A: One file per benchmark (11 `.cpp` files)
  - Option B: Group by type (memory, compute, latency)

**Proposal**:
```
src/rocmGPUBenches/
 benchmarks/           # NEW: All benchmark implementations
   ├── base_benchmark.hpp
   ├── gpu_cache_benchmark.{hpp,cpp}
   ├── gpu_latency_benchmark.{hpp,cpp}
   └── ...
 kernels/              # Kernel source generators
   ├── cache_kernels.hpp
   ├── latency_kernels.hpp
   └── ...
 utils/                # Shared utilities
 hip_rtc_compiler.hpp
```

#### TODO 5: Define Steps to Add New Benchmarks 📝
**Priority**: HIGH
**Effort**: Documentation task

**Goal**: Clear workflow for adding remaining 10 benchmarks

**Checklist for adding a new benchmark**:
1. [ ] Create kernel source generator: `kernels/<name>_kernels.hpp`
   - Define kernel as raw string literal
   - Add helper function: `get_<name>_kernel_source()`
2. [ ] Create benchmark class: `benchmarks/<name>_benchmark.{hpp,cpp}`
   - Inherit from `GPUBenchmark` base class
   - Implement required virtual methods
   - Add result struct
3. [ ] Create pybind11 bindings: `benchmarks/<name>_bindings.cpp`
   - Expose class to Python
   - Add to module registration in `rocmGPUBenches.cpp`
4. [ ] Update CMakeLists.txt:
   - Add new `.cpp` files to `pybind11_add_module()`
   - No other changes needed!
5. [ ] Update `__init__.py`:
   - Import new benchmark class
   - Add to `__all__`
6. [ ] Test:
   - Create `test_<name>.py`
   - Verify compilation and execution

**Key insight**: With the runtime compilation approach, **no CMake/build changes needed for kernels**, only for C++ source files!

**Priority**: MEDIUM  #### TODO 6: Benchmark Results Storage & Visualization
**Effort**: 2-3 sessions

**Current state**: Results exist only in memory during benchmark run

**Requirements**:
- [ ] Persistent storage for historical comparisons
- [ ] Query results by GPU, date, benchmark type
- [ ] Export for visualization tools

**Proposed solution: SQLite database**

```sql
CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    gpu_name TEXT,
    gpu_arch TEXT,
    benchmark_type TEXT,  -- 'cache', 'latency', etc.
    parameters JSON,      -- {N: 256, blockSize: 256, ...}
    results JSON,         -- {exec_time_ms: 23.5, bandwidth_gbs: 414, ...}
    rocm_version TEXT
);
```

**Implementation**:
- [ ] Create `BenchmarkDB` class
- [ ] Add `save_result()` method to base benchmark class
- [ ] Create query API: `db.get_results(gpu='MI325X', benchmark='cache')`
- [ ] Add CLI: `rocmgpubench history --benchmark cache`
- [ ] Export functions: `export_csv()`, `export_json()`

**Visualization**:
- [ ] matplotlib plots: bandwidth vs. data size
- [ ] Compare across GPUs
- [ ] Interactive: plotly/bokeh
- [ ] Integration with Jupyter notebooks

### 📋 Additional TODOs

#### TODO 7: Testing Infrastructure 🧪
**Priority**: HIGH
**Effort**: 2 sessions

- [ ] Add pytest framework
- [ ] Unit tests for `HipRTCCompiler`
- [ ] Integration tests for each benchmark
- [ ] CI/CD with GitHub Actions
- [ ] Test on multiple ROCm versions

#### TODO 8: Documentation 📚
**Priority**: MEDIUM
**Effort**: 2-3 sessions

- [ ] API documentation with Sphinx
- [ ] Installation guide
- [ ] Tutorial notebooks
- [ ] Performance analysis guide
- [ ] Contributing guidelines

#### TODO 9: Packaging & Distribution 📦
**Priority**: LOW
**Effort**: 1-2 sessions

- [ ] Prepare for PyPI
- [ ] Add version management
- [ ] Create release workflow
- [ ] Docker container with ROCm

### 🎯 Phase 5: Benchmark API Design (Next Up)

**Goal**: Implement generic framework and add more benchmarks

1. Implement `GPUBenchmark` base class (TODO 1)
2. Optimize kernel compilation (TODO 2)
3. Refactor `GPUCacheBenchmark` to use base class
4. Add 2-3 more benchmarks as proof of concept:
   - GPU Latency (simple, good test case)
   - GPU Stream (different pattern)
   - GPU Roofline (complex)

**Estimated effort**: 4-5 sessions

### 🎨 Phase 6: Visualization & Analysis (Future)

**Goal**: Create compelling visualizations

1. Implement SQLite storage (TODO 6)
2. Create plotting utilities
3. Generate comparison charts
4. Interactive dashboards

**Estimated effort**: 3-4 sessions

### 📓 Phase 7: Jupyter Integration (Future)

**Goal**: Interactive exploration

1. Create example notebooks
2. Add magic commands
3. Live performance monitoring
4. Tutorial content

**Estimated effort**: 2-3 sessions

### ✅ Phase 8: Testing & CI (Future)

**Goal**: Ensure reliability

1. Implement TODO 7
2. Set up CI/CD
3. Performance regression tests

**Estimated effort**: 2-3 sessions

### 📖 Phase 9: Documentation (Future)

**Goal**: Make it accessible

1. Complete TODO 8
2. Video tutorials
3. Blog posts

**Estimated effort**: 2-3 sessions

## Notes for Future AI Agents

### Architecture Decisions Made

1. **Runtime Compilation Only**: No HIP device code compiled at build time
   - All kernels compiled via hipRTC at runtime
   - CMakeLists.txt only links HIP runtime + hipRTC libraries
   - Build uses g++ for pure C++ host code

2. **Kernel Storage**: Kernels as C++ string literals in header files
   - Easy to read and modify
   - No file I/O at runtime
   - Example: `gpu_cache_kernel_source.hpp`

3. **Caching Strategy**: In-memory kernel cache per session
   - Key: kernel configuration (needs optimization!)
   - Future: Add disk-based cache

4. **Python Bindings**: pybind11 for C++ ↔ Python interface
   - Clean, type-safe API
   - Result structs exposed as Python classes

### Development Workflow

```bash
# Activate environment
micromamba activate rocm-gpubench-env

# Build and install
cd /devel/rocmGpubenches
pip install -e . --no-build-isolation

# Test
python -c "from rocmGPUBenches import GPUCacheBenchmark; bench = GPUCacheBenchmark(); print(bench.run(256, 256, 5))"

# Check GPU
rocminfo | grep gfx

# Commit
git add -A && git commit -m "Your message" && git push
```

### Current Test Results

**GPU**: AMD Instinct MI325X (gfx942)
**Compute Units**: 304
**Test**: GPUCacheBenchmark, N=256, blockSize=256, 5 iterations
**Result**:
- Execution Time: 23.488 ms
- Bandwidth: 414,499 GB/s (414 TB/s!)
- Spread: 3.57%

### Known Issues

1. **Slow sweep tests**: Each new N value requires kernel recompilation
   - **Fix**: TODO 2 (use runtime parameters)
2. **Terminal hangs**: Python process hangs on larger sweeps
   - Likely due to excessive compilation time
   - **Fix**: TODO 2
3. **No result persistence**: Results lost after run
   - **Fix**: TODO 6 (SQLite storage)

## Timeline Estimate

- ✅ **Phase 1-4** (Setup through First Benchmark): Complete!
- **TODO 1-2** (Generalize & Optimize): 3-4 sessions
- **TODO 3-5** (Structure & Process): 3-4 sessions
- **TODO 6** (Storage & Viz): 2-3 sessions
- **Phase 7-9** (Jupyter, Testing, Docs): 8-10 sessions

**Total Remaining**: ~16-21 work sessions

## Success Criteria

The project will be considered successful when:

1. ✅ All CUDA benchmarks converted to HIP
2. ✅ hipRTC integration working
3. ✅ Can run at least 1 benchmark end-to-end from Python (GPUCache working!)
4. ⬜ Generic framework supports all 11 benchmarks
5. ⬜ Visualizations match quality of original gpu-benches
6. ⬜ Example Jupyter notebooks demonstrate all features
7. ⬜ Documentation covers installation and usage
8. ⬜ CI/CD pipeline ensures builds work

## Maintenance

### Long-term Considerations

- Keep up with ROCm updates
- Support new GPU architectures as they release
- Consider supporting multi-GPU benchmarks
- Potential for performance regression testing
- Community contributions and issue management

---

**Last Updated**: October 16, 2025
**Current Phase**: Phase 5 (Generalizing Benchmark Framework)
**Current Status**: First benchmark (GPU Cache) working with runtime hipRTC compilation!
**Next Session Focus**: Optimize kernel compilation (TODO 2), then generalize framework (TODO 1)
**Maintainer**: @diptorupd
