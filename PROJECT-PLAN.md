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
3. 🔄 **Python API**: Create a clean Python interface for all benchmarks
4. 🔄 **Execution Framework**: Execute compiled kernels and collect results
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
- [x] Configured CMake build system
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
- [x] Created `HipRTCCompiler` C++ class wrapper
- [x] Implemented pybind11 bindings for Python access
- [x] Configured CMake to link HIP and hipRTC libraries
- [x] Successfully tested kernel compilation
- [x] Added GPU architecture auto-detection

**Current GPU:** `gfx942:sramecc+:xnack-` (MI300 series)

### 🔄 In Progress

Currently at the beginning of Phase 4.

### 
**Goal**: Execute compiled kernels and measure performance

#### 4.1 Kernel Execution Infrastructure
- [ ] Extend `HipRTCCompiler` to execute loaded kernels
- [ ] Add parameter passing mechanism for kernel arguments
- [ ] Implement device memory allocation/deallocation wrappers
- [ ] Add synchronization and error checking utilities

#### 4.2 Memory Management
- [ ] Create Python-friendly memory buffer classes
- [ ] Support for different data types (float, double, int, etc.)
- [ ] Implement data transfer (host ↔ device)
- [ ] Add memory pooling for efficiency

#### 4.3 Timing and Metrics
- [ ] Integrate HIP events for accurate timing
- [ ] Calculate effective bandwidth
- [ ] Measure kernel launch overhead
- [ ] Collect hardware counters (if available)

**Estimated effort**: 2-3 sessions

### � TODO: Phase 5 - Benchmark API Design

**Goal**: Create clean Python APIs for each benchmark

#### 5.1 Base Benchmark Class
- [ ] Design abstract `Benchmark` base class
- [ ] Standard interface: `setup()`, `run()`, `teardown()`
- [ ] Configuration via Python dictionaries
- [ ] Result reporting in structured format

#### 5.2 Implement Individual Benchmarks
- [ ] `GPUCacheBenchmark` - Cache hierarchy characterization
- [ ] `GPUStreamBenchmark` - Memory bandwidth measurement
- [ ] `GPULatencyBenchmark` - Memory latency profiling
- [ ] `GPURooflineBenchmark` - Roofline model generation
- [ ] Additional benchmarks as needed

#### 5.3 Benchmark Registry
- [ ] Create registry for discovering available benchmarks
- [ ] Support for listing and filtering benchmarks
- [ ] Dynamic loading of benchmark implementations

**Estimated effort**: 3-4 sessions

### 📋 TODO: Phase 6 - Visualization

**Goal**: Generate meaningful plots and reports

#### 6.1 Data Collection
- [ ] Structured result storage (JSON/CSV)
- [ ] Support for multiple runs and averaging
- [ ] Statistical analysis (mean, std, percentiles)

#### 6.2 Plotting Infrastructure
- [ ] Integrate matplotlib/plotly
- [ ] Reuse plotting code from original gpu-benches
- [ ] Create standard plot templates:
  - Bandwidth vs. data size
  - Latency vs. stride pattern
  - Roofline plots
  - Comparison plots across runs

#### 6.3 Report Generation
- [ ] Generate markdown/HTML reports
- [ ] Include plots and summary statistics
- [ ] Support for comparing different GPUs/configurations

**Estimated effort**: 2-3 sessions

### 📋 TODO: Phase 7 - Jupyter Integration

**Goal**: Interactive benchmark exploration

#### 7.1 Notebook Examples
- [ ] Create example notebooks for each benchmark
- [ ] Interactive parameter exploration with widgets
- [ ] Real-time visualization updates

#### 7.2 IPython Magic Commands
- [ ] `%gpu_benchmark` magic for quick benchmarking
- [ ] `%%hip_kernel` cell magic for inline kernel definition
- [ ] Auto-detection of GPU in notebook environment

#### 7.3 Dashboard
- [ ] Interactive dashboard using ipywidgets
- [ ] Benchmark selection and configuration UI
- [ ] Live results display

**Estimated effort**: 2-3 sessions

### 📋 TODO: Phase 8 - Testing & CI/CD

**Goal**: Ensure reliability and maintainability

#### 8.1 Unit Tests
- [ ] Test hipRTC compilation with various kernels
- [ ] Test memory management utilities
- [ ] Test benchmark execution
- [ ] Mock GPU operations for CPU-only testing

#### 8.2 Integration Tests
- [ ] End-to-end benchmark runs
- [ ] Test result parsing and visualization
- [ ] Test notebook execution

#### 8.3 CI/CD Pipeline
- [ ] GitHub Actions workflow
- [ ] Build and test on ROCm container
- [ ] Generate and upload documentation
- [ ] Package building and distribution

**Estimated effort**: 2-3 sessions

### 📋 TODO: Phase 9 - Documentation & Polish

**Goal**: Make the project accessible and maintainable

#### 9.1 Documentation
- [ ] README with quick start guide
- [ ] API documentation (Sphinx)
- [ ] Architecture documentation
- [ ] Contributing guidelines

#### 9.2 Examples
- [ ] Standalone Python scripts
- [ ] Jupyter notebooks
- [ ] Performance tuning guides

#### 9.3 Packaging
- [ ] Prepare for PyPI distribution
- [ ] Add installation instructions for different platforms
- [ ] Create conda recipe

**Estimated effort**: 2-3 sessions

## Technical Details

### Development Environment

- **OS**: Ubuntu 24.04 LTS (in dev container)
- **Python**: 3.12
- **ROCm**: 6.4.1
- **GPU**: AMD MI300 series (gfx942)
- **Build System**: scikit-build-core + CMake 4.1.2
- **Conda Environment**: `rocm-gpubench-env`

### Dependencies

**Python Packages:**
- scikit-build-core
- pybind11
- numpy (for data handling)
- matplotlib/plotly (for visualization)
- jupyter/ipywidgets (for notebooks)

**System Libraries:**
- HIP runtime
- hipRTC
- libstdc++-14-dev

### Project Structure

```
rocmGPUBenches/
 CMakeLists.txt              # Build configuration
 pyproject.toml              # Python package metadata
 LICENSE                     # GPLv3
 README.md                   # User documentation
 PROJECT-PLAN.md            # This file
 src/rocmGPUBenches/
   ├── __init__.py            # Python package entry point
   ├── main.cpp               # Basic bindings
   ├── rocmGPUBenches.cpp     # pybind11 module definition
   ├── hip_rtc_compiler.hpp   # hipRTC wrapper (header-only)
   ├── hip_rtc_bindings.cpp   # hipRTC Python bindings
   └── kernels/               # Hipified GPU kernels (11 benchmarks)
       ├── gpu-cache.hip
       ├── gpu-stream.hip
       └── ... (9 more)
 tests/                     # Unit tests (TODO)
```

### Key Design Decisions

1. **Header-only HipRTCCompiler**: Simplifies build, inline implementation in .hpp
2. **Explicit Library Linking**: Direct path to `libhiprtc.so` to avoid CMake config issues
3. **Architecture Auto-detection**: Uses `hipGetDeviceProperties()` to get `gcnArchName`
4. **Editable Install**: Development with `pip install -e .` for rapid iteration

## Development Workflow

### Daily Workflow
1. Activate conda environment: `micromamba activate rocm-gpubench-env`
2. Navigate to project: `cd /devel/rocmGpubenches`
3. Check git status: `git status` and `git log --oneline -5`
4. Make changes
5. Rebuild: `pip install -e . --force-reinstall --no-deps`
6. Test changes: `python -c "import rocmGPUBenches; ..."`
7. Commit: `git add -A && git commit -m "Description"`
8. Push: `git push`

### Testing hipRTC
```python
import rocmGPUBenches

compiler = rocmGPUBenches.HipRTCCompiler()
arch = compiler.get_gpu_arch()
code = compiler.compile(kernel_source, "kernel.hip", [f"--gpu-architecture={arch}"])
```

### Resuming Work

When returning to the project:
1. The workspace in `/devel/rocmGpubenches` persists
2. Git history is preserved
3. Conda environment should persist
4. Start a new AI chat and reference this PROJECT-PLAN.md
5. Check the most recent commits: `git log --oneline -10`

## Resources

- **GitHub Repository**: https://github.com/diptorupd/rocmGPUBenches
- **Original gpu-benches**: https://github.com/te42kyfo/gpu-benches
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **hipRTC Documentation**: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/

## Notes for Future AI Agents

### Context for Continuation

When resuming work on this project:

1. **Read this file first** to understand goals and current status
2. **Check recent commits**: `git log --oneline -10` to see latest work
3. **Verify environment**: 
   - `micromamba activate rocm-gpubench-env`
   - `cd /devel/rocmGpubenches`
   - `python -c "import rocmGPUBenches; print(rocmGPUBenches.__version__)"`
4. **Look at the next TODO phase** and start from there

### Common Issues

- **hipRTC linking**: Library path is explicitly set in CMakeLists.txt
- **CMake version**: Using 3.21+ with CMAKE_POLICY_VERSION_MINIMUM=3.5 workaround
- **Import errors**: Always rebuild after C++ changes: `pip install -e . --force-reinstall --no-deps`

### Quick Commands Reference

```bash
# Rebuild package
pip install -e . --force-reinstall --no-deps

# Test import
python -c "import rocmGPUBenches; print(rocmGPUBenches.HipRTCCompiler())"

# Check GPU
rocminfo | grep gfx

# Commit and push
git add -A && git commit -m "Your message" && git push
```

## Timeline Estimate

- **Phase 4** (Kernel Execution): 2-3 sessions
- **Phase 5** (Benchmark API): 3-4 sessions  
- **Phase 6** (Visualization): 2-3 sessions
- **Phase 7** (Jupyter): 2-3 sessions
- **Phase 8** (Testing/CI): 2-3 sessions
- **Phase 9** (Documentation): 2-3 sessions

**Total**: ~15-20 work sessions to complete all phases

## Success Criteria

The project will be considered successful when:

1. ✅ All CUDA benchmarks converted to HIP
2. ✅ hipRTC integration working
3. ⬜ Can run at least 3 benchmarks end-to-end from Python
4. ⬜ Visualizations match quality of original gpu-benches
5. ⬜ Example Jupyter notebooks demonstrate all features
6. ⬜ Documentation covers installation and usage
7. ⬜ CI/CD pipeline ensures builds work

## Maintenance

### Long-term Considerations

- Keep up with ROCm updates
- Support new GPU architectures as they release
- Consider supporting multi-GPU benchmarks
- Potential for performance regression testing
- Community contributions and issue management

---

**Last Updated**: October 15, 2025  
**Current Phase**: Phase 4 (Kernel Execution Framework)  
**Project Status**: Active Development  
**Maintainer**: @diptorupd
