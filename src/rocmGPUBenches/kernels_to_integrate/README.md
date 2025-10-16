# Kernels To Integrate

This directory contains original `.hip` kernel implementations from the [gpu-benches](https://github.com/te42kyfo/gpu-benches) repository that are candidates for integration into the rocmGPUBenches framework.

## Status

### ✅ Integrated
- **gpu-latency.hip** → `../kernels/latency_kernels.hpp` + `../benchmarks/latency_benchmark_config.hpp`
  - Status: Baseline integrated (measurements need debugging)
  
### 📋 To Integrate (Priority Order)
1. **gpu-stream.hip** - Memory bandwidth benchmark (STREAM triad)
   - Good next candidate: simple, well-understood metric
   
2. **gpu-roofline.hip** - Roofline model (compute vs memory bound)
   - Useful for performance analysis
   
3. **gpu-l2-cache.hip** - L2 cache-specific benchmark
   - Complements existing cache benchmark
   
4. **gpu-l2-stream.hip** - L2 streaming patterns
   
5. **gpu-strides.hip** - Memory stride patterns
   - Good for understanding memory access patterns
   
6. **gpu-small-kernels.hip** - Kernel launch overhead
   
7. **cuda-incore.hip** - In-core compute benchmark
   
8. **cuda-memcpy.hip** - Memory copy patterns
   
9. **um-stream.hip** - Unified memory streaming

## Integration Process

For each kernel:
1. Extract kernel source to `../kernels/<name>_kernels.hpp` as string literal
2. Create `../benchmarks/<name>_benchmark_config.hpp` with BenchmarkConfig
3. Add factory function to `../framework/benchmark_runner_bindings.cpp`
4. Export in `../__init__.py`
5. Test with framework (run, sweep, storage, visualization)
6. Remove/archive original `.hip` file

## Reference
Original implementations: `/devel/gpu-benches/` or https://github.com/te42kyfo/gpu-benches
