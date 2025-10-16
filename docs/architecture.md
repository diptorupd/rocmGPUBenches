# Architecture & Adding New Benchmarks

## Overview

rocmGPUBenches uses a **configuration-based architecture** where benchmarks are defined by:
1. **Kernel Code** (HIP/C++)
2. **Configuration** (C++ struct)
3. **Factory Function** (Python binding)

This design separates the "what to benchmark" (configuration) from "how to run benchmarks" (framework).

## Directory Structure

```
src/rocmGPUBenches/
 framework/              # Generic infrastructure (DON'T MODIFY for new benchmarks)
   ├── benchmark_runner.hpp
   ├── benchmark_runner.cpp
   └── benchmark_runner_bindings.cpp

 benchmarks/             # Benchmark configurations (ADD NEW CONFIGS HERE)
   ├── cache_benchmark_config.hpp
   └── [your_benchmark]_config.hpp

 kernels/                # Kernel implementations (ADD NEW KERNELS HERE)
   ├── cache_kernels.hpp
 [your_benchmark]_kernels.hpp   └─

 hiprtc_utils/           # Runtime compilation utilities
 storage/                # Database layer (Python)
 visualization/          # Plotting functions (Python)
 utils/                  # Measurement utilities
```

## How Benchmarks Work

### 1. Kernel Code (`kernels/*.hpp`)

Kernels are HIP code compiled at runtime via hipRTC. They should:
- Be self-contained (include all dependencies in the file)
- Use runtime parameters (not template parameters)
- Return meaningful metrics

**Example**: `kernels/cache_kernels.hpp`
```cpp
extern "C" __global__ void cache_benchmark_kernel(
    float* __restrict__ ptr,
    long int size,
    long int stride,
    long int iter
) {
    long int start_ind = threadIdx.x + blockIdx.x * blockDim.x;
    long int ind = start_ind;
    
    for(long int i = 0; i < iter; i++) {
        ind = ((long int*)ptr)[ind];
    }
    
    ((long int*)ptr)[start_ind] = ind;  // Prevent optimization
}
```

### 2. Configuration (`benchmarks/*_config.hpp`)

Configurations define:
- Kernel file path
- Memory requirements
- Default parameters
- Grid/block dimensions
- Metric calculation

**Template**:
```cpp
#pragma once
#include "../framework/benchmark_runner.hpp"

inline BenchmarkConfig create_[name]_benchmark_config() {
    BenchmarkConfig config;
    
    // 1. Kernel source
    config.name = "[name]";
    config.kernel_file = "src/rocmGPUBenches/kernels/[name]_kernels.hpp";
    
    // 2. Memory requirements
    config.allocate_memory = [](const ParamMap& params) -> size_t {
        int problem_size = std::get<int>(params.at("problem_size"));
        return problem_size * sizeof(float);  // or appropriate type
    };
    
    // 3. Grid/block dimensions
    config.grid_dim = [](const ParamMap& params, int max_threads) {
        int problem_size = std::get<int>(params.at("problem_size"));
        int block_size = std::get<int>(params.at("block_size"));
        return dim3((problem_size + block_size - 1) / block_size);
    };
    
    config.block_dim = [](const ParamMap& params, int max_threads) {
        return dim3(std::get<int>(params.at("block_size")));
    };
    
    // 4. Kernel arguments
    config.kernel_args = [](void* dev_ptr, const ParamMap& params) 
                         -> std::vector<void*> {
        static int problem_size;
        problem_size = std::get<int>(params.at("problem_size"));
        
        return {
            (void*)&dev_ptr,
            (void*)&problem_size
            // Add more parameters as needed
        };
    };
    
    // 5. Metric calculation
    config.metric_name = "your_metric_name";
    config.calculate_metric = [](double elapsed_ms, const ParamMap& params) {
        // Calculate your metric (bandwidth, latency, FLOPS, etc.)
        int problem_size = std::get<int>(params.at("problem_size"));
        double bytes = problem_size * sizeof(float) * 2;  // read + write
        return (bytes / 1e9) / (elapsed_ms / 1000.0);  // GB/s
    };
    
    // 6. Default parameters
    config.default_params = {
        {"problem_size", 256},
        {"block_size", 256}
    };
    
    // 7. Iteration count (for measurement stability)
    config.iterations = 1000;
    config.warmup_iterations = 100;
    
    return config;
}
```

### 3. Python Binding (`framework/benchmark_runner_bindings.cpp`)

Add a factory function to create your benchmark runner:

```cpp
// In benchmark_runner_bindings.cpp

#include "../benchmarks/[name]_benchmark_config.hpp"

// Add to init_benchmark_runner() function:
m.def("create_[name]_benchmark_runner", []() {
    auto config = create_[name]_benchmark_config();
    return std::make_shared<BenchmarkRunner>(config);
}, "Create a benchmark runner for [name] benchmark");
```

### 4. Python Export (`src/rocmGPUBenches/__init__.py`)

Add to the import list:
```python
create_[name]_benchmark_runner = _rocmGPUBenches.create_[name]_benchmark_runner
```

And to `__all__`:
```python
__all__.extend([
    'create_[name]_benchmark_runner'
])
```

## Step-by-Step: Adding a New Benchmark

### Example: Adding "GPU Latency" Benchmark

**Step 1**: Create kernel file `src/rocmGPUBenches/kernels/latency_kernels.hpp`
```cpp
extern "C" __global__ void latency_kernel(
    long* ptr,
    long size,
    long iterations
) {
    long idx = 0;
    for(long i = 0; i < iterations; i++) {
        idx = ptr[idx];
    }
    ptr[0] = idx;  // Prevent optimization
}
```

**Step 2**: Create config `src/rocmGPUBenches/benchmarks/latency_benchmark_config.hpp`
```cpp
#pragma once
#include "../framework/benchmark_runner.hpp"

inline BenchmarkConfig create_latency_benchmark_config() {
    BenchmarkConfig config;
    config.name = "latency";
    config.kernel_file = "src/rocmGPUBenches/kernels/latency_kernels.hpp";
    
    config.allocate_memory = [](const ParamMap& params) {
        return std::get<int>(params.at("array_size")) * sizeof(long);
    };
    
    config.grid_dim = [](auto&, auto) { return dim3(1); };
    config.block_dim = [](auto&, auto) { return dim3(1); };
    
    config.kernel_args = [](void* dev_ptr, const ParamMap& params) {
        static int array_size = std::get<int>(params.at("array_size"));
        static int iterations = std::get<int>(params.at("iterations"));
        return std::vector<void*>{
            (void*)&dev_ptr,
            (void*)&array_size,
            (void*)&iterations
        };
    };
    
    config.metric_name = "latency_ns";
    config.calculate_metric = [](double elapsed_ms, const ParamMap& params) {
        int iterations = std::get<int>(params.at("iterations"));
        return (elapsed_ms * 1e6) / iterations;  // nanoseconds per access
    };
    
    config.default_params = {
        {"array_size", 1024 * 1024},
        {"iterations", 10000}
    };
    
    config.iterations = 100;
    return config;
}
```

**Step 3**: Add binding in `src/rocmGPUBenches/framework/benchmark_runner_bindings.cpp`
```cpp
#include "../benchmarks/latency_benchmark_config.hpp"

// In init_benchmark_runner():
m.def("create_latency_benchmark_runner", []() {
    return std::make_shared<BenchmarkRunner>(
        create_latency_benchmark_config()
    );
}, "Create latency benchmark runner");
```

**Step 4**: Export in Python `src/rocmGPUBenches/__init__.py`
```python
create_latency_benchmark_runner = _rocmGPUBenches.create_latency_benchmark_runner

__all__.extend(['create_latency_benchmark_runner'])
```

**Step 5**: Rebuild and test
```bash
pip install -e . --no-build-isolation
python -c "from rocmGPUBenches import create_latency_benchmark_runner; print('✓')"
```

**Step 6**: Use it!
```python
from rocmGPUBenches import create_latency_benchmark_runner, BenchmarkDB

runner = create_latency_benchmark_runner()
result = runner.run('latency', array_size=1024*1024)
print(f"Latency: {result.primary_metric:.2f} ns")
```

## Design Patterns

### Parameter Types
Use `std::variant<int, double, std::string>` for flexibility:
```cpp
int size = std::get<int>(params.at("problem_size"));
double threshold = std::get<double>(params.at("threshold"));
std::string mode = std::get<std::string>(params.at("mode"));
```

### Memory Patterns
- **Simple**: Fixed size allocation
- **Dynamic**: Size based on parameters
- **Multiple buffers**: Return max of all allocations

```cpp
config.allocate_memory = [](const ParamMap& params) {
    int n = std::get<int>(params.at("n"));
    size_t input_buf = n * sizeof(float);
    size_t output_buf = n * sizeof(float);
    return std::max(input_buf, output_buf);  // Or sum for multiple
};
```

### Metric Calculations
Common patterns:
- **Bandwidth**: `(bytes / 1e9) / (time_s)` → GB/s
- **Latency**: `(time_ms * 1e6) / operations` → ns
- **Throughput**: `operations / time_s` → ops/s
- **FLOPS**: `(flops / 1e9) / time_s` → GFLOPS

## Testing Your Benchmark

1. **Single run**:
```python
result = runner.run('mybench', param1=value1)
assert result.spread_percent < 5.0  # Check stability
```

2. **Parameter sweep**:
```python
results = runner.sweep('mybench', 'param1', [1, 2, 4, 8])
assert all(r.spread_percent < 5.0 for r in results)
```

3. **Storage integration**:
```python
db = BenchmarkDB('test.db')
db.save_result('mybench', result, {'param1': value1}, gpu_info)
df = db.query(benchmark='mybench')
assert len(df) == 1
```

4. **Visualization**:
```python
from rocmGPUBenches import plot_sweep
plot_sweep(df, x='param1', y='primary_metric')
```

## Best Practices

1. **Kernel Design**:
   - Use `__restrict__` for pointers
   - Prevent dead code elimination (write results)
   - Avoid branches in hot paths
   - Use appropriate memory access patterns

2. **Parameter Naming**:
   - Use snake_case: `problem_size`, not `problemSize`
   - Be descriptive: `array_size_kb`, not `size`
   - Match gpu-benches conventions when applicable

3. **Metric Selection**:
   - Choose meaningful units (GB/s for bandwidth, ns for latency)
   - Document calculation in comments
   - Verify against known values (e.g., peak bandwidth)

4. **Iteration Counts**:
   - More iterations = more stable, but slower
   - Start with 100-1000, adjust based on spread_percent
   - Warmup iterations help with cold cache effects

5. **Error Handling**:
   - Framework handles HIP errors automatically
   - Focus on logical correctness in your config

## Common Patterns from gpu-benches

### Cache Benchmark Pattern
- Pointer chasing through array
- Vary size to hit L1/L2/L3
- Measure bandwidth at each level

### Stream Pattern
- Large contiguous transfers
- Copy, scale, add, triad operations
- Peak bandwidth measurement

### Latency Pattern
- Single-threaded pointer chase
- Power-of-2 stride increments
- Measure access latency vs stride

### Roofline Pattern
- Vary compute intensity
- Plot FLOPS vs bandwidth
- Identify bottlenecks

## Troubleshooting

**"Kernel compilation failed"**:
- Check kernel syntax (must be valid HIP/C++)
- Verify `extern "C"` for kernel functions
- Check include paths in kernel file

**"Undefined symbol in kernel"**:
- Ensure all functions are defined in the kernel file
- Don't rely on external libraries in kernel code
- Use inline functions or macros

**"Incorrect metric values"**:
- Verify `calculate_metric` formula
- Check parameter extraction (`std::get<Type>`)
- Add debug prints in lambda

**"High spread_percent"**:
- Increase `config.iterations`
- Add more `warmup_iterations`
- Check for thermal throttling

## Reference Benchmarks

Study these for patterns:
- `benchmarks/cache_benchmark_config.hpp` - Complete working example
- Original gpu-benches: `/devel/gpu-benches/gpu-*/` - Reference implementations

## Questions?

See [PROJECT-PLAN.md](../PROJECT-PLAN.md) for overall architecture decisions and rationale.
