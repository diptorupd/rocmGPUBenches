#include "benchmark_runner.hpp"
#include <algorithm>

namespace rocmgpubenches {

BenchmarkRunner::BenchmarkRunner() {
    hipError_t err = hipGetDevice(&device_id_);
    if (err != hipSuccess) {
        throw std::runtime_error("hipGetDevice failed");
    }
    err = hipGetDeviceProperties(&props_, device_id_);
    if (err != hipSuccess) {
        throw std::runtime_error("hipGetDeviceProperties failed");
    }
    sm_count_ = props_.multiProcessorCount;
}

BenchmarkRunner::~BenchmarkRunner() {
    cleanup();
}

void BenchmarkRunner::cleanup() {
    for (auto& [name, compiled] : compiled_cache_) {
        if (compiled.module != nullptr) {
            hipModuleUnload(compiled.module);
        }
    }
    compiled_cache_.clear();
}

std::string BenchmarkRunner::get_device_name() const {
    return std::string(props_.name);
}

int BenchmarkRunner::get_sm_count() const {
    return sm_count_;
}

void BenchmarkRunner::register_benchmark(const BenchmarkConfig& config) {
    benchmarks_[config.name] = config;
    std::cout << "Registered benchmark: " << config.name << std::endl;
}

void BenchmarkRunner::validate_params(const BenchmarkConfig& config, const ParamMap& params) {
    // Check all required params are present
    for (const auto& req_param : config.required_params) {
        if (params.find(req_param) == params.end()) {
            throw std::runtime_error("Missing required parameter: " + req_param);
        }
    }
}

BenchmarkRunner::ParamMap BenchmarkRunner::merge_params(
    const BenchmarkConfig& config, 
    const ParamMap& user_params) {
    
    ParamMap merged = config.default_params;
    
    // Override with user params
    for (const auto& [key, val] : user_params) {
        merged[key] = val;
    }
    
    return merged;
}

void BenchmarkRunner::ensure_compiled(const std::string& benchmark_name) {
    // Check if already compiled
    if (compiled_cache_.find(benchmark_name) != compiled_cache_.end()) {
        return;
    }
    
    auto it = benchmarks_.find(benchmark_name);
    if (it == benchmarks_.end()) {
        throw std::runtime_error("Benchmark '" + benchmark_name + "' not registered");
    }
    
    const auto& config = it->second;
    
    std::cout << "Compiling " << benchmark_name 
              << " kernel with optimizations (one-time compilation)..." << std::endl;
    
    // Standard optimization flags
    std::vector<std::string> compile_options = {
        "-O3",
        "-ffast-math",
        "--gpu-max-threads-per-block=1024"
    };
    
    std::cout << "Compile flags: -O3 -ffast-math --gpu-max-threads-per-block=1024" << std::endl;
    
    // Compile kernel source
    std::vector<char> compiled_code = compiler_.compile(
        config.kernel_source,
        config.name,
        compile_options
    );
    
    // Load module
    hipModule_t module;
    hipError_t err = hipModuleLoadData(&module, compiled_code.data());
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to load HIP module: " + 
                               std::string(hipGetErrorString(err)));
    }
    
    // Extract all kernel functions
    CompiledBenchmark compiled;
    compiled.module = module;
    
    for (const auto& kernel_name : config.kernel_names) {
        hipFunction_t func;
        err = hipModuleGetFunction(&func, module, kernel_name.c_str());
        if (err != hipSuccess) {
            throw std::runtime_error("Failed to get kernel function '" + kernel_name + "': " + 
                                   std::string(hipGetErrorString(err)));
        }
        compiled.kernels[kernel_name] = func;
    }
    
    compiled_cache_[benchmark_name] = compiled;
    
    std::cout << "Kernel compilation complete! Loaded " << config.kernel_names.size() 
              << " kernel functions." << std::endl;
}

BenchmarkRunner::BenchmarkResult BenchmarkRunner::run(
    const std::string& benchmark_name,
    const ParamMap& params) {
    
    auto it = benchmarks_.find(benchmark_name);
    if (it == benchmarks_.end()) {
        throw std::runtime_error("Benchmark '" + benchmark_name + "' not registered");
    }
    
    const auto& config = it->second;
    
    // Merge user params with defaults
    ParamMap merged_params = merge_params(config, params);
    
    // Validate required params
    validate_params(config, merged_params);
    
    // Ensure kernel is compiled
    ensure_compiled(benchmark_name);
    
    const auto& compiled = compiled_cache_[benchmark_name];
    
    // Get iterations parameter (common to all benchmarks)
    int iterations = get_param<int>(merged_params, "iterations", 15);
    
    // Allocate device memory according to specs
    std::map<std::string, void*> device_memory;
    for (const auto& mem_spec : config.memory_specs) {
        size_t num_elements = mem_spec.size_func(merged_params);
        size_t bytes = num_elements * mem_spec.element_size;
        
        void* ptr;
        hipError_t err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess) {
            // Cleanup any already allocated memory
            for (auto& [name, mem_ptr] : device_memory) {
                hipFree(mem_ptr);
            }
            throw std::runtime_error("hipMalloc failed for " + mem_spec.name);
        }
        device_memory[mem_spec.name] = ptr;
    }
    
    // Run benchmark with timing
    MeasurementSeries time_series;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    for (int i = 0; i < iterations; i++) {
        hipEventRecord(start);
        
        // Execute benchmark-specific launch function
        config.launch_func(merged_params, device_memory, compiled.kernels, sm_count_);
        
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        time_series.add(milliseconds / 1000.0);  // Convert to seconds
    }
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    // Calculate metric
    auto [metric_value, metric_name] = config.metric_func(
        time_series.minValue(),
        merged_params,
        device_memory,
        sm_count_
    );
    
    // Cleanup device memory
    for (auto& [name, ptr] : device_memory) {
        hipFree(ptr);
    }
    
    // Return result
    BenchmarkResult result;
    result.benchmark_name = benchmark_name;
    result.exec_time_ms = time_series.value() * 1000.0;
    result.primary_metric = metric_value;
    result.metric_name = metric_name;
    result.params = merged_params;
    result.spread_percent = time_series.spread() * 100.0;
    
    return result;
}

std::vector<BenchmarkRunner::BenchmarkResult> BenchmarkRunner::sweep(
    const std::string& benchmark_name,
    const std::string& sweep_param,
    const std::vector<int>& sweep_values,
    const ParamMap& fixed_params) {
    
    std::vector<BenchmarkResult> results;
    
    std::cout << "Running benchmark sweep with " << sweep_values.size() 
              << " configurations..." << std::endl;
    
    for (int value : sweep_values) {
        try {
            std::cout << "Testing " << sweep_param << "=" << value << "..." << std::endl;
            
            // Create params with sweep value
            ParamMap params = fixed_params;
            params[sweep_param] = value;
            
            auto result = run(benchmark_name, params);
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Skipping " << sweep_param << "=" << value 
                      << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    std::cout << "Sweep complete!" << std::endl;
    return results;
}

} // namespace rocmgpubenches
