#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <variant>
#include <functional>
#include <stdexcept>
#include <iostream>
#include "../hip_rtc_compiler.hpp"
#include "../utils/MeasurementSeries.hpp"

namespace rocmgpubenches {

/**
 * BenchmarkRunner: Configuration-based GPU benchmark framework
 * 
 * Design:
 * - Compile-time config defines benchmark (kernel, memory, metrics)
 * - Runtime params control execution (problem_size, block_size, etc.)
 * - Flexible ParamMap allows different benchmarks to have different params
 */
class BenchmarkRunner {
public:
    // Flexible parameter container - supports int, double, string
    using ParamMap = std::map<std::string, std::variant<int, double, std::string>>;
    
    struct BenchmarkResult {
        std::string benchmark_name;
        double exec_time_ms;
        double primary_metric;
        std::string metric_name;
        ParamMap params;
        double spread_percent;
    };
    
    // Configuration for a benchmark
    struct BenchmarkConfig {
        std::string name;
        std::string kernel_source;
        std::vector<std::string> kernel_names;
        
        // Parameter specification
        std::set<std::string> required_params;
        ParamMap default_params;
        
        // Memory allocation specs
        using MemorySizeFunc = std::function<size_t(const ParamMap&)>;
        struct MemorySpec {
            std::string name;
            std::string dtype;
            MemorySizeFunc size_func;
            size_t element_size;  // sizeof(float4), etc.
        };
        std::vector<MemorySpec> memory_specs;
        
        // Kernel launch function
        using LaunchFunc = std::function<void(
            const ParamMap& params,
            const std::map<std::string, void*>& device_memory,
            const std::map<std::string, hipFunction_t>& kernels,
            int sm_count
        )>;
        LaunchFunc launch_func;
        
        // Metric calculation function
        using MetricFunc = std::function<std::pair<double, std::string>(
            double time_sec,
            const ParamMap& params,
            const std::map<std::string, void*>& device_memory,
            int sm_count
        )>;
        MetricFunc metric_func;
    };
    
    // Constructor
    BenchmarkRunner();
    ~BenchmarkRunner();
    
    // Register a benchmark configuration
    void register_benchmark(const BenchmarkConfig& config);
    
    // Run a benchmark with given parameters
    BenchmarkResult run(const std::string& benchmark_name, const ParamMap& params);
    
    // Sweep over one parameter
    std::vector<BenchmarkResult> sweep(
        const std::string& benchmark_name,
        const std::string& sweep_param,
        const std::vector<int>& sweep_values,
        const ParamMap& fixed_params
    );
    
    // Device info
    std::string get_device_name() const;
    int get_sm_count() const;
    
    // Helper: Get parameter with type checking
    template<typename T>
    static T get_param(const ParamMap& params, const std::string& key) {
        auto it = params.find(key);
        if (it == params.end()) {
            throw std::runtime_error("Required parameter '" + key + "' not found");
        }
        return std::get<T>(it->second);
    }
    
    template<typename T>
    static T get_param(const ParamMap& params, const std::string& key, const T& default_val) {
        auto it = params.find(key);
        if (it == params.end()) return default_val;
        return std::get<T>(it->second);
    }

private:
    int device_id_;
    hipDeviceProp_t props_;
    int sm_count_;
    
    HipRTCCompiler compiler_;
    
    // Registered benchmarks
    std::map<std::string, BenchmarkConfig> benchmarks_;
    
    // Compiled kernel cache (key: benchmark_name)
    struct CompiledBenchmark {
        hipModule_t module;
        std::map<std::string, hipFunction_t> kernels;
    };
    std::map<std::string, CompiledBenchmark> compiled_cache_;
    
    // Internal methods
    void validate_params(const BenchmarkConfig& config, const ParamMap& params);
    ParamMap merge_params(const BenchmarkConfig& config, const ParamMap& user_params);
    void ensure_compiled(const std::string& benchmark_name);
    void cleanup();
};

} // namespace rocmgpubenches
