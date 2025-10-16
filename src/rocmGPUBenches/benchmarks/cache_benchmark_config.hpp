#pragma once

#include "benchmark_runner.hpp"
#include "../kernels/cache_kernels.hpp"

namespace rocmgpubenches {

/**
 * Configuration for GPU Cache Benchmark
 * 
 * This benchmark measures cache/memory hierarchy performance by
 * repeatedly accessing data with varying problem sizes.
 */
inline BenchmarkRunner::BenchmarkConfig get_cache_benchmark_config() {
    using ParamMap = BenchmarkRunner::ParamMap;
    using BR = BenchmarkRunner;
    
    BenchmarkRunner::BenchmarkConfig config;
    
    config.name = "cache";
    config.kernel_source = get_gpu_cache_kernel_source();
    config.kernel_names = {"initKernel", "sumKernel"};
    
    // Required: problem_size
    config.required_params = {"problem_size"};
    
    // Defaults
    config.default_params = {
        {"block_size", 256},
        {"iterations", 15}
    };
    
    // Memory specs
    config.memory_specs = {
        {
            .name = "dA",
            .dtype = "float4",
            .size_func = [](const ParamMap& p) -> size_t {
                int N = BR::get_param<int>(p, "problem_size");
                return 2 * N + 1282;
            },
            .element_size = sizeof(float4)
        },
        {
            .name = "dB",
            .dtype = "float4",
            .size_func = [](const ParamMap& p) -> size_t {
                int N = BR::get_param<int>(p, "problem_size");
                return 2 * N + 1282;
            },
            .element_size = sizeof(float4)
        }
    };
    
    // Launch function
    config.launch_func = [](const ParamMap& params,
                           const std::map<std::string, void*>& device_memory,
                           const std::map<std::string, hipFunction_t>& kernels,
                           int sm_count) {
        
        int N = BR::get_param<int>(params, "problem_size");
        int block_size = BR::get_param<int>(params, "block_size");
        int iters = 1000000000 / N + 2;
        
        float4* dA = static_cast<float4*>(device_memory.at("dA"));
        float4* dB = static_cast<float4*>(device_memory.at("dB"));
        
        size_t bufferCount = 2 * N + 1282;
        
        // Initialize buffers
        hipFunction_t init_kernel = kernels.at("initKernel");
        void* init_args_a[] = {(void*)&dA, (void*)&bufferCount};
        hipModuleLaunchKernel(init_kernel, 52, 1, 1, 256, 1, 1, 0, 0, init_args_a, nullptr);
        
        void* init_args_b[] = {(void*)&dB, (void*)&bufferCount};
        hipModuleLaunchKernel(init_kernel, 52, 1, 1, 256, 1, 1, 0, 0, init_args_b, nullptr);
        
        hipDeviceSynchronize();
        
        // Run sum kernel with runtime parameters
        hipFunction_t sum_kernel = kernels.at("sumKernel");
        void* sum_args[] = {(void*)&dA, (void*)&dB, (void*)&N, (void*)&iters, (void*)&block_size};
        hipModuleLaunchKernel(sum_kernel, sm_count, 1, 1, block_size, 1, 1, 0, 0, sum_args, nullptr);
    };
    
    // Metric calculation
    config.metric_func = [](double time_sec,
                           const ParamMap& params,
                           const std::map<std::string, void*>& device_memory,
                           int sm_count) -> std::pair<double, std::string> {
        
        int N = BR::get_param<int>(params, "problem_size");
        int iters = 1000000000 / N + 2;
        
        double blockDataVolume = 2 * N * sizeof(float4);
        double bandwidth = (blockDataVolume * sm_count * iters) / time_sec / 1.0e9;
        
        return {bandwidth, "bandwidth_gbs"};
    };
    
    return config;
}

} // namespace rocmgpubenches
