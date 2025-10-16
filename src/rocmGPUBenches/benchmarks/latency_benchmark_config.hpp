#pragma once

#include "../framework/benchmark_runner.hpp"
#include "../kernels/latency_kernels.hpp"
#include <hip/hip_runtime.h>
#include <random>
#include <algorithm>
#include <cstring>

namespace rocmgpubenches {

/**
 * Configuration for GPU Latency Benchmark
 *
 * This benchmark measures memory access latency using pointer chasing.
 * 
 * FIXME: Current measurements are ~24x lower than expected!
 * Expected (MI300X @ 2092 MHz): L1=~120 cycles, L2=~213 cycles, HBM=~456 cycles
 * Actual measurements: ~5-6 cycles for L1-sized buffers
 * 
 * Possible issues:
 * - Compiler optimizing away loop iterations
 * - Pointer chain setup not creating true dependency chain
 * - Need warmup runs and multiple measurements
 * - Kernel may not execute full iteration count
 * 
 * TODO: Debug by comparing with original gpu-latency benchmark
 */
inline BenchmarkRunner::BenchmarkConfig get_latency_benchmark_config() {
    using ParamMap = BenchmarkRunner::ParamMap;
    using BR = BenchmarkRunner;

    BenchmarkRunner::BenchmarkConfig config;

    config.name = "latency";
    config.kernel_source = LATENCY_KERNEL_SOURCE;
    config.kernel_names = {"pchase_kernel"};

    // Required: problem_size (chain length)
    config.required_params = {"problem_size"};

    // Defaults
    config.default_params = {
        {"iterations", 100000}  // Number of pointer chase iterations
    };

    // Memory specs - need buffer for pointer chain and dummy output
    config.memory_specs = {
        {
            .name = "chain",
            .dtype = "int64_t",
            .size_func = [](const ParamMap& p) -> size_t {
                int problem_size = BR::get_param<int>(p, "problem_size");
                return problem_size;
            },
            .element_size = sizeof(int64_t)
        },
        {
            .name = "dummy",
            .dtype = "int64_t",
            .size_func = [](const ParamMap&) -> size_t {
                return 1;  // Just one element for dummy output
            },
            .element_size = sizeof(int64_t)
        }
    };

    // Launch function - sets up pointer chain and runs kernel
    config.launch_func = [](const ParamMap& params,
                           const std::map<std::string, void*>& device_memory,
                           const std::map<std::string, hipFunction_t>& kernels,
                           int /*sm_count*/) {

        int problem_size = BR::get_param<int>(params, "problem_size");
        int iterations = BR::get_param<int>(params, "iterations");

        int64_t* d_chain = static_cast<int64_t*>(device_memory.at("chain"));
        int64_t* d_dummy = static_cast<int64_t*>(device_memory.at("dummy"));

        // Create randomized pointer chain on host
        std::vector<int64_t> h_chain(problem_size);

        // Create indices and shuffle
        std::vector<int> indices(problem_size);
        for (int i = 0; i < problem_size; i++) {
            indices[i] = i;
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Create circular pointer chain with cache line spreading
        const int cache_line_spread = 16;  // 128 bytes / 8 bytes per int64
        for (size_t i = 0; i < indices.size() - 1; i++) {
            int next_idx = (indices[i] + cache_line_spread) % problem_size;
            h_chain[indices[i]] = reinterpret_cast<int64_t>(&d_chain[next_idx]);
        }
        h_chain[indices.back()] = reinterpret_cast<int64_t>(&d_chain[indices[0]]);

        // Copy to device
        hipMemcpy(d_chain, h_chain.data(), problem_size * sizeof(int64_t), hipMemcpyHostToDevice);

        // Launch kernel - single thread
        hipFunction_t kernel = kernels.at("pchase_kernel");
        void* args[] = {(void*)&d_chain, (void*)&d_dummy, (void*)&iterations};
        hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, nullptr);
    };

    // Metric calculation - convert time to latency in cycles
    config.metric_func = [](double time_sec,
                           const ParamMap& params,
                           const std::map<std::string, void*>& /*device_memory*/,
                           int /*sm_count*/) -> std::pair<double, std::string> {

        // Clock frequency verified from MI300X reference: 2092 MHz
        const double gpu_freq_ghz = 2.092;

        int iterations = BR::get_param<int>(params, "iterations");
        int problem_size = BR::get_param<int>(params, "problem_size");

        // Total cycles for all accesses
        double total_cycles = time_sec * gpu_freq_ghz * 1e9;

        // Average cycles per memory access
        double cycles_per_access = total_cycles / (iterations * problem_size);

        return {cycles_per_access, "latency_cycles"};
    };

    return config;
}

} // namespace rocmgpubenches
