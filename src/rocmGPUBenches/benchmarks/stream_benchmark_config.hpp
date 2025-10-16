#pragma once

#include "../framework/benchmark_runner.hpp"
#include "../kernels/stream_kernels.hpp"
#include <hip/hip_runtime.h>
#include <map>
#include <string>

namespace rocmgpubenches {

/**
 * Configuration for GPU STREAM Benchmark
 *
 * Classic STREAM benchmark measuring memory bandwidth with different access patterns.
 * Based on McCalpin's STREAM benchmark adapted for GPUs.
 * 
 * Supported kernel types:
 * - init:   A = scalar (write-only, 1 stream)
 * - read:   temp = B (read-only, 1 stream)
 * - scale:  A = α·B (2 streams)
 * - triad:  A = B·D + C (4 streams - classic STREAM)
 * 
 * FIXME: stencil3pt and stencil5pt kernels disabled due to memory access faults
 */
inline BenchmarkRunner::BenchmarkConfig get_stream_benchmark_config() {
    using ParamMap = BenchmarkRunner::ParamMap;
    using BR = BenchmarkRunner;

    BenchmarkRunner::BenchmarkConfig config;

    config.name = "stream";
    config.kernel_source = STREAM_KERNEL_SOURCE;
    config.kernel_names = {"init_kernel", "read_kernel", "scale_kernel", "triad_kernel"};

    // Required: problem_size (number of doubles)
    config.required_params = {"problem_size"};

    // Defaults
    config.default_params = {
        {"kernel_type", std::string("triad")},  // Which STREAM kernel to run
        {"block_size", 256}                      // Threads per block
    };

    // Memory specs - 4 arrays for full STREAM triad: A = B * D + C
    config.memory_specs = {
        {
            .name = "A",
            .dtype = "double",
            .size_func = [](const ParamMap& p) -> size_t {
                return BR::get_param<int>(p, "problem_size");
            },
            .element_size = sizeof(double)
        },
        {
            .name = "B",
            .dtype = "double",
            .size_func = [](const ParamMap& p) -> size_t {
                return BR::get_param<int>(p, "problem_size");
            },
            .element_size = sizeof(double)
        },
        {
            .name = "C",
            .dtype = "double",
            .size_func = [](const ParamMap& p) -> size_t {
                return BR::get_param<int>(p, "problem_size");
            },
            .element_size = sizeof(double)
        },
        {
            .name = "D",
            .dtype = "double",
            .size_func = [](const ParamMap& p) -> size_t {
                return BR::get_param<int>(p, "problem_size");
            },
            .element_size = sizeof(double)
        }
    };

    // Launch function
    config.launch_func = [](const ParamMap& params,
                           const std::map<std::string, void*>& device_memory,
                           const std::map<std::string, hipFunction_t>& kernels,
                           int sm_count) {

        int N = BR::get_param<int>(params, "problem_size");
        int block_size = BR::get_param<int>(params, "block_size");
        std::string kernel_type = BR::get_param<std::string>(params, "kernel_type");

        double* dA = static_cast<double*>(device_memory.at("A"));
        double* dB = static_cast<double*>(device_memory.at("B"));
        double* dC = static_cast<double*>(device_memory.at("C"));
        double* dD = static_cast<double*>(device_memory.at("D"));

        int grid_size = (N + block_size - 1) / block_size;

        // Initialize arrays first
        hipFunction_t init = kernels.at("init_kernel");
        void* init_args[] = {(void*)&dA, (void*)&dB, (void*)&dC, (void*)&dD, (void*)&N};
        hipModuleLaunchKernel(init, grid_size, 1, 1, block_size, 1, 1, 0, 0, init_args, nullptr);
        hipModuleLaunchKernel(init, grid_size, 1, 1, block_size, 1, 1, 0, 0, init_args, nullptr);
        hipModuleLaunchKernel(init, grid_size, 1, 1, block_size, 1, 1, 0, 0, init_args, nullptr);
        hipModuleLaunchKernel(init, grid_size, 1, 1, block_size, 1, 1, 0, 0, init_args, nullptr);
        hipDeviceSynchronize();

        // Map kernel_type string to actual kernel
        std::map<std::string, std::string> kernel_map = {
            {"init", "init_kernel"},
            {"read", "read_kernel"},
            {"scale", "scale_kernel"},
            {"triad", "triad_kernel"}
        };

        hipFunction_t kernel = kernels.at(kernel_map[kernel_type]);
        void* kernel_args[] = {(void*)&dA, (void*)&dB, (void*)&dC, (void*)&dD, (void*)&N};
        hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, kernel_args, nullptr);
    };

    // Metric calculation - compute bandwidth based on kernel type
    config.metric_func = [](double time_sec,
                           const ParamMap& params,
                           const std::map<std::string, void*>& /*device_memory*/,
                           int /*sm_count*/) -> std::pair<double, std::string> {

        int N = BR::get_param<int>(params, "problem_size");
        std::string kernel_type = BR::get_param<std::string>(params, "kernel_type");

        // Number of memory streams (read + write) per kernel type
        std::map<std::string, int> stream_counts = {
            {"init", 1},   // 1 write
            {"read", 1},   // 1 read
            {"scale", 2},  // 1 read + 1 write
            {"triad", 4}   // 3 reads + 1 write
        };

        int stream_count = stream_counts[kernel_type];
        double bytes_transferred = stream_count * N * sizeof(double);
        double bandwidth_gbs = bytes_transferred / time_sec / 1.0e9;

        return {bandwidth_gbs, "bandwidth_gbs"};
    };

    return config;
}

} // namespace rocmgpubenches
