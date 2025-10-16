#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <map>
#include "hip_rtc_compiler.hpp"
#include "utils/MeasurementSeries.hpp"
#include "utils/gpu-error.h"
#include "gpu_cache_kernel_source.hpp"

namespace rocmgpubenches {

class GPUCacheBenchmark {
public:
    struct BenchmarkResult {
        double exec_time_ms;
        double bandwidth_gbs;
        size_t data_size_kb;
        double spread_percent;
        int N;
        int blockSize;
    };

    GPUCacheBenchmark();
    ~GPUCacheBenchmark();

    // Run a single benchmark configuration
    BenchmarkResult run(int N, int blockSize = 256, int iterations = 15);
    
    // Run full benchmark sweep
    std::vector<BenchmarkResult> run_sweep();

    std::string get_device_name() const;
    int get_sm_count() const;

private:
    int device_id_;
    hipDeviceProp_t props_;
    int sm_count_;
    
    // HipRTC compiler instance
    HipRTCCompiler compiler_;
    
    // Cache for compiled kernels (key: "N_iters_blockSize")
    std::map<std::string, std::vector<char>> kernel_cache_;
    std::map<std::string, hipModule_t> module_cache_;
    std::map<std::string, hipFunction_t> function_cache_;
    
    void cleanup();
    void compile_and_load_kernel(int N, int iters, int blockSize);
    std::string make_kernel_key(int N, int iters, int blockSize);
};

} // namespace rocmgpubenches
