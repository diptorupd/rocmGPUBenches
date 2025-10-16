#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
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
    
    // Single compiled module and functions (no per-parameter caching needed)
    hipModule_t module_;
    hipFunction_t init_kernel_;
    hipFunction_t sum_kernel_;
    bool kernel_compiled_;
    
    void cleanup();
    void ensure_kernel_compiled();
};

} // namespace rocmgpubenches
