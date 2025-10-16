#pragma once

#include <string>

namespace rocmgpubenches {

// GPU latency benchmark kernel - pointer chasing to measure memory latency
// Uses a single thread to follow a randomized pointer chain through memory
const char* LATENCY_KERNEL_SOURCE = R"(
#include <hip/hip_runtime.h>

extern "C" __global__ void pchase_kernel(
    long long* buf,
    long long* dummy_buf,
    long long N
) {
    long long* idx = buf;
    
    const int unroll_factor = 32;
    #pragma unroll 1
    for (long long n = 0; n < N; n += unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; u++) {
            idx = (long long*)*idx;
        }
    }
    
    // Write result to prevent optimization (condition to prevent always writing)
    if (threadIdx.x + blockIdx.x * blockDim.x > 12313) {
        dummy_buf[0] = (long long)idx;
    }
}
)";

} // namespace rocmgpubenches
