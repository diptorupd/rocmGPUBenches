#pragma once

#include <string>

namespace rocmgpubenches {

// Generate GPU cache benchmark kernel source for runtime compilation via hipRTC
// This kernel is parameterized - N, iters, and blockSize are passed as kernel arguments
// instead of being compiled into the source. This allows us to compile once and reuse
// for different parameter values.
inline std::string get_gpu_cache_kernel_source() {
    return R"(
#include <hip/hip_runtime.h>

extern "C" __global__ void initKernel(float4 *A, size_t N) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
        A[idx] = make_float4(1.1f, 1.1f, 1.1f, 1.1f);
    }
}

extern "C" __global__ void sumKernel(float4 *__restrict__ A, 
                                      const float4 *__restrict__ B,
                                      int N,
                                      int iters,
                                      int BLOCKSIZE) {
    float4 localSum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    B += threadIdx.x;

    for (int iter = 0; iter < iters; iter++) {
        float4 *B2 = (float4*)B + N;
        for (int i = 0; i < N; i += BLOCKSIZE) {
            float4 b1 = B[i];
            float4 b2 = B2[i];
            localSum.x += b1.x * b2.x;
            localSum.y += b1.y * b2.y;
            localSum.z += b1.z * b2.z;
            localSum.w += b1.w * b2.w;
        }
        localSum.x *= 1.3f;
        localSum.y *= 1.3f;
        localSum.z *= 1.3f;
        localSum.w *= 1.3f;
    }
    
    if (localSum.x == 1233.0f)
        A[threadIdx.x] = localSum;
}
)";
}

} // namespace rocmgpubenches
