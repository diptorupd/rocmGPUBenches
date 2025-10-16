#include <hip/hip_runtime.h>

using dtype = float4;

// Initialization kernel
__global__ void initKernel(dtype *A, size_t N) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
        A[idx] = make_float4(1.1f, 1.1f, 1.1f, 1.1f);
    }
}

// Cache benchmark kernel  
template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B) {
    dtype localSum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    B += threadIdx.x;

    for (int iter = 0; iter < iters; iter++) {
        auto B2 = B + N;
        #pragma unroll 8
        for (int i = 0; i < N; i += BLOCKSIZE) {
            dtype b1 = B[i];
            dtype b2 = B2[i];
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

// Explicit template instantiations for common sizes and block sizes
// N=1 to 16384, blockSize=32, 64, 128, 256
#define INSTANTIATE_KERNEL(N, BS) \
    template __global__ void sumKernel<N, (1000000000 / N + 2), BS>(dtype *__restrict__, const dtype *__restrict__);

// Block size 32
INSTANTIATE_KERNEL(1, 32)
INSTANTIATE_KERNEL(2, 32)
INSTANTIATE_KERNEL(4, 32)
INSTANTIATE_KERNEL(8, 32)
INSTANTIATE_KERNEL(16, 32)
INSTANTIATE_KERNEL(32, 32)
INSTANTIATE_KERNEL(64, 32)
INSTANTIATE_KERNEL(128, 32)
INSTANTIATE_KERNEL(256, 32)
INSTANTIATE_KERNEL(512, 32)
INSTANTIATE_KERNEL(1024, 32)
INSTANTIATE_KERNEL(2048, 32)
INSTANTIATE_KERNEL(4096, 32)
INSTANTIATE_KERNEL(8192, 32)
INSTANTIATE_KERNEL(16384, 32)

// Block size 64
INSTANTIATE_KERNEL(1, 64)
INSTANTIATE_KERNEL(2, 64)
INSTANTIATE_KERNEL(4, 64)
INSTANTIATE_KERNEL(8, 64)
INSTANTIATE_KERNEL(16, 64)
INSTANTIATE_KERNEL(32, 64)
INSTANTIATE_KERNEL(64, 64)
INSTANTIATE_KERNEL(128, 64)
INSTANTIATE_KERNEL(256, 64)
INSTANTIATE_KERNEL(512, 64)
INSTANTIATE_KERNEL(1024, 64)
INSTANTIATE_KERNEL(2048, 64)
INSTANTIATE_KERNEL(4096, 64)
INSTANTIATE_KERNEL(8192, 64)
INSTANTIATE_KERNEL(16384, 64)

// Block size 128
INSTANTIATE_KERNEL(1, 128)
INSTANTIATE_KERNEL(2, 128)
INSTANTIATE_KERNEL(4, 128)
INSTANTIATE_KERNEL(8, 128)
INSTANTIATE_KERNEL(16, 128)
INSTANTIATE_KERNEL(32, 128)
INSTANTIATE_KERNEL(64, 128)
INSTANTIATE_KERNEL(128, 128)
INSTANTIATE_KERNEL(256, 128)
INSTANTIATE_KERNEL(512, 128)
INSTANTIATE_KERNEL(1024, 128)
INSTANTIATE_KERNEL(2048, 128)
INSTANTIATE_KERNEL(4096, 128)
INSTANTIATE_KERNEL(8192, 128)
INSTANTIATE_KERNEL(16384, 128)

// Block size 256
INSTANTIATE_KERNEL(1, 256)
INSTANTIATE_KERNEL(2, 256)
INSTANTIATE_KERNEL(4, 256)
INSTANTIATE_KERNEL(8, 256)
INSTANTIATE_KERNEL(16, 256)
INSTANTIATE_KERNEL(32, 256)
INSTANTIATE_KERNEL(64, 256)
INSTANTIATE_KERNEL(128, 256)
INSTANTIATE_KERNEL(256, 256)
INSTANTIATE_KERNEL(512, 256)
INSTANTIATE_KERNEL(1024, 256)
INSTANTIATE_KERNEL(2048, 256)
INSTANTIATE_KERNEL(4096, 256)
INSTANTIATE_KERNEL(8192, 256)
INSTANTIATE_KERNEL(16384, 256)
