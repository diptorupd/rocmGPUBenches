#pragma once

#include <string>

namespace rocmgpubenches {

// GPU STREAM benchmark kernels
// Tests memory bandwidth with various access patterns
const char* STREAM_KERNEL_SOURCE = R"(
#include <hip/hip_runtime.h>

extern "C" __global__ void init_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    A[tidx] = 0.23;
}

extern "C" __global__ void read_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    double temp = B[tidx];
    if (temp == 123.0)  // Never true, prevents optimization
        A[tidx] = temp;
}

extern "C" __global__ void scale_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    A[tidx] = B[tidx] * 1.2;
}

extern "C" __global__ void triad_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    A[tidx] = B[tidx] * D[tidx] + C[tidx];
}

// FIXME: Stencil kernels cause memory access faults
// Issue: Boundary conditions not properly handled, accessing out-of-bounds memory
// TODO: Fix boundary checks or add proper padding to arrays
/*
extern "C" __global__ void stencil1d3pt_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N - 1 || tidx == 0) return;
    A[tidx] = 0.5 * B[tidx - 1] - 1.0 * B[tidx] + 0.5 * B[tidx + 1];
}

extern "C" __global__ void stencil1d5pt_kernel(
    double *A, const double *__restrict__ B,
    const double *__restrict__ C, const double *__restrict__ D,
    const size_t N
) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N - 2 || tidx < 2) return;
    A[tidx] = 0.25 * B[tidx - 2] + 0.25 * B[tidx - 1] - 1.0 * B[tidx] +
              0.5 * B[tidx + 1] + 0.5 * B[tidx + 2];
}
*/
)";

} // namespace rocmgpubenches
