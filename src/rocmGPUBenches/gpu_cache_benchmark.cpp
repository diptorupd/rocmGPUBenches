#include "gpu_cache_benchmark.hpp"
#include <stdexcept>
#include <iostream>

namespace rocmgpubenches {

GPUCacheBenchmark::GPUCacheBenchmark() : kernel_compiled_(false), module_(nullptr) {
    hipError_t err = hipGetDevice(&device_id_);
    if (err != hipSuccess) throw std::runtime_error("hipGetDevice failed");
    err = hipGetDeviceProperties(&props_, device_id_);
    if (err != hipSuccess) throw std::runtime_error("hipGetDeviceProperties failed");
    sm_count_ = props_.multiProcessorCount;
    
    // Compile the kernel once during construction
    ensure_kernel_compiled();
}

GPUCacheBenchmark::~GPUCacheBenchmark() {
    cleanup();
}

void GPUCacheBenchmark::cleanup() {
    if (module_ != nullptr) {
        hipModuleUnload(module_);
        module_ = nullptr;
    }
    kernel_compiled_ = false;
}

std::string GPUCacheBenchmark::get_device_name() const {
    return std::string(props_.name);
}

int GPUCacheBenchmark::get_sm_count() const {
    return sm_count_;
}

void GPUCacheBenchmark::ensure_kernel_compiled() {
    if (kernel_compiled_) {
        return;
    }
    
    std::cout << "Compiling GPU cache kernel (one-time compilation)..." << std::endl;
    
    // Get the parameterized kernel source
    std::string kernel_source = get_gpu_cache_kernel_source();
    
    // Compile using hipRTC
    std::string kernel_name = "gpu_cache_kernel";
    std::vector<char> compiled_code = compiler_.compile(kernel_source, kernel_name);
    
    // Load module
    hipError_t err = hipModuleLoadData(&module_, compiled_code.data());
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to load HIP module: " + std::string(hipGetErrorString(err)));
    }
    
    // Get function handles
    err = hipModuleGetFunction(&init_kernel_, module_, "initKernel");
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to get initKernel function: " + std::string(hipGetErrorString(err)));
    }
    
    err = hipModuleGetFunction(&sum_kernel_, module_, "sumKernel");
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to get sumKernel function: " + std::string(hipGetErrorString(err)));
    }
    
    kernel_compiled_ = true;
    std::cout << "Kernel compilation complete!" << std::endl;
}

GPUCacheBenchmark::BenchmarkResult GPUCacheBenchmark::run(int N, int blockSize, int iterations) {
    const int iters = 1000000000 / N + 2;
    const size_t blockCount = sm_count_;
    
    // Ensure kernel is compiled (already done in constructor, but check anyway)
    ensure_kernel_compiled();
    
    // Allocate memory
    size_t bufferCount = 2 * N + 1282;
    float4 *dA, *dB;
    hipError_t err = hipMalloc(&dA, bufferCount * sizeof(float4));
    if (err != hipSuccess) throw std::runtime_error("hipMalloc failed for dA");
    err = hipMalloc(&dB, bufferCount * sizeof(float4));
    if (err != hipSuccess) throw std::runtime_error("hipMalloc failed for dB");
    
    // Initialize buffers using hipModuleLaunchKernel
    void* init_args[] = {(void*)&dA, (void*)&bufferCount};
    err = hipModuleLaunchKernel(init_kernel_, 52, 1, 1, 256, 1, 1, 0, 0, init_args, nullptr);
    if (err != hipSuccess) throw std::runtime_error("Failed to launch init kernel for dA");
    
    void* init_args_b[] = {(void*)&dB, (void*)&bufferCount};
    err = hipModuleLaunchKernel(init_kernel_, 52, 1, 1, 256, 1, 1, 0, 0, init_args_b, nullptr);
    if (err != hipSuccess) throw std::runtime_error("Failed to launch init kernel for dB");
    
    err = hipDeviceSynchronize();
    if (err != hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed after init");
    
    // Run benchmark - NOW WITH RUNTIME PARAMETERS!
    MeasurementSeries time_series;
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    if (err != hipSuccess) throw std::runtime_error("hipEventCreate failed for start");
    err = hipEventCreate(&stop);
    if (err != hipSuccess) throw std::runtime_error("hipEventCreate failed for stop");
    
    for (int i = 0; i < iterations; i++) {
        err = hipEventRecord(start);
        if (err != hipSuccess) throw std::runtime_error("hipEventRecord failed for start");
        
        // Pass N, iters, and blockSize as kernel arguments (runtime parameters!)
        void* sum_args[] = {(void*)&dA, (void*)&dB, (void*)&N, (void*)&iters, (void*)&blockSize};
        err = hipModuleLaunchKernel(sum_kernel_, blockCount, 1, 1, blockSize, 1, 1, 0, 0, sum_args, nullptr);
        if (err != hipSuccess) throw std::runtime_error("Failed to launch sum kernel");
        
        err = hipEventRecord(stop);
        if (err != hipSuccess) throw std::runtime_error("hipEventRecord failed for stop");
        err = hipEventSynchronize(stop);
        if (err != hipSuccess) throw std::runtime_error("hipEventSynchronize failed for stop");
        
        float milliseconds = 0;
        err = hipEventElapsedTime(&milliseconds, start, stop);
        if (err != hipSuccess) throw std::runtime_error("hipEventElapsedTime failed");
        time_series.add(milliseconds / 1000.0);  // Convert to seconds
    }
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(dA);
    hipFree(dB);
    
    // Calculate results
    double blockDataVolume = 2 * N * sizeof(float4);
    double bandwidth = (blockDataVolume * blockCount * iters) / time_series.minValue() / 1.0e9;
    
    BenchmarkResult result;
    result.exec_time_ms = time_series.value() * 1000.0;
    result.bandwidth_gbs = bandwidth;
    result.data_size_kb = blockDataVolume / 1024;
    result.spread_percent = time_series.spread() * 100.0;
    result.N = N;
    result.blockSize = blockSize;
    
    return result;
}

std::vector<GPUCacheBenchmark::BenchmarkResult> GPUCacheBenchmark::run_sweep() {
    std::vector<BenchmarkResult> results;
    std::vector<int> N_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    
    std::cout << "Running benchmark sweep with " << N_values.size() << " configurations..." << std::endl;
    
    for (int N : N_values) {
        try {
            std::cout << "Testing N=" << N << "..." << std::endl;
            auto result = run(N, 256, 15);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Skipping N=" << N << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    std::cout << "Sweep complete!" << std::endl;
    return results;
}

} // namespace rocmgpubenches
