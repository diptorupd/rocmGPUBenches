#include "gpu_cache_benchmark.hpp"
#include <stdexcept>
#include <iostream>

namespace rocmgpubenches {

GPUCacheBenchmark::GPUCacheBenchmark() {
    hipError_t err = hipGetDevice(&device_id_);
    if (err != hipSuccess) throw std::runtime_error("hipGetDevice failed");
    err = hipGetDeviceProperties(&props_, device_id_);
    if (err != hipSuccess) throw std::runtime_error("hipGetDeviceProperties failed");
    sm_count_ = props_.multiProcessorCount;
}

GPUCacheBenchmark::~GPUCacheBenchmark() {
    cleanup();
}

void GPUCacheBenchmark::cleanup() {
    // Unload all cached modules
    for (auto& pair : module_cache_) {
        hipModuleUnload(pair.second);
    }
    module_cache_.clear();
    function_cache_.clear();
    kernel_cache_.clear();
}

std::string GPUCacheBenchmark::get_device_name() const {
    return std::string(props_.name);
}

int GPUCacheBenchmark::get_sm_count() const {
    return sm_count_;
}

std::string GPUCacheBenchmark::make_kernel_key(int N, int iters, int blockSize) {
    return std::to_string(N) + "_" + std::to_string(iters) + "_" + std::to_string(blockSize);
}

void GPUCacheBenchmark::compile_and_load_kernel(int N, int iters, int blockSize) {
    std::string key = make_kernel_key(N, iters, blockSize);
    
    // Check if already compiled and loaded
    if (function_cache_.find(key) != function_cache_.end()) {
        return;
    }
    
    // Generate kernel source
    std::string kernel_source = generate_gpu_cache_kernel_source(N, iters, blockSize);
    
    // Compile using hipRTC
    std::string kernel_name = "gpu_cache_kernel";
    std::vector<char> compiled_code = compiler_.compile(kernel_source, kernel_name);
    kernel_cache_[key] = compiled_code;
    
    // Load module
    hipModule_t module;
    hipError_t err = hipModuleLoadData(&module, compiled_code.data());
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to load HIP module: " + std::string(hipGetErrorString(err)));
    }
    module_cache_[key] = module;
    
    // Get function handles
    hipFunction_t init_func, sum_func;
    err = hipModuleGetFunction(&init_func, module, "initKernel");
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to get initKernel function: " + std::string(hipGetErrorString(err)));
    }
    
    err = hipModuleGetFunction(&sum_func, module, "sumKernel");
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to get sumKernel function: " + std::string(hipGetErrorString(err)));
    }
    
    // Store function handles (we'll use "init_" and "sum_" prefixes)
    function_cache_["init_" + key] = init_func;
    function_cache_["sum_" + key] = sum_func;
}

GPUCacheBenchmark::BenchmarkResult GPUCacheBenchmark::run(int N, int blockSize, int iterations) {
    const size_t iters = 1000000000 / N + 2;
    const size_t blockCount = sm_count_;
    
    // Compile and load kernel if not cached
    compile_and_load_kernel(N, iters, blockSize);
    
    std::string key = make_kernel_key(N, iters, blockSize);
    hipFunction_t init_kernel = function_cache_["init_" + key];
    hipFunction_t sum_kernel = function_cache_["sum_" + key];
    
    // Allocate memory
    size_t bufferCount = 2 * N + 1282;
    float4 *dA, *dB;
    hipError_t err = hipMalloc(&dA, bufferCount * sizeof(float4));
    if (err != hipSuccess) throw std::runtime_error("hipMalloc failed for dA");
    err = hipMalloc(&dB, bufferCount * sizeof(float4));
    if (err != hipSuccess) throw std::runtime_error("hipMalloc failed for dB");
    
    // Initialize buffers using hipModuleLaunchKernel
    void* init_args[] = {(void*)&dA, (void*)&bufferCount};
    err = hipModuleLaunchKernel(init_kernel, 52, 1, 1, 256, 1, 1, 0, 0, init_args, nullptr);
    if (err != hipSuccess) throw std::runtime_error("Failed to launch init kernel for dA");
    
    void* init_args_b[] = {(void*)&dB, (void*)&bufferCount};
    err = hipModuleLaunchKernel(init_kernel, 52, 1, 1, 256, 1, 1, 0, 0, init_args_b, nullptr);
    if (err != hipSuccess) throw std::runtime_error("Failed to launch init kernel for dB");
    
    err = hipDeviceSynchronize();
    if (err != hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed after init");
    
    // Run benchmark
    MeasurementSeries time_series;
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    if (err != hipSuccess) throw std::runtime_error("hipEventCreate failed for start");
    err = hipEventCreate(&stop);
    if (err != hipSuccess) throw std::runtime_error("hipEventCreate failed for stop");
    
    for (int i = 0; i < iterations; i++) {
        err = hipEventRecord(start);
        if (err != hipSuccess) throw std::runtime_error("hipEventRecord failed for start");
        
        void* sum_args[] = {(void*)&dA, (void*)&dB};
        err = hipModuleLaunchKernel(sum_kernel, blockCount, 1, 1, blockSize, 1, 1, 0, 0, sum_args, nullptr);
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
    
    for (int N : N_values) {
        try {
            auto result = run(N, 256, 15);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Skipping N=" << N << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    return results;
}

} // namespace rocmgpubenches
