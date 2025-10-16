#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include "../hip_rtc_compiler.hpp"

namespace rocmgpubenches {

/**
 * Abstract base class for all GPU benchmarks
 * 
 * Provides common infrastructure:
 * - hipRTC compilation with optimization flags
 * - GPU device management
 * - One-time kernel compilation
 * - Common result structure
 * 
 * Derived classes must implement:
 * - get_kernel_source(): Return HIP kernel source as string
 * - run(): Execute benchmark with specific parameters
 * - run_sweep(): Run across range of parameters
 */
class GPUBenchmark {
public:
    struct BenchmarkResult {
        double exec_time_ms;
        double bandwidth_gbs;
        size_t data_size_kb;
        double spread_percent;
        std::string benchmark_name;
        // Derived classes can add specific fields
    };

    GPUBenchmark(const std::string& name) 
        : benchmark_name_(name), kernel_compiled_(false), module_(nullptr) {
        hipError_t err = hipGetDevice(&device_id_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipGetDevice failed");
        }
        err = hipGetDeviceProperties(&props_, device_id_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipGetDeviceProperties failed");
        }
        sm_count_ = props_.multiProcessorCount;
    }

    virtual ~GPUBenchmark() {
        cleanup();
    }

    // Pure virtual methods - must be implemented by derived classes
    virtual std::string get_kernel_source() = 0;
    
    // Common methods available to all benchmarks
    std::string get_device_name() const {
        return std::string(props_.name);
    }

    int get_sm_count() const {
        return sm_count_;
    }

    std::string get_benchmark_name() const {
        return benchmark_name_;
    }

protected:
    // Common members available to derived classes
    std::string benchmark_name_;
    int device_id_;
    hipDeviceProp_t props_;
    int sm_count_;
    
    // hipRTC compilation infrastructure
    HipRTCCompiler compiler_;
    hipModule_t module_;
    bool kernel_compiled_;
    
    // Compile kernel with standard optimization flags
    void compile_kernel(const std::string& kernel_name, 
                       hipFunction_t& kernel_func,
                       const std::vector<std::string>& extra_options = {}) {
        if (kernel_compiled_) {
            return; // Already compiled
        }
        
        std::cout << "Compiling " << benchmark_name_ 
                  << " kernel with optimizations (one-time compilation)..." << std::endl;
        
        // Get kernel source from derived class
        std::string kernel_source = get_kernel_source();
        
        // Standard optimization flags
        std::vector<std::string> compile_options = {
            "-O3",
            "-ffast-math",
            "--gpu-max-threads-per-block=1024"
        };
        
        // Add any extra options from derived class
        compile_options.insert(compile_options.end(), 
                              extra_options.begin(), 
                              extra_options.end());
        
        std::cout << "Compile flags: -O3 -ffast-math --gpu-max-threads-per-block=1024";
        for (const auto& opt : extra_options) {
            std::cout << " " << opt;
        }
        std::cout << std::endl;
        
        // Compile using hipRTC
        std::vector<char> compiled_code = compiler_.compile(kernel_source, 
                                                            kernel_name, 
                                                            compile_options);
        
        // Load module
        hipError_t err = hipModuleLoadData(&module_, compiled_code.data());
        if (err != hipSuccess) {
            throw std::runtime_error("Failed to load HIP module: " + 
                                   std::string(hipGetErrorString(err)));
        }
        
        // Get function handle
        err = hipModuleGetFunction(&kernel_func, module_, kernel_name.c_str());
        if (err != hipSuccess) {
            throw std::runtime_error("Failed to get kernel function: " + 
                                   std::string(hipGetErrorString(err)));
        }
        
        kernel_compiled_ = true;
        std::cout << "Kernel compilation complete!" << std::endl;
    }
    
    void cleanup() {
        if (module_ != nullptr) {
            hipModuleUnload(module_);
            module_ = nullptr;
        }
        kernel_compiled_ = false;
    }
};

} // namespace rocmgpubenches
