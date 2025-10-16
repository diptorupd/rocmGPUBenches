#pragma once

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

namespace rocmgpubenches {

class HipRTCCompiler {
public:
    HipRTCCompiler() = default;
    ~HipRTCCompiler() = default;

    // Compile a HIP kernel from source code
    std::vector<char> compile(const std::string& kernel_source, 
                              const std::string& kernel_name,
                              const std::vector<std::string>& options = {}) {
        hiprtcProgram prog;
        
        // Create the program
        hiprtcResult result = hiprtcCreateProgram(
            &prog, 
            kernel_source.c_str(), 
            kernel_name.c_str(), 
            0, 
            nullptr, 
            nullptr
        );
        
        if (result != HIPRTC_SUCCESS) {
            throw std::runtime_error("Failed to create hipRTC program: " + 
                                   std::string(hiprtcGetErrorString(result)));
        }

        // Convert options to C-style array
        std::vector<const char*> opt_ptrs;
        for (const auto& opt : options) {
            opt_ptrs.push_back(opt.c_str());
        }

        // Compile the program
        result = hiprtcCompileProgram(prog, opt_ptrs.size(), opt_ptrs.data());
        
        if (result != HIPRTC_SUCCESS) {
            // Get compilation log
            size_t log_size;
            hiprtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size);
            hiprtcGetProgramLog(prog, log.data());
            
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("Compilation failed:\n" + std::string(log.data()));
        }

        // Get compiled code size
        size_t code_size;
        result = hiprtcGetCodeSize(prog, &code_size);
        if (result != HIPRTC_SUCCESS) {
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("Failed to get code size");
        }

        // Get compiled code
        std::vector<char> code(code_size);
        result = hiprtcGetCode(prog, code.data());
        if (result != HIPRTC_SUCCESS) {
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("Failed to get compiled code");
        }

        hiprtcDestroyProgram(&prog);
        return code;
    }

    // Load compiled code as a module and get kernel function
    hipFunction_t load_kernel(const std::vector<char>& code, 
                             const std::string& kernel_name) {
        hipModule_t module;
        hipError_t hip_result = hipModuleLoadData(&module, code.data());
        if (hip_result != hipSuccess) {
            throw std::runtime_error("Failed to load module: " + 
                                   std::string(hipGetErrorString(hip_result)));
        }

        hipFunction_t kernel;
        hip_result = hipModuleGetFunction(&kernel, module, kernel_name.c_str());
        if (hip_result != hipSuccess) {
            throw std::runtime_error("Failed to get kernel function: " + 
                                   std::string(hipGetErrorString(hip_result)));
        }

        // Store module for cleanup (in real implementation, manage lifecycle)
        modules_.push_back(module);
        
        return kernel;
    }

    // Get current GPU architecture
    std::string get_gpu_arch() {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, 0);
        return std::string(props.gcnArchName);
    }

private:
    std::vector<hipModule_t> modules_;
};

} // namespace rocmgpubenches
