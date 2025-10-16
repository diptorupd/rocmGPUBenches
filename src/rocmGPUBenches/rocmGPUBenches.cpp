#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void bind_hip_rtc(py::module &);
void init_benchmark_runner(py::module_ &);

PYBIND11_MODULE(rocmGPUBenches, m) {
    m.doc() = "ROCm GPU Benchmarks - Performance microbenchmarks for AMD GPUs";
    
    // Register submodules
    bind_hip_rtc(m);
    init_benchmark_runner(m);  // New flexible benchmark runner
}
