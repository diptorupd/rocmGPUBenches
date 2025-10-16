#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_main(py::module &m);
void bind_hip_rtc(py::module &m);
void bind_gpu_cache_benchmark(py::module &m);

PYBIND11_MODULE(rocmGPUBenches, m) {
    m.doc() = "ROCm GPU Benchmarks with hipRTC support";
    bind_main(m);
    bind_hip_rtc(m);
    bind_gpu_cache_benchmark(m);
}
