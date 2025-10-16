#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_main(py::module &m);

PYBIND11_MODULE(rocmGPUBenches, m) {
    m.doc() = "ROCm GPU Benchmarks";
    bind_main(m);
}
