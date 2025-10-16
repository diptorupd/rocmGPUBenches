#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_main(py::module &m) {
    m.def("hello", []() {
        return "Hello from rocmGPUBenches!";
    });
}
