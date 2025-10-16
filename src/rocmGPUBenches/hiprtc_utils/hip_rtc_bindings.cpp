#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hip_rtc_compiler.hpp"

namespace py = pybind11;
using namespace rocmgpubenches;

void bind_hip_rtc(py::module &m) {
    py::class_<HipRTCCompiler>(m, "HipRTCCompiler")
        .def(py::init<>())
        .def("compile", &HipRTCCompiler::compile,
             py::arg("kernel_source"),
             py::arg("kernel_name"),
             py::arg("options") = std::vector<std::string>(),
             "Compile a HIP kernel from source code")
        .def("get_gpu_arch", &HipRTCCompiler::get_gpu_arch,
             "Get the current GPU architecture string");
}
