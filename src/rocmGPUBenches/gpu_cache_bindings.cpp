#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpu_cache_benchmark.hpp"

namespace py = pybind11;
using namespace rocmgpubenches;

void bind_gpu_cache_benchmark(py::module &m) {
    py::class_<GPUCacheBenchmark::BenchmarkResult>(m, "GPUCacheBenchmarkResult")
        .def_readonly("exec_time_ms", &GPUCacheBenchmark::BenchmarkResult::exec_time_ms)
        .def_readonly("bandwidth_gbs", &GPUCacheBenchmark::BenchmarkResult::bandwidth_gbs)
        .def_readonly("data_size_kb", &GPUCacheBenchmark::BenchmarkResult::data_size_kb)
        .def_readonly("spread_percent", &GPUCacheBenchmark::BenchmarkResult::spread_percent)
        .def_readonly("N", &GPUCacheBenchmark::BenchmarkResult::N)
        .def_readonly("blockSize", &GPUCacheBenchmark::BenchmarkResult::blockSize)
        .def("__repr__", [](const GPUCacheBenchmark::BenchmarkResult &r) {
            return "GPUCacheBenchmarkResult(N=" + std::to_string(r.N) + 
                   ", blockSize=" + std::to_string(r.blockSize) +
                   ", data_size_kb=" + std::to_string(r.data_size_kb) + 
                   ", exec_time_ms=" + std::to_string(r.exec_time_ms) + ")";
        });
    
    py::class_<GPUCacheBenchmark>(m, "GPUCacheBenchmark")
        .def(py::init<>())
        .def("run", &GPUCacheBenchmark::run, 
             py::arg("N"), 
             py::arg("blockSize") = 256, 
             py::arg("iterations") = 15,
             "Run a single benchmark configuration")
        .def("run_sweep", &GPUCacheBenchmark::run_sweep,
             "Run full benchmark sweep across different data sizes")
        .def("get_device_name", &GPUCacheBenchmark::get_device_name)
        .def("get_sm_count", &GPUCacheBenchmark::get_sm_count);
}
