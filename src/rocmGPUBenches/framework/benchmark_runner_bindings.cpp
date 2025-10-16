#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "benchmark_runner.hpp"
#include "../benchmarks/cache_benchmark_config.hpp"

namespace py = pybind11;
using namespace rocmgpubenches;

void init_benchmark_runner(py::module_ &m) {
    // BenchmarkResult
    py::class_<BenchmarkRunner::BenchmarkResult>(m, "BenchmarkResult")
        .def_readonly("benchmark_name", &BenchmarkRunner::BenchmarkResult::benchmark_name)
        .def_readonly("exec_time_ms", &BenchmarkRunner::BenchmarkResult::exec_time_ms)
        .def_readonly("primary_metric", &BenchmarkRunner::BenchmarkResult::primary_metric)
        .def_readonly("metric_name", &BenchmarkRunner::BenchmarkResult::metric_name)
        .def_readonly("spread_percent", &BenchmarkRunner::BenchmarkResult::spread_percent)
        .def("__repr__", [](const BenchmarkRunner::BenchmarkResult &r) {
            return "<BenchmarkResult " + r.benchmark_name + 
                   ": " + std::to_string(r.primary_metric) + " " + r.metric_name + ">";
        });
    
    // BenchmarkRunner
    py::class_<BenchmarkRunner, std::shared_ptr<BenchmarkRunner>>(m, "BenchmarkRunner")
        .def(py::init<>())
        .def("get_device_name", &BenchmarkRunner::get_device_name)
        .def("get_sm_count", &BenchmarkRunner::get_sm_count)
        .def("run", [](BenchmarkRunner& self, 
                      const std::string& benchmark_name,
                      py::kwargs kwargs) {
            // Convert Python kwargs to ParamMap
            BenchmarkRunner::ParamMap params;
            for (auto item : kwargs) {
                std::string key = py::str(item.first);
                
                // Try to convert to int first
                try {
                    int val = py::cast<int>(item.second);
                    params[key] = val;
                    continue;
                } catch (...) {}
                
                // Try double
                try {
                    double val = py::cast<double>(item.second);
                    params[key] = val;
                    continue;
                } catch (...) {}
                
                // Fall back to string
                try {
                    std::string val = py::cast<std::string>(item.second);
                    params[key] = val;
                } catch (...) {
                    throw std::runtime_error("Unsupported parameter type for key: " + key);
                }
            }
            
            return self.run(benchmark_name, params);
        }, py::arg("benchmark_name"),
           "Run a benchmark with given parameters as keyword arguments")
        .def("sweep", [](BenchmarkRunner& self,
                        const std::string& benchmark_name,
                        const std::string& sweep_param,
                        const std::vector<int>& sweep_values,
                        py::kwargs kwargs) {
            // Convert kwargs to fixed params
            BenchmarkRunner::ParamMap fixed_params;
            for (auto item : kwargs) {
                std::string key = py::str(item.first);
                
                try {
                    int val = py::cast<int>(item.second);
                    fixed_params[key] = val;
                    continue;
                } catch (...) {}
                
                try {
                    double val = py::cast<double>(item.second);
                    fixed_params[key] = val;
                    continue;
                } catch (...) {}
                
                try {
                    std::string val = py::cast<std::string>(item.second);
                    fixed_params[key] = val;
                } catch (...) {}
            }
            
            return self.sweep(benchmark_name, sweep_param, sweep_values, fixed_params);
        }, py::arg("benchmark_name"),
           py::arg("sweep_param"),
           py::arg("sweep_values"),
           "Sweep over a parameter with fixed values for other parameters");
    
    // Helper function to create and register cache benchmark
    m.def("create_cache_benchmark_runner", []() {
        auto runner = std::make_shared<BenchmarkRunner>();
        runner->register_benchmark(get_cache_benchmark_config());
        return runner;
    }, "Create a BenchmarkRunner with cache benchmark pre-registered");
}
