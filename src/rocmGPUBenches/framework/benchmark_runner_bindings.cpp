#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "benchmark_runner.hpp"
#include "../benchmarks/cache_benchmark_config.hpp"
#include "../benchmarks/latency_benchmark_config.hpp"

namespace py = pybind11;
using namespace rocmgpubenches;

void init_benchmark_runner(py::module_& m) {
    // Expose BenchmarkResult
    py::class_<BenchmarkRunner::BenchmarkResult>(m, "BenchmarkResult")
        .def_readonly("benchmark_name", &BenchmarkRunner::BenchmarkResult::benchmark_name)
        .def_readonly("exec_time_ms", &BenchmarkRunner::BenchmarkResult::exec_time_ms)
        .def_readonly("primary_metric", &BenchmarkRunner::BenchmarkResult::primary_metric)
        .def_readonly("metric_name", &BenchmarkRunner::BenchmarkResult::metric_name)
        .def_readonly("spread_percent", &BenchmarkRunner::BenchmarkResult::spread_percent)
        .def("__repr__", [](const BenchmarkRunner::BenchmarkResult &r) {
            return "<BenchmarkResult: " + r.benchmark_name + " = " + std::to_string(r.primary_metric) + " " + r.metric_name + ">";
        });

    // Expose BenchmarkRunner
    py::class_<BenchmarkRunner, std::shared_ptr<BenchmarkRunner>>(m, "BenchmarkRunner")
        .def(py::init<>())
        .def("get_device_name", &BenchmarkRunner::get_device_name)
        .def("get_sm_count", &BenchmarkRunner::get_sm_count)
        .def("run", [](BenchmarkRunner& self,
                       const std::string& benchmark_name,
                       const py::dict& py_params) {
            BenchmarkRunner::ParamMap params;
            for (auto item : py_params) {
                std::string key = py::cast<std::string>(item.first);
                // Try different types
                try {
                    int val = py::cast<int>(item.second);
                    params[key] = val;
                } catch (const py::cast_error&) {
                    try {
                        double val = py::cast<double>(item.second);
                        params[key] = val;
                    } catch (const py::cast_error&) {
                        try {
                            std::string val = py::cast<std::string>(item.second);
                            params[key] = val;
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("Unsupported parameter type for key: " + key);
                        }
                    }
                }
            }
            return self.run(benchmark_name, params);
        })
        .def("sweep", [](BenchmarkRunner& self,
                        const std::string& benchmark_name,
                        const std::string& sweep_param,
                        const std::vector<int>& sweep_values,
                        const py::dict& py_fixed_params) {
            BenchmarkRunner::ParamMap fixed_params;
            for (auto item : py_fixed_params) {
                std::string key = py::cast<std::string>(item.first);
                try {
                    int val = py::cast<int>(item.second);
                    fixed_params[key] = val;
                } catch (const py::cast_error&) {
                    try {
                        double val = py::cast<double>(item.second);
                        fixed_params[key] = val;
                    } catch (const py::cast_error&) {
                        try {
                            std::string val = py::cast<std::string>(item.second);
                            fixed_params[key] = val;
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("Unsupported parameter type for key: " + key);
                        }
                    }
                }
            }
            return self.sweep(benchmark_name, sweep_param, sweep_values, fixed_params);
        });

    // Factory function for cache benchmark
    m.def("create_cache_benchmark_runner", []() {
        auto runner = std::make_shared<BenchmarkRunner>();
        runner->register_benchmark(get_cache_benchmark_config());
        return runner;
    }, "Create a BenchmarkRunner configured for the cache benchmark");

    // Factory function for latency benchmark
    m.def("create_latency_benchmark_runner", []() {
        auto runner = std::make_shared<BenchmarkRunner>();
        runner->register_benchmark(get_latency_benchmark_config());
        return runner;
    }, "Create a BenchmarkRunner configured for the latency benchmark");
}
