cmake_minimum_required(VERSION 3.18)
project(rocmGPUBenches LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# Create the Python extension module
pybind11_add_module(rocmGPUBenches 
    src/rocmGPUBenches/rocmGPUBenches.cpp
    src/rocmGPUBenches/main.cpp
)

# Install the module to the correct location
install(TARGETS rocmGPUBenches LIBRARY DESTINATION rocmGPUBenches)
