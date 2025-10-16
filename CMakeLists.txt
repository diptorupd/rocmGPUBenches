cmake_minimum_required(VERSION 3.21)
project(rocmGPUBenches LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Get HIP installation path
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to HIP installation")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP installation")
    endif()
endif()

# Create the Python module
pybind11_add_module(rocmGPUBenches 
    src/rocmGPUBenches/main.cpp
    src/rocmGPUBenches/rocmGPUBenches.cpp
    src/rocmGPUBenches/hiprtc_utils/hip_rtc_bindings.cpp
    src/rocmGPUBenches/framework/benchmark_runner.cpp
    src/rocmGPUBenches/framework/benchmark_runner_bindings.cpp
)

# Include directories
target_include_directories(rocmGPUBenches PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocmGPUBenches
    ${HIP_PATH}/include
)

# Link libraries directly
target_link_libraries(rocmGPUBenches PRIVATE 
    ${HIP_PATH}/lib/libamdhip64.so
    ${HIP_PATH}/lib/libhiprtc.so
)

# Set compile definitions
target_compile_definitions(rocmGPUBenches PRIVATE 
    USE_PROF_API=1
    __HIP_PLATFORM_AMD__=1
)

# Installation
install(TARGETS rocmGPUBenches LIBRARY DESTINATION .)
