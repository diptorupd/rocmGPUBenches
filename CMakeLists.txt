cmake_minimum_required(VERSION 3.21)

# Work around old hiprtc CMake config
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(rocmGPUBenches LANGUAGES CXX)

# Find HIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to HIP installation")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP installation")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH})

# Find HIP (for runtime library only)
find_package(hip REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the Python extension module (pure C++ - no device code)
pybind11_add_module(rocmGPUBenches 
    src/rocmGPUBenches/rocmGPUBenches.cpp
    src/rocmGPUBenches/main.cpp
    src/rocmGPUBenches/hip_rtc_bindings.cpp
    src/rocmGPUBenches/gpu_cache_bindings.cpp
    src/rocmGPUBenches/gpu_cache_benchmark.cpp
)

# Link against HIP runtime and hipRTC (host libraries only)
target_link_libraries(rocmGPUBenches PRIVATE hip::host)
target_link_libraries(rocmGPUBenches PRIVATE ${HIP_PATH}/lib/libhiprtc.so)

# Add include directories
target_include_directories(rocmGPUBenches PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocmGPUBenches
    ${HIP_PATH}/include
)

# Set C++ standard
target_compile_features(rocmGPUBenches PRIVATE cxx_std_17)

# Install the module to the correct location
install(TARGETS rocmGPUBenches LIBRARY DESTINATION rocmGPUBenches)
