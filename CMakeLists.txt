cmake_minimum_required(VERSION 3.16)
project(gpu_image_pipeline VERSION 1.0 LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
# C++ Standard
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# ─────────────────────────────────────────────────────────────────────────────
# Find Dependencies
# ─────────────────────────────────────────────────────────────────────────────

# OpenCL
find_package(OpenCL REQUIRED)
message(STATUS "OpenCL Found: ${OpenCL_VERSION_STRING}")
message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")

# Python & pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 version: ${Python3_VERSION}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

# pybind11 (dapat diinstall via pip: pip install pybind11)
# Cari di beberapa lokasi umum
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# ─────────────────────────────────────────────────────────────────────────────
# Source Files
# ─────────────────────────────────────────────────────────────────────────────
set(CPP_SOURCES
    cpp/opencl_manager.cpp
    cpp/image_processor.cpp
)

set(BINDING_SOURCES
    cpp/bindings.cpp
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Library: gpu_pipeline (Python module)
# ─────────────────────────────────────────────────────────────────────────────
pybind11_add_module(gpu_pipeline
    ${CPP_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(gpu_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(gpu_pipeline PRIVATE
    ${OpenCL_LIBRARIES}
)

# Compiler flags
target_compile_options(gpu_pipeline PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -O3 -ffast-math
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /O2
    >
)

# Definisi untuk OpenCL: disable deprecated warnings
target_compile_definitions(gpu_pipeline PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)

# ─────────────────────────────────────────────────────────────────────────────
# Output directory: letakkan .so/.pyd ke folder python/
# ─────────────────────────────────────────────────────────────────────────────
set_target_properties(gpu_pipeline PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# ─────────────────────────────────────────────────────────────────────────────
# Standalone C++ benchmark executable (tanpa Python)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(benchmark_cpp
    ${CPP_SOURCES}
    cpp/benchmark_main.cpp
)

target_include_directories(benchmark_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(benchmark_cpp PRIVATE
    ${OpenCL_LIBRARIES}
)

target_compile_options(benchmark_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -O3 -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

target_compile_definitions(benchmark_cpp PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)

set_target_properties(benchmark_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────
install(TARGETS gpu_pipeline DESTINATION ${CMAKE_SOURCE_DIR}/python)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "  Module output : ${CMAKE_SOURCE_DIR}/python/")
message(STATUS "  Benchmark     : ${CMAKE_SOURCE_DIR}/benchmark_cpp")
message(STATUS "===========================")
