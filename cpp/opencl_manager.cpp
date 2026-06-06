/**
 * @file opencl_manager.cpp
 * @brief Implementasi OpenCL Lifecycle Manager
 *
 * Mengimplementasikan seluruh OpenCL dari scratch tanpa framework wrapper:
 *   - clGetPlatformIDs, clGetDeviceIDs
 *   - clCreateContext, clCreateCommandQueue
 *   - clCreateProgramWithSource, clBuildProgram
 *   - clCreateKernel
 *   - clCreateBuffer, clEnqueueWriteBuffer, clEnqueueReadBuffer
 *   - clEnqueueNDRangeKernel
 *   - clFinish, clReleaseXxx
 */

#include "opencl_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ocl {

OpenCLManager::OpenCLManager() = default;

OpenCLManager::~OpenCLManager() {
    // Release semua kernel
    for (auto& [id, k] : kernels_) {
        if (k) clReleaseKernel(k);
    }
    // Release semua program
    for (auto& [name, p] : programs_) {
        if (p) clReleaseProgram(p);
    }
    if (queue_)   clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    // Device dan platform tidak perlu di-release secara eksplisit
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialize: Platform → Device → Context → Command Queue
// ─────────────────────────────────────────────────────────────────────────────
void OpenCLManager::initialize() {
    cl_int err;

    // ── 1. Platform Discovery ──────────────────────────────────────────────
    // Cari semua OpenCL platform (Intel, NVIDIA, AMD, Apple, dll)
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0)
        throw std::runtime_error("Tidak ada OpenCL platform yang ditemukan.");

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
    platform_ = platforms[0];  // Gunakan platform pertama

    // Print nama platform
    char platform_name[256];
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
    std::cout << "[OpenCL] Platform: " << platform_name << "\n";

    // ── 2. Device Discovery ────────────────────────────────────────────────
    // Prioritas: GPU → CPU fallback
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err == CL_DEVICE_NOT_FOUND) {
        std::cout << "[OpenCL] GPU tidak ditemukan, fallback ke CPU.\n";
        CL_CHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr));
    } else {
        CL_CHECK(err);
    }

    printDeviceInfo();

    // ── 3. Context Creation ────────────────────────────────────────────────
    // Context menghubungkan host program dengan device
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);

    // ── 4. Command Queue ───────────────────────────────────────────────────
    // Queue untuk mengirim perintah ke device (kernel dispatch, memcpy, dll)
    // CL_QUEUE_PROFILING_ENABLE: memungkinkan pengukuran waktu kernel
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    CL_CHECK(err);

    initialized_ = true;
    std::cout << "[OpenCL] Inisialisasi berhasil.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Load & Compile OpenCL Program dari .cl source
// ─────────────────────────────────────────────────────────────────────────────
void OpenCLManager::loadProgram(const std::string& kernel_path,
                                const std::string& program_name) {
    if (!initialized_)
        throw std::runtime_error("OpenCLManager belum diinisialisasi.");

    std::string source = readFile(kernel_path);
    const char* src_ptr = source.c_str();
    size_t      src_len = source.size();

    cl_int err;

    // Buat program object dari source string
    cl_program program = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);

    // Compile program untuk device yang dipilih
    // Build options: -cl-fast-relaxed-math untuk optimasi floating point
    const char* build_options = "-cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &device_, build_options, nullptr, nullptr);

    // Jika build gagal, ambil build log untuk debugging
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error(
            std::string("Build program '") + program_name + "' gagal:\n" + log.data());
    }

    programs_[program_name] = program;
    std::cout << "[OpenCL] Program '" << program_name << "' berhasil dikompilasi.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Create Kernel dari Program
// ─────────────────────────────────────────────────────────────────────────────
void OpenCLManager::createKernel(const std::string& program_name,
                                  const std::string& kernel_name,
                                  const std::string& kernel_id) {
    auto it = programs_.find(program_name);
    if (it == programs_.end())
        throw std::runtime_error("Program '" + program_name + "' belum di-load.");

    cl_int err;
    cl_kernel kernel = clCreateKernel(it->second, kernel_name.c_str(), &err);
    CL_CHECK(err);

    kernels_[kernel_id] = kernel;
    std::cout << "[OpenCL] Kernel '" << kernel_id << "' berhasil dibuat.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer Management
// ─────────────────────────────────────────────────────────────────────────────
cl_mem OpenCLManager::allocBuffer(size_t size, cl_mem_flags flags) {
    cl_int err;
    cl_mem buf = clCreateBuffer(context_, flags, size, nullptr, &err);
    CL_CHECK(err);
    return buf;
}

void OpenCLManager::writeBuffer(cl_mem buffer, size_t size, const void* data, bool blocking) {
    CL_CHECK(clEnqueueWriteBuffer(
        queue_, buffer,
        blocking ? CL_TRUE : CL_FALSE,
        0, size, data,
        0, nullptr, nullptr));
}

void OpenCLManager::readBuffer(cl_mem buffer, size_t size, void* data, bool blocking) {
    CL_CHECK(clEnqueueReadBuffer(
        queue_, buffer,
        blocking ? CL_TRUE : CL_FALSE,
        0, size, data,
        0, nullptr, nullptr));
}

void OpenCLManager::releaseBuffer(cl_mem buffer) {
    if (buffer) clReleaseMemObject(buffer);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Dispatch: 2D NDRange
// ─────────────────────────────────────────────────────────────────────────────
void OpenCLManager::enqueueKernel2D(const std::string& kernel_id,
                                     const size_t global_work[2],
                                     const size_t* local_work) {
    auto it = kernels_.find(kernel_id);
    if (it == kernels_.end())
        throw std::runtime_error("Kernel '" + kernel_id + "' tidak ditemukan.");

    // Round up global_work agar habis dibagi local_work (jika local_work diberikan)
    size_t rounded[2] = { global_work[0], global_work[1] };
    if (local_work) {
        rounded[0] = ((global_work[0] + local_work[0] - 1) / local_work[0]) * local_work[0];
        rounded[1] = ((global_work[1] + local_work[1] - 1) / local_work[1]) * local_work[1];
    }

    CL_CHECK(clEnqueueNDRangeKernel(
        queue_,
        it->second,
        2,           // work dimensions
        nullptr,     // global work offset
        rounded,     // global work size
        local_work,  // local work size (work-group size)
        0, nullptr, nullptr));
}

void OpenCLManager::finish() {
    CL_CHECK(clFinish(queue_));
}

cl_kernel OpenCLManager::getKernel(const std::string& kernel_id) {
    auto it = kernels_.find(kernel_id);
    if (it == kernels_.end())
        throw std::runtime_error("Kernel '" + kernel_id + "' tidak ditemukan.");
    return it->second;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────
void OpenCLManager::printDeviceInfo() const {
    char name[256], vendor[256];
    cl_ulong global_mem, local_mem;
    cl_uint compute_units, max_freq;
    size_t max_work_group;

    clGetDeviceInfo(device_, CL_DEVICE_NAME,              sizeof(name),    name,          nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VENDOR,            sizeof(vendor),  vendor,        nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,   sizeof(cl_ulong),&global_mem,   nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,    sizeof(cl_ulong),&local_mem,    nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units,nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_freq,   nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group, nullptr);

    std::cout << "[OpenCL] Device Info:\n"
              << "  Nama         : " << name << "\n"
              << "  Vendor       : " << vendor << "\n"
              << "  Global Mem   : " << global_mem / (1024*1024) << " MB\n"
              << "  Local Mem    : " << local_mem / 1024 << " KB\n"
              << "  Compute Units: " << compute_units << "\n"
              << "  Max Freq     : " << max_freq << " MHz\n"
              << "  Max WG Size  : " << max_work_group << "\n";
}

std::string OpenCLManager::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Gagal membuka file: " + path);
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace ocl
