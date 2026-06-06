#pragma once
/**
 * @file opencl_manager.h
 * @brief OpenCL Lifecycle Manager
 *
 * Bertanggung jawab atas seluruh OpenCL boilerplate:
 *   - Platform & Device discovery
 *   - Context & Command Queue creation
 *   - Program compilation dari .cl source
 *   - Kernel object management
 *   - Error handling
 */

#include <CL/cl.h>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <vector>

// Macro untuk error checking OpenCL
#define CL_CHECK(err) \
    if ((err) != CL_SUCCESS) { \
        throw std::runtime_error(std::string("OpenCL Error [") + \
            std::to_string(err) + "] at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }

namespace ocl {

/**
 * @class OpenCLManager
 * @brief Singleton-style manager untuk seluruh resource OpenCL.
 *
 * Lifecycle yang dikelola:
 *   1. Platform discovery  → memilih platform OpenCL pertama yang tersedia
 *   2. Device discovery    → memilih GPU device; fallback ke CPU jika tidak ada GPU
 *   3. Context creation    → membuat context untuk device yang dipilih
 *   4. Command Queue       → antrian perintah untuk dispatch kernel
 *   5. Program compilation → compile .cl source menjadi binary
 *   6. Kernel creation     → buat kernel object dari program
 */
class OpenCLManager {
public:
    OpenCLManager();
    ~OpenCLManager();

    // Non-copyable
    OpenCLManager(const OpenCLManager&) = delete;
    OpenCLManager& operator=(const OpenCLManager&) = delete;

    /**
     * @brief Inisialisasi OpenCL: platform, device, context, command queue.
     * @throws std::runtime_error jika tidak ada device yang compatible
     */
    void initialize();

    /**
     * @brief Compile OpenCL program dari source file .cl
     * @param kernel_path   Path ke file .cl
     * @param program_name  Nama identifier untuk cache
     */
    void loadProgram(const std::string& kernel_path, const std::string& program_name);

    /**
     * @brief Buat kernel object dari program yang sudah dicompile
     * @param program_name  Nama program (harus sudah di-load)
     * @param kernel_name   Nama fungsi kernel dalam .cl
     * @param kernel_id     Identifier untuk cache kernel
     */
    void createKernel(const std::string& program_name,
                      const std::string& kernel_name,
                      const std::string& kernel_id);

    /**
     * @brief Alokasi buffer di device memory (GPU global memory)
     * @param size   Ukuran buffer dalam bytes
     * @param flags  CL_MEM_READ_ONLY / CL_MEM_WRITE_ONLY / CL_MEM_READ_WRITE
     * @return cl_mem handle
     */
    cl_mem allocBuffer(size_t size, cl_mem_flags flags);

    /**
     * @brief Transfer data dari host (CPU) ke device (GPU)
     * @param buffer   cl_mem target buffer
     * @param size     Ukuran data dalam bytes
     * @param data     Pointer ke data host
     * @param blocking Jika true, tunggu hingga transfer selesai
     */
    void writeBuffer(cl_mem buffer, size_t size, const void* data, bool blocking = true);

    /**
     * @brief Transfer data dari device (GPU) ke host (CPU)
     */
    void readBuffer(cl_mem buffer, size_t size, void* data, bool blocking = true);

    /**
     * @brief Dispatch kernel dengan 2D global work size
     * @param kernel_id     Identifier kernel (harus sudah dibuat)
     * @param global_work   Array 2 elemen: {width, height}
     * @param local_work    Array 2 elemen: {local_x, local_y}, nullptr untuk auto
     */
    void enqueueKernel2D(const std::string& kernel_id,
                         const size_t global_work[2],
                         const size_t* local_work = nullptr);

    /**
     * @brief Tunggu hingga semua command dalam queue selesai dieksekusi
     */
    void finish();

    /**
     * @brief Bebaskan cl_mem buffer
     */
    void releaseBuffer(cl_mem buffer);

    /**
     * @brief Getter untuk kernel object (untuk set argument)
     */
    cl_kernel getKernel(const std::string& kernel_id);

    /**
     * @brief Print informasi device yang dipilih (nama, memori, compute units, dll)
     */
    void printDeviceInfo() const;

    /**
     * @brief Baca file teks dari disk
     */
    static std::string readFile(const std::string& path);

private:
    cl_platform_id   platform_   = nullptr;
    cl_device_id     device_     = nullptr;
    cl_context       context_    = nullptr;
    cl_command_queue queue_      = nullptr;

    // Cache: program_name -> cl_program
    std::unordered_map<std::string, cl_program> programs_;

    // Cache: kernel_id -> cl_kernel
    std::unordered_map<std::string, cl_kernel>  kernels_;

    bool initialized_ = false;
};

} // namespace ocl
