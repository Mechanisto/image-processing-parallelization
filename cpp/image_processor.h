#pragma once
/**
 * @file image_processor.h
 * @brief GPU Image Processing Pipeline menggunakan OpenCL
 *
 * Menyediakan API high-level untuk pipeline:
 *   1. Resize (Nearest Neighbor)
 *   2. Grayscale Conversion
 *   3. Gaussian Blur
 *   4. Flatten to 1D float vector
 *
 * Setiap operasi berjalan di GPU melalui OpenCL kernel.
 * Juga menyediakan CPU reference implementation untuk benchmark.
 */

#include "opencl_manager.h"
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>

namespace ocl {

// ─────────────────────────────────────────────────────────────────────────────
// Struct untuk hasil timing benchmark
// ─────────────────────────────────────────────────────────────────────────────
struct TimingResult {
    double resize_ms     = 0.0;
    double grayscale_ms  = 0.0;
    double gaussian_ms   = 0.0;
    double flatten_ms    = 0.0;
    double transfer_ms   = 0.0;  // Overhead CPU-GPU transfer
    double total_ms      = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Struct untuk menyimpan image data
// ─────────────────────────────────────────────────────────────────────────────
struct ImageData {
    std::vector<uint8_t> pixels;  // RGBA, 4 bytes per pixel
    int width  = 0;
    int height = 0;
    int channels = 4;  // selalu RGBA

    size_t sizeBytes() const {
        return (size_t)width * height * channels;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Results
// ─────────────────────────────────────────────────────────────────────────────
struct PipelineResult {
    ImageData resized;
    ImageData grayscale;
    ImageData blurred;
    std::vector<float> flattened;  // 1D float vector, normalized [0,1]
    TimingResult timing;
};

// ─────────────────────────────────────────────────────────────────────────────
// ImageProcessor: GPU-accelerated pipeline
// ─────────────────────────────────────────────────────────────────────────────
class ImageProcessor {
public:
    explicit ImageProcessor(OpenCLManager& manager, const std::string& kernels_dir);

    /**
     * @brief Jalankan pipeline GPU lengkap: resize → grayscale → blur → flatten
     * @param input      Input image (RGBA)
     * @param out_width  Target width
     * @param out_height Target height
     * @return PipelineResult berisi semua intermediate + waktu
     */
    PipelineResult processPipeline(const ImageData& input,
                                   int out_width, int out_height);

    // Individual GPU operations (untuk benchmark per-operasi)
    ImageData gpuResize    (const ImageData& src, int out_w, int out_h, double& ms);
    ImageData gpuGrayscale (const ImageData& src, double& ms);
    ImageData gpuGaussianBlur(const ImageData& src, double& ms);
    std::vector<float> gpuFlatten(const ImageData& src, double& ms);

    // CPU reference implementations
    ImageData cpuResize    (const ImageData& src, int out_w, int out_h, double& ms);
    ImageData cpuGrayscale (const ImageData& src, double& ms);
    ImageData cpuGaussianBlur(const ImageData& src, double& ms);
    std::vector<float> cpuFlatten(const ImageData& src, double& ms);

    // CPU full pipeline
    PipelineResult processPipelineCPU(const ImageData& input, int out_w, int out_h);

private:
    OpenCLManager& mgr_;
    std::string kernels_dir_;
    bool kernels_loaded_ = false;

    void ensureKernelsLoaded();

    // Helper: measure time
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    static double elapsed_ms(TimePoint start, TimePoint end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

} // namespace ocl
