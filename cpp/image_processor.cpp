/**
 * @file image_processor.cpp
 * @brief Implementasi GPU dan CPU Image Processing Pipeline
 */

#include "image_processor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace ocl {

ImageProcessor::ImageProcessor(OpenCLManager& manager, const std::string& kernels_dir)
    : mgr_(manager), kernels_dir_(kernels_dir) {}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy-load kernels (only once)
// ─────────────────────────────────────────────────────────────────────────────
void ImageProcessor::ensureKernelsLoaded() {
    if (kernels_loaded_) return;

    mgr_.loadProgram(kernels_dir_ + "/resize.cl",    "resize");
    mgr_.loadProgram(kernels_dir_ + "/grayscale.cl", "grayscale");
    mgr_.loadProgram(kernels_dir_ + "/gaussian.cl",  "gaussian");
    mgr_.loadProgram(kernels_dir_ + "/flatten.cl",   "flatten");

    mgr_.createKernel("resize",    "resize_nearest",  "resize");
    mgr_.createKernel("grayscale", "grayscale",        "grayscale");
    mgr_.createKernel("gaussian",  "gaussian_blur",    "gaussian");
    mgr_.createKernel("flatten",   "flatten",          "flatten");

    kernels_loaded_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU OPERATIONS
// ─────────────────────────────────────────────────────────────────────────────

ImageData ImageProcessor::gpuResize(const ImageData& src, int out_w, int out_h, double& ms) {
    ensureKernelsLoaded();

    ImageData dst;
    dst.width = out_w; dst.height = out_h; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    size_t src_bytes = src.sizeBytes();
    size_t dst_bytes = dst.sizeBytes();

    // Alokasi buffer di GPU
    cl_mem src_buf = mgr_.allocBuffer(src_bytes, CL_MEM_READ_ONLY);
    cl_mem dst_buf = mgr_.allocBuffer(dst_bytes, CL_MEM_WRITE_ONLY);

    // Transfer data host → device
    auto t0 = Clock::now();
    mgr_.writeBuffer(src_buf, src_bytes, src.pixels.data());

    // Set kernel arguments
    cl_kernel k = mgr_.getKernel("resize");
    clSetKernelArg(k, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(k, 2, sizeof(int),    &src.width);
    clSetKernelArg(k, 3, sizeof(int),    &src.height);
    clSetKernelArg(k, 4, sizeof(int),    &out_w);
    clSetKernelArg(k, 5, sizeof(int),    &out_h);

    // Dispatch: global = {out_w, out_h}, local = {16, 16}
    size_t global[2] = { (size_t)out_w, (size_t)out_h };
    size_t local[2]  = { 16, 16 };
    mgr_.enqueueKernel2D("resize", global, local);
    mgr_.finish();

    // Transfer result device → host
    mgr_.readBuffer(dst_buf, dst_bytes, dst.pixels.data());
    auto t1 = Clock::now();
    ms = elapsed_ms(t0, t1);

    mgr_.releaseBuffer(src_buf);
    mgr_.releaseBuffer(dst_buf);
    return dst;
}

ImageData ImageProcessor::gpuGrayscale(const ImageData& src, double& ms) {
    ensureKernelsLoaded();

    ImageData dst;
    dst.width = src.width; dst.height = src.height; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    size_t bytes = src.sizeBytes();
    cl_mem src_buf = mgr_.allocBuffer(bytes, CL_MEM_READ_ONLY);
    cl_mem dst_buf = mgr_.allocBuffer(bytes, CL_MEM_WRITE_ONLY);

    auto t0 = Clock::now();
    mgr_.writeBuffer(src_buf, bytes, src.pixels.data());

    cl_kernel k = mgr_.getKernel("grayscale");
    clSetKernelArg(k, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(k, 2, sizeof(int),    &src.width);
    clSetKernelArg(k, 3, sizeof(int),    &src.height);

    size_t global[2] = { (size_t)src.width, (size_t)src.height };
    size_t local[2]  = { 16, 16 };
    mgr_.enqueueKernel2D("grayscale", global, local);
    mgr_.finish();

    mgr_.readBuffer(dst_buf, bytes, dst.pixels.data());
    auto t1 = Clock::now();
    ms = elapsed_ms(t0, t1);

    mgr_.releaseBuffer(src_buf);
    mgr_.releaseBuffer(dst_buf);
    return dst;
}

ImageData ImageProcessor::gpuGaussianBlur(const ImageData& src, double& ms) {
    ensureKernelsLoaded();

    ImageData dst;
    dst.width = src.width; dst.height = src.height; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    size_t bytes = src.sizeBytes();
    cl_mem src_buf = mgr_.allocBuffer(bytes, CL_MEM_READ_ONLY);
    cl_mem dst_buf = mgr_.allocBuffer(bytes, CL_MEM_WRITE_ONLY);

    auto t0 = Clock::now();
    mgr_.writeBuffer(src_buf, bytes, src.pixels.data());

    cl_kernel k = mgr_.getKernel("gaussian");
    clSetKernelArg(k, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(k, 2, sizeof(int),    &src.width);
    clSetKernelArg(k, 3, sizeof(int),    &src.height);

    size_t global[2] = { (size_t)src.width, (size_t)src.height };
    size_t local[2]  = { 16, 16 };
    mgr_.enqueueKernel2D("gaussian", global, local);
    mgr_.finish();

    mgr_.readBuffer(dst_buf, bytes, dst.pixels.data());
    auto t1 = Clock::now();
    ms = elapsed_ms(t0, t1);

    mgr_.releaseBuffer(src_buf);
    mgr_.releaseBuffer(dst_buf);
    return dst;
}

std::vector<float> ImageProcessor::gpuFlatten(const ImageData& src, double& ms) {
    ensureKernelsLoaded();

    size_t num_floats = (size_t)src.width * src.height * 3;
    std::vector<float> result(num_floats);

    size_t src_bytes = src.sizeBytes();
    size_t dst_bytes = num_floats * sizeof(float);

    cl_mem src_buf = mgr_.allocBuffer(src_bytes, CL_MEM_READ_ONLY);
    cl_mem dst_buf = mgr_.allocBuffer(dst_bytes, CL_MEM_WRITE_ONLY);

    auto t0 = Clock::now();
    mgr_.writeBuffer(src_buf, src_bytes, src.pixels.data());

    cl_kernel k = mgr_.getKernel("flatten");
    clSetKernelArg(k, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(k, 2, sizeof(int),    &src.width);
    clSetKernelArg(k, 3, sizeof(int),    &src.height);

    size_t global[2] = { (size_t)src.width, (size_t)src.height };
    size_t local[2]  = { 16, 16 };
    mgr_.enqueueKernel2D("flatten", global, local);
    mgr_.finish();

    mgr_.readBuffer(dst_buf, dst_bytes, result.data());
    auto t1 = Clock::now();
    ms = elapsed_ms(t0, t1);

    mgr_.releaseBuffer(src_buf);
    mgr_.releaseBuffer(dst_buf);
    return result;
}

PipelineResult ImageProcessor::processPipeline(const ImageData& input,
                                                int out_w, int out_h) {
    PipelineResult result;
    auto t_start = Clock::now();

    result.resized   = gpuResize    (input,          out_w, out_h, result.timing.resize_ms);
    result.grayscale = gpuGrayscale (result.resized,              result.timing.grayscale_ms);
    result.blurred   = gpuGaussianBlur(result.grayscale,          result.timing.gaussian_ms);
    result.flattened = gpuFlatten   (result.blurred,              result.timing.flatten_ms);

    auto t_end = Clock::now();
    result.timing.total_ms = elapsed_ms(t_start, t_end);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU REFERENCE IMPLEMENTATIONS
// ─────────────────────────────────────────────────────────────────────────────

ImageData ImageProcessor::cpuResize(const ImageData& src, int out_w, int out_h, double& ms) {
    auto t0 = Clock::now();

    ImageData dst;
    dst.width = out_w; dst.height = out_h; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    float scale_x = (float)src.width  / out_w;
    float scale_y = (float)src.height / out_h;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int sx = std::min((int)(x * scale_x), src.width  - 1);
            int sy = std::min((int)(y * scale_y), src.height - 1);

            int src_idx = (sy * src.width + sx) * 4;
            int dst_idx = (y  * out_w     + x ) * 4;

            dst.pixels[dst_idx + 0] = src.pixels[src_idx + 0];
            dst.pixels[dst_idx + 1] = src.pixels[src_idx + 1];
            dst.pixels[dst_idx + 2] = src.pixels[src_idx + 2];
            dst.pixels[dst_idx + 3] = src.pixels[src_idx + 3];
        }
    }

    ms = elapsed_ms(t0, Clock::now());
    return dst;
}

ImageData ImageProcessor::cpuGrayscale(const ImageData& src, double& ms) {
    auto t0 = Clock::now();

    ImageData dst;
    dst.width = src.width; dst.height = src.height; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    for (int i = 0; i < src.width * src.height; i++) {
        int idx = i * 4;
        uint8_t luma = (uint8_t)(
            0.299f * src.pixels[idx + 0] +
            0.587f * src.pixels[idx + 1] +
            0.114f * src.pixels[idx + 2]);
        dst.pixels[idx + 0] = luma;
        dst.pixels[idx + 1] = luma;
        dst.pixels[idx + 2] = luma;
        dst.pixels[idx + 3] = 255;
    }

    ms = elapsed_ms(t0, Clock::now());
    return dst;
}

ImageData ImageProcessor::cpuGaussianBlur(const ImageData& src, double& ms) {
    auto t0 = Clock::now();

    // Gaussian kernel 5×5 (normalized)
    static const float kern[5][5] = {
        { 1/256.f,  4/256.f,  6/256.f,  4/256.f,  1/256.f },
        { 4/256.f, 16/256.f, 24/256.f, 16/256.f,  4/256.f },
        { 6/256.f, 24/256.f, 36/256.f, 24/256.f,  6/256.f },
        { 4/256.f, 16/256.f, 24/256.f, 16/256.f,  4/256.f },
        { 1/256.f,  4/256.f,  6/256.f,  4/256.f,  1/256.f }
    };

    ImageData dst;
    dst.width = src.width; dst.height = src.height; dst.channels = 4;
    dst.pixels.resize(dst.sizeBytes());

    for (int y = 0; y < src.height; y++) {
        for (int x = 0; x < src.width; x++) {
            float r = 0, g = 0, b = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int sx = std::clamp(x + kx, 0, src.width  - 1);
                    int sy = std::clamp(y + ky, 0, src.height - 1);
                    int si = (sy * src.width + sx) * 4;
                    float w = kern[ky+2][kx+2];
                    r += w * src.pixels[si + 0];
                    g += w * src.pixels[si + 1];
                    b += w * src.pixels[si + 2];
                }
            }
            int di = (y * src.width + x) * 4;
            dst.pixels[di + 0] = (uint8_t)std::clamp(r, 0.f, 255.f);
            dst.pixels[di + 1] = (uint8_t)std::clamp(g, 0.f, 255.f);
            dst.pixels[di + 2] = (uint8_t)std::clamp(b, 0.f, 255.f);
            dst.pixels[di + 3] = 255;
        }
    }

    ms = elapsed_ms(t0, Clock::now());
    return dst;
}

std::vector<float> ImageProcessor::cpuFlatten(const ImageData& src, double& ms) {
    auto t0 = Clock::now();

    size_t n = (size_t)src.width * src.height * 3;
    std::vector<float> result(n);

    for (int i = 0; i < src.width * src.height; i++) {
        result[i * 3 + 0] = src.pixels[i * 4 + 0] / 255.0f;
        result[i * 3 + 1] = src.pixels[i * 4 + 1] / 255.0f;
        result[i * 3 + 2] = src.pixels[i * 4 + 2] / 255.0f;
    }

    ms = elapsed_ms(t0, Clock::now());
    return result;
}

PipelineResult ImageProcessor::processPipelineCPU(const ImageData& input, int out_w, int out_h) {
    PipelineResult result;
    auto t_start = Clock::now();

    result.resized   = cpuResize    (input,          out_w, out_h, result.timing.resize_ms);
    result.grayscale = cpuGrayscale (result.resized,              result.timing.grayscale_ms);
    result.blurred   = cpuGaussianBlur(result.grayscale,          result.timing.gaussian_ms);
    result.flattened = cpuFlatten   (result.blurred,              result.timing.flatten_ms);

    auto t_end = Clock::now();
    result.timing.total_ms = elapsed_ms(t_start, t_end);
    return result;
}

} // namespace ocl
