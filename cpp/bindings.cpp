/**
 * @file bindings.cpp
 * @brief Python Bindings menggunakan pybind11
 *
 * Jembatan antara Python (controller) dan C++ (OpenCL host).
 *
 * Alur Data:
 *   Python numpy array (uint8 RGBA)
 *   → pybind11 buffer protocol
 *   → C++ std::vector<uint8_t>
 *   → OpenCL buffer (GPU)
 *   → Kernel execution
 *   → C++ std::vector<uint8_t>
 *   → pybind11 → Python numpy array
 *
 * Menggunakan pybind11 karena:
 *   1. Type safety yang lebih baik dibanding ctypes
 *   2. Support langsung numpy array via buffer protocol
 *   3. Exception Python yang proper dari C++ exceptions
 *   4. Overhead minimal dibanding ctypes manual marshaling
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "opencl_manager.h"
#include "image_processor.h"

namespace py = pybind11;
using namespace ocl;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: Konversi numpy array ↔ ImageData
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Konversi numpy array uint8 (H, W, 4) → ImageData
 * numpy array harus RGBA dan C-contiguous
 */
ImageData numpyToImageData(py::array_t<uint8_t> arr) {
    auto buf = arr.request();

    if (buf.ndim != 3 || buf.shape[2] != 4) {
        throw std::invalid_argument(
            "Input harus numpy array shape (H, W, 4) dengan dtype uint8. "
            "Gunakan cv2.cvtColor + alpha channel jika perlu.");
    }

    ImageData img;
    img.height   = (int)buf.shape[0];
    img.width    = (int)buf.shape[1];
    img.channels = 4;
    img.pixels.resize(img.sizeBytes());

    std::memcpy(img.pixels.data(), buf.ptr, img.sizeBytes());
    return img;
}

/**
 * Konversi ImageData → numpy array uint8 (H, W, 4)
 */
py::array_t<uint8_t> imageDataToNumpy(const ImageData& img) {
    auto arr = py::array_t<uint8_t>(
        { img.height, img.width, img.channels },
        { img.width * img.channels, img.channels, 1 }
    );
    auto buf = arr.request();
    std::memcpy(buf.ptr, img.pixels.data(), img.sizeBytes());
    return arr;
}

/**
 * Konversi TimingResult → Python dict
 */
py::dict timingToDict(const TimingResult& t) {
    py::dict d;
    d["resize_ms"]    = t.resize_ms;
    d["grayscale_ms"] = t.grayscale_ms;
    d["gaussian_ms"]  = t.gaussian_ms;
    d["flatten_ms"]   = t.flatten_ms;
    d["transfer_ms"]  = t.transfer_ms;
    d["total_ms"]     = t.total_ms;
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper class yang di-expose ke Python
// ─────────────────────────────────────────────────────────────────────────────
class GPUImagePipeline {
public:
    GPUImagePipeline(const std::string& kernels_dir) {
        manager_.initialize();
        processor_ = std::make_unique<ImageProcessor>(manager_, kernels_dir);
    }

    /**
     * @brief Jalankan full GPU pipeline
     * @param img_rgba  numpy uint8 (H, W, 4)
     * @param out_w     Target width
     * @param out_h     Target height
     * @return dict dengan keys: resized, grayscale, blurred, flattened, timing
     */
    py::dict process(py::array_t<uint8_t> img_rgba, int out_w, int out_h) {
        ImageData input = numpyToImageData(img_rgba);
        PipelineResult result = processor_->processPipeline(input, out_w, out_h);

        py::dict out;
        out["resized"]   = imageDataToNumpy(result.resized);
        out["grayscale"] = imageDataToNumpy(result.grayscale);
        out["blurred"]   = imageDataToNumpy(result.blurred);
        out["flattened"] = py::array_t<float>(
            { (py::ssize_t)result.flattened.size() },
            result.flattened.data()
        );
        out["timing"] = timingToDict(result.timing);
        return out;
    }

    /**
     * @brief Jalankan full CPU pipeline (reference)
     */
    py::dict processCPU(py::array_t<uint8_t> img_rgba, int out_w, int out_h) {
        ImageData input = numpyToImageData(img_rgba);
        PipelineResult result = processor_->processPipelineCPU(input, out_w, out_h);

        py::dict out;
        out["resized"]   = imageDataToNumpy(result.resized);
        out["grayscale"] = imageDataToNumpy(result.grayscale);
        out["blurred"]   = imageDataToNumpy(result.blurred);
        out["flattened"] = py::array_t<float>(
            { (py::ssize_t)result.flattened.size() },
            result.flattened.data()
        );
        out["timing"] = timingToDict(result.timing);
        return out;
    }

    void printDeviceInfo() {
        manager_.printDeviceInfo();
    }

private:
    OpenCLManager manager_;
    std::unique_ptr<ImageProcessor> processor_;
};

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 Module Definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(gpu_pipeline, m) {
    m.doc() = "GPU Image Processing Pipeline menggunakan OpenCL";

    py::class_<GPUImagePipeline>(m, "GPUImagePipeline")
        .def(py::init<const std::string&>(),
             py::arg("kernels_dir"),
             "Inisialisasi pipeline GPU. kernels_dir = path ke folder .cl files")
        .def("process", &GPUImagePipeline::process,
             py::arg("img_rgba"), py::arg("out_w"), py::arg("out_h"),
             "Jalankan pipeline GPU: resize → grayscale → blur → flatten\n"
             "Input: numpy array (H, W, 4) uint8 RGBA\n"
             "Output: dict {resized, grayscale, blurred, flattened, timing}")
        .def("process_cpu", &GPUImagePipeline::processCPU,
             py::arg("img_rgba"), py::arg("out_w"), py::arg("out_h"),
             "Jalankan pipeline CPU (reference implementation)")
        .def("print_device_info", &GPUImagePipeline::printDeviceInfo,
             "Print informasi GPU/device yang digunakan");
}
