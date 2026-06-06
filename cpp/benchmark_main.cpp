/**
 * @file benchmark_main.cpp
 * @brief Standalone C++ benchmark executable (tanpa Python)
 *
 * Dapat dijalankan langsung: ./benchmark_cpp <image_path> <kernels_dir>
 * Menghasilkan output tabel benchmark ke stdout dan CSV ke benchmark_results/
 */

#include "opencl_manager.h"
#include "image_processor.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

// Buat dummy 512x512 RGBA image (gradient pattern untuk testing)
ocl::ImageData createTestImage(int w, int h) {
    ocl::ImageData img;
    img.width = w; img.height = h; img.channels = 4;
    img.pixels.resize(img.sizeBytes());

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 4;
            img.pixels[idx + 0] = (uint8_t)((x * 255) / w);        // R
            img.pixels[idx + 1] = (uint8_t)((y * 255) / h);        // G
            img.pixels[idx + 2] = (uint8_t)(((x+y) * 255) / (w+h)); // B
            img.pixels[idx + 3] = 255;                               // A
        }
    }
    return img;
}

void printTable(const std::vector<std::tuple<std::string,
                    ocl::TimingResult, ocl::TimingResult>>& rows) {
    const char* sep = "+----------------+----------+----------+----------+----------+----------+";
    std::cout << "\n=== BENCHMARK RESULTS ===\n\n";

    // Header
    std::cout << sep << "\n";
    std::cout << "| Resolution     | CPU (ms) | GPU (ms) | Speedup  | Op       | Detail   |\n";
    std::cout << sep << "\n";

    for (auto& [label, cpu, gpu] : rows) {
        auto speedup = [](double c, double g) -> std::string {
            if (g <= 0) return "N/A";
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2fx", c / g);
            return buf;
        };

        auto row = [&](const char* op, double c_ms, double g_ms) {
            std::cout << "| " << std::left << std::setw(14) << label
                      << " | " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << c_ms
                      << " | " << std::setw(8) << g_ms
                      << " | " << std::setw(8) << speedup(c_ms, g_ms)
                      << " | " << std::left << std::setw(8) << op
                      << " |          |\n";
        };

        row("Resize",    cpu.resize_ms,    gpu.resize_ms);
        row("Grayscale", cpu.grayscale_ms, gpu.grayscale_ms);
        row("Gaussian",  cpu.gaussian_ms,  gpu.gaussian_ms);
        row("Flatten",   cpu.flatten_ms,   gpu.flatten_ms);
        row("TOTAL",     cpu.total_ms,     gpu.total_ms);
        std::cout << sep << "\n";
    }
}

int main(int argc, char** argv) {
    std::string kernels_dir = "./kernels";
    if (argc > 1) kernels_dir = argv[1];

    std::cout << "=== GPU Image Processing Benchmark (OpenCL) ===\n";
    std::cout << "Kernels dir: " << kernels_dir << "\n\n";

    try {
        ocl::OpenCLManager mgr;
        mgr.initialize();

        ocl::ImageProcessor proc(mgr, kernels_dir);

        // Test image source: 1024x768 dummy
        ocl::ImageData source = createTestImage(1024, 768);

        std::vector<std::pair<int,int>> sizes = {{128,128}, {256,256}, {512,512}};
        std::vector<std::tuple<std::string, ocl::TimingResult, ocl::TimingResult>> results;

        for (auto [w, h] : sizes) {
            std::string label = std::to_string(w) + "x" + std::to_string(h);
            std::cout << "\n[Benchmarking " << label << "...]\n";

            // Warmup
            proc.processPipeline(source, w, h);

            // CPU run
            auto cpu_result = proc.processPipelineCPU(source, w, h);

            // GPU run (rata-rata 3x)
            ocl::TimingResult gpu_avg = {};
            int runs = 3;
            for (int i = 0; i < runs; i++) {
                auto r = proc.processPipeline(source, w, h);
                gpu_avg.resize_ms    += r.timing.resize_ms;
                gpu_avg.grayscale_ms += r.timing.grayscale_ms;
                gpu_avg.gaussian_ms  += r.timing.gaussian_ms;
                gpu_avg.flatten_ms   += r.timing.flatten_ms;
                gpu_avg.total_ms     += r.timing.total_ms;
            }
            gpu_avg.resize_ms    /= runs;
            gpu_avg.grayscale_ms /= runs;
            gpu_avg.gaussian_ms  /= runs;
            gpu_avg.flatten_ms   /= runs;
            gpu_avg.total_ms     /= runs;

            results.emplace_back(label, cpu_result.timing, gpu_avg);

            std::cout << "  CPU total: " << cpu_result.timing.total_ms << " ms\n";
            std::cout << "  GPU total: " << gpu_avg.total_ms << " ms\n";
            std::cout << "  Speedup:   " << cpu_result.timing.total_ms / gpu_avg.total_ms << "x\n";
        }

        printTable(results);

        // Save CSV
        std::ofstream csv("benchmark_results/results_cpp.csv");
        csv << "resolution,operation,cpu_ms,gpu_ms,speedup\n";
        for (auto& [label, cpu, gpu] : results) {
            auto write = [&](const char* op, double c, double g) {
                csv << label << "," << op << "," << c << "," << g << ","
                    << (g > 0 ? c/g : 0.0) << "\n";
            };
            write("resize",    cpu.resize_ms,    gpu.resize_ms);
            write("grayscale", cpu.grayscale_ms, gpu.grayscale_ms);
            write("gaussian",  cpu.gaussian_ms,  gpu.gaussian_ms);
            write("flatten",   cpu.flatten_ms,   gpu.flatten_ms);
            write("total",     cpu.total_ms,     gpu.total_ms);
        }
        std::cout << "\nCSV saved: benchmark_results/results_cpp.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
