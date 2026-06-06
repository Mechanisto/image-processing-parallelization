# GPU Image Processing Pipeline — OpenCL

> Proyek Tugas Akhir Arsitektur Komputer  
> Image Processing Python menggunakan OpenCL + C++ + Python

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Application                          │
│  main.py (controller)  │  benchmark.py  │  utils.py            │
│                                                                 │
│  - cv2.imread()  ← OpenCV HANYA untuk I/O                      │
│  - cv2.imwrite()                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │  pybind11 (shared library .so/.pyd)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    C++ OpenCL Host                              │
│                                                                 │
│  bindings.cpp          → Python ↔ C++ bridge                   │
│  opencl_manager.cpp    → OpenCL lifecycle management            │
│  image_processor.cpp   → Pipeline orchestration                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │  OpenCL API (clEnqueueNDRangeKernel)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GPU (OpenCL Device)                         │
│                                                                 │
│  resize.cl      → Nearest Neighbor Resize                      │
│  grayscale.cl   → ITU-R BT.601 Luminance                       │
│  gaussian.cl    → 5×5 Gaussian Blur Convolution                │
│  flatten.cl     → Image → 1D Float Vector                      │
└─────────────────────────────────────────────────────────────────┘
```

### Alur Data Detail

```
Python numpy (H,W,4 uint8)
    │
    │ pybind11 buffer protocol (zero-copy jika possible)
    ▼
C++ ImageData struct (std::vector<uint8_t>)
    │
    │ clEnqueueWriteBuffer → PCIe transfer
    ▼
GPU Global Memory (cl_mem buffer)
    │
    │ clEnqueueNDRangeKernel [global={W,H}, local={16,16}]
    ▼
OpenCL Kernel (setiap work-item = 1 pixel)
    │
    │ clEnqueueReadBuffer → PCIe transfer
    ▼
C++ result (std::vector<uint8_t>)
    │
    │ pybind11 → numpy array
    ▼
Python numpy (H,W,4 uint8) → cv2.imwrite()
```

---

## Struktur Proyek

```
project/
│
├── python/
│   ├── main.py          # Controller utama: baca gambar, jalankan pipeline, simpan
│   ├── benchmark.py     # Sistem benchmark CPU vs GPU + grafik + analisis
│   └── utils.py         # Helpers: I/O gambar, display, timing, formatting
│
├── cpp/
│   ├── opencl_manager.h/.cpp    # OpenCL lifecycle dari scratch
│   ├── image_processor.h/.cpp   # Pipeline GPU + CPU reference
│   ├── bindings.cpp             # pybind11 Python-C++ bridge
│   └── benchmark_main.cpp       # Standalone C++ benchmark executable
│
├── kernels/
│   ├── resize.cl        # Nearest Neighbor Resize
│   ├── grayscale.cl     # RGB → Grayscale (BT.601)
│   ├── gaussian.cl      # Gaussian Blur 5×5
│   └── flatten.cl       # Image → 1D float vector
│
├── datasets/            # Letakkan gambar JPG/PNG di sini
├── outputs/             # Hasil pemrosesan
├── benchmark_results/   # Hasil benchmark (CSV, JSON, charts)
└── CMakeLists.txt       # Build system
```

---

## Build Instructions

### Prerequisites

#### Linux (Ubuntu/Debian)
```bash
# Compiler
sudo apt-get install -y build-essential cmake

# OpenCL
# Pilih sesuai GPU:
# NVIDIA:
sudo apt-get install -y nvidia-opencl-dev
# AMD:
sudo apt-get install -y opencl-headers ocl-icd-opencl-dev
# Intel:
sudo apt-get install -y intel-opencl-icd opencl-headers ocl-icd-opencl-dev
# Universal (CPU fallback):
sudo apt-get install -y pocl-opencl-icd

# Python dependencies
sudo apt-get install -y python3-dev python3-pip
pip3 install pybind11 opencv-python numpy matplotlib

# Verifikasi OpenCL
clinfo  # harus menampilkan device info
```

#### Windows
```powershell
# Install Visual Studio 2022 (dengan C++ workload)
# Install CMake dari https://cmake.org/download/

# OpenCL SDK:
# NVIDIA: Install CUDA Toolkit (includes OpenCL headers)
# AMD: Install AMD APP SDK
# Intel: Install Intel OpenCL SDK

# Python dependencies
pip install pybind11 opencv-python numpy matplotlib

# Temukan path opencl.lib:
# NVIDIA: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\OpenCL.lib
# AMD:    C:\Program Files\AMD APP SDK\3.0\lib\x86_64\OpenCL.lib
# Intel:  C:\Program Files (x86)\Intel\OpenCL SDK\lib\x64\OpenCL.lib
```

### Build

#### Linux
```bash
cd project/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Output: python/gpu_pipeline.so dan ./benchmark_cpp
```

#### Windows
```cmd
cd project
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
# Output: python\gpu_pipeline.pyd dan .\benchmark_cpp.exe
```

#### macOS (Apple Silicon/Intel)
```bash
# OpenCL sudah tersedia secara native (Apple OpenCL)
brew install cmake
pip3 install pybind11 opencv-python numpy matplotlib

cd project/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

---

## Cara Menjalankan

### 1. Siapkan dataset
```bash
# Salin gambar JPG/PNG ke folder datasets/
cp /path/to/your/images/*.jpg datasets/
# atau download sample:
# wget -P datasets/ https://sample-image-url/photo.jpg
```

### 2. Jalankan pipeline utama
```bash
cd python/

# Proses semua gambar, output 256×256
python main.py --input ../datasets/ --output ../outputs/ --size 256

# Proses dan tampilkan hasil (membutuhkan GUI)
python main.py --input ../datasets/ --show

# Proses file tunggal
python main.py --input ../datasets/photo.jpg --size 512

# Lihat info GPU
python main.py --device-info
```

### 3. Jalankan benchmark
```bash
cd python/

# Benchmark lengkap (CPU vs GPU, 3 resolusi, charts)
python benchmark.py --input ../datasets/ --output ../benchmark_results/

# Hasilkan di:
#   benchmark_results/results.csv
#   benchmark_results/results.json
#   benchmark_results/charts/total_pipeline_comparison.png
#   benchmark_results/charts/per_operation_breakdown.png
#   benchmark_results/charts/speedup_heatmap.png
```

### 4. Jalankan standalone C++ benchmark
```bash
# Dari root project (setelah build)
./benchmark_cpp kernels/

# Atau dari build directory:
./build/benchmark_cpp ../kernels/
```

---

## Penjelasan Kernel OpenCL

### resize.cl — Nearest Neighbor
- **Mapping koordinat**: `src_x = dst_x × (src_W / dst_W)`
- **Global ID**: `get_global_id(0)` = kolom, `get_global_id(1)` = baris
- **Work-items**: `dst_W × dst_H`, setiap work-item = 1 pixel output
- **Memory**: 1 read (src), 1 write (dst), tidak ada dependency

### grayscale.cl — ITU-R BT.601
- **Formula**: `Y = 0.299R + 0.587G + 0.114B`
- **Paralelisasi**: Setiap pixel independen → embarrassingly parallel
- **Work-items**: `W × H`, total `W×H` multiply-add operations parallel

### gaussian.cl — 5×5 Convolution
- **Kernel 5×5**: Binomial coefficients, sum=256, normalized /256
- **Boundary**: Clamp-to-edge (koordinat di-clamp ke [0, W-1] / [0, H-1])
- **Computation**: 25 multiply-add per pixel per channel
- **Memory**: 25 global reads per pixel (tidak ada local memory optimization)

### flatten.cl — Image → 1D
- **Output format**: HWC (Height×Width×Channels), RGB 3-channel
- **Normalisasi**: `float_val = uchar_val / 255.0f` → range [0.0, 1.0]
- **Alpha**: Di-drop (tidak termasuk dalam output vector)
- **Indexing**: `dst[y×W×3 + x×3 + c] = src[y×W×4 + x×4 + c] / 255`

---

## Lifecycle OpenCL

```
clGetPlatformIDs()          → Temukan platform (Intel, NVIDIA, AMD)
    ↓
clGetDeviceIDs()            → Temukan device (GPU/CPU)
    ↓
clCreateContext()           → Buat context (binding platform-device)
    ↓
clCreateCommandQueue()      → Antrian perintah (dengan profiling enable)
    ↓
clCreateProgramWithSource() → Load source .cl
    ↓
clBuildProgram()            → Compile kernel untuk device
    ↓
clCreateKernel()            → Buat kernel object dari fungsi di .cl
    ↓
clCreateBuffer()            → Alokasi memori di GPU
    ↓
clEnqueueWriteBuffer()      → Transfer data CPU → GPU
    ↓
clSetKernelArg()            → Set argumen kernel
    ↓
clEnqueueNDRangeKernel()    → Dispatch kernel (parallel execution)
    ↓
clFinish()                  → Tunggu eksekusi selesai
    ↓
clEnqueueReadBuffer()       → Transfer hasil GPU → CPU
    ↓
clReleaseXxx()              → Bebaskan resource
```

---

## Kompleksitas Komputasi

| Operasi       | CPU Complexity | GPU Complexity    | Work-items       |
|---------------|----------------|-------------------|-----------------|
| Resize        | O(W×H)         | O(1) per work-item | W×H             |
| Grayscale     | O(W×H)         | O(1) per work-item | W×H             |
| Gaussian Blur | O(W×H×25)      | O(25) per work-item| W×H             |
| Flatten       | O(W×H×3)       | O(3) per work-item | W×H             |

Untuk 512×512:
- Resize/Grayscale/Flatten: 262,144 operasi parallel
- Gaussian Blur: 262,144 × 25 = 6.5 juta multiply-add parallel

---

## Alasan Pemilihan OpenCL

1. **Cross-platform**: Berjalan di GPU Intel, AMD, NVIDIA, dan CPU
2. **Open standard**: Tidak tergantung vendor (berbeda dengan CUDA/NVIDIA-only)
3. **Low-level control**: Akses langsung ke pipeline GPU tanpa abstraksi berlebih
4. **Pendidikan**: Memaksa pemahaman mendalam tentang arsitektur GPU
5. **Production ready**: Digunakan di industri (khususnya embedded systems, mobile)

---

## Alasan Menggunakan Shared Library (pybind11)

1. **Performance**: Overhead minimal vs ctypes; mendukung zero-copy numpy array
2. **Safety**: Type checking otomatis, exception handling yang proper
3. **Productivity**: Python sebagai controller/orchestrator, C++ sebagai engine
4. **Maintainability**: Separation of concerns yang jelas antara UI dan komputasi
5. **Flexibility**: Python ecosystem (matplotlib, numpy) untuk analisis dan visualisasi

---

## Dependencies

| Library    | Versi    | Keperluan                          |
|------------|----------|------------------------------------|
| OpenCL     | 1.2+     | GPU computing framework            |
| pybind11   | 2.11+    | Python-C++ bridge                  |
| OpenCV     | 4.x      | Baca/tulis/tampil gambar SAJA      |
| numpy      | 1.24+    | Array handling di Python           |
| matplotlib | 3.7+     | Visualisasi benchmark              |
| CMake      | 3.16+    | Build system                       |

---

## Lisensi

MIT License — bebas digunakan untuk keperluan akademis dan komersial.
