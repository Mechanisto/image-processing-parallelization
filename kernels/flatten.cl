/**
 * @file flatten.cl
 * @brief OpenCL Kernel: Image Flatten to 1D Vector
 *
 * Fungsi:
 *   Mengubah image 2D (H × W × C) menjadi vektor 1D float yang dinormalisasi [0,1].
 *   Normalisasi: pixel_value / 255.0
 *
 * Layout Memori Output (Row-Major, Channel-Last):
 *   output[y * width * channels + x * channels + c]
 *   Urutan: R0G0B0 R1G1B1 ... RnGnBn (interleaved / HWC format)
 *   Ini kompatibel dengan format NumPy array HWC yang umum digunakan di ML.
 *
 * Alternatif layout CHW (Channel-First, digunakan PyTorch):
 *   output[c * width * height + y * width + x]
 *   Dapat diaktifkan dengan mengubah indexing di bawah.
 *
 * Indexing:
 *   Input:  src[y * width * 4 + x * 4 + channel]  (RGBA uchar)
 *   Output: dst[y * width * 3 + x * 3 + channel]  (RGB float, tanpa alpha)
 *
 * Setiap work-item memproses satu pixel (3 channel sekaligus).
 * Total work-items = width * height
 */

__kernel void flatten(
    __global const uchar* src,   // Input: RGBA image (uchar)
    __global       float* dst,   // Output: flattened RGB float vector
    const int width,
    const int height
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Source index (RGBA, 4 channel)
    int src_idx = (y * width + x) * 4;

    // Destination index (RGB, 3 channel, HWC layout)
    int dst_idx = (y * width + x) * 3;

    // Normalisasi ke [0.0, 1.0] dan drop alpha channel
    dst[dst_idx + 0] = (float)src[src_idx + 0] / 255.0f;  // R
    dst[dst_idx + 1] = (float)src[src_idx + 1] / 255.0f;  // G
    dst[dst_idx + 2] = (float)src[src_idx + 2] / 255.0f;  // B
}
