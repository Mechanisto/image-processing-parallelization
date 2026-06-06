/**
 * @file gaussian.cl
 * @brief OpenCL Kernel: Gaussian Blur 5×5
 *
 * Kernel Gaussian 5×5 (unnormalized):
 *   1  4  6  4  1
 *   4 16 24 16  4
 *   6 24 36 24  6
 *   4 16 24 16  4
 *   1  4  6  4  1
 *   Total sum = 256 → normalisasi dengan / 256
 *
 * Asal kernel ini dari binomial coefficients C(4,k):
 *   [1,4,6,4,1] adalah row 4 dari segitiga Pascal
 *   Outer product dari [1,4,6,4,1]^T x [1,4,6,4,1] membentuk kernel 5×5
 *
 * Convolution:
 *   Untuk setiap pixel output (x,y), kalikan neighborhood 5×5 dengan kernel
 *   lalu sum dan bagi 256.
 *
 * Boundary Handling: Clamp-to-Edge
 *   Jika koordinat source keluar batas, clamp ke tepi image.
 *   Alternatif: zero-padding (border hitam), mirror, wrap.
 *   Clamp dipilih karena menghasilkan visual paling natural di tepi.
 *
 * Memory Access Pattern:
 *   Global memory read: 25 read per pixel (5×5 neighborhood per channel)
 *   Ini adalah bottleneck; solusi lanjutan: gunakan local memory (shared).
 *
 * Input:  RGBA image (grayscale sudah dikonversi ke RGBA)
 * Output: RGBA image setelah blur
 */

// Definisi kernel Gaussian 5×5
__constant float gaussian_kernel[25] = {
     1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256,  1.0f/256,
     4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256,  4.0f/256,
     6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256,  6.0f/256,
     4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256,  4.0f/256,
     1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256,  1.0f/256
};

__kernel void gaussian_blur(
    __global const uchar* src,   // Input: RGBA image
    __global       uchar* dst,   // Output: RGBA blurred image
    const int width,
    const int height
)
{
    int x = get_global_id(0);  // kolom pixel output
    int y = get_global_id(1);  // baris pixel output

    if (x >= width || y >= height) return;

    // Akumulasi per channel (R, G, B)
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;

    // Iterasi atas 5×5 neighborhood
    // Kernel offset: radius = 2 (half of 5)
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {

            // Koordinat source dengan boundary handling (clamp)
            int sx = clamp(x + kx, 0, width  - 1);
            int sy = clamp(y + ky, 0, height - 1);

            // Offset dalam buffer RGBA
            int src_idx = (sy * width + sx) * 4;

            // Indeks dalam flattened kernel array
            int ki = (ky + 2) * 5 + (kx + 2);
            float w = gaussian_kernel[ki];

            // Akumulasi weighted sum per channel
            sum_r += w * (float)src[src_idx + 0];
            sum_g += w * (float)src[src_idx + 1];
            sum_b += w * (float)src[src_idx + 2];
        }
    }

    // Tulis hasil ke output buffer
    int dst_idx = (y * width + x) * 4;
    dst[dst_idx + 0] = (uchar)clamp(sum_r, 0.0f, 255.0f);
    dst[dst_idx + 1] = (uchar)clamp(sum_g, 0.0f, 255.0f);
    dst[dst_idx + 2] = (uchar)clamp(sum_b, 0.0f, 255.0f);
    dst[dst_idx + 3] = src[(y * width + x) * 4 + 3];  // preserve alpha
}
