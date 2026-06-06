/**
 * @file grayscale.cl
 * @brief OpenCL Kernel: RGB to Grayscale Conversion
 *
 * Formula ITU-R BT.601 (Luminance):
 *   Y = 0.299*R + 0.587*G + 0.114*B
 *
 * Koefisien ini berdasarkan sensitivitas mata manusia:
 *   - Green channel paling dominan (0.587) karena mata paling sensitif terhadap hijau
 *   - Red channel menengah (0.299)
 *   - Blue channel paling kecil (0.114) karena mata kurang sensitif terhadap biru
 *
 * Paralelisasi:
 *   - Setiap work-item memproses SATU pixel secara independen
 *   - Tidak ada dependency antar pixel → embarrassingly parallel
 *   - Total work-items = width * height
 *
 * Input:  RGBA image (4 bytes per pixel)
 * Output: Grayscale image disimpan sebagai RGBA (R=G=B=Y, A=255)
 *         agar kompatibel dengan OpenCV imwrite
 */

__kernel void grayscale(
    __global const uchar* src,   // Input: RGBA image
    __global       uchar* dst,   // Output: Grayscale sebagai RGBA
    const int width,
    const int height
)
{
    // Setiap work-item = satu pixel
    int x = get_global_id(0);  // kolom
    int y = get_global_id(1);  // baris

    // Boundary check
    if (x >= width || y >= height) return;

    // Hitung offset pixel dalam buffer RGBA
    int idx = (y * width + x) * 4;

    // Ambil komponen warna
    float r = (float)src[idx + 0];
    float g = (float)src[idx + 1];
    float b = (float)src[idx + 2];

    // Terapkan formula luminance ITU-R BT.601
    uchar luma = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);

    // Simpan sebagai RGBA grayscale (R=G=B=luma, A=255)
    dst[idx + 0] = luma;
    dst[idx + 1] = luma;
    dst[idx + 2] = luma;
    dst[idx + 3] = 255;
}
