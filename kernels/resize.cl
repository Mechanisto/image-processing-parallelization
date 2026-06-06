/**
 * @file resize.cl
 * @brief OpenCL Kernel: Nearest Neighbor Image Resize
 *
 * Algoritma: Nearest Neighbor Interpolation
 * Untuk setiap pixel output (dst_x, dst_y), kita mapping balik ke koordinat
 * source image dan ambil pixel terdekat.
 *
 * Mapping Formula:
 *   src_x = dst_x * (src_width  / dst_width)
 *   src_y = dst_y * (src_height / dst_height)
 *
 * Work-item Organization:
 *   - Global ID (0) = pixel column (x)
 *   - Global ID (1) = pixel row    (y)
 *   - Total work-items = dst_width * dst_height
 *
 * Memory Layout: RGBA interleaved (4 bytes per pixel)
 *   pixel[i] = buffer[y * width * 4 + x * 4 + channel]
 */

__kernel void resize_nearest(
    __global const uchar* src,   // Source image buffer (RGBA)
    __global       uchar* dst,   // Destination image buffer (RGBA)
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height
)
{
    // Global work-item ID -> pixel coordinates dalam output image
    int dst_x = get_global_id(0);  // kolom output
    int dst_y = get_global_id(1);  // baris output

    // Boundary check: pastikan work-item tidak melampaui dimensi output
    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Nearest Neighbor Mapping: balik dari koordinat output ke koordinat input
    // Gunakan float untuk presisi, lalu floor ke integer terdekat
    float scale_x = (float)src_width  / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;

    int src_x = (int)(dst_x * scale_x);
    int src_y = (int)(dst_y * scale_y);

    // Clamp: pastikan src koordinat tidak keluar batas
    src_x = min(src_x, src_width  - 1);
    src_y = min(src_y, src_height - 1);

    // Hitung offset dalam buffer (RGBA = 4 channel)
    int src_idx = (src_y * src_width  + src_x) * 4;
    int dst_idx = (dst_y * dst_width  + dst_x) * 4;

    // Copy 4 channel: R, G, B, A
    dst[dst_idx + 0] = src[src_idx + 0];  // Red
    dst[dst_idx + 1] = src[src_idx + 1];  // Green
    dst[dst_idx + 2] = src[src_idx + 2];  // Blue
    dst[dst_idx + 3] = src[src_idx + 3];  // Alpha
}
