"""
main.py
─────────────────────────────────────────────────────────────────────────────
Aplikasi utama: GPU Image Processing Pipeline menggunakan OpenCL

Alur Data:
  1. Python membaca gambar dengan cv2.imread()
  2. Konversi BGR → RGBA numpy array
  3. Kirim ke C++ melalui pybind11 shared library
  4. C++ OpenCL host melakukan:
     a. Transfer data CPU → GPU (clEnqueueWriteBuffer)
     b. Dispatch resize kernel
     c. Dispatch grayscale kernel
     d. Dispatch gaussian blur kernel
     e. Dispatch flatten kernel
     f. Transfer hasil GPU → CPU (clEnqueueReadBuffer)
  5. Python menerima kembali hasil sebagai numpy array
  6. Simpan hasil dengan cv2.imwrite()

Penggunaan:
  python main.py --input datasets/ --output outputs/ --size 256
  python main.py --input datasets/ --show
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ── Import modul yang dibutuhkan ──────────────────────────────────────────────
# utils menggunakan OpenCV HANYA untuk imread/imwrite/imshow
from utils import (
    load_image_rgba,
    save_image,
    save_flattened,
    collect_images,
    display_results,
    create_output_dirs,
    format_timing,
)

# ── Import C++ extension module ───────────────────────────────────────────────
# gpu_pipeline.so / gpu_pipeline.pyd dicompile dari bindings.cpp
try:
    import gpu_pipeline
except ImportError as e:
    print(f"[ERROR] Gagal import gpu_pipeline: {e}")
    print("Pastikan sudah build dengan CMake. Lihat README.md untuk instruksi.")
    sys.exit(1)


def get_kernels_dir() -> str:
    """Tentukan path ke folder kernels/ dari posisi script ini."""
    script_dir = Path(__file__).parent
    kernels_dir = script_dir.parent / "kernels"
    if not kernels_dir.exists():
        raise FileNotFoundError(f"Folder kernels tidak ditemukan: {kernels_dir}")
    return str(kernels_dir)


def process_single_image(pipeline: 'gpu_pipeline.GPUImagePipeline',
                          img_path: str,
                          out_dirs: dict,
                          out_size: int,
                          show: bool = False) -> dict:
    """
    Proses satu gambar melalui pipeline GPU.

    Args:
        pipeline: Instance GPUImagePipeline (sudah diinisialisasi)
        img_path: Path ke gambar input
        out_dirs: Dict output directories
        out_size: Ukuran output (width = height = out_size)
        show: Tampilkan hasil di jendela

    Returns:
        Dict timing results
    """
    stem = Path(img_path).stem
    print(f"\n[Processing] {Path(img_path).name} → {out_size}×{out_size}")

    # ── 1. Baca gambar menggunakan OpenCV ────────────────────────────────────
    img_rgba = load_image_rgba(img_path)
    if img_rgba is None:
        return {}

    orig_h, orig_w = img_rgba.shape[:2]
    print(f"  Input: {orig_w}×{orig_h} pixels")

    # ── 2. Kirim ke GPU melalui pybind11 ─────────────────────────────────────
    result = pipeline.process(img_rgba, out_size, out_size)

    timing = result['timing']
    print(f"  Timing GPU:\n{format_timing(timing)}")

    # ── 3. Simpan hasil menggunakan OpenCV ───────────────────────────────────
    # resized
    save_image(os.path.join(out_dirs['resized'],   f"{stem}_{out_size}.png"),
               result['resized'])

    # grayscale
    save_image(os.path.join(out_dirs['grayscale'], f"{stem}_{out_size}_gray.png"),
               result['grayscale'])

    # blurred
    save_image(os.path.join(out_dirs['blurred'],   f"{stem}_{out_size}_blur.png"),
               result['blurred'])

    # flattened vector
    save_flattened(os.path.join(out_dirs['flattened'], f"{stem}_{out_size}_flat.npy"),
                   result['flattened'])

    flat_shape = result['flattened'].shape
    print(f"  Flattened shape: {flat_shape}  (= {out_size}×{out_size}×3 = {out_size*out_size*3})")

    # ── 4. Opsional: tampilkan hasil ─────────────────────────────────────────
    if show:
        display_results(
            original=img_rgba,
            resized=result['resized'],
            grayscale=result['grayscale'],
            blurred=result['blurred'],
            window_name=f"Pipeline: {Path(img_path).name} → {out_size}×{out_size}"
        )

    return timing


def main():
    parser = argparse.ArgumentParser(
        description="GPU Image Processing Pipeline (OpenCL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Proses semua gambar di folder datasets/, output ke outputs/
  python main.py --input datasets/ --output outputs/

  # Proses dengan ukuran 512x512
  python main.py --input datasets/ --size 512

  # Proses dan tampilkan hasil
  python main.py --input datasets/ --show

  # Proses file tunggal
  python main.py --input datasets/photo.jpg --size 256
        """
    )

    parser.add_argument('--input',  type=str, default='../datasets',
                        help='Folder atau file gambar input')
    parser.add_argument('--output', type=str, default='../outputs',
                        help='Folder output hasil')
    parser.add_argument('--size',   type=int, default=256,
                        choices=[128, 256, 512],
                        help='Ukuran output (128, 256, atau 512)')
    parser.add_argument('--show',   action='store_true',
                        help='Tampilkan hasil di jendela (membutuhkan GUI)')
    parser.add_argument('--device-info', action='store_true',
                        help='Print informasi GPU/device')
    args = parser.parse_args()

    # ── Inisialisasi Pipeline ─────────────────────────────────────────────────
    kernels_dir = get_kernels_dir()
    print(f"[Init] Kernel directory: {kernels_dir}")
    print("[Init] Menginisialisasi OpenCL pipeline...\n")

    pipeline = gpu_pipeline.GPUImagePipeline(kernels_dir)

    if args.device_info:
        pipeline.print_device_info()

    # ── Kumpulkan gambar input ────────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    else:
        image_paths = collect_images(str(input_path))

    if not image_paths:
        print("[ERROR] Tidak ada gambar yang ditemukan. Masukkan gambar ke folder datasets/")
        sys.exit(1)

    # ── Buat output directories ───────────────────────────────────────────────
    out_dirs = create_output_dirs(args.output)
    print(f"[Output] Folder: {args.output}")

    # ── Proses setiap gambar ──────────────────────────────────────────────────
    total_images = len(image_paths)
    print(f"\n[Pipeline] Memproses {total_images} gambar → {args.size}×{args.size}\n")
    print("=" * 60)

    all_timings = []
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{total_images}]", end="")
        timing = process_single_image(
            pipeline=pipeline,
            img_path=path,
            out_dirs=out_dirs,
            out_size=args.size,
            show=args.show,
        )
        if timing:
            all_timings.append(timing)

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    if all_timings:
        print("\n" + "=" * 60)
        print(f"[Summary] Processed {len(all_timings)} gambar @ {args.size}×{args.size}")
        print(f"  Avg Resize     : {np.mean([t['resize_ms']    for t in all_timings]):.3f} ms")
        print(f"  Avg Grayscale  : {np.mean([t['grayscale_ms'] for t in all_timings]):.3f} ms")
        print(f"  Avg Gaussian   : {np.mean([t['gaussian_ms']  for t in all_timings]):.3f} ms")
        print(f"  Avg Flatten    : {np.mean([t['flatten_ms']   for t in all_timings]):.3f} ms")
        print(f"  Avg TOTAL      : {np.mean([t['total_ms']     for t in all_timings]):.3f} ms")
        print(f"\nOutput tersimpan di: {args.output}")


if __name__ == "__main__":
    main()
