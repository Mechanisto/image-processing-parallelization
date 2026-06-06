"""
utils.py
─────────────────────────────────────────────────────────────────────────────
Utility functions untuk:
  - Loading gambar dari folder menggunakan OpenCV
  - Konversi BGR → RGBA (format yang dibutuhkan kernel OpenCL)
  - Menyimpan hasil ke disk
  - Formatting output
─────────────────────────────────────────────────────────────────────────────
CATATAN:
  OpenCV HANYA digunakan untuk:
    - cv2.imread()   → membaca gambar
    - cv2.imwrite()  → menyimpan gambar
    - cv2.imshow()   → menampilkan gambar
  Semua operasi resize, grayscale, blur, flatten dilakukan di kernel OpenCL.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image_rgba(path: str) -> Optional[np.ndarray]:
    """
    Baca gambar dari disk menggunakan OpenCV dan konversi ke RGBA uint8.

    Args:
        path: Path ke file gambar (JPG atau PNG)

    Returns:
        numpy array shape (H, W, 4) dtype uint8 (RGBA), atau None jika gagal
    """
    # OpenCV membaca dalam format BGR
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARNING] Gagal membaca: {path}", file=sys.stderr)
        return None

    # Konversi BGR → BGRA, lalu reorder ke RGBA
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    # BGRA → RGBA: swap channel 0 dan 2
    img_rgba = img_bgra[:, :, [2, 1, 0, 3]].copy()

    return img_rgba.astype(np.uint8)


def rgba_to_bgr(rgba: np.ndarray) -> np.ndarray:
    """
    Konversi numpy array RGBA (H,W,4) ke BGR (H,W,3) untuk cv2.imwrite.
    """
    return rgba[:, :, [2, 1, 0]]  # RGBA → BGR (drop alpha)


def save_image(path: str, rgba: np.ndarray) -> bool:
    """
    Simpan gambar RGBA ke disk menggunakan cv2.imwrite.

    Args:
        path: Output path
        rgba: numpy array (H, W, 4) uint8

    Returns:
        True jika berhasil
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bgr = rgba_to_bgr(rgba)
    return cv2.imwrite(path, bgr)


def save_flattened(path: str, flat: np.ndarray) -> None:
    """
    Simpan flattened vector sebagai .npy file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, flat)


def collect_images(folder: str,
                   extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
                   ) -> List[str]:
    """
    Kumpulkan semua path gambar dalam folder (rekursif).

    Args:
        folder: Path folder input
        extensions: Tuple ekstensi yang didukung

    Returns:
        List path gambar yang valid
    """
    paths = []
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder tidak ditemukan: {folder}", file=sys.stderr)
        return paths

    for ext in extensions:
        paths.extend(folder_path.rglob(f"*{ext}"))
        paths.extend(folder_path.rglob(f"*{ext.upper()}"))

    # Deduplicate dan sort
    paths = sorted(set(str(p) for p in paths))
    print(f"[INFO] Ditemukan {len(paths)} gambar di '{folder}'")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def display_results(original: np.ndarray,
                    resized: np.ndarray,
                    grayscale: np.ndarray,
                    blurred: np.ndarray,
                    window_name: str = "Pipeline Results") -> None:
    """
    Tampilkan hasil pipeline menggunakan cv2.imshow dalam satu jendela.
    """
    def to_bgr(arr):
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr

    # Resize semua ke ukuran yang sama untuk display
    target_h = 256
    def resize_display(img):
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        bgr = to_bgr(img)
        return cv2.resize(bgr, (new_w, target_h))

    row1 = np.hstack([
        resize_display(original),
        resize_display(resized),
    ])
    row2 = np.hstack([
        resize_display(grayscale),
        resize_display(blurred),
    ])

    # Pad agar sama lebar
    max_w = max(row1.shape[1], row2.shape[1])
    def pad_row(r):
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            return np.hstack([r, pad])
        return r

    display = np.vstack([pad_row(row1), pad_row(row2)])

    # Label
    cv2.putText(display, "Original",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(display, "Resized",   (display.shape[1]//2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(display, "Grayscale", (10, target_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(display, "Gaussian",  (display.shape[1]//2 + 10, target_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager untuk mengukur waktu eksekusi."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.name:
            print(f"  [{self.name}] {self.elapsed_ms:.3f} ms")


def create_output_dirs(base_output: str) -> dict:
    """
    Buat struktur folder output.

    Returns:
        dict dengan keys: resized, grayscale, blurred, flattened
    """
    dirs = {
        'resized':   os.path.join(base_output, 'resized'),
        'grayscale': os.path.join(base_output, 'grayscale'),
        'blurred':   os.path.join(base_output, 'blurred'),
        'flattened': os.path.join(base_output, 'flattened'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def format_timing(timing: dict) -> str:
    """Format timing dict menjadi string yang mudah dibaca."""
    lines = [
        f"  Resize     : {timing.get('resize_ms',    0):.3f} ms",
        f"  Grayscale  : {timing.get('grayscale_ms', 0):.3f} ms",
        f"  Gaussian   : {timing.get('gaussian_ms',  0):.3f} ms",
        f"  Flatten    : {timing.get('flatten_ms',   0):.3f} ms",
        f"  ─────────────────────────",
        f"  TOTAL      : {timing.get('total_ms',     0):.3f} ms",
    ]
    return "\n".join(lines)
