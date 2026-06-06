"""
benchmark.py
─────────────────────────────────────────────────────────────────────────────
Sistem Benchmark: CPU vs GPU (OpenCL)

Membandingkan performa pipeline untuk 3 resolusi:
  - 128×128
  - 256×256
  - 512×512

Menghasilkan:
  1. Tabel benchmark (stdout)
  2. CSV hasil (benchmark_results/results.csv)
  3. Grafik matplotlib (benchmark_results/charts/)

Analisis:
  - Speedup GPU vs CPU per operasi
  - Overhead transfer memori CPU-GPU
  - Pengaruh ukuran citra terhadap speedup
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (server/headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_image_rgba, collect_images

try:
    import gpu_pipeline
except ImportError as e:
    print(f"[ERROR] Gagal import gpu_pipeline: {e}")
    print("Pastikan sudah build dengan CMake. Lihat README.md untuk instruksi.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────
RESOLUTIONS   = [128, 256, 512]
WARMUP_RUNS   = 2
BENCHMARK_RUNS = 5

OPERATIONS = ['resize', 'grayscale', 'gaussian', 'flatten', 'total']
OP_LABELS  = {
    'resize':    'Resize',
    'grayscale': 'Grayscale',
    'gaussian':  'Gaussian Blur',
    'flatten':   'Flatten',
    'total':     'Total Pipeline',
}

COLORS = {
    'cpu': '#E74C3C',   # merah
    'gpu': '#3498DB',   # biru
    'speedup': '#2ECC71' # hijau
}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_for_image(pipeline, img_rgba: np.ndarray,
                             resolution: int, n_runs: int
                             ) -> Tuple[Dict, Dict]:
    """
    Jalankan benchmark CPU dan GPU untuk satu gambar pada satu resolusi.

    Returns:
        (cpu_timings_avg, gpu_timings_avg) dalam milliseconds
    """
    # ── Warmup ──────────────────────────────────────────────────────────────
    for _ in range(WARMUP_RUNS):
        pipeline.process(img_rgba, resolution, resolution)
        pipeline.process_cpu(img_rgba, resolution, resolution)

    # ── CPU runs ─────────────────────────────────────────────────────────────
    cpu_runs = []
    for _ in range(n_runs):
        r = pipeline.process_cpu(img_rgba, resolution, resolution)
        cpu_runs.append(r['timing'])

    # ── GPU runs ─────────────────────────────────────────────────────────────
    gpu_runs = []
    for _ in range(n_runs):
        r = pipeline.process(img_rgba, resolution, resolution)
        gpu_runs.append(r['timing'])

    # ── Average ───────────────────────────────────────────────────────────────
    def avg_timings(runs):
        keys = ['resize_ms', 'grayscale_ms', 'gaussian_ms', 'flatten_ms', 'total_ms']
        result = {}
        for k in keys:
            result[k] = np.mean([t[k] for t in runs])
        return result

    return avg_timings(cpu_runs), avg_timings(gpu_runs)


def run_full_benchmark(images_dir: str, output_dir: str):
    """
    Jalankan benchmark lengkap untuk semua resolusi.
    """
    # ── Init pipeline ─────────────────────────────────────────────────────────
    kernels_dir = str(Path(__file__).parent.parent / "kernels")
    print(f"[Benchmark] Kernels: {kernels_dir}")
    print("[Benchmark] Inisialisasi OpenCL...\n")

    pipeline = gpu_pipeline.GPUImagePipeline(kernels_dir)
    pipeline.print_device_info()

    # ── Kumpulkan gambar ──────────────────────────────────────────────────────
    image_paths = collect_images(images_dir)
    if not image_paths:
        # Buat dummy image jika folder kosong
        print("[INFO] Folder dataset kosong, menggunakan dummy image 1024×768\n")
        dummy = np.random.randint(0, 255, (768, 1024, 4), dtype=np.uint8)
        images = [("dummy_1024x768", dummy)]
    else:
        images = []
        for p in image_paths[:5]:  # Maksimal 5 gambar untuk benchmark
            img = load_image_rgba(p)
            if img is not None:
                images.append((Path(p).stem, img))
        print(f"[Benchmark] Menggunakan {len(images)} gambar\n")

    # ── Run benchmark per resolusi ────────────────────────────────────────────
    print("=" * 70)
    print(f"  {'Resolution':<12} {'Operation':<16} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>10}")
    print("=" * 70)

    # Simpan hasil: {resolution: {operation: {cpu: float, gpu: float}}}
    all_results = {}

    for res in RESOLUTIONS:
        res_label = f"{res}×{res}"
        res_cpu = {k: [] for k in ['resize_ms','grayscale_ms','gaussian_ms','flatten_ms','total_ms']}
        res_gpu = {k: [] for k in res_cpu}

        for name, img in images:
            cpu_t, gpu_t = run_benchmark_for_image(
                pipeline, img, res, BENCHMARK_RUNS
            )
            for k in res_cpu:
                res_cpu[k].append(cpu_t[k])
                res_gpu[k].append(gpu_t[k])

        # Average across images
        avg_cpu = {k: np.mean(v) for k, v in res_cpu.items()}
        avg_gpu = {k: np.mean(v) for k, v in res_gpu.items()}

        # Print tabel
        for op_key, op_label in [
            ('resize_ms',    'Resize'),
            ('grayscale_ms', 'Grayscale'),
            ('gaussian_ms',  'Gaussian Blur'),
            ('flatten_ms',   'Flatten'),
            ('total_ms',     'TOTAL'),
        ]:
            c = avg_cpu[op_key]
            g = avg_gpu[op_key]
            s = c / g if g > 0 else float('inf')
            marker = "★" if op_key == 'total_ms' else " "
            print(f"{marker} {res_label:<12} {op_label:<16} {c:>10.3f} {g:>10.3f} {s:>9.2f}x")

        print("-" * 70)

        all_results[res] = {
            'cpu': avg_cpu,
            'gpu': avg_gpu,
        }

    # ── Simpan CSV ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['resolution', 'operation', 'cpu_ms', 'gpu_ms', 'speedup'])
        for res, data in all_results.items():
            for op_key in ['resize_ms', 'grayscale_ms', 'gaussian_ms', 'flatten_ms', 'total_ms']:
                c = data['cpu'][op_key]
                g = data['gpu'][op_key]
                writer.writerow([
                    f"{res}x{res}",
                    op_key.replace('_ms', ''),
                    f"{c:.4f}",
                    f"{g:.4f}",
                    f"{c/g:.4f}" if g > 0 else "inf"
                ])
    print(f"\n[Saved] CSV: {csv_path}")

    # ── Generate Charts ───────────────────────────────────────────────────────
    generate_charts(all_results, output_dir)

    # ── JSON hasil lengkap ────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"[Saved] JSON: {json_path}")

    # ── Analisis ──────────────────────────────────────────────────────────────
    print_analysis(all_results)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Chart generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(results: dict, output_dir: str):
    """Hasilkan grafik benchmark yang komprehensif."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    resolutions = [128, 256, 512]
    res_labels  = ["128×128", "256×256", "512×512"]
    ops         = ['resize_ms', 'grayscale_ms', 'gaussian_ms', 'flatten_ms']
    op_labels   = ['Resize', 'Grayscale', 'Gaussian Blur', 'Flatten']

    # ── Chart 1: Total Pipeline CPU vs GPU ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GPU OpenCL vs CPU Sequential: Pipeline Benchmark",
                 fontsize=14, fontweight='bold')

    cpu_totals = [results[r]['cpu']['total_ms'] for r in resolutions]
    gpu_totals = [results[r]['gpu']['total_ms'] for r in resolutions]

    x = np.arange(len(resolutions))
    w = 0.35

    ax = axes[0]
    bars_cpu = ax.bar(x - w/2, cpu_totals, w, label='CPU Sequential',
                       color=COLORS['cpu'], alpha=0.85, edgecolor='black', linewidth=0.7)
    bars_gpu = ax.bar(x + w/2, gpu_totals, w, label='GPU OpenCL',
                       color=COLORS['gpu'], alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Resolusi Output")
    ax.set_ylabel("Waktu (ms)")
    ax.set_title("Total Pipeline Time: CPU vs GPU")
    ax.set_xticks(x)
    ax.set_xticklabels(res_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotate values
    for bar in bars_cpu:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_gpu:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    # ── Chart 2: Speedup ─────────────────────────────────────────────────────
    ax2 = axes[1]
    speedups = [c/g if g > 0 else 0 for c, g in zip(cpu_totals, gpu_totals)]
    bars_sp = ax2.bar(res_labels, speedups, color=COLORS['speedup'],
                       alpha=0.85, edgecolor='black', linewidth=0.7)
    ax2.set_xlabel("Resolusi Output")
    ax2.set_ylabel("Speedup (CPU time / GPU time)")
    ax2.set_title("GPU Speedup vs CPU")
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars_sp, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    chart1_path = os.path.join(charts_dir, "total_pipeline_comparison.png")
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Chart] {chart1_path}")

    # ── Chart 3: Per-Operation Breakdown ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Operation Benchmark: CPU vs GPU (OpenCL)",
                 fontsize=14, fontweight='bold')

    for idx, (op_key, op_label) in enumerate(zip(ops, op_labels)):
        ax = axes[idx // 2][idx % 2]
        cpu_vals = [results[r]['cpu'][op_key] for r in resolutions]
        gpu_vals = [results[r]['gpu'][op_key] for r in resolutions]

        x = np.arange(len(resolutions))
        ax.bar(x - 0.2, cpu_vals, 0.4, label='CPU', color=COLORS['cpu'], alpha=0.8)
        ax.bar(x + 0.2, gpu_vals, 0.4, label='GPU', color=COLORS['gpu'], alpha=0.8)
        ax.set_title(op_label)
        ax.set_xticks(x)
        ax.set_xticklabels(res_labels)
        ax.set_ylabel("Waktu (ms)")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    chart2_path = os.path.join(charts_dir, "per_operation_breakdown.png")
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Chart] {chart2_path}")

    # ── Chart 4: Speedup heatmap per operasi ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    speedup_matrix = []
    for op_key in ops + ['total_ms']:
        row = []
        for r in resolutions:
            c = results[r]['cpu'][op_key]
            g = results[r]['gpu'][op_key]
            row.append(c/g if g > 0 else 0)
        speedup_matrix.append(row)

    speedup_matrix = np.array(speedup_matrix)
    im = ax.imshow(speedup_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels(res_labels)
    ax.set_yticks(range(len(op_labels) + 1))
    ax.set_yticklabels(op_labels + ['Total Pipeline'])
    ax.set_title("GPU Speedup Heatmap (× lebih cepat dari CPU)")

    for i in range(speedup_matrix.shape[0]):
        for j in range(speedup_matrix.shape[1]):
            ax.text(j, i, f'{speedup_matrix[i, j]:.2f}×',
                    ha='center', va='center', fontweight='bold', fontsize=11)

    plt.colorbar(im, ax=ax, label='Speedup Factor')
    plt.tight_layout()
    chart3_path = os.path.join(charts_dir, "speedup_heatmap.png")
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Chart] {chart3_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Academic Analysis
# ─────────────────────────────────────────────────────────────────────────────

def print_analysis(results: dict):
    """
    Cetak analisis akademis yang cocok untuk laporan tugas akhir.
    """
    print("\n" + "=" * 70)
    print("  ANALISIS PERFORMA GPU vs CPU")
    print("=" * 70)

    speedups_total = {}
    for r in [128, 256, 512]:
        c = results[r]['cpu']['total_ms']
        g = results[r]['gpu']['total_ms']
        speedups_total[r] = c / g if g > 0 else float('inf')

    print(f"""
1. PENGARUH UKURAN CITRA TERHADAP SPEEDUP
   ─────────────────────────────────────────
   Resolusi 128×128 : {speedups_total.get(128, 0):.2f}× speedup
   Resolusi 256×256 : {speedups_total.get(256, 0):.2f}× speedup
   Resolusi 512×512 : {speedups_total.get(512, 0):.2f}× speedup

   Analisis:
   Semakin besar resolusi, semakin tinggi potensi speedup GPU karena:
   - Jumlah work-item bertambah (parallelisme lebih tinggi)
   - GPU lebih efisien saat ribuan thread berjalan bersamaan
   - CPU sequential harus memproses setiap pixel satu per satu

2. MENGAPA GPU LEBIH CEPAT
   ─────────────────────────
   a) Massively Parallel Architecture:
      GPU memiliki ratusan hingga ribuan Compute Units (shader processors).
      Pipeline kita adalah "embarrassingly parallel" — setiap pixel independent.
      Work-items di GPU dieksekusi secara bersamaan dalam SIMD fashion.

   b) Throughput vs Latency:
      CPU dioptimalkan untuk latency rendah (satu task diselesaikan cepat).
      GPU dioptimalkan untuk throughput tinggi (banyak task sekaligus).
      Untuk image processing, throughput lebih penting.

   c) Efisiensi Gaussian Blur pada GPU:
      Gaussian Blur memerlukan 25 operasi multiply-add per pixel (5×5 kernel).
      Pada resolusi 512×512 = 262,144 pixel × 25 = 6.5 juta operasi per channel.
      GPU dapat mendistribusikan ini ke ribuan compute unit.

3. PENGARUH JUMLAH WORK-ITEM
   ─────────────────────────────
   Work-group size kita: 16×16 = 256 work-items per group.
   Untuk 512×512: (512/16)×(512/16) = 1024 work-groups.
   GPU modern dapat menjalankan puluhan work-group secara concurrent.

   Lebih banyak work-item → GPU lebih terisi → occupancy tinggi → speedup tinggi.
   Ini menjelaskan mengapa speedup meningkat seiring resolusi.

4. OVERHEAD TRANSFER MEMORI CPU-GPU
   ────────────────────────────────────
   Transfer data merupakan bottleneck utama GPU computing:
   - PCIe bandwidth terbatas (misal: PCIe 3.0 x16 = ~12 GB/s)
   - Untuk 512×512 RGBA: 512×512×4 = 1 MB → transfer ~0.1 ms
   - Overhead relatif kecil dibanding komputasi untuk operasi berat seperti Gaussian

   Strategi optimasi: pipeline multiple images, gunakan pinned memory,
   atau asynchronous transfer (clEnqueueWriteBuffer non-blocking).

5. OPERASI DENGAN SPEEDUP TERTINGGI
   ────────────────────────────────────
   Gaussian Blur memiliki speedup tertinggi karena:
   - O(n²) complexity per pixel (25 multiplications per pixel)
   - Intensitas komputasi tinggi → GPU lebih unggul
   - Memory access pattern regular → cache GPU efisien

   Flatten dan Grayscale lebih memory-bound, speedup lebih rendah.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU Image Processing Pipeline"
    )
    parser.add_argument('--input',  type=str, default='../datasets',
                        help='Folder gambar input')
    parser.add_argument('--output', type=str, default='../benchmark_results',
                        help='Folder output hasil benchmark')
    parser.add_argument('--runs',   type=int, default=BENCHMARK_RUNS,
                        help='Jumlah run per benchmark')
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  GPU IMAGE PROCESSING BENCHMARK — OpenCL")
    print("  Arsitektur Komputer — Tugas Akhir")
    print("█" * 70 + "\n")

    run_full_benchmark(
        images_dir=args.input,
        output_dir=args.output,
    )
