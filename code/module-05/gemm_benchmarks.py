"""Benchmark GEMM shapes and precision on a CUDA GPU.

The measurements are intended for a Lightning NVIDIA T4.  They report
device-side latency and achieved TFLOP/s, not a guarantee of peak hardware
performance.  Warmup, CUDA events, and median samples keep launch and Python
timing noise out of the reported device time as much as practical.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Measurement:
    """One matrix multiplication measurement."""

    label: str
    m: int
    k: int
    n: int
    dtype: torch.dtype
    milliseconds: float
    tflops: float


def require_cuda() -> torch.device:
    """Return the first CUDA device or fail with a useful instruction."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")
    return torch.device("cuda:0")


def benchmark_matmul(
    device: torch.device,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    label: str,
    warmup: int,
    repeats: int,
) -> Measurement:
    """Measure ``C = A @ B`` with GPU-resident inputs."""
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    c = torch.empty((m, n), device=device, dtype=dtype)

    for _ in range(warmup):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    milliseconds = statistics.median(samples_ms)
    flops = 2 * m * k * n
    tflops = flops / (milliseconds / 1000.0) / 1e12
    return Measurement(label, m, k, n, dtype, milliseconds, tflops)


def print_measurement(measurement: Measurement) -> None:
    """Print one measurement in a compact, copyable format."""
    print(
        f"{measurement.label:18s} "
        f"shape=({measurement.m:4d},{measurement.k:4d},{measurement.n:4d}) "
        f"dtype={str(measurement.dtype).removeprefix('torch.'):7s} "
        f"time={measurement.milliseconds:9.3f} ms "
        f"TFLOP/s={measurement.tflops:7.2f}"
    )


def print_tile_model(tile_sizes: list[int]) -> None:
    """Print the simplified shared-memory tile reuse model."""
    print("=== Tile-size model (FP32 inputs) ===")
    print("Tile | Input storage | FLOPs | Arithmetic intensity | Threads")
    for tile in tile_sizes:
        storage_kib = 2 * tile * tile * 4 / 1024
        flops = 2 * tile**3
        intensity = flops / (storage_kib * 1024)
        print(
            f"{tile:4d} | {storage_kib:10.1f} KiB | {flops:5,d} | "
            f"{intensity:20.1f} FLOPs/byte | {tile * tile:7,d}"
        )


def measure_precision_error(device: torch.device) -> None:
    """Compare FP16-input/FP16-accumulate and FP32 matrix products."""
    print("=== FP16 versus FP32 numerical difference ===")
    torch.manual_seed(42)
    a32 = torch.randn((512, 512), device=device, dtype=torch.float32)
    b32 = torch.randn((512, 512), device=device, dtype=torch.float32)
    c32 = a32 @ b32
    c16 = a32.half() @ b32.half()
    difference = (c32 - c16.float()).abs()
    print(f"Mean absolute difference: {difference.mean().item():.6f}")
    print(f"Maximum absolute difference: {difference.max().item():.6f}")


def run_cpu_gemm(size: int) -> None:
    """Run deliberately slow triple-loop FP32 GEMM for comparison."""
    import numpy as np

    print(f"=== Optional naive CPU GEMM ({size}x{size}) ===")
    rng = np.random.default_rng(42)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    c = np.zeros((size, size), dtype=np.float32)
    started = time.perf_counter()
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c[i, j] += a[i, k] * b[k, j]
    elapsed_ms = (time.perf_counter() - started) * 1000
    correct = np.allclose(c, a @ b, atol=1e-3)
    print(f"Python loop time: {elapsed_ms:.2f} ms")
    print(f"Correct against NumPy: {correct}")


def save_plot(results: list[Measurement], output: str) -> None:
    """Save a square-GEMM throughput plot without requiring plotting by default."""
    import matplotlib.pyplot as plt

    sizes = [result.m for result in results]
    tflops = [result.tflops for result in results]
    plt.figure(figsize=(8, 4))
    plt.plot(sizes, tflops, marker="o")
    plt.xlabel("Square matrix dimension")
    plt.ylabel("Achieved TFLOP/s")
    plt.title("NVIDIA T4 — FP16 GEMM Throughput")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved plot: {output}")


def main() -> None:
    """Run square-GEMM, precision, and projection-shape experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--skip-fp32", action="store_true")
    parser.add_argument("--skip-tile-model", action="store_true")
    parser.add_argument("--precision-error", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="run the deliberately slow CPU loop")
    parser.add_argument("--cpu-size", type=int, default=64)
    parser.add_argument(
        "--plot",
        metavar="PATH",
        help="save the square FP16 throughput plot (requires matplotlib)",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1 or args.cpu_size < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if any(size < 1 for size in args.sizes) or any(tokens < 1 for tokens in args.tokens):
        parser.error("matrix sizes and token counts must be positive")

    device = require_cuda()
    properties = torch.cuda.get_device_properties(device)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print()

    if not args.skip_tile_model:
        print_tile_model([8, 16, 32, 64])
        print()

    print("=== Experiment A: square FP16 GEMM ===")
    square_results: list[Measurement] = []
    for size in args.sizes:
        result = benchmark_matmul(
            device, size, size, size, torch.float16, f"N={size}", args.warmup, args.repeats
        )
        square_results.append(result)
        print_measurement(result)

    if args.plot:
        save_plot(square_results, args.plot)

    if not args.skip_fp32:
        print("\n=== Experiment B: FP16 versus FP32 ===")
        for dtype in (torch.float16, torch.float32):
            for size in args.sizes:
                print_measurement(
                    benchmark_matmul(
                        device,
                        size,
                        size,
                        size,
                        dtype,
                        f"{dtype} N={size}",
                        args.warmup,
                        args.repeats,
                    )
                )

        if args.precision_error:
            measure_precision_error(device)

    elif args.precision_error:
        measure_precision_error(device)

    print("\n=== Experiment C: projection shapes (M tokens, K=N=4096) ===")
    for tokens in args.tokens:
        print_measurement(
            benchmark_matmul(
                device,
                tokens,
                4096,
                4096,
                torch.float16,
                f"tokens={tokens}",
                args.warmup,
                args.repeats,
            )
        )

    if args.cpu:
        run_cpu_gemm(args.cpu_size)


if __name__ == "__main__":
    main()
