"""Measure memory-bound and compute-bound workloads for a T4 roofline lab."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch


T4_FP32_TFLOPS = 8.1
T4_FP16_TFLOPS = 65.0
T4_BANDWIDTH_GBS = 320.0


@dataclass(frozen=True)
class Measurement:
    """One timed GPU workload."""

    label: str
    milliseconds: float
    tflops: float
    arithmetic_intensity: float
    effective_bandwidth_gbs: float | None = None


def require_cuda() -> torch.device:
    """Return the first CUDA device or fail with an actionable message."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")
    return torch.device("cuda:0")


def gpu_benchmark(
    function,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    """Return median device-side milliseconds using CUDA events."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def print_roofline_summary() -> None:
    """Print the simplified T4 roofline ceilings and balance points."""
    fp32_balance = T4_FP32_TFLOPS * 1_000 / T4_BANDWIDTH_GBS
    fp16_balance = T4_FP16_TFLOPS * 1_000 / T4_BANDWIDTH_GBS

    print("=== Simplified T4 roofline ===")
    print(f"FP32 compute peak: {T4_FP32_TFLOPS:.1f} TFLOPS")
    print(f"FP16 Tensor Core peak: {T4_FP16_TFLOPS:.1f} TFLOPS")
    print(f"Peak memory bandwidth: {T4_BANDWIDTH_GBS:.1f} GB/s")
    print(f"FP32 balance point: {fp32_balance:.1f} FLOPs/byte")
    print(f"FP16 balance point: {fp16_balance:.1f} FLOPs/byte")
    print("These are theoretical ceilings, not measured application results.")


def vector_add_experiment(
    device: torch.device,
    elements: int,
    warmup: int,
    repeats: int,
) -> Measurement:
    """Measure an FP32 vector addition and its useful-data rate."""
    a = torch.randn(elements, device=device, dtype=torch.float32)
    b = torch.randn(elements, device=device, dtype=torch.float32)
    c = torch.empty_like(a)

    def vector_add() -> None:
        torch.add(a, b, out=c)

    milliseconds = gpu_benchmark(vector_add, device, warmup, repeats)
    useful_bytes = 3 * elements * a.element_size()
    bandwidth_gbs = useful_bytes / (milliseconds / 1000) / 1e9
    arithmetic_intensity = 1 / 12
    tflops = elements / (milliseconds / 1000) / 1e12

    measurement = Measurement(
        "FP32 vector add",
        milliseconds,
        tflops,
        arithmetic_intensity,
        bandwidth_gbs,
    )
    print("\n=== Experiment A: memory-bound vector addition ===")
    print(f"Elements: {elements:,}")
    print(f"Median device time: {milliseconds:.3f} ms")
    print(f"Effective useful-data rate: {bandwidth_gbs:.2f} GB/s")
    print(f"Arithmetic intensity: {arithmetic_intensity:.4f} FLOPs/byte")
    return measurement


def gemm_experiment(
    device: torch.device,
    size: int,
    warmup: int,
    repeats: int,
) -> Measurement:
    """Measure a GPU-resident FP16 square GEMM."""
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)
    c = torch.empty((size, size), device=device, dtype=torch.float16)

    def gemm() -> None:
        torch.mm(a, b, out=c)

    milliseconds = gpu_benchmark(gemm, device, warmup, repeats)
    flops = 2 * size**3
    bytes_min = (a.numel() + b.numel() + c.numel()) * a.element_size()
    arithmetic_intensity = flops / bytes_min
    tflops = flops / (milliseconds / 1000) / 1e12

    measurement = Measurement(
        f"FP16 GEMM {size}x{size}",
        milliseconds,
        tflops,
        arithmetic_intensity,
    )
    print("\n=== Experiment B: compute-intensive GEMM ===")
    print(f"Matrix size: {size}x{size}")
    print(f"Median device time: {milliseconds:.3f} ms")
    print(f"Achieved throughput: {tflops:.2f} TFLOPS")
    print(f"Idealized arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")
    return measurement


def projection_experiment(
    device: torch.device,
    batches: list[int],
    hidden_size: int,
    warmup: int,
    repeats: int,
) -> list[Measurement]:
    """Measure how the token dimension changes projection efficiency."""
    weight = torch.randn(
        (hidden_size, hidden_size),
        device=device,
        dtype=torch.float16,
    )
    results = []

    print("\n=== Experiment C: projection shape and batch size ===")
    print("Batch | Time (ms) | TFLOPS | Ideal AI (FLOPs/byte)")
    for batch in batches:
        x = torch.randn(
            (batch, hidden_size),
            device=device,
            dtype=torch.float16,
        )
        y = torch.empty_like(x)

        def projection() -> None:
            torch.mm(x, weight, out=y)

        milliseconds = gpu_benchmark(projection, device, warmup, repeats)
        flops = 2 * batch * hidden_size * hidden_size
        bytes_min = (x.numel() + weight.numel() + y.numel()) * 2
        arithmetic_intensity = flops / bytes_min
        tflops = flops / (milliseconds / 1000) / 1e12
        results.append(
            Measurement(
                f"Projection batch {batch}",
                milliseconds,
                tflops,
                arithmetic_intensity,
            )
        )
        print(
            f"{batch:5d} | {milliseconds:9.3f} | {tflops:7.2f} | "
            f"{arithmetic_intensity:20.2f}"
        )
        del x, y

    del weight
    return results


def save_projection_plot(results: list[Measurement], output: str) -> None:
    """Save a plot of achieved projection throughput versus batch size."""
    import matplotlib.pyplot as plt

    batches = [
        int(measurement.label.rsplit(" ", 1)[-1]) for measurement in results
    ]
    tflops = [measurement.tflops for measurement in results]

    plt.figure(figsize=(8, 4))
    plt.plot(batches, tflops, marker="o")
    plt.xlabel("Input rows / token positions")
    plt.ylabel("Achieved TFLOPS")
    plt.title("T4 projection efficiency versus batch size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved plot: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=16_000_000)
    parser.add_argument("--gemm-size", type=int, default=4096)
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[1, 4, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--skip-projection", action="store_true")
    parser.add_argument("--plot", metavar="PATH")
    args = parser.parse_args()

    if (
        args.elements < 1
        or args.gemm_size < 1
        or args.hidden_size < 1
        or args.warmup < 0
        or args.repeats < 1
        or any(batch < 1 for batch in args.batches)
    ):
        parser.error("all numeric arguments must be positive")

    device = require_cuda()
    properties = torch.cuda.get_device_properties(device)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print_roofline_summary()

    if not args.skip_vector:
        vector_add_experiment(device, args.elements, args.warmup, args.repeats)
    if not args.skip_gemm:
        gemm_experiment(device, args.gemm_size, args.warmup, args.repeats)
    if not args.skip_projection:
        results = projection_experiment(
            device,
            args.batches,
            args.hidden_size,
            args.warmup,
            args.repeats,
        )
        if args.plot:
            save_projection_plot(results, args.plot)


if __name__ == "__main__":
    main()
