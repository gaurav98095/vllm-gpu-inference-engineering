"""Run the Module 09 end-to-end GPU benchmarking laboratory.

The suite measures device memory traffic, GEMM throughput and shape efficiency,
small-kernel launch cost, host/device transfers, and PyTorch allocator state on
one CUDA GPU. Results are workload measurements, not hardware guarantees.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "Module 09 requires PyTorch. Install a CUDA-enabled PyTorch build "
            "in the Lightning runtime before running this lab."
        ) from exc
    raise


T4_BANDWIDTH_GBS = 300.0
T4_FP16_TFLOPS = 65.0
T4_FP32_TFLOPS = 8.1


def require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device, for example cuda:0")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"{device} is not visible")
    return device


def benchmark_gpu(
    operation: Callable[[], None],
    device: torch.device,
    warmup: int,
    repeats: int,
    inner: int = 1,
) -> float:
    """Return median CUDA-event milliseconds per operation."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return statistics.median(samples)


def benchmark_wall(
    operation: Callable[[], None],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    """Return median CPU-observed milliseconds including GPU completion."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def memory_report(device: torch.device) -> dict[str, float]:
    allocated = torch.cuda.memory_allocated(device) / 2**20
    reserved = torch.cuda.memory_reserved(device) / 2**20
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mib": round(allocated, 2),
        "reserved_mib": round(reserved, 2),
        "free_mib": round(free / 2**20, 2),
        "total_mib": round(total / 2**20, 2),
    }


def device_report(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    report = {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "visible_gpus": torch.cuda.device_count(),
        "total_vram_gib": props.total_memory / 2**30,
    }
    print("=== Experiment 1: GPU inspection ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("memory:", memory_report(device))
    return report


def vector_add_experiment(
    device: torch.device, vector_mib: int, warmup: int, repeats: int
) -> dict[str, object]:
    elements = vector_mib * 1024**2 // 4
    a = torch.randn(elements, device=device, dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    def operation() -> None:
        torch.add(a, b, out=c)

    milliseconds = benchmark_gpu(operation, device, warmup, repeats)
    useful_bytes = 3 * elements * 4
    bandwidth = useful_bytes / (milliseconds / 1000) / 1e9
    result = {
        "experiment": "FP32 vector addition",
        "milliseconds": milliseconds,
        "effective_bandwidth_gbs": bandwidth,
        "arithmetic_intensity": 1 / 12,
        "elements": elements,
    }
    print("\n=== Experiment 2: memory bandwidth ===")
    print(f"Logical traffic: {useful_bytes / 1e6:.1f} MB")
    print(f"Median device time: {milliseconds:.3f} ms")
    print(f"Useful-data rate: {bandwidth:.2f} GB/s")
    print(f"Arithmetic intensity: {1 / 12:.4f} FLOPs/byte")
    assert torch.allclose(c[:100], a[:100] + b[:100])
    return result


def gemm_experiment(
    device: torch.device,
    sizes: list[int],
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    print("\n=== Experiment 3: GEMM throughput ===")
    results = []
    for size in sizes:
        a = torch.randn((size, size), device=device, dtype=torch.float16)
        b = torch.randn((size, size), device=device, dtype=torch.float16)
        c = torch.empty_like(a)

        def operation() -> None:
            torch.mm(a, b, out=c)

        milliseconds = benchmark_gpu(operation, device, warmup, repeats)
        flops = 2 * size**3
        bytes_min = (a.numel() + b.numel() + c.numel()) * a.element_size()
        result = {
            "experiment": f"FP16 GEMM {size}x{size}",
            "size": size,
            "milliseconds": milliseconds,
            "tflops": flops / (milliseconds / 1000) / 1e12,
            "ideal_arithmetic_intensity": flops / bytes_min,
        }
        results.append(result)
        print(
            f"N={size:4d} | {milliseconds:9.3f} ms | "
            f"{result['tflops']:7.2f} TFLOPS | "
            f"AI={result['ideal_arithmetic_intensity']:7.1f}"
        )
        del a, b, c
    return results


def projection_experiment(
    device: torch.device,
    batches: list[int],
    hidden_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    print("\n=== Experiment 4: GEMV-like versus GEMM-like shapes ===")
    weight = torch.randn((hidden_size, hidden_size), device=device, dtype=torch.float16)
    results = []
    for batch in batches:
        x = torch.randn((batch, hidden_size), device=device, dtype=torch.float16)
        y = torch.empty_like(x)

        def operation() -> None:
            torch.mm(x, weight, out=y)

        milliseconds = benchmark_gpu(operation, device, warmup, repeats)
        flops = 2 * batch * hidden_size * hidden_size
        bytes_min = (x.numel() + weight.numel() + y.numel()) * 2
        result = {
            "experiment": f"Projection batch {batch}",
            "batch": batch,
            "milliseconds": milliseconds,
            "tflops": flops / (milliseconds / 1000) / 1e12,
            "ideal_arithmetic_intensity": flops / bytes_min,
        }
        results.append(result)
        print(
            f"batch={batch:4d} | {milliseconds:9.3f} ms | "
            f"{result['tflops']:7.2f} TFLOPS | "
            f"AI={result['ideal_arithmetic_intensity']:7.1f}"
        )
        del x, y
    del weight
    return results


def launch_experiment(
    device: torch.device,
    elements: int,
    count: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    x_small = torch.ones(elements, device=device)
    y_small = torch.empty_like(x_small)
    x_batched = torch.ones((count, elements), device=device)
    y_batched = torch.empty_like(x_batched)

    def many_launches() -> None:
        for _ in range(count):
            torch.mul(x_small, 2, out=y_small)

    def one_launch() -> None:
        torch.mul(x_batched, 2, out=y_batched)

    many_ms = benchmark_wall(many_launches, device, warmup, repeats)
    one_ms = benchmark_wall(one_launch, device, warmup, repeats)
    result = {
        "experiment": "Kernel launch overhead",
        "many_launches_ms": many_ms,
        "one_launch_ms": one_ms,
        "ratio": many_ms / one_ms,
        "launch_count": count,
    }
    print("\n=== Experiment 5: kernel-launch overhead ===")
    print(f"{count} small launches: {many_ms:.3f} ms")
    print(f"One batched launch: {one_ms:.3f} ms")
    print(f"Ratio: {many_ms / one_ms:.2f}x")
    return result


def transfer_experiment(
    device: torch.device,
    transfer_mib: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    elements = transfer_mib * 1024**2 // 4
    cpu_tensor = torch.randn(elements, dtype=torch.float32)
    gpu_tensor = torch.empty(elements, device=device, dtype=torch.float32)
    cpu_destination = torch.empty_like(cpu_tensor)

    def host_to_device() -> None:
        gpu_tensor.copy_(cpu_tensor)

    def device_to_host() -> None:
        cpu_destination.copy_(gpu_tensor)

    h2d_ms = benchmark_wall(host_to_device, device, warmup, repeats)
    d2h_ms = benchmark_wall(device_to_host, device, warmup, repeats)
    bytes_transferred = cpu_tensor.numel() * cpu_tensor.element_size()
    h2d_gbs = bytes_transferred / (h2d_ms / 1000) / 1e9
    d2h_gbs = bytes_transferred / (d2h_ms / 1000) / 1e9
    result = {
        "experiment": "CPU-GPU transfers",
        "transfer_mib": transfer_mib,
        "h2d_ms": h2d_ms,
        "h2d_gbs": h2d_gbs,
        "d2h_ms": d2h_ms,
        "d2h_gbs": d2h_gbs,
    }
    print("\n=== Experiment 6: CPU-GPU transfers ===")
    print(f"Host to device: {h2d_ms:.3f} ms, {h2d_gbs:.2f} GB/s")
    print(f"Device to host: {d2h_ms:.3f} ms, {d2h_gbs:.2f} GB/s")
    print("These are PCIe/system-path measurements, not HBM bandwidth.")
    return result


def allocation_experiment(device: torch.device) -> dict[str, object]:
    print("\n=== Experiment 7: memory allocation ===")
    before = memory_report(device)
    tensor = torch.empty((4096, 4096), device=device, dtype=torch.float32)
    tensor.zero_()
    torch.cuda.synchronize(device)
    during = memory_report(device)
    del tensor
    after_delete = memory_report(device)
    torch.cuda.empty_cache()
    after_cache = memory_report(device)
    result = {
        "experiment": "PyTorch allocator",
        "logical_tensor_mib": 64.0,
        "before": before,
        "during": during,
        "after_delete": after_delete,
        "after_empty_cache": after_cache,
    }
    for label, row in (("before", before), ("during", during), ("after delete", after_delete), ("after empty_cache", after_cache)):
        print(f"{label:16s}: {row}")
    print("Logical tensor size: 4096 x 4096 x FP32 = 64 MiB")
    return result


def save_plots(
    output_dir: str,
    gemm: list[dict[str, object]],
    projections: list[dict[str, object]],
    launch: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sizes = [row["size"] for row in gemm]
    plt.figure(figsize=(8, 4))
    plt.plot(sizes, [row["tflops"] for row in gemm], marker="o")
    plt.axhline(T4_FP16_TFLOPS, linestyle="--", color="gray", label="T4 FP16 peak")
    plt.xlabel("Square matrix dimension")
    plt.ylabel("Achieved TFLOPS")
    plt.title("FP16 GEMM throughput")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "gemm_tflops.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot([row["batch"] for row in projections], [row["tflops"] for row in projections], marker="o")
    plt.xlabel("Input rows / token positions")
    plt.ylabel("Achieved TFLOPS")
    plt.title("Projection efficiency versus batch size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory / "projection_tflops.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Many launches", "One launch"], [launch["many_launches_ms"], launch["one_launch_ms"]])
    plt.ylabel("Wall time (ms)")
    plt.title("Small-kernel dispatch overhead")
    plt.tight_layout()
    plt.savefig(directory / "launch_overhead.png", dpi=150)
    plt.close()
    print(f"Saved plots in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-mib", type=int, default=128)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--launch-elements", type=int, default=1024)
    parser.add_argument("--launch-count", type=int, default=100)
    parser.add_argument("--transfer-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--plot-dir", metavar="PATH")
    args = parser.parse_args()
    if (
        args.vector_mib < 1 or args.transfer_mib < 1 or args.hidden_size < 1
        or any(size < 1 for size in args.sizes)
        or any(batch < 1 for batch in args.batches)
        or args.launch_elements < 1 or args.launch_count < 1
        or args.warmup < 0 or args.repeats < 1
    ):
        parser.error("sizes, counts, memory values, and repeats must be positive")

    device = require_cuda(args.device)
    if device.index is not None:
        torch.cuda.set_device(device)
    metadata = device_report(device)
    allocation = allocation_experiment(device)
    vector = vector_add_experiment(device, args.vector_mib, args.warmup, args.repeats)
    gemm = gemm_experiment(device, args.sizes, args.warmup, args.repeats)
    projections = projection_experiment(device, args.batches, args.hidden_size, args.warmup, args.repeats)
    launch = launch_experiment(device, args.launch_elements, args.launch_count, args.warmup, args.repeats)
    transfers = transfer_experiment(device, args.transfer_mib, args.warmup, args.repeats)
    payload = {
        "metadata": metadata,
        "reference": {"t4_bandwidth_gbs": T4_BANDWIDTH_GBS, "t4_fp16_tflops": T4_FP16_TFLOPS, "t4_fp32_tflops": T4_FP32_TFLOPS},
        "allocation": allocation,
        "vector_add": vector,
        "gemm": gemm,
        "projections": projections,
        "launch": launch,
        "transfers": transfers,
    }
    if args.plot_dir:
        save_plots(args.plot_dir, gemm, projections, launch)
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON report: {args.json}")


if __name__ == "__main__":
    main()
