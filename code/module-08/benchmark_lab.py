"""Run a reproducible GPU benchmarking laboratory.

The experiments are designed for a Lightning AI NVIDIA T4. They measure
device-side memory traffic, FP16/FP32 GEMM throughput, and the cost of
launching small GPU operations. Results are workload measurements, not
guarantees of peak hardware performance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


DEFAULT_T4_BANDWIDTH_GBS = 300.0
DEFAULT_T4_FP16_TFLOPS = 65.0
DEFAULT_T4_FP32_TFLOPS = 8.1


@dataclass(frozen=True)
class Measurement:
    """A single benchmark result with enough context to interpret it."""

    category: str
    label: str
    milliseconds: float
    throughput: float
    throughput_unit: str
    arithmetic_intensity: float | None = None
    bytes_moved: int | None = None
    flops: int | None = None
    p10_milliseconds: float | None = None
    p90_milliseconds: float | None = None


def require_cuda(device_name: str) -> torch.device:
    """Return the requested CUDA device or fail with an actionable message."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning GPU runtime")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device, for example cuda:0")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device}; only {torch.cuda.device_count()} CUDA device(s) are visible"
        )
    return device


def percentile(samples: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile from a non-empty sample list."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def gpu_samples(
    operation: Callable[[], None],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Measure operation latency with CUDA events in milliseconds."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize_samples(samples: list[float]) -> tuple[float, float, float]:
    """Return median, p10, and p90 latency."""
    return (
        statistics.median(samples),
        percentile(samples, 0.10),
        percentile(samples, 0.90),
    )


def print_samples(label: str, samples: list[float]) -> tuple[float, float, float]:
    """Print a compact latency summary and return median, p10, and p90."""
    median, p10, p90 = summarize_samples(samples)
    print(
        f"{label:28s} median={median:9.3f} ms "
        f"p10={p10:9.3f} ms p90={p90:9.3f} ms"
    )
    return median, p10, p90


def print_device_report(device: torch.device) -> dict[str, object]:
    """Print runtime and device information and return JSON-safe metadata."""
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    metadata: dict[str, object] = {
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_gib": total_bytes / 1024**3,
        "free_memory_gib": free_bytes / 1024**3,
    }

    print("=== Device report ===")
    for key, value in metadata.items():
        display_key = key.replace("_", " ").title()
        if isinstance(value, float) and key.endswith("gib"):
            print(f"{display_key}: {value:.2f} GiB")
        else:
            print(f"{display_key}: {value}")
    return metadata


def memory_experiment(device: torch.device, probe_fraction: float) -> None:
    """Report VRAM capacity and optionally perform a bounded allocation probe."""
    free_before, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    print("\n=== Experiment A: VRAM capacity and availability ===")
    print(f"Physical device memory: {total / 1024**3:.2f} GiB")
    print(f"Free according to CUDA: {free_before / 1024**3:.2f} GiB")
    print(f"PyTorch allocated: {allocated / 1024**3:.2f} GiB")
    print(f"PyTorch reserved: {reserved / 1024**3:.2f} GiB")
    print("Free memory changes with other processes and prior allocations.")

    if probe_fraction == 0:
        print("Allocation probe: skipped (use --probe-fraction to opt in)")
        return

    probe_bytes = int(free_before * probe_fraction)
    print(
        f"Allocation probe: {probe_bytes / 1024**3:.2f} GiB "
        f"({probe_fraction:.0%} of free memory)"
    )
    probe = torch.empty(probe_bytes, dtype=torch.uint8, device=device)
    probe.fill_(0)
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    print(f"Free after probe: {free_after / 1024**3:.2f} GiB")
    del probe
    torch.cuda.empty_cache()


def bandwidth_experiment(
    device: torch.device,
    vector_mib: int,
    warmup: int,
    repeats: int,
) -> list[Measurement]:
    """Measure copy and vector-add useful-data bandwidth."""
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    elements = vector_mib * 1024**2 // element_size
    source = torch.randn(elements, device=device, dtype=torch.float32)
    other = torch.randn_like(source)
    destination = torch.empty_like(source)
    measurements: list[Measurement] = []

    print("\n=== Experiment B: memory bandwidth ===")
    print(f"Vector size: {vector_mib} MiB per tensor ({elements:,} FP32 elements)")

    copy_samples = gpu_samples(
        lambda: destination.copy_(source), device, warmup, repeats
    )
    copy_ms, copy_p10, copy_p90 = print_samples(
        "FP32 device-to-device copy", copy_samples
    )
    copy_bytes = 2 * source.numel() * source.element_size()
    copy_bandwidth = copy_bytes / (copy_ms / 1000) / 1e9
    measurements.append(
        Measurement(
            "bandwidth",
            "FP32 device-to-device copy",
            copy_ms,
            copy_bandwidth,
            "GB/s",
            bytes_moved=copy_bytes,
            p10_milliseconds=copy_p10,
            p90_milliseconds=copy_p90,
        )
    )
    print(f"  Useful-data bandwidth: {copy_bandwidth:.2f} GB/s")

    add_samples = gpu_samples(
        lambda: torch.add(source, other, out=destination),
        device,
        warmup,
        repeats,
    )
    add_ms, add_p10, add_p90 = print_samples("FP32 vector add", add_samples)
    add_bytes = 3 * source.numel() * source.element_size()
    add_flops = source.numel()
    add_bandwidth = add_bytes / (add_ms / 1000) / 1e9
    measurements.append(
        Measurement(
            "bandwidth",
            "FP32 vector add",
            add_ms,
            add_bandwidth,
            "GB/s",
            arithmetic_intensity=add_flops / add_bytes,
            bytes_moved=add_bytes,
            flops=add_flops,
            p10_milliseconds=add_p10,
            p90_milliseconds=add_p90,
        )
    )
    print(f"  Useful-data bandwidth: {add_bandwidth:.2f} GB/s")

    del source, other, destination
    torch.cuda.empty_cache()
    return measurements


def gemm_experiment(
    device: torch.device,
    sizes: list[int],
    warmup: int,
    repeats: int,
) -> list[Measurement]:
    """Measure square FP16 and FP32 GEMM throughput."""
    measurements: list[Measurement] = []
    print("\n=== Experiment C: matrix throughput ===")
    print("The FLOP count is 2 * M * K * N for C = A @ B.")

    for dtype in (torch.float16, torch.float32):
        print(f"\n{dtype} GEMM")
        for size in sizes:
            a = torch.randn((size, size), device=device, dtype=dtype)
            b = torch.randn((size, size), device=device, dtype=dtype)
            c = torch.empty_like(a)

            samples = gpu_samples(
                lambda: torch.mm(a, b, out=c), device, warmup, repeats
            )
            milliseconds, p10, p90 = print_samples(f"{size} x {size}", samples)
            flops = 2 * size**3
            bytes_moved = (a.numel() + b.numel() + c.numel()) * a.element_size()
            tflops = flops / (milliseconds / 1000) / 1e12
            arithmetic_intensity = flops / bytes_moved
            measurements.append(
                Measurement(
                    "gemm",
                    f"{dtype} {size}x{size}",
                    milliseconds,
                    tflops,
                    "TFLOP/s",
                    arithmetic_intensity=arithmetic_intensity,
                    bytes_moved=bytes_moved,
                    flops=flops,
                    p10_milliseconds=p10,
                    p90_milliseconds=p90,
                )
            )
            print(
                f"  Achieved: {tflops:.2f} TFLOP/s | "
                f"idealized AI: {arithmetic_intensity:.1f} FLOPs/byte"
            )
            del a, b, c
            torch.cuda.empty_cache()
    return measurements


def launch_experiment(
    device: torch.device,
    elements: int,
    launch_count: int,
    warmup: int,
    repeats: int,
) -> list[Measurement]:
    """Measure small-kernel latency and host-observed launch cost."""
    source = torch.randn(elements, device=device, dtype=torch.float32)
    destination = torch.empty_like(source)

    def operation() -> None:
        torch.add(source, 1.0, out=destination)

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)

    gpu_samples_per_launch: list[float] = []
    host_samples_per_launch: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launch_count):
            operation()
        end.record()
        end.synchronize()
        gpu_samples_per_launch.append(start.elapsed_time(end) / launch_count)

        torch.cuda.synchronize(device)
        host_start = time.perf_counter()
        for _ in range(launch_count):
            operation()
        torch.cuda.synchronize(device)
        host_samples_per_launch.append(
            (time.perf_counter() - host_start) * 1000 / launch_count
        )

    print("\n=== Experiment D: small-kernel launch overhead ===")
    print(f"Operation: add one scalar to {elements:,} FP32 elements")
    print(f"Launches per sample: {launch_count:,}")
    gpu_ms, gpu_p10, gpu_p90 = print_samples(
        "GPU event time per launch", gpu_samples_per_launch
    )
    host_ms, host_p10, host_p90 = print_samples(
        "Host-observed time per launch", host_samples_per_launch
    )
    print("GPU event time excludes most CPU submission overhead; host time includes it.")

    del source, destination
    torch.cuda.empty_cache()
    return [
        Measurement(
            "launch",
            "GPU event time per launch",
            gpu_ms,
            1 / (gpu_ms / 1000),
            "ops/s",
            p10_milliseconds=gpu_p10,
            p90_milliseconds=gpu_p90,
        ),
        Measurement(
            "launch",
            "Host-observed time per launch",
            host_ms,
            1 / (host_ms / 1000),
            "ops/s",
            p10_milliseconds=host_p10,
            p90_milliseconds=host_p90,
        ),
    ]


def print_roofline_interpretation(
    measurements: list[Measurement],
    memory_bandwidth_gbs: float,
    fp16_tflops: float,
    fp32_tflops: float,
) -> None:
    """Compare measured arithmetic throughput with simplified roofline ceilings."""
    print("\n=== Interpretation: simplified roofline ceilings ===")
    print(
        f"Using {memory_bandwidth_gbs:.1f} GB/s memory bandwidth, "
        f"{fp16_tflops:.1f} FP16 TFLOP/s, and {fp32_tflops:.1f} FP32 TFLOP/s peaks."
    )
    print("These ceilings are reference lines, not claims about achieved performance.")
    for measurement in measurements:
        if measurement.arithmetic_intensity is None:
            continue
        compute_peak = fp32_tflops
        if "float16" in measurement.label:
            compute_peak = fp16_tflops
        bandwidth_ceiling = measurement.arithmetic_intensity * memory_bandwidth_gbs / 1000
        ceiling = min(compute_peak, bandwidth_ceiling)
        measured_tflops = measurement.throughput
        if measurement.throughput_unit == "GB/s":
            measured_tflops = (
                measurement.throughput * measurement.arithmetic_intensity / 1000
            )
        print(
            f"{measurement.label:28s} AI={measurement.arithmetic_intensity:9.2f} | "
            f"ceiling={ceiling:8.3f} TFLOP/s | "
            f"measured={measured_tflops:8.3f} TFLOP/s"
        )


def write_json(path: str, metadata: dict[str, object], measurements: list[Measurement]) -> None:
    """Write machine-readable results for later comparison."""
    payload = {
        "device": metadata,
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved JSON results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-mib", type=int, default=128)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--launch-elements", type=int, default=1024)
    parser.add_argument("--launch-count", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--probe-fraction",
        type=float,
        default=0.0,
        help="optionally allocate this fraction of currently free VRAM (0 disables it)",
    )
    parser.add_argument("--memory-bandwidth-gbs", type=float, default=DEFAULT_T4_BANDWIDTH_GBS)
    parser.add_argument("--fp16-tflops", type=float, default=DEFAULT_T4_FP16_TFLOPS)
    parser.add_argument("--fp32-tflops", type=float, default=DEFAULT_T4_FP32_TFLOPS)
    parser.add_argument("--skip-bandwidth", action="store_true")
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="save results as JSON")
    args = parser.parse_args()

    if (
        args.vector_mib < 1
        or any(size < 1 for size in args.sizes)
        or args.launch_elements < 1
        or args.launch_count < 1
        or args.warmup < 0
        or args.repeats < 1
        or not 0 <= args.probe_fraction < 1
        or args.memory_bandwidth_gbs <= 0
        or args.fp16_tflops <= 0
        or args.fp32_tflops <= 0
    ):
        parser.error(
            "sizes, memory settings, and counts must be positive; "
            "probe fraction must be in [0, 1)"
        )

    device = require_cuda(args.device)
    metadata = print_device_report(device)
    memory_experiment(device, args.probe_fraction)
    measurements: list[Measurement] = []

    if not args.skip_bandwidth:
        measurements.extend(
            bandwidth_experiment(device, args.vector_mib, args.warmup, args.repeats)
        )
    if not args.skip_gemm:
        measurements.extend(gemm_experiment(device, args.sizes, args.warmup, args.repeats))
    if not args.skip_launch:
        measurements.extend(
            launch_experiment(
                device,
                args.launch_elements,
                args.launch_count,
                args.warmup,
                args.repeats,
            )
        )

    print_roofline_interpretation(
        measurements,
        args.memory_bandwidth_gbs,
        args.fp16_tflops,
        args.fp32_tflops,
    )
    if args.json:
        write_json(args.json, metadata, measurements)


if __name__ == "__main__":
    main()
