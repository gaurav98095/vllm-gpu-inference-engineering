"""Inspect the local PyTorch/CUDA environment and benchmark square GEMM."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch


def median_seconds(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[float, torch.Tensor]:
    """Warm up and return median execution time plus the final result."""
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)
        else:
            start_time = time.perf_counter()
            result = fn()
            samples.append(time.perf_counter() - start_time)
    return statistics.median(samples), result


def inspect_environment() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("No CUDA device found; running CPU-only measurements.")
        return

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {props.name} | compute capability "
            f"{props.major}.{props.minor} | VRAM {props.total_memory / 2**30:.2f} GiB"
        )


def benchmark(size: int, repeats: int, device: torch.device, include_transfer: bool) -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    a_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    b_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)

    cpu_time, cpu_result = median_seconds(lambda: a_cpu @ b_cpu, repeats)

    if device.type != "cuda":
        print(
            f"N={size:5d} | cpu={cpu_time * 1000:.3f} ms | "
            f"cpu_throughput={2 * size**3 / cpu_time / 1e12:.3f} TFLOP/s | gpu=skipped"
        )
        return

    a = a_cpu.to(device)
    b = b_cpu.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_time, gpu_result = median_seconds(lambda: a @ b, repeats)
    max_error = (cpu_result.to(device) - gpu_result).abs().max().item()
    allocated = torch.cuda.memory_allocated(device) / 2**20
    reserved = torch.cuda.memory_reserved(device) / 2**20
    peak = torch.cuda.max_memory_allocated(device) / 2**20

    transfer_text = ""
    if include_transfer:
        transfer_samples: list[float] = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            a_roundtrip = a_cpu.to(device)
            b_roundtrip = b_cpu.to(device)
            result_roundtrip = (a_roundtrip @ b_roundtrip).cpu()
            torch.cuda.synchronize(device)
            del a_roundtrip, b_roundtrip, result_roundtrip
            transfer_samples.append(time.perf_counter() - start_time)
        transfer_time = statistics.median(transfer_samples)
        transfer_text = f" | transfer+compute+copy-back={transfer_time * 1000:.3f} ms"

    print(
        f"N={size:5d} | cpu={cpu_time * 1000:.3f} ms | "
        f"gpu={gpu_time * 1000:.3f} ms | "
        f"speedup={cpu_time / gpu_time:.2f}x | "
        f"gpu_throughput={2 * size**3 / gpu_time / 1e12:.3f} TFLOP/s | "
        f"allocated={allocated:.1f} MiB | reserved={reserved:.1f} MiB | "
        f"peak={peak:.1f} MiB | max_error={max_error:.3e}{transfer_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--include-transfer", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")

    inspect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in args.sizes:
        benchmark(size, args.repeats, device, args.include_transfer)


if __name__ == "__main__":
    main()
