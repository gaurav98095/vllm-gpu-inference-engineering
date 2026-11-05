"""Inspect the local PyTorch/CUDA environment and benchmark square GEMM."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch


def median_seconds(
    fn: Callable[[], torch.Tensor], repeats: int, device: torch.device
) -> tuple[float, torch.Tensor]:
    """Warm up and return median execution time plus the final result."""
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
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

    cpu_time, cpu_result = median_seconds(
        lambda: a_cpu @ b_cpu, repeats, torch.device("cpu")
    )

    if device.type != "cuda":
        print(
            f"N={size:5d} | cpu={cpu_time * 1000:.3f} ms | "
            f"cpu_throughput={2 * size**3 / cpu_time / 1e12:.3f} TFLOP/s | gpu=skipped"
        )
        return

    a = a_cpu.to(device)
    b = b_cpu.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_time, gpu_result = median_seconds(lambda: a @ b, repeats, device)
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


def benchmark_transfer(size: int, repeats: int, device: torch.device) -> None:
    """Compare GPU-resident compute with transfer + compute + copy-back."""
    if device.type != "cuda":
        print("Transfer experiment skipped: CUDA is not available.")
        return

    generator = torch.Generator(device="cpu").manual_seed(17)
    a_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    b_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    for _ in range(5):
        _ = a_gpu @ b_gpu
    torch.cuda.synchronize(device)

    compute_time, _ = median_seconds(lambda: a_gpu @ b_gpu, repeats, device)

    roundtrip_samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        c_cpu = (a @ b).cpu()
        torch.cuda.synchronize(device)
        del a, b, c_cpu
        roundtrip_samples.append(time.perf_counter() - start_time)

    total_time = statistics.median(roundtrip_samples)
    print(f"Transfer experiment N={size}")
    print(f"Compute only: {compute_time * 1000:.3f} ms")
    print(f"Transfer + compute + return: {total_time * 1000:.3f} ms")


def benchmark_projection_batches(
    hidden_size: int, batches: tuple[int, ...], repeats: int, device: torch.device
) -> None:
    """Measure how batch size changes a projection's work and memory footprint."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    weight = torch.randn((hidden_size, hidden_size), device=device, dtype=dtype)
    weight_bytes = weight.numel() * weight.element_size()

    print(
        f"Projection weights: {weight.shape}, {dtype}, "
        f"{weight_bytes / 2**20:.2f} MiB"
    )
    for batch in batches:
        x = torch.randn((batch, hidden_size), device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed, y = median_seconds(lambda: x @ weight, repeats, device)
        flops = 2 * batch * hidden_size * hidden_size
        memory = ""
        if device.type == "cuda":
            memory = (
                f" | allocated={torch.cuda.memory_allocated(device) / 2**20:.1f} MiB"
                f" | peak={torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB"
            )
        print(
            f"B={batch:3d} | X={tuple(x.shape)} | Y={tuple(y.shape)} | "
            f"FLOPs={flops / 1e9:.3f}B | time={elapsed * 1000:.3f} ms"
            f"{memory}"
        )
        del x, y
    del weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--include-transfer", action="store_true")
    parser.add_argument("--include-projection", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")

    inspect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in args.sizes:
        benchmark(size, args.repeats, device, args.include_transfer)

    print(
        f"1B parameters in FP16: "
        f"{1_000_000_000 * torch.tensor([], dtype=torch.float16).element_size() / 2**30:.2f} GiB"
    )
    if args.include_transfer:
        benchmark_transfer(2048, min(args.repeats, 20), device)
    if args.include_projection:
        benchmark_projection_batches(4096, (1, 64), min(args.repeats, 20), device)


if __name__ == "__main__":
    main()
