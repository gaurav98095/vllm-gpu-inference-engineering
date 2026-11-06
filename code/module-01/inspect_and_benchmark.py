"""Inspect CUDA and compare CPU/GPU matrix-multiplication behavior.

The measurements intentionally separate GPU-resident compute from a
transfer-inclusive path.  That distinction is important for inference:
model weights normally remain on the GPU, while repeatedly moving inputs or
outputs across the PCIe boundary adds latency.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch


def median_seconds(
    fn: Callable[[], torch.Tensor], repeats: int, device: torch.device
) -> tuple[float, torch.Tensor]:
    """Warm up and return median execution time plus the final result.

    CUDA launches are asynchronous, so CPU wall-clock timing is not enough
    for device work.  CUDA events measure time on the GPU stream itself;
    CPU timing is used only for the CPU path.
    """
    # The first call may include library setup and kernel selection overhead.
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            # Recording events on the same stream brackets the device work.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)
        else:
            # CPU operations are synchronous, so a high-resolution host timer
            # is appropriate for this branch.
            start_time = time.perf_counter()
            result = fn()
            samples.append(time.perf_counter() - start_time)
    return statistics.median(samples), result


def inspect_environment() -> None:
    """Print the runtime and hardware information used by the experiments."""
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
    """Benchmark one square GEMM and optionally include host-device copies."""
    # Generate one reproducible CPU input pair so CPU and GPU results can be
    # compared for correctness rather than benchmarking different data.
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

    # This is the normal steady-state serving path: inputs are already on the
    # GPU before the timed matrix multiplication begins.
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
        # This path deliberately measures host-to-device copies, compute, and
        # device-to-host copy-back together.  It is an end-to-end latency
        # illustration, not a replacement for the resident-compute timing.
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
    """Compare GPU-resident compute with transfer + compute + copy-back.

    The host timer is synchronized before and after the round trip so it
    includes the asynchronous CUDA work as well as the explicit copies.
    """
    if device.type != "cuda":
        print("Transfer experiment skipped: CUDA is not available.")
        return

    generator = torch.Generator(device="cpu").manual_seed(17)
    a_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    b_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # Warm up the GEMM path before collecting samples.
    for _ in range(5):
        _ = a_gpu @ b_gpu
    torch.cuda.synchronize(device)

    compute_time, _ = median_seconds(lambda: a_gpu @ b_gpu, repeats, device)

    roundtrip_samples: list[float] = []
    for _ in range(repeats):
        # Ensure the previous iteration has finished before starting the host
        # timer; otherwise asynchronous work could leak between samples.
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
    """Measure how batch size changes projection work and memory footprint."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Keep one weight matrix fixed while changing only the number of token
    # positions that use it.  This models batch-one versus batched inference.
    weight = torch.randn((hidden_size, hidden_size), device=device, dtype=dtype)
    weight_bytes = weight.numel() * weight.element_size()

    print(
        f"Projection weights: {weight.shape}, {dtype}, "
        f"{weight_bytes / 2**20:.2f} MiB"
    )
    for batch in batches:
        # The weight storage is unchanged; the input/output tensors and FLOPs
        # grow with the number of token positions in the batch.
        x = torch.randn((batch, hidden_size), device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed, y = median_seconds(lambda: x @ weight, repeats, device)
        # One multiply and one add are counted as roughly two FLOPs.
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
    """Parse options and run the selected Module 01 experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--include-transfer", action="store_true")
    parser.add_argument("--include-projection", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")

    # Fall back to CPU-only measurements so the script remains useful outside
    # Lightning AI, while still making the missing CUDA device explicit.
    inspect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in args.sizes:
        benchmark(size, args.repeats, device, args.include_transfer)

    # This is a storage estimate only; activations, KV cache, and workspaces
    # also consume VRAM during real model inference.
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
