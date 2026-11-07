"""Explore CUDA's execution model on a CUDA-enabled GPU.

The experiments are intentionally small:

* inspect block, thread, global, and warp indices;
* process a large array with a grid-stride loop;
* reduce 256 values with shared memory and block synchronization;
* compare many tiny PyTorch launches with one larger launch.

Run this on the Lightning NVIDIA T4 used by the course.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from numba import cuda, float32


@cuda.jit
def inspect_threads(output):
    """Write block, local, global, and warp indices for each thread."""
    global_id = cuda.grid(1)
    if global_id < output.shape[0]:
        output[global_id, 0] = cuda.blockIdx.x
        output[global_id, 1] = cuda.threadIdx.x
        output[global_id, 2] = global_id
        output[global_id, 3] = cuda.threadIdx.x // 32


@cuda.jit
def double_grid_stride(source, destination, size):
    """Double an array using a grid-stride loop."""
    index = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(index, size, stride):
        destination[i] = 2.0 * source[i]


@cuda.jit
def block_sum(source, output):
    """Reduce 256 values in one block using shared memory."""
    thread_id = cuda.threadIdx.x
    shared = cuda.shared.array(shape=256, dtype=float32)
    shared[thread_id] = source[thread_id]
    cuda.syncthreads()

    stride = 128
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] += shared[thread_id + stride]
        cuda.syncthreads()
        stride //= 2

    if thread_id == 0:
        output[0] = shared[0]


def require_cuda() -> None:
    """Fail early with a useful message when CUDA is unavailable."""
    if not cuda.is_available() or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")


def inspect_gpu() -> None:
    """Print the device properties used to interpret the experiments."""
    device = cuda.get_current_device()
    props = torch.cuda.get_device_properties(0)
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    print(f"GPU: {name}")
    print(f"Compute capability: {device.compute_capability}")
    print(f"Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"Threads per block limit: {device.MAX_THREADS_PER_BLOCK}")
    print(f"Threads per SM: {getattr(props, 'max_threads_per_multi_processor', 'unknown')}")


def run_thread_inspection(blocks: int, threads: int) -> None:
    """Show the index mapping for a small representative launch."""
    size = blocks * threads
    output = cuda.device_array((size, 4), dtype=np.int32)
    inspect_threads[blocks, threads](output)
    cuda.synchronize()
    result = output.copy_to_host()

    print("\n=== Thread and block indices ===")
    print(f"Blocks: {blocks}")
    print(f"Threads per block: {threads}")
    print(f"Total threads: {size}")
    print("Block | Local thread | Global ID | Warp in block")
    for row in result[: min(10, size)]:
        print(f"{row[0]:5d} | {row[1]:12d} | {row[2]:9d} | {row[3]:13d}")

    selected = sorted({0, 31, 32, threads - 1, threads, size - 1})
    print("\nSelected boundary threads:")
    for index in selected:
        if 0 <= index < size:
            print(
                f"global={index:4d} | block={result[index, 0]:2d} | "
                f"local={result[index, 1]:3d} | warp={result[index, 3]:2d}"
            )


def run_grid_stride(size: int, blocks: int, threads: int) -> None:
    """Process more elements than there are launched threads."""
    source = np.arange(size, dtype=np.float32)
    device_source = cuda.to_device(source)
    device_destination = cuda.device_array_like(device_source)

    double_grid_stride[blocks, threads](
        device_source, device_destination, size
    )
    cuda.synchronize()
    destination = device_destination.copy_to_host()

    print("\n=== Grid-stride loop ===")
    print(f"Elements: {size:,}")
    print(f"Threads launched: {blocks * threads:,}")
    print(f"Grid stride: {blocks * threads:,}")
    print(f"Result correct: {np.allclose(destination, source * 2)}")
    print(f"First five results: {destination[:5]}")


def run_block_reduction() -> None:
    """Reduce 256 ones and demonstrate block-wide synchronization."""
    source = np.ones(256, dtype=np.float32)
    device_source = cuda.to_device(source)
    device_output = cuda.device_array(1, dtype=np.float32)

    block_sum[1, 256](device_source, device_output)
    cuda.synchronize()
    result = device_output.copy_to_host()[0]

    print("\n=== Shared-memory block reduction ===")
    print(f"Input values: {source.size}")
    print(f"Expected sum: {source.sum():.1f}")
    print(f"GPU sum: {result:.1f}")
    print(f"Result correct: {np.isclose(result, source.sum())}")


def run_launch_overhead(repeats: int, trials: int) -> None:
    """Compare repeated tiny launches with one larger elementwise launch."""
    device = torch.device("cuda:0")
    small_size = 1024
    x = torch.ones(small_size, device=device)
    y = torch.empty_like(x)
    batched_x = torch.ones((repeats, small_size), device=device)
    batched_y = torch.empty_like(batched_x)

    def many_small_launches() -> None:
        for _ in range(repeats):
            torch.mul(x, 2.0, out=y)

    def one_large_launch() -> None:
        torch.mul(batched_x, 2.0, out=batched_y)

    for _ in range(5):
        many_small_launches()
        one_large_launch()
    torch.cuda.synchronize(device)

    def measure(function) -> float:
        samples = []
        for _ in range(trials):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            function()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples)

    small_ms = measure(many_small_launches)
    large_ms = measure(one_large_launch)

    print("\n=== Launch overhead ===")
    print(f"Elementwise operations: {repeats * small_size:,}")
    print(f"{repeats} small launches: {small_ms:.3f} ms")
    print(f"1 large launch:       {large_ms:.3f} ms")
    print(f"Time ratio:           {small_ms / large_ms:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--skip-thread-inspection", action="store_true")
    parser.add_argument("--skip-grid-stride", action="store_true")
    parser.add_argument("--skip-reduction", action="store_true")
    parser.add_argument("--skip-launch-overhead", action="store_true")
    args = parser.parse_args()

    if (
        args.blocks < 1
        or args.threads < 1
        or args.elements < 1
        or args.repeats < 1
        or args.trials < 1
    ):
        parser.error("all numeric arguments must be positive")
    if args.threads > 1024:
        parser.error("--threads cannot exceed the T4's 1024-thread block limit")

    require_cuda()
    inspect_gpu()

    if not args.skip_thread_inspection:
        run_thread_inspection(args.blocks, args.threads)
    if not args.skip_grid_stride:
        run_grid_stride(args.elements, max(args.blocks, 80), 256)
    if not args.skip_reduction:
        run_block_reduction()
    if not args.skip_launch_overhead:
        run_launch_overhead(args.repeats, args.trials)


if __name__ == "__main__":
    main()
