"""Standalone Module 02 CUDA thread/block vector-add lab.

Run with:
    python code/module-02/gpu_threads.py --benchmark
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from numba import cuda


@cuda.jit
def vector_add(a, b, c, size):
    """Map one CUDA thread to one vector element."""
    index = cuda.grid(1)
    if index < size:
        c[index] = a[index] + b[index]


def inspect_gpu() -> None:
    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    print("GPU:", name)
    print("Compute capability:", device.compute_capability)
    print("SMs:", device.MULTIPROCESSOR_COUNT)
    print("Max threads/block:", device.MAX_THREADS_PER_BLOCK)


def run_vector_add(size: int, threads_per_block: int) -> None:
    """Launch, synchronize, copy back, and verify a vector addition."""
    host_a = np.arange(size, dtype=np.float32)
    host_b = np.arange(size, dtype=np.float32) * 10
    device_a = cuda.to_device(host_a)
    device_b = cuda.to_device(host_b)
    device_c = cuda.device_array(size, dtype=np.float32)
    blocks = (size + threads_per_block - 1) // threads_per_block
    vector_add[blocks, threads_per_block](device_a, device_b, device_c, size)
    cuda.synchronize()
    result = device_c.copy_to_host()
    np.testing.assert_allclose(result, host_a + host_b)
    print("Elements:", size)
    print("Blocks:", blocks)
    print("Threads/block:", threads_per_block)
    print("First outputs:", result[:10])
    print("Correctness: PASS")


def benchmark_block_sizes(size: int, repeats: int) -> None:
    """Keep useful work fixed while changing threads per block."""
    host_a = np.ones(size, dtype=np.float32)
    host_b = np.ones(size, dtype=np.float32)
    device_a = cuda.to_device(host_a)
    device_b = cuda.to_device(host_b)
    device_c = cuda.device_array_like(device_a)
    for threads in (32, 64, 128, 256, 512, 1024):
        blocks = (size + threads - 1) // threads
        vector_add[blocks, threads](device_a, device_b, device_c, size)
        cuda.synchronize()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            vector_add[blocks, threads](device_a, device_b, device_c, size)
            cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        print(f"threads/block={threads:4d} blocks={blocks:6d} median_ms={statistics.median(samples):.4f}")
    np.testing.assert_allclose(device_c.copy_to_host(), 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not cuda.is_available():
        parser.error("CUDA is unavailable; run on a CUDA-enabled GPU")
    inspect_gpu()
    if args.benchmark:
        benchmark_block_sizes(args.elements, args.repeats)
    else:
        run_vector_add(args.elements, args.threads)


if __name__ == "__main__":
    main()
