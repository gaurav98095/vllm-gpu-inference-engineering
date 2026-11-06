"""Launch and benchmark a simple vector-add CUDA kernel with Numba.

The example makes the CUDA hierarchy concrete: a grid contains blocks, each
block contains threads, and every thread computes one output element.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from numba import cuda


@cuda.jit
def vector_add(a, b, c, size):
    """Add one pair of elements per CUDA thread."""
    # cuda.grid(1) converts the block and thread coordinates into one global
    # element index for this one-dimensional workload.
    index = cuda.grid(1)
    # The final block may contain extra threads, so guard every array access.
    if index < size:
        c[index] = a[index] + b[index]


def inspect_gpu() -> None:
    """Print the CUDA device limits that constrain the launch."""
    device = cuda.get_current_device()
    print(f"GPU: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"Compute capability: {device.compute_capability}")
    print(f"Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"Threads per block limit: {device.MAX_THREADS_PER_BLOCK}")


def launch_vector_add(size: int, threads_per_block: int) -> None:
    """Launch one vector-add kernel and verify every output element."""
    # Host arrays make the input easy to inspect; cuda.to_device copies them
    # to device memory before the kernel is launched.
    host_a = np.arange(size, dtype=np.float32)
    host_b = np.arange(size, dtype=np.float32) * 10
    device_a = cuda.to_device(host_a)
    device_b = cuda.to_device(host_b)
    device_c = cuda.device_array(size, dtype=np.float32)

    # Ceiling division ensures enough threads for a size that is not evenly
    # divisible by the selected block size.
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    vector_add[blocks_per_grid, threads_per_block](
        device_a, device_b, device_c, size
    )
    # Kernel launches are asynchronous relative to Python.  Synchronize before
    # copying the result back and before reporting correctness.
    cuda.synchronize()
    result = device_c.copy_to_host()

    print(f"Elements: {size}")
    print(f"Blocks: {blocks_per_grid}")
    print(f"Threads per block: {threads_per_block}")
    print(f"Total threads launched: {blocks_per_grid * threads_per_block}")
    print(f"First 10 outputs: {result[:10]}")
    np.testing.assert_allclose(result, host_a + host_b)
    print("Correctness: PASS")


def benchmark_block_sizes(size: int, repeats: int) -> None:
    """Compare launch-plus-execution time for several block sizes."""
    # Keep the data and useful work fixed so the main variable is the number
    # of threads assigned to each block.
    host_a = np.ones(size, dtype=np.float32)
    host_b = np.ones(size, dtype=np.float32)
    device_a = cuda.to_device(host_a)
    device_b = cuda.to_device(host_b)
    device_c = cuda.device_array_like(device_a)

    for threads_per_block in (32, 64, 128, 256, 512, 1024):
        # More threads per block means fewer blocks for the same element count.
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        vector_add[blocks_per_grid, threads_per_block](
            device_a, device_b, device_c, size
        )
        # Warm up this configuration so compilation and first-launch costs do
        # not dominate the samples.
        cuda.synchronize()

        samples = []
        for _ in range(repeats):
            # Synchronization makes the host timer include the complete kernel
            # execution.  This is intentionally simple; device events can be
            # introduced later for more isolated kernel timing.
            start = time.perf_counter()
            vector_add[blocks_per_grid, threads_per_block](
                device_a, device_b, device_c, size
            )
            cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)

        print(
            f"Threads/block: {threads_per_block:4d} | "
            f"Blocks: {blocks_per_grid:6d} | "
            f"Median time: {statistics.median(samples):.4f} ms"
        )

    np.testing.assert_allclose(device_c.copy_to_host(), 2.0)


def main() -> None:
    """Parse options and run either correctness or block-size measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if args.elements < 1 or args.threads < 1 or args.repeats < 1:
        parser.error("elements, threads, and repeats must be positive")
    if not cuda.is_available():
        parser.error("CUDA is not available; run this on a CUDA-enabled machine")

    inspect_gpu()
    if args.benchmark:
        benchmark_block_sizes(args.elements, args.repeats)
    else:
        launch_vector_add(args.elements, args.threads)


if __name__ == "__main__":
    main()
