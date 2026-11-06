"""Measure T4 memory allocation, bandwidth, and access-pattern behavior.

The results are teaching measurements rather than hardware certification:
allocator state, cache effects, launch overhead, and the Lightning runtime
can all influence the observed values.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def require_cuda() -> torch.device:
    """Return the first CUDA device or fail with an actionable message."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on a Lightning T4")
    return torch.device("cuda:0")


def show_memory(device: torch.device) -> None:
    """Print live tensor allocation and PyTorch allocator reservation.

    ``allocated`` tracks live tensors known to PyTorch.  ``reserved`` also
    includes unused blocks held by its caching allocator, so the two values
    are expected to differ after allocations are released.
    """
    allocated = torch.cuda.memory_allocated(device) / 2**20
    reserved = torch.cuda.memory_reserved(device) / 2**20
    print(f"Allocated: {allocated:.2f} MiB")
    print(f"Reserved:  {reserved:.2f} MiB")


def tensor_memory_experiment(device: torch.device) -> None:
    """Show logical tensor size and allocator behavior for a 4096² tensor."""
    print("=== Tensor memory ===")
    show_memory(device)
    # An empty FP32 tensor still reserves storage for 4096 * 4096 * 4 bytes;
    # the values are uninitialized, so no random-data initialization is timed.
    tensor = torch.empty((4096, 4096), dtype=torch.float32, device=device)
    logical_mib = tensor.numel() * tensor.element_size() / 2**20
    print(f"Logical tensor memory: {logical_mib:.2f} MiB")
    show_memory(device)

    # Deleting the Python reference makes the tensor block reusable.  The
    # caching allocator may retain that block in ``reserved`` memory.
    del tensor
    print("After deleting tensor:")
    show_memory(device)
    torch.cuda.empty_cache()
    print("After empty_cache():")
    show_memory(device)


def bandwidth_experiment(device: torch.device, elements: int, repeats: int) -> None:
    """Estimate useful data throughput for a large elementwise add."""
    print("=== Effective global-memory bandwidth ===")
    a = torch.randn(elements, device=device)
    b = torch.randn(elements, device=device)
    c = torch.empty_like(a)

    def add_kernel() -> None:
        # Each output element logically requires two reads and one write.
        torch.add(a, b, out=c)

    # Warm up allocation and kernel selection before measuring steady-state
    # device execution.
    for _ in range(10):
        add_kernel()
    torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(repeats):
        # CUDA events measure elapsed time on the device stream and avoid
        # charging Python scheduling time to the kernel itself.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        add_kernel()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(times_ms)
    # This is useful logical traffic, not a direct DRAM counter: cache hits and
    # transaction overhead mean physical traffic can differ.
    useful_bytes = 3 * elements * a.element_size()
    effective_gbs = useful_bytes / (median_ms / 1000) / 1e9
    print(f"Elements: {elements:,}")
    print(f"Useful traffic estimate: {useful_bytes / 1e6:.2f} MB")
    print(f"Median time: {median_ms:.4f} ms")
    print(f"Effective useful-data rate: {effective_gbs:.2f} GB/s")


def coalescing_experiment(elements: int, repeats: int) -> None:
    """Compare a contiguous gather with increasingly strided gathers."""
    from numba import cuda
    import numpy as np

    print("=== Coalesced versus strided access ===")

    @cuda.jit
    def gather_kernel(src, dst, stride, size):
        # Neighboring threads use neighboring indices when stride=1.  Larger
        # strides separate their addresses and can require more transactions.
        index = cuda.grid(1)
        # Protect the final partially filled block from an out-of-range read.
        if index < size:
            dst[index] = src[index * stride]

    # Allocate enough source elements for the largest stride we test.
    max_stride = 32
    source = cuda.to_device(np.arange(elements * max_stride, dtype=np.float32))
    destination = cuda.device_array(elements, dtype=np.float32)
    threads = 256
    blocks = (elements + threads - 1) // threads

    for stride in (1, 2, 4, 8, 16, 32):
        # Warm up each stride so Numba compilation and first-launch overhead
        # do not contaminate the timed samples.
        gather_kernel[blocks, threads](source, destination, stride, elements)
        cuda.synchronize()
        samples = []
        for _ in range(repeats):
            # Synchronize because the kernel launch itself is asynchronous.
            start = time.perf_counter()
            gather_kernel[blocks, threads](source, destination, stride, elements)
            cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        print(f"Stride {stride:2d}: {statistics.median(samples):.4f} ms")


def main() -> None:
    """Parse options and run the selected memory experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--coalescing", action="store_true")
    parser.add_argument("--elements", type=int, default=8_000_000)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if args.elements < 1 or args.repeats < 1:
        parser.error("elements and repeats must be positive")

    device = require_cuda()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 2**30:.2f} GiB")
    tensor_memory_experiment(device)

    if args.bandwidth:
        bandwidth_experiment(device, args.elements, args.repeats)
    if args.coalescing:
        coalescing_experiment(args.elements // 8, args.repeats)


if __name__ == "__main__":
    main()
