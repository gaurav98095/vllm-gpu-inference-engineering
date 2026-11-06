"""Follow a simplified 1B-model execution on an NVIDIA T4.

The script does not load a real language model.  It isolates two useful pieces
of the story: model-sized FP16 storage and a large GPU-resident matrix
multiplication.  Run it on Lightning AI with a T4.
"""

from __future__ import annotations

import argparse
import statistics

import torch


PARAMETER_COUNT = 1_000_000_000
HIDDEN_ROWS = 1_024
HIDDEN_SIZE = 4_096


def require_cuda() -> torch.device:
    """Return the first CUDA device or explain how to select one."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select a Lightning T4 runtime")
    return torch.device("cuda:0")


def show_memory(device: torch.device, label: str) -> None:
    """Print PyTorch live allocation and allocator reservation in GiB."""
    # Allocated is live tensor storage known to PyTorch.  Reserved also
    # includes cached blocks that can be reused by later allocations.
    allocated = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    print(f"{label}: allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB")


def inspect_gpu(device: torch.device) -> None:
    """Print the device facts needed to interpret the measurements."""
    props = torch.cuda.get_device_properties(device)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {props.total_memory / 2**30:.2f} GiB")


def allocate_model_sized_tensor(device: torch.device) -> None:
    """Simulate storage for 1B FP16 parameters and report its footprint."""
    print("=== Experiment A: 1B FP16 parameter storage ===")
    show_memory(device, "Before allocation")

    # This is storage simulation, not a complete model.  An FP16 element uses
    # two bytes, so the tensor represents approximately 2 GB of raw weights.
    weights = torch.empty(PARAMETER_COUNT, dtype=torch.float16, device=device)

    # Touch the allocation so the experiment observes real device writes.  A
    # synchronize makes the memory update complete before counters are read.
    weights.zero_()
    torch.cuda.synchronize(device)
    expected_gib = PARAMETER_COUNT * weights.element_size() / 2**30
    print(f"Raw tensor size: {expected_gib:.3f} GiB")
    show_memory(device, "After allocation")

    # Keep the tensor alive until after the observation above, then explicitly
    # release it so the matrix experiment starts with a cleaner allocator.
    del weights
    torch.cuda.empty_cache()
    show_memory(device, "After release")


def benchmark_matmul(device: torch.device, repeats: int) -> None:
    """Measure a GPU-resident FP16 matrix multiplication with CUDA events."""
    print("=== Experiment B: GPU-resident matrix multiplication ===")
    dtype = torch.float16

    # A and B remain on the GPU for the timed region.  This isolates arithmetic
    # and device memory behavior from host-to-device transfer latency.
    a = torch.randn(HIDDEN_ROWS, HIDDEN_SIZE, dtype=dtype, device=device)
    b = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, dtype=dtype, device=device)

    # Warm up library selection and kernel setup before collecting samples.
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        # CUDA events measure elapsed time on the device stream.  A Python
        # timer alone could stop before asynchronous GPU work has completed.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a @ b
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(samples_ms)
    flops = 2 * HIDDEN_ROWS * HIDDEN_SIZE * HIDDEN_SIZE
    tflops = flops / (median_ms / 1000) / 1e12
    print(f"A shape: {tuple(a.shape)}")
    print(f"B shape: {tuple(b.shape)}")
    print(f"Dtype: {dtype}")
    print(f"Median device time: {median_ms:.3f} ms")
    print(f"Approximate throughput: {tflops:.2f} TFLOP/s")
    show_memory(device, "After matrix benchmark")


def main() -> None:
    """Parse options and run the selected T4 experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocate-model",
        action="store_true",
        help="allocate and initialize a 1B-element FP16 storage simulation",
    )
    parser.add_argument("--matmul-repeats", type=int, default=20)
    args = parser.parse_args()

    if args.matmul_repeats < 1:
        parser.error("--matmul-repeats must be positive")

    device = require_cuda()
    inspect_gpu(device)
    if args.allocate_model:
        allocate_model_sized_tensor(device)
    benchmark_matmul(device, args.matmul_repeats)


if __name__ == "__main__":
    main()
