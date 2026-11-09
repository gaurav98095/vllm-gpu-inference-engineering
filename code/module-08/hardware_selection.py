"""Compare inference GPU constraints and inspect the attached CUDA device.

The capacity and bandwidth calculations run without CUDA.  ``--probe``
inspects the GPU visible to the current Python process and ``--topology``
prints the system's NVIDIA GPU topology when ``nvidia-smi`` is available.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    architecture: str
    memory_type: str
    memory_gb: float
    bandwidth_gbs: float
    precision: str
    interconnect: str
    form_factor: str

    @property
    def usable_gb(self) -> float:
        """Return a conservative 90% planning budget."""
        return self.memory_gb * 0.90


GPU_SPECS = (
    GpuSpec("T4", "Turing", "GDDR6", 16, 300, "FP16 · INT8 · INT4", "PCIe Gen3 x16", "PCIe"),
    GpuSpec("A100 80GB", "Ampere", "HBM2e", 80, 2_040, "FP16 · BF16 · TF32", "PCIe/NVLink", "SXM"),
    GpuSpec("H100", "Hopper", "HBM3", 80, 3_350, "FP16 · BF16 · FP8", "PCIe/NVLink", "SXM"),
    GpuSpec("H200", "Hopper", "HBM3e", 141, 4_800, "FP16 · BF16 · FP8", "PCIe/NVLink", "SXM"),
    GpuSpec("B200", "Blackwell", "HBM3e", 180, 8_000, "FP16 · BF16 · FP8 · FP4", "NVLink/NVSwitch", "HGX/SXM"),
)


def print_specs() -> None:
    """Print representative capacity, bandwidth, and precision constraints."""
    print("=== Representative inference GPU specifications ===")
    print("GPU        Architecture  Memory       VRAM   BW         Precision                 Link")
    for spec in GPU_SPECS:
        print(
            f"{spec.name:10s} {spec.architecture:12s} {spec.memory_type:11s} "
            f"{spec.memory_gb:5.0f} GB {spec.bandwidth_gbs:7.0f} GB/s "
            f"{spec.precision:25s} {spec.interconnect}"
        )
    print("These are representative manufacturer specifications, not application benchmarks.")
    print("Check the exact SKU, form factor, clocks, and platform before deployment.")


def estimate_budget(
    parameters_b: float,
    weight_bytes: float,
    reserve_gb: float,
    kv_gb_per_1k_tokens: float,
    tokens: int,
) -> dict[str, float]:
    """Return a transparent decimal-GB serving-memory estimate."""
    weights_gb = parameters_b * weight_bytes
    kv_gb = kv_gb_per_1k_tokens * tokens / 1_000
    return {
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "reserve_gb": reserve_gb,
        "total_gb": weights_gb + kv_gb + reserve_gb,
    }


def print_budget(
    parameters_b: float,
    weight_bytes: float,
    reserve_gb: float,
    kv_gb_per_1k_tokens: float,
    tokens: int,
) -> None:
    """Print the estimate and conservative fit status for every GPU."""
    budget = estimate_budget(
        parameters_b, weight_bytes, reserve_gb, kv_gb_per_1k_tokens, tokens
    )
    print("\n=== Estimated serving-memory budget ===")
    print(f"Parameters: {parameters_b:.2f}B")
    print(f"Weight storage: {budget['weights_gb']:.2f} GB")
    print(f"KV-cache estimate: {budget['kv_gb']:.2f} GB for {tokens:,} active tokens")
    print(f"Runtime/workspace reserve: {budget['reserve_gb']:.2f} GB")
    print(f"Estimated total: {budget['total_gb']:.2f} GB")
    print("GPU        Conservative budget  Fits?  Headroom")
    for spec in GPU_SPECS:
        headroom = spec.usable_gb - budget["total_gb"]
        print(
            f"{spec.name:10s} {spec.usable_gb:18.2f} GB  "
            f"{'yes' if headroom >= 0 else 'no':>5s}  {headroom:8.2f} GB"
        )
    print("This is planning arithmetic; measure the real checkpoint and runtime allocations.")


def print_lower_bounds(weight_gb: float) -> None:
    """Print idealized one-pass weight-streaming times."""
    print(f"\n=== Idealized lower bound for reading {weight_gb:.2f} GB once ===")
    print("GPU        Bandwidth       Lower bound")
    for spec in GPU_SPECS:
        milliseconds = weight_gb / spec.bandwidth_gbs * 1_000
        print(f"{spec.name:10s} {spec.bandwidth_gbs:7.0f} GB/s     {milliseconds:8.3f} ms")
    print("These are memory-interface lower bounds, not token-generation latencies.")


def probe_device(show_topology: bool) -> None:
    """Inspect all visible CUDA devices and optionally print topology."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--probe requires a PyTorch installation") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning GPU runtime")

    print("\n=== Attached CUDA devices ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"\nGPU {index}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  VRAM: {total_bytes / 1024**3:.2f} GiB")
        print(f"  Free VRAM: {free_bytes / 1024**3:.2f} GiB")
        print(f"  SM count: {props.multi_processor_count}")

    if show_topology:
        print("\n=== nvidia-smi topology ===")
        subprocess.run(["nvidia-smi", "topo", "-m"], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-specs", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--model-params-b", type=float)
    parser.add_argument("--weight-bytes", type=float, default=2.0)
    parser.add_argument("--reserve-gb", type=float, default=3.0)
    parser.add_argument("--kv-gb-per-1k-tokens", type=float, default=0.125)
    parser.add_argument("--tokens", type=int, default=4_096)
    parser.add_argument("--lower-bound-gb", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.topology and not args.probe:
        raise SystemExit("--topology requires --probe")
    if not any((args.show_specs, args.probe, args.model_params_b is not None, args.lower_bound_gb)):
        args.show_specs = True
    if (
        args.model_params_b is not None and args.model_params_b <= 0
        or args.weight_bytes <= 0
        or args.reserve_gb < 0
        or args.kv_gb_per_1k_tokens < 0
        or args.tokens < 0
        or args.lower_bound_gb is not None and args.lower_bound_gb <= 0
    ):
        raise SystemExit("memory and model arguments are out of range")

    if args.show_specs:
        print_specs()
    if args.model_params_b is not None:
        print_budget(
            args.model_params_b,
            args.weight_bytes,
            args.reserve_gb,
            args.kv_gb_per_1k_tokens,
            args.tokens,
        )
    if args.lower_bound_gb is not None:
        print_lower_bounds(args.lower_bound_gb)
    if args.probe:
        probe_device(args.topology)


if __name__ == "__main__":
    main()
