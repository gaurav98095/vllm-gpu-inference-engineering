"""Compare inference-oriented GPU constraints and inspect a CUDA device.

The capacity and roofline calculations are intentionally hardware-agnostic, so
they can be run without CUDA. ``--probe`` inspects the GPU attached to the
current Python environment; ``--measure-h2d`` optionally measures host-to-
device PCIe/system transfers and should not be confused with HBM bandwidth.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    architecture: str
    memory_type: str
    memory_gb: float
    bandwidth_gbs: float
    fp32_tflops: float
    fp16_tflops: float
    nvlink_gbs: float | None
    pcie: str
    form_factor: str
    tdp_w: int

    @property
    def fp16_balance(self) -> float:
        return self.fp16_tflops * 1_000 / self.bandwidth_gbs

    @property
    def usable_gb(self) -> float:
        return self.memory_gb * 0.90


GPU_SPECS = (
    GpuSpec("T4", "Turing", "GDDR6", 16, 320, 8.1, 65, None, "Gen3 x16", "PCIe", 70),
    GpuSpec("A100 80GB SXM", "Ampere", "HBM2e", 80, 2_039, 19.5, 312, 600, "Gen4", "SXM", 400),
    GpuSpec("H100 80GB SXM", "Hopper", "HBM3", 80, 3_350, 67, 989, 900, "Gen5", "SXM", 700),
)


def gb(value: float) -> str:
    return f"{value:,.2f} GB"


def print_specs() -> None:
    print("=== Representative inference GPU specifications ===")
    print(
        "GPU               Arch    Memory   VRAM   BW        FP32   "
        "FP16/BF16  AI balance  NVLink"
    )
    for spec in GPU_SPECS:
        nvlink = "-" if spec.nvlink_gbs is None else f"{spec.nvlink_gbs:g} GB/s"
        print(
            f"{spec.name:18s} {spec.architecture:7s} {spec.memory_type:7s} "
            f"{spec.memory_gb:4.0f} GB {spec.bandwidth_gbs:7.0f} "
            f"{spec.fp32_tflops:6.1f} {spec.fp16_tflops:9.1f} "
            f"{spec.fp16_balance:10.1f} {nvlink:>9s}"
        )
    print("AI balance uses dense FP16/BF16 peak divided by memory bandwidth.")
    print("All values are theoretical published ceilings, not benchmark results.")
    print("Form factors and TDP: " + "; ".join(
        f"{spec.name}={spec.form_factor}, {spec.tdp_w} W" for spec in GPU_SPECS
    ))


def estimate_budget(
    parameters_b: float,
    weight_bytes: float,
    reserve_gb: float,
    kv_gb_per_1k_tokens: float,
    tokens: int,
) -> dict[str, float]:
    """Return a transparent, decimal-GB planning estimate."""
    weights_gb = parameters_b * weight_bytes
    kv_gb = kv_gb_per_1k_tokens * tokens / 1_000
    return {
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "reserve_gb": reserve_gb,
        "total_gb": weights_gb + reserve_gb + kv_gb,
    }


def print_budget(
    parameters_b: float,
    weight_bytes: float,
    reserve_gb: float,
    kv_gb_per_1k_tokens: float,
    tokens: int,
) -> None:
    budget = estimate_budget(
        parameters_b,
        weight_bytes,
        reserve_gb,
        kv_gb_per_1k_tokens,
        tokens,
    )
    weights_gb = budget["weights_gb"]
    kv_gb = budget["kv_gb"]
    total_gb = budget["total_gb"]

    print("\n=== Estimated model memory budget ===")
    print(f"Parameters: {parameters_b:.3f} billion")
    print(f"Weight bytes/parameter: {weight_bytes:.3f}")
    print(f"Weights: {gb(weights_gb)}")
    print(f"KV-cache estimate: {gb(kv_gb)} for {tokens:,} tokens")
    print(f"Reserve/workspace: {gb(reserve_gb)}")
    print(f"Estimated total: {gb(total_gb)}")
    print("GPU                 Usable at 90%   Fits?   Headroom")
    for spec in GPU_SPECS:
        headroom = spec.usable_gb - total_gb
        status = "yes" if headroom >= 0 else "no"
        print(
            f"{spec.name:18s} {spec.usable_gb:10.2f} GB   {status:>4s}   "
            f"{headroom:8.2f} GB"
        )
    print(
        "This is a planning estimate. Check the real checkpoint, quantization "
        "metadata, allocator behavior, and vLLM runtime allocations."
    )


def cuda_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--probe requires a PyTorch installation") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning GPU runtime")
    return torch


def probe_device(measure_h2d: bool, transfer_mb: int, repeats: int) -> None:
    torch = cuda_device()
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print("\n=== Attached CUDA device ===")
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Total memory: {total_bytes / 1e9:.2f} GB")
    print(f"Free memory: {free_bytes / 1e9:.2f} GB")
    print(f"Multiprocessors: {props.multi_processor_count}")

    if not measure_h2d:
        return

    if transfer_mb < 1 or repeats < 1:
        raise ValueError("transfer_mb and repeats must be positive")
    print("\n=== Optional host-to-device transfer test ===")
    print("This estimates the host/system path, not HBM bandwidth.")
    host = torch.empty(
        transfer_mb * 1024 * 1024 // 4,
        dtype=torch.float32,
        pin_memory=True,
    )
    destination = torch.empty_like(host, device=device)
    for _ in range(5):
        destination.copy_(host, non_blocking=True)
    torch.cuda.synchronize(device)

    samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        destination.copy_(host, non_blocking=True)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    median_ms = statistics.median(samples_ms)
    rate = transfer_mb / (median_ms / 1_000) / 1_000
    print(f"Transfer: {transfer_mb} MiB")
    print(f"Median device time: {median_ms:.3f} ms")
    print(f"Approximate rate: {rate:.2f} GB/s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-specs", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--measure-h2d", action="store_true")
    parser.add_argument("--transfer-mb", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model-params-b", type=float)
    parser.add_argument("--weight-bytes", type=float, default=2.0)
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    parser.add_argument("--kv-gb-per-1k-tokens", type=float, default=0.125)
    parser.add_argument("--tokens", type=int, default=4_096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.measure_h2d and not args.probe:
        raise SystemExit("--measure-h2d requires --probe")
    if not any((args.show_specs, args.probe, args.model_params_b is not None)):
        args.show_specs = True

    if args.show_specs:
        print_specs()
    if args.model_params_b is not None:
        if (
            args.model_params_b <= 0
            or args.weight_bytes <= 0
            or args.reserve_gb < 0
            or args.kv_gb_per_1k_tokens < 0
            or args.tokens < 0
        ):
            raise SystemExit("model and memory arguments must be non-negative")
        print_budget(
            args.model_params_b,
            args.weight_bytes,
            args.reserve_gb,
            args.kv_gb_per_1k_tokens,
            args.tokens,
        )
    if args.probe:
        probe_device(args.measure_h2d, args.transfer_mb, args.repeats)


if __name__ == "__main__":
    main()
