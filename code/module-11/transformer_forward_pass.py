"""Inspect and benchmark Qwen Transformer-layer workloads on one T4."""

from __future__ import annotations

import argparse
import statistics

import torch


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")
    return torch.device("cuda:0")


def load_model(model_id: str, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies with: pip install -U "
            "'transformers>=4.45,<5' accelerate safetensors pandas"
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device).eval()
    return model


def projection_modules(model):
    layer = model.model.layers[0]
    return {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }


def linear_stats(module, tokens: int) -> dict[str, float]:
    out_features, in_features = module.weight.shape
    flops = 2 * tokens * in_features * out_features
    weight_bytes = module.weight.numel() * module.weight.element_size()
    return {
        "flops": float(flops),
        "weight_bytes": float(weight_bytes),
        "gflops": flops / 1e9,
        "weight_mib": weight_bytes / 2**20,
        "flops_per_weight_byte": flops / weight_bytes,
    }


def benchmark_module(module, x, device, warmup: int, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def inspect_hooks(model, device, tokens: int) -> list[dict[str, object]]:
    modules = projection_modules(model)
    captured = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            captured.append({
                "module": name,
                "input_shape": tuple(inputs[0].shape),
                "output_shape": tuple(output.shape),
                "dtype": str(output.dtype),
            })
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    input_ids = torch.randint(100, 50_000, (1, tokens), device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("tokens and repeats must be positive; warmup cannot be negative")

    device = require_cuda()
    model = load_model(args.model, device)
    config = model.config
    print("GPU:", torch.cuda.get_device_name(device))
    print("Hidden size:", config.hidden_size)
    print("Layers:", config.num_hidden_layers)
    print("Query heads:", config.num_attention_heads)
    print("KV heads:", config.num_key_value_heads)
    print("MLP intermediate:", config.intermediate_size)

    modules = projection_modules(model)
    print("\nProjection statistics for one layer")
    for name, module in modules.items():
        stats = linear_stats(module, args.tokens)
        print(
            f"{name:10s} shape={tuple(module.weight.shape)!s:18s} "
            f"GFLOPs={stats['gflops']:7.3f} "
            f"weight={stats['weight_mib']:7.2f} MiB "
            f"FLOPs/weight-byte={stats['flops_per_weight_byte']:7.2f}"
        )

    print("\nCaptured first-layer shapes")
    for row in inspect_hooks(model, device, args.tokens):
        print(row)

    print("\nProjection timings")
    for tokens in (1, args.tokens):
        for name, module in modules.items():
            input_features = module.weight.shape[1]
            x = torch.randn(
                1, tokens, input_features, device=device, dtype=torch.float16
            )
            milliseconds = benchmark_module(
                module, x, device, args.warmup, args.repeats
            )
            flops = 2 * tokens * input_features * module.weight.shape[0]
            tflops = flops / (milliseconds / 1000) / 1e12
            print(f"tokens={tokens:4d} {name:10s} {milliseconds:8.3f} ms {tflops:7.3f} TFLOPS")

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        input_ids = torch.randint(100, 50_000, (1, args.tokens), device=device)
        with torch.inference_mode():
            for _ in range(3):
                model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False, return_dict=True)
            torch.cuda.synchronize(device)
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
