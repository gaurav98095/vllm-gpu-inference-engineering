"""Trace a causal language model from prompt text to generated tokens.

This is an educational Module 09 lab. It uses Hugging Face Transformers to
load a small causal LM, inspect tensor shapes, measure a prompt forward pass
(prefill), and run a simple cached greedy decode loop. The decode loop is
intentionally explicit; production engines use more specialized scheduling,
memory management, sampling, and kernels.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


DEFAULT_MODEL = "distilgpt2"
DEFAULT_PROMPT = "GPU inference is useful because"


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cuda/cpu and fail clearly when CUDA was requested."""
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning GPU runtime")
    return torch.device(requested)


def choose_dtype(name: str, device: torch.device) -> torch.dtype | None:
    """Return a model dtype, leaving CPU auto-loading at the default dtype."""
    if name == "auto":
        return torch.float16 if device.type == "cuda" else None
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype | None,
):
    """Download/load a causal LM and move it to the selected device."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Module 09 requires transformers; install it with "
            "pip install transformers safetensors"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Return device memory values in MiB, or an empty dict on CPU."""
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        "free_mib": free / 2**20,
        "total_mib": total / 2**20,
        "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def parameter_count(model: torch.nn.Module) -> int:
    """Count parameters without constructing a duplicate tensor."""
    return sum(parameter.numel() for parameter in model.parameters())


def config_value(config: Any, *names: str) -> Any:
    """Read the first available architecture-specific config attribute."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return "unknown"


def print_model_summary(model: torch.nn.Module, tokenizer: Any, device: torch.device) -> None:
    """Print the architecture facts needed to interpret the lab."""
    config = model.config
    print("=== Model and runtime ===")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Parameter count: {parameter_count(model):,}")
    print(f"Vocabulary size: {config_value(config, 'vocab_size')}")
    print(f"Hidden size: {config_value(config, 'hidden_size', 'n_embd')}")
    print(f"Layers: {config_value(config, 'num_hidden_layers', 'n_layer')}")
    print(f"Attention heads: {config_value(config, 'num_attention_heads', 'n_head')}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}")
        print(f"Compute capability: {properties.major}.{properties.minor}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_input_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize on CPU, optionally truncate, then move model inputs to device."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=max_input_tokens is not None,
        max_length=max_input_tokens,
    )
    return {name: value.to(device) for name, value in inputs.items()}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    """Return median milliseconds for a callable on CPU or CUDA."""
    for _ in range(warmup):
        function()
    synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            function()
            samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def run_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    capture_hidden_states: bool = False,
):
    """Run one causal-LM forward pass with cache enabled."""
    with torch.inference_mode():
        return model(
            **inputs,
            use_cache=True,
            output_hidden_states=capture_hidden_states,
            return_dict=True,
        )


def cache_description(cache: Any) -> str:
    """Describe old tuple caches and newer Transformers Cache objects."""
    if cache is None:
        return "not returned"
    if hasattr(cache, "get_seq_length"):
        return f"{type(cache).__name__}, sequence length={cache.get_seq_length()}"
    if isinstance(cache, (tuple, list)):
        if not cache:
            return "empty tuple cache"
        first_layer = cache[0]
        shapes = [tuple(tensor.shape) for tensor in first_layer]
        return f"tuple cache, layers={len(cache)}, first-layer shapes={shapes}"
    return type(cache).__name__


def print_forward_shapes(outputs: Any) -> None:
    """Print the tensors that connect a Transformer forward pass to logits."""
    print("\n=== Forward-pass tensors ===")
    print(f"Logits: {tuple(outputs.logits.shape)}")
    print(f"Logits dtype: {outputs.logits.dtype}")
    print(f"Past KV cache: {cache_description(outputs.past_key_values)}")
    if outputs.hidden_states is not None:
        print(f"Hidden-state tensors returned: {len(outputs.hidden_states)}")
        print(f"Embedding output shape: {tuple(outputs.hidden_states[0].shape)}")
        print(f"Final hidden-state shape: {tuple(outputs.hidden_states[-1].shape)}")


@torch.inference_mode()
def greedy_decode(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate tokens one at a time while reusing the returned KV cache.

    This intentionally simple loop is for observing the lifecycle. It uses
    torch.cat for readability; a production engine avoids repeated dynamic
    allocation and schedules many sequences together.
    """
    generated = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(generated)
    past_key_values = None

    for _ in range(max_new_tokens):
        model_inputs: dict[str, Any] = {
            "input_ids": generated if past_key_values is None else generated[:, -1:],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        outputs = model(**model_inputs)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token)),
            dim=1,
        )
        past_key_values = outputs.past_key_values
    return generated


def print_next_token_candidates(
    tokenizer: Any,
    logits: torch.Tensor,
    count: int = 5,
) -> None:
    """Show the highest-scoring next-token candidates before sampling."""
    probabilities = torch.softmax(logits[0, -1, :].float(), dim=-1)
    values, indices = torch.topk(probabilities, k=count)
    print("\n=== Next-token candidates ===")
    for probability, index in zip(values.tolist(), indices.tolist()):
        token_text = tokenizer.decode([index]).replace("\n", "\\n")
        print(f"{index:>8d}  {probability:8.4f}  {token_text!r}")


def inspect_lifecycle(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete lab and return JSON-serializable measurements."""
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)
    print_model_summary(model, tokenizer, device)
    before_inputs = memory_snapshot(device)

    inputs = tokenize_prompt(
        tokenizer,
        args.prompt,
        device,
        args.max_input_tokens,
    )
    input_ids = inputs["input_ids"]
    print("\n=== Tokenization ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Input IDs shape: {tuple(input_ids.shape)}")
    print(f"Input token count: {input_ids.shape[1]}")
    print(f"First token IDs: {input_ids[0, : min(12, input_ids.shape[1])].tolist()}")
    print(f"Round-trip text: {tokenizer.decode(input_ids[0])!r}")

    outputs = run_forward(model, inputs, args.capture_hidden_states)
    synchronize(device)
    print_forward_shapes(outputs)
    print_next_token_candidates(tokenizer, outputs.logits)
    after_forward = memory_snapshot(device)

    forward_ms = benchmark_callable(
        lambda: run_forward(model, inputs),
        device,
        args.warmup,
        args.repeats,
    )
    print(f"\nPrefill forward median: {forward_ms:.3f} ms")

    generated = greedy_decode(model, inputs, args.max_new_tokens)
    synchronize(device)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n=== Greedy generation ===")
    print(f"Generated token count: {generated.shape[1] - input_ids.shape[1]}")
    print(f"Text: {generated_text!r}")

    generation_ms = benchmark_callable(
        lambda: greedy_decode(model, inputs, args.max_new_tokens),
        device,
        max(1, min(args.warmup, 2)),
        args.generation_repeats,
    )
    per_token_ms = generation_ms / args.max_new_tokens
    print(f"Generation median: {generation_ms:.3f} ms")
    print(f"Approximate inter-token time: {per_token_ms:.3f} ms/token")
    print(f"Approximate generated tokens/sec: {1000 / per_token_ms:.2f}")

    result = {
        "model": args.model,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": args.max_new_tokens,
        "prefill_median_ms": forward_ms,
        "generation_median_ms": generation_ms,
        "inter_token_ms": per_token_ms,
        "generated_tokens_per_second": 1000 / per_token_ms,
        "memory_before_inputs": before_inputs,
        "memory_after_forward": after_forward,
        "memory_after_lab": memory_snapshot(device),
    }
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved measurements: {output_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-hidden-states", action="store_true")
    parser.add_argument("--json-output", metavar="PATH")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.max_new_tokens < 1
        or args.warmup < 0
        or args.repeats < 1
        or args.generation_repeats < 1
        or (args.max_input_tokens is not None and args.max_input_tokens < 1)
    ):
        parser.error("token, warmup, and repeat values must be positive")
    try:
        inspect_lifecycle(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Module 09 could not run: {exc}\n")


if __name__ == "__main__":
    main()
