"""Inspect Qwen2.5 KV-cache shapes, storage, growth, and decode reuse.

Run on one CUDA GPU:
    python code/module-13/kv_cache_lab.py --contexts 64 128 256 512
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Iterable

import torch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA-enabled Lightning AI GPU runtime.")
    return torch.device("cuda:0")


def load_model(model_id: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device).eval()
    return tokenizer, model


def cache_layers(cache: Any) -> Iterable[Any]:
    """Support current DynamicCache and older tuple-like cache APIs."""
    if hasattr(cache, "layers"):
        return cache.layers
    return cache


def layer_key_value(cache: Any, index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[index]
        return layer.keys, layer.values
    return cache[index]


def kv_storage_bytes(cache: Any) -> int:
    """Count live logical K/V tensor bytes, excluding allocator overhead."""
    total = 0
    for layer in cache_layers(cache):
        if hasattr(layer, "keys"):
            tensors = (layer.keys, layer.values)
        else:
            tensors = layer
        for tensor in tensors:
            total += tensor.numel() * tensor.element_size()
    return total


def make_prompt(tokenizer: Any, device: torch.device) -> Any:
    messages = [{
        "role": "user",
        "content": "Explain why the KV cache helps autoregressive decoding.",
    }]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


@torch.inference_mode()
def create_cache(model: Any, input_ids: torch.Tensor) -> tuple[Any, Any]:
    output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return output, output.past_key_values


@torch.inference_mode()
def inspect_cache(model: Any, encoded: Any) -> dict[str, Any]:
    output, cache = create_cache(model, encoded["input_ids"])
    key, value = layer_key_value(cache)
    return {
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "cache_type": type(cache).__name__,
        "cached_positions": int(cache.get_seq_length()),
        "layers": len(list(cache_layers(cache))),
        "key_shape": tuple(key.shape),
        "value_shape": tuple(value.shape),
        "dtype": str(key.dtype),
        "logical_bytes": kv_storage_bytes(cache),
        "output": output,
        "cache": cache,
    }


@torch.inference_mode()
def measure_cache_growth(model: Any, device: torch.device, lengths: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for seq_len in lengths:
        input_ids = torch.randint(100, 50000, (1, seq_len), device=device)
        output, cache = create_cache(model, input_ids)
        measured = kv_storage_bytes(cache)
        key, _ = layer_key_value(cache)
        bytes_per_token = measured / seq_len
        rows.append({
            "tokens": seq_len,
            "measured_MiB": measured / 1024**2,
            "bytes_per_token": bytes_per_token,
            "key_shape": tuple(key.shape),
        })
        del output, cache, input_ids
    return rows


@torch.inference_mode()
def decode_once(model: Any, encoded: Any, cache: Any, next_token: torch.Tensor) -> tuple[Any, float]:
    old_len = cache.get_seq_length()
    attention_mask = torch.cat(
        [encoded["attention_mask"], encoded["attention_mask"].new_ones((1, 1))],
        dim=-1,
    )
    cache_position = torch.tensor([old_len], dtype=torch.long, device=next_token.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(
        input_ids=next_token,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000


@torch.inference_mode()
def compare_cached_and_uncached(
    model: Any,
    input_ids: torch.Tensor,
    steps: int = 8,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    rows = []
    for use_cache in (False, True):
        samples = []
        final_length = 0
        for trial in range(repeats + 1):
            sequence = input_ids.clone()
            current_input = input_ids
            cache = None
            attention_mask = torch.ones_like(sequence)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(steps):
                output = model(
                    input_ids=current_input if use_cache else sequence,
                    attention_mask=attention_mask,
                    past_key_values=cache if use_cache else None,
                    use_cache=use_cache,
                    return_dict=True,
                )
                token = output.logits[:, -1:].argmax(dim=-1)
                sequence = torch.cat([sequence, token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, 1))], dim=-1
                )
                if use_cache:
                    cache = output.past_key_values
                    current_input = token
            torch.cuda.synchronize()
            if trial > 0:
                samples.append((time.perf_counter() - started) * 1000)
            final_length = int(sequence.shape[1])
        rows.append({
            "use_cache": use_cache,
            "steps": steps,
            "final_sequence_length": final_length,
            "median_ms": statistics.median(samples),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--contexts", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()
    tokenizer, model = load_model(args.model, device)
    encoded = make_prompt(tokenizer, device)
    cfg = model.config
    print("GPU:", torch.cuda.get_device_name(0))
    print("Layers:", cfg.num_hidden_layers)
    print("Query heads:", cfg.num_attention_heads)
    print("KV heads:", cfg.num_key_value_heads)
    print("Head dimension:", cfg.hidden_size // cfg.num_attention_heads)

    inspected = inspect_cache(model, encoded)
    print("\nPrompt tokens:", inspected["prompt_tokens"])
    print("Cache type:", inspected["cache_type"])
    print("Layer-0 K shape:", inspected["key_shape"])
    print("Layer-0 V shape:", inspected["value_shape"])
    print("Logical KV MiB:", inspected["logical_bytes"] / 1024**2)

    print("\nKV growth:")
    for row in measure_cache_growth(model, device, args.contexts):
        print(row)

    first_token = inspected["output"].logits[:, -1:].argmax(dim=-1)
    old_bytes = kv_storage_bytes(inspected["cache"])
    decode_output, decode_ms = decode_once(model, encoded, inspected["cache"], first_token)
    new_bytes = kv_storage_bytes(decode_output.past_key_values)
    print("\nOne-token decode:")
    print("Decode milliseconds:", round(decode_ms, 3))
    print("Cache length after decode:", decode_output.past_key_values.get_seq_length())
    print("Logical bytes added:", new_bytes - old_bytes)

    print("\nCached versus uncached generation:")
    for row in compare_cached_and_uncached(
        model, encoded["input_ids"], steps=args.steps, repeats=args.repeats
    ):
        print(row)


if __name__ == "__main__":
    main()
