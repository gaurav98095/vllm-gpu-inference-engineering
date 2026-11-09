"""Trace Qwen2.5 inference from tokenization through cached generation on a T4."""

from __future__ import annotations

import argparse
import statistics
import time

import torch


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Lightning T4 runtime")
    return torch.device("cuda:0")


def load_model(device: torch.device, model_id: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the lab dependencies with: "
            "pip install -U 'transformers>=4.45,<5' accelerate safetensors"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device).eval()
    return tokenizer, model


def tokenize_prompt(tokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def benchmark_prefill(model, inputs, device: torch.device, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(3):
            model(**inputs, use_cache=True, return_dict=True)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(**inputs, use_cache=True, return_dict=True)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def generate_greedy(model, tokenizer, inputs, device, max_new_tokens: int):
    with torch.inference_mode():
        result = model(**inputs, use_cache=True, return_dict=True)
        cache = result.past_key_values
        next_token = result.logits[:, -1:].argmax(dim=-1)
        generated = [next_token]
        attention_mask = inputs["attention_mask"]
        decode_times_ms = []

        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((1, 1))],
                dim=-1,
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            torch.cuda.synchronize(device)
            decode_times_ms.append((time.perf_counter() - started) * 1000)
            cache = result.past_key_values
            next_token = result.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)

    generated_ids = torch.cat(generated, dim=1)
    text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return text, cache, decode_times_ms


def cache_bytes(cache) -> int:
    total = 0
    for layer in range(len(cache)):
        key_tensor, value_tensor = cache[layer]
        total += key_tensor.numel() * key_tensor.element_size()
        total += value_tensor.numel() * value_tensor.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--prompt",
        default="What is the capital of France? Answer briefly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prefill-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.prefill_repeats < 1:
        parser.error("numeric arguments must be positive")

    device = require_cuda()
    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    tokenizer, model = load_model(device, args.model)
    inputs = tokenize_prompt(tokenizer, args.prompt, device)
    prompt_tokens = inputs["input_ids"].shape[1]
    print("Prompt tokens:", prompt_tokens)
    print("Input IDs shape:", tuple(inputs["input_ids"].shape))
    print("Embedding shape:", tuple(model.get_input_embeddings()(inputs["input_ids"]).shape))

    prefill_ms = benchmark_prefill(model, inputs, device, args.prefill_repeats)
    print("Median prefill ms:", round(prefill_ms, 3))
    print("Prefill tokens/sec:", round(prompt_tokens / (prefill_ms / 1000), 1))

    response, cache, decode_times_ms = generate_greedy(
        model, tokenizer, inputs, device, args.max_new_tokens
    )
    print("Generated text:", response)
    print("Cache sequence length:", cache.get_seq_length())
    print("Logical KV-cache MiB:", round(cache_bytes(cache) / 2**20, 4))
    if decode_times_ms:
        print("Mean incremental decode ms:", round(statistics.mean(decode_times_ms), 3))


if __name__ == "__main__":
    main()
