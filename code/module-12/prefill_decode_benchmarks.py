"""Benchmark prefill and incremental decode on one CUDA GPU.

The measurements intentionally keep cache construction outside the timed
decode region. This makes the two phases comparable and avoids reporting a
prefill operation as decode latency.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Iterable

import torch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def require_cuda() -> torch.device:
    """Return the first CUDA device or fail with an actionable message."""
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA-enabled Lightning AI GPU runtime.")
    return torch.device("cuda:0")


def load_model(model_id: str, device: torch.device) -> Any:
    """Load the small FP16 model used throughout the inference modules."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    return model.to(device).eval()


def gpu_event_time_ms(fn: Any) -> tuple[float, Any]:
    """Time one CUDA operation with events instead of CPU dispatch time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


@torch.inference_mode()
def measure_prefill(
    model: Any,
    input_ids: torch.Tensor,
    repeats: int = 8,
    warmup: int = 2,
) -> float:
    """Return median GPU milliseconds for a full prompt forward pass."""
    for _ in range(warmup):
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        del output
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        elapsed, output = gpu_event_time_ms(
            lambda: model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
        )
        samples.append(elapsed)
        del output
    return statistics.median(samples)


@torch.inference_mode()
def measure_decode(
    model: Any,
    next_token: torch.Tensor,
    cache: Any,
    repeats: int = 8,
) -> float:
    """Return median GPU milliseconds for one incremental decode forward.

    DynamicCache implementations may be updated in place. A deepcopy keeps
    every repetition at the same context length and happens outside timing.
    """
    import copy

    samples: list[float] = []
    for _ in range(repeats):
        trial_cache = copy.deepcopy(cache)
        elapsed, output = gpu_event_time_ms(
            lambda: model(
                input_ids=next_token,
                past_key_values=trial_cache,
                use_cache=True,
                return_dict=True,
            )
        )
        samples.append(elapsed)
        del output, trial_cache
    return statistics.median(samples)


@torch.inference_mode()
def benchmark_prefill_lengths(
    model: Any,
    device: torch.device,
    lengths: Iterable[int],
    repeats: int = 8,
) -> list[dict[str, float]]:
    """Measure prompt forward time and prompt-token throughput."""
    rows: list[dict[str, float]] = []
    for seq_len in lengths:
        input_ids = torch.randint(100, 50000, (1, seq_len), device=device)
        milliseconds = measure_prefill(model, input_ids, repeats=repeats)
        rows.append(
            {
                "prompt_tokens": float(seq_len),
                "prefill_ms": milliseconds,
                "prompt_tokens_per_sec": seq_len / (milliseconds / 1000.0),
            }
        )
        del input_ids
    return rows


@torch.inference_mode()
def benchmark_decode_contexts(
    model: Any,
    device: torch.device,
    context_lengths: Iterable[int],
    repeats: int = 8,
) -> list[dict[str, float]]:
    """Build each cache outside timing, then measure one new-token step."""
    rows: list[dict[str, float]] = []
    for context_len in context_lengths:
        prompt = torch.randint(100, 50000, (1, context_len), device=device)
        # Cache construction is deliberately excluded from decode timing.
        prefill = model(input_ids=prompt, use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        before = int(cache.get_seq_length())
        milliseconds = measure_decode(
            model, next_token, cache, repeats=repeats
        )
        rows.append(
            {
                "context_tokens": float(context_len),
                "decode_ms": milliseconds,
                "cache_length_before": float(before),
                "cache_length_after": float(before + 1),
            }
        )
        del prompt, prefill, cache, next_token
        torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def benchmark_generation_lengths(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    output_lengths: Iterable[int],
    prompt_text: str = "Explain why prefill and decode have different GPU behavior.",
    repeats: int = 3,
) -> list[dict[str, float]]:
    """Measure end-to-end local generation time for requested output lengths."""
    messages = [{"role": "user", "content": prompt_text}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    rows: list[dict[str, float]] = []
    for requested in output_lengths:
        samples: list[float] = []
        actual = 0
        for trial in range(repeats + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(
                encoded,
                min_new_tokens=requested,
                max_new_tokens=requested,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            actual = int(output.shape[1] - encoded.shape[1])
            if trial > 0:
                samples.append(elapsed_ms)
        median_ms = statistics.median(samples)
        rows.append(
            {
                "requested_tokens": float(requested),
                "generated_tokens": float(actual),
                "generation_ms": median_ms,
                "output_tokens_per_sec": actual / (median_ms / 1000.0),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--prefill-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--decode-contexts",
        type=int,
        nargs="+",
        default=[64, 256, 512, 1024],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = require_cuda()
    model = load_model(args.model, device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {args.model}")
    print(f"Dtype: {next(model.parameters()).dtype}")

    prefill_rows = benchmark_prefill_lengths(
        model, device, args.prefill_lengths, repeats=args.repeats
    )
    print("\nPrefill results:")
    for row in prefill_rows:
        print(row)

    decode_rows = benchmark_decode_contexts(
        model, device, args.decode_contexts, repeats=args.repeats
    )
    print("\nDecode results:")
    for row in decode_rows:
        print(row)


if __name__ == "__main__":
    main()
