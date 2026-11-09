# Module 10 — LLM Inference Lifecycle: From Tokenization to Autoregressive Generation

_vLLM & GPU Inference Engineering Masterclass · Module 10 of 64 · Phase 2: LLM Inference Fundamentals_

**Hands-on:** Lightning AI · NVIDIA T4 · 1 GPU

In the first nine modules, we studied GPUs as computing machines. We learned about CUDA cores, Tensor Cores, matrix multiplication, memory bandwidth, the Roofline Model, hardware selection, and measurement.

Now we move to the central question of LLM serving:

> What exactly happens, step by step, when a prompt becomes generated text?

For example, when you ask:

> What is the capital of France?

the model eventually produces an answer such as:

> Paris is the capital of France.

That result passes through several distinct stages:

1. Text is converted into token IDs.
2. Token IDs select embedding vectors.
3. Transformer layers run attention, MLPs, normalization, and residual operations.
4. The model produces logits for the vocabulary.
5. A token-selection rule chooses the next token.
6. The selected token is fed back into the model for the next decoding step.

This module follows that lifecycle as an inference engineer: by tracking tensors, devices, memory, and latency rather than treating the model as a black box.

## 1. The complete inference lifecycle

This is a simplified path:

```text
text → tokenizer → token IDs → embeddings → Transformer layers
     → hidden states → vocabulary logits → next token → repeat
```

A production vLLM server also handles request queues, scheduling, continuous batching, KV-cache allocation, streaming, sampling, and error handling. We will expose the core model path first; later modules will study those serving systems in detail.

## 2. The model for today's lesson

We'll use [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), a small decoder-only causal language model that fits comfortably on a single 16 GB T4 while exposing the same major inference stages as larger models.

The published model card describes approximately 0.49B parameters, 24 layers, 14 query-attention heads, 2 key/value heads, RoPE, and a 32,768-token context length. The exact checkpoint and runtime configuration should always be inspected rather than assumed.

| Property | Value |
| --- | ---: |
| Transformer layers | 24 |
| Hidden dimension | 896 |
| Query attention heads | 14 |
| Key/value heads | 2 |
| Head dimension | 64 |
| Vocabulary size | 151,936 |
| Maximum configured positions | 32,768 |
| Model type | Causal language model |

The model weights are stored as BF16 on the model hub. We explicitly load FP16 for this T4 lab because Turing Tensor Cores do not provide the native BF16 path available on newer architectures. FP16 parameter storage is roughly 1 GB, with additional runtime and cache memory.

## 3. Phase A — Tokenization

The model does not consume words directly. A tokenizer converts text into integer IDs. For illustration, a tokenizer might represent:

```text
The capital of France is
```

as a sequence such as:

```text
The       → 101
 capital  → 245
 of       → 78
 France   → 920
 is       → 34
```

Those IDs are fictional. The actual Qwen tokenizer may split the text differently and assigns different IDs.

For a batch of one and 20 tokens:

$$
\text{input\_ids}\in\mathbb{Z}^{1\times20}
$$

For eight padded prompts with 20 positions:

$$
\text{input\_ids}\in\mathbb{Z}^{8\times20}
$$

These are integer IDs, not FP16 embedding vectors. Tokenization normally occurs on the CPU before the model's GPU forward pass, so tokenization time can contribute to latency for large prompts and high request rates.

## 4. Phase B — Embedding lookup

Qwen2.5-0.5B uses a hidden dimension of 896. Each token ID selects one row from the embedding matrix:

$$
E\in\mathbb{R}^{151936\times896}
$$

For 20 tokens in one request:

$$
\text{input\_ids}\in\mathbb{Z}^{1\times20}
$$
$$
\boxed{X\in\mathbb{R}^{1\times20\times896}}
$$

For each token ID $t$, the model retrieves row $E_t$. Positional information is incorporated with RoPE inside attention rather than by simply adding a conventional learned position vector at the input.

## 5. Phase C — Prefill

The initial forward pass over the prompt is called **prefill**. If the prompt contains 20 tokens, the model processes all 20 positions to produce the first answer-token distribution.

The hidden-state shape remains:

$$
[1,20,896]
$$

through the main Transformer stack. Each of the 24 layers performs attention, feed-forward computation, normalization, and residual operations.

One illustrative projection is:

$$
Y=XW,
\qquad X\in\mathbb{R}^{20\times896},
\qquad W\in\mathbb{R}^{896\times896}
$$

Its approximate arithmetic work is:

$$
2\times20\times896^2\approx32.1\text{ million FLOPs}
$$

That is only one projection in one part of the model. The complete forward pass launches many matrix multiplications, attention kernels, normalizations, and memory operations.

Because all prompt positions are processed together, prefill can form relatively large matrix operations and use the GPU efficiently. Causal attention still prevents a position from attending to future positions.

## 6. Phase D — Logits and token selection

The language-model head maps hidden states to vocabulary scores. With vocabulary size $V=151936$ and 20 prompt positions:

$$
\boxed{\text{logits}\in\mathbb{R}^{1\times20\times151936}}
$$

To generate the next token, we usually need only the final position:

$$
\text{next\_logits}=\text{logits}[:, -1, :]
$$

$$
\boxed{\text{next\_logits}\in\mathbb{R}^{1\times151936}}
$$

Logits are raw, unnormalized scores. Softmax converts them into probabilities:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Greedy decoding chooses:

$$
t_{\text{next}}=\arg\max_i z_i
$$

Sampling can apply temperature, top-k, and top-p filtering before choosing a token. This module uses greedy decoding so that measurements are deterministic.

Token selection can remain on the GPU. The entire vocabulary-logit tensor does not need to be transferred to the CPU just to find its maximum.

## 7. Phase E — Autoregressive decoding and the KV cache

Generation is autoregressive:

$$
P(y_1,y_2,\ldots,y_T\mid x)=\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$

After the first token is selected, it is fed back as the next input. Decoder-only Transformers use a key/value cache so that previous attention keys and values do not need to be recomputed at every step.

During prefill, the model computes and stores K and V tensors for the prompt positions. During incremental decode, it processes the new token, appends its K/V state, and reuses the cached history.

For Qwen2.5-0.5B, one layer's cache for a batch of one and 20 cached positions has shapes:

$$
K,V\in\mathbb{R}^{1\times2\times20\times64}
$$

The cache contains 24 layer-level key/value pairs. With FP16 elements, the theoretical KV-cache bytes per token are:

$$
M_{\text{KV/token}}
=2\times L\times H_{\text{KV}}\times D\times2
$$

$$
=2\times24\times2\times64\times2
=\boxed{12{,}288\text{ bytes}=12\text{ KiB}}
$$

This is logical tensor storage only. Allocator reservations, temporary attention memory, padding, and other runtime state add overhead.

## 8. Lightning AI laboratory

This lab uses PyTorch and Hugging Face Transformers rather than vLLM so that each stage is visible. Install compatible dependencies in the Lightning AI terminal if needed:

```bash
pip install -U "transformers>=4.45,<5" accelerate safetensors
```

### Lab 1 — Inspect the T4 and load the model

```python
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "Select one T4 GPU."
device = torch.device("cuda:0")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA capability:", torch.cuda.get_device_capability(0))
print("PyTorch:", torch.__version__)

tokenizer = AutoTokenizer.from_pretrained(model_id)
torch.cuda.reset_peak_memory_stats(device)

started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device).eval()
torch.cuda.synchronize(device)

print("Model load seconds:", round(time.perf_counter() - started, 2))
print("Model device:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Allocated MiB:", round(torch.cuda.memory_allocated(device) / 2**20, 1))
```

Model download, deserialization, allocation, and device transfer contribute to load time. That is different from per-request latency; production servers normally load the model once and reuse it.

### Lab 2 — Inspect real tokenization

```python
messages = [
    {
        "role": "user",
        "content": "What is the capital of France? Answer briefly.",
    }
]

started = time.perf_counter()
inputs_cpu = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
tokenization_ms = (time.perf_counter() - started) * 1000

print("Tokenization time (ms):", round(tokenization_ms, 3))
print("Input shape:", tuple(inputs_cpu["input_ids"].shape))
for token_id in inputs_cpu["input_ids"][0].tolist():
    print(token_id, repr(tokenizer.decode([token_id])))
```

The chat template adds role markers and special tokens expected by the instruction-tuned checkpoint. Inspecting the actual IDs is more useful than relying on a word-level illustration.

### Lab 3 — Move inputs and inspect embeddings

```python
started = time.perf_counter()
inputs = {key: value.to(device) for key, value in inputs_cpu.items()}
torch.cuda.synchronize(device)
print("Input transfer ms:", round((time.perf_counter() - started) * 1000, 3))

with torch.inference_mode():
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(inputs["input_ids"])

print("Input IDs:", tuple(inputs["input_ids"].shape))
print("Embedding matrix:", tuple(embedding_layer.weight.shape))
print("Output embeddings:", tuple(embeddings.shape))
print("Embedding dtype/device:", embeddings.dtype, embeddings.device)
```

For $S$ input tokens, the embedding output should have shape $[1,S,896]$.

### Lab 4 — Run prefill and inspect logits/cache

```python
with torch.inference_mode():
    outputs = model(**inputs, use_cache=True, return_dict=True)

print("Logits shape:", tuple(outputs.logits.shape))
print("Logits dtype:", outputs.logits.dtype)
print("Cache type:", type(outputs.past_key_values).__name__)
print("Cache length:", outputs.past_key_values.get_seq_length())

next_logits = outputs.logits[:, -1, :]
next_token = next_logits.argmax(dim=-1, keepdim=True)
print("Next-token logits:", tuple(next_logits.shape))
print("Selected token:", repr(tokenizer.decode(next_token[0].tolist())))
```

The full logits shape is $[1,S,151936]$ and the final-position logits shape is $[1,151936]$. The first generated token is selected from logits already computed during prefill.

### Lab 5 — Measure model-only prefill

```python
def benchmark_prefill(model, inputs, warmup=3, repeats=10):
    with torch.inference_mode():
        for _ in range(warmup):
            model(**inputs, use_cache=True)
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

prefill_ms = benchmark_prefill(model, inputs)
prompt_tokens = inputs["input_ids"].shape[1]
print("Prompt tokens:", prompt_tokens)
print("Median prefill ms:", round(prefill_ms, 3))
print("Prompt tokens/sec:", round(prompt_tokens / (prefill_ms / 1000), 1))
```

This is model-only prefill time. API-level TTFT also includes tokenization, transfers, scheduling, and token selection.

### Lab 6 — Manual incremental decoding

```python
def generate_greedy(model, inputs, tokenizer, max_new_tokens=16):
    with torch.inference_mode():
        result = model(**inputs, use_cache=True, return_dict=True)
        cache = result.past_key_values
        next_token = result.logits[:, -1:].argmax(dim=-1)
        generated = [next_token]
        attention_mask = inputs["attention_mask"]

        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break

            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((1, 1))],
                dim=-1,
            )
            result = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = result.past_key_values
            next_token = result.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)

    generated_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True), cache

response, cache = generate_greedy(model, inputs, tokenizer)
print(response)
print("Cached positions:", cache.get_seq_length())
```

Each incremental forward pass processes one new token and reuses the cached prompt state. A production engine minimizes Python overhead and synchronization, but the causal dataflow is the same.

### Lab 7 — Calculate actual KV-cache bytes

```python
kv_bytes = 0
for layer in range(model.config.num_hidden_layers):
    key_tensor, value_tensor = cache[layer]
    kv_bytes += (
        key_tensor.numel() * key_tensor.element_size()
        + value_tensor.numel() * value_tensor.element_size()
    )

print("Logical KV storage MiB:", round(kv_bytes / 2**20, 4))
print("Cached positions:", cache.get_seq_length())
print("Bytes per cached position:", kv_bytes / cache.get_seq_length())
```

The measured bytes per cached position should be close to the theoretical 12,288-byte value for this model in FP16, excluding allocator and temporary-workspace effects.

## 9. Cache versus no-cache decoding

Without a KV cache, every decoding step recomputes the entire accumulated sequence. With a cache, only the newly generated token is processed and previous keys and values are reused.

For short prompts and a small number of generated tokens, the speedup may be modest because Python, dispatch, and synchronization overheads dominate. As context and output length grow, recomputing the full history becomes increasingly expensive.

When comparing the paths, record:

- generated token IDs and decoded text;
- total generation time;
- average incremental forward time;
- cache sequence length;
- allocated and peak GPU memory.

The comparison is an educational experiment, not a benchmark of vLLM's paged KV-cache implementation.

## 10. End-to-end request timing

An end-to-end local request includes more than the model forward pass:

```text
request → tokenization → CPU/GPU transfer → prefill → decode loop
        → token selection → detokenization → response
```

Track these metrics separately:

| Metric | Meaning |
| --- | --- |
| Model load time | Download, deserialize, allocate, and transfer weights |
| Tokenization time | CPU conversion from text to token IDs |
| Input transfer time | CPU-to-GPU movement of token tensors |
| Prefill latency | Time to process the prompt and produce first-token logits |
| TTFT | Request arrival to first streamed token |
| ITL | Time between successive generated tokens |
| Peak allocated memory | Maximum PyTorch allocation during the request |
| Total latency | End-to-end local request duration |

TTFT includes preprocessing, scheduling, prefill, and first-token selection. ITL mostly describes the incremental decode path after the first token. vLLM adds queueing, batching, and scheduler effects around these model operations.

## 11. Tensor-shape summary

Assume batch size 1 and prompt length $S$:

| Stage | Tensor shape | Device |
| --- | --- | --- |
| Raw user text | String | CPU |
| Tokenization | `[1, S]` integer IDs | CPU |
| Token transfer | `[1, S]` integer IDs | GPU |
| Token embeddings | `[1, S, 896]` | GPU |
| Transformer hidden states | `[1, S, 896]` | GPU |
| Full logits | `[1, S, 151936]` | GPU |
| Final-position logits | `[1, 151936]` | GPU |
| Selected next-token ID | `[1, 1]` | GPU |
| Per-layer K cache | `[1, 2, S_cached, 64]` | GPU |
| Per-layer V cache | `[1, 2, S_cached, 64]` | GPU |
| Decoded output text | String | CPU |

This table is the bridge between a user-visible response and the GPU tensors that produced it.

## References

- Qwen Team, [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — Architecture, chat template, tokenizer, and generation examples.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer architecture.
- Hugging Face, [KV-cache strategies](https://huggingface.co/docs/transformers/main/kv_cache) — Dynamic caches and generation behavior.
- PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) — Asynchronous execution, timing, and memory accounting.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM KV-cache management.

## Module 10 recap

$$
\boxed{\text{Text}\rightarrow\text{Tokens}\rightarrow\text{Embeddings}\rightarrow\text{Transformer}\rightarrow\text{Logits}\rightarrow\text{Next token}}
$$

The three points to retain are:

1. Tokenization, model loading, prefill, and decode are different operations with different performance characteristics.
2. Prefill processes many token positions together, while incremental decode usually processes one new position per active sequence.
3. The KV cache exchanges GPU memory capacity for reduced recomputation.

**Module 10 complete.**

Next: Module 11 — Transformer Forward Pass as GPU Workloads: QKV Projections, Attention, MLPs, Normalization & Matrix Shapes.
