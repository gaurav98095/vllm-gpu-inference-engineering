# vLLM & GPU Inference Engineering

## 64-Module Practical Masterclass

This course studies vLLM and GPU inference engineering through a practical loop:

> Learn a concept → implement it → measure GPU behavior → connect it to vLLM internals.

The primary environment is Lightning AI with NVIDIA T4 GPUs. The course gradually moves from GPU fundamentals and CUDA to vLLM internals, inference optimization, distributed inference, and production-style serving.

## Practical setup

- **1× NVIDIA T4:** GPU fundamentals, CUDA, kernels, vLLM internals, and benchmarking.
- **2× NVIDIA T4:** Tensor parallelism, NCCL, model sharding, and distributed overhead.
- **3× NVIDIA T4:** Replica serving, load balancing, and supported three-way experiments.

Each T4 provides 16 GB of VRAM. Three GPUs provide 48 GB of physical VRAM in total; they do not form one shared memory pool.

The primary models will be in the 0.5B–3B range. Some modern kernels, data types, and optimizations require newer GPUs, so those topics may be studied through traces, source code, or simulations when they cannot be run efficiently on a T4.

## Course phases

1. **GPU Fundamentals (01–08)** — Hardware, memory, execution, and benchmarking.
2. **LLM Inference Fundamentals (09–16)** — Transformer computation, prefill, decode, KV cache, batching, and metrics.
3. **vLLM Internals (17–24)** — vLLM V1, PagedAttention, KV-cache management, scheduling, and tracing.
4. **CUDA, Kernels & GPU Optimization (25–32)** — CUDA, GEMM, Triton, attention kernels, CUDA Graphs, and profiling.
5. **Inference Performance Optimization (33–40)** — Precision, quantization, speculative decoding, long context, LoRA, and tuning.
6. **Multi-GPU & Distributed Inference (41–48)** — Interconnects, NCCL, parallelism, replicas, MoE, and disaggregation.
7. **Production LLM Serving (49–56)** — Deployment, workloads, load testing, observability, scaling, reliability, and cost.
8. **Core Inference Engineering (57–64)** — Source debugging, extensions, architecture-specific inference, multimodality, comparisons, and the capstone.

## Module format

Each topic has one Markdown file and a separate code directory. Modules contain explanations, implementation work, measurements, GPU observations, vLLM connections, exercises, and references.

## Start here

[Module 01 — CPU vs GPU: Why GPU Computing Exists](modules/01-cpu-vs-gpu.md)
