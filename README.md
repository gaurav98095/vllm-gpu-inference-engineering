# vLLM & GPU Inference Engineering

## 64-Module Practical Masterclass

This repository contains my notes, experiments, code, and documentation for studying vLLM and GPU inference engineering.

The learning loop is:

> Learn a concept → implement it → measure GPU behavior → connect it to vLLM internals.

The course is designed around hands-on work with NVIDIA T4 GPUs on Lightning AI. We will gradually move from GPU fundamentals to CUDA kernels, vLLM internals, inference optimization, distributed inference, and production-style serving.

## Practical setup

- **1× NVIDIA T4:** GPU fundamentals, CUDA, kernels, vLLM internals, and benchmarking.
- **2× NVIDIA T4:** Tensor parallelism, NCCL, model sharding, and distributed overhead.
- **3× NVIDIA T4:** Replica serving, load balancing, and three-way tensor parallelism where supported.

Each T4 provides 16 GB of VRAM. Three GPUs provide 48 GB of physical VRAM in total; they do not form one shared memory pool.

The primary models will be in the 0.5B–3B range, with larger quantized models used occasionally. Some modern kernels, data types, and optimizations require newer GPUs, so those topics may be studied through traces, source code, or simulations when they cannot be run efficiently on a T4.

## Course phases

1. **GPU Fundamentals (Modules 01–08)** — CPU vs GPU, GPU architecture, memory, CUDA execution, and benchmarking.
2. **LLM Inference Fundamentals (Modules 09–16)** — Transformer computation, prefill, decode, KV cache, batching, and metrics.
3. **vLLM Internals (Modules 17–24)** — vLLM V1, PagedAttention, KV-cache management, scheduling, and tracing.
4. **CUDA, Kernels & GPU Optimization (Modules 25–32)** — CUDA, GEMM, Triton, attention kernels, CUDA Graphs, and profiling.
5. **Inference Performance Optimization (Modules 33–40)** — Precision, quantization, speculative decoding, long context, LoRA, and tuning.
6. **Multi-GPU & Distributed Inference (Modules 41–48)** — Interconnects, NCCL, tensor parallelism, pipeline parallelism, replicas, MoE, and disaggregation.
7. **Production LLM Serving (Modules 49–56)** — Deployment, workloads, load testing, observability, scaling, reliability, and cost.
8. **Core Inference Engineering (Modules 57–64)** — Source-code debugging, extensions, architecture-specific inference, multimodality, comparisons, and the capstone.

## Repository structure

Each topic will have one Markdown file. Code is kept separately under `code/`.

```text
docs/modules/01-cpu-vs-gpu.md       Module documentation
code/module-01/                     Module 01 code and experiments
```

The documentation is published as a GitHub Pages website with a layout inspired by [Tiny-LLM](https://skyzh.github.io/tiny-llm/). The site is built automatically from the docs/ directory on pushes to main.

## Module format

Each module should contain:

- the core concept explained clearly;
- implementation instructions;
- a measurable experiment;
- GPU behavior and results;
- a connection to vLLM internals;
- exercises or a completion checklist;
- references and follow-up questions.

Measurements should record the GPU, software versions, tensor shapes, dtypes, batch sizes, sequence lengths, repetitions, latency, throughput, and memory usage whenever relevant.

## Current progress

- [x] Module 01 — [CPU vs GPU: Why GPU Computing Exists](docs/modules/01-cpu-vs-gpu.md)
- [x] Module 02 — [GPU Architecture: SMs, CUDA Cores, Tensor Cores, Warps](docs/modules/02-gpu-architecture.md)
- [x] Module 03 — [GPU Memory Hierarchy: Registers, Shared Memory, L1/L2 Cache & VRAM](docs/modules/03-gpu-memory-hierarchy.md)
- [x] Module 04 — [A 1B-Parameter Model on an NVIDIA T4: Memory and Execution](docs/modules/04-inside-t4-1b-model.md)
- [x] Module 05 — [Matrix Multiplication on GPUs: GEMM, FLOPs, Tiling & Tensor Cores](docs/modules/05-matrix-multiplication.md)
- [x] Module 06 — [CUDA Execution Model: Kernels, Threads, Blocks, Grids & Scheduling](docs/modules/06-cuda-execution-model.md)
- [ ] Modules 07–64

## Start here

Begin with [Module 01](docs/modules/01-cpu-vs-gpu.md) and run its experiment:

```bash
python code/module-01/inspect_and_benchmark.py --sizes 128 512 1024 2048 --repeats 30 --include-transfer
```
