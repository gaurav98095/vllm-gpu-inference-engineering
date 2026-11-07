# vLLM & GPU Inference Engineering

## A practical 64-module masterclass

Learn a concept → implement it → measure GPU behavior → connect it to vLLM internals.

This course starts with GPU fundamentals and CUDA, then moves through Transformer inference, vLLM internals, kernel optimization, distributed inference, and production serving.

**Progress:** 6 of 64 modules complete · **9.4%**

**Primary environment:** Lightning AI · NVIDIA T4 · Linux · Python · PyTorch

## Start here

[Start Module 01 — CPU vs GPU: Why GPU Computing Exists](modules/01-cpu-vs-gpu.md)

Run the first experiment:

~~~bash
python code/module-01/inspect_and_benchmark.py \
  --sizes 128 512 1024 2048 \
  --repeats 30 \
  --include-transfer
~~~

## What you need

- Linux or a compatible CUDA environment;
- Python and a CUDA-enabled PyTorch installation;
- one NVIDIA T4 for Modules 01–06;
- optional access to two or three T4 GPUs for later distributed modules.

The lessons also explain which experiments can be studied through CPU execution, traces, source code, or simulations when a T4 cannot run a newer kernel or data type.

## Course roadmap

| Phase | Modules | Focus |
| --- | ---: | --- |
| GPU Fundamentals | 01–08 | Hardware, memory, execution, CUDA, benchmarking |
| LLM Inference Fundamentals | 09–16 | Transformers, prefill, decode, KV cache, batching, metrics |
| vLLM Internals | 17–24 | V1, PagedAttention, scheduling, tracing |
| CUDA and Kernel Optimization | 25–32 | CUDA, GEMM, Triton, attention kernels, graphs, profiling |
| Inference Optimization | 33–40 | Precision, quantization, speculation, long context, LoRA |
| Distributed Inference | 41–48 | NCCL, parallelism, replicas, MoE, disaggregation |
| Production Serving | 49–56 | Deployment, load testing, observability, scaling, reliability, cost |
| Core Inference Engineering | 57–64 | Debugging, extensions, multimodality, comparisons, capstone |

## How to use each module

1. Read the mental model and learning objectives.
2. Run the companion script or notebook.
3. Record the GPU, software versions, shapes, dtypes, repetitions, latency, throughput, and memory usage.
4. Explain the result in terms of GPU behavior and vLLM.

The source repository contains the Markdown lessons under docs/modules/ and matching experiments under code/.
