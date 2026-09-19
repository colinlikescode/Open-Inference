# OpenBaseten

## What does it do?

Give OpenBaseten a language model, your existing NVIDIA GPU machines, and a time budget for inference optimization.

The **Pi** harness acts as your inference engineer, iteratively experimenting with **vLLM and SGLang**, distributed GPU layouts, serving settings, and **Triton/CUDA kernels** to improve performance for your workload.

OpenBaseten controls the machines over **SSH** and configures **PyTorch Distributed, NCCL, Ray, or the engine's native distributed runtime** as needed. The machines only need to exist, be networked, and be reachable over SSH.

A deterministic verifier checks correctness and benchmarks every experiment before accepting an improvement.

Pi accesses its reasoning model through an **OpenAI-compatible endpoint via LiteLLM**.

At the end of the optimization budget, OpenBaseten launches the best verified configuration it found and gives you:

- an **OpenAI-compatible inference endpoint**
- a performance and capacity report
- the complete experiment history
- a reproducible deployment recipe

You bring the GPU machines. OpenBaseten handles optimization and serving.

## Why?

The best inference engine, parallelism strategy, batching configuration, serving settings, and kernels depend on your model, hardware, and traffic.

OpenBaseten automates that experimentation to get more throughput or lower latency from GPU hardware you already have.

See [instructions.md](instructions.md) for more details.
