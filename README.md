# Open-Baseten-Inference

Finds the fastest way to serve an LLM on NVIDIA GPUs, then serves it, in your own cloud account.

**What it is.** You bring a GCP, Azure or AWS account (1-8 GPUs single node, or a Ray
cluster). ServePilot benchmarks the real serving layouts (replicas, GPUs per replica, vLLM or
SGLang, load) for time to first token and throughput, picks the winner, and puts it behind one
OpenAI-compatible URL. SkyPilot creates the machines in your account; ServePilot serves on them
and tears them down. GCP, Azure and AWS are the only clouds supported.

**Why it exists.** Picking a serving layout by hand is guesswork: vLLM or SGLang, how many GPUs
per model instance, and how many instances to run. If the full LLM fits on one GPU, ServePilot
can run it there. If it needs more memory, ServePilot automatically splits it across multiple
GPUs that work together as one instance, including across machines for supported setups. When
memory allows, it also tests running multiple independent instances of the same full model to
handle more requests. ServePilot benchmarks these arrangements and serving engines, then selects
the best measured configuration for your workload’s throughput or response latency.

```bash
git clone https://github.com/colinlikescode/Open-Baseten-Inference.git
cd Open-Baseten-Inference
pip install -e .             # plus: pip install vllm   and/or   pip install "sglang[all]"
servepilot serve Qwen/Qwen3-32B
```

API is at `http://127.0.0.1:8000/v1`. Ctrl-C stops everything and frees the GPUs.

```bash
servepilot serve Qwen/Qwen3-32B --objective latency --expected-concurrency 32
servepilot serve Qwen/Qwen3-32B --profile long-context --gpus 0,1,2,3 --engine sglang

# in your own GCP / Azure / AWS account (billed to you). Credentials must already be set up:
# gcloud auth, az login, or aws configure. `sky check` confirms SkyPilot can see them.
pip install -e ".[cloud]" && sky check
servepilot launch Qwen/Qwen3-32B --cloud gcp --accelerators H100:8
servepilot launch Qwen/Qwen3-235B-A22B --cloud aws --instance p5.48xlarge --nodes 2
servepilot down servepilot-qwen3-32b

# what would fit on a machine you have not rented yet (no GPUs or credentials needed)
servepilot plan Qwen/Qwen3-235B-A22B --cloud azure --instance Standard_ND96isr_H100_v5 --nodes 2
```

Cloud launches from this checkout build and upload a wheel of the current code, including local
fixes. Only the wheel is uploaded. Use `--package /path/to/servepilot.whl` or `--package
git+https://...@COMMIT` to deploy a specific build. `--dry-run` prepares the package and task
without renting machines. ServePilot is currently installed from source, not PyPI.

## Commands

```text
servepilot serve MODEL                 tune if needed, then serve        --retune --no-tune --dry-run
servepilot plan | tune MODEL           show layouts / benchmark them and save the winner
servepilot launch MODEL --cloud gcp|azure|aws --instance TYPE | --accelerators H100:8 [--nodes N]
servepilot clusters | down NAME        list / tear down launched machines
servepilot benchmark URL               benchmark any OpenAI-compatible server
servepilot doctor | status | stop      check the machine / what is running / stop it
servepilot inspect hardware|model|cloud
```

Every command takes `--json`. `-v` shows progress, `-vv` shows engine logs.

## Docs

[how it works](docs/architecture.md) · [planner](docs/planner.md) ·
[benchmarking](docs/benchmarking.md) · [engines](docs/engines.md) ·
[GPU validation results](docs/validation.md) · [config file](servepilot.example.yaml) ·
[security](SECURITY.md)

Needs Linux, NVIDIA drivers, Python 3.11+. Tests and planning run anywhere:
`pip install -e ".[dev]" && pytest`. Apache 2.0.
