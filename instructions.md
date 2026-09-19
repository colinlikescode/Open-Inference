# Open BaseTen — Product and Implementation Instructions

**Give Open BaseTen your model, your existing GPU machines, and an hour. It experiments with the inference stack, deploys the best verified configuration it finds, and saves a reproducible optimization report.**

An autonomous inference performance engineer for GPUs you already have.

This is the detailed setup and operating guide. GPU runtime paths are implemented, but this revision has been tested without real GPUs; see [validation](docs/validation.md) for the distinction between CPU checks and historical GPU measurements.

The product is named **Open BaseTen**. The existing Python package and CLI are still named `servepilot`; command examples below use that current executable name.

## The product boundary

You provide:

- Existing Linux machines with NVIDIA GPUs and working drivers.
- SSH access from the head node to the workers, and networking between the machines.
- A model, a workload, an optimization objective, and a time budget.

Open BaseTen returns:

- An OpenAI-compatible inference endpoint.
- The best verified configuration discovered within the budget.
- A performance and capacity report.
- A reproducible deployment recipe, including any code changes.
- The complete experiment history, which you can resume later.

**Open BaseTen does not provision infrastructure.** It does not create or terminate machines, select cloud regions, compare GPU rental prices, or manage cloud credentials or quotas. Provisioning commands and dependencies have been removed.

The machines can be in any cloud, a private datacenter, a university cluster, or on premises. The product begins after those machines exist. No Kubernetes cluster or preconfigured Ray cluster is required.

## Basic usage

Install the controller on the CPU cluster head or dedicated Linux machine:

```bash
git clone https://github.com/colinlikescode/Open-Baseten-Inference.git
cd Open-Baseten-Inference
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[hf]'
# Node.js 22.19+ is required for Pi.
npm install -g @earendil-works/pi-coding-agent@0.83.0
```

Use Python 3.11 or newer for the controller. GPU workers need Linux, working NVIDIA drivers,
SSH, and storage for engine images and model weights. Bootstrap supports apt/dnf machines and
needs root or passwordless sudo when installing Docker/NVIDIA Container Toolkit. Existing
installations work with `--no-bootstrap`. The controller needs storage for exact winning image
archives. It never downloads Hub model weights onto the CPU head just to inspect the model.

Pi uses LiteLLM's **OpenAI-compatible** API. If you already run LiteLLM, set `LITELLM_BASE_URL`
and `LITELLM_API_KEY`, then pass its model alias with `--agent-model`. To run the supplied Gemini
example, install `litellm[proxy]==1.101.0` in a separate Python 3.12 environment, set
`GEMINI_API_KEY` and a local proxy `LITELLM_API_KEY`, and start:

```bash
DEBUG=false litellm --config examples/litellm.yaml --host 127.0.0.1 --port 4000
```

Run optimization in another terminal with `LITELLM_API_KEY` set. Provider settings are in
`examples/agent.yaml`; change the alias there and use `--agent-config` if needed. Keys belong
in environment variables, not YAML or Git. Set `HF_TOKEN` only when model access requires it.

Check workers before starting a long run:

```bash
servepilot inspect hardware --nodes nodes.yaml
servepilot doctor --nodes nodes.yaml
```

Doctor checks SSH/GPU discovery, controller reachability, pairwise TCP bandwidth, Docker/NVIDIA
runtime availability, and local Pi configuration. Optimization additionally checks the selected
images against CUDA and runs an inter-node NCCL collective. Nodes must be mutually reachable;
an SSH alias can specify an `address` for the inference network.

SSH into the machine you want to use as the head node:

```bash
ssh ubuntu@10.0.0.1
```

Describe your existing machines in `nodes.yaml`:

```yaml
nodes:
  - host: 10.0.0.1
  - host: 10.0.0.2
  - host: 10.0.0.3
  - host: 10.0.0.4

ssh_user: ubuntu
```

The first listed GPU machine is the distributed runtime head by default; `head: HOST` overrides this. Run the controller there or on a separate CPU machine with network access to every worker.

Alternatively, clone and run Open BaseTen on a dedicated Linux head machine with SSH access to the GPU workers listed in `nodes.yaml`. This control machine can be CPU-only; GPU discovery and inference then run on the workers, while Pi, the verifier, and the public endpoint run on the control machine.

Then optimize:

```bash
servepilot optimize \
  --model Qwen/Qwen3-235B-A22B \
  --nodes nodes.yaml \
  --minutes 60 \
  --objective throughput \
  --max-p95-ttft 300ms \
  --output ./servepilot-output
```

For a single machine, omit `--nodes` to use the local GPUs:

```bash
servepilot optimize --model Qwen/Qwen3-32B --minutes 15
```

For an overnight run:

```bash
servepilot optimize \
  --model Qwen/Qwen3-235B-A22B \
  --nodes nodes.yaml \
  --hours 24 \
  --objective balanced
```

Open BaseTen handles runtime setup, distributed launch, experiments, benchmarking, rollback, and final deployment. You do not manually configure Ray, inference workers, TP/PP layouts, NCCL settings, or benchmark and production launch scripts.

SSH is used for setup and process control. Inference requests go directly to the HTTP endpoint, not through SSH.

## How optimization works

```text
Existing GPUs + SSH
        ↓
Inspect hardware, topology, network, and model
        ↓
Launch and verify a baseline
        ↓
Pi proposes an experiment
        ↓
Apply changes in an isolated experiment environment
        ↓
Launch → correctness tests → performance benchmark
        ↓
Open BaseTen's deterministic verifier accepts or rejects it
        ↓
Keep the improvement or revert; record the result
        ↓
Repeat until the budget expires or the search is stopped
        ↓
Deploy the best verified configuration
        ↓
Endpoint + report + reproducible recipe
```

Before launching experiments, Open BaseTen checks SSH access, GPU availability, driver and CUDA compatibility, inter-node connectivity, bandwidth, and the communication requirements of the selected runtime, including NCCL where needed. Failures identify the affected machine and check.

The optimization budget includes inspection, setup, baseline evaluation, and experiments. Open BaseTen stops starting new experiments when the budget expires and bounds running experiments by the remaining time. Cleanup and final deployment can take additional time and are reported separately.

The search can also stop manually or after a configured period without improvement. Stopping preserves completed experiments and the best verified result. If there is no verified result, Open BaseTen reports that outcome instead of claiming success.

Open BaseTen reports **the best verified configuration discovered within the allocated budget**. It does not claim a global optimum.

## Pi is the inference engineer

Pi decides what experiment to try next. It receives the model, hardware and network topology, current configuration, best metrics, experiment history, failed attempts, profiler output, objective, and remaining time.

Pi accesses its reasoning model through an OpenAI-compatible endpoint via LiteLLM.

Its tools cover cluster inspection, file reads and edits, patching, shell commands, dependency installation, server lifecycle, profiling, documentation lookup, and requests to run benchmarks and correctness tests.

The target scope includes all three levels:

| Level | Examples |
| --- | --- |
| Serving configuration | vLLM versus SGLang; tensor, pipeline, data, and expert parallelism; replica count; batching; scheduling; memory utilization; KV cache; context limits; chunked prefill; prefix caching; speculative decoding; quantization; CUDA graphs; compilation settings. |
| Distributed topology | GPU grouping, replica placement, TP within or across machines, PP across machines, expert placement, network-aware layouts, and prefill/decode separation. |
| Code optimization | Changes to vLLM or SGLang, Triton and CUDA kernels, custom operators, attention and MoE implementations, memory management, routing, and batching. |

This is intended to go beyond a fixed sweep of serving flags. Every configuration or code change is an experiment, and every accepted change must pass verification.

## Open BaseTen owns verification

**Pi proposes changes. Open BaseTen decides whether they worked.**

The deterministic evaluator controls benchmark inputs, correctness checks, measurements, scoring, and acceptance. Pi cannot edit scores, change the objective, disable checks, silently loosen SLOs, declare a winner, or rewrite historical results.

Experiments run in isolated environments. Agent write access is limited to the experimental stack; verifier code and authoritative results remain outside that writable environment. Rejected experiments are stopped and reverted. Accepted changes are saved as reproducible artifacts.

Correctness is a required gate, not an agent judgment. The verification suite and any numerical or quality tolerances are fixed before the search. All required checks must pass. Changes such as quantization or custom kernels must meet those same checks; a faster run is insufficient by itself. The report records what was tested and the limits of that verification.

For example, a throughput objective can require:

```text
Maximize output tokens per second
Subject to:
  P95 TTFT < 300 ms
  P95 TPOT < 30 ms
  All required correctness checks pass
  No OOMs, crashes, or failed benchmark requests
```

The presets are `throughput`, `latency`, and `balanced`. Explicit SLOs constrain selection, including latency ceilings and throughput floors. Comparisons use a consistent workload and benchmark protocol, with confirmation measurements before accepting improvements. Missing required metrics or violated constraints cannot be reported as a verified win.

## Optimize for your workload

Describe the traffic distribution in YAML:

```yaml
workload:
  input_tokens:
    p50: 2000
    p95: 12000
  output_tokens:
    p50: 500
    p95: 2000
  concurrency:
    expected: 64
    peak: 128
```

```bash
servepilot optimize \
  --model Qwen/Qwen3-32B \
  --minutes 60 \
  --workload workload.yaml
```

Or replay representative requests:

```bash
servepilot optimize \
  --model Qwen/Qwen3-32B \
  --minutes 60 \
  --workload requests.jsonl
```

Benchmark inputs, generation settings, seeds, and measured outputs are saved with the run so comparisons can be reproduced.

## Multi-node inference

One machine and sixteen machines use the same top-level interface. Open BaseTen determines how to distribute models that cannot fit or perform well on one machine.

The runtime is an implementation choice for each supported experiment: vLLM with Ray, native distributed execution in vLLM or SGLang, or PyTorch distributed and NCCL where appropriate. Ray can be installed and managed when a selected engine configuration needs it; it is not the foundation or a universal requirement of Open BaseTen.

Open BaseTen validates engine and hardware compatibility before attempting a layout. Unsupported combinations and failed communication checks are reported explicitly.

## Results and capacity planning

At the end of optimization, Open BaseTen creates a clean deployment from the best verified experiment and reports its endpoint after readiness checks pass.

The report includes:

- Model revision, cluster inventory, engine versions, and software environment.
- Allocated budget, actual search duration, setup and deployment time, and stopping reason.
- Baseline versus best throughput, TTFT, TPOT, and end-to-end latency.
- Correctness results, request failures, SLO compliance, and measured improvement.
- Performance at each tested concurrency, plus recommended production concurrency and the highest tested concurrency satisfying the SLOs.
- Every experiment's hypothesis, changes, logs, measurements, acceptance decision, and failure reason where applicable.

Capacity claims are based on tested load levels, not extrapolated promises. If no configuration satisfies the constraints, the report says so rather than labeling an infeasible candidate successful.

## Reproducible artifacts

```text
servepilot-output/
├── recipe.yaml
├── report.html
├── experiments.jsonl
├── experiments/
│   ├── experiment_0001/
│   └── experiment_0002/
├── benchmark-results/
├── profiler-results/
├── patches/
├── kernels/
├── Dockerfile
└── launch.sh
```

The recipe captures the model revision, exact engine versions, runtime dependencies, GPU layout, parallelism, serving flags, benchmark configuration, and any accepted patches or kernels. Logs retain the exact launch commands without embedding credentials. The deployment artifacts reproduce the selected environment and changes.

Deploy the recipe on equivalent hardware without repeating the optimization search:

```bash
servepilot deploy ./servepilot-output/recipe.yaml --nodes nodes.yaml
```

Deployment validates hardware and runtime compatibility and performs readiness checks. It does not assume that a recipe's performance measurements transfer unchanged to different machines.

Continue an optimization run with more time:

```bash
servepilot optimize --resume ./servepilot-output --minutes 120
```

Resuming loads the original objective, constraints, workload, best verified configuration, and experiment history. It validates compatibility with the current environment and appends new experiments without rewriting the earlier results.

If the budget expires during initial setup, the output directory still contains a setup
checkpoint and report. Use the same `--resume` command to continue preparation with more time;
the original configuration and workload files must remain unchanged.

## Commands

| Command | Purpose |
| --- | --- |
| `servepilot optimize` | Optimize a model on existing machines for a time budget, then deploy the best verified result. |
| `servepilot deploy recipe.yaml` | Reproduce a saved deployment without repeating the search. |
| `servepilot benchmark URL` | Benchmark an existing OpenAI-compatible endpoint. |
| `servepilot inspect` | Inspect hardware, models, and network topology. |
| `servepilot doctor` | Check machine prerequisites and connectivity. |
| `servepilot status` | Show optimization or deployment status. |
| `servepilot stop` | Stop Open BaseTen-managed processes, preserving artifacts and leaving the machines running. |

Cloud provisioning commands such as `launch`, `clusters`, and `down`, cloud instance planning, and the SkyPilot dependency are outside the product and have been removed.

### Operational details

- Optimization serves in the foreground after saving its winner. Keep it under your service
  manager for an unattended deployment. `--no-deploy` saves artifacts and exits.
- `servepilot status` shows setup/search progress or the live deployment. `servepilot stop`
  stops managed processes; completed experiments and recipes remain. Ctrl-C during search
  stops the search and saves the best result without starting a new serving process.
- `--minutes` and `--hours` are mutually exclusive. `--plateau-minutes` stops after that long
  without a verified improvement. `--concurrency 8,16,32` fixes the tested load levels;
  `--requests-per-trial` and `--repetitions` control confirmation effort.
- SLO values accept `300ms`, `0.3s`, or bare milliseconds. `--min-throughput` adds a hard output
  tokens/second floor. There is no fallback to a result that fails the constraints.
- `--correctness examples/correctness.yaml` supplies user goldens and fixed tolerances.
  Baseline comparisons alone are regression checks, not a semantic-quality evaluation.
- `--runtime-config examples/runtime.yaml` selects images and shared-memory size. Accepted
  runtime edits are exported as checksummed Docker image archives under `images/`, in addition
  to the Dockerfile and recorded commands. Exact archive replay is used by `deploy`.
- Resume requires the original GPU identities, topology, model, and engine environment.
  Deploy accepts equivalent hardware and remaps node names; performance is not re-estimated.
  Local-model recipes require the original model directory and verify its contents by hash.
- Pi can request GPU profiles with `profile: true`. Captures run after scored trials, retain
  per-node archives and summaries, and never change the score. Shell/file/dependency edits
  happen inside isolated images; GPU execution happens through verified experiments.
- SGLang prefill/decode experiments currently support TP/PP prefill groups and one decode
  group, using NIXL or Mooncake plus the SGLang gateway. Unsupported combinations fail with
  a specific compatibility error. See [engine details](docs/engines.md).

## Development

The repository is currently installed from source:

```bash
git clone https://github.com/colinlikescode/Open-Baseten-Inference.git
cd Open-Baseten-Inference
pip install -e ".[dev]"
pytest
```

Real inference requires Linux and supported NVIDIA GPUs and drivers. The autonomous search requires Pi and a configured LiteLLM endpoint. Open BaseTen manages the inference runtimes on the supplied machines; provider authentication and machine access are supplied by the user.

See [architecture](docs/architecture.md), [planner](docs/planner.md), [benchmarking](docs/benchmarking.md), and [engine](docs/engines.md) for implementation details. Historical [GPU validation results](docs/validation.md) do not validate the new SSH/container optimizer.

Apache 2.0.
