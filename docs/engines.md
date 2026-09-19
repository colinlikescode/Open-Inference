# Engines

ServePilot does not run models itself. It starts vLLM or SGLang and talks to them over
their OpenAI-compatible HTTP APIs.

## Optimizer runtimes

`optimize` prepares separate official vLLM and SGLang container images on the supplied GPU
workers. Defaults are `vllm/vllm-openai:v0.28.0` and `lmsysorg/sglang:v0.5.19`; tags are resolved
to immutable digests before experiments. `--runtime-config` overrides images and shared-memory
size. Docker and NVIDIA Container Toolkit can be bootstrapped; drivers and machine access must
already work. The CPU controller does not need either inference engine installed.

Each replica gets only its selected GPU UUIDs and a read-only model cache. Native vLLM
multi-node execution uses `vllm serve`, `--nnodes`, `--node-rank`, and headless workers.
SGLang uses `--nnodes`, `--node-rank`, and `--dist-init-addr`. A plan selecting Ray gets a managed
head and workers inside its engine image; no existing Ray cluster is required.

SGLang prefill/decode experiments use `CandidatePlan.prefill`: one or more prefill groups,
their TP/PP layout, memory fraction, and `nixl` or `mooncake`. The plan's ordinary GPU group is
the decode stage. The controller starts the stages, waits for each, and starts a GPU-free
SGLang model gateway. This path currently supports one decode group with TP/PP; unsupported
DP-attention or engine combinations are rejected before launch. Transfer-engine and router
dependencies must exist in the experimental image; Pi can install them through runtime tools.

`profile: true` captures diagnostic Torch traces after the scored trials, with separate
prefill and decode captures. Traces and summaries are saved under `profiler-results/`.
They cannot contribute a score. Runtime commands, image archives, environment settings,
patches, and source files are retained in the recipe.

## Finding an engine for manual commands

For each engine, in order:

1. `SERVEPILOT_VLLM_PYTHON` / `SERVEPILOT_SGLANG_PYTHON`: a Python interpreter that has it.
2. The interpreter running ServePilot, if the package is installed there.
3. A console script on `PATH` (`vllm`); its shebang tells us the interpreter.
4. `~/engines/vllm/bin/python`, `~/engines/sglang/bin/python`, `/opt/vllm`, `/opt/sglang`.

Keeping each engine in its own virtual environment is the safest setup; their CUDA
dependencies often conflict. The engine's `bin/` directory is put on `PATH` for the engine
process so its own tools (`ninja` for JIT kernels, for example) are found.

`servepilot doctor` shows what was found and which interpreter it will use.

## vLLM

Started as `PYTHON -m vllm.entrypoints.openai.api_server --model MODEL ...`. Flags used:
`--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size` (0.8+),
`--distributed-executor-backend ray` (cross-node), `--gpu-memory-utilization`,
`--max-model-len`, `--max-num-seqs`, `--enable-expert-parallel` (0.6.4+),
`--enable-chunked-prefill` / `--no-enable-chunked-prefill`, `--max-num-batched-tokens`,
`--kv-cache-dtype`, `--dtype`, `--quantization` (only for gguf/bitsandbytes; the rest is
auto-detected), `--revision`, `--served-model-name`, `--trust-remote-code`,
`--disable-uvicorn-access-log` (0.7+), `--performance-mode` (0.17+). Minimum version 0.6.0.

`--performance-mode` follows the objective: `throughput` → `throughput` (larger CUDA graphs,
double the default batch limits), `latency` → `interactivity` (fine-grained CUDA graphs for
small batches), `balanced` → vLLM's default.

DP-attention candidates for MoE models (0.9+) launch as `--tensor-parallel-size 1
--data-parallel-size N --enable-expert-parallel` on the same N GPUs: attention runs data
parallel, experts are sharded. `--max-num-seqs` applies per DP rank, so the plan's limit is
divided by N.

Version differences live in one small table in `engines/vllm.py`.

## SGLang

Started as `PYTHON -m sglang.launch_server --model-path MODEL ...`. Flags used: `--tp-size`,
`--pp-size`, `--dp-size`, `--ep-size`, `--enable-dp-attention`, `--mem-fraction-static`,
`--context-length`, `--max-running-requests`, `--cuda-graph-max-bs-decode` (0.5.18+;
`--cuda-graph-max-bs` before), `--chunked-prefill-size`, `--kv-cache-dtype`, `--dtype`,
`--quantization` (gguf/bitsandbytes), `--revision`, `--served-model-name`,
`--trust-remote-code`, `--log-level warning`. Minimum version 0.4.0.

`--mem-fraction-static` is not the plan's memory fraction verbatim. SGLang's fraction covers
weights + KV pool only and allocates activations and CUDA graphs outside it, whereas
ServePilot's fraction budgets all of those. The adapter subtracts the estimate's activation,
CUDA-graph, communication and overhead bytes so SGLang gets the same total budget as vLLM
(on an idle 80 GB GPU: 0.91 → about 0.85). Without an estimate the fraction passes through.

On GPUs with 60 GiB or more, decode CUDA graphs are captured up to the running-request limit
(at most 512) whenever that exceeds SGLang's default of 256; larger decode batches would
otherwise run eagerly.

DP attention is only offered for architectures it is documented to work with (DeepSeek V2/V3,
Qwen MoE, GLM-4 MoE).

## What every adapter does

- `supports(model, plan)`: can this engine run this model in this layout? Returns reasons and
  warnings, so an excluded candidate always says why.
- `build_launch_spec(...)`: the exact command, environment (`CUDA_VISIBLE_DEVICES`,
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, forwarded `HF_*` variables) and a redacted display string.
- `wait_until_ready(...)`: polls `/v1/models` until it answers, or the process exits, or the
  timeout passes. Process exit is classified from the log tail.
- `classify_failure(...)`: OOM, CUDA error, NCCL error, unsupported model, bad argument, port
  conflict, download/auth error, crash, timeout.

Extra engine flags go through `constraints.engine_args` in the config file
(`enable-prefix-caching: true`, `seed: 7`).

## Environment forwarded to engines

`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`,
`HF_ENDPOINT`, `TRANSFORMERS_CACHE`, `PATH`, `HOME`, `LD_LIBRARY_PATH`, `NCCL_*`,
`GLOO_SOCKET_IFNAME`, `RAY_ADDRESS`, plus `PYTHONUNBUFFERED=1`. Token values are redacted
everywhere ServePilot prints a command.

## The fake engine

`SERVEPILOT_ENABLE_FAKE_ENGINE=1` adds an engine named `fake` that launches
`servepilot.testing.fake_openai` as a real subprocess: an OpenAI-compatible server with
configurable time to first token, time per token, capacity and startup failure modes (OOM,
crash, hang, unsupported model, NCCL error). Together with `SERVEPILOT_FAKE_HARDWARE` it runs
the whole tune-and-serve loop on any machine.
