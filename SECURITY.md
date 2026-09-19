# Security

Report vulnerabilities privately to the maintainers listed on the repository page.

## Optimization boundary

Pi connects to its reasoning model through your LiteLLM OpenAI-compatible endpoint. Its
provider key is read from the configured environment variable and is not saved in recipes,
logs, or generated provider configuration. Pi's built-in shell, filesystem tools, extensions,
skills, and context discovery are disabled. A private authenticated controller bridge exposes
only the experiment tools.

Agent commands run inside disposable engine containers on machines in `nodes.yaml`. These
containers do not mount the controller filesystem, Docker socket, host credentials, or
writable model weights. Successful edits become image layers. The controller owns the fixed
workload, correctness suite, measurements, score calculation, and append-only experiment
history. Profiler traces and engine logs are diagnostic data, never authoritative scores.

The controller stages private models using a trusted, digest-pinned base image. Hugging Face
credentials travel through SSH stdin to that downloader, and are not provided to experimental
images. Engine containers read a model cache volume. The older manual `serve`/`tune` commands
forward model-download credentials to their engine processes; they do not provide the
optimizer's container boundary.

Containers share the host kernel. GPU serving uses host networking and selected NVIDIA devices;
InfiniBand devices are exposed when present. This boundary is intended for controlled runtime
experiments on trusted infrastructure, not for running arbitrary hostile tenants together.
SSH host-key verification remains enabled. Bootstrap can install Docker and NVIDIA Container
Toolkit using root or passwordless sudo on supplied workers; `--no-bootstrap` disables it.

## Process and API ownership

Remote guardians stop owned processes and containers when their controller connection closes
or its heartbeat lease expires. Cleanup targets the run's container names/labels. Commands
never create or terminate machines, prune Docker globally, or stop unrelated workloads.
Local stop operations validate PID creation times before signaling a process.

The public inference API defaults to `127.0.0.1` and does not add authentication. Use your own
authenticated reverse proxy when exposing it. Keep worker inference and distributed-runtime
ports on the cluster network. `--trust-remote-code` is opt-in.

Run artifacts can contain your model prompts and outputs, machine addresses, source patches,
and diagnostic traces. Store and share them accordingly. Reasoning requests include cluster
metadata and experiment evidence with the LiteLLM provider you configure.
