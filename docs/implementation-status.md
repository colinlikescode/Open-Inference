# Open Sandbox implementation audit

This records the implemented scope in [instructions.md](../instructions.md) and the evidence
available without GPUs. Checked items mean implementation and applicable CPU verification are
complete. They do not certify real GPU execution or measured inference performance.

- [x] Existing machines only: provisioning commands, dependencies, and current setup documentation removed.
- [x] SSH inventory supports a CPU controller and GPU workers; local control uses the same interface.
- [x] Bootstrap, hardware/driver/CUDA discovery, topology, pairwise TCP/bandwidth and NCCL preflight paths.
- [x] Single-node and multi-node vLLM/SGLang, native or managed Ray execution, and owned-process cleanup.
- [x] Pi uses a LiteLLM OpenAI-compatible endpoint, restricted experiment tools, and experiment history.
- [x] Isolated configuration, topology, engine-code, CUDA/Triton source and dependency experiments.
- [x] Independent correctness checks, fixed objectives/SLOs, controller measurements and deterministic acceptance.
- [x] Workload YAML and JSONL replay with recorded inputs, outputs, seeds and generation parameters.
- [x] `optimize`: baseline, bounded experiments, rollback, plateau/manual/deadline stopping, verified deployment.
- [x] Durable history, setup checkpoints, interrupted-baseline recovery and compatible resume.
- [x] Measured capacity curves and recommended/max-tested concurrency under the supplied constraints.
- [x] Recipe, HTML report, JSONL history, raw benchmarks, optional profiles, patches/kernels, Dockerfile and launch script.
- [x] `deploy`: hardware compatibility, checksummed exact-image replay, readiness and correctness verification.
- [x] Inspect, doctor, status and stop cover managed workers without terminating machines.
- [x] CPU unit/integration tests, real Pi API flow, lint, type checking, shell syntax and packaging checks.
- [x] CPU validation and remaining hardware qualification are documented separately.

## Verification evidence

| Area | Evidence |
| --- | --- |
| Optimization and deployment | `tests/integration/test_optimization_workflow.py` runs real HTTP inference subprocesses through baseline, improvement, failed launch, impossible SLO, resume, clean recipe deployment and setup/startup deadlines. The model and GPU data are simulated. |
| Agent boundary | `tests/integration/test_pi_agent.py` runs the installed Pi CLI against a compatible test endpoint and exercises controller tools. A separate live Pi → LiteLLM → Gemini test completed a tool call; see [validation](validation.md). |
| Correctness and evidence | `tests/unit/test_optimization_contract.py` checks streaming parity, tolerances, repeated trials, missing/invalid evidence, immutable history, locks, deadlines, replay and altered/missing/escaping image archives. Benchmark client tests reject inflated engine token counts. |
| Remote control | `tests/unit/test_ssh_control.py` exercises quoting, CPU-controller inventory, live guardian processes, child cleanup and connection loss. |
| Distributed containers | `tests/unit/test_container_runtime.py` records native vLLM/SGLang ranks, GPU UUID placement, vLLM data parallel ranks, SGLang prefill/decode stages, isolation and failed-launch cleanup. Its TCP preflight uses real local sockets. Docker containers are not executed by these tests. |
| Profiling | `tests/unit/test_profiling.py` validates bounded trace processing and archive handling without extracting untrusted paths. GPU trace capture needs target hardware. |
| Distribution | Ruff lint/format, strict mypy, bootstrap shell syntax, source/wheel builds, packaged scripts and CLI entry points checked. |

## Hardware qualification still required

No GPU or running Docker daemon was available for this revision. Target-cluster qualification
still needs image pulls/builds/imports, NVIDIA runtime compatibility, NCCL, distributed
engine startup, profiler capture and model-specific numerical behavior. Automatic GPU
preflight, correctness checks and repeated measurements must pass before a production winner
is accepted. Historical GPU results in `validation.md` belong to the previous manual serving path.

SGLang prefill/decode currently supports TP/PP prefill groups and one decode group; unsupported
layouts fail explicitly. Default baseline comparisons detect regressions; user-supplied
goldens are needed to test task quality. Exact image archives can require substantial storage.
