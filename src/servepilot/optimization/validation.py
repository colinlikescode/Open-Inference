"""Validate agent-authored configurations against the immutable run contract."""

from __future__ import annotations

import math

from servepilot.engines.registry import EngineRegistry
from servepilot.exceptions import ConfigurationError
from servepilot.optimization.schemas import ExperimentProposal, RunDefinition

# These fields belong to the plan/model contract, not the extra argument escape hatch.
OWNED_ARGUMENTS = {
    "model",
    "model-path",
    "revision",
    "tokenizer",
    "tokenizer-path",
    "tokenizer-revision",
    "served-model-name",
    "host",
    "port",
    "api-key",
    "trust-remote-code",
    "max-model-len",
    "context-length",
    "tp",
    "tp-size",
    "tensor-parallel-size",
    "pp",
    "pp-size",
    "pipeline-parallel-size",
    "dp",
    "dp-size",
    "data-parallel-size",
    "data-parallel-size-local",
    "data-parallel-start-rank",
    "data-parallel-address",
    "data-parallel-rpc-port",
    "data-parallel-backend",
    "headless",
    "nnodes",
    "node-rank",
    "master-addr",
    "master-port",
    "dist-init-addr",
    "distributed-executor-backend",
    "gpu-memory-utilization",
    "mem-fraction-static",
    "max-num-seqs",
    "max-running-requests",
    "cuda-visible-devices",
    "load-format",
    "profiler-config",
    "disaggregation-mode",
    "disaggregation-transfer-backend",
    "disaggregation-bootstrap-port",
    "base-gpu-id",
}


def validate_proposal(
    proposal: ExperimentProposal, definition: RunDefinition, registry: EngineRegistry
) -> None:
    plan = proposal.plan
    if plan.engine.value not in definition.engine_versions:
        raise ConfigurationError(f"engine {plan.engine} was not prepared for this run")
    if not plan.gpu_groups or len(plan.gpu_groups) != plan.replica_count:
        raise ConfigurationError("each replica must have one nonempty GPU group")
    ids = plan.gpu_ids
    if len(set(ids)) != len(ids) or not ids or any(not group for group in plan.gpu_groups):
        raise ConfigurationError("a GPU cannot belong to multiple replicas")
    available = {gpu.index for gpu in definition.hardware.gpus}
    if not set(ids) <= available:
        raise ConfigurationError("proposal refers to GPUs outside the supplied inventory")
    required = plan.tensor_parallel_size * plan.pipeline_parallel_size
    if not plan.dp_attention_enabled:
        required *= plan.data_parallel_size
    if any(len(group) != required for group in plan.gpu_groups):
        raise ConfigurationError("GPU groups must match the requested parallelism")
    if plan.prefill:
        if (
            plan.engine != "sglang"
            or plan.replica_count != 1
            or plan.data_parallel_size != 1
            or plan.dp_attention_enabled
        ):
            raise ConfigurationError(
                "PD execution currently requires SGLang, one decode group, and TP/PP rather than DP attention"
            )
        size = plan.prefill.tensor_parallel_size * plan.prefill.pipeline_parallel_size
        if any(len(group) != size for group in plan.prefill.gpu_groups):
            raise ConfigurationError("prefill GPU groups must match prefill TP × PP")
    if plan.context_length < definition.workload.max_context_tokens:
        raise ConfigurationError("an experiment cannot reduce the required context length")
    if plan.objective is not None and plan.objective != definition.workload.objective:
        raise ConfigurationError("an experiment cannot change the user's objective")
    if plan.memory_fraction is not None and (
        not math.isfinite(plan.memory_fraction) or not 0 < plan.memory_fraction < 1
    ):
        raise ConfigurationError("memory_fraction must be finite and between zero and one")
    for value in (
        plan.max_concurrency,
        plan.max_num_seqs,
        plan.max_running_requests,
        plan.chunked_prefill_size,
    ):
        if value is not None and value < 1:
            raise ConfigurationError("batch and concurrency limits must be positive")
    for key in plan.engine_args:
        normalized = key.lstrip("-").replace("_", "-").split("=", 1)[0].split(".", 1)[0]
        if normalized in OWNED_ARGUMENTS:
            raise ConfigurationError(f"engine_args cannot override controller-owned field {key!r}")
    actual_nodes = [
        [definition.hardware.gpu(index).node_id or "local" for index in group]
        for group in plan.gpu_groups
    ]
    if plan.replica_nodes is not None and plan.replica_nodes != actual_nodes:
        raise ConfigurationError("replica node placement does not match the GPU groups")
    if any(len(set(nodes)) > 1 for nodes in actual_nodes) and not plan.distributed_backend:
        raise ConfigurationError("a multi-node replica must select a distributed runtime")
    support = registry.require(plan.engine).supports(definition.model, plan)
    if not support.supported:
        raise ConfigurationError("engine does not support this experiment", hints=support.reasons)
