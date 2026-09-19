"""Candidate topology generation.

For ``G`` selected GPUs the planner enumerates tensor-parallel sizes from the divisors of the
per-machine GPU count, derives the replica count that fully uses the GPUs, assigns GPU groups by
topology, asks each engine adapter whether it supports the model, and attaches a static memory
estimate that decides *estimated* viability. Benchmarking decides the rest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from servepilot.constants import BUSY_GPU_USED_FRACTION, GIB
from servepilot.engines.base import InferenceEngine
from servepilot.exceptions import ConfigurationError, HardwareError, NoViablePlanError
from servepilot.models.kv_cache import estimate_kv
from servepilot.planner.gpu_groups import GPUGroupPlanner
from servepilot.planner.memory import MemoryModelConfig, estimate_memory, format_bytes
from servepilot.planner.pruning import (
    check_context_length,
    check_ep_divisibility,
    check_memory,
    check_tp_divisibility,
    dedupe,
    divisors,
)
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import (
    CandidatePlan,
    CandidateViability,
    EngineName,
    ExcludedCandidate,
    KVEstimate,
    MemoryEstimate,
    PlanConstraints,
    PlanningResult,
)
from servepilot.schemas.workload import Objective, WorkloadProfile

MAX_NUM_SEQS_FLOOR = 16
MAX_NUM_SEQS_CEILING = 1024


@dataclass
class GPUSelection:
    snapshot: HardwareSnapshot
    gpu_ids: list[int]
    warnings: list[str]


def select_gpus(snapshot: HardwareSnapshot, constraints: PlanConstraints) -> GPUSelection:
    """Apply ``--gpus`` and enforce homogeneity; fail clearly on mixed hardware."""
    warnings: list[str] = []
    if snapshot.gpu_count == 0:
        raise HardwareError(
            "no NVIDIA GPUs are visible to ServePilot.",
            hints=[
                "Run `nvidia-smi` to confirm the driver sees your GPUs.",
                "Check CUDA_VISIBLE_DEVICES is not hiding every device.",
            ],
        )
    available = {g.index for g in snapshot.gpus}
    if constraints.gpu_ids is not None:
        missing = sorted(set(constraints.gpu_ids) - available)
        if missing:
            raise ConfigurationError(
                f"--gpus references GPU indices {missing} that are not present (available: {sorted(available)})"
            )
        ids = sorted(set(constraints.gpu_ids))
    else:
        ids = sorted(available)
    selected = snapshot.select(ids)

    groups = selected.homogeneous_groups()
    if len(groups) > 1:
        lines = [
            "Mixed GPU types/memory sizes were detected. Heterogeneous tensor-parallel topologies are not supported in v1."
        ]
        lines.append("Homogeneous subsets you can select explicitly:")
        for (name, mem), idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines.append(
                f"  --gpus {','.join(str(i) for i in idx)}   ({len(idx)} × {name}, {mem / GIB:.0f} GiB)"
            )
        raise HardwareError(
            "\n".join(lines), hints=["Re-run with --gpus <indices> naming one homogeneous subset."]
        )

    busy = [g for g in selected.gpus if g.used_fraction > BUSY_GPU_USED_FRACTION]
    if busy:
        detail = ", ".join(f"GPU {g.index}: {format_bytes(g.used_memory_bytes)} used" for g in busy)
        if constraints.allow_busy_gpus:
            warnings.append(
                f"Selected GPUs already have allocations ({detail}); planning against free memory only."
            )
        else:
            warnings.append(
                f"Selected GPUs already have allocations ({detail}). ServePilot plans against free memory only; "
                "pass --allow-busy-gpus to silence this warning or --gpus to choose idle devices."
            )
    return GPUSelection(selected, ids, warnings)


def resolve_context_length(
    model: ModelProfile, workload: WorkloadProfile, constraints: PlanConstraints
) -> tuple[int, list[str]]:
    notes: list[str] = []
    requested = constraints.context_length or workload.max_context_tokens
    limit = model.max_position_embeddings
    if limit is not None and requested > limit and constraints.context_length is None:
        needed = workload.p95_sequence_tokens
        if needed > limit:
            # Benchmark requests at the p95 lengths would be rejected by the engine, so the
            # measurements could never be valid; say so instead of tuning into failures.
            raise ConfigurationError(
                f"the workload needs {needed} tokens of context (p95 prompt + p95 output) but the "
                f"model's maximum is {limit}.",
                hints=[
                    "Lower --input-tokens-p95 / --output-tokens-p95 or pick a shorter profile.",
                    f"Or force it: --context-length {needed} --allow-context-override "
                    "(quality beyond the trained length is not guaranteed).",
                ],
            )
        notes.append(
            f"workload context {requested} exceeds the model's maximum ({limit}); using {limit}. "
            "Pass --context-length with --allow-context-override to force a longer context."
        )
        requested = limit
    return requested, notes


def _plan_id(engine: EngineName, tp: int, pp: int, replicas: int, suffix: str = "") -> str:
    base = f"{engine.value}-tp{tp}"
    if pp > 1:
        base += f"-pp{pp}"
    base += f"-x{replicas}{suffix}"
    return base


def _max_num_seqs(
    engine: InferenceEngine, est_c50: int | None, constraints: PlanConstraints
) -> int:
    if constraints.max_concurrency is not None:
        return constraints.max_concurrency
    if est_c50 is None:
        return engine.default_max_num_seqs()
    return max(MAX_NUM_SEQS_FLOOR, min(MAX_NUM_SEQS_CEILING, est_c50))


def generate_candidates(
    snapshot: HardwareSnapshot,
    model: ModelProfile,
    workload: WorkloadProfile,
    engines: Sequence[InferenceEngine],
    constraints: PlanConstraints | None = None,
    memory_cfg: MemoryModelConfig | None = None,
) -> PlanningResult:
    constraints = constraints or PlanConstraints()
    memory_cfg = memory_cfg or MemoryModelConfig()
    if constraints.memory_headroom is not None:
        memory_cfg = MemoryModelConfig(
            **{**memory_cfg.__dict__, "headroom_fraction": constraints.memory_headroom}
        )
    if not engines:
        raise NoViablePlanError("no inference engines were provided to the planner")

    selection = select_gpus(snapshot, constraints)
    selected = selection.snapshot
    result = PlanningResult(selected_gpu_ids=selection.gpu_ids, warnings=list(selection.warnings))

    context_length, ctx_notes = resolve_context_length(model, workload, constraints)
    result.notes.extend(ctx_notes)
    ctx_error = check_context_length(model, context_length, constraints.allow_context_override)
    if ctx_error:
        raise ConfigurationError(ctx_error)

    by_node = selected.gpus_by_node()
    node_sizes = [len(v) for v in by_node.values()]
    gpus_per_node = min(node_sizes)
    if len(set(node_sizes)) > 1:
        result.warnings.append(
            f"nodes expose different GPU counts {sorted(node_sizes)}; per-node topologies use {gpus_per_node} GPUs per node"
        )
    total_gpus = len(selection.gpu_ids)
    planner = GPUGroupPlanner(selected)

    def kv_for(tp: int) -> KVEstimate:
        return estimate_kv(model, tp, constraints.kv_cache_dtype)

    tp_values = divisors(gpus_per_node)
    if constraints.tensor_parallel_size is not None:
        tp = constraints.tensor_parallel_size
        if tp not in tp_values:
            if tp > gpus_per_node and selected.is_cluster:
                tp_values = []  # cross-node only, handled below
            else:
                raise ConfigurationError(
                    f"--tp {tp} is not possible on {gpus_per_node} GPUs per node (valid values: {tp_values})"
                )
        else:
            tp_values = [tp]

    plans: list[CandidatePlan] = []
    for tp in tp_values:
        reason = check_tp_divisibility(model, tp)
        if reason:
            result.excluded.append(
                ExcludedCandidate(description=f"TP={tp}", reason=reason, tensor_parallel_size=tp)
            )
            continue
        max_replicas = sum(n // tp for n in node_sizes)
        replicas = max_replicas
        if constraints.replica_count is not None:
            if constraints.replica_count > max_replicas:
                raise ConfigurationError(
                    f"--replicas {constraints.replica_count} with TP={tp} needs {constraints.replica_count * tp} GPUs "
                    f"but only {total_gpus} are selected"
                )
            replicas = constraints.replica_count
        groups = planner.plan(tp, replicas, selection.gpu_ids)
        used_gpus = [g for grp in groups for g in grp]
        est_kv = kv_for(tp)
        estimate = estimate_memory(
            model,
            gpus=[selected.gpu(i) for i in used_gpus],
            tensor_parallel_size=tp,
            kv=est_kv,
            context_length=context_length,
            workload=workload,
            cfg=memory_cfg,
            memory_fraction=constraints.memory_fraction,
        )
        mem_reason = check_memory(estimate)
        replica_nodes = (
            [[selected.node_of(g) or "local" for g in grp] for grp in groups]
            if selected.nodes
            else None
        )

        for engine in engines:
            if constraints.engine is not None and engine.engine_name != constraints.engine:
                continue
            variants: list[tuple[str, dict[str, object], list[str]]] = [("", {}, [])]
            if model.is_moe and tp > 1 and engine.supports_expert_parallel(model):
                ep_reason = check_ep_divisibility(model, tp)
                if ep_reason is None:
                    variants.append(
                        (
                            "-ep",
                            {"expert_parallel_enabled": True, "expert_parallel_size": tp},
                            [
                                f"Expert parallelism (EP={tp}) shards experts instead of replicating them; {engine.name()} supports it for this model."
                            ],
                        )
                    )
                else:
                    result.excluded.append(
                        ExcludedCandidate(
                            description=f"{engine.name()} TP={tp} EP={tp}",
                            reason=ep_reason,
                            engine=engine.engine_name,
                            tensor_parallel_size=tp,
                        )
                    )
            if model.is_moe and tp > 1 and engine.supports_dp_attention(model):
                variants.append(
                    (
                        "-dpa",
                        {
                            "expert_parallel_enabled": True,
                            "expert_parallel_size": tp,
                            "data_parallel_size": tp,
                            "dp_attention_enabled": True,
                        },
                        [
                            f"DP attention with EP={tp} is known to work for this architecture on {engine.name()}."
                        ],
                    )
                )

            for suffix, overrides, extra_rationale in variants:
                plan = CandidatePlan(
                    id=_plan_id(engine.engine_name, tp, 1, replicas, suffix),
                    engine=engine.engine_name,
                    gpu_groups=groups,
                    tensor_parallel_size=tp,
                    replica_count=replicas,
                    context_length=context_length,
                    objective=workload.objective,
                    memory_fraction=estimate.memory_fraction,
                    max_num_seqs=_max_num_seqs(
                        engine, estimate.estimated_max_concurrency_p50, constraints
                    ),
                    kv_cache_dtype=constraints.kv_cache_dtype,
                    engine_args=dict(constraints.extra_engine_args),
                    estimated_memory=estimate,
                    replica_nodes=replica_nodes,
                    **overrides,  # type: ignore[arg-type]
                )
                plan.max_running_requests = plan.max_num_seqs
                plan.max_concurrency = plan.max_num_seqs * replicas if plan.max_num_seqs else None
                support = engine.supports(model, plan)
                if not support.supported:
                    result.excluded.append(
                        ExcludedCandidate(
                            description=plan.label(),
                            reason="; ".join(support.reasons)
                            or f"{engine.name()} does not support this model",
                            engine=engine.engine_name,
                            tensor_parallel_size=tp,
                            replica_count=replicas,
                            plan=plan,
                        )
                    )
                    continue
                rationale = list(extra_rationale)
                rationale.extend(support.warnings)
                if mem_reason:
                    plan.viability = CandidateViability.ESTIMATED_IMPOSSIBLE
                    plan.rationale = [mem_reason, *rationale]
                    result.excluded.append(
                        ExcludedCandidate(
                            description=plan.label(),
                            reason=mem_reason,
                            engine=engine.engine_name,
                            tensor_parallel_size=tp,
                            replica_count=replicas,
                            plan=plan,
                        )
                    )
                    continue
                plan.viability = (
                    CandidateViability.UNKNOWN
                    if estimate.confidence == "unknown"
                    else CandidateViability.ESTIMATED_VIABLE
                )
                plan.rationale = (
                    _base_rationale(plan, estimate, planner, gpus_per_node, selected) + rationale
                )
                plans.append(plan)

    # Cross-node candidates: only when nothing fits inside one machine (or the user forced TP>node).
    if selected.is_cluster and (not plans or selected.provider == "ssh"):
        plans.extend(
            _cross_node_candidates(
                selected,
                model,
                workload,
                engines,
                constraints,
                memory_cfg,
                context_length,
                result,
                kv_for,
            )
        )

    unique, dupes = dedupe(plans)
    for d in dupes:
        result.excluded.append(
            ExcludedCandidate(
                description=d.label(),
                reason="duplicate of an identical topology",
                engine=d.engine,
                plan=d,
            )
        )
    _rank(unique, workload.objective)
    result.candidates = unique
    viable_tps = [p.tensor_parallel_size * p.pipeline_parallel_size for p in unique]
    result.estimated_minimum_tp = min(viable_tps) if viable_tps else None
    if not unique:
        reasons = "\n".join(f"  - {e.description}: {e.reason}" for e in result.excluded[:12])
        raise NoViablePlanError(
            "No candidate topology is estimated to fit this model on the selected GPUs.\n"
            + reasons,
            hints=[
                "Choose a quantized variant of the model or a smaller model.",
                "Reduce --context-length or lower --memory-headroom.",
                "Select more GPUs with --gpus, or use a Ray cluster with more machines.",
            ],
        )
    return result


def _base_rationale(
    plan: CandidatePlan,
    estimate: MemoryEstimate,
    planner: GPUGroupPlanner,
    gpus_per_node: int,
    snapshot: HardwareSnapshot,
) -> list[str]:
    lines: list[str] = []
    tp = plan.tensor_parallel_size
    if tp == 1:
        lines.append(
            f"Model weights ({format_bytes(estimate.weights_bytes)}) fit on one GPU; TP=1 avoids inter-GPU communication."
        )
    else:
        lines.append(f"TP={tp} splits weights to ~{format_bytes(estimate.weights_bytes)} per GPU.")
    if plan.replica_count > 1:
        lines.append(
            f"{plan.replica_count} independent replicas use all {plan.gpu_count} selected GPUs; the router load-balances across them."
        )
    if estimate.estimated_max_concurrency_p50 is not None:
        lines.append(
            f"Estimated KV cache {format_bytes(estimate.kv_cache_bytes_available)} per GPU ≈ "
            f"{estimate.estimated_max_concurrency_p50} concurrent p50 sequences per replica (confidence: {estimate.kv_confidence})."
        )
    elif estimate.kv_confidence == "unknown":
        lines.append(
            "KV-cache capacity for this architecture is unknown statically; launch validation will establish it."
        )
    if tp > 1:
        lines.extend("Group " + d for d in planner.describe(plan.gpu_groups)[:4])
    if snapshot.is_cluster:
        lines.append(
            f"Replicas are placed within machines ({gpus_per_node} GPUs per node); no replica spans the network."
        )
    return lines


def _cross_node_candidates(
    snapshot: HardwareSnapshot,
    model: ModelProfile,
    workload: WorkloadProfile,
    engines: Sequence[InferenceEngine],
    constraints: PlanConstraints,
    memory_cfg: MemoryModelConfig,
    context_length: int,
    result: PlanningResult,
    kv_for: Callable[[int], KVEstimate],
) -> list[CandidatePlan]:
    """vLLM Ray-backend candidates spanning every node (TP inside nodes, PP across)."""
    plans: list[CandidatePlan] = []
    by_node = snapshot.gpus_by_node()
    node_ids = sorted(k for k in by_node if k is not None)
    if len(node_ids) < 2:
        return plans
    if constraints.replica_count not in (None, 1):
        result.excluded.append(
            ExcludedCandidate(
                description="multi-node layouts",
                reason=f"a replica spanning machines is always a single copy; --replicas {constraints.replica_count} cannot be honoured across nodes",
            )
        )
        return plans
    gpus_per_node = min(len(by_node[n]) for n in node_ids)
    all_gpus = [g for n in node_ids for g in sorted(by_node[n])[:gpus_per_node]]
    pp = len(node_ids)
    tp = gpus_per_node
    shapes: list[tuple[int, int, str]] = [
        (
            tp,
            pp,
            "TP within each machine, pipeline stages across machines (vLLM's recommended multi-node layout)",
        )
    ]
    total = tp * pp
    if check_tp_divisibility(model, total) is None:
        shapes.append(
            (total, 1, "single tensor-parallel group spanning machines (higher communication cost)")
        )
    if constraints.tensor_parallel_size is not None:
        wanted = constraints.tensor_parallel_size
        shapes = [s for s in shapes if s[0] == wanted]
        if not shapes:
            result.excluded.append(
                ExcludedCandidate(
                    description=f"TP={wanted} multi-node",
                    reason=f"multi-node layouts on this cluster use TP={tp} per machine or TP={total} across all machines",
                    tensor_parallel_size=wanted,
                )
            )
            return plans
    for engine in engines:
        if constraints.engine is not None and engine.engine_name != constraints.engine:
            continue
        native = snapshot.provider == "ssh" and engine.supports_native_backend()
        if not engine.supports_ray_backend() and not native:
            result.excluded.append(
                ExcludedCandidate(
                    description=f"{engine.name()} multi-node",
                    reason=f"{engine.name()} has no supported distributed executor on this connection",
                    engine=engine.engine_name,
                )
            )
            continue
        for tp_size, pp_size, why in shapes:
            backend = (
                ("mp" if engine.engine_name == EngineName.VLLM else "native") if native else "ray"
            )
            if pp_size > 1 and not engine.supports_pipeline_parallel():
                continue
            estimate = estimate_memory(
                model,
                gpus=[snapshot.gpu(i) for i in all_gpus],
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                kv=kv_for(tp_size),
                context_length=context_length,
                workload=workload,
                cfg=memory_cfg,
                memory_fraction=constraints.memory_fraction,
            )
            plan = CandidatePlan(
                id=_plan_id(engine.engine_name, tp_size, pp_size, 1, f"-{backend}"),
                engine=engine.engine_name,
                gpu_groups=[all_gpus],
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                replica_count=1,
                context_length=context_length,
                objective=workload.objective,
                memory_fraction=estimate.memory_fraction,
                max_num_seqs=_max_num_seqs(
                    engine, estimate.estimated_max_concurrency_p50, constraints
                ),
                kv_cache_dtype=constraints.kv_cache_dtype,
                engine_args=dict(constraints.extra_engine_args),
                estimated_memory=estimate,
                distributed_backend=backend,
                replica_nodes=[[snapshot.node_of(g) or "local" for g in all_gpus]],
            )
            plan.max_running_requests = plan.max_num_seqs
            plan.max_concurrency = plan.max_num_seqs
            mem_reason = check_memory(estimate)
            support = engine.supports(model, plan)
            if not support.supported or mem_reason:
                result.excluded.append(
                    ExcludedCandidate(
                        description=plan.label(),
                        reason=mem_reason or "; ".join(support.reasons),
                        engine=engine.engine_name,
                        tensor_parallel_size=tp_size,
                        replica_count=1,
                        plan=plan,
                    )
                )
                continue
            plan.viability = (
                CandidateViability.ESTIMATED_VIABLE
                if estimate.confidence != "unknown"
                else CandidateViability.UNKNOWN
            )
            plan.rationale = [
                f"One replica spans the existing machines using {engine.name()}'s {backend} runtime.",
                why,
                f"Estimated weights per GPU: {format_bytes(estimate.weights_bytes)}.",
                *support.warnings,
            ]
            plans.append(plan)
    return plans


def _rank(plans: list[CandidatePlan], objective: Objective) -> None:
    """Heuristic ordering used to decide which structural candidates to benchmark first."""

    def key(p: CandidatePlan) -> tuple[int, int, int, str]:
        variant = 0 if not (p.expert_parallel_enabled or p.dp_attention_enabled) else 1
        if objective == Objective.LATENCY:
            # Larger TP typically lowers per-token latency; try it first.
            return (-p.tensor_parallel_size, p.pipeline_parallel_size, variant, p.engine.value)
        return (p.tensor_parallel_size, p.pipeline_parallel_size, variant, p.engine.value)

    plans.sort(key=key)
    for i, p in enumerate(plans):
        p.heuristic_rank = i
