"""SGLang adapter (``<python> -m sglang.launch_server``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from servepilot.constants import GIB
from servepilot.engines.base import (
    InferenceEngine,
    LaunchSpec,
    SupportResult,
    engine_environment,
    redacted_command,
)
from servepilot.engines.interpreter import EngineRuntime, find_engine_runtime, parse_version
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName

SGLANG_MIN_SUPPORTED_VERSION = (0, 4, 0)

SGLANG_QUANTIZATION_METHODS = {
    "awq",
    "awq_marlin",
    "gptq",
    "gptq_marlin",
    "fp8",
    "blockwise_int8",
    "compressed-tensors",
    "compressed_tensors",
    "w8a8_int8",
    "w8a8_fp8",
    "bitsandbytes",
    "gguf",
    "modelopt",
    "modelopt_fp4",
    "moe_wna16",
    "mxfp4",
    "qoq",
    "petit_nvfp4",
}

# MoE model types where SGLang's expert parallelism is exercised regularly.
SGLANG_EP_MODEL_TYPES = {
    "deepseek_v2",
    "deepseek_v3",
    "qwen2_moe",
    "qwen3_moe",
    "mixtral",
    "glm4_moe",
    "llama4",
    "gpt_oss",
    "kimi_k2",
    "minimax_m1",
}
# DP attention is documented/validated for these architectures.
SGLANG_DP_ATTENTION_MODEL_TYPES = {
    "deepseek_v2",
    "deepseek_v3",
    "qwen2_moe",
    "qwen3_moe",
    "glm4_moe",
}


# SGLang captures decode CUDA graphs up to this batch size by default on 80 GB-class GPUs with
# TP < 4; larger decode batches run eagerly and are markedly slower. ServePilot raises the
# capture limit to the plan's running-request limit, up to the same 512 ceiling vLLM uses.
SGLANG_DEFAULT_DECODE_CUDA_GRAPH_MAX_BS = 256
SGLANG_DECODE_CUDA_GRAPH_MAX_BS_CEILING = 512
SGLANG_LARGE_GPU_BYTES = 60 * GIB
# Never hand SGLang less than this for weights + KV cache.
SGLANG_MIN_STATIC_FRACTION = 0.30


@dataclass(frozen=True)
class _SGLangFlags:
    version: tuple[int, ...]

    @property
    def supports_ep(self) -> bool:
        return self.version >= (0, 4, 0)

    @property
    def supports_pp(self) -> bool:
        return self.version >= (0, 4, 5)

    @property
    def decode_cuda_graph_max_bs(self) -> str:
        # Renamed when prefill CUDA graphs were added; the old name is a deprecated alias since.
        return "--cuda-graph-max-bs-decode" if self.version >= (0, 5, 18) else "--cuda-graph-max-bs"


def static_memory_fraction(plan: CandidatePlan) -> float | None:
    """Translate ServePilot's memory fraction into SGLang's ``--mem-fraction-static``.

    ServePilot's fraction budgets everything the engine allocates (weights, KV cache, activations,
    CUDA graphs, communication buffers, runtime overhead). SGLang's static fraction covers only
    weights + KV pool and allocates the rest *outside* it, so passing ServePilot's number through
    leaves SGLang too little room for activations and graph capture. Subtract the non-static parts
    of the estimate so both engines get the same total budget.
    """
    if plan.memory_fraction is None:
        return None
    est = plan.estimated_memory
    if est is None or est.device_total_bytes <= 0:
        return plan.memory_fraction
    dynamic = (
        est.activations_bytes
        + est.cuda_graph_bytes
        + est.communication_bytes
        + est.engine_overhead_bytes
    )
    static = plan.memory_fraction - dynamic / est.device_total_bytes
    return max(SGLANG_MIN_STATIC_FRACTION, round(static, 2))


def decode_cuda_graph_max_bs(plan: CandidatePlan, running_limit: int) -> int | None:
    """Decode CUDA-graph capture limit to request, or None to keep SGLang's default."""
    est = plan.estimated_memory
    if est is None or est.device_total_bytes < SGLANG_LARGE_GPU_BYTES or plan.dp_attention_enabled:
        return None
    target = min(running_limit, SGLANG_DECODE_CUDA_GRAPH_MAX_BS_CEILING)
    return target if target > SGLANG_DEFAULT_DECODE_CUDA_GRAPH_MAX_BS else None


class SGLangEngine(InferenceEngine):
    engine_name = EngineName.SGLANG

    def __init__(self, runtime: EngineRuntime | None = None, *, probe: bool = True) -> None:
        self._runtime = runtime
        if runtime is None and probe:
            self._runtime = find_engine_runtime(
                "sglang",
                env_var="SERVEPILOT_SGLANG_PYTHON",
                console_scripts=(),
                conventional_dirs=("~/engines/sglang", "/opt/sglang"),
            )

    def name(self) -> str:
        return "sglang"

    def is_available(self) -> bool:
        return self._runtime is not None

    def version(self) -> str | None:
        return self._runtime.version if self._runtime else None

    def python(self) -> str | None:
        return self._runtime.python if self._runtime else None

    def unavailable_hints(self) -> list[str]:
        return [
            'pip install "sglang[all]"   (in this environment), or',
            "export SERVEPILOT_SGLANG_PYTHON=/path/to/venv/bin/python to use another environment",
        ]

    def _flags(self) -> _SGLangFlags:
        return _SGLangFlags(parse_version(self.version()))

    def supports_expert_parallel(self, model: ModelProfile) -> bool:
        return (
            model.is_moe
            and (model.model_type or "").lower() in SGLANG_EP_MODEL_TYPES
            and self._flags().supports_ep
        )

    def supports_dp_attention(self, model: ModelProfile) -> bool:
        return model.is_moe and (model.model_type or "").lower() in SGLANG_DP_ATTENTION_MODEL_TYPES

    def supports_pipeline_parallel(self) -> bool:
        return self._flags().supports_pp

    def supports_ray_backend(self) -> bool:
        return False

    def supports_native_backend(self) -> bool:
        return parse_version(self.version()) >= (0, 4, 5)

    def default_max_num_seqs(self) -> int:
        return 256

    def supports(self, model: ModelProfile, plan: CandidatePlan) -> SupportResult:
        reasons: list[str] = []
        warnings: list[str] = []
        version = parse_version(self.version())
        if version and version < SGLANG_MIN_SUPPORTED_VERSION:
            reasons.append(
                f"SGLang {self.version()} is older than the minimum ServePilot supports (0.4.0)"
            )
        method = (model.quantization_method or "").lower()
        if method and method not in SGLANG_QUANTIZATION_METHODS:
            reasons.append(f"quantization method {method!r} is not known to be supported by SGLang")
        if plan.distributed_backend == "ray":
            reasons.append("SGLang uses its native multi-node runtime, not a Ray executor")
        if plan.dp_attention_enabled and not self.supports_dp_attention(model):
            reasons.append(
                f"DP attention optimization is not known to support model type {model.model_type!r}"
            )
        if plan.expert_parallel_enabled and not self.supports_expert_parallel(model):
            reasons.append(
                f"expert parallelism is not known to work for model type {model.model_type!r} in SGLang"
            )
        if model.is_multimodal:
            warnings.append("multimodal model: text-only benchmarks are used for tuning")
        if model.trust_remote_code_required:
            warnings.append(
                "model requires trust_remote_code; ServePilot only passes it when --trust-remote-code is set"
            )
        confidence: Literal["high", "medium", "low"] = (
            "high" if not reasons and model.model_type and not model.is_moe else "medium"
        )
        return SupportResult(
            supported=not reasons, confidence=confidence, reasons=reasons, warnings=warnings
        )

    def build_launch_spec(
        self,
        model: ModelProfile,
        plan: CandidatePlan,
        *,
        replica_index: int,
        host: str,
        port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        node: tuple[str, str] | None = None,
        local_gpu_ids: list[int] | None = None,
    ) -> LaunchSpec:
        if self._runtime is None:
            raise RuntimeError("SGLang is not available; check SGLangEngine.is_available() first")
        gpu_ids = plan.gpu_groups[replica_index]
        device_ids = local_gpu_ids if local_gpu_ids is not None else gpu_ids
        model_ref = model.local_path or model.model_id
        args: list[str] = [
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_ref,
            "--host",
            host,
            "--port",
            str(port),
            "--tp-size",
            str(plan.tensor_parallel_size),
        ]
        if plan.pipeline_parallel_size > 1:
            args += ["--pp-size", str(plan.pipeline_parallel_size)]
        if plan.data_parallel_size > 1:
            args += ["--dp-size", str(plan.data_parallel_size)]
        if plan.expert_parallel_enabled and plan.expert_parallel_size:
            args += ["--ep-size", str(plan.expert_parallel_size)]
        if plan.dp_attention_enabled:
            args.append("--enable-dp-attention")
        static_fraction = static_memory_fraction(plan)
        if static_fraction is not None:
            args += ["--mem-fraction-static", f"{static_fraction:.2f}"]
        args += ["--context-length", str(plan.context_length)]
        limit = plan.max_running_requests or plan.max_num_seqs
        if limit is not None:
            args += ["--max-running-requests", str(limit)]
            graph_bs = decode_cuda_graph_max_bs(plan, limit)
            if graph_bs is not None:
                args += [self._flags().decode_cuda_graph_max_bs, str(graph_bs)]
        if plan.chunked_prefill_enabled is False:
            args += ["--chunked-prefill-size", "-1"]
        elif plan.chunked_prefill_size:
            args += ["--chunked-prefill-size", str(plan.chunked_prefill_size)]
        if plan.kv_cache_dtype:
            args += ["--kv-cache-dtype", plan.kv_cache_dtype]
        if model.configured_dtype in ("bfloat16", "float16") and not model.is_quantized:
            args += ["--dtype", model.configured_dtype]
        if model.quantization_method and model.quantization_method.lower() in {
            "gguf",
            "bitsandbytes",
        }:
            args += ["--quantization", model.quantization_method.lower()]
        if model.revision and not model.local_path:
            args += ["--revision", model.revision]
        if served_model_name:
            args += ["--served-model-name", served_model_name]
        if trust_remote_code:
            args.append("--trust-remote-code")
        args += ["--log-level", "warning"]
        for key, value in sorted(plan.engine_args.items()):
            flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"
            if value is True:
                args.append(flag)
            elif value is False or value is None:
                continue
            else:
                args += [flag, str(value)]

        env = engine_environment(self._runtime.python, list(device_ids))
        executable = self._runtime.python
        return LaunchSpec(
            executable=executable,
            args=args,
            env=env,
            host=host,
            port=port,
            gpu_ids=list(gpu_ids),
            redacted_display_command=redacted_command([executable, *args], env),
            replica_id=f"{plan.id}-r{replica_index}",
            node_id=node[0] if node else None,
            node_ip=node[1] if node else None,
            readiness_path="/v1/models",
            health_path="/health",
        )
