"""vLLM adapter.

All vLLM CLI knowledge is concentrated in :meth:`VLLMEngine.build_launch_spec` and the
version-gated :class:`_VLLMFlags` table. The server is started as
``<python> -m vllm.entrypoints.openai.api_server --model MODEL ...`` which has been stable
across vLLM releases and works with an interpreter from a different virtual environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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
from servepilot.schemas.workload import Objective

VLLM_MIN_SUPPORTED_VERSION = (0, 6, 0)

# Quantization methods vLLM accepts as `--quantization` / detects from config.
VLLM_QUANTIZATION_METHODS = {
    "awq",
    "awq_marlin",
    "gptq",
    "gptq_marlin",
    "gptq_marlin_24",
    "fp8",
    "fbgemm_fp8",
    "compressed-tensors",
    "compressed_tensors",
    "bitsandbytes",
    "gguf",
    "modelopt",
    "modelopt_fp4",
    "marlin",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "mxfp4",
    "auto-round",
}

# MoE model types with expert-parallel support in vLLM.
VLLM_EP_MODEL_TYPES = {
    "mixtral",
    "deepseek_v2",
    "deepseek_v3",
    "qwen2_moe",
    "qwen3_moe",
    "llama4",
    "gpt_oss",
    "glm4_moe",
    "dbrx",
    "olmoe",
    "phimoe",
    "granitemoe",
    "minimax_m1",
    "kimi_k2",
    "ernie4_5_moe",
    "hunyuan_v1_moe",
}


@dataclass(frozen=True)
class _VLLMFlags:
    """Flag names that differ between vLLM versions (extend here, nowhere else)."""

    version: tuple[int, ...]

    @property
    def chunked_prefill_off(self) -> str:
        # Boolean flags gained --no- forms in 0.6.x; older versions used explicit values.
        return (
            "--no-enable-chunked-prefill"
            if self.version >= (0, 6, 4)
            else "--enable-chunked-prefill=False"
        )

    @property
    def supports_expert_parallel(self) -> bool:
        return self.version >= (0, 6, 4)

    @property
    def supports_disable_uvicorn_access_log(self) -> bool:
        return self.version >= (0, 7, 0)

    @property
    def supports_data_parallel(self) -> bool:
        return self.version >= (0, 8, 0)

    @property
    def supports_dp_attention(self) -> bool:
        # DP attention for MoE: one process, `--tensor-parallel-size 1 --data-parallel-size N
        # --enable-expert-parallel`, with internal load balancing across the DP ranks.
        return self.version >= (0, 9, 0)

    @property
    def supports_performance_mode(self) -> bool:
        return self.version >= (0, 17, 0)


# ServePilot objective → vLLM `--performance-mode` (0.17+). "balanced" is vLLM's default.
VLLM_PERFORMANCE_MODES = {
    Objective.THROUGHPUT: "throughput",
    Objective.LATENCY: "interactivity",
}


class VLLMEngine(InferenceEngine):
    engine_name = EngineName.VLLM

    def __init__(self, runtime: EngineRuntime | None = None, *, probe: bool = True) -> None:
        self._runtime = runtime
        self._probed = runtime is not None or not probe
        if not self._probed:
            self._runtime = find_engine_runtime(
                "vllm",
                env_var="SERVEPILOT_VLLM_PYTHON",
                console_scripts=("vllm",),
                conventional_dirs=("~/engines/vllm", "/opt/vllm"),
            )
            self._probed = True

    # ------------------------------------------------------------------ identity
    def name(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        return self._runtime is not None

    def version(self) -> str | None:
        return self._runtime.version if self._runtime else None

    def python(self) -> str | None:
        return self._runtime.python if self._runtime else None

    def unavailable_hints(self) -> list[str]:
        return [
            "pip install vllm   (in this environment), or",
            "point ServePilot at an environment that has it: export SERVEPILOT_VLLM_PYTHON=/path/to/venv/bin/python",
        ]

    def _flags(self) -> _VLLMFlags:
        return _VLLMFlags(parse_version(self.version()))

    # ------------------------------------------------------------------ capabilities
    def supports_expert_parallel(self, model: ModelProfile) -> bool:
        return (
            model.is_moe
            and (model.model_type or "").lower() in VLLM_EP_MODEL_TYPES
            and self._flags().supports_expert_parallel
        )

    def supports_dp_attention(self, model: ModelProfile) -> bool:
        return self.supports_expert_parallel(model) and self._flags().supports_dp_attention

    def supports_pipeline_parallel(self) -> bool:
        return True

    def supports_ray_backend(self) -> bool:
        return True

    def supports_native_backend(self) -> bool:
        return parse_version(self.version()) >= (0, 17, 0)

    def default_max_num_seqs(self) -> int:
        return 256

    def supports(self, model: ModelProfile, plan: CandidatePlan) -> SupportResult:
        reasons: list[str] = []
        warnings: list[str] = []
        confidence: Literal["high", "medium", "low"] = "medium"
        version = parse_version(self.version())
        if version and version < VLLM_MIN_SUPPORTED_VERSION:
            reasons.append(
                f"vLLM {self.version()} is older than the minimum ServePilot supports "
                f"({'.'.join(map(str, VLLM_MIN_SUPPORTED_VERSION))}); upgrade vLLM"
            )
        method = (model.quantization_method or "").lower()
        if method and method not in VLLM_QUANTIZATION_METHODS:
            reasons.append(f"quantization method {method!r} is not known to be supported by vLLM")
        if plan.dp_attention_enabled and not self.supports_dp_attention(model):
            reasons.append(
                f"DP attention (data parallel + expert parallel) is not known to work for model type "
                f"{model.model_type!r} in vLLM {self.version()}"
            )
        if plan.expert_parallel_enabled and not self.supports_expert_parallel(model):
            reasons.append(
                f"expert parallelism is not known to work for model type {model.model_type!r} in vLLM"
            )
        if plan.distributed_backend == "ray" and plan.pipeline_parallel_size > 1 and model.is_moe:
            warnings.append(
                "pipeline parallelism with MoE models is less mature in vLLM; expect longer startup"
            )
        if model.trust_remote_code_required:
            warnings.append(
                "model requires trust_remote_code; ServePilot only passes it when --trust-remote-code is set"
            )
        if model.is_multimodal:
            warnings.append("multimodal model: text-only benchmarks are used for tuning")
        if model.attention_kind.value == "unknown" and not model.architecture_names:
            reasons.append("model configuration has no architecture information")
        if not reasons and model.model_type:
            confidence = "high" if not model.is_moe else "medium"
        return SupportResult(
            supported=not reasons, confidence=confidence, reasons=reasons, warnings=warnings
        )

    # ------------------------------------------------------------------ launch
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
            raise RuntimeError("vLLM is not available; check VLLMEngine.is_available() first")
        flags = self._flags()
        gpu_ids = plan.gpu_groups[replica_index]
        device_ids = local_gpu_ids if local_gpu_ids is not None else gpu_ids
        model_ref = model.local_path or model.model_id
        # DP attention (MoE): attention runs data-parallel on every GPU of the group while the
        # experts are sharded across them, so vLLM sees TP=1 × DP=N rather than TP=N.
        dp_attention = plan.dp_attention_enabled and plan.data_parallel_size > 1
        args: list[str] = [
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_ref,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            "1" if dp_attention else str(plan.tensor_parallel_size),
        ]
        if plan.pipeline_parallel_size > 1:
            args += ["--pipeline-parallel-size", str(plan.pipeline_parallel_size)]
        if plan.data_parallel_size > 1 and flags.supports_data_parallel:
            args += ["--data-parallel-size", str(plan.data_parallel_size)]
        if plan.distributed_backend:
            args += ["--distributed-executor-backend", plan.distributed_backend]
        if plan.memory_fraction is not None:
            args += ["--gpu-memory-utilization", f"{plan.memory_fraction:.2f}"]
        args += ["--max-model-len", str(plan.context_length)]
        if plan.max_num_seqs is not None:
            # `--max-num-seqs` applies per DP rank; the plan's value is the whole replica's.
            per_rank = (
                math.ceil(plan.max_num_seqs / plan.data_parallel_size)
                if dp_attention
                else plan.max_num_seqs
            )
            args += ["--max-num-seqs", str(max(1, per_rank))]
        if plan.expert_parallel_enabled and flags.supports_expert_parallel:
            args.append("--enable-expert-parallel")
        mode = VLLM_PERFORMANCE_MODES.get(plan.objective) if plan.objective else None
        if mode and flags.supports_performance_mode:
            args += ["--performance-mode", mode]
        if plan.chunked_prefill_enabled is False:
            args.append(flags.chunked_prefill_off)
        elif plan.chunked_prefill_enabled and plan.chunked_prefill_size:
            args += [
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                str(plan.chunked_prefill_size),
            ]
        if plan.kv_cache_dtype:
            args += ["--kv-cache-dtype", plan.kv_cache_dtype]
        if model.configured_dtype in ("bfloat16", "float16") and not model.is_quantized:
            args += ["--dtype", model.configured_dtype]
        # vLLM auto-detects quantization from the config; pass it only for methods that need it.
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
        if flags.supports_disable_uvicorn_access_log:
            args.append("--disable-uvicorn-access-log")
        for key, value in sorted(plan.engine_args.items()):
            flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"
            if value is True:
                args.append(flag)
            elif value is False or value is None:
                continue
            else:
                args += [flag, str(value)]

        env = engine_environment(
            self._runtime.python, None if plan.spans_nodes else list(device_ids)
        )
        if (
            plan.context_length
            and model.max_position_embeddings
            and plan.context_length > model.max_position_embeddings
        ):
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        executable = self._runtime.python
        spec = LaunchSpec(
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
        return spec
