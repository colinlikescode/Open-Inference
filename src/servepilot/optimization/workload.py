"""Load user traffic without silently replacing it with synthetic prompts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from servepilot.benchmark.workload import BenchmarkRequest
from servepilot.exceptions import ConfigurationError
from servepilot.models.tokenizer import TokenCounter
from servepilot.optimization.schemas import Contract
from servepilot.schemas.workload import WorkloadProfile


class TokenDistribution(Contract):
    p50: int = Field(ge=1)
    p95: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> TokenDistribution:
        if self.p95 < self.p50:
            raise ValueError("p95 must be >= p50")
        return self


class ConcurrencyDistribution(Contract):
    expected: int = Field(ge=1)
    peak: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> ConcurrencyDistribution:
        if self.peak < self.expected:
            raise ValueError("peak must be >= expected concurrency")
        return self


class TrafficDistribution(Contract):
    input_tokens: TokenDistribution
    output_tokens: TokenDistribution
    concurrency: ConcurrencyDistribution
    max_context_tokens: int | None = Field(default=None, ge=16)
    request_rate: float | None = Field(default=None, gt=0)
    shared_prefix_fraction: float = Field(default=0, ge=0, le=1)


class ReplayEntry(Contract):
    endpoint: str = "chat"
    request: dict[str, Any]

    @model_validator(mode="after")
    def valid(self) -> ReplayEntry:
        if self.endpoint not in ("chat", "completions"):
            raise ValueError("endpoint must be chat or completions")
        required = "messages" if self.endpoint == "chat" else "prompt"
        if not self.request.get(required):
            raise ValueError(f"replay request requires {required}")
        other = "prompt" if required == "messages" else "messages"
        if other in self.request:
            raise ValueError("request must use messages or prompt, not both")
        reserved = {"model", "stream", "stream_options", "n"} & self.request.keys()
        if reserved:
            raise ValueError(f"controller owns replay fields: {sorted(reserved)}")
        maximum = self.request.get("max_tokens")
        if type(maximum) is not int or maximum < 1:
            raise ValueError("every replay request needs a positive integer max_tokens")
        if self.endpoint == "chat":
            messages = self.request["messages"]
            if not isinstance(messages, list) or not all(
                isinstance(m, dict)
                and isinstance(m.get("role"), str)
                and isinstance(m.get("content"), str)
                for m in messages
            ):
                raise ValueError("replay messages must contain text role/content objects")
        elif not isinstance(self.request["prompt"], str):
            raise ValueError("replay prompt must be text")
        return self


def load_traffic(
    path: Path, base: WorkloadProfile, tokenizer: TokenCounter
) -> tuple[WorkloadProfile, list[int], list[ReplayEntry]]:
    """Return profile, fixed load levels, and exact requests (empty for synthetic traffic)."""
    try:
        text = path.read_text()
        if path.suffix.lower() == ".jsonl":
            entries = []
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "request" not in data:
                        data = {
                            "endpoint": "chat" if "messages" in data else "completions",
                            "request": data,
                        }
                    entries.append(ReplayEntry.model_validate(data))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
            if not entries:
                raise ValueError("replay workload is empty")
            requests = replay_requests(entries, tokenizer)
            inputs = sorted(r.input_tokens for r in requests)
            outputs = sorted(r.max_tokens for r in requests)
            data = base.model_dump()
            data.update(
                name="replay",
                input_tokens_p50=inputs[len(inputs) // 2],
                input_tokens_p95=inputs[min(len(inputs) - 1, math.ceil(len(inputs) * 0.95) - 1)],
                output_tokens_p50=outputs[len(outputs) // 2],
                output_tokens_p95=outputs[
                    min(len(outputs) - 1, math.ceil(len(outputs) * 0.95) - 1)
                ],
                max_context_tokens=max(base.max_context_tokens, max(inputs) + max(outputs)),
            )
            concurrency = base.expected_concurrency or 16
            return (
                WorkloadProfile.model_validate(data),
                sorted({max(1, concurrency // 2), concurrency, concurrency * 2}),
                entries,
            )
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict) or set(payload) != {"workload"}:
            raise ValueError("workload YAML must contain exactly one top-level workload mapping")
        distribution = TrafficDistribution.model_validate(payload["workload"])
        data = base.model_dump()
        data.update(
            name="custom",
            input_tokens_p50=distribution.input_tokens.p50,
            input_tokens_p95=distribution.input_tokens.p95,
            output_tokens_p50=distribution.output_tokens.p50,
            output_tokens_p95=distribution.output_tokens.p95,
            expected_concurrency=distribution.concurrency.expected,
            target_request_rate=distribution.request_rate,
            shared_prefix_fraction=distribution.shared_prefix_fraction,
            max_context_tokens=distribution.max_context_tokens
            or max(
                base.max_context_tokens,
                distribution.input_tokens.p95 + distribution.output_tokens.p95,
            ),
        )
        levels = sorted(
            {
                max(1, distribution.concurrency.expected // 2),
                distribution.concurrency.expected,
                distribution.concurrency.peak,
            }
        )
        return WorkloadProfile.model_validate(data), levels, []
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid workload {path}: {exc}") from exc


def replay_requests(entries: list[ReplayEntry], tokenizer: TokenCounter) -> list[BenchmarkRequest]:
    requests = []
    for index, entry in enumerate(entries):
        prompt = entry.request.get("prompt")
        if prompt is None:
            prompt = "\n".join(m["content"] for m in entry.request["messages"])
        requests.append(
            BenchmarkRequest(
                index=index,
                prompt=prompt,
                input_tokens=max(1, tokenizer.count(prompt)),
                max_tokens=entry.request["max_tokens"],
                payload=dict(entry.request),
                endpoint=entry.endpoint,
            )
        )
    return requests
