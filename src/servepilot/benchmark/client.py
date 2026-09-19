"""Async OpenAI-compatible benchmark client.

Measures TTFT (first SSE event carrying model output), end-to-end latency and TPOT per request.
Token counts come from the API ``usage`` object when present, otherwise from the tokenizer.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from servepilot.benchmark.workload import BenchmarkRequest
from servepilot.models.tokenizer import TokenCounter
from servepilot.schemas.benchmark import BenchmarkEndpoint, RequestBenchmarkResult


def build_payload(
    request: BenchmarkRequest,
    *,
    model: str,
    endpoint: BenchmarkEndpoint,
    stream: bool,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if ignore_eos:
        # vLLM/SGLang extension: force the full output length so measured output tokens match
        # the workload profile instead of depending on where the model chooses to stop.
        payload["ignore_eos"] = True
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": request.prompt}]
    else:
        payload["prompt"] = request.prompt
    if request.payload is not None:
        # Replay the actual messages and generation parameters. Routing and transport fields
        # remain controller-owned, and real traffic is not forced to ignore EOS.
        payload.pop("ignore_eos", None)
        payload.update(request.payload)
        payload.update(model=model, stream=stream)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        else:
            payload.pop("stream_options", None)
    return payload


def _parse_response(data: str | bytes) -> dict[str, Any]:
    chunk = json.loads(data)
    if not isinstance(chunk, dict):
        raise ValueError("response must be a JSON object")
    if chunk.get("error") is not None:
        raise ValueError(f"API error: {str(chunk['error'])[:200]}")
    return chunk


def _first_choice(chunk: dict[str, Any]) -> dict[str, Any]:
    choices = chunk.get("choices")
    if choices is None:
        return {}
    if not isinstance(choices, list):
        raise ValueError("choices must be an array")
    if not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choice must be an object")
    return first


def _extract_text(chunk: dict[str, Any], endpoint: BenchmarkEndpoint) -> str:
    first = _first_choice(chunk)
    if endpoint == "chat":
        message = first.get("delta")
        if message is None:
            message = first.get("message")
        if message is None:
            return ""
        if not isinstance(message, dict):
            raise ValueError("message or delta must be an object")
        content = message.get("content")
    else:
        content = first.get("text")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("model text must be a string")
    return content


def _usage_counts(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = chunk.get("usage")
    if usage is None:
        return None, None
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object")
    counts = []
    for name in ("prompt_tokens", "completion_tokens"):
        value = usage.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"usage.{name} must be a non-negative integer")
        counts.append(value)
    return counts[0], counts[1]


class BenchmarkClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        tokenizer: TokenCounter,
        timeout_seconds: float,
        api_key: str | None = None,
        max_connections: int = 2048,
        ignore_eos: bool = True,
        trust_server_usage: bool = True,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._tok = tokenizer
        self._ignore_eos = ignore_eos
        self._trust_server_usage = trust_server_usage
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=30.0),
            headers=headers,
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> BenchmarkClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _path(self, endpoint: BenchmarkEndpoint) -> str:
        return "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"

    async def run_request(
        self, request: BenchmarkRequest, *, endpoint: BenchmarkEndpoint, stream: bool
    ) -> RequestBenchmarkResult:
        if request.endpoint is not None:
            if request.endpoint not in ("chat", "completions"):
                raise ValueError(f"unknown replay endpoint {request.endpoint}")
            endpoint = "chat" if request.endpoint == "chat" else "completions"
        payload = build_payload(
            request,
            model=self._model,
            endpoint=endpoint,
            stream=stream,
            ignore_eos=self._ignore_eos,
        )
        url = self._base + self._path(endpoint)
        started = time.perf_counter()
        if stream:
            result = await self._run_streaming(url, payload, request, endpoint, started)
        else:
            result = await self._run_blocking(url, payload, request, endpoint, started)
        result.request_index = request.index
        return result

    async def _run_blocking(
        self,
        url: str,
        payload: dict[str, Any],
        request: BenchmarkRequest,
        endpoint: BenchmarkEndpoint,
        started: float,
    ) -> RequestBenchmarkResult:
        try:
            resp = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                completed_at=time.perf_counter(),
                error=f"{type(exc).__name__}: {exc}",
            )
        completed = time.perf_counter()
        if resp.status_code >= 400:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                completed_at=completed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        try:
            body = _parse_response(resp.content)
            input_count, output_count = _usage_counts(body)
            text = _extract_text(body, endpoint)
        except ValueError as exc:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                completed_at=completed,
                error=f"invalid response: {exc}",
                status_code=resp.status_code,
            )
        output_tokens = (
            output_count
            if self._trust_server_usage and output_count is not None
            else self._tok.count(text)
        )
        input_tokens = (
            input_count
            if self._trust_server_usage and input_count is not None
            else request.input_tokens
        )
        e2e_ms = (completed - started) * 1000.0
        return RequestBenchmarkResult(
            success=output_tokens > 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started,
            completed_at=completed,
            e2e_latency_ms=e2e_ms,
            status_code=resp.status_code,
            error=None if output_tokens > 0 else "no output tokens",
            output_text=text,
        )

    async def _run_streaming(
        self,
        url: str,
        payload: dict[str, Any],
        request: BenchmarkRequest,
        endpoint: BenchmarkEndpoint,
        started: float,
    ) -> RequestBenchmarkResult:
        first_token_at: float | None = None
        text_parts: list[str] = []
        input_count: int | None = None
        output_count: int | None = None
        finished = False
        status_code: int | None = None
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                status_code = resp.status_code
                if resp.status_code >= 400:
                    body = await resp.aread()
                    return RequestBenchmarkResult(
                        success=False,
                        input_tokens=request.input_tokens,
                        started_at=started,
                        completed_at=time.perf_counter(),
                        error=f"HTTP {resp.status_code}: {body[:200]!r}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        finished = True
                        break
                    chunk = _parse_response(data)
                    prompt_count, completion_count = _usage_counts(chunk)
                    if prompt_count is not None:
                        input_count = prompt_count
                    if completion_count is not None:
                        output_count = completion_count
                    text = _extract_text(chunk, endpoint)
                    if _first_choice(chunk).get("finish_reason") is not None:
                        finished = True
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(text)
            if not finished:
                raise ValueError("incomplete stream: no finish reason or [DONE] marker")
        except (httpx.HTTPError, ValueError) as exc:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                first_token_at=first_token_at,
                completed_at=time.perf_counter(),
                error=f"{type(exc).__name__}: {exc}",
                status_code=status_code,
            )
        completed = time.perf_counter()
        output_tokens = (
            output_count
            if self._trust_server_usage and output_count is not None
            else self._tok.count("".join(text_parts))
        )
        input_tokens = (
            input_count
            if self._trust_server_usage and input_count is not None
            else request.input_tokens
        )
        e2e_ms = (completed - started) * 1000.0
        ttft_ms = (first_token_at - started) * 1000.0 if first_token_at is not None else None
        tpot_ms: float | None = None
        if ttft_ms is not None and output_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
        return RequestBenchmarkResult(
            success=output_tokens > 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started,
            first_token_at=first_token_at,
            completed_at=completed,
            ttft_ms=ttft_ms,
            e2e_latency_ms=e2e_ms,
            tpot_ms=tpot_ms,
            status_code=status_code,
            error=None if output_tokens > 0 else "no output tokens",
            output_text="".join(text_parts),
        )
