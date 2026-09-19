"""Benchmark measurements must reflect output tokens and report backend failures."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from servepilot.benchmark.client import BenchmarkClient
from servepilot.benchmark.workload import BenchmarkRequest
from servepilot.models.tokenizer import ApproximateTokenizer
from servepilot.schemas.benchmark import RequestBenchmarkResult


@pytest.fixture
async def client() -> AsyncIterator[BenchmarkClient]:
    async with BenchmarkClient(
        "http://benchmark.test", model="test", tokenizer=ApproximateTokenizer(), timeout_seconds=1
    ) as client:
        yield client


async def respond(client: BenchmarkClient, response: httpx.Response) -> None:
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))


async def run(client: BenchmarkClient, *, stream: bool = True) -> RequestBenchmarkResult:
    return await client.run_request(
        BenchmarkRequest(index=0, prompt="hello", input_tokens=2, max_tokens=16),
        endpoint="chat",
        stream=stream,
    )


def sse(*chunks: object, done: bool = True) -> httpx.Response:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def delta(text: str) -> dict[str, object]:
    return {"choices": [{"delta": {"content": text}}]}


async def test_fallback_counts_text_independently_of_chunk_boundaries(
    client: BenchmarkClient,
) -> None:
    text = "hello there general kenobi"
    await respond(client, sse(delta(text)))
    combined = await run(client)
    await respond(
        client, sse(*(delta(part) for part in ["hello ", "there ", "general ", "kenobi"]))
    )
    split = await run(client)
    assert combined.success and split.success
    assert combined.output_tokens == split.output_tokens == ApproximateTokenizer().count(text)


async def test_usage_counts_take_precedence(client: BenchmarkClient) -> None:
    await respond(
        client, sse(delta("hello"), {"usage": {"completion_tokens": 7, "prompt_tokens": 9}})
    )
    result = await run(client)
    assert result.success and result.output_tokens == 7 and result.input_tokens == 9


@pytest.mark.parametrize("stream", [True, False])
async def test_optimization_counts_tokens_independently_of_untrusted_engine_usage(
    stream: bool,
) -> None:
    async with BenchmarkClient(
        "http://benchmark.test",
        model="test",
        tokenizer=ApproximateTokenizer(),
        trust_server_usage=False,
        timeout_seconds=1,
    ) as client:
        choice = {"delta" if stream else "message": {"content": "hello there"}}
        body = {
            "choices": [choice],
            "usage": {"prompt_tokens": 9000000, "completion_tokens": 9000000},
        }
        await respond(client, sse(body) if stream else httpx.Response(200, json=body))
        result = await run(client, stream=stream)
        assert result.success
        assert result.output_tokens == ApproximateTokenizer().count("hello there")
        assert result.input_tokens == 2


@pytest.mark.parametrize("stream", [True, False])
async def test_zero_usage_is_not_replaced_with_estimated_tokens(
    client: BenchmarkClient, stream: bool
) -> None:
    body = {**delta("hello"), "usage": {"completion_tokens": 0, "prompt_tokens": 0}}
    await respond(client, sse(body) if stream else httpx.Response(200, json=body))
    result = await run(client, stream=stream)
    assert not result.success and result.output_tokens == 0 and result.input_tokens == 0


@pytest.mark.parametrize("stream", [True, False])
async def test_malformed_json_is_a_failure(client: BenchmarkClient, stream: bool) -> None:
    await respond(client, httpx.Response(200, text="data: {broken}\n\n" if stream else "{broken}"))
    result = await run(client, stream=stream)
    assert not result.success and result.completed_at is not None and result.error


@pytest.mark.parametrize("stream", [True, False])
async def test_completions_endpoint_counts_text(client: BenchmarkClient, stream: bool) -> None:
    body = {"choices": [{"text": "hello there"}]}
    await respond(client, sse(body) if stream else httpx.Response(200, json=body))
    result = await client.run_request(
        BenchmarkRequest(index=0, prompt="hello", input_tokens=2, max_tokens=16),
        endpoint="completions",
        stream=stream,
    )
    assert result.success and result.output_tokens == ApproximateTokenizer().count("hello there")


async def test_stream_error_after_partial_output_is_a_failure(client: BenchmarkClient) -> None:
    await respond(client, sse(delta("partial"), {"error": {"message": "engine failed"}}))
    result = await run(client)
    assert not result.success and "engine failed" in result.error
    assert result.first_token_at is not None and result.completed_at is not None


async def test_unfinished_stream_is_a_failure(client: BenchmarkClient) -> None:
    await respond(client, sse(delta("partial"), done=False))
    result = await run(client)
    assert not result.success and "incomplete" in result.error


async def test_finish_reason_without_done_is_accepted(client: BenchmarkClient) -> None:
    await respond(
        client, sse(delta("hello"), {"choices": [{"finish_reason": "length"}]}, done=False)
    )
    assert (await run(client)).success


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "body",
    [
        [],
        {"choices": "bad"},
        {"choices": [None]},
        {"choices": [{"delta": {"content": ["bad"]}}]},
        {"choices": [{"message": "bad"}]},
        {"usage": "bad"},
        {"usage": {"completion_tokens": "bad"}},
        {"usage": {"completion_tokens": -1}},
        {"usage": {"completion_tokens": True}},
        {"error": {"message": "engine failed"}},
    ],
)
async def test_invalid_response_is_recorded_instead_of_raising(
    client: BenchmarkClient, body: object, stream: bool
) -> None:
    await respond(client, sse(body) if stream else httpx.Response(200, json=body))
    result = await run(client, stream=stream)
    assert not result.success and result.error
    assert result.status_code == 200 and result.completed_at is not None


async def test_nonstreaming_has_no_ttft_or_tpot(client: BenchmarkClient) -> None:
    await respond(
        client, httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
    )
    result = await run(client, stream=False)
    assert result.success and result.e2e_latency_ms is not None
    assert result.ttft_ms is None and result.tpot_ms is None


async def test_nonstreaming_transport_failure_has_completion_time(client: BenchmarkClient) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    result = await run(client, stream=False)
    assert not result.success and result.completed_at is not None
    assert result.completed_at >= result.started_at
